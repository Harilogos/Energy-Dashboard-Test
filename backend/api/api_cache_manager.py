"""
API Cache Manager for optimizing data fetching performance.

This module provides intelligent caching for API responses to minimize
repeated API calls and improve application performance.
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.logs.logger_setup import setup_logger
from backend.config.app_config import CONFIG

logger = setup_logger('api_cache_manager', 'api_cache.log')

class APICacheManager:
    """
    Manages caching of API responses with intelligent invalidation and bulk fetching.
    """

    def __init__(self, cache_dir: str = "cache/api_data"):
        """
        Initialize the API cache manager.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache configuration
        self.default_ttl = CONFIG["data"].get("cache_ttl", 3600)  # 1 hour default
        self.bulk_fetch_months = 6  # Fetch 6 months of data
        self.enable_bulk_fetch = True
        self.max_workers = CONFIG["data"].get("max_concurrent_requests", 4)

        # Thread safety
        self.cache_lock = threading.Lock()

        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

        logger.info(f"API Cache Manager initialized with cache dir: {self.cache_dir}")

    def _load_metadata(self) -> Dict:
        """Load cache metadata from file."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")

        return {
            "last_bulk_fetch": {},
            "cache_entries": {},
            "version": "1.0"
        }

    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _get_cache_key(self, plant_id: str, data_type: str, start_date: str,
                      end_date: str, granularity: str = "15m") -> str:
        """
        Generate a unique cache key for the given parameters.

        Args:
            plant_id: Plant identifier
            data_type: Type of data (generation, consumption, etc.)
            start_date: Start date string
            end_date: End date string
            granularity: Data granularity

        Returns:
            Unique cache key string
        """
        key_string = f"{plant_id}_{data_type}_{start_date}_{end_date}_{granularity}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_key: str, ttl_hours: int = None) -> bool:
        """
        Check if cached data is still valid.

        Args:
            cache_key: Cache key to check
            ttl_hours: Time-to-live in hours (uses default if None)

        Returns:
            True if cache is valid, False otherwise
        """
        if cache_key not in self.metadata["cache_entries"]:
            return False

        cache_info = self.metadata["cache_entries"][cache_key]
        cache_time = datetime.fromisoformat(cache_info["timestamp"])
        ttl = ttl_hours or (self.default_ttl / 3600)

        return datetime.now() - cache_time < timedelta(hours=ttl)

    def get_cached_data(self, plant_id: str, data_type: str, start_date: str,
                       end_date: str, granularity: str = "15m") -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and valid.

        Args:
            plant_id: Plant identifier
            data_type: Type of data
            start_date: Start date string
            end_date: End date string
            granularity: Data granularity

        Returns:
            Cached DataFrame or None if not available/valid
        """
        cache_key = self._get_cache_key(plant_id, data_type, start_date, end_date, granularity)

        with self.cache_lock:
            if not self._is_cache_valid(cache_key):
                return None

            cache_file = self._get_cache_file_path(cache_key)
            if not cache_file.exists():
                return None

            try:
                with open(cache_file, 'rb') as f:
                    df = pickle.load(f)

                logger.info(f"Cache hit for {plant_id} {data_type} ({start_date} to {end_date})")
                return df

            except Exception as e:
                logger.error(f"Failed to load cached data: {e}")
                return None

    def cache_data(self, plant_id: str, data_type: str, start_date: str,
                  end_date: str, data: pd.DataFrame, granularity: str = "15m"):
        """
        Cache API response data.

        Args:
            plant_id: Plant identifier
            data_type: Type of data
            start_date: Start date string
            end_date: End date string
            data: DataFrame to cache
            granularity: Data granularity
        """
        cache_key = self._get_cache_key(plant_id, data_type, start_date, end_date, granularity)
        cache_file = self._get_cache_file_path(cache_key)

        with self.cache_lock:
            try:
                # Save data to pickle file
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Update metadata
                self.metadata["cache_entries"][cache_key] = {
                    "plant_id": plant_id,
                    "data_type": data_type,
                    "start_date": start_date,
                    "end_date": end_date,
                    "granularity": granularity,
                    "timestamp": datetime.now().isoformat(),
                    "file_size": cache_file.stat().st_size,
                    "rows": len(data)
                }

                self._save_metadata()
                logger.info(f"Cached data for {plant_id} {data_type} ({len(data)} rows)")

            except Exception as e:
                logger.error(f"Failed to cache data: {e}")

    def should_bulk_fetch(self, plant_id: str) -> bool:
        """
        Check if bulk fetch is needed for a plant.

        Args:
            plant_id: Plant identifier

        Returns:
            True if bulk fetch is needed
        """
        if not self.enable_bulk_fetch:
            return False

        last_fetch_key = f"{plant_id}_bulk"
        if last_fetch_key not in self.metadata["last_bulk_fetch"]:
            return True

        last_fetch_time = datetime.fromisoformat(
            self.metadata["last_bulk_fetch"][last_fetch_key]
        )

        # Bulk fetch once per day
        return datetime.now() - last_fetch_time > timedelta(days=1)

    def bulk_fetch_plant_data(self, plant_id: str, integration, plant_type: str = "solar"):
        """
        Bulk fetch 6 months of data for a plant.

        Args:
            plant_id: Plant identifier
            integration: API integration object
            plant_type: Type of plant (solar/wind)
        """
        if not self.should_bulk_fetch(plant_id):
            logger.info(f"Bulk fetch not needed for {plant_id}")
            return

        logger.info(f"Starting bulk fetch for {plant_id} ({plant_type})")

        # Calculate date range (6 months back from today)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # ~6 months

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            # Fetch different granularities and data types
            # 15-minute is the industry standard for energy data
            fetch_configs = [
                {"granularity": "15m", "data_type": "generation"},  # Primary: 15-minute (industry standard)
                {"granularity": "1h", "data_type": "generation"},   # Secondary: hourly for aggregation
                {"granularity": "1d", "data_type": "generation"}    # Tertiary: daily for overview
            ]

            for config in fetch_configs:
                try:
                    # Check if already cached
                    cached_data = self.get_cached_data(
                        plant_id, config["data_type"], start_str, end_str, config["granularity"]
                    )

                    if cached_data is not None:
                        logger.info(f"Data already cached for {plant_id} {config['granularity']}")
                        continue

                    # Fetch from API
                    df = self._fetch_api_data(integration, plant_id, plant_type,
                                            start_str, end_str, config["granularity"])

                    if df is not None and not df.empty:
                        # Cache the data
                        self.cache_data(plant_id, config["data_type"], start_str, end_str,
                                      df, config["granularity"])
                        logger.info(f"Bulk cached {len(df)} rows for {plant_id} {config['granularity']}")

                except Exception as e:
                    logger.error(f"Failed to bulk fetch {config} for {plant_id}: {e}")

            # Update last bulk fetch time
            self.metadata["last_bulk_fetch"][f"{plant_id}_bulk"] = datetime.now().isoformat()
            self._save_metadata()

            logger.info(f"Completed bulk fetch for {plant_id}")

        except Exception as e:
            logger.error(f"Bulk fetch failed for {plant_id}: {e}")

    def _fetch_api_data(self, integration, plant_id: str, plant_type: str,
                       start_date: str, end_date: str, granularity: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from API with proper parameters based on plant type.

        Args:
            integration: API integration object
            plant_id: Plant identifier
            plant_type: Type of plant (solar/wind)
            start_date: Start date string
            end_date: End date string
            granularity: Data granularity

        Returns:
            DataFrame with fetched data or None if failed
        """
        try:
            if plant_type.lower() == "solar":
                df = integration.fetchDataV2(
                    plant_id,
                    "Plant",
                    ["Daily Energy"],
                    None,
                    start_date,
                    end_date,
                    granularity=granularity,
                    condition={"Daily Energy": "last"}
                )
            else:  # wind
                df = integration.fetchDataV2(
                    plant_id,
                    "Turbine",
                    ["WTUR.Generation today"],
                    None,
                    start_date,
                    end_date,
                    granularity=granularity,
                    condition={"Generation today": "last"}
                )

            return df if df is not None and not isinstance(df, str) else None

        except Exception as e:
            logger.error(f"API fetch failed for {plant_id}: {e}")
            return None

    def get_data_with_cache(self, plant_id: str, data_type: str, start_date: str,
                           end_date: str, granularity: str, integration,
                           plant_type: str = "solar") -> Optional[pd.DataFrame]:
        """
        Get data with intelligent caching - tries cache first, then API.

        Args:
            plant_id: Plant identifier
            data_type: Type of data
            start_date: Start date string
            end_date: End date string
            granularity: Data granularity
            integration: API integration object
            plant_type: Type of plant

        Returns:
            DataFrame with requested data
        """
        # Try cache first
        cached_data = self.get_cached_data(plant_id, data_type, start_date, end_date, granularity)
        if cached_data is not None:
            return cached_data

        # Cache miss - fetch from API
        logger.info(f"Cache miss for {plant_id} {data_type}, fetching from API")

        df = self._fetch_api_data(integration, plant_id, plant_type,
                                start_date, end_date, granularity)

        if df is not None and not df.empty:
            # Cache the fetched data
            self.cache_data(plant_id, data_type, start_date, end_date, df, granularity)

        return df

    def cleanup_old_cache(self, days_old: int = 30):
        """
        Clean up cache entries older than specified days.

        Args:
            days_old: Remove cache entries older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        removed_count = 0

        with self.cache_lock:
            entries_to_remove = []

            for cache_key, cache_info in self.metadata["cache_entries"].items():
                cache_time = datetime.fromisoformat(cache_info["timestamp"])
                if cache_time < cutoff_date:
                    entries_to_remove.append(cache_key)

            for cache_key in entries_to_remove:
                try:
                    # Remove cache file
                    cache_file = self._get_cache_file_path(cache_key)
                    if cache_file.exists():
                        cache_file.unlink()

                    # Remove from metadata
                    del self.metadata["cache_entries"][cache_key]
                    removed_count += 1

                except Exception as e:
                    logger.error(f"Failed to remove cache entry {cache_key}: {e}")

            if removed_count > 0:
                self._save_metadata()
                logger.info(f"Cleaned up {removed_count} old cache entries")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total_entries = len(self.metadata["cache_entries"])
        total_size = sum(
            entry.get("file_size", 0)
            for entry in self.metadata["cache_entries"].values()
        )

        return {
            "total_entries": total_entries,
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
            "last_cleanup": self.metadata.get("last_cleanup", "Never")
        }


# Global cache manager instance
api_cache_manager = APICacheManager()
