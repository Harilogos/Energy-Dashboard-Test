"""
Optimized Data Manager for improved performance in data fetching and preprocessing.
"""
import pandas as pd
import streamlit as st
import numpy as np
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import pickle
import os
import hashlib
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union
import threading
from pathlib import Path

from backend.config.app_config import CONFIG
from backend.logs.logger_setup import setup_logger

logger = setup_logger('data_manager', 'data_manager.log')

class OptimizedDataManager:
    """
    High-performance data manager with caching, concurrent processing, and optimized I/O.
    """

    def __init__(self):
        self.csv_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

        # Configuration
        self.enable_csv_caching = CONFIG["data"].get("enable_csv_caching", True)
        self.csv_cache_ttl = CONFIG["data"].get("csv_cache_ttl", 7200)
        self.enable_concurrent = CONFIG["data"].get("enable_concurrent_processing", True)
        self.max_workers = CONFIG["data"].get("max_concurrent_requests", 4)
        self.chunk_size = CONFIG["data"].get("chunk_size", 1000)

        logger.info(f"DataManager initialized with caching: {self.enable_csv_caching}, "
                   f"concurrent: {self.enable_concurrent}, workers: {self.max_workers}")

    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_str = "_".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _is_cache_valid(self, cache_file: Path, ttl: int) -> bool:
        """Check if cache file is still valid."""
        if not cache_file.exists():
            return False

        file_age = datetime.now().timestamp() - cache_file.stat().st_mtime
        return file_age < ttl

    @st.cache_data(ttl=7200)  # 2 hours cache
    def load_csv_optimized(_self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV with optimized reading and caching.
        """
        try:
            cache_key = _self._get_cache_key("csv", csv_path)
            cache_file = _self.cache_dir / f"{cache_key}.pkl"

            # Try to load from cache first
            if _self.enable_csv_caching and _self._is_cache_valid(cache_file, _self.csv_cache_ttl):
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    logger.info(f"Loaded CSV from cache: {csv_path}")
                    return df
                except Exception as e:
                    logger.warning(f"Cache read failed: {e}")

            # Read CSV with optimizations
            logger.info(f"Reading CSV file: {csv_path}")

            # Use efficient data types and chunked reading for large files
            dtypes = {
                'Plant Short Name': 'category',
                'energy_kwh': 'float32'
            }

            # Read in chunks for memory efficiency
            chunks = []
            for chunk in pd.read_csv(csv_path, chunksize=_self.chunk_size, dtype=dtypes, low_memory=False):
                # Optimize datetime parsing
                if 'time' in chunk.columns:
                    chunk['time'] = pd.to_datetime(chunk['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)

            # Pre-compute commonly used columns
            if 'time' in df.columns:
                df['date'] = df['time'].dt.date
                df['hour'] = df['time'].dt.hour
                df['datetime'] = df['time']

            # Save to cache
            if _self.enable_csv_caching:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
                    logger.info(f"Saved CSV to cache: {cache_file}")
                except Exception as e:
                    logger.warning(f"Cache write failed: {e}")

            logger.info(f"CSV loaded successfully: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return pd.DataFrame()

    def filter_data_optimized(self, df: pd.DataFrame, plant_id: str,
                            start_date, end_date) -> pd.DataFrame:
        """
        Optimized data filtering with vectorized operations.
        """
        try:
            if df.empty:
                return df

            # Convert dates to proper format
            start_date_obj = start_date.date() if hasattr(start_date, 'date') else start_date
            end_date_obj = end_date.date() if hasattr(end_date, 'date') else end_date

            # Handle categorical columns properly
            plant_column = df['Plant Short Name']
            if plant_column.dtype.name == 'category':
                # Convert categorical to string for comparison
                plant_mask = plant_column.astype(str) == plant_id
            else:
                plant_mask = plant_column == plant_id

            # Use vectorized operations for filtering
            mask = (
                plant_mask &
                (df['date'] >= start_date_obj) &
                (df['date'] <= end_date_obj)
            )

            filtered_df = df[mask].copy()

            # Sort by time for better performance in subsequent operations
            if 'time' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('time')

            logger.info(f"Filtered data: {len(filtered_df)} rows for {plant_id}")
            return filtered_df

        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            return pd.DataFrame()

    async def fetch_multiple_api_data(self, requests: List[Dict]) -> Dict:
        """
        Fetch multiple API requests concurrently.

        Args:
            requests: List of dicts with keys: plant_name, start_date, end_date, data_type
        """
        results = {}

        if not self.enable_concurrent:
            # Fallback to sequential processing
            for req in requests:
                key = f"{req['plant_name']}_{req['data_type']}"
                # This would call the actual API function
                results[key] = None  # Placeholder
            return results

        # Concurrent processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {}

            for req in requests:
                key = f"{req['plant_name']}_{req['data_type']}"
                # Submit API call to thread pool
                future = executor.submit(self._fetch_single_api_data, req)
                future_to_key[future] = key

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(f"API request failed for {key}: {e}")
                    results[key] = pd.DataFrame()

        return results

    def _fetch_single_api_data(self, request: Dict) -> pd.DataFrame:
        """
        Fetch single API request (placeholder for actual implementation).
        """
        # This would integrate with your existing API functions
        # For now, return empty DataFrame
        return pd.DataFrame()

    @lru_cache(maxsize=128)
    def get_plant_mapping(self) -> Dict:
        """
        Cached plant ID mapping from client.json.
        """
        try:
            import json
            client_path = os.path.join('src', 'client.json')
            with open(client_path, 'r') as f:
                client_data = json.load(f)

            plant_mapping = {}

            # Process solar plants
            for company, plants in client_data.get('solar', {}).items():
                for plant in plants:
                    if plant.get('name') and plant.get('plant_id'):
                        plant_mapping[plant['name']] = plant['plant_id']

            # Process wind plants
            for company, plants in client_data.get('wind', {}).items():
                for plant in plants:
                    if plant.get('name') and plant.get('plant_id'):
                        plant_mapping[plant['name']] = plant['plant_id']

            logger.info(f"Loaded plant mapping: {len(plant_mapping)} plants")
            return plant_mapping

        except Exception as e:
            logger.error(f"Error loading plant mapping: {e}")
            return {}

    def preprocess_consumption_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized preprocessing for consumption data.
        """
        try:
            if df.empty:
                return df

            # Vectorized operations for better performance
            processed_df = df.copy()

            # Ensure proper data types
            if 'energy_kwh' in processed_df.columns:
                processed_df['energy_kwh'] = pd.to_numeric(processed_df['energy_kwh'], errors='coerce')

            # Remove any rows with invalid data
            processed_df = processed_df.dropna(subset=['time', 'energy_kwh'])

            # Sort by time for better performance
            processed_df = processed_df.sort_values('time')

            logger.info(f"Preprocessed consumption data: {len(processed_df)} rows")
            return processed_df

        except Exception as e:
            logger.error(f"Error preprocessing consumption data: {e}")
            return pd.DataFrame()

    def clear_cache(self):
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

# Global instance
data_manager = OptimizedDataManager()
