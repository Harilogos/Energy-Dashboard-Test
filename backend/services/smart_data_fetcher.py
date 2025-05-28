"""
Smart Data Fetcher with intelligent caching and bulk fetching capabilities.

This module provides optimized data fetching that minimizes API calls
through intelligent caching and bulk data retrieval.
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from backend.api.api_cache_manager import api_cache_manager
from backend.logs.logger_setup import setup_logger
from src.integration_utilities import PrescintoIntegrationUtilities
from backend.data.data import get_api_credentials

logger = setup_logger('smart_data_fetcher', 'smart_data_fetcher.log')

class SmartDataFetcher:
    """
    Smart data fetcher with caching and bulk fetching capabilities.
    """

    def __init__(self):
        """Initialize the smart data fetcher."""
        self.integration = None
        self.plants_info = None
        self._initialize_api()
        self._load_plants_info()

    def _initialize_api(self):
        """Initialize API integration."""
        try:
            server, token = get_api_credentials()
            self.integration = PrescintoIntegrationUtilities(server=server, token=token)
            logger.info("API integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API integration: {e}")
            self.integration = None

    def _load_plants_info(self):
        """Load plants information from client.json."""
        try:
            client_path = Path('src/client.json')
            with open(client_path, 'r') as f:
                self.plants_info = json.load(f)
            logger.info("Plants information loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load plants info: {e}")
            self.plants_info = {}

    def _get_plant_type(self, plant_id: str) -> str:
        """
        Determine plant type (solar/wind) from plant_id.

        Args:
            plant_id: Plant identifier

        Returns:
            Plant type ('solar' or 'wind')
        """
        # Check in solar plants
        for company, plants in self.plants_info.get('solar', {}).items():
            for plant in plants:
                if plant.get('plant_id') == plant_id:
                    return 'solar'

        # Check in wind plants
        for company, plants in self.plants_info.get('wind', {}).items():
            for plant in plants:
                if plant.get('plant_id') == plant_id:
                    return 'wind'

        # Default to solar if not found
        logger.warning(f"Plant type not found for {plant_id}, defaulting to solar")
        return 'solar'

    def initialize_bulk_cache(self):
        """
        Initialize bulk cache for all plants.
        This should be called once when the application starts.
        """
        if self.integration is None:
            logger.error("API integration not available for bulk caching")
            return

        logger.info("Starting bulk cache initialization for all plants")

        all_plants = []

        # Collect all plant IDs
        for plant_type in ['solar', 'wind']:
            for company, plants in self.plants_info.get(plant_type, {}).items():
                for plant in plants:
                    plant_id = plant.get('plant_id')
                    if plant_id:
                        all_plants.append({
                            'plant_id': plant_id,
                            'plant_type': plant_type,
                            'name': plant.get('name', plant_id)
                        })

        logger.info(f"Found {len(all_plants)} plants for bulk caching")

        # Bulk fetch data for each plant
        for plant_info in all_plants:
            try:
                api_cache_manager.bulk_fetch_plant_data(
                    plant_info['plant_id'],
                    self.integration,
                    plant_info['plant_type']
                )
            except Exception as e:
                logger.error(f"Bulk fetch failed for {plant_info['plant_id']}: {e}")

        logger.info("Bulk cache initialization completed")

    def get_generation_data_smart(self, plant_id: str, start_date: str, end_date: str,
                                 granularity: str = "15m") -> Optional[pd.DataFrame]:
        """
        Get generation data with smart caching.

        Args:
            plant_id: Plant identifier
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            granularity: Data granularity (15m, 1h, 1d)

        Returns:
            DataFrame with generation data or None if failed
        """
        if self.integration is None:
            logger.error("API integration not available")
            return None

        plant_type = self._get_plant_type(plant_id)

        # Try to get data from cache first
        df = api_cache_manager.get_data_with_cache(
            plant_id=plant_id,
            data_type="generation",
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            integration=self.integration,
            plant_type=plant_type
        )

        if df is not None and not df.empty:
            # Filter data to exact date range if needed
            df = self._filter_date_range(df, start_date, end_date)
            logger.info(f"Retrieved {len(df)} rows for {plant_id} ({start_date} to {end_date})")

        return df

    def _filter_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Filter DataFrame to exact date range.

        Args:
            df: Input DataFrame
            start_date: Start date string
            end_date: End date string

        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df

        try:
            # Ensure we have a time column
            if 'time' not in df.columns and 'Time' not in df.columns:
                return df

            time_col = 'time' if 'time' in df.columns else 'Time'

            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])

            # Filter by date range - handle timezone issues
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include end date

            # Remove timezone info from both sides to avoid comparison issues
            if df[time_col].dt.tz is not None:
                df[time_col] = df[time_col].dt.tz_localize(None)
            if start_dt.tz is not None:
                start_dt = start_dt.tz_localize(None)
            if end_dt.tz is not None:
                end_dt = end_dt.tz_localize(None)

            mask = (df[time_col] >= start_dt) & (df[time_col] < end_dt)
            filtered_df = df[mask].copy()

            return filtered_df

        except Exception as e:
            logger.error(f"Failed to filter date range: {e}")
            return df

    def get_multiple_plants_data(self, plant_requests: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple plants efficiently.

        Args:
            plant_requests: List of dicts with keys: plant_id, start_date, end_date, granularity

        Returns:
            Dictionary mapping plant_id to DataFrame
        """
        results = {}

        for request in plant_requests:
            plant_id = request['plant_id']
            try:
                df = self.get_generation_data_smart(
                    plant_id=plant_id,
                    start_date=request['start_date'],
                    end_date=request['end_date'],
                    granularity=request.get('granularity', '15m')
                )
                results[plant_id] = df if df is not None else pd.DataFrame()

            except Exception as e:
                logger.error(f"Failed to get data for {plant_id}: {e}")
                results[plant_id] = pd.DataFrame()

        return results

    def preload_common_data(self):
        """
        Preload commonly requested data ranges.
        This can be called periodically to warm up the cache.
        """
        if self.integration is None:
            return

        logger.info("Preloading common data ranges")

        # Common date ranges
        today = datetime.now()
        date_ranges = [
            # Last 7 days
            (today - timedelta(days=7), today),
            # Last 30 days
            (today - timedelta(days=30), today),
            # Current month
            (today.replace(day=1), today),
        ]

        # Get all plant IDs
        all_plant_ids = []
        for plant_type in ['solar', 'wind']:
            for company, plants in self.plants_info.get(plant_type, {}).items():
                for plant in plants:
                    plant_id = plant.get('plant_id')
                    if plant_id:
                        all_plant_ids.append(plant_id)

        # Preload data for common ranges
        for start_date, end_date in date_ranges:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            for plant_id in all_plant_ids[:3]:  # Limit to first 3 plants to avoid overload
                try:
                    self.get_generation_data_smart(plant_id, start_str, end_str, "15m")
                except Exception as e:
                    logger.error(f"Preload failed for {plant_id}: {e}")

        logger.info("Preloading completed")

    def get_cache_status(self) -> Dict:
        """Get cache status and statistics."""
        return api_cache_manager.get_cache_stats()

    def cleanup_cache(self, days_old: int = 30):
        """Clean up old cache entries."""
        api_cache_manager.cleanup_old_cache(days_old)


# Global smart data fetcher instance
smart_data_fetcher = SmartDataFetcher()
