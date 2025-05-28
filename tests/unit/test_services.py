"""
Unit tests for backend services.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
import pickle
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from tests.conftest import (
    assert_dataframe_structure,
    TestDataGenerator,
    TEST_PLANT_ID,
    TEST_PLANT_NAME,
    TEST_START_DATE,
    TEST_END_DATE
)


class TestSmartDataFetcher:
    """Test cases for backend.services.smart_data_fetcher module."""
    
    @patch('backend.services.smart_data_fetcher.api_cache_manager')
    @patch('backend.services.smart_data_fetcher.PrescintoIntegrationUtilities')
    def test_smart_data_fetcher_initialization(self, mock_integration_class, mock_cache_manager):
        """Test SmartDataFetcher initialization."""
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        mock_integration = Mock()
        mock_integration_class.return_value = mock_integration
        
        fetcher = SmartDataFetcher()
        
        assert fetcher.integration == mock_integration
        assert hasattr(fetcher, 'plant_type_cache')
    
    @patch('backend.services.smart_data_fetcher.api_cache_manager')
    def test_get_generation_data_smart_success(self, mock_cache_manager, sample_generation_data):
        """Test successful generation data retrieval with smart caching."""
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        mock_cache_manager.get_data_with_cache.return_value = sample_generation_data
        
        fetcher = SmartDataFetcher()
        result = fetcher.get_generation_data_smart(TEST_PLANT_ID, "2024-01-01", "2024-01-07", "15m")
        
        assert_dataframe_structure(result, ['time', 'generation_kwh'], min_rows=1)
        mock_cache_manager.get_data_with_cache.assert_called_once()
    
    @patch('backend.services.smart_data_fetcher.api_cache_manager')
    def test_get_generation_data_smart_empty_result(self, mock_cache_manager):
        """Test generation data retrieval with empty result."""
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        mock_cache_manager.get_data_with_cache.return_value = pd.DataFrame()
        
        fetcher = SmartDataFetcher()
        result = fetcher.get_generation_data_smart(TEST_PLANT_ID, "2024-01-01", "2024-01-07", "15m")
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    @patch('backend.services.smart_data_fetcher.api_cache_manager')
    def test_get_generation_data_smart_cache_error(self, mock_cache_manager):
        """Test generation data retrieval when cache fails."""
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        mock_cache_manager.get_data_with_cache.side_effect = Exception("Cache error")
        
        fetcher = SmartDataFetcher()
        result = fetcher.get_generation_data_smart(TEST_PLANT_ID, "2024-01-01", "2024-01-07", "15m")
        
        # Should return empty DataFrame on error
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_get_plant_type_solar(self):
        """Test plant type detection for solar plants."""
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        fetcher = SmartDataFetcher()
        
        # Test solar plant ID patterns
        assert fetcher._get_plant_type("SOLAR.PLANT.001") == "solar"
        assert fetcher._get_plant_type("TEST.SOLAR.001") == "solar"
        assert fetcher._get_plant_type("solar_plant_id") == "solar"
    
    def test_get_plant_type_wind(self):
        """Test plant type detection for wind plants."""
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        fetcher = SmartDataFetcher()
        
        # Test wind plant ID patterns
        assert fetcher._get_plant_type("WIND.PLANT.001") == "wind"
        assert fetcher._get_plant_type("TEST.WIND.001") == "wind"
        assert fetcher._get_plant_type("wind_plant_id") == "wind"
    
    def test_get_plant_type_unknown(self):
        """Test plant type detection for unknown plants."""
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        fetcher = SmartDataFetcher()
        
        # Test unknown plant ID
        assert fetcher._get_plant_type("UNKNOWN.PLANT.001") == "unknown"
        assert fetcher._get_plant_type("") == "unknown"
    
    def test_filter_date_range(self, sample_generation_data):
        """Test date range filtering."""
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        fetcher = SmartDataFetcher()
        
        # Filter to a smaller date range
        start_date = "2024-01-02"
        end_date = "2024-01-04"
        
        result = fetcher._filter_date_range(sample_generation_data, start_date, end_date)
        
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            # Check that dates are within range
            result_dates = pd.to_datetime(result['time']).dt.date
            assert all(date(2024, 1, 2) <= d <= date(2024, 1, 4) for d in result_dates)
    
    @patch('backend.services.smart_data_fetcher.api_cache_manager')
    def test_get_multiple_plants_data(self, mock_cache_manager, sample_generation_data):
        """Test getting data for multiple plants."""
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        mock_cache_manager.get_data_with_cache.return_value = sample_generation_data
        
        fetcher = SmartDataFetcher()
        
        plant_requests = [
            {"plant_id": "PLANT.001", "start_date": "2024-01-01", "end_date": "2024-01-07", "granularity": "15m"},
            {"plant_id": "PLANT.002", "start_date": "2024-01-01", "end_date": "2024-01-07", "granularity": "15m"}
        ]
        
        results = fetcher.get_multiple_plants_data(plant_requests)
        
        assert isinstance(results, dict)
        assert "PLANT.001" in results
        assert "PLANT.002" in results
        assert isinstance(results["PLANT.001"], pd.DataFrame)
        assert isinstance(results["PLANT.002"], pd.DataFrame)


class TestCacheInitializer:
    """Test cases for backend.services.cache_initializer module."""
    
    @patch('backend.services.cache_initializer.CONFIG')
    def test_initialize_cache_system_enabled(self, mock_config):
        """Test cache system initialization when enabled."""
        from backend.services.cache_initializer import initialize_cache_system
        
        mock_config.__getitem__.return_value = {"enable_smart_caching": True, "bulk_fetch_enabled": True}
        
        with patch('backend.services.cache_initializer.should_run_bulk_fetch') as mock_should_run:
            mock_should_run.return_value = True
            
            with patch('backend.services.cache_initializer.smart_data_fetcher') as mock_fetcher:
                mock_fetcher.initialize_bulk_cache.return_value = None
                
                with patch('backend.services.cache_initializer.mark_bulk_fetch_completed') as mock_mark:
                    # Should not raise exception
                    initialize_cache_system()
                    
                    mock_fetcher.initialize_bulk_cache.assert_called_once()
                    mock_mark.assert_called_once()
    
    @patch('backend.services.cache_initializer.CONFIG')
    def test_initialize_cache_system_disabled(self, mock_config):
        """Test cache system initialization when disabled."""
        from backend.services.cache_initializer import initialize_cache_system
        
        mock_config.__getitem__.return_value = {"enable_smart_caching": False}
        
        # Should not raise exception and should return early
        initialize_cache_system()
    
    @patch('backend.services.cache_initializer.CONFIG')
    def test_initialize_cache_system_bulk_not_needed(self, mock_config):
        """Test cache system initialization when bulk fetch not needed."""
        from backend.services.cache_initializer import initialize_cache_system
        
        mock_config.__getitem__.return_value = {"enable_smart_caching": True, "bulk_fetch_enabled": True}
        
        with patch('backend.services.cache_initializer.should_run_bulk_fetch') as mock_should_run:
            mock_should_run.return_value = False
            
            with patch('backend.services.cache_initializer.smart_data_fetcher') as mock_fetcher:
                initialize_cache_system()
                
                # Should not call bulk cache initialization
                mock_fetcher.initialize_bulk_cache.assert_not_called()
    
    def test_should_run_bulk_fetch_no_marker(self):
        """Test should_run_bulk_fetch when no marker file exists."""
        from backend.services.cache_initializer import should_run_bulk_fetch
        
        with patch('backend.services.cache_initializer.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            
            result = should_run_bulk_fetch()
            assert result == True
    
    def test_should_run_bulk_fetch_old_marker(self):
        """Test should_run_bulk_fetch when marker file is old."""
        from backend.services.cache_initializer import should_run_bulk_fetch
        
        with patch('backend.services.cache_initializer.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            # Mock old timestamp
            old_time = datetime.now() - timedelta(days=2)
            mock_path.return_value.stat.return_value.st_mtime = old_time.timestamp()
            
            result = should_run_bulk_fetch()
            assert result == True
    
    def test_should_run_bulk_fetch_recent_marker(self):
        """Test should_run_bulk_fetch when marker file is recent."""
        from backend.services.cache_initializer import should_run_bulk_fetch
        
        with patch('backend.services.cache_initializer.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            
            # Mock recent timestamp
            recent_time = datetime.now() - timedelta(hours=1)
            mock_path.return_value.stat.return_value.st_mtime = recent_time.timestamp()
            
            result = should_run_bulk_fetch()
            assert result == False
    
    def test_mark_bulk_fetch_completed(self):
        """Test marking bulk fetch as completed."""
        from backend.services.cache_initializer import mark_bulk_fetch_completed
        
        with patch('backend.services.cache_initializer.Path') as mock_path:
            mock_file = Mock()
            mock_path.return_value.open.return_value.__enter__.return_value = mock_file
            
            mark_bulk_fetch_completed()
            
            # Should create marker file
            mock_path.return_value.touch.assert_called_once()


class TestAPICacheManager:
    """Test cases for backend.api.api_cache_manager module."""
    
    def test_api_cache_manager_initialization(self, temp_cache_dir):
        """Test APICacheManager initialization."""
        from backend.api.api_cache_manager import APICacheManager
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        assert cache_manager.cache_dir == Path(temp_cache_dir)
        assert cache_manager.cache_dir.exists()
        assert hasattr(cache_manager, 'metadata')
        assert hasattr(cache_manager, 'cache_lock')
    
    def test_generate_cache_key(self, temp_cache_dir):
        """Test cache key generation."""
        from backend.api.api_cache_manager import APICacheManager
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        key = cache_manager._generate_cache_key(
            plant_id=TEST_PLANT_ID,
            data_type="generation",
            start_date="2024-01-01",
            end_date="2024-01-07",
            granularity="15m"
        )
        
        assert isinstance(key, str)
        assert len(key) > 0
        # Should be deterministic
        key2 = cache_manager._generate_cache_key(
            plant_id=TEST_PLANT_ID,
            data_type="generation",
            start_date="2024-01-01",
            end_date="2024-01-07",
            granularity="15m"
        )
        assert key == key2
    
    def test_cache_data_and_retrieve(self, temp_cache_dir, sample_generation_data):
        """Test caching data and retrieving it."""
        from backend.api.api_cache_manager import APICacheManager
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        cache_key = "test_cache_key"
        
        # Cache the data
        cache_manager._cache_data(cache_key, sample_generation_data, ttl_seconds=3600)
        
        # Retrieve the data
        retrieved_data = cache_manager._get_cached_data(cache_key)
        
        assert retrieved_data is not None
        assert isinstance(retrieved_data, pd.DataFrame)
        assert len(retrieved_data) == len(sample_generation_data)
    
    def test_cache_expiry(self, temp_cache_dir, sample_generation_data):
        """Test cache expiry functionality."""
        from backend.api.api_cache_manager import APICacheManager
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        cache_key = "test_expiry_key"
        
        # Cache with very short TTL
        cache_manager._cache_data(cache_key, sample_generation_data, ttl_seconds=0)
        
        # Should be expired immediately
        retrieved_data = cache_manager._get_cached_data(cache_key)
        assert retrieved_data is None
    
    def test_cleanup_old_cache(self, temp_cache_dir, sample_generation_data):
        """Test cleaning up old cache entries."""
        from backend.api.api_cache_manager import APICacheManager
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        # Create some cache entries
        cache_manager._cache_data("old_key", sample_generation_data, ttl_seconds=0)
        cache_manager._cache_data("new_key", sample_generation_data, ttl_seconds=3600)
        
        # Clean up old entries
        cache_manager.cleanup_old_cache(days_old=0)
        
        # Old entry should be gone, new entry should remain
        assert cache_manager._get_cached_data("old_key") is None
        assert cache_manager._get_cached_data("new_key") is not None
    
    def test_get_cache_stats(self, temp_cache_dir):
        """Test getting cache statistics."""
        from backend.api.api_cache_manager import APICacheManager
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        stats = cache_manager.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "total_entries" in stats
        assert "cache_size_mb" in stats
        assert "oldest_entry" in stats
        assert "newest_entry" in stats
    
    @patch('backend.api.api_cache_manager.PrescintoIntegrationUtilities')
    def test_get_data_with_cache_hit(self, mock_integration, temp_cache_dir, sample_generation_data):
        """Test getting data with cache hit."""
        from backend.api.api_cache_manager import APICacheManager
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        # Pre-populate cache
        cache_key = cache_manager._generate_cache_key(
            plant_id=TEST_PLANT_ID,
            data_type="generation",
            start_date="2024-01-01",
            end_date="2024-01-07",
            granularity="15m"
        )
        cache_manager._cache_data(cache_key, sample_generation_data, ttl_seconds=3600)
        
        # Mock integration
        mock_integration_instance = Mock()
        
        result = cache_manager.get_data_with_cache(
            plant_id=TEST_PLANT_ID,
            data_type="generation",
            start_date="2024-01-01",
            end_date="2024-01-07",
            granularity="15m",
            integration=mock_integration_instance,
            plant_type="solar"
        )
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Should not have called the integration (cache hit)
        mock_integration_instance.fetchDataV2.assert_not_called()
    
    @patch('backend.api.api_cache_manager.PrescintoIntegrationUtilities')
    def test_get_data_with_cache_miss(self, mock_integration, temp_cache_dir, sample_generation_data):
        """Test getting data with cache miss."""
        from backend.api.api_cache_manager import APICacheManager
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        # Mock integration to return data
        mock_integration_instance = Mock()
        mock_integration_instance.fetchDataV2.return_value = sample_generation_data
        
        result = cache_manager.get_data_with_cache(
            plant_id=TEST_PLANT_ID,
            data_type="generation",
            start_date="2024-01-01",
            end_date="2024-01-07",
            granularity="15m",
            integration=mock_integration_instance,
            plant_type="solar"
        )
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Should have called the integration (cache miss)
        mock_integration_instance.fetchDataV2.assert_called_once()


class TestServiceIntegration:
    """Integration tests for backend services."""
    
    @patch('backend.services.smart_data_fetcher.api_cache_manager')
    @patch('backend.services.cache_initializer.smart_data_fetcher')
    def test_cache_system_integration(self, mock_smart_fetcher, mock_cache_manager, sample_generation_data):
        """Test integration between cache initializer and smart data fetcher."""
        from backend.services.cache_initializer import initialize_cache_system
        
        # Mock configuration
        with patch('backend.services.cache_initializer.CONFIG') as mock_config:
            mock_config.__getitem__.return_value = {
                "enable_smart_caching": True,
                "bulk_fetch_enabled": True
            }
            
            with patch('backend.services.cache_initializer.should_run_bulk_fetch') as mock_should_run:
                mock_should_run.return_value = True
                
                # Initialize cache system
                initialize_cache_system()
                
                # Verify that smart fetcher was called
                mock_smart_fetcher.initialize_bulk_cache.assert_called_once()
    
    def test_service_error_handling(self, temp_cache_dir):
        """Test error handling in services."""
        from backend.api.api_cache_manager import APICacheManager
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        # Test with invalid data
        try:
            cache_manager._cache_data("test_key", "invalid_data", ttl_seconds=3600)
            # Should handle gracefully
        except Exception as e:
            pytest.fail(f"Service should handle errors gracefully: {e}")
    
    def test_concurrent_cache_access(self, temp_cache_dir, sample_generation_data):
        """Test concurrent access to cache."""
        from backend.api.api_cache_manager import APICacheManager
        import threading
        
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        def cache_operation(key_suffix):
            cache_key = f"concurrent_test_{key_suffix}"
            cache_manager._cache_data(cache_key, sample_generation_data, ttl_seconds=3600)
            retrieved = cache_manager._get_cached_data(cache_key)
            assert retrieved is not None
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=cache_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All operations should have completed successfully
