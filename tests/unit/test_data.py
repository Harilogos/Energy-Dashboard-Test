"""
Unit tests for data layer modules.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from tests.conftest import (
    assert_dataframe_structure, 
    assert_valid_plot_data,
    TestDataGenerator,
    TEST_PLANT_ID,
    TEST_PLANT_NAME,
    TEST_START_DATE,
    TEST_END_DATE
)


class TestDataFunctions:
    """Test cases for backend.data.data module functions."""
    
    @patch('backend.data.data.integration')
    def test_get_generation_data_smart_success(self, mock_integration, sample_generation_data):
        """Test successful generation data retrieval with smart caching."""
        from backend.data.data import get_generation_data_smart
        
        # Mock the smart data fetcher
        with patch('backend.data.data.smart_data_fetcher') as mock_fetcher:
            mock_fetcher.get_generation_data_smart.return_value = sample_generation_data
            
            result = get_generation_data_smart(TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE)
            
            assert_dataframe_structure(result, ['time', 'generation_kwh'], min_rows=1)
            mock_fetcher.get_generation_data_smart.assert_called_once()
    
    @patch('backend.data.data.integration')
    def test_get_generation_data_smart_fallback(self, mock_integration, sample_generation_data):
        """Test fallback when smart caching fails."""
        from backend.data.data import get_generation_data_smart, get_generation_data
        
        with patch('backend.data.data.smart_data_fetcher') as mock_fetcher:
            # Make smart fetcher fail
            mock_fetcher.get_generation_data_smart.side_effect = Exception("Cache error")
            
            with patch('backend.data.data.get_generation_data') as mock_fallback:
                mock_fallback.return_value = sample_generation_data
                
                result = get_generation_data_smart(TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE)
                
                assert_dataframe_structure(result, ['time', 'generation_kwh'], min_rows=1)
                mock_fallback.assert_called_once()
    
    def test_get_consumption_data_from_csv_success(self, temp_csv_file, sample_consumption_data):
        """Test successful consumption data loading from CSV."""
        from backend.data.data import get_consumption_data_from_csv
        
        with patch('backend.data.data.CONFIG') as mock_config:
            mock_config.__getitem__.return_value = {"consumption_csv_path": temp_csv_file}
            
            result = get_consumption_data_from_csv(TEST_PLANT_ID, TEST_START_DATE, TEST_END_DATE)
            
            assert isinstance(result, pd.DataFrame)
            # Should have time-related columns after processing
            assert 'time' in result.columns or 'hour' in result.columns
    
    def test_get_consumption_data_from_csv_file_not_found(self):
        """Test consumption data loading when CSV file doesn't exist."""
        from backend.data.data import get_consumption_data_from_csv
        
        with patch('backend.data.data.CONFIG') as mock_config:
            mock_config.__getitem__.return_value = {"consumption_csv_path": "nonexistent.csv"}
            
            result = get_consumption_data_from_csv(TEST_PLANT_ID, TEST_START_DATE, TEST_END_DATE)
            
            assert isinstance(result, pd.DataFrame)
            assert result.empty
    
    def test_get_plant_id_mapping(self):
        """Test plant ID mapping functionality."""
        from backend.data.data import get_plant_id
        
        with patch('backend.data.data.load_client_data') as mock_load:
            mock_load.return_value = {
                "solar": {
                    "Test Client": {
                        "plants": [{"name": TEST_PLANT_NAME, "plant_id": TEST_PLANT_ID}]
                    }
                },
                "wind": {}
            }
            
            plant_id = get_plant_id(TEST_PLANT_NAME)
            assert plant_id == TEST_PLANT_ID
    
    def test_get_plant_id_not_found(self):
        """Test plant ID mapping when plant not found."""
        from backend.data.data import get_plant_id
        
        with patch('backend.data.data.load_client_data') as mock_load:
            mock_load.return_value = {"solar": {}, "wind": {}}
            
            plant_id = get_plant_id("Nonexistent Plant")
            assert plant_id is None
    
    def test_compare_generation_consumption(self, sample_generation_data, sample_consumption_data):
        """Test generation vs consumption comparison."""
        from backend.data.data import compare_generation_consumption
        
        # Align the data for comparison
        sample_consumption_data['time'] = sample_generation_data['time'][:len(sample_consumption_data)]
        sample_consumption_data = sample_consumption_data.rename(columns={'energy_kwh': 'consumption_kwh'})
        
        result = compare_generation_consumption(sample_generation_data, sample_consumption_data)
        
        assert_dataframe_structure(result, ['time', 'generation_kwh', 'consumption_kwh'], min_rows=1)
        assert len(result) > 0
    
    def test_get_tod_binned_data(self, sample_generation_data):
        """Test Time-of-Day binned data generation."""
        from backend.data.data import get_tod_binned_data
        
        with patch('backend.data.data.get_generation_consumption_comparison') as mock_get_data:
            # Create mock data with consumption
            mock_gen_data = sample_generation_data.copy()
            mock_cons_data = sample_generation_data.copy()
            mock_cons_data = mock_cons_data.rename(columns={'generation_kwh': 'consumption_kwh'})
            mock_get_data.return_value = (mock_gen_data, mock_cons_data)
            
            result = get_tod_binned_data(TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE)
            
            if not result.empty:
                assert 'tod_bin' in result.columns
                # Should have ToD classifications
                tod_values = result['tod_bin'].unique()
                assert any(val in ['Peak', 'Off-Peak'] for val in tod_values)
    
    def test_calculate_power_cost_metrics(self, sample_generation_data, sample_consumption_data):
        """Test power cost metrics calculation."""
        from backend.data.data import calculate_power_cost_metrics
        
        grid_rate = 5.0  # Rs per kWh
        
        with patch('backend.data.data.get_generation_consumption_comparison') as mock_get_data:
            # Prepare mock data
            mock_gen_data = sample_generation_data.copy()
            mock_cons_data = sample_consumption_data.copy()
            mock_cons_data['time'] = mock_gen_data['time'][:len(mock_cons_data)]
            mock_cons_data = mock_cons_data.rename(columns={'energy_kwh': 'consumption_kwh'})
            mock_get_data.return_value = (mock_gen_data, mock_cons_data)
            
            result = calculate_power_cost_metrics(TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE, grid_rate)
            
            if not result.empty:
                expected_columns = ['time', 'generation_kwh', 'consumption_kwh', 'grid_power_kwh', 
                                  'grid_cost', 'actual_cost', 'savings']
                assert_dataframe_structure(result, expected_columns, min_rows=1)
                
                # Verify cost calculations
                assert (result['grid_cost'] >= 0).all()
                assert (result['actual_cost'] >= 0).all()


class TestDataValidation:
    """Test cases for data validation functions."""
    
    def test_validate_date_range(self):
        """Test date range validation."""
        from backend.data.data import validate_date_range
        
        # Valid date range
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 7)
        assert validate_date_range(start_date, end_date) == True
        
        # Invalid date range (end before start)
        start_date = date(2024, 1, 7)
        end_date = date(2024, 1, 1)
        assert validate_date_range(start_date, end_date) == False
        
        # Same date (valid for single day)
        start_date = end_date = date(2024, 1, 1)
        assert validate_date_range(start_date, end_date) == True
    
    def test_validate_plant_name(self):
        """Test plant name validation."""
        from backend.data.data import validate_plant_name
        
        # Valid plant name
        assert validate_plant_name(TEST_PLANT_NAME) == True
        
        # Invalid plant names
        assert validate_plant_name("") == False
        assert validate_plant_name(None) == False
        assert validate_plant_name("   ") == False
    
    def test_validate_dataframe_structure(self):
        """Test DataFrame structure validation."""
        from backend.data.data import validate_dataframe_structure
        
        # Valid DataFrame
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=5, freq='H'),
            'generation_kwh': [100, 150, 200, 175, 125]
        })
        assert validate_dataframe_structure(df, ['time', 'generation_kwh']) == True
        
        # Missing required columns
        df_invalid = pd.DataFrame({'time': pd.date_range('2024-01-01', periods=5, freq='H')})
        assert validate_dataframe_structure(df_invalid, ['time', 'generation_kwh']) == False
        
        # Empty DataFrame
        df_empty = pd.DataFrame()
        assert validate_dataframe_structure(df_empty, ['time']) == False


class TestDataTransformation:
    """Test cases for data transformation functions."""
    
    def test_resample_to_hourly(self, sample_generation_data):
        """Test resampling data to hourly frequency."""
        from backend.data.data import resample_to_hourly
        
        # Ensure we have 15-minute data
        sample_generation_data['time'] = pd.date_range('2024-01-01', periods=len(sample_generation_data), freq='15min')
        
        result = resample_to_hourly(sample_generation_data, 'generation_kwh')
        
        assert_dataframe_structure(result, ['time', 'generation_kwh'], min_rows=1)
        
        # Check that frequency is hourly
        time_diff = result['time'].diff().dropna().iloc[0]
        assert time_diff == pd.Timedelta(hours=1)
    
    def test_resample_to_daily(self, sample_generation_data):
        """Test resampling data to daily frequency."""
        from backend.data.data import resample_to_daily
        
        # Ensure we have hourly data spanning multiple days
        sample_generation_data['time'] = pd.date_range('2024-01-01', periods=len(sample_generation_data), freq='H')
        
        result = resample_to_daily(sample_generation_data, 'generation_kwh')
        
        assert_dataframe_structure(result, ['date', 'generation_kwh'], min_rows=1)
        
        # Check that we have daily aggregation
        assert len(result) <= len(sample_generation_data) // 24 + 1
    
    def test_add_tod_classification(self, sample_generation_data):
        """Test adding Time-of-Day classification to data."""
        from backend.data.data import add_tod_classification
        
        # Ensure time column has proper datetime format
        sample_generation_data['time'] = pd.to_datetime(sample_generation_data['time'])
        
        result = add_tod_classification(sample_generation_data)
        
        assert_dataframe_structure(result, ['time', 'generation_kwh', 'tod_bin'], min_rows=1)
        
        # Check ToD classifications
        tod_values = result['tod_bin'].unique()
        assert all(val in ['Peak', 'Off-Peak'] for val in tod_values)
    
    def test_aggregate_by_tod(self, sample_tod_data):
        """Test aggregation by Time-of-Day bins."""
        from backend.data.data import aggregate_by_tod
        
        result = aggregate_by_tod(sample_tod_data, ['generation_kwh', 'consumption_kwh'])
        
        assert_dataframe_structure(result, ['tod_bin', 'generation_kwh', 'consumption_kwh'], min_rows=1)
        
        # Check that aggregation worked
        assert len(result) <= len(sample_tod_data)
        assert result['generation_kwh'].sum() <= sample_tod_data['generation_kwh'].sum()


class TestDataCaching:
    """Test cases for data caching functionality."""
    
    def test_cache_key_generation(self):
        """Test cache key generation for data requests."""
        from backend.data.data import generate_cache_key
        
        key = generate_cache_key(TEST_PLANT_ID, TEST_START_DATE, TEST_END_DATE, "generation", "15m")
        
        assert isinstance(key, str)
        assert len(key) > 0
        assert TEST_PLANT_ID in key or str(hash(TEST_PLANT_ID)) in key
    
    def test_cache_expiry_check(self):
        """Test cache expiry checking."""
        from backend.data.data import is_cache_expired
        
        # Recent timestamp (not expired)
        recent_time = datetime.now() - timedelta(minutes=30)
        assert is_cache_expired(recent_time, ttl_seconds=3600) == False
        
        # Old timestamp (expired)
        old_time = datetime.now() - timedelta(hours=2)
        assert is_cache_expired(old_time, ttl_seconds=3600) == True
    
    @patch('backend.data.data.CONFIG')
    def test_smart_caching_enabled_check(self, mock_config):
        """Test smart caching enabled configuration check."""
        from backend.data.data import is_smart_caching_enabled
        
        # Caching enabled
        mock_config.__getitem__.return_value = {"enable_smart_caching": True}
        assert is_smart_caching_enabled() == True
        
        # Caching disabled
        mock_config.__getitem__.return_value = {"enable_smart_caching": False}
        assert is_smart_caching_enabled() == False
