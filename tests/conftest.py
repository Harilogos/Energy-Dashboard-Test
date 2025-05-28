"""
Pytest configuration and shared fixtures for the Energy Generation Dashboard tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from pathlib import Path

# Test data constants
TEST_PLANT_ID = "TEST.PLANT.001"
TEST_PLANT_NAME = "Test Plant"
TEST_CLIENT_NAME = "Test Client"
TEST_START_DATE = date(2024, 1, 1)
TEST_END_DATE = date(2024, 1, 7)

@pytest.fixture
def sample_generation_data():
    """Sample generation data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='15min')
    return pd.DataFrame({
        'time': dates,
        'generation_kwh': np.random.uniform(50, 200, len(dates)),
        'plant_id': TEST_PLANT_ID
    })

@pytest.fixture
def sample_consumption_data():
    """Sample consumption data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='H')
    return pd.DataFrame({
        'time': dates,
        'energy_kwh': np.random.uniform(100, 300, len(dates)),
        'Plant Short Name': TEST_PLANT_ID
    })

@pytest.fixture
def sample_daily_data():
    """Sample daily aggregated data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='D')
    return pd.DataFrame({
        'date': dates,
        'generation_kwh': np.random.uniform(1000, 3000, len(dates)),
        'consumption_kwh': np.random.uniform(1500, 2500, len(dates))
    })

@pytest.fixture
def sample_tod_data():
    """Sample Time-of-Day data for testing."""
    return pd.DataFrame({
        'tod_bin': ['Peak', 'Off-Peak', 'Peak', 'Off-Peak'],
        'generation_kwh': [150, 200, 180, 120],
        'consumption_kwh': [250, 180, 220, 100],
        'time_period': ['06:00-10:00', '10:00-18:00', '18:00-22:00', '22:00-06:00']
    })

@pytest.fixture
def sample_client_data():
    """Sample client configuration data for testing."""
    return {
        "solar": {
            TEST_CLIENT_NAME: {
                "plants": [
                    {"name": "Test Solar Plant", "plant_id": "TEST.SOLAR.001"}
                ]
            }
        },
        "wind": {
            TEST_CLIENT_NAME: {
                "plants": [
                    {"name": "Test Wind Plant", "plant_id": "TEST.WIND.001"}
                ]
            }
        }
    }

@pytest.fixture
def temp_client_json(sample_client_data):
    """Create a temporary client.json file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_client_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

@pytest.fixture
def temp_csv_file(sample_consumption_data):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_consumption_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "app": {
            "title": "Test Energy Dashboard",
            "page_icon": "⚡",
            "layout": "wide"
        },
        "data": {
            "cache_ttl": 3600,
            "max_rows": 10000,
            "date_format": "%Y-%m-%d",
            "consumption_csv_path": "test_data.csv",
            "enable_smart_caching": True,
            "api_cache_ttl": 21600,
            "bulk_fetch_enabled": True,
            "bulk_fetch_months": 6
        },
        "visualization": {
            "default_height": 6,
            "default_width": 12,
            "dpi": 100,
            "colors": {
                "primary": "#4285F4",
                "consumption": "#00897B",
                "generation": "#34A853"
            }
        }
    }

@pytest.fixture
def mock_integration():
    """Mock Prescinto integration for testing."""
    mock = Mock()
    mock.fetchDataV2.return_value = pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=96, freq='15min'),
        'generation_kwh': np.random.uniform(50, 200, 96)
    })
    return mock

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing."""
    with patch('streamlit.selectbox') as mock_selectbox, \
         patch('streamlit.date_input') as mock_date_input, \
         patch('streamlit.pyplot') as mock_pyplot, \
         patch('streamlit.metric') as mock_metric, \
         patch('streamlit.columns') as mock_columns, \
         patch('streamlit.header') as mock_header, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.info') as mock_info:
        
        mock_selectbox.return_value = TEST_PLANT_NAME
        mock_date_input.return_value = TEST_START_DATE
        mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
        
        yield {
            'selectbox': mock_selectbox,
            'date_input': mock_date_input,
            'pyplot': mock_pyplot,
            'metric': mock_metric,
            'columns': mock_columns,
            'header': mock_header,
            'warning': mock_warning,
            'error': mock_error,
            'info': mock_info
        }

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    
    yield
    
    # Cleanup after test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']

class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_time_series(start_date, end_date, freq='15min', value_range=(50, 200)):
        """Generate time series data for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        values = np.random.uniform(value_range[0], value_range[1], len(dates))
        return pd.DataFrame({
            'time': dates,
            'value': values
        })
    
    @staticmethod
    def generate_plant_data(plant_id, start_date, end_date, data_type='generation'):
        """Generate plant-specific data for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq='15min')
        column_name = f'{data_type}_kwh'
        return pd.DataFrame({
            'time': dates,
            column_name: np.random.uniform(50, 200, len(dates)),
            'plant_id': plant_id
        })

# Test utilities
def assert_dataframe_structure(df, required_columns, min_rows=0):
    """Assert that a DataFrame has the required structure."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"
    
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col}"

def assert_valid_plot_data(df, x_col, y_col):
    """Assert that DataFrame contains valid plot data."""
    assert not df.empty, "DataFrame is empty"
    assert x_col in df.columns, f"Missing x-axis column: {x_col}"
    assert y_col in df.columns, f"Missing y-axis column: {y_col}"
    assert not df[y_col].isna().all(), f"All values in {y_col} are NaN"
