"""
Unit tests for configuration modules.
"""
import pytest
import os
import tempfile
import yaml
from unittest.mock import patch, mock_open
from pathlib import Path

from backend.config.app_config import load_config, setup_page, DEFAULT_CONFIG
from backend.config.api_config import get_api_credentials, INTEGRATION_SERVER, INTEGRATION_TOKEN
from backend.config.tod_config import get_tod_slot, get_tod_slots, TOD_SLOTS


class TestAppConfig:
    """Test cases for app_config.py"""
    
    def test_default_config_structure(self):
        """Test that default configuration has required structure."""
        assert "app" in DEFAULT_CONFIG
        assert "data" in DEFAULT_CONFIG
        assert "visualization" in DEFAULT_CONFIG
        
        # Test app section
        app_config = DEFAULT_CONFIG["app"]
        assert "title" in app_config
        assert "page_icon" in app_config
        assert "layout" in app_config
        
        # Test data section
        data_config = DEFAULT_CONFIG["data"]
        assert "cache_ttl" in data_config
        assert "max_rows" in data_config
        assert "consumption_csv_path" in data_config
        assert "enable_smart_caching" in data_config
        
        # Test visualization section
        viz_config = DEFAULT_CONFIG["visualization"]
        assert "default_height" in viz_config
        assert "default_width" in viz_config
        assert "colors" in viz_config
    
    def test_load_config_with_existing_file(self):
        """Test loading configuration from existing YAML file."""
        test_config = {
            "app": {"title": "Test App"},
            "data": {"cache_ttl": 7200},
            "visualization": {"default_height": 8}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(test_config, f)
            temp_path = f.name
        
        try:
            with patch('backend.config.app_config.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                with patch('builtins.open', mock_open(read_data=yaml.dump(test_config))):
                    config = load_config()
                    assert config["app"]["title"] == "Test App"
                    assert config["data"]["cache_ttl"] == 7200
        finally:
            os.unlink(temp_path)
    
    def test_load_config_file_not_exists(self):
        """Test loading configuration when file doesn't exist."""
        with patch('backend.config.app_config.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            config = load_config()
            assert config == DEFAULT_CONFIG
    
    def test_load_config_invalid_yaml(self):
        """Test loading configuration with invalid YAML."""
        invalid_yaml = "invalid: yaml: content: ["
        
        with patch('backend.config.app_config.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('builtins.open', mock_open(read_data=invalid_yaml)):
                config = load_config()
                assert config == DEFAULT_CONFIG
    
    @patch('streamlit.set_page_config')
    def test_setup_page(self, mock_set_page_config):
        """Test page setup function."""
        setup_page()
        mock_set_page_config.assert_called_once()
        
        # Check that it was called with expected parameters
        call_args = mock_set_page_config.call_args[1]
        assert "page_title" in call_args
        assert "page_icon" in call_args
        assert "layout" in call_args


class TestApiConfig:
    """Test cases for api_config.py"""
    
    def test_get_api_credentials_default(self):
        """Test getting API credentials with default values."""
        server, token = get_api_credentials()
        assert server == INTEGRATION_SERVER
        assert token == INTEGRATION_TOKEN
        assert server == "IN"
        assert isinstance(token, str)
        assert len(token) > 0
    
    @patch.dict(os.environ, {'PRESCINTO_API_TOKEN': 'test-token-123'})
    def test_get_api_credentials_from_env(self):
        """Test getting API credentials from environment variable."""
        # Need to reload the module to pick up the new environment variable
        import importlib
        from backend.config import api_config
        importlib.reload(api_config)
        
        server, token = api_config.get_api_credentials()
        assert server == "IN"
        assert token == "test-token-123"
    
    def test_api_credentials_not_none(self):
        """Test that API credentials are never None."""
        server, token = get_api_credentials()
        assert server is not None
        assert token is not None
        assert len(server) > 0
        assert len(token) > 0


class TestTodConfig:
    """Test cases for tod_config.py"""
    
    def test_tod_slots_structure(self):
        """Test that TOD_SLOTS has correct structure."""
        assert isinstance(TOD_SLOTS, list)
        assert len(TOD_SLOTS) > 0
        
        for slot in TOD_SLOTS:
            assert "start_hour" in slot
            assert "end_hour" in slot
            assert "name" in slot
            assert "description" in slot
            assert isinstance(slot["start_hour"], int)
            assert isinstance(slot["end_hour"], int)
            assert 0 <= slot["start_hour"] <= 23
            assert 0 <= slot["end_hour"] <= 23
    
    def test_get_tod_slot_valid_hours(self):
        """Test getting ToD slot for valid hours."""
        # Test morning peak (6-10)
        slot = get_tod_slot(8)
        assert slot["name"] == "Peak"
        assert slot["start_hour"] == 6
        assert slot["end_hour"] == 10
        
        # Test off-peak (10-18)
        slot = get_tod_slot(14)
        assert slot["name"] == "Off-Peak"
        assert slot["start_hour"] == 10
        assert slot["end_hour"] == 18
        
        # Test evening peak (18-22)
        slot = get_tod_slot(20)
        assert slot["name"] == "Peak"
        assert slot["start_hour"] == 18
        assert slot["end_hour"] == 22
        
        # Test night off-peak (22-6)
        slot = get_tod_slot(2)
        assert slot["name"] == "Off-Peak"
        assert slot["start_hour"] == 22
        assert slot["end_hour"] == 6
    
    def test_get_tod_slot_boundary_hours(self):
        """Test getting ToD slot for boundary hours."""
        # Test exact boundary hours
        slot_6 = get_tod_slot(6)
        assert slot_6["name"] == "Peak"
        
        slot_10 = get_tod_slot(10)
        assert slot_10["name"] == "Off-Peak"
        
        slot_18 = get_tod_slot(18)
        assert slot_18["name"] == "Peak"
        
        slot_22 = get_tod_slot(22)
        assert slot_22["name"] == "Off-Peak"
    
    def test_get_tod_slot_invalid_hour(self):
        """Test getting ToD slot for invalid hours."""
        # Test negative hour
        slot = get_tod_slot(-1)
        assert slot is None
        
        # Test hour > 23
        slot = get_tod_slot(25)
        assert slot is None
    
    def test_get_tod_slots(self):
        """Test getting all ToD slots."""
        slots = get_tod_slots()
        assert isinstance(slots, list)
        assert len(slots) == len(TOD_SLOTS)
        assert slots == TOD_SLOTS
    
    def test_tod_coverage_24_hours(self):
        """Test that ToD slots cover all 24 hours."""
        covered_hours = set()
        
        for hour in range(24):
            slot = get_tod_slot(hour)
            assert slot is not None, f"Hour {hour} not covered by any ToD slot"
            covered_hours.add(hour)
        
        assert len(covered_hours) == 24, "Not all hours are covered by ToD slots"
    
    def test_tod_slot_names(self):
        """Test that ToD slot names are valid."""
        valid_names = {"Peak", "Off-Peak"}
        
        for slot in TOD_SLOTS:
            assert slot["name"] in valid_names, f"Invalid ToD slot name: {slot['name']}"
    
    def test_tod_slot_descriptions(self):
        """Test that ToD slot descriptions exist and are non-empty."""
        for slot in TOD_SLOTS:
            assert "description" in slot
            assert isinstance(slot["description"], str)
            assert len(slot["description"]) > 0


class TestConfigIntegration:
    """Integration tests for configuration modules."""
    
    def test_config_modules_import(self):
        """Test that all configuration modules can be imported."""
        try:
            from backend.config import app_config
            from backend.config import api_config
            from backend.config import tod_config
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import configuration modules: {e}")
    
    def test_config_consistency(self):
        """Test consistency between different configuration modules."""
        # Test that visualization colors are properly defined
        colors = DEFAULT_CONFIG["visualization"]["colors"]
        required_colors = ["primary", "consumption", "generation"]
        
        for color in required_colors:
            assert color in colors, f"Missing required color: {color}"
            assert isinstance(colors[color], str), f"Color {color} should be a string"
            assert colors[color].startswith("#"), f"Color {color} should be a hex color"
    
    @patch.dict(os.environ, {'TESTING': 'true'})
    def test_config_in_test_environment(self):
        """Test configuration behavior in test environment."""
        # Verify that test environment is detected
        assert os.getenv('TESTING') == 'true'
        
        # Configuration should still work in test environment
        config = load_config()
        assert config is not None
        assert isinstance(config, dict)





