"""
Unit tests for UI components.
"""
import pytest
import json
import tempfile
import os
from datetime import date, timedelta
from unittest.mock import Mock, patch, mock_open

from tests.conftest import TEST_PLANT_NAME, TEST_CLIENT_NAME, TEST_START_DATE


class TestUIComponents:
    """Test cases for frontend.components.ui_components module."""
    
    def test_load_client_data_success(self, sample_client_data, temp_client_json):
        """Test successful loading of client data from JSON."""
        from frontend.components.ui_components import load_client_data
        
        with patch('frontend.components.ui_components.os.path.join') as mock_join:
            mock_join.return_value = temp_client_json
            
            result = load_client_data()
            
            assert isinstance(result, dict)
            assert "solar" in result
            assert "wind" in result
            assert TEST_CLIENT_NAME in result["solar"]
    
    def test_load_client_data_file_not_found(self):
        """Test loading client data when file doesn't exist."""
        from frontend.components.ui_components import load_client_data
        
        with patch('frontend.components.ui_components.os.path.join') as mock_join:
            mock_join.return_value = "nonexistent.json"
            
            result = load_client_data()
            
            assert isinstance(result, dict)
            assert result == {"solar": {}, "wind": {}}
    
    def test_load_client_data_invalid_json(self):
        """Test loading client data with invalid JSON."""
        from frontend.components.ui_components import load_client_data
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            temp_path = f.name
        
        try:
            with patch('frontend.components.ui_components.os.path.join') as mock_join:
                mock_join.return_value = temp_path
                
                result = load_client_data()
                
                assert isinstance(result, dict)
                assert result == {"solar": {}, "wind": {}}
        finally:
            os.unlink(temp_path)
    
    @patch('streamlit.selectbox')
    @patch('streamlit.sidebar')
    def test_create_client_plant_filters_single_client(self, mock_sidebar, mock_selectbox, sample_client_data):
        """Test creating client and plant filters with single client."""
        from frontend.components.ui_components import create_client_plant_filters
        
        # Mock single client data
        single_client_data = {
            "solar": {TEST_CLIENT_NAME: sample_client_data["solar"][TEST_CLIENT_NAME]},
            "wind": {}
        }
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = single_client_data
            mock_selectbox.side_effect = [TEST_CLIENT_NAME, TEST_PLANT_NAME]
            
            result = create_client_plant_filters()
            
            assert isinstance(result, dict)
            assert "selected_client" in result
            assert "selected_plant" in result
            assert "has_solar" in result
            assert "has_wind" in result
            assert result["selected_client"] == TEST_CLIENT_NAME
            assert result["has_solar"] == True
            assert result["has_wind"] == False
    
    @patch('streamlit.selectbox')
    @patch('streamlit.sidebar')
    def test_create_client_plant_filters_multiple_clients(self, mock_sidebar, mock_selectbox, sample_client_data):
        """Test creating client and plant filters with multiple clients."""
        from frontend.components.ui_components import create_client_plant_filters
        
        # Add another client
        multi_client_data = sample_client_data.copy()
        multi_client_data["solar"]["Another Client"] = {
            "plants": [{"name": "Another Plant", "plant_id": "ANOTHER.001"}]
        }
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = multi_client_data
            mock_selectbox.side_effect = [TEST_CLIENT_NAME, TEST_PLANT_NAME]
            
            result = create_client_plant_filters()
            
            assert isinstance(result, dict)
            assert result["selected_client"] == TEST_CLIENT_NAME
            # Should have called selectbox for client selection
            assert mock_selectbox.call_count >= 2
    
    @patch('streamlit.selectbox')
    @patch('streamlit.sidebar')
    def test_create_client_plant_filters_combined_view(self, mock_sidebar, mock_selectbox, sample_client_data):
        """Test creating filters with combined view option."""
        from frontend.components.ui_components import create_client_plant_filters
        
        # Client with both solar and wind
        both_types_data = {
            "solar": {TEST_CLIENT_NAME: sample_client_data["solar"][TEST_CLIENT_NAME]},
            "wind": {TEST_CLIENT_NAME: sample_client_data["wind"][TEST_CLIENT_NAME]}
        }
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = both_types_data
            mock_selectbox.side_effect = [TEST_CLIENT_NAME, "Combined View"]
            
            result = create_client_plant_filters()
            
            assert result["selected_plant"] == "Combined View"
            assert result["has_solar"] == True
            assert result["has_wind"] == True
    
    @patch('streamlit.date_input')
    @patch('streamlit.sidebar')
    def test_create_date_filters_default(self, mock_sidebar, mock_date_input):
        """Test creating date filters with default values."""
        from frontend.components.ui_components import create_date_filters
        
        # Mock date inputs
        today = date.today()
        week_ago = today - timedelta(days=7)
        mock_date_input.side_effect = [week_ago, today]
        
        start_date, end_date = create_date_filters()
        
        assert start_date == week_ago
        assert end_date == today
        assert mock_date_input.call_count == 2
    
    @patch('streamlit.date_input')
    @patch('streamlit.sidebar')
    def test_create_date_filters_single_day(self, mock_sidebar, mock_date_input):
        """Test creating date filters for single day selection."""
        from frontend.components.ui_components import create_date_filters
        
        # Mock same date for both inputs
        target_date = TEST_START_DATE
        mock_date_input.side_effect = [target_date, target_date]
        
        start_date, end_date = create_date_filters()
        
        assert start_date == target_date
        assert end_date == target_date
    
    @patch('streamlit.date_input')
    @patch('streamlit.sidebar')
    def test_create_date_filters_invalid_range(self, mock_sidebar, mock_date_input):
        """Test creating date filters with invalid range (end before start)."""
        from frontend.components.ui_components import create_date_filters
        
        # Mock invalid date range
        start_date = date(2024, 1, 7)
        end_date = date(2024, 1, 1)
        mock_date_input.side_effect = [start_date, end_date]
        
        with patch('streamlit.error') as mock_error:
            result_start, result_end = create_date_filters()
            
            # Should show error and potentially correct the dates
            mock_error.assert_called()
    
    def test_get_plant_options_solar_only(self, sample_client_data):
        """Test getting plant options for solar-only client."""
        from frontend.components.ui_components import get_plant_options
        
        solar_only_data = {
            "solar": {TEST_CLIENT_NAME: sample_client_data["solar"][TEST_CLIENT_NAME]},
            "wind": {}
        }
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = solar_only_data
            
            options = get_plant_options(TEST_CLIENT_NAME)
            
            assert isinstance(options, list)
            assert len(options) > 0
            # Should not include "Combined View" for single type
            assert "Combined View" not in [opt if isinstance(opt, str) else opt.get('name', '') for opt in options]
    
    def test_get_plant_options_both_types(self, sample_client_data):
        """Test getting plant options for client with both solar and wind."""
        from frontend.components.ui_components import get_plant_options
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = sample_client_data
            
            options = get_plant_options(TEST_CLIENT_NAME)
            
            assert isinstance(options, list)
            assert len(options) > 0
            # Should include "Combined View" for multiple types
            option_names = [opt if isinstance(opt, str) else opt.get('name', '') for opt in options]
            assert "Combined View" in option_names
    
    def test_get_plant_options_no_plants(self):
        """Test getting plant options for client with no plants."""
        from frontend.components.ui_components import get_plant_options
        
        empty_data = {"solar": {}, "wind": {}}
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = empty_data
            
            options = get_plant_options("Nonexistent Client")
            
            assert isinstance(options, list)
            assert len(options) == 0


class TestUIHelpers:
    """Test cases for UI helper functions."""
    
    def test_format_plant_name_display(self):
        """Test formatting plant names for display."""
        from frontend.components.ui_components import format_plant_name_display
        
        # Test with plant object
        plant_obj = {"name": "Test Solar Plant", "plant_id": "TEST.SOLAR.001"}
        formatted = format_plant_name_display(plant_obj)
        assert "Test Solar Plant" in formatted
        assert "TEST.SOLAR.001" not in formatted  # Should not show plant_id
        
        # Test with string
        plant_str = "Test Plant Name"
        formatted = format_plant_name_display(plant_str)
        assert formatted == plant_str
    
    def test_extract_plant_id_from_selection(self):
        """Test extracting plant ID from selection."""
        from frontend.components.ui_components import extract_plant_id_from_selection
        
        # Test with plant object
        plant_obj = {"name": "Test Solar Plant", "plant_id": "TEST.SOLAR.001"}
        plant_id = extract_plant_id_from_selection(plant_obj)
        assert plant_id == "TEST.SOLAR.001"
        
        # Test with string (should return the string itself)
        plant_str = "TEST.PLANT.001"
        plant_id = extract_plant_id_from_selection(plant_str)
        assert plant_id == plant_str
        
        # Test with Combined View
        plant_id = extract_plant_id_from_selection("Combined View")
        assert plant_id == "Combined View"
    
    def test_validate_client_selection(self, sample_client_data):
        """Test validating client selection."""
        from frontend.components.ui_components import validate_client_selection
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = sample_client_data
            
            # Valid client
            assert validate_client_selection(TEST_CLIENT_NAME) == True
            
            # Invalid client
            assert validate_client_selection("Nonexistent Client") == False
            
            # Empty/None client
            assert validate_client_selection("") == False
            assert validate_client_selection(None) == False
    
    def test_get_client_plant_types(self, sample_client_data):
        """Test getting plant types for a client."""
        from frontend.components.ui_components import get_client_plant_types
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = sample_client_data
            
            plant_types = get_client_plant_types(TEST_CLIENT_NAME)
            
            assert isinstance(plant_types, dict)
            assert "has_solar" in plant_types
            assert "has_wind" in plant_types
            assert plant_types["has_solar"] == True
            assert plant_types["has_wind"] == True
    
    def test_get_client_plant_types_solar_only(self, sample_client_data):
        """Test getting plant types for solar-only client."""
        from frontend.components.ui_components import get_client_plant_types
        
        solar_only_data = {
            "solar": {TEST_CLIENT_NAME: sample_client_data["solar"][TEST_CLIENT_NAME]},
            "wind": {}
        }
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = solar_only_data
            
            plant_types = get_client_plant_types(TEST_CLIENT_NAME)
            
            assert plant_types["has_solar"] == True
            assert plant_types["has_wind"] == False


class TestUIIntegration:
    """Integration tests for UI components."""
    
    @patch('streamlit.selectbox')
    @patch('streamlit.date_input')
    @patch('streamlit.sidebar')
    def test_complete_filter_creation_flow(self, mock_sidebar, mock_date_input, mock_selectbox, sample_client_data):
        """Test complete flow of creating all filters."""
        from frontend.components.ui_components import create_client_plant_filters, create_date_filters
        
        # Setup mocks
        mock_selectbox.side_effect = [TEST_CLIENT_NAME, TEST_PLANT_NAME]
        today = date.today()
        week_ago = today - timedelta(days=7)
        mock_date_input.side_effect = [week_ago, today]
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = sample_client_data
            
            # Create filters
            plant_filters = create_client_plant_filters()
            start_date, end_date = create_date_filters()
            
            # Verify results
            assert plant_filters["selected_client"] == TEST_CLIENT_NAME
            assert plant_filters["selected_plant"] == TEST_PLANT_NAME
            assert start_date == week_ago
            assert end_date == today
    
    def test_ui_components_error_handling(self):
        """Test error handling in UI components."""
        from frontend.components.ui_components import load_client_data, create_client_plant_filters
        
        # Test with file system errors
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            result = load_client_data()
            assert result == {"solar": {}, "wind": {}}
        
        # Test with JSON decode errors
        with patch('builtins.open', mock_open(read_data="invalid json")):
            with patch('frontend.components.ui_components.os.path.join') as mock_join:
                mock_join.return_value = "test.json"
                result = load_client_data()
                assert result == {"solar": {}, "wind": {}}
