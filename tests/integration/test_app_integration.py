"""
Integration tests for the main application.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from tests.conftest import (
    assert_dataframe_structure,
    TestDataGenerator,
    TEST_PLANT_ID,
    TEST_PLANT_NAME,
    TEST_CLIENT_NAME,
    TEST_START_DATE,
    TEST_END_DATE
)


class TestAppIntegration:
    """Integration tests for the main application flow."""
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.selectbox')
    @patch('streamlit.date_input')
    @patch('streamlit.pyplot')
    def test_main_app_flow(self, mock_pyplot, mock_date_input, mock_selectbox, mock_set_page_config, 
                          sample_client_data, sample_generation_data, sample_consumption_data):
        """Test the main application flow from start to finish."""
        from app import main
        
        # Mock Streamlit components
        mock_selectbox.side_effect = [TEST_CLIENT_NAME, TEST_PLANT_NAME]
        mock_date_input.side_effect = [TEST_START_DATE, TEST_END_DATE]
        
        # Mock data loading
        with patch('frontend.components.ui_components.load_client_data') as mock_load_client:
            mock_load_client.return_value = sample_client_data
            
            with patch('backend.data.data.get_generation_consumption_comparison') as mock_get_data:
                mock_get_data.return_value = (sample_generation_data, sample_consumption_data)
                
                with patch('backend.services.cache_initializer.initialize_cache_system'):
                    with patch('backend.logs.error_logger.setup_error_logging'):
                        # Should not raise exception
                        try:
                            main()
                        except SystemExit:
                            # Streamlit may call sys.exit(), which is normal
                            pass
                        
                        # Verify that page was configured
                        mock_set_page_config.assert_called_once()
    
    def test_data_pipeline_integration(self, sample_generation_data, sample_consumption_data):
        """Test the complete data pipeline from API to visualization."""
        from backend.data.data import get_generation_consumption_comparison, compare_generation_consumption
        from backend.utils.visualization import create_comparison_plot
        
        # Mock API integration
        with patch('backend.data.data.integration') as mock_integration:
            mock_integration.fetchDataV2.return_value = sample_generation_data
            
            with patch('backend.data.data.get_consumption_data_from_csv') as mock_csv:
                mock_csv.return_value = sample_consumption_data
                
                # Test data retrieval
                gen_data, cons_data = get_generation_consumption_comparison(TEST_PLANT_NAME, TEST_START_DATE)
                
                assert isinstance(gen_data, pd.DataFrame)
                assert isinstance(cons_data, pd.DataFrame)
                
                # Test data comparison
                comparison_data = compare_generation_consumption(gen_data, cons_data)
                
                if not comparison_data.empty:
                    assert_dataframe_structure(comparison_data, ['time'], min_rows=1)
                    
                    # Test visualization
                    fig = create_comparison_plot(comparison_data, TEST_PLANT_NAME, TEST_START_DATE)
                    assert fig is not None
                    
                    import matplotlib.pyplot as plt
                    plt.close(fig)  # Clean up
    
    def test_cache_integration_flow(self, temp_cache_dir, sample_generation_data):
        """Test the complete caching flow."""
        from backend.api.api_cache_manager import APICacheManager
        from backend.services.smart_data_fetcher import SmartDataFetcher
        
        # Initialize cache manager
        cache_manager = APICacheManager(cache_dir=temp_cache_dir)
        
        # Mock integration
        with patch('backend.services.smart_data_fetcher.api_cache_manager', cache_manager):
            with patch('backend.services.smart_data_fetcher.PrescintoIntegrationUtilities') as mock_integration_class:
                mock_integration = Mock()
                mock_integration.fetchDataV2.return_value = sample_generation_data
                mock_integration_class.return_value = mock_integration
                
                # Test smart data fetcher
                fetcher = SmartDataFetcher()
                
                # First call should fetch from API and cache
                result1 = fetcher.get_generation_data_smart(TEST_PLANT_ID, "2024-01-01", "2024-01-07", "15m")
                
                # Second call should use cache
                result2 = fetcher.get_generation_data_smart(TEST_PLANT_ID, "2024-01-01", "2024-01-07", "15m")
                
                # Both results should be valid
                if not result1.empty:
                    assert_dataframe_structure(result1, ['time'], min_rows=1)
                if not result2.empty:
                    assert_dataframe_structure(result2, ['time'], min_rows=1)
    
    def test_ui_to_data_integration(self, sample_client_data, sample_generation_data):
        """Test integration from UI components to data retrieval."""
        from frontend.components.ui_components import create_client_plant_filters
        from backend.data.data import get_generation_data_smart
        
        with patch('streamlit.selectbox') as mock_selectbox:
            with patch('streamlit.sidebar'):
                mock_selectbox.side_effect = [TEST_CLIENT_NAME, TEST_PLANT_NAME]
                
                with patch('frontend.components.ui_components.load_client_data') as mock_load:
                    mock_load.return_value = sample_client_data
                    
                    # Get UI selections
                    filters = create_client_plant_filters()
                    
                    assert filters["selected_client"] == TEST_CLIENT_NAME
                    assert filters["selected_plant"] == TEST_PLANT_NAME
                    
                    # Use selections to fetch data
                    with patch('backend.data.data.smart_data_fetcher') as mock_fetcher:
                        mock_fetcher.get_generation_data_smart.return_value = sample_generation_data
                        
                        result = get_generation_data_smart(filters["selected_plant"], TEST_START_DATE, TEST_END_DATE)
                        
                        if not result.empty:
                            assert_dataframe_structure(result, ['time'], min_rows=1)
    
    def test_error_handling_integration(self):
        """Test error handling across the application."""
        from app import main
        
        # Test with various error conditions
        with patch('backend.config.app_config.setup_page', side_effect=Exception("Config error")):
            with patch('streamlit.error') as mock_error:
                try:
                    main()
                except:
                    pass  # Expected to handle errors gracefully
                
                # Should show error to user
                mock_error.assert_called()


class TestDisplayComponentsIntegration:
    """Integration tests for display components."""
    
    def test_summary_tab_integration(self, sample_generation_data, sample_consumption_data):
        """Test summary tab display integration."""
        from src.display_components import display_generation_consumption_view
        
        with patch('backend.data.data.get_generation_consumption_comparison') as mock_get_data:
            mock_get_data.return_value = (sample_generation_data, sample_consumption_data)
            
            with patch('backend.utils.visualization.create_comparison_plot') as mock_plot:
                mock_fig = Mock()
                mock_plot.return_value = mock_fig
                
                with patch('streamlit.pyplot') as mock_pyplot:
                    # Should not raise exception
                    display_generation_consumption_view(TEST_PLANT_NAME, TEST_START_DATE, section="summary")
                    
                    # Should have created and displayed plot
                    mock_plot.assert_called_once()
                    mock_pyplot.assert_called_once()
    
    def test_tod_tab_integration(self, sample_tod_data):
        """Test ToD tab display integration."""
        from src.display_components import display_tod_binned_view
        
        with patch('backend.data.data.get_tod_binned_data') as mock_get_data:
            mock_get_data.return_value = sample_tod_data
            
            with patch('backend.utils.visualization.create_tod_binned_plot') as mock_plot:
                mock_fig = Mock()
                mock_plot.return_value = mock_fig
                
                with patch('streamlit.pyplot') as mock_pyplot:
                    # Should not raise exception
                    display_tod_binned_view(TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE, section="tod")
                    
                    # Should have created and displayed plot
                    mock_plot.assert_called_once()
                    mock_pyplot.assert_called_once()
    
    def test_power_cost_integration(self, sample_generation_data, sample_consumption_data):
        """Test power cost analysis integration."""
        from src.display_components import display_power_cost_analysis
        
        # Create cost data
        cost_data = pd.DataFrame({
            'time': sample_generation_data['time'][:min(len(sample_generation_data), len(sample_consumption_data))],
            'generation_kwh': sample_generation_data['generation_kwh'][:min(len(sample_generation_data), len(sample_consumption_data))],
            'consumption_kwh': sample_consumption_data['energy_kwh'][:min(len(sample_generation_data), len(sample_consumption_data))],
            'grid_cost': np.random.uniform(100, 300, min(len(sample_generation_data), len(sample_consumption_data))),
            'actual_cost': np.random.uniform(50, 200, min(len(sample_generation_data), len(sample_consumption_data))),
            'savings': np.random.uniform(20, 100, min(len(sample_generation_data), len(sample_consumption_data)))
        })
        
        with patch('backend.data.data.calculate_power_cost_metrics') as mock_calc:
            mock_calc.return_value = cost_data
            
            with patch('backend.data.data.get_power_cost_summary') as mock_summary:
                mock_summary.return_value = {
                    'total_grid_cost': 5000,
                    'total_actual_cost': 3000,
                    'total_savings': 2000
                }
                
                with patch('streamlit.number_input') as mock_input:
                    mock_input.return_value = 5.0  # Grid rate
                    
                    with patch('streamlit.metric') as mock_metric:
                        with patch('streamlit.pyplot') as mock_pyplot:
                            # Should not raise exception
                            display_power_cost_analysis(TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE, False)
                            
                            # Should have displayed metrics
                            mock_metric.assert_called()


class TestConfigurationIntegration:
    """Integration tests for configuration loading and usage."""
    
    def test_config_loading_integration(self):
        """Test configuration loading and usage across modules."""
        from backend.config.app_config import CONFIG, load_config
        from backend.config.api_config import get_api_credentials
        from backend.config.tod_config import get_tod_slots
        
        # Test that all configurations load successfully
        config = load_config()
        assert isinstance(config, dict)
        assert "app" in config
        assert "data" in config
        assert "visualization" in config
        
        # Test API configuration
        server, token = get_api_credentials()
        assert server is not None
        assert token is not None
        
        # Test ToD configuration
        tod_slots = get_tod_slots()
        assert isinstance(tod_slots, list)
        assert len(tod_slots) > 0
    
    def test_config_usage_in_services(self, temp_cache_dir):
        """Test configuration usage in services."""
        from backend.api.api_cache_manager import APICacheManager
        
        with patch('backend.config.app_config.CONFIG') as mock_config:
            mock_config.__getitem__.return_value = {
                "cache_ttl": 3600,
                "max_concurrent_requests": 4
            }
            
            # Should use configuration values
            cache_manager = APICacheManager(cache_dir=temp_cache_dir)
            assert cache_manager.default_ttl == 3600
            assert cache_manager.max_workers == 4


class TestEndToEndScenarios:
    """End-to-end integration test scenarios."""
    
    def test_single_day_analysis_scenario(self, sample_client_data, sample_generation_data, sample_consumption_data):
        """Test complete single-day analysis scenario."""
        # Mock the complete flow for single day analysis
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = sample_client_data
            
            with patch('backend.data.data.get_generation_consumption_comparison') as mock_get_data:
                mock_get_data.return_value = (sample_generation_data, sample_consumption_data)
                
                with patch('backend.data.data.get_tod_binned_data') as mock_tod_data:
                    mock_tod_data.return_value = pd.DataFrame({
                        'tod_bin': ['Peak', 'Off-Peak'],
                        'generation_kwh': [150, 200],
                        'consumption_kwh': [250, 180]
                    })
                    
                    with patch('streamlit.selectbox') as mock_selectbox:
                        with patch('streamlit.date_input') as mock_date_input:
                            with patch('streamlit.pyplot') as mock_pyplot:
                                mock_selectbox.side_effect = [TEST_CLIENT_NAME, TEST_PLANT_NAME]
                                mock_date_input.side_effect = [TEST_START_DATE, TEST_START_DATE]  # Same date
                                
                                from app import main
                                
                                try:
                                    main()
                                except SystemExit:
                                    pass  # Normal Streamlit behavior
                                
                                # Should have displayed multiple plots
                                assert mock_pyplot.call_count >= 2
    
    def test_multi_day_analysis_scenario(self, sample_client_data, sample_generation_data, sample_consumption_data):
        """Test complete multi-day analysis scenario."""
        # Mock the complete flow for multi-day analysis
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = sample_client_data
            
            with patch('backend.data.data.get_daily_generation_consumption_comparison') as mock_get_data:
                mock_get_data.return_value = (sample_generation_data, sample_consumption_data)
                
            with patch('backend.data.data.get_daily_consumption_data') as mock_daily_cons:
                mock_daily_cons.return_value = sample_consumption_data
                
                with patch('streamlit.selectbox') as mock_selectbox:
                    with patch('streamlit.date_input') as mock_date_input:
                        with patch('streamlit.pyplot') as mock_pyplot:
                            mock_selectbox.side_effect = [TEST_CLIENT_NAME, TEST_PLANT_NAME]
                            mock_date_input.side_effect = [TEST_START_DATE, TEST_END_DATE]  # Different dates
                            
                            from app import main
                            
                            try:
                                main()
                            except SystemExit:
                                pass  # Normal Streamlit behavior
                            
                            # Should have displayed multiple plots
                            assert mock_pyplot.call_count >= 2
    
    def test_combined_view_scenario(self, sample_client_data, sample_generation_data):
        """Test combined view scenario for clients with multiple plant types."""
        # Modify client data to have both solar and wind
        combined_client_data = {
            "solar": {TEST_CLIENT_NAME: sample_client_data["solar"][TEST_CLIENT_NAME]},
            "wind": {TEST_CLIENT_NAME: sample_client_data["wind"][TEST_CLIENT_NAME]}
        }
        
        with patch('frontend.components.ui_components.load_client_data') as mock_load:
            mock_load.return_value = combined_client_data
            
            with patch('backend.data.data.get_combined_wind_solar_generation') as mock_combined:
                mock_combined.return_value = sample_generation_data
                
                with patch('streamlit.selectbox') as mock_selectbox:
                    with patch('streamlit.date_input') as mock_date_input:
                        with patch('streamlit.pyplot') as mock_pyplot:
                            mock_selectbox.side_effect = [TEST_CLIENT_NAME, "Combined View"]
                            mock_date_input.side_effect = [TEST_START_DATE, TEST_END_DATE]
                            
                            from app import main
                            
                            try:
                                main()
                            except SystemExit:
                                pass  # Normal Streamlit behavior
                            
                            # Should have displayed combined view
                            mock_combined.assert_called()
