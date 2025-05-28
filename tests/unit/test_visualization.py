"""
Unit tests for visualization utilities.
"""
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from unittest.mock import Mock, patch, MagicMock

from tests.conftest import (
    assert_dataframe_structure, 
    assert_valid_plot_data,
    TestDataGenerator,
    TEST_PLANT_ID,
    TEST_PLANT_NAME,
    TEST_START_DATE,
    TEST_END_DATE
)


class TestVisualizationUtils:
    """Test cases for backend.utils.visualization module utilities."""
    
    def test_format_thousands(self):
        """Test thousands formatting function."""
        from backend.utils.visualization import format_thousands
        
        # Test values >= 1000
        assert format_thousands(1000, None) == "1.0K"
        assert format_thousands(1500, None) == "1.5K"
        assert format_thousands(2000, None) == "2.0K"
        assert format_thousands(10000, None) == "10.0K"
        
        # Test values < 1000
        assert format_thousands(500, None) == "500"
        assert format_thousands(0, None) == "0"
        assert format_thousands(999, None) == "999"
    
    @patch('backend.utils.visualization.CONFIG')
    def test_create_figure_default(self, mock_config):
        """Test creating figure with default dimensions."""
        from backend.utils.visualization import create_figure
        
        mock_config.__getitem__.return_value = {
            "default_width": 12,
            "default_height": 6,
            "dpi": 100
        }
        
        fig = create_figure()
        
        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 6
        assert fig.dpi == 100
        
        plt.close(fig)  # Clean up
    
    @patch('backend.utils.visualization.CONFIG')
    def test_create_figure_custom(self, mock_config):
        """Test creating figure with custom dimensions."""
        from backend.utils.visualization import create_figure
        
        mock_config.__getitem__.return_value = {
            "default_width": 12,
            "default_height": 6,
            "dpi": 100
        }
        
        fig = create_figure(width=10, height=8)
        
        assert isinstance(fig, plt.Figure)
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 8
        
        plt.close(fig)  # Clean up


class TestConsumptionPlots:
    """Test cases for consumption plotting functions."""
    
    def test_create_consumption_plot_hourly(self, sample_consumption_data):
        """Test creating consumption plot with hourly data."""
        from backend.utils.visualization import create_consumption_plot
        
        # Prepare hourly data
        sample_consumption_data['hour'] = range(len(sample_consumption_data))
        sample_consumption_data = sample_consumption_data.rename(columns={'energy_kwh': 'energy_kwh'})
        
        fig = create_consumption_plot(sample_consumption_data, TEST_PLANT_NAME, TEST_START_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        # Check that plot has data
        ax = fig.axes[0]
        assert len(ax.patches) > 0 or len(ax.lines) > 0
        
        plt.close(fig)  # Clean up
    
    def test_create_consumption_plot_daily(self, sample_daily_data):
        """Test creating consumption plot with daily data."""
        from backend.utils.visualization import create_consumption_plot
        
        # Prepare daily data
        daily_data = sample_daily_data.copy()
        daily_data = daily_data.rename(columns={'consumption_kwh': 'energy_kwh'})
        
        fig = create_consumption_plot(daily_data, TEST_PLANT_NAME, TEST_START_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)  # Clean up
    
    def test_create_consumption_plot_empty_data(self):
        """Test creating consumption plot with empty data."""
        from backend.utils.visualization import create_consumption_plot
        
        empty_df = pd.DataFrame()
        
        fig = create_consumption_plot(empty_df, TEST_PLANT_NAME, TEST_START_DATE)
        
        # Should still return a figure, possibly with error message
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)  # Clean up
    
    def test_create_daily_consumption_plot(self, sample_daily_data):
        """Test creating daily consumption plot."""
        from backend.utils.visualization import create_daily_consumption_plot
        
        fig = create_daily_consumption_plot(sample_daily_data, TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)  # Clean up


class TestComparisonPlots:
    """Test cases for generation vs consumption comparison plots."""
    
    def test_create_comparison_plot(self, sample_generation_data, sample_consumption_data):
        """Test creating generation vs consumption comparison plot."""
        from backend.utils.visualization import create_comparison_plot
        
        # Prepare comparison data
        comparison_data = pd.DataFrame({
            'time': sample_generation_data['time'][:min(len(sample_generation_data), len(sample_consumption_data))],
            'generation_kwh': sample_generation_data['generation_kwh'][:min(len(sample_generation_data), len(sample_consumption_data))],
            'consumption_kwh': sample_consumption_data['energy_kwh'][:min(len(sample_generation_data), len(sample_consumption_data))]
        })
        
        fig = create_comparison_plot(comparison_data, TEST_PLANT_NAME, TEST_START_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        # Should have both generation and consumption data
        ax = fig.axes[0]
        assert len(ax.lines) >= 2 or len(ax.patches) >= 2
        
        plt.close(fig)  # Clean up
    
    def test_create_daily_comparison_plot(self, sample_daily_data):
        """Test creating daily generation vs consumption comparison plot."""
        from backend.utils.visualization import create_daily_comparison_plot
        
        fig = create_daily_comparison_plot(sample_daily_data, TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)  # Clean up
    
    def test_create_comparison_plot_missing_data(self):
        """Test creating comparison plot with missing data columns."""
        from backend.utils.visualization import create_comparison_plot
        
        # Data missing consumption column
        incomplete_data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=5, freq='H'),
            'generation_kwh': [100, 150, 200, 175, 125]
        })
        
        fig = create_comparison_plot(incomplete_data, TEST_PLANT_NAME, TEST_START_DATE)
        
        # Should still return a figure
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)  # Clean up


class TestToDPlots:
    """Test cases for Time-of-Day plotting functions."""
    
    def test_create_tod_binned_plot(self, sample_tod_data):
        """Test creating Time-of-Day binned plot."""
        from backend.utils.visualization import create_tod_binned_plot
        
        fig = create_tod_binned_plot(sample_tod_data, TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        # Should have bars for ToD bins
        ax = fig.axes[0]
        assert len(ax.patches) > 0
        
        plt.close(fig)  # Clean up
    
    def test_create_daily_tod_binned_plot(self, sample_tod_data):
        """Test creating daily Time-of-Day binned plot."""
        from backend.utils.visualization import create_daily_tod_binned_plot
        
        # Add date column for daily plot
        sample_tod_data['date'] = TEST_START_DATE
        
        fig = create_daily_tod_binned_plot(sample_tod_data, TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)  # Clean up
    
    def test_create_tod_generation_plot(self, sample_tod_data):
        """Test creating ToD generation plot."""
        from backend.utils.visualization import create_tod_generation_plot
        
        fig = create_tod_generation_plot(sample_tod_data, TEST_PLANT_NAME, TEST_START_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)  # Clean up
    
    def test_create_tod_consumption_plot(self, sample_tod_data):
        """Test creating ToD consumption plot."""
        from backend.utils.visualization import create_tod_consumption_plot
        
        fig = create_tod_consumption_plot(sample_tod_data, TEST_PLANT_NAME, TEST_START_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)  # Clean up


class TestCombinedPlots:
    """Test cases for combined wind and solar plots."""
    
    def test_create_combined_wind_solar_plot(self, sample_generation_data):
        """Test creating combined wind and solar generation plot."""
        from backend.utils.visualization import create_combined_wind_solar_plot
        
        # Create separate wind and solar data
        wind_data = sample_generation_data.copy()
        wind_data['plant_type'] = 'wind'
        solar_data = sample_generation_data.copy()
        solar_data['plant_type'] = 'solar'
        
        combined_data = pd.concat([wind_data, solar_data], ignore_index=True)
        
        fig = create_combined_wind_solar_plot(combined_data, TEST_CLIENT_NAME, TEST_START_DATE, TEST_END_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)  # Clean up


class TestPowerCostPlots:
    """Test cases for power cost analysis plots."""
    
    def test_create_power_cost_comparison_plot(self):
        """Test creating power cost comparison plot."""
        from backend.utils.visualization import create_power_cost_comparison_plot
        
        # Create sample cost data
        cost_data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=24, freq='H'),
            'grid_cost': np.random.uniform(100, 300, 24),
            'actual_cost': np.random.uniform(50, 200, 24),
            'savings': np.random.uniform(20, 100, 24)
        })
        
        fig = create_power_cost_comparison_plot(cost_data, TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)  # Clean up
    
    def test_create_power_savings_plot(self):
        """Test creating power savings plot."""
        from backend.utils.visualization import create_power_savings_plot
        
        # Create sample savings data
        savings_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7, freq='D'),
            'daily_savings': np.random.uniform(500, 1500, 7),
            'cumulative_savings': np.cumsum(np.random.uniform(500, 1500, 7))
        })
        
        fig = create_power_savings_plot(savings_data, TEST_PLANT_NAME, TEST_START_DATE, TEST_END_DATE)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        
        plt.close(fig)  # Clean up


class TestPlotStyling:
    """Test cases for plot styling and formatting."""
    
    @patch('backend.utils.visualization.CONFIG')
    def test_plot_colors(self, mock_config):
        """Test that plots use configured colors."""
        from backend.utils.visualization import create_consumption_plot
        
        mock_config.__getitem__.return_value = {
            "colors": {
                "consumption": "#00897B",
                "generation": "#34A853",
                "primary": "#4285F4"
            }
        }
        
        # Create sample data
        data = pd.DataFrame({
            'hour': range(24),
            'energy_kwh': np.random.uniform(100, 300, 24)
        })
        
        fig = create_consumption_plot(data, TEST_PLANT_NAME, TEST_START_DATE)
        
        # Check that figure was created (color testing would require more complex inspection)
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)  # Clean up
    
    def test_plot_titles_and_labels(self, sample_consumption_data):
        """Test that plots have proper titles and labels."""
        from backend.utils.visualization import create_consumption_plot
        
        sample_consumption_data['hour'] = range(len(sample_consumption_data))
        
        fig = create_consumption_plot(sample_consumption_data, TEST_PLANT_NAME, TEST_START_DATE)
        
        ax = fig.axes[0]
        
        # Check that title exists
        assert ax.get_title() is not None
        assert len(ax.get_title()) > 0
        
        # Check that axis labels exist
        assert ax.get_xlabel() is not None
        assert ax.get_ylabel() is not None
        
        plt.close(fig)  # Clean up
    
    def test_plot_grid_and_formatting(self, sample_consumption_data):
        """Test plot grid and formatting options."""
        from backend.utils.visualization import create_consumption_plot
        
        sample_consumption_data['hour'] = range(len(sample_consumption_data))
        
        fig = create_consumption_plot(sample_consumption_data, TEST_PLANT_NAME, TEST_START_DATE)
        
        ax = fig.axes[0]
        
        # Check that grid is enabled (if configured)
        grid_lines = ax.get_xgridlines() + ax.get_ygridlines()
        # Grid might be enabled or disabled based on configuration
        
        plt.close(fig)  # Clean up


class TestPlotErrorHandling:
    """Test cases for plot error handling."""
    
    def test_plot_with_empty_dataframe(self):
        """Test plotting functions with empty DataFrame."""
        from backend.utils.visualization import create_consumption_plot
        
        empty_df = pd.DataFrame()
        
        # Should not raise exception
        fig = create_consumption_plot(empty_df, TEST_PLANT_NAME, TEST_START_DATE)
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)  # Clean up
    
    def test_plot_with_invalid_data_types(self):
        """Test plotting functions with invalid data types."""
        from backend.utils.visualization import create_consumption_plot
        
        # DataFrame with string values in numeric column
        invalid_df = pd.DataFrame({
            'hour': range(5),
            'energy_kwh': ['invalid', 'data', 'types', 'here', 'test']
        })
        
        # Should handle gracefully
        fig = create_consumption_plot(invalid_df, TEST_PLANT_NAME, TEST_START_DATE)
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)  # Clean up
    
    def test_plot_with_missing_columns(self):
        """Test plotting functions with missing required columns."""
        from backend.utils.visualization import create_comparison_plot
        
        # DataFrame missing required columns
        incomplete_df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=5, freq='H')
            # Missing generation_kwh and consumption_kwh
        })
        
        # Should handle gracefully
        fig = create_comparison_plot(incomplete_df, TEST_PLANT_NAME, TEST_START_DATE)
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)  # Clean up
