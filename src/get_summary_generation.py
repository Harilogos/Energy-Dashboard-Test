"""
Function for fetching solar generation data for the summary tab for a single day.
"""
import pandas as pd
import streamlit as st
from functools import wraps
import time
import traceback
from backend.logs.logger_setup import setup_logger
from src.integration_utilities import PrescintoIntegrationUtilities
from backend.config.api_config import get_api_credentials

# Configure logging
logger = setup_logger('summary_generation', 'data.log')

# Initialize API integration
INTEGRATION_SERVER, INTEGRATION_TOKEN = get_api_credentials()
integration = PrescintoIntegrationUtilities(server=INTEGRATION_SERVER, token=INTEGRATION_TOKEN)

def retry_on_exception(max_retries=3, retry_delay=1):
    """Decorator to retry a function on exception"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            logger.error(f"Function {func.__name__} failed after {max_retries} attempts")
            raise last_exception
        return wrapper
    return decorator

@st.cache_data(ttl=3600)
@retry_on_exception()
def get_summary_generation_data(plant_name, date):
    """
    Get solar generation data for the summary tab for a single day
    
    Args:
        plant_name (str): Name of the plant
        date (datetime): Date to retrieve data for
        
    Returns:
        DataFrame: Generation data with 'time' as first column and generation values
    """
    try:
        # Format date for API call
        date_str = date.strftime('%Y-%m-%d')
        
        # For solar plants
        df = integration.fetchDataV2(
            plant_name,      # pName
            "Plant",         # catList
            ["Daily Energy"], # paramList
            None,            # deviceList
            date_str,        # sDate
            date_str,        # eDate
            granularity="1d",
            condition={"Daily Energy": "max"}
        )
        
        # Check if API returned None or empty DataFrame
        if df is None or df.empty:
            logger.warning(f"No data returned from API for {plant_name}")
            return pd.DataFrame()
        
        # Process the dataframe
        result_df = pd.DataFrame()
        result_df['time'] = pd.to_datetime(df.iloc[:, 0])  # First column is time
        result_df['Generation'] = df.iloc[:, 1]  # Second column is generation data
        
        logger.info(f"Retrieved generation data for {plant_name} on {date_str}")
        return result_df
        
    except Exception as e:
        logger.error(f"Failed to retrieve generation data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()
