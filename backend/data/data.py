"""
Data retrieval and processing functions.

This module is organized into two main sections:
1. Summary Section - Functions that support plots in the Summary tab
2. ToD (Time of Day) Section - Functions that support plots in the ToD tab

Within each section, functions are arranged in the order they appear in the frontend.
"""
import pandas as pd
import numpy as np
import streamlit as st
import traceback
from datetime import datetime, timedelta
import os
from functools import wraps
import time
from backend.config.api_config import get_api_credentials
from src.integration_utilities import PrescintoIntegrationUtilities
# Configure logging
from backend.logs.logger_setup import setup_logger

logger = setup_logger('data', 'data.log')

# Import smart caching if enabled
try:
    from backend.config.app_config import CONFIG
    SMART_CACHING_ENABLED = CONFIG["data"].get("enable_smart_caching", False)
    if SMART_CACHING_ENABLED:
        logger.info("Smart caching enabled")
    else:
        logger.info("Smart caching disabled")
except Exception as e:
    SMART_CACHING_ENABLED = False
    logger.warning(f"Smart caching not available: {e}")

# Initialize API integration
try:
    INTEGRATION_SERVER, INTEGRATION_TOKEN = get_api_credentials()
    logger.info(f"Initializing Prescinto API integration with server: {INTEGRATION_SERVER}")
    integration = PrescintoIntegrationUtilities(server=INTEGRATION_SERVER, token=INTEGRATION_TOKEN)
    logger.info(f"API integration initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize API integration: {e}")
    logger.error(traceback.format_exc())
    # Create a placeholder integration object to avoid errors
    integration = None

# Snowflake functions have been removed as we're now using the API exclusively

#######################################################################
# Utility Functions - Used across both Summary and ToD sections
#######################################################################

def get_plant_display_name(plant_obj):
    """
    Extract the display name from a plant object or return the name if it's a string.

    Args:
        plant_obj: Plant object (dict) or name (str)

    Returns:
        str: Plant display name for UI
    """
    if isinstance(plant_obj, dict):
        return plant_obj.get('name', 'Unknown Plant')
    return plant_obj

def get_plant_id(plant_name):
    """
    Get the plant_id from the plant name using client.json configuration.
    If plant_name is already a plant_id, it will be returned as is.

    Args:
        plant_name (str or dict): Name, ID, or plant object

    Returns:
        str: plant_id for API calls
    """
    import json
    import os

    # Extract plant_id if plant_name is a dictionary
    if isinstance(plant_name, dict):
        plant_id = plant_name.get('plant_id', '')
        if plant_id:
            return plant_id
        plant_name = plant_name.get('name', 'Unknown')
        # Only log if debugging is needed
        # logger.info(f"Extracted plant name from dictionary: {plant_name}")

    # Special case for "Combined View" which is not a real plant
    if plant_name == "Combined View":
        # logger.info(f"Plant {plant_name} is a special view")
        return plant_name

    try:
        # Load client.json
        client_path = os.path.join('src', 'client.json')
        with open(client_path, 'r') as f:
            client_data = json.load(f)

        # Check if the input is already a plant_id
        # Check solar plants
        for company, plants in client_data.get('solar', {}).items():
            for plant in plants:
                if plant.get('plant_id') == plant_name:
                    # logger.debug(f"Input {plant_name} is already a plant_id")
                    return plant_name
                if plant.get('name') == plant_name:
                    plant_id = plant.get('plant_id')
                    # logger.debug(f"Found plant_id {plant_id} for plant name {plant_name}")
                    return plant_id

        # Check wind plants
        for company, plants in client_data.get('wind', {}).items():
            for plant in plants:
                if plant.get('plant_id') == plant_name:
                    # logger.debug(f"Input {plant_name} is already a plant_id")
                    return plant_name
                if plant.get('name') == plant_name:
                    plant_id = plant.get('plant_id')
                    # logger.debug(f"Found plant_id {plant_id} for plant name {plant_name}")
                    return plant_id

        # If not found, log warning and return the original name
        logger.warning(f"Plant {plant_name} not found in client.json, using as is")
        return plant_name
    except Exception as e:
        logger.error(f"Error getting plant_id: {e}")
        logger.error(traceback.format_exc())
        # Return the original name if there's an error
        return plant_name

def is_solar_plant(plant_name):
    """
    Determine if a plant is solar or wind based on the client.json configuration

    Args:
        plant_name (str): Name of the plant

    Returns:
        bool: True if solar plant, False if wind plant
    """
    import json
    import os

    # Extract plant_id if plant_name is a dictionary
    if isinstance(plant_name, dict):
        plant_id = plant_name.get('plant_id', '')
        plant_name = plant_id if plant_id else plant_name.get('name', 'Unknown')
        # logger.debug(f"Extracted plant name from dictionary: {plant_name}")

    # Special case for "Combined View" which is not a real plant
    if plant_name == "Combined View":
        # logger.debug(f"Plant {plant_name} is a special view, treating as solar")
        return True

    try:
        # Load client.json
        client_path = os.path.join('src', 'client.json')
        with open(client_path, 'r') as f:
            client_data = json.load(f)

        # Check if plant is in solar section
        for company, plants in client_data.get('solar', {}).items():
            for plant in plants:
                if plant.get('plant_id') == plant_name or plant.get('name') == plant_name:
                    # logger.debug(f"Plant {plant_name} identified as solar")
                    return True

        # Check if plant is in wind section
        for company, plants in client_data.get('wind', {}).items():
            for plant in plants:
                if plant.get('plant_id') == plant_name or plant.get('name') == plant_name:
                    # logger.debug(f"Plant {plant_name} identified as wind")
                    return False

        # If not found, log warning and default to solar
        logger.warning(f"Plant {plant_name} not found in client.json, defaulting to solar")
        return True
    except Exception as e:
        logger.error(f"Error determining plant type: {e}")
        logger.error(traceback.format_exc())
        # Default to solar if there's an error
        return True

def standardize_dataframe_columns(df):
    """
    Standardize DataFrame column names to lowercase and handle common column name variations

    Args:
        df (DataFrame): DataFrame to standardize

    Returns:
        DataFrame: DataFrame with standardized column names
    """
    if df.empty:
        return df

    # Convert all column names to lowercase
    df.columns = [col.lower() for col in df.columns]

    # Handle common column name variations
    column_mapping = {
        'datevalue': 'date',
        'date_value': 'date',
        'plant_generation': 'generation_kwh',
        'generation': 'generation_kwh',
        'energy_kwh': 'consumption_kwh',
        'consumption': 'consumption_kwh',
        'plant_long_name': 'plant_name'
    }

    # Apply mapping for columns that exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    return df



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

@st.cache_data(ttl=3600)  # Cache data for 1 hour
@retry_on_exception()
def get_plants():
    """Get list of available plants from client.json"""
    import json
    import os

    try:
        # Load client.json
        client_path = os.path.join('src', 'client.json')
        with open(client_path, 'r') as f:
            client_data = json.load(f)

        plants = []

        # Get solar plants
        for company, company_plants in client_data.get('solar', {}).items():
            for plant in company_plants:
                if plant.get('name'):
                    plants.append(plant.get('name'))

        # Get wind plants
        for company, company_plants in client_data.get('wind', {}).items():
            for plant in company_plants:
                if plant.get('name'):
                    plants.append(plant.get('name'))

        # Sort plants alphabetically
        plants.sort()

        logger.info(f"Retrieved {len(plants)} plants from client.json")
        return plants
    except Exception as e:
        logger.error(f"Failed to retrieve plants from client.json: {e}")
        logger.error(traceback.format_exc())
        return []
















def get_consumption_data_from_csv(plant_name, start_date, end_date=None):
    """
    Get consumption data from CSV file for the specified plant and date

    Args:
        plant_name (str): Name of the plant
        start_date (datetime): Date to retrieve data for
        end_date (datetime, optional): End date for date range. If None, only start_date is used.

    Returns:
        DataFrame: Consumption data grouped by plant and hour
    """
    from backend.config.app_config import CONFIG

    # If end_date is not provided, use start_date
    if end_date is None:
        end_date = start_date

    # Use path from config if not provided, with fallback to default path
    try:
        csv_path = CONFIG["data"].get("consumption_csv_path",
                                     "Data/csv/Consumption data Cloud nine - processed_data.csv")
    except (KeyError, TypeError):
        # Fallback to default path if CONFIG doesn't have the expected structure
        csv_path = "Data/csv/Consumption data Cloud nine - processed_data.csv"
        logger.warning(f"consumption_csv_path not found in config, using default: {csv_path}")

    # Get plant_id from plant_name using client.json
    plant_id = get_plant_id(plant_name)
    logger.info(f"Loading consumption data from: {csv_path} for plant_id: {plant_id}, date: {start_date}")

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Convert time column to datetime and extract date, hour, and minute for 15-minute granularity
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['date'] = df['time'].dt.date
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            df['quarter_hour'] = df['hour'] + (df['minute'] / 60.0)  # Convert to decimal hours for 15-minute intervals
        else:
            logger.error("time column not found in consumption data")
            return pd.DataFrame()

        # Convert start_date and end_date to date objects if they're datetime objects
        start_date_obj = start_date.date() if hasattr(start_date, 'date') else start_date
        end_date_obj = end_date.date() if hasattr(end_date, 'date') else end_date

        # Filter by plant short name (which matches plant_id) and date
        filtered_df = df[
            (df['Plant Short Name'] == plant_id) &
            (df['date'] >= start_date_obj) &
            (df['date'] <= end_date_obj)
        ]

        # If no data found with Plant Short Name, try with Plant Long Name as fallback
        if filtered_df.empty:
            logger.warning(f"No data found for plant_id {plant_id}, trying with Plant Long Name {plant_name}")
            filtered_df = df[
                (df['Plant Long Name'] == plant_name) &
                (df['date'] >= start_date_obj) &
                (df['date'] <= end_date_obj)
            ]

        # Group by plant_long_name, date, and quarter_hour for 15-minute granularity
        # This ensures we keep the date information and don't aggregate across multiple dates
        result_df = filtered_df.groupby(['Plant Long Name', 'date', 'quarter_hour'])['Energy_kWh'].sum().reset_index()

        # Rename columns to match what's expected in the visualization function
        result_df = result_df.rename(columns={
            'Plant Long Name': 'plant_long_name',
            'Energy_kWh': 'energy_kwh'
        })

        # Add additional time-related columns for compatibility
        result_df['hour'] = result_df['quarter_hour'].astype(int)
        result_df['minute'] = ((result_df['quarter_hour'] % 1) * 60).astype(int)
        result_df['time_str'] = result_df['hour'].astype(str).str.zfill(2) + ':' + result_df['minute'].astype(str).str.zfill(2)

        # Create datetime column for plotting
        result_df['datetime'] = pd.to_datetime(result_df['date']) + pd.to_timedelta(result_df['hour'], unit='h') + pd.to_timedelta(result_df['minute'], unit='m')

        return result_df
    except FileNotFoundError:
        logger.error(f"Consumption data file not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing consumption data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def get_hourly_block_consumption(plant_name, date):
    """
    Get consumption data for the specified plant and date, grouped into 3-hour blocks.

    Args:
        plant_name (str): Name of the plant
        date (datetime): Date to retrieve data for

    Returns:
        DataFrame: Consumption data grouped by 3-hour blocks
    """
    try:
        # Get hourly consumption data
        df = get_consumption_data_from_csv(plant_name, date)

        if df.empty:
            logger.warning(f"No consumption data found for {plant_name} on {date}")
            return pd.DataFrame()

        # Convert hour to numeric to ensure proper calculations
        df['hour'] = pd.to_numeric(df['hour'], errors='coerce')

        # Create a new column for the hour block (similar to generation data)
        df['HOUR_BLOCK'] = df['hour'].apply(lambda x: np.floor(x / 3.0) * 3.0)

        # Group by plant_long_name and hour_block
        result_df = df.groupby(['plant_long_name', 'HOUR_BLOCK'])['energy_kwh'].sum().reset_index()

        # Rename the energy column to match the generation data format
        result_df = result_df.rename(columns={'energy_kwh': 'TOTAL_CONSUMPTION'})

        logger.info(f"Retrieved {len(result_df)} rows of hourly block consumption data for {plant_name}")
        return result_df

    except Exception as e:
        logger.error(f"Failed to retrieve hourly block consumption data: {e}")
        return pd.DataFrame()


def get_generation_consumption_comparison(plant_name, date):
    """
    Get both generation and consumption data for comparison with 15-minute granularity

    Args:
        plant_name (str): Name of the plant
        date (datetime): Date to retrieve data for

    Returns:
        tuple: (generation_df, consumption_df) - DataFrames with generation and consumption data
    """
    try:
        # Get 15-minute generation data using the API for better granularity
        generation_df = get_generation_only_data(plant_name, date)

        if generation_df.empty:
            # Only log warning for actual plants, not for "Combined View"
            if plant_name != "Combined View":
                logger.warning(f"No 15-minute generation data found for {plant_name} on {date}")
            return pd.DataFrame(), pd.DataFrame()

        # Add quarter_hour column if not present
        if 'quarter_hour' not in generation_df.columns and 'hour' in generation_df.columns and 'minute' in generation_df.columns:
            generation_df['quarter_hour'] = generation_df['hour'] + (generation_df['minute'] / 60.0)
        elif 'quarter_hour' not in generation_df.columns and 'time' in generation_df.columns:
            # Extract quarter_hour from time column
            generation_df['hour'] = generation_df['time'].dt.hour
            generation_df['minute'] = generation_df['time'].dt.minute
            generation_df['quarter_hour'] = generation_df['hour'] + (generation_df['minute'] / 60.0)

        # Get consumption data for the same date (already returns 15-minute data)
        consumption_df = get_consumption_data_from_csv(
            plant_name,
            date,
            date
        )

        return generation_df, consumption_df
    except Exception as e:
        logger.error(f"Error getting generation and consumption data for comparison: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame(), pd.DataFrame()


def compare_generation_consumption(generation_df, consumption_df):
    """
    Compare generation and consumption data

    Args:
        generation_df (DataFrame): Generation data
        consumption_df (DataFrame): Consumption data

    Returns:
        DataFrame: Combined data for comparison
    """
    # Prepare generation data - handle different column naming conventions including 15-minute data
    if not generation_df.empty:
        # Check which column names are present and standardize
        if 'generation_kwh' in generation_df.columns and 'quarter_hour' in generation_df.columns:
            # 15-minute data with quarter_hour column
            gen_data = generation_df[['quarter_hour', 'generation_kwh']].copy()
            gen_data = gen_data.rename(columns={'quarter_hour': 'time_interval'})
        elif 'generation_kwh' in generation_df.columns and 'hour' in generation_df.columns:
            # Hourly data
            gen_data = generation_df[['hour', 'generation_kwh']].copy()
            gen_data = gen_data.rename(columns={'hour': 'time_interval'})
        elif 'HOUR_NO' in generation_df.columns and 'PLANT_GENERATION' in generation_df.columns:
            gen_data = generation_df[['HOUR_NO', 'PLANT_GENERATION']].copy()
            gen_data = gen_data.rename(columns={'HOUR_NO': 'time_interval', 'PLANT_GENERATION': 'generation_kwh'})
        else:
            # Try to identify columns by position if they exist
            columns = generation_df.columns.tolist()
            if len(columns) >= 2:
                gen_data = generation_df.iloc[:, [1, 2]].copy()  # Assuming time is column 1, generation is column 2
                gen_data.columns = ['time_interval', 'generation_kwh']
            else:
                # Create empty dataframe with required columns
                gen_data = pd.DataFrame(columns=['time_interval', 'generation_kwh'])
    else:
        gen_data = pd.DataFrame(columns=['time_interval', 'generation_kwh'])

    # Prepare consumption data - handle both 15-minute and hourly data
    if not consumption_df.empty:
        if 'quarter_hour' in consumption_df.columns and 'energy_kwh' in consumption_df.columns:
            # 15-minute consumption data
            cons_data = consumption_df[['quarter_hour', 'energy_kwh']].copy()
            cons_data = cons_data.rename(columns={'quarter_hour': 'time_interval'})
        elif 'hour' in consumption_df.columns and 'energy_kwh' in consumption_df.columns:
            # Hourly consumption data
            cons_data = consumption_df[['hour', 'energy_kwh']].copy()
            cons_data = cons_data.rename(columns={'hour': 'time_interval'})
        else:
            # Try to identify columns by position
            columns = consumption_df.columns.tolist()
            if len(columns) >= 2:
                cons_data = consumption_df.iloc[:, [1, 2]].copy()
                cons_data.columns = ['time_interval', 'energy_kwh']
            else:
                cons_data = pd.DataFrame(columns=['time_interval', 'energy_kwh'])
    else:
        cons_data = pd.DataFrame(columns=['time_interval', 'energy_kwh'])

    # Ensure time_interval is numeric and same data type for proper merging
    gen_data['time_interval'] = pd.to_numeric(gen_data['time_interval'], errors='coerce').astype(float)
    cons_data['time_interval'] = pd.to_numeric(cons_data['time_interval'], errors='coerce').astype(float)

    # Round to avoid floating point precision issues
    gen_data['time_interval'] = gen_data['time_interval'].round(2)
    cons_data['time_interval'] = cons_data['time_interval'].round(2)

    # Merge the dataframes on time_interval
    merged_df = pd.merge(gen_data, cons_data, on='time_interval', how='outer')

    # Fill NaN values with 0 and ensure proper data types
    merged_df = merged_df.fillna(0)
    # Ensure we maintain proper data types after filling NaN values
    merged_df = merged_df.infer_objects(copy=False)

    # Sort by time_interval (supports both 15-minute and hourly data)
    merged_df = merged_df.sort_values('time_interval')

    # Add backward compatibility: create 'hour' column for visualization code
    merged_df['hour'] = merged_df['time_interval'].astype(int)

    return merged_df





def get_daily_consumption_data(plant_name, start_date, end_date):
    """
    Get hourly consumption data for the specified plant and date range
    (Despite the function name, this now returns hourly data for better granularity)

    Args:
        plant_name (str): Name of the plant
        start_date (datetime): Start date to retrieve data for
        end_date (datetime): End date to retrieve data for

    Returns:
        DataFrame: Consumption data with hourly granularity
    """
    # Path to the CSV file
    csv_path = "Data/csv/Consumption data Cloud nine - processed_data.csv"

    # Get plant_id from plant_name using client.json
    plant_id = get_plant_id(plant_name)
    logger.info(f"Loading hourly consumption data from: {csv_path} for plant_id: {plant_id}, date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Convert time column to datetime and extract date, hour, and minute for 15-minute granularity
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['date'] = df['time'].dt.date
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            df['quarter_hour'] = df['hour'] + (df['minute'] / 60.0)  # Convert to decimal hours for 15-minute intervals
            # Keep the full datetime for plotting
            df['datetime'] = df['time']
        else:
            logger.error("time column not found in consumption data")
            return pd.DataFrame()

        # Convert start_date and end_date to date objects if they're datetime objects
        start_date_obj = start_date.date() if hasattr(start_date, 'date') else start_date
        end_date_obj = end_date.date() if hasattr(end_date, 'date') else end_date

        # Filter by plant short name (which matches plant_id) and date range
        filtered_df = df[
            (df['Plant Short Name'] == plant_id) &
            (df['date'] >= start_date_obj) &
            (df['date'] <= end_date_obj)
        ]

        # If no data found with Plant Short Name, try with Plant Long Name as fallback
        if filtered_df.empty:
            logger.warning(f"No data found for plant_id {plant_id}, trying with Plant Long Name {plant_name}")
            filtered_df = df[
                (df['Plant Long Name'] == plant_name) &
                (df['date'] >= start_date_obj) &
                (df['date'] <= end_date_obj)
            ]

        # Keep 15-minute granularity - group by plant_long_name, date, quarter_hour, and datetime
        result_df = filtered_df.groupby(['Plant Long Name', 'date', 'quarter_hour', 'datetime'])['Energy_kWh'].sum().reset_index()

        # Rename columns to match what's expected in the visualization function
        result_df = result_df.rename(columns={
            'Plant Long Name': 'plant_long_name',
            'Energy_kWh': 'consumption_kwh'
        })

        # Convert date back to datetime for consistency
        result_df['date'] = pd.to_datetime(result_df['date'])

        # Format quarter_hour as time string (e.g., 9.25 becomes "09:15")
        result_df['hour'] = result_df['quarter_hour'].astype(int)
        result_df['minute'] = ((result_df['quarter_hour'] % 1) * 60).astype(int)
        result_df['time_str'] = result_df['hour'].astype(str).str.zfill(2) + ':' + result_df['minute'].astype(str).str.zfill(2)

        return result_df
    except FileNotFoundError:
        logger.error(f"Consumption data file not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing daily consumption data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def get_daily_generation_consumption_comparison(selected_plant, start_date, end_date):
    """
    Get daily generation and consumption data for comparison

    Args:
        selected_plant (str): Name of the plant
        start_date (datetime): Start date to retrieve data for
        end_date (datetime): End date to retrieve data for

    Returns:
        DataFrame: Combined DataFrame with daily generation and consumption data
    """
    try:
        # Get generation data using the smart wrapper with caching (15-minute granularity - industry standard)
        generation_df = get_generation_data_smart_wrapper(selected_plant, start_date, end_date, granularity="15m")

        if generation_df.empty:
            logger.warning(f"No generation data found for {selected_plant} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Rename columns to match expected format
        generation_df = generation_df.rename(columns={
            'DATE': 'date',
            'TOTAL_GENERATION': 'generation_kwh'
        })

        # Standardize column names
        generation_df = standardize_dataframe_columns(generation_df)

    except Exception as e:
        logger.error(f"Error getting daily generation data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

    # Get daily consumption data
    try:
        # Get consumption data from CSV
        consumption_df = get_daily_consumption_data(selected_plant, start_date, end_date)

        if consumption_df.empty:
            logger.warning(f"No consumption data found for {selected_plant} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Standardize column names
        consumption_df = standardize_dataframe_columns(consumption_df)

    except Exception as e:
        logger.error(f"Error getting daily consumption data: {e}")
        return pd.DataFrame()

    # Merge the two dataframes on date
    try:
        # Convert date columns to same format if needed
        generation_df['date'] = pd.to_datetime(generation_df['date'])
        consumption_df['date'] = pd.to_datetime(consumption_df['date'])

        # Check if consumption_df has sub-daily data (has 'hour' or 'quarter_hour' column)
        if 'hour' in consumption_df.columns or 'quarter_hour' in consumption_df.columns:
            # Group by date to get daily totals - use a safer approach
            try:
                # First create a copy to avoid modifying the original
                agg_df = consumption_df.copy()
                # Then group by date and sum consumption values
                agg_df = agg_df.groupby('date', as_index=False)['consumption_kwh'].sum()
                # Replace the original dataframe with the aggregated one
                consumption_df = agg_df
            except Exception as agg_error:
                logger.error(f"Error during aggregation: {agg_error}")
                # If aggregation fails, create a simple dataframe with the required columns
                consumption_df = pd.DataFrame({
                    'date': pd.to_datetime(consumption_df['date'].unique()),
                    'consumption_kwh': [consumption_df['consumption_kwh'].sum()]
                })

        # Use outer join to include all dates from both datasets
        # This ensures we don't lose dates where consumption might be zero
        result_df = pd.merge(
            generation_df,
            consumption_df,
            on='date',
            how='outer',
            validate='one_to_one'  # Ensure we have one-to-one mapping
        )

        # Fill NaN values with 0 for missing data
        result_df['generation_kwh'] = result_df['generation_kwh'].fillna(0)
        result_df['consumption_kwh'] = result_df['consumption_kwh'].fillna(0)

        # Log information about zero consumption days for debugging
        zero_consumption_days = result_df[result_df['consumption_kwh'] == 0]
        if not zero_consumption_days.empty:
            logger.info(f"Found {len(zero_consumption_days)} days with zero consumption for {selected_plant}")
            logger.info(f"Zero consumption dates: {zero_consumption_days['date'].tolist()}")

        # Only exclude rows where BOTH generation and consumption are zero
        # This preserves days with zero consumption but non-zero generation
        valid_rows = (result_df['generation_kwh'] > 0) | (result_df['consumption_kwh'] > 0)
        if valid_rows.any():
            result_df = result_df[valid_rows].copy()
            logger.info(f"Kept {len(result_df)} days with valid generation or consumption data")
        else:
            logger.warning(f"No valid data found for {selected_plant} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Calculate surplus metrics
        result_df['surplus_generation'] = result_df.apply(
            lambda row: max(0, row['generation_kwh'] - row['consumption_kwh']), axis=1
        )
        result_df['surplus_demand'] = result_df.apply(
            lambda row: max(0, row['consumption_kwh'] - row['generation_kwh']), axis=1
        )

        return result_df
    except Exception as e:
        logger.error(f"Error merging generation and consumption data: {str(e)}")
        # Print more detailed information about the dataframes
        logger.error(f"Generation DataFrame shape: {generation_df.shape}, columns: {generation_df.columns.tolist()}")
        logger.error(f"Consumption DataFrame shape: {consumption_df.shape}, columns: {consumption_df.columns.tolist()}")

        # Try to identify the issue
        if 'date' in generation_df.columns and 'date' in consumption_df.columns:
            # Check data types
            logger.error(f"Generation date type: {generation_df['date'].dtype}")
            logger.error(f"Consumption date type: {consumption_df['date'].dtype}")

            # Check for sample values
            if not generation_df.empty:
                logger.error(f"Generation date sample: {generation_df['date'].iloc[0]}")
            if not consumption_df.empty:
                logger.error(f"Consumption date sample: {consumption_df['date'].iloc[0]}")

        return pd.DataFrame()




#######################################################################
# ToD (Time of Day) Section Functions
#######################################################################

def get_generation_only_data(plant_name, start_date, end_date=None):
    """
    Get generation data only for a specific plant and date range.

    Args:
        plant_name (str): Name of the plant (display name from client.json)
        start_date (datetime): Start date to retrieve data for
        end_date (datetime, optional): End date. If None, only start_date is used.

    Returns:
        DataFrame: Generation data with 'time' as first column and generation values
    """
    try:
        # Special case for "Combined View" - handle separately
        if plant_name == "Combined View":
            logger.info(f"Combined View is not a real plant, handling separately")
            return pd.DataFrame()

        # Get plant_id from client.json using the display name
        plant_id = get_plant_id(plant_name)
        if not plant_id:
            logger.error(f"Could not find plant_id for plant name: {plant_name}")
            return pd.DataFrame()

        # Determine if this is a solar or wind plant
        is_solar = is_solar_plant(plant_name)

        # Format dates for API call
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d') if end_date else start_str

        # Determine if single day or date range
        is_single_day = end_date is None or start_date == end_date

        # Check if integration is initialized
        if integration is None:
            logger.error(f"API integration not initialized. Cannot fetch data for plant {plant_name}")
            return pd.DataFrame()

        df = None
        if is_single_day:
            # For single day, use 15m granularity (15-minute data)
            if is_solar:
                logger.info(f"Fetching 15-minute solar generation data for {plant_name}")
                df = integration.fetchDataV2(
                    plant_id,        # pName - Use plant_id for API calls
                    "Plant",         # catList
                    ["Daily Energy"], # paramList
                    None,            # deviceList
                    start_str,       # sDate
                    start_str,       # eDate
                    granularity="15m",
                    condition={"Daily Energy": "last"}
                )
            else:
                logger.info(f"Fetching 15-minute wind generation data for {plant_name}")
                df = integration.fetchDataV2(
                    plant_id,        # pName - Use plant_id for API calls
                    "Turbine",       # catList
                    ["WTUR.Generation today"],  # paramList
                    None,            # deviceList
                    start_str,       # Start Date
                    start_str,       # End Date
                    granularity="15m",
                    condition={"Generation today": "last"}
                )
        else:
            # For date ranges, use 15m granularity for detailed data
            if is_solar:
                logger.info(f"Fetching 15-minute solar generation data for {plant_name}")
                df = integration.fetchDataV2(
                    plant_id,        # pName - Use plant_id for API calls
                    "Plant",         # catList
                    ["Daily Energy"], # paramList
                    None,            # deviceList
                    start_str,       # sDate
                    end_str,         # eDate
                    granularity="15m",
                    condition={"Daily Energy": "last"}
                )
            else:
                logger.info(f"Fetching 15-minute wind generation data for {plant_name}")
                df = integration.fetchDataV2(
                    plant_id,        # pName - Use plant_id for API calls
                    "Turbine",       # catList
                    ["WTUR.Generation today"],  # paramList
                    None,            # deviceList
                    start_str,       # Start Date
                    end_str,         # End Date
                    granularity="15m",
                    condition={"Generation today": "last"}
                )

        # Log the shape of the returned dataframe
        if df is not None and not isinstance(df, str):
            logger.info(f"API returned generation dataframe with shape: {df.shape} and columns: {df.columns.tolist()}")
            if not df.empty:
                logger.info(f"First few rows of generation data: {df.head(3).to_dict()}")
        else:
            logger.warning(f"API returned non-dataframe result for generation data: {type(df)}")

        # Check if API returned None or empty DataFrame
        if df is None or df.empty:
            logger.warning(f"No generation data returned from API for {plant_name}")
            return pd.DataFrame()

        # Process the dataframe to match expected format
        if not df.empty:
            try:
                # Ensure 'time' is the first column
                if 'time' not in df.columns:
                    logger.warning(f"API response doesn't contain 'time' column: {df.columns}")
                    # Create a time column if it doesn't exist
                    if is_single_day:
                        df['time'] = pd.date_range(start=start_date, periods=24, freq='H')
                    else:
                        df['time'] = pd.date_range(start=start_date, end=end_date, freq='D')

                # Convert time to datetime
                df['time'] = pd.to_datetime(df['time'])

                if is_single_day:
                    # For single day, extract hour and minute for 15-minute data
                    df['hour'] = df['time'].dt.hour
                    df['minute'] = df['time'].dt.minute
                    df['quarter_hour'] = df['hour'] + (df['minute'] / 60.0)  # Convert to decimal hours

                    # Process generation data based on plant type - use the same logic as get_hourly_generation_data
                    if len(df.columns) > 1:
                        # Create a new dataframe with 'time' as first column
                        result_df = pd.DataFrame()
                        result_df['time'] = df['time']
                        result_df['hour'] = df['hour']

                        # For wind plants, sum all columns from index 1 onwards
                        # For solar plants, just take the second column
                        if is_solar:
                            # Get the name of the second column (generation data)
                            gen_col_name = df.columns[1]
                            cumulative_values = pd.to_numeric(df[gen_col_name], errors='coerce').fillna(0)

                            # Check if this is cumulative daily energy data
                            if 'Daily Energy' in gen_col_name:
                                logger.info(f"Converting cumulative daily energy to 15-minute incremental values")
                                # Convert cumulative to incremental (difference between consecutive values)
                                incremental_values = cumulative_values.diff().fillna(0)
                                # Set first value to the cumulative value (start of day)
                                incremental_values.iloc[0] = cumulative_values.iloc[0]
                                # Ensure no negative values (can happen with data issues)
                                incremental_values = incremental_values.clip(lower=0)
                                result_df['generation_kwh'] = incremental_values
                                logger.info(f"Converted cumulative to incremental: Total = {incremental_values.sum():.2f} kWh")
                            else:
                                # Use values as-is if not cumulative
                                result_df['generation_kwh'] = cumulative_values
                        else:
                            # Sum all columns from index 1 onwards (all turbine columns)
                            result_df['generation_kwh'] = df.iloc[:, 1:].sum(axis=1)

                        logger.info(f"Successfully processed single day generation data for {plant_name}: {len(result_df)} records")
                        return result_df
                    else:
                        logger.warning(f"API returned dataframe with unexpected columns: {df.columns}")
                        return pd.DataFrame()

                else:
                    # For date ranges, process daily data
                    if len(df.columns) > 1:
                        # Create a new dataframe with 'time' as first column
                        result_df = pd.DataFrame()
                        result_df['time'] = df['time']

                        # For wind plants, sum all columns from index 1 onwards
                        # For solar plants, just take the second column
                        if is_solar:
                            # Get the name of the second column (generation data)
                            gen_col_name = df.columns[1]
                            result_df['generation_kwh'] = pd.to_numeric(df[gen_col_name], errors='coerce').fillna(0)
                        else:
                            # Sum all columns from index 1 onwards (all turbine columns)
                            result_df['generation_kwh'] = df.iloc[:, 1:].sum(axis=1)

                        # For date ranges with 60m granularity, we need to aggregate to daily
                        if len(result_df) > (end_date - start_date).days + 1:
                            # We have hourly data, aggregate to daily
                            result_df['date'] = result_df['time'].dt.date
                            daily_df = result_df.groupby('date')['generation_kwh'].sum().reset_index()
                            daily_df['time'] = pd.to_datetime(daily_df['date'])
                            result_df = daily_df[['time', 'generation_kwh']].copy()

                        logger.info(f"Successfully processed date range generation data for {plant_name}: {len(result_df)} records")
                        return result_df
                    else:
                        logger.warning(f"API returned dataframe with unexpected columns: {df.columns}")
                        return pd.DataFrame()

            except Exception as process_error:
                logger.error(f"Error processing API response for {plant_name}: {process_error}")
                logger.error(traceback.format_exc())
                return pd.DataFrame()

        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error fetching generation data for {plant_name}: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def get_tod_binned_data(plant_name, start_date, end_date=None):
    """
    Get generation and consumption data binned into custom Time-of-Day (ToD) intervals
    based on the configuration settings.

    Args:
        plant_name (str): Name of the plant
        start_date (datetime): Start date
        end_date (datetime, optional): End date. If None, only start_date is used.

    Returns:
        DataFrame: Data binned by custom ToD intervals
    """
    try:
        # Determine if we're looking at a single day or multiple days
        single_day = end_date is None or start_date == end_date

        if single_day:
            # Get hourly generation data for the single day using the API
            logger.info(f"Getting ToD binned data for {plant_name} on {start_date}")

            try:
                # Get hourly generation data using smart caching
                hourly_df = get_hourly_generation_data_smart_wrapper(plant_name, start_date)

                if hourly_df.empty:
                    logger.warning(f"No hourly generation data found for {plant_name} on {start_date}")
                    return pd.DataFrame()

                # Rename columns to match expected format for ToD binning
                generation_df = hourly_df.rename(columns={
                    'HOUR_NO': 'hour',
                    'PLANT_GENERATION': 'generation_kwh'
                })

                # Ensure we have an 'hour' column for merging
                if 'hour' not in generation_df.columns:
                    if 'time' in generation_df.columns:
                        # Extract hour from time column
                        generation_df['hour'] = pd.to_datetime(generation_df['time']).dt.hour
                        logger.info("Created 'hour' column from 'time' column for ToD binning")
                    else:
                        logger.error("Cannot create 'hour' column for ToD binning - no time information available")
                        return pd.DataFrame()

                # Check for generation column and standardize name
                gen_col = None
                for col in generation_df.columns:
                    if 'generation' in col.lower() or 'energy' in col.lower():
                        gen_col = col
                        break

                if gen_col and gen_col != 'generation_kwh':
                    generation_df = generation_df.rename(columns={gen_col: 'generation_kwh'})
                    logger.info(f"Renamed generation column from '{gen_col}' to 'generation_kwh'")

                logger.info(f"Retrieved {len(generation_df)} rows of generation data for ToD binning")
                logger.info(f"Generation data columns: {generation_df.columns.tolist()}")
                logger.info(f"Generation data sample: {generation_df.head(2).to_dict()}")
            except Exception as e:
                logger.error(f"Error getting generation data for ToD binning: {e}")
                logger.error(traceback.format_exc())
                return pd.DataFrame()

            # Get hourly consumption data for the single day
            logger.info(f"Getting consumption data for ToD binning for {plant_name} on {start_date}")
            consumption_df = get_consumption_data_from_csv(plant_name, start_date)

            if not consumption_df.empty:
                logger.info(f"Retrieved {len(consumption_df)} rows of consumption data for ToD binning")
                logger.info(f"Consumption data columns: {consumption_df.columns.tolist()}")
                logger.info(f"Consumption data sample: {consumption_df.head(2).to_dict()}")
            else:
                logger.warning(f"No consumption data found for {plant_name} on {start_date}")
                # Create an empty consumption DataFrame with the same structure
                consumption_df = pd.DataFrame({'hour': range(24), 'energy_kwh': [0] * 24})
                logger.info("Created empty consumption DataFrame with zeros")

            if generation_df.empty:
                logger.warning(f"No generation data found for {plant_name} on {start_date}")
                return pd.DataFrame()

            # Merge generation and consumption data
            logger.info("Merging generation and consumption data for ToD binning")
            try:
                # Standardize column names before merging
                logger.info(f"Generation DataFrame columns before standardization: {generation_df.columns.tolist()}")
                logger.info(f"Consumption DataFrame columns before standardization: {consumption_df.columns.tolist()}")

                # Convert column names to lowercase
                generation_df.columns = [col.lower() for col in generation_df.columns]

                # Ensure we have the 'hour' column in both dataframes
                if 'hour' not in generation_df.columns and 'HOUR' in generation_df.columns:
                    generation_df = generation_df.rename(columns={'HOUR': 'hour'})
                elif 'hour' not in generation_df.columns and 'hour_no' in generation_df.columns:
                    generation_df = generation_df.rename(columns={'hour_no': 'hour'})

                # Rename generation column if needed
                if 'generation_kwh' not in generation_df.columns and 'GENERATION_KWH' in generation_df.columns:
                    generation_df = generation_df.rename(columns={'GENERATION_KWH': 'generation_kwh'})

                logger.info(f"Generation DataFrame columns after standardization: {generation_df.columns.tolist()}")
                logger.info(f"Consumption DataFrame columns after standardization: {consumption_df.columns.tolist()}")

                # Log the data before merging
                logger.info(f"Generation data shape before merge: {generation_df.shape}")
                logger.info(f"Consumption data shape before merge: {consumption_df.shape}")

                # Handle different time column structures for merging
                # Generation data has 'hour' column (24 hourly records)
                # Consumption data has 'quarter_hour' column (96 15-minute records)

                # For ToD binning, we need to aggregate consumption data to hourly first
                if 'quarter_hour' in consumption_df.columns:
                    logger.info("Aggregating 15-minute consumption data to hourly for ToD binning")
                    # Create hour column from quarter_hour
                    consumption_df['hour'] = consumption_df['quarter_hour'].astype(int)
                    # Aggregate to hourly
                    consumption_df = consumption_df.groupby(['plant_long_name', 'hour'], as_index=False)['energy_kwh'].sum()
                    logger.info(f"Aggregated consumption data shape: {consumption_df.shape}")

                # Now merge the dataframes on hour
                merged_df = pd.merge(
                    generation_df,
                    consumption_df,
                    on='hour',
                    how='outer'
                )

                logger.info(f"Merged data shape: {merged_df.shape}")
                logger.info(f"Merged data columns: {merged_df.columns.tolist()}")

                # Fill NaN values with 0
                merged_df = merged_df.fillna(0)

                # Apply the same filtering logic as in compare_generation_consumption
                # to ensure consistency between regular and ToD plots
                if 'generation_kwh' in merged_df.columns and ('energy_kwh' in merged_df.columns or 'consumption_kwh' in merged_df.columns):
                    consumption_col = 'consumption_kwh' if 'consumption_kwh' in merged_df.columns else 'energy_kwh'

                    # Log the data before filtering
                    total_gen_before = merged_df['generation_kwh'].sum()
                    total_cons_before = merged_df[consumption_col].sum()
                    logger.info(f"ToD - Before filtering - Total generation: {total_gen_before:.2f} kWh, Total consumption: {total_cons_before:.2f} kWh")

                    # Only keep rows where both generation and consumption are > 0
                    # This ensures we're comparing the same hours as in the regular plot
                    valid_rows = (merged_df['generation_kwh'] > 0) & (merged_df[consumption_col] > 0)
                    if valid_rows.any():
                        filtered_df = merged_df[valid_rows].copy()

                        # Log the data after filtering
                        total_gen_after = filtered_df['generation_kwh'].sum()
                        total_cons_after = filtered_df[consumption_col].sum()
                        logger.info(f"ToD - After filtering - Total generation: {total_gen_after:.2f} kWh, Total consumption: {total_cons_after:.2f} kWh")
                        logger.info(f"ToD - Filtered out {len(merged_df) - len(filtered_df)} rows with zero generation or consumption")

                        # Use the filtered data if we have enough valid hours
                        if len(filtered_df) >= 3:  # Require at least 3 valid hours
                            merged_df = filtered_df
                            logger.info(f"ToD - Using filtered data with {len(filtered_df)} valid hours")

                # Import ToD configuration
                from backend.config.tod_config import get_tod_slot, get_tod_bin_labels

                def assign_tod_bin(hour_str):
                    # Convert hour string to integer and get the appropriate ToD bin
                    try:
                        # Get the ToD slot for this hour
                        slot = get_tod_slot(hour_str)
                        start_hour = slot["start_hour"]
                        end_hour = slot["end_hour"]
                        name = slot["name"]

                        # Format hours in 12-hour format with AM/PM
                        start_str = f"{start_hour if start_hour <= 12 else start_hour - 12} {'AM' if start_hour < 12 else 'PM'}"
                        end_str = f"{end_hour if end_hour <= 12 else end_hour - 12} {'AM' if end_hour < 12 else 'PM'}"

                        return f"{start_str} - {end_str} ({name})"
                    except Exception as e:
                        logger.error(f"Error in assign_tod_bin for hour {hour_str}: {e}")
                        # Default to first slot if there's an error
                        return "6 AM - 10 AM (Peak)"

                logger.info("Assigning ToD bins to data")
                # Add ToD bin column
                merged_df['tod_bin'] = merged_df['hour'].apply(assign_tod_bin)

                # Add peak/off-peak indicator
                merged_df['is_peak'] = merged_df['tod_bin'].apply(lambda x: "Peak" in x)

                logger.info("Grouping data by ToD bin")
                # Check if we have the expected columns
                logger.info(f"Columns before grouping: {merged_df.columns.tolist()}")

                # Make sure we have the right column names for aggregation
                agg_columns = {}

                # Check for generation column
                if 'generation_kwh' in merged_df.columns:
                    agg_columns['generation_kwh'] = 'sum'
                elif 'plant_generation' in merged_df.columns:
                    merged_df = merged_df.rename(columns={'plant_generation': 'generation_kwh'})
                    agg_columns['generation_kwh'] = 'sum'

                # Check for consumption/energy column
                if 'energy_kwh' in merged_df.columns:
                    agg_columns['energy_kwh'] = 'sum'
                elif 'consumption_kwh' in merged_df.columns:
                    merged_df = merged_df.rename(columns={'consumption_kwh': 'energy_kwh'})
                    agg_columns['energy_kwh'] = 'sum'

                # Add is_peak to aggregation
                agg_columns['is_peak'] = 'first'

                logger.info(f"Aggregation columns: {agg_columns}")

                # Group by ToD bin and aggregate
                result_df = merged_df.groupby('tod_bin').agg(agg_columns).reset_index()

                logger.info(f"Grouped data shape: {result_df.shape}")
                logger.info(f"Grouped data columns: {result_df.columns.tolist()}")
                logger.info(f"Grouped data: {result_df.to_dict()}")
            except Exception as e:
                logger.error(f"Error during data merging and binning: {e}")
                logger.error(traceback.format_exc())
                return pd.DataFrame()

            # Define the correct order of bins for display using hardcoded values for reliability
            # Define ToD slots directly
            TOD_SLOTS_LOCAL = [
                {
                    "start_hour": 6,
                    "end_hour": 10,
                    "name": "Peak",
                    "description": "Morning peak demand period"
                },
                {
                    "start_hour": 10,
                    "end_hour": 18,
                    "name": "Off-Peak",
                    "description": "Daytime off-peak period"
                },
                {
                    "start_hour": 18,
                    "end_hour": 22,
                    "name": "Peak",
                    "description": "Evening peak demand period"
                },
                {
                    "start_hour": 22,
                    "end_hour": 6,
                    "name": "Off-Peak",
                    "description": "Nighttime off-peak period"
                }
            ]

            # Generate bin labels in the correct order
            bin_order = []
            for slot in TOD_SLOTS_LOCAL:
                start_hour = slot["start_hour"]
                end_hour = slot["end_hour"]
                name = slot["name"]

                # Format hours in 12-hour format with AM/PM
                start_str = f"{start_hour if start_hour <= 12 else start_hour - 12} {'AM' if start_hour < 12 else 'PM'}"
                end_str = f"{end_hour if end_hour <= 12 else end_hour - 12} {'AM' if end_hour < 12 else 'PM'}"

                bin_order.append(f"{start_str} - {end_str} ({name})")

            # Ensure all bins are present (even if no data)
            for bin_name in bin_order:
                if bin_name not in result_df['tod_bin'].values:
                    result_df = pd.concat([
                        result_df,
                        pd.DataFrame({
                            'tod_bin': [bin_name],
                            'generation_kwh': [0],
                            'energy_kwh': [0],
                            'is_peak': ["Peak" in bin_name]
                        })
                    ], ignore_index=True)

            # Sort by the defined bin order
            result_df['bin_order'] = result_df['tod_bin'].apply(lambda x: bin_order.index(x))
            result_df = result_df.sort_values('bin_order').drop('bin_order', axis=1)

            # Rename columns for clarity - use assignment instead of inplace
            if 'energy_kwh' in result_df.columns:
                result_df = result_df.rename(columns={'energy_kwh': 'consumption_kwh'})
                logger.info(f"Renamed energy_kwh to consumption_kwh for ToD binned data")

            return result_df

        else:
            # For multiple days, we need to aggregate data across days
            # Initialize an empty DataFrame to store results
            all_days_df = pd.DataFrame()

            # Process each day in the date range
            current_date = start_date
            while current_date <= end_date:
                # Get data for the current day
                day_df = get_tod_binned_data(plant_name, current_date)

                if not day_df.empty:
                    # Add date column
                    day_df['date'] = current_date

                    # Append to the all days DataFrame
                    all_days_df = pd.concat([all_days_df, day_df], ignore_index=True)

                # Move to the next day
                current_date += pd.Timedelta(days=1)

            if all_days_df.empty:
                logger.warning(f"No data found for {plant_name} between {start_date} and {end_date}")
                return pd.DataFrame()

            # Check if we have any generation data
            if 'generation_kwh' not in all_days_df.columns or all_days_df['generation_kwh'].sum() == 0:
                logger.warning(f"No generation data found for {plant_name} between {start_date} and {end_date}")
                return pd.DataFrame()

            # Check if we have energy_kwh instead of consumption_kwh
            if 'energy_kwh' in all_days_df.columns and 'consumption_kwh' not in all_days_df.columns:
                all_days_df = all_days_df.rename(columns={'energy_kwh': 'consumption_kwh'})
                logger.info(f"Renamed energy_kwh to consumption_kwh for multi-day ToD binned data")

            # Log the data before aggregation
            total_gen_before = all_days_df['generation_kwh'].sum()
            total_cons_before = all_days_df['consumption_kwh'].sum()
            logger.info(f"ToD Multi-day - Before aggregation - Total generation: {total_gen_before:.2f} kWh, Total consumption: {total_cons_before:.2f} kWh")

            # Apply filtering to ensure consistency with regular Generation vs Consumption plot
            # Only keep rows where both generation and consumption are > 0
            valid_rows = (all_days_df['generation_kwh'] > 0) & (all_days_df['consumption_kwh'] > 0)
            if valid_rows.any():
                filtered_df = all_days_df[valid_rows].copy()

                # Log the data after filtering
                total_gen_after = filtered_df['generation_kwh'].sum()
                total_cons_after = filtered_df['consumption_kwh'].sum()
                logger.info(f"ToD Multi-day - After filtering - Total generation: {total_gen_after:.2f} kWh, Total consumption: {total_cons_after:.2f} kWh")
                logger.info(f"ToD Multi-day - Filtered out {len(all_days_df) - len(filtered_df)} rows with zero generation or consumption")

                # Use the filtered data if we have enough valid rows
                if len(filtered_df) >= len(all_days_df) * 0.5:  # Require at least 50% of the original data
                    all_days_df = filtered_df
                    logger.info(f"ToD Multi-day - Using filtered data with {len(filtered_df)} valid rows")

            # Group by ToD bin and aggregate across all days
            result_df = all_days_df.groupby('tod_bin').agg({
                'generation_kwh': 'sum',
                'consumption_kwh': 'sum',
                'is_peak': 'first'
            }).reset_index()

            # Import ToD configuration to get bin labels
            from backend.config.tod_config import get_tod_bin_labels

            # Get bin labels in the correct order
            bin_order = get_tod_bin_labels("full")

            # Sort by the defined bin order
            result_df['bin_order'] = result_df['tod_bin'].apply(lambda x: bin_order.index(x))
            result_df = result_df.sort_values('bin_order').drop('bin_order', axis=1)

            return result_df

    except Exception as e:
        logger.error(f"Error getting ToD binned data: {e}")
        return pd.DataFrame()

#######################################################################
# Summary Section Functions
#######################################################################

def get_generation_data_smart_wrapper(plant_name, start_date, end_date, granularity="15m"):
    """
    Smart wrapper for generation data fetching with caching.

    Args:
        plant_name (str): Name of the plant (display name from client.json)
        start_date (datetime): Start date to retrieve data for
        end_date (datetime): End date to retrieve data for
        granularity (str): Data granularity (15m, 1h, 1d)

    Returns:
        DataFrame: Generation data with smart caching
    """
    if SMART_CACHING_ENABLED:
        try:
            # Import smart data fetcher only when needed to avoid circular imports
            from backend.services.smart_data_fetcher import smart_data_fetcher

            # Get plant_id from plant_name
            plant_id = get_plant_id(plant_name)
            if not plant_id:
                logger.warning(f"Could not find plant_id for {plant_name}")
                return get_generation_data(plant_name, start_date, end_date)

            # Convert dates to strings
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')

            # Use smart data fetcher
            df = smart_data_fetcher.get_generation_data_smart(
                plant_id, start_str, end_str, granularity
            )

            if df is not None and not df.empty:
                logger.info(f"Smart cache returned {len(df)} rows for {plant_name}")
                return df
            else:
                logger.warning(f"Smart cache returned empty data for {plant_name}, falling back to original method")

        except Exception as e:
            logger.error(f"Smart caching failed for {plant_name}: {e}, falling back to original method")

    # Fallback to original method
    return get_generation_data(plant_name, start_date, end_date)


def get_hourly_generation_data_smart_wrapper(plant_name, date):
    """
    Smart wrapper for hourly generation data fetching with caching.

    Args:
        plant_name (str): Name of the plant (display name from client.json)
        date (datetime): Date to retrieve data for

    Returns:
        DataFrame: Hourly generation data with smart caching
    """
    if SMART_CACHING_ENABLED:
        try:
            # Import smart data fetcher only when needed to avoid circular imports
            from backend.services.smart_data_fetcher import smart_data_fetcher

            # Get plant_id from plant_name
            plant_id = get_plant_id(plant_name)
            if not plant_id:
                logger.warning(f"Could not find plant_id for {plant_name}")
                return get_hourly_generation_data(plant_name, date)

            # Convert date to string
            date_str = date.strftime('%Y-%m-%d')

            # Use smart data fetcher with 1h granularity for hourly data
            df = smart_data_fetcher.get_generation_data_smart(
                plant_id, date_str, date_str, granularity="1h"
            )

            if df is not None and not df.empty:
                logger.info(f"Smart cache returned {len(df)} hourly rows for {plant_name}")
                return df
            else:
                logger.warning(f"Smart cache returned empty hourly data for {plant_name}, falling back to original method")

        except Exception as e:
            logger.error(f"Smart caching failed for hourly data {plant_name}: {e}, falling back to original method")

    # Fallback to original method
    return get_hourly_generation_data(plant_name, date)


def get_generation_data(plant_name, start_date, end_date):
    """
    Get generation data for the specified plant and date range

    Args:
        plant_name (str): Name of the plant (display name from client.json)
        start_date (datetime): Start date to retrieve data for
        end_date (datetime): End date to retrieve data for

    Returns:
        DataFrame: Generation data grouped by date with 'time' as first column
    """
    # Special case for "Combined View" - handle separately
    if plant_name == "Combined View":
        logger.info(f"Combined View is not a real plant, handling separately")
        # Implement Combined View logic if needed
        return pd.DataFrame()

    try:
        # Get the plant_id for API calls
        plant_id = get_plant_id(plant_name)
        logger.info(f"Using plant_id {plant_id} for API calls (display name: {plant_name})")

        # Determine if it's a solar or wind plant
        is_solar = is_solar_plant(plant_name)

        # Format dates for API call
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        try:
            # Check if integration is initialized
            if integration is None:
                logger.error(f"API integration not initialized. Cannot fetch data for {plant_name}")
                return pd.DataFrame()

            logger.info(f"Fetching data for plant: {plant_name} (ID: {plant_id}) from {start_str} to {end_str}")

            if is_solar:
                # For solar plants
                logger.info(f"Fetching solar plant data with category: Plant, parameter: Daily Energy")
                df = integration.fetchDataV2(
                    plant_id,        # pName - Use plant_id for API calls
                    "Plant",         # catList
                    ["Daily Energy"], # paramList
                    None,            # deviceList
                    start_str,       # sDate
                    end_str,         # eDate
                    granularity="1d",
                    condition={"Daily Energy": "max"}
                )
                logger.info(f"Solar data fetch completed for {plant_name}")
            else:
                # For wind plants
                logger.info(f"Fetching wind plant data with category: Turbine, parameter: WTUR.Generation today")
                df = integration.fetchDataV2(
                    plant_id,        # pName - Use plant_id for API calls
                    "Turbine",       # catList
                    ["WTUR.Generation today"],  # paramList
                    None,            # deviceList
                    start_str,       # Start Date
                    end_str,         # End Date
                    granularity="1d",
                    condition={"Generation today": "last"}
                )
                logger.info(f"Wind data fetch completed for {plant_name}")

            # Log the shape of the returned dataframe
            if df is not None and not isinstance(df, str):
                logger.info(f"API returned dataframe with shape: {df.shape} and columns: {df.columns.tolist()}")
            else:
                logger.warning(f"API returned non-dataframe result: {type(df)}")

        except Exception as api_error:
            logger.error(f"API call failed for {plant_name}: {api_error}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

        # Check if API returned None
        if df is None:
            logger.warning(f"API returned None for {plant_name}")
            return pd.DataFrame()

        # Process the dataframe to match expected format
        if not df.empty:
            try:
                # Ensure 'time' is the first column
                if 'time' not in df.columns:
                    logger.warning(f"API response doesn't contain 'time' column: {df.columns}")
                    # Create a time column if it doesn't exist
                    df['time'] = pd.date_range(start=start_date, end=end_date, freq='D')

                # Ensure we have generation data in the second column
                if len(df.columns) > 1:
                    # Get the name of the second column (generation data)
                    gen_col_name = df.columns[1]

                    # Create a new dataframe with 'time' as first column and generation data as second
                    result_df = pd.DataFrame()
                    result_df['time'] = pd.to_datetime(df['time'])
                    result_df['generation'] = df[gen_col_name]

                    # Add additional columns for backward compatibility
                    result_df['DATEVALUE'] = result_df['time'].dt.date
                    result_df['DATE'] = result_df['time']
                    result_df['TOTAL_GENERATION'] = result_df['generation']

                    logger.info(f"Retrieved {len(result_df)} rows of generation data for {plant_name}")
                    return result_df
                else:
                    logger.warning(f"API returned dataframe with unexpected columns: {df.columns}")
                    return pd.DataFrame()
            except Exception as process_error:
                logger.error(f"Error processing API response for {plant_name}: {process_error}")
                logger.error(traceback.format_exc())
                return pd.DataFrame()
        else:
            logger.warning(f"API returned empty dataframe for {plant_name}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Failed to retrieve generation data from API: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def get_hourly_generation_data(plant_name, date):
    """
    Get hourly generation data for the specified plant and date.

    Args:
        plant_name (str): Name of the plant (display name from client.json)
        date (datetime): Date to retrieve data for

    Returns:
        DataFrame: Hourly generation data with 'time' as first column
    """
    # Special case for "Combined View" - handle separately
    if plant_name == "Combined View" or (isinstance(plant_name, dict) and plant_name.get('name') == "Combined View"):
        logger.info(f"Combined View is not a real plant, handling separately")
        # Implement Combined View logic if needed
        return pd.DataFrame()

    try:
        # Get the plant_id for API calls
        plant_id = get_plant_id(plant_name)
        logger.info(f"Using plant_id {plant_id} for API calls (display name: {plant_name})")

        # Determine if it's a solar or wind plant
        is_solar = is_solar_plant(plant_name)
        logger.info(f"Plant {plant_name} is_solar: {is_solar}")

        # Format date for API call
        date_str = date.strftime('%Y-%m-%d')

        try:
            # Check if integration is initialized
            if integration is None:
                logger.error(f"API integration not initialized. Cannot fetch hourly data for {plant_name}")
                return pd.DataFrame()

            logger.info(f"Fetching hourly data for plant: {plant_name} (ID: {plant_id}) on {date_str}")

            # Add more detailed logging for debugging
            df = None
            if is_solar:
                # For solar plants
                logger.info(f"Fetching hourly solar plant data with category: Plant, parameter: Daily Energy")
                df = integration.fetchDataV2(
                    plant_id,        # pName - Use plant_id for API calls
                    "Plant",         # catList
                    ["Daily Energy"], # paramList
                    None,            # deviceList
                    date_str,        # sDate
                    date_str,        # eDate
                    granularity="60m",
                    condition={"Daily Energy": "last"}
                )
                logger.info(f"Hourly solar data fetch completed for {plant_name}")
            else:
                # For wind plants
                logger.info(f"Fetching hourly wind plant data")
                df = integration.fetchDataV2(
                    plant_id,        # pName - Use plant_id for API calls
                    "Turbine",       # catList
                    ["WTUR.Generation today"],  # paramList
                    None,            # deviceList
                    date_str,        # Start Date
                    date_str,        # End Date
                    granularity="60m",
                    condition={"Generation today": "last"}
                )
                logger.info(f"Hourly wind data fetch completed for {plant_name}")

            # Log the shape of the returned dataframe
            if df is not None and not isinstance(df, str):
                logger.info(f"API returned hourly dataframe with shape: {df.shape} and columns: {df.columns.tolist()}")
                if not df.empty:
                    logger.info(f"First few rows of data: {df.head(3).to_dict()}")
            else:
                logger.warning(f"API returned non-dataframe result for hourly data: {type(df)}")

        except Exception as api_error:
            logger.error(f"API call failed for hourly data {plant_name}: {api_error}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()

        # Check if API returned None
        if df is None:
            logger.warning(f"API returned None for {plant_name}")
            return pd.DataFrame()

        # Process the dataframe to match expected format
        if not df.empty:
            try:
                # Ensure 'time' is the first column
                if 'time' not in df.columns:
                    logger.warning(f"API response doesn't contain 'time' column: {df.columns}")
                    # Create a time column with hourly intervals for the given date
                    df['time'] = pd.date_range(start=date, periods=24, freq='H')

                # Convert time to datetime and extract hour
                df['time'] = pd.to_datetime(df['time'])
                df['HOUR_NO'] = df['time'].dt.hour

                # Process generation data based on plant type
                if len(df.columns) > 1:
                    # Create a new dataframe with 'time' as first column
                    result_df = pd.DataFrame()
                    result_df['time'] = df['time']

                    # For wind plants, sum all columns from index 1 onwards
                    # For solar plants, just take the second column
                    if is_solar:
                        # Get the name of the second column (generation data)
                        gen_col_name = df.columns[1]
                        result_df['generation'] = df[gen_col_name]
                    else:
                        # Sum all columns from index 1 onwards (all turbine columns)
                        result_df['generation'] = df.iloc[:, 1:].sum(axis=1)

                    # Add additional columns for backward compatibility
                    result_df['HOUR_NO'] = df['HOUR_NO']
                    result_df['PLANT_GENERATION'] = result_df['generation']
                    result_df['PLANT_LONG_NAME'] = plant_name

                    # Reorder columns to match expected format for backward compatibility
                    # But keep 'time' as the first column
                    final_df = pd.DataFrame()
                    final_df['time'] = result_df['time']
                    final_df['PLANT_LONG_NAME'] = result_df['PLANT_LONG_NAME']
                    final_df['HOUR_NO'] = result_df['HOUR_NO']
                    final_df['PLANT_GENERATION'] = result_df['PLANT_GENERATION']

                    logger.info(f"Retrieved {len(final_df)} rows of hourly generation data for {plant_name}")
                    return final_df
                else:
                    logger.warning(f"API returned dataframe with unexpected columns: {df.columns}")
                    return pd.DataFrame()
            except Exception as process_error:
                logger.error(f"Error processing API response for {plant_name}: {process_error}")
                logger.error(traceback.format_exc())
                return pd.DataFrame()
        else:
            logger.warning(f"API returned empty dataframe for {plant_name}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Failed to retrieve hourly generation data from API: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()


# Note: get_generation_consumption_comparison function is defined earlier in the file (around line 407)
# This duplicate function has been removed to avoid conflicts with the 15-minute granularity implementation


# Note: compare_generation_consumption function is defined earlier in the file (around line 452)
# This duplicate function has been removed to avoid conflicts with the 15-minute granularity implementation


def get_daily_generation_consumption_comparison(selected_plant, start_date, end_date):
    """
    Get daily generation and consumption data for comparison

    Args:
        selected_plant (str): Name of the plant
        start_date (datetime): Start date to retrieve data for
        end_date (datetime): End date to retrieve data for

    Returns:
        DataFrame: Combined DataFrame with daily generation and consumption data
    """
    try:
        # Get generation data using the smart wrapper with caching (daily granularity for daily comparison)
        generation_df = get_generation_data_smart_wrapper(selected_plant, start_date, end_date, granularity="1d")

        if generation_df.empty:
            logger.warning(f"No generation data found for {selected_plant} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Rename columns to match expected format
        generation_df = generation_df.rename(columns={
            'DATE': 'date',
            'TOTAL_GENERATION': 'generation_kwh'
        })

        # Standardize column names
        generation_df = standardize_dataframe_columns(generation_df)

        # Ensure we have a date column for merging
        if 'time' in generation_df.columns and 'date' not in generation_df.columns:
            # Convert time to date for daily aggregation
            generation_df['date'] = pd.to_datetime(generation_df['time']).dt.date

            # Find the generation column (should be the second column or any column with generation data)
            gen_col = None
            if 'generation_kwh' in generation_df.columns:
                gen_col = 'generation_kwh'
            else:
                # Find any column that might contain generation data
                for col in generation_df.columns:
                    if col != 'time' and col != 'date' and 'energy' in col.lower():
                        gen_col = col
                        break

                # If still not found, use the second column
                if gen_col is None and len(generation_df.columns) > 1:
                    gen_col = generation_df.columns[1]

            if gen_col:
                # Check if this is already processed incremental data or raw cumulative data
                # If the data comes from get_generation_data_smart_wrapper, it's already processed
                # and contains incremental values that should be summed
                # If it's raw API data, it might be cumulative and need different handling

                # Check if we have incremental data (generation_kwh column exists and values are reasonable)
                if gen_col == 'generation_kwh' and 'time' in generation_df.columns:
                    # This is likely already processed incremental data from smart wrapper
                    # Sum the incremental values to get daily total
                    logger.info(f"Aggregating incremental generation data for daily view")
                    generation_df = generation_df.groupby('date', as_index=False)[gen_col].sum()
                else:
                    # This might be raw cumulative data - take the last value of each day
                    logger.info(f"Processing potentially cumulative generation data for daily view")
                    # For cumulative data, we want the last (maximum) value of each day
                    generation_df = generation_df.groupby('date', as_index=False)[gen_col].last()

                # Rename the generation column to standard name
                generation_df = generation_df.rename(columns={gen_col: 'generation_kwh'})
                # Convert back to datetime for consistency
                generation_df['date'] = pd.to_datetime(generation_df['date'])

                # Log the aggregation result for debugging
                if not generation_df.empty:
                    total_gen = generation_df['generation_kwh'].sum()
                    logger.info(f"Daily aggregation result: {len(generation_df)} days, total generation: {total_gen:.2f} kWh")
            else:
                logger.error("No generation column found in data")
                return pd.DataFrame()
        elif 'date' in generation_df.columns:
            # Ensure date column is datetime
            generation_df['date'] = pd.to_datetime(generation_df['date'])

    except Exception as e:
        logger.error(f"Error getting daily generation data: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

    # Get daily consumption data
    try:
        # Get consumption data from CSV
        consumption_df = get_daily_consumption_data(selected_plant, start_date, end_date)

        if consumption_df.empty:
            logger.warning(f"No consumption data found for {selected_plant} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Standardize column names
        consumption_df = standardize_dataframe_columns(consumption_df)

    except Exception as e:
        logger.error(f"Error getting daily consumption data: {e}")
        return pd.DataFrame()

    # Merge the two dataframes on date
    try:
        # Fix duplicate columns in generation_df
        if 'date' in generation_df.columns:
            # Check for duplicate column names
            duplicate_cols = generation_df.columns.duplicated()
            if any(duplicate_cols):
                logger.warning(f"Found duplicate columns in generation_df: {generation_df.columns[duplicate_cols].tolist()}")
                # Keep only the first occurrence of each column name
                generation_df = generation_df.loc[:, ~generation_df.columns.duplicated()]
                logger.info(f"Fixed generation_df columns: {generation_df.columns.tolist()}")

            # Check for duplicates in date values
            if generation_df['date'].duplicated().any():
                duplicate_count = generation_df['date'].duplicated().sum()
                logger.info(f"Found {duplicate_count} duplicate dates in generation data. Aggregating by date for data consistency.")
                # Aggregate by date to handle duplicates
                generation_df = generation_df.groupby('date', as_index=False)['generation_kwh'].sum()

            # Convert to datetime safely
            try:
                generation_df['date'] = pd.to_datetime(generation_df['date'])
            except ValueError as e:
                logger.error(f"Error converting generation date to datetime: {e}")
                # Try to fix the issue by dropping duplicates
                generation_df = generation_df.drop_duplicates(subset=['date'])
                generation_df['date'] = pd.to_datetime(generation_df['date'])

        # Handle consumption data date conversion and duplicates
        if 'date' in consumption_df.columns:
            # Check for duplicate dates in consumption data
            if consumption_df['date'].duplicated().any():
                duplicate_count = consumption_df['date'].duplicated().sum()
                logger.info(f"Found {duplicate_count} duplicate dates in consumption data. Aggregating by date for data consistency.")
                # Group by date and sum consumption values
                consumption_df = consumption_df.groupby('date', as_index=False)['consumption_kwh'].sum()

            try:
                consumption_df['date'] = pd.to_datetime(consumption_df['date'])
            except ValueError as e:
                logger.error(f"Error converting consumption date to datetime: {e}")
                consumption_df = consumption_df.drop_duplicates(subset=['date'])
                consumption_df['date'] = pd.to_datetime(consumption_df['date'])

        # Check if consumption_df has hourly data (has 'hour' column)
        if 'hour' in consumption_df.columns:
            # Group by date to get daily totals
            try:
                consumption_df = consumption_df.groupby('date')['consumption_kwh'].sum().reset_index()
            except Exception as agg_error:
                logger.error(f"Error aggregating consumption data: {agg_error}")
                # Try an alternative approach if the standard groupby fails
                # First ensure date is datetime
                consumption_df['date'] = pd.to_datetime(consumption_df['date'])
                # Then group by date
                consumption_df = consumption_df.groupby(consumption_df['date'].dt.date)['consumption_kwh'].sum().reset_index()
                consumption_df['date'] = pd.to_datetime(consumption_df['date'])

        # Use outer join to include all dates from both datasets
        # This ensures we don't lose dates where consumption might be zero
        result_df = pd.merge(
            generation_df,
            consumption_df,
            on='date',
            how='outer',
            validate='one_to_one'  # Ensure we have one-to-one mapping
        )

        # Fill NaN values with 0 for missing data
        result_df['generation_kwh'] = result_df['generation_kwh'].fillna(0)
        result_df['consumption_kwh'] = result_df['consumption_kwh'].fillna(0)

        # Log information about zero consumption days for debugging
        zero_consumption_days = result_df[result_df['consumption_kwh'] == 0]
        if not zero_consumption_days.empty:
            logger.info(f"Found {len(zero_consumption_days)} days with zero consumption for {selected_plant}")
            logger.info(f"Zero consumption dates: {zero_consumption_days['date'].tolist()}")

        # Only exclude rows where BOTH generation and consumption are zero
        # This preserves days with zero consumption but non-zero generation
        valid_rows = (result_df['generation_kwh'] > 0) | (result_df['consumption_kwh'] > 0)
        if valid_rows.any():
            result_df = result_df[valid_rows].copy()
            logger.info(f"Kept {len(result_df)} days with valid generation or consumption data")
        else:
            logger.warning(f"No valid data found for {selected_plant} between {start_date} and {end_date}")
            return pd.DataFrame()

        # Calculate surplus metrics
        result_df['surplus_generation'] = result_df.apply(
            lambda row: max(0, row['generation_kwh'] - row['consumption_kwh']), axis=1
        )
        result_df['surplus_demand'] = result_df.apply(
            lambda row: max(0, row['consumption_kwh'] - row['generation_kwh']), axis=1
        )

        return result_df
    except Exception as e:
        logger.error(f"Error merging generation and consumption data: {str(e)}")
        # Print more detailed information about the dataframes
        logger.error(f"Generation DataFrame shape: {generation_df.shape}, columns: {generation_df.columns.tolist()}")
        logger.error(f"Consumption DataFrame shape: {consumption_df.shape}, columns: {consumption_df.columns.tolist()}")

        # Try to identify the issue
        if 'date' in generation_df.columns and 'date' in consumption_df.columns:
            # Check data types
            logger.error(f"Generation date type: {generation_df['date'].dtypes}")
            logger.error(f"Consumption date type: {consumption_df['date'].dtypes}")

            # Check for duplicate dates
            if 'date' in generation_df.columns:
                logger.error(f"Generation date has duplicates: {generation_df['date'].duplicated().any()}")
                if generation_df['date'].duplicated().any():
                    logger.error(f"Number of duplicate dates: {generation_df['date'].duplicated().sum()}")

            if 'date' in consumption_df.columns:
                logger.error(f"Consumption date has duplicates: {consumption_df['date'].duplicated().any()}")
                if consumption_df['date'].duplicated().any():
                    logger.error(f"Number of duplicate dates: {consumption_df['date'].duplicated().sum()}")

            # Check for sample values
            if not generation_df.empty:
                logger.error(f"Generation date sample: {generation_df['date'].iloc[0]}")
            if not consumption_df.empty:
                logger.error(f"Consumption date sample: {consumption_df['date'].iloc[0]}")

        return pd.DataFrame()


def get_combined_wind_solar_generation(client_name, start_date, end_date):
    """
    Get combined wind and solar generation data for a client

    Args:
        client_name (str): Name of the client
        start_date (datetime): Start date to retrieve data for
        end_date (datetime): End date to retrieve data for

    Returns:
        DataFrame: Combined wind and solar generation data
    """
    import json
    import os

    # Load client data to get wind and solar plants
    json_path = os.path.join('src', 'client.json')
    try:
        with open(json_path, 'r') as f:
            client_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading client data: {e}")
        return pd.DataFrame()

    # Get solar plants for the client
    solar_plants_data = client_data.get('solar', {}).get(client_name, [])
    solar_plants = [plant.get('plant_id') for plant in solar_plants_data if plant.get('plant_id')]
    logger.info(f"Solar plants for {client_name}: {solar_plants}")

    # Get wind plants for the client
    wind_plants_data = client_data.get('wind', {}).get(client_name, [])
    wind_plants = [plant.get('plant_id') for plant in wind_plants_data if plant.get('plant_id')]
    logger.info(f"Wind plants for {client_name}: {wind_plants}")

    if not solar_plants and not wind_plants:
        logger.warning(f"No wind or solar plants found for client: {client_name}")
        return pd.DataFrame()

    # Format dates for API calls
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Create a DataFrame to hold all plants' data
    combined_df = pd.DataFrame()

    # Get data for each solar plant
    for plant_name in solar_plants:
        try:
            # Get generation data for this solar plant
            try:
                # Check if integration is initialized
                if integration is None:
                    logger.error(f"API integration not initialized. Cannot fetch data for solar plant {plant_name}")
                    continue

                logger.info(f"Fetching data for solar plant: {plant_name} from {start_str} to {end_str}")
                condition = {"Daily Energy": "last"}
                plant_df = integration.fetchDataV2(
                    plant_name,      # pName
                    "Plant",         # catList
                    ["Daily Energy"], # paramList
                    None,            # deviceList
                    start_str,       # sDate
                    end_str,         # eDate
                    granularity="1d",
                    condition=condition
                )

                # Log the result
                if plant_df is not None and not isinstance(plant_df, str):
                    logger.info(f"API returned solar plant dataframe with shape: {plant_df.shape}")
                else:
                    logger.warning(f"API returned non-dataframe result for solar plant: {type(plant_df)}")

                # Check if API returned None
                if plant_df is None:
                    logger.warning(f"API returned None for solar plant {plant_name}")
                    continue

                if not plant_df.empty:
                    # Ensure 'time' is the first column
                    if 'time' not in plant_df.columns:
                        logger.warning(f"API response doesn't contain 'time' column: {plant_df.columns}")
                        # Create a time column if it doesn't exist
                        plant_df['time'] = pd.date_range(start=start_date, end=end_date, freq='D')

                    # The second column contains generation values - always use iloc
                    if len(plant_df.columns) > 1:
                        # Get the name of the second column (generation data)
                        gen_col_name = plant_df.columns[1]

                        # Create a new dataframe with 'time' as first column
                        result_df = pd.DataFrame()
                        result_df['time'] = pd.to_datetime(plant_df['time'])
                        result_df['generation_kwh'] = plant_df[gen_col_name]

                        # Add plant name and source columns
                        result_df['PLANT_LONG_NAME'] = plant_name
                        result_df['source'] = 'Solar'

                        # Add date column for backward compatibility
                        result_df['date'] = result_df['time']

                        # Add to combined DataFrame
                        combined_df = pd.concat([combined_df, result_df[['time', 'date', 'PLANT_LONG_NAME', 'generation_kwh', 'source']]], ignore_index=True)
                        logger.info(f"Added {len(result_df)} rows of solar generation data for {plant_name}")
                    else:
                        logger.warning(f"API returned dataframe with unexpected columns for {plant_name}: {plant_df.columns}")
            except Exception as api_error:
                logger.error(f"API call failed for solar plant {plant_name}: {api_error}")
                logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error getting solar generation data for {plant_name}: {e}")
            logger.error(traceback.format_exc())

    # Get data for each wind plant
    for plant_name in wind_plants:
        try:
            # Get generation data for this wind plant
            try:
                # Check if integration is initialized
                if integration is None:
                    logger.error(f"API integration not initialized. Cannot fetch data for wind plant {plant_name}")
                    continue

                logger.info(f"Fetching data for wind plant: {plant_name} from {start_str} to {end_str}")
                plant_df = integration.fetchDataV2(
                    plant_name,      # pName
                    "Turbine",       # catList
                    ["WTUR.Generation today"],  # paramList
                    None,            # deviceList
                    start_str,       # Start Date
                    end_str,         # End Date
                    granularity="1d",
                    condition={"Generation today": "last"}
                )

                # Log the result
                if plant_df is not None and not isinstance(plant_df, str):
                    logger.info(f"API returned wind plant dataframe with shape: {plant_df.shape}")
                else:
                    logger.warning(f"API returned non-dataframe result for wind plant: {type(plant_df)}")



                # Check if API returned None
                if plant_df is None:
                    logger.warning(f"API returned None for wind plant {plant_name}")
                    continue

                if not plant_df.empty:
                    # Ensure 'time' is the first column
                    if 'time' not in plant_df.columns:
                        logger.warning(f"API response doesn't contain 'time' column: {plant_df.columns}")
                        # Create a time column if it doesn't exist
                        plant_df['time'] = pd.date_range(start=start_date, end=end_date, freq='D')

                    # For wind plants, we need to sum all turbine columns (all columns from index 1 onwards)
                    if len(plant_df.columns) > 1:
                        # Create a new dataframe with 'time' as first column
                        result_df = pd.DataFrame()
                        result_df['time'] = pd.to_datetime(plant_df['time'])

                        # Sum all columns from index 1 onwards (all turbine columns)
                        result_df['generation_kwh'] = plant_df.iloc[:, 1:].sum(axis=1)

                        # Add plant name and source columns
                        result_df['PLANT_LONG_NAME'] = plant_name
                        result_df['source'] = 'Wind'

                        # Add date column for backward compatibility
                        result_df['date'] = result_df['time']

                        # Add to combined DataFrame
                        combined_df = pd.concat([combined_df, result_df[['time', 'date', 'PLANT_LONG_NAME', 'generation_kwh', 'source']]], ignore_index=True)
                        logger.info(f"Added {len(result_df)} rows of wind generation data for {plant_name}")
                    else:
                        logger.warning(f"API returned dataframe with unexpected columns for {plant_name}: {plant_df.columns}")
            except Exception as api_error:
                logger.error(f"API call failed for wind plant {plant_name}: {api_error}")
                logger.error(traceback.format_exc())

                # Add plant name and source columns
                plant_df['PLANT_LONG_NAME'] = plant_name
                plant_df['source'] = 'Wind'

                # Add to combined DataFrame
                combined_df = pd.concat([combined_df, plant_df[['date', 'PLANT_LONG_NAME', 'generation_kwh', 'source']]], ignore_index=True)
                logger.info(f"Added {len(plant_df)} rows of wind generation data for {plant_name}")
        except Exception as e:
            logger.error(f"Error getting wind generation data for {plant_name}: {e}")
            logger.error(traceback.format_exc())

    if combined_df.empty:
        logger.warning(f"No generation data found for client {client_name} between {start_date} and {end_date}")
        return pd.DataFrame()

    # Convert date column to datetime
    combined_df['date'] = pd.to_datetime(combined_df['date'])

    # Log unique plants and their sources
    for plant in combined_df['PLANT_LONG_NAME'].unique():
        source = combined_df[combined_df['PLANT_LONG_NAME'] == plant]['source'].iloc[0]
        logger.info(f"Plant: {plant}, Source: {source}")

    # Print column names for debugging
    logger.info(f"Combined wind and solar data columns: {combined_df.columns.tolist()}")

    logger.info(f"Retrieved combined wind and solar generation data for {client_name}: {len(combined_df)} rows")
    return combined_df



def get_consumption_data_by_timeframe(plant_name, start_date, end_date=None):
    """
    Get consumption data with appropriate granularity based on date selection

    Args:
        plant_name (str): Name of the plant
        start_date (datetime): Start date to retrieve data for
        end_date (datetime, optional): End date for date range. If None or same as start_date,
                                      hourly data for start_date is returned.

    Returns:
        DataFrame: Consumption data with hourly or daily granularity
    """
    # Determine if we need hourly or daily data
    is_single_day = end_date is None or start_date == end_date

    if is_single_day:
        # For single day, get hourly data
        return get_consumption_data_from_csv(plant_name, start_date, start_date)
    else:
        # For date range, get daily data
        return get_daily_consumption_data(plant_name, start_date, end_date)

def get_generation_consumption_by_timeframe(plant_name, start_date, end_date=None):
    """
    Get generation and consumption comparison data with appropriate granularity

    Args:
        plant_name (str): Name of the plant
        start_date (datetime): Start date to retrieve data for
        end_date (datetime, optional): End date for date range. If None or same as start_date,
                                      hourly data for start_date is returned.

    Returns:
        DataFrame: Combined generation and consumption data with appropriate granularity
    """
    # Determine if we need hourly or daily data
    is_single_day = end_date is None or start_date == end_date

    if is_single_day:
        # For single day, get hourly comparison
        generation_df, consumption_df = get_generation_consumption_comparison(plant_name, start_date)
        if not generation_df.empty and not consumption_df.empty:
            return compare_generation_consumption(generation_df, consumption_df)
        return pd.DataFrame()
    else:
        # For date range, get daily comparison
        return get_daily_generation_consumption_comparison(plant_name, start_date, end_date)


def calculate_power_cost_metrics(plant_name, start_date, end_date, grid_rate_per_kwh):
    """
    Calculate power cost metrics including grid cost, actual cost, and savings.

    Args:
        plant_name (str): Name of the plant
        start_date (datetime): Start date
        end_date (datetime): End date
        grid_rate_per_kwh (float): Grid power cost per kWh

    Returns:
        DataFrame: DataFrame with cost metrics
    """
    try:
        logger.info(f"Starting power cost calculation for {plant_name} from {start_date} to {end_date}")

        # Get generation and consumption data using smart caching
        generation_df = get_generation_data_smart_wrapper(plant_name, start_date, end_date)
        consumption_df = get_consumption_data_by_timeframe(plant_name, start_date, end_date)

        logger.info(f"Retrieved generation data: {len(generation_df)} rows, columns: {generation_df.columns.tolist() if not generation_df.empty else 'empty'}")
        logger.info(f"Retrieved consumption data: {len(consumption_df)} rows, columns: {consumption_df.columns.tolist() if not consumption_df.empty else 'empty'}")

        if generation_df.empty or consumption_df.empty:
            logger.warning(f"No data available for cost calculation for {plant_name}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'time', 'date', 'consumption_kwh', 'generation_kwh',
                'net_consumption_kwh', 'grid_cost', 'actual_cost', 'savings', 'energy_offset_kwh'
            ])

        # Standardize column names for merging
        generation_df = standardize_dataframe_columns(generation_df)
        consumption_df = standardize_dataframe_columns(consumption_df)

        logger.info(f"After standardization - Generation columns: {generation_df.columns.tolist()}")
        logger.info(f"After standardization - Consumption columns: {consumption_df.columns.tolist()}")

        # Enhanced generation column detection and standardization
        generation_column_found = False

        # Check for various generation column patterns
        generation_patterns = [
            'generation_kwh', 'generation', 'plant_generation', 'total_generation',
            'energy', 'daily_energy', 'plant_energy', 'kwh', 'energy_kwh'
        ]

        for pattern in generation_patterns:
            matching_cols = [col for col in generation_df.columns if pattern in col.lower()]
            if matching_cols:
                # Use the first matching column
                original_col = matching_cols[0]
                if original_col != 'generation_kwh':
                    generation_df = generation_df.rename(columns={original_col: 'generation_kwh'})
                    logger.info(f"Renamed generation column from '{original_col}' to 'generation_kwh'")
                generation_column_found = True
                break

        if not generation_column_found:
            logger.error(f"No generation column found in generation data. Available columns: {generation_df.columns.tolist()}")
            return pd.DataFrame(columns=[
                'time', 'date', 'consumption_kwh', 'generation_kwh',
                'net_consumption_kwh', 'grid_cost', 'actual_cost', 'savings', 'energy_offset_kwh'
            ])

        # Enhanced consumption column detection and standardization
        consumption_column_found = False

        # Check for various consumption column patterns
        consumption_patterns = [
            'consumption_kwh', 'consumption', 'energy_kwh', 'total_consumption',
            'plant_consumption', 'kwh'
        ]

        for pattern in consumption_patterns:
            matching_cols = [col for col in consumption_df.columns if pattern in col.lower()]
            if matching_cols:
                # Use the first matching column
                original_col = matching_cols[0]
                if original_col != 'consumption_kwh':
                    consumption_df = consumption_df.rename(columns={original_col: 'consumption_kwh'})
                    logger.info(f"Renamed consumption column from '{original_col}' to 'consumption_kwh'")
                consumption_column_found = True
                break

        if not consumption_column_found:
            logger.error(f"No consumption column found in consumption data. Available columns: {consumption_df.columns.tolist()}")
            return pd.DataFrame(columns=[
                'time', 'date', 'consumption_kwh', 'generation_kwh',
                'net_consumption_kwh', 'grid_cost', 'actual_cost', 'savings', 'energy_offset_kwh'
            ])

        # Final validation
        logger.info(f"Final generation columns: {generation_df.columns.tolist()}")
        logger.info(f"Final consumption columns: {consumption_df.columns.tolist()}")

        # Determine merge column based on data granularity
        is_single_day = start_date == end_date

        # Clean up any duplicate columns before merging
        generation_df = generation_df.loc[:, ~generation_df.columns.duplicated()]
        consumption_df = consumption_df.loc[:, ~consumption_df.columns.duplicated()]

        # Validate that we have the required columns after standardization
        if 'generation_kwh' not in generation_df.columns:
            logger.error(f"generation_kwh column missing after standardization. Available: {generation_df.columns.tolist()}")
            return pd.DataFrame(columns=[
                'time', 'date', 'consumption_kwh', 'generation_kwh',
                'net_consumption_kwh', 'grid_cost', 'actual_cost', 'savings', 'energy_offset_kwh'
            ])

        if 'consumption_kwh' not in consumption_df.columns:
            logger.error(f"consumption_kwh column missing after standardization. Available: {consumption_df.columns.tolist()}")
            return pd.DataFrame(columns=[
                'time', 'date', 'consumption_kwh', 'generation_kwh',
                'net_consumption_kwh', 'grid_cost', 'actual_cost', 'savings', 'energy_offset_kwh'
            ])

        logger.info(f"Data validation passed. Generation: {len(generation_df)} rows, Consumption: {len(consumption_df)} rows")

        if is_single_day:
            # For single day power cost analysis, we need to distribute daily generation across consumption intervals
            logger.info("Processing single day data for power cost analysis")

            if 'datetime' in consumption_df.columns:
                # Use consumption datetime as the time reference
                consumption_df['time'] = consumption_df['datetime']
            elif 'quarter_hour' in consumption_df.columns and 'date' in consumption_df.columns:
                # Create time from date and quarter_hour
                consumption_df['time'] = pd.to_datetime(consumption_df['date']) + pd.to_timedelta(consumption_df['quarter_hour'], unit='h')

            # Get total daily generation and consumption
            total_daily_generation = generation_df['generation_kwh'].sum()
            total_daily_consumption = consumption_df['consumption_kwh'].sum()

            logger.info(f"Total daily generation: {total_daily_generation:.2f} kWh")
            logger.info(f"Total daily consumption: {total_daily_consumption:.2f} kWh")

            # Start with consumption data as base
            merged_df = consumption_df.copy()

            # Distribute generation proportionally based on consumption in each interval
            if total_daily_consumption > 0:
                # Proportional distribution: each interval gets generation proportional to its consumption
                merged_df['generation_kwh'] = (merged_df['consumption_kwh'] / total_daily_consumption) * total_daily_generation
            else:
                # If no consumption, distribute equally across all intervals
                merged_df['generation_kwh'] = total_daily_generation / len(merged_df) if len(merged_df) > 0 else 0

            logger.info(f"Distributed {total_daily_generation:.2f} kWh generation across {len(merged_df)} consumption intervals")
        else:
            # For date range, merge on date
            logger.info("Processing date range data for power cost analysis")

            if 'date' in generation_df.columns and 'date' in consumption_df.columns:
                # Ensure both date columns have the same data type
                generation_df['date'] = pd.to_datetime(generation_df['date']).dt.date
                consumption_df['date'] = pd.to_datetime(consumption_df['date']).dt.date
                merged_df = pd.merge(generation_df, consumption_df, on='date', how='outer')
                logger.info(f"Merged on date column: {len(merged_df)} rows")
            elif 'time' in generation_df.columns and 'time' in consumption_df.columns:
                # Convert time to date for daily aggregation
                generation_df_copy = generation_df.copy()
                consumption_df_copy = consumption_df.copy()

                generation_df_copy['date'] = pd.to_datetime(generation_df_copy['time']).dt.date
                consumption_df_copy['date'] = pd.to_datetime(consumption_df_copy['time']).dt.date

                # Aggregate by date
                gen_daily = generation_df_copy.groupby('date')['generation_kwh'].sum().reset_index()
                cons_daily = consumption_df_copy.groupby('date')['consumption_kwh'].sum().reset_index()

                merged_df = pd.merge(gen_daily, cons_daily, on='date', how='outer')
                logger.info(f"Merged time-based data aggregated by date: {len(merged_df)} rows")
            elif 'time' in generation_df.columns and 'date' in consumption_df.columns:
                # Generation has time, consumption has date - convert generation time to date
                generation_df_copy = generation_df.copy()
                generation_df_copy['date'] = pd.to_datetime(generation_df_copy['time']).dt.date

                # Aggregate generation by date
                gen_daily = generation_df_copy.groupby('date')['generation_kwh'].sum().reset_index()

                # Ensure consumption date is proper date type and aggregate by date
                consumption_df_copy = consumption_df.copy()
                consumption_df_copy['date'] = pd.to_datetime(consumption_df_copy['date']).dt.date

                # Aggregate consumption by date for multiple days view
                cons_daily = consumption_df_copy.groupby('date')['consumption_kwh'].sum().reset_index()

                merged_df = pd.merge(gen_daily, cons_daily, on='date', how='outer')
                logger.info(f"Merged generation (time-based) with consumption (date-based) - both aggregated by date: {len(merged_df)} rows")
            else:
                logger.warning("No common time/date columns found, attempting simple concatenation")
                # Reset indices to ensure proper alignment
                generation_df = generation_df.reset_index(drop=True)
                consumption_df = consumption_df.reset_index(drop=True)
                merged_df = pd.concat([generation_df, consumption_df], axis=1)
                logger.info(f"Concatenated data: {len(merged_df)} rows")

        # Fill missing values with 0
        merged_df = merged_df.fillna(0)
        logger.info(f"After merging and filling NaN: {len(merged_df)} rows, columns: {merged_df.columns.tolist()}")

        # Final validation before calculations
        if 'generation_kwh' not in merged_df.columns or 'consumption_kwh' not in merged_df.columns:
            logger.error(f"Required columns missing after merge. Available: {merged_df.columns.tolist()}")
            return pd.DataFrame(columns=[
                'time', 'date', 'consumption_kwh', 'generation_kwh',
                'net_consumption_kwh', 'grid_cost', 'actual_cost', 'savings', 'energy_offset_kwh'
            ])

        # Calculate power cost metrics
        logger.info("Calculating power cost metrics...")

        # Grid power cost (total consumption × grid rate)
        merged_df['grid_cost'] = merged_df['consumption_kwh'] * grid_rate_per_kwh

        # Net consumption (consumption - generation, minimum 0)
        merged_df['net_consumption_kwh'] = np.maximum(
            merged_df['consumption_kwh'] - merged_df['generation_kwh'], 0
        )

        # Actual power cost (net consumption × grid rate)
        merged_df['actual_cost'] = merged_df['net_consumption_kwh'] * grid_rate_per_kwh

        # Period savings (grid cost - actual cost)
        merged_df['savings'] = merged_df['grid_cost'] - merged_df['actual_cost']

        # Energy offset (generation used to offset consumption)
        merged_df['energy_offset_kwh'] = np.minimum(
            merged_df['generation_kwh'], merged_df['consumption_kwh']
        )

        logger.info(f"Power cost calculation completed. Final DataFrame: {len(merged_df)} rows")
        logger.info(f"Sample calculations - Grid cost: Rs.{merged_df['grid_cost'].sum():.2f}, Savings: Rs.{merged_df['savings'].sum():.2f}")

        return merged_df

    except Exception as e:
        logger.error(f"Error calculating power cost metrics: {e}")
        logger.error(traceback.format_exc())
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'time', 'date', 'consumption_kwh', 'generation_kwh',
            'net_consumption_kwh', 'grid_cost', 'actual_cost', 'savings', 'energy_offset_kwh'
        ])


def get_power_cost_summary(cost_df):
    """
    Generate summary statistics for power cost analysis.

    Args:
        cost_df (DataFrame): DataFrame with cost metrics

    Returns:
        dict: Summary statistics
    """
    try:
        if cost_df.empty:
            logger.warning("Empty DataFrame provided for power cost summary")
            return {}

        # Validate required columns exist
        required_columns = ['consumption_kwh', 'generation_kwh', 'net_consumption_kwh',
                          'grid_cost', 'actual_cost', 'savings', 'energy_offset_kwh']

        missing_columns = [col for col in required_columns if col not in cost_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for summary: {missing_columns}")
            logger.error(f"Available columns: {cost_df.columns.tolist()}")
            return {}

        logger.info(f"Generating summary for {len(cost_df)} rows of cost data")

        summary = {
            'total_consumption_kwh': cost_df['consumption_kwh'].sum(),
            'total_generation_kwh': cost_df['generation_kwh'].sum(),
            'total_net_consumption_kwh': cost_df['net_consumption_kwh'].sum(),
            'total_grid_cost': cost_df['grid_cost'].sum(),
            'total_actual_cost': cost_df['actual_cost'].sum(),
            'total_savings': cost_df['savings'].sum(),
            'total_energy_offset_kwh': cost_df['energy_offset_kwh'].sum(),
            'avg_daily_savings': cost_df['savings'].mean() if len(cost_df) > 0 else 0,
            'savings_percentage': (cost_df['savings'].sum() / cost_df['grid_cost'].sum() * 100) if cost_df['grid_cost'].sum() > 0 else 0
        }

        logger.info(f"Summary generated successfully: Total savings = Rs.{summary['total_savings']:.2f}")
        return summary

    except Exception as e:
        logger.error(f"Error generating power cost summary: {e}")
        logger.error(traceback.format_exc())
        return {}
