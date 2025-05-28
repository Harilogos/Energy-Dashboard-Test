"""
Configuration settings for the Prescinto API integration.
"""
import os
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# API Configuration - Using the same values as in the test notebook
INTEGRATION_SERVER = "IN"
INTEGRATION_TOKEN = os.getenv("PRESCINTO_API_TOKEN")

# Configure logging
from backend.logs.logger_setup import setup_logger
logger = setup_logger('api_config', 'api_config.log')

def get_api_credentials():
    """
    Get API credentials for Prescinto integration

    Returns:
        tuple: (server, token) for API authentication
    """
    try:
        # Return the credentials
        return INTEGRATION_SERVER, INTEGRATION_TOKEN
    except Exception as e:
        logger.error(f"Error getting API credentials: {e}")
        logger.error(traceback.format_exc())
        # Return default values if there's an error
        return "Error"
