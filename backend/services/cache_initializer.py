"""
Cache Initializer for setting up bulk data caching.

This module provides functions to initialize and manage the smart caching system.
"""

import streamlit as st
from datetime import datetime

from backend.logs.logger_setup import setup_logger

logger = setup_logger('cache_initializer', 'cache_initializer.log')

def initialize_cache_system():
    """
    Initialize the cache system when the app starts.
    This function should be called once during app startup.
    """
    try:
        from backend.config.app_config import CONFIG

        # Check if smart caching is enabled
        if not CONFIG["data"].get("enable_smart_caching", False):
            logger.info("Smart caching is disabled in configuration")
            return

        from backend.services.smart_data_fetcher import smart_data_fetcher
        from backend.api.api_cache_manager import api_cache_manager

        logger.info("Initializing cache system...")

        # Initialize bulk cache if enabled
        if CONFIG["data"].get("bulk_fetch_enabled", True):
            # Check if we should run bulk fetch (avoid running on every app restart)
            if should_run_bulk_fetch():
                logger.info("Starting bulk cache initialization...")

                # Run cache initialization silently in background
                try:
                    smart_data_fetcher.initialize_bulk_cache()
                    logger.info("Cache initialization completed successfully")
                except Exception as e:
                    logger.error(f"Cache initialization failed: {e}")

                # Mark bulk fetch as completed
                mark_bulk_fetch_completed()
            else:
                logger.info("Bulk fetch not needed at this time")

        # Cleanup old cache entries
        cleanup_days = CONFIG["data"].get("auto_cleanup_days", 30)
        api_cache_manager.cleanup_old_cache(cleanup_days)

        logger.info("Cache system initialization completed")

    except Exception as e:
        logger.error(f"Failed to initialize cache system: {e}")

def should_run_bulk_fetch() -> bool:
    """
    Check if bulk fetch should be run.

    Returns:
        True if bulk fetch should be run
    """
    try:
        # Check session state to avoid running multiple times in same session
        if 'bulk_fetch_completed' in st.session_state:
            return False

        from backend.api.api_cache_manager import api_cache_manager

        # Simple check - if we have very few cache entries, run bulk fetch
        cache_stats = api_cache_manager.get_cache_stats()
        if cache_stats["total_entries"] < 10:
            return True

        return False

    except Exception as e:
        logger.error(f"Error checking bulk fetch status: {e}")
        return False

def mark_bulk_fetch_completed():
    """Mark bulk fetch as completed in session state."""
    try:
        st.session_state['bulk_fetch_completed'] = True
        st.session_state['bulk_fetch_time'] = datetime.now()
    except:
        pass  # Not in Streamlit context
