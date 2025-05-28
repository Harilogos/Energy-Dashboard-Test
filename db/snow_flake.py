"""
Snowflake connection management module with connection pooling.
"""
import snowflake.connector
import os
from dotenv import load_dotenv
from contextlib import contextmanager
import pandas as pd
from threading import Lock
import time

# Configure logging
from backend.logs.logger_setup import setup_logger


logger = setup_logger('snow_flake', 'snow_flake.log')

# Load environment variables
load_dotenv()

class SnowflakeConnectionPool:
    """Thread-safe connection pool for Snowflake connections"""

    def __init__(self, max_size=5, connection_timeout=300):
        self.max_size = max_size
        self.connection_timeout = connection_timeout
        self.pool = []
        self.lock = Lock()
        self.config = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        }
        logger.info(f"Initialized Snowflake connection pool with max size {max_size}")

    def get_connection(self):
        """Get a connection from the pool or create a new one"""
        with self.lock:
            # Try to get a connection from the pool
            if self.pool:
                conn = self.pool.pop()
                try:
                    # Test if connection is still alive
                    conn.cursor().execute("SELECT 1")
                    logger.debug("Reusing existing connection from pool")
                    return conn
                except Exception as e:
                    logger.warning(f"Discarding stale connection: {e}")
                    try:
                        conn.close()
                    except:
                        pass

            # Create a new connection
            try:
                logger.info("Creating new Snowflake connection")
                return snowflake.connector.connect(**self.config)
            except Exception as e:
                logger.error(f"Failed to connect to Snowflake: {e}")
                raise ConnectionError(f"Could not establish Snowflake connection: {e}")

    def release_connection(self, conn):
        """Return a connection to the pool"""
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(conn)
                logger.debug("Connection returned to pool")
            else:
                logger.debug("Pool full, closing connection")
                conn.close()

# Global connection pool instance
_connection_pool = SnowflakeConnectionPool()

@contextmanager
def snowflake_connection():
    """Context manager for Snowflake connections"""
    conn = None
    try:
        conn = _connection_pool.get_connection()
        yield conn
    except Exception as e:
        logger.error(f"Error with Snowflake connection: {e}")
        raise
    finally:
        if conn:
            _connection_pool.release_connection(conn)

def execute_query(query, params=None, retries=3, retry_delay=1):
    """
    Execute a query and return the results as a DataFrame

    Args:
        query (str): SQL query to execute
        params (tuple, optional): Parameters for the query
        retries (int): Number of retries on failure
        retry_delay (int): Delay between retries in seconds

    Returns:
        DataFrame: Query results
    """
    last_error = None

    for attempt in range(retries):
        try:
            with snowflake_connection() as conn:
                if params:
                    df = pd.read_sql(query, conn, params=params)
                else:
                    df = pd.read_sql(query, conn)
                return df
        except Exception as e:
            last_error = e
            logger.warning(f"Query attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)

    # If we get here, all retries failed
    logger.error(f"Query execution failed after {retries} attempts")
    logger.error(f"Query: {query}")
    logger.error(f"Params: {params}")
    logger.error(f"Last error: {last_error}")
    raise last_error

def get_connection():
    """Legacy function to maintain compatibility"""
    try:
        return _connection_pool.get_connection()
    except Exception as e:
        logger.error(f"Failed to get Snowflake connection: {e}")
        raise ConnectionError(f"Could not establish Snowflake connection: {e}")
