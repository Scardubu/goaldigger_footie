"""
PostgreSQL Production Optimization Script

This script sets up and optimizes PostgreSQL for production use by:
1. Validating PostgreSQL configuration
2. Testing the connection
3. Optimizing connection pool settings
4. Creating a connection manager singleton for app-wide use
5. Migrating data from SQLite if needed
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from database.db_manager import (
    DatabaseManager,
    _build_postgres_uri_from_env,
    _connect_args_for_postgres_from_env,
)

# Import database utilities
from database.pg_config_validator import (
    test_postgres_connection,
    validate_postgres_config,
)


def validate_environment() -> bool:
    """
    Validate that the environment has the necessary PostgreSQL configuration.
    
    Returns:
        Boolean indicating if environment is valid
    """
    logger.info("Validating PostgreSQL environment variables...")
    is_valid, validation = validate_postgres_config()
    
    if not is_valid:
        logger.error("PostgreSQL configuration is invalid")
        if validation.get("missing_required"):
            logger.error(f"Missing required variables: {validation['missing_required']}")
        return False
    
    logger.info("PostgreSQL configuration is valid")
    return True

def test_connection() -> bool:
    """
    Test the PostgreSQL connection.
    
    Returns:
        Boolean indicating if connection test was successful
    """
    logger.info("Testing PostgreSQL connection...")
    
    # Get PostgreSQL URI
    uri = os.getenv("DATABASE_URL")
    if not uri:
        uri, connect_args = _build_postgres_uri_from_env()
    else:
        connect_args = _connect_args_for_postgres_from_env()
    
    if not uri:
        logger.error("No PostgreSQL URI available")
        return False
    
    # Test connection
    success, result = test_postgres_connection(uri, connect_args)
    
    if success:
        logger.info(f"PostgreSQL connection successful (latency: {result['latency_ms']:.2f}ms)")
        if 'server_version' in result:
            logger.info(f"PostgreSQL version: {result['server_version'].split(',')[0]}")
        return True
    else:
        logger.error(f"PostgreSQL connection failed: {result['error']}")
        return False

def migrate_data_from_sqlite(source_path: str, target_pg_uri: str) -> bool:
    """
    Migrate data from SQLite database to PostgreSQL.
    
    Args:
        source_path: Path to SQLite database
        target_pg_uri: Target PostgreSQL URI
        
    Returns:
        Boolean indicating if migration was successful
    """
    try:
        import pandas as pd
        from sqlalchemy import MetaData, Table, create_engine
        
        logger.info(f"Starting data migration from SQLite ({source_path}) to PostgreSQL...")
        
        # Create engines
        sqlite_uri = f"sqlite:///{source_path}"
        sqlite_engine = create_engine(sqlite_uri)
        pg_engine = create_engine(target_pg_uri)
        
        # Get SQLite tables
        metadata = MetaData()
        metadata.reflect(bind=sqlite_engine)
        tables = metadata.tables
        
        # Create tables in PostgreSQL
        metadata.create_all(pg_engine)
        
        # Migrate data for each table
        for table_name, table in tables.items():
            logger.info(f"Migrating table: {table_name}")
            
            # Read data from SQLite
            df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_engine)
            
            if not df.empty:
                # Write data to PostgreSQL
                df.to_sql(table_name, pg_engine, if_exists='append', index=False)
                logger.info(f"Migrated {len(df)} rows from {table_name}")
            else:
                logger.info(f"Table {table_name} is empty, skipping")
        
        logger.info("Data migration completed successfully")
        return True
        
    except ImportError:
        logger.error("Required packages (pandas) not available for data migration")
        return False
    except Exception as e:
        logger.error(f"Error during data migration: {e}")
        return False

def optimize_pool_settings() -> Tuple[int, int, int, int]:
    """
    Calculate optimal pool settings based on environment.
    
    Returns:
        Tuple containing:
            - pool_size
            - max_overflow
            - timeout
            - recycle
    """
    # Calculate based on available CPUs
    cpu_count = os.cpu_count() or 4
    
    # Get settings from environment or use calculated defaults
    pool_size = int(os.getenv('POSTGRES_POOL_SIZE', str(max(5, cpu_count * 2))))
    max_overflow = int(os.getenv('POSTGRES_MAX_OVERFLOW', str(max(10, cpu_count * 4))))
    timeout = int(os.getenv('POSTGRES_POOL_TIMEOUT', '30'))
    recycle = int(os.getenv('POSTGRES_POOL_RECYCLE', '1800'))  # 30 minutes
    
    return pool_size, max_overflow, timeout, recycle

def setup_production_database() -> Optional[DatabaseManager]:
    """
    Set up the production PostgreSQL database with optimized settings.
    
    Returns:
        DatabaseManager instance if successful, None otherwise
    """
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return None
    
    # Test connection
    if not test_connection():
        logger.error("Connection test failed")
        return None
    
    # Get PostgreSQL URI
    uri = os.getenv("DATABASE_URL")
    if not uri:
        uri, connect_args = _build_postgres_uri_from_env()
    else:
        connect_args = _connect_args_for_postgres_from_env()
    
    if not uri:
        logger.error("No PostgreSQL URI available")
        return None
    
    # Get optimized pool settings
    pool_size, max_overflow, timeout, recycle = optimize_pool_settings()
    
    try:
        # Create database manager
        logger.info(f"Creating database manager with optimized settings: pool_size={pool_size}, max_overflow={max_overflow}")
        db_manager = DatabaseManager(uri)
        
        # Configure connection pool
        db_manager._setup_connection_pool(
            uri,
            pool_size=pool_size,
            max_overflow=max_overflow,
            timeout=timeout,
            recycle=recycle
        )
        
        # Create tables
        logger.info("Creating tables...")
        db_manager.create_tables()
        
        logger.info("Production PostgreSQL database set up successfully")
        return db_manager
        
    except Exception as e:
        logger.error(f"Error setting up production database: {e}")
        return None

def main():
    """Main function to parse arguments and run setup."""
    parser = argparse.ArgumentParser(description='PostgreSQL Production Optimization Script')
    
    parser.add_argument('--migrate', action='store_true', help='Migrate data from SQLite to PostgreSQL')
    parser.add_argument('--sqlite-path', type=str, help='Path to SQLite database for migration')
    parser.add_argument('--validate-only', action='store_true', help='Only validate environment and connection')
    
    args = parser.parse_args()
    
    # Validate environment and connection
    if args.validate_only:
        if validate_environment() and test_connection():
            logger.info("Validation successful")
            return 0
        else:
            logger.error("Validation failed")
            return 1
    
    # Set up production database
    db_manager = setup_production_database()
    if not db_manager:
        logger.error("Failed to set up production database")
        return 1
    
    # Migrate data if requested
    if args.migrate:
        if not args.sqlite_path:
            logger.error("SQLite path not provided for migration")
            return 1
        
        if os.path.exists(args.sqlite_path):
            if migrate_data_from_sqlite(args.sqlite_path, db_manager.db_uri):
                logger.info("Data migration completed successfully")
            else:
                logger.error("Data migration failed")
                return 1
        else:
            logger.error(f"SQLite database not found: {args.sqlite_path}")
            return 1
    
    logger.info("PostgreSQL production optimization completed successfully")
    return 0

if __name__ == '__main__':
    sys.exit(main())