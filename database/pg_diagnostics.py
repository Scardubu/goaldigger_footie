"""
PostgreSQL Connection Diagnostics Utility

This script provides diagnostics and setup tools for PostgreSQL connections.
It helps validate configuration, test connections, and optimize settings.
"""

import argparse
import logging
import os
import sys
from typing import Any, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.pg_config_validator import (
    get_postgres_config_status,
    print_postgres_diagnostic_report,
    test_postgres_connection,
    validate_postgres_config,
    verify_postgres_permissions,
)


def run_diagnostics():
    """Run comprehensive PostgreSQL diagnostics and print report."""
    print_postgres_diagnostic_report()

def check_config():
    """Check PostgreSQL configuration and return status."""
    is_valid, validation = validate_postgres_config()
    print(f"PostgreSQL Configuration Valid: {is_valid}")
    if not is_valid and validation.get("missing_required"):
        print(f"Missing required variables: {validation['missing_required']}")
    return is_valid

def setup_postgres_pool():
    """Configure optimized PostgreSQL connection pooling."""
    # Import database manager here to avoid circular imports
    from database.db_manager import DatabaseManager

    # Get PostgreSQL URI from environment
    uri = os.getenv("DATABASE_URL")
    if not uri:
        from database.db_manager import _build_postgres_uri_from_env
        uri, _ = _build_postgres_uri_from_env()
    
    if not uri:
        print("Error: No PostgreSQL URI available. Set DATABASE_URL or POSTGRES_* variables.")
        return False
    
    # Get optimal pool settings based on environment
    cpu_count = os.cpu_count() or 4
    recommended_pool_size = max(5, cpu_count * 2)
    recommended_max_overflow = max(10, cpu_count * 4)
    
    # Create database manager with optimized settings
    print(f"Setting up optimized PostgreSQL pool: size={recommended_pool_size}, max_overflow={recommended_max_overflow}")
    
    try:
        db_manager = DatabaseManager(uri)
        db_manager._setup_connection_pool(
            uri,
            pool_size=recommended_pool_size,
            max_overflow=recommended_max_overflow,
            timeout=30,
            recycle=1800
        )
        
        # Test connection with pool
        if db_manager.test_connection():
            print("PostgreSQL connection pool configured and tested successfully.")
            return True
        else:
            print("Failed to connect to PostgreSQL with optimized pool settings.")
            return False
            
    except Exception as e:
        print(f"Error setting up PostgreSQL pool: {e}")
        return False

def main():
    """Main function to parse arguments and run selected operation."""
    parser = argparse.ArgumentParser(description='PostgreSQL Connection Diagnostics Utility')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Diagnostics command
    diag_parser = subparsers.add_parser('diagnose', help='Run comprehensive PostgreSQL diagnostics')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check PostgreSQL configuration')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up optimized PostgreSQL connection pool')
    
    args = parser.parse_args()
    
    # Run selected command
    if args.command == 'diagnose':
        run_diagnostics()
    elif args.command == 'check':
        check_config()
    elif args.command == 'setup':
        setup_postgres_pool()
    else:
        # Default to diagnostics if no command specified
        run_diagnostics()

if __name__ == '__main__':
    main()