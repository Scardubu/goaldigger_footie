#!/usr/bin/env python
"""
Database Schema Update for GoalDiggers Platform

This script updates the database schema to ensure compatibility with the latest model definitions,
handling the addition of new fields like 'tla' in the Team model.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import after setting up logging
from database.db_manager import DatabaseManager
from database.schema import (Base, League, Match, MatchStats, Odds, Prediction,
                             Team, TeamStats, ValueBet)


def update_schema():
    """Update the database schema to match the current model definitions."""
    logger.info("Starting database schema update...")
    
    # Initialize database manager
    db_manager = DatabaseManager()
    
    # First, create tables if they don't exist
    db_manager.create_tables()
    
    logger.info("Database schema update completed successfully.")
    return 0

def main():
    """Main function to parse arguments and run the update."""
    parser = argparse.ArgumentParser(description="Update the database schema for the GoalDiggers platform.")
    parser.add_argument("--db-uri", type=str, help="Database URI (optional, uses config if not provided)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Use provided URI if given
    if args.db_uri:
        os.environ["DATABASE_URI"] = args.db_uri
        logger.info(f"Using provided database URI: {args.db_uri}")
    
    try:
        return update_schema()
    except Exception as e:
        logger.error(f"Error updating database schema: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
