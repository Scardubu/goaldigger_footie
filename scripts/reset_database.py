#!/usr/bin/env python
"""
Database reset and initialization script for GoalDiggers.
This script will create a fresh database with all required tables.
"""
import asyncio
import logging
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.storage.database import DBManager
from utils.config import Config
from utils.logging_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

async def reset_database():
    """Reset and initialize the database with required tables."""
    logger.info("Starting database reset and initialization...")
    
    db_path = Config.get('database.path', 'database/football.db')
    
    # Create a new DB Manager instance
    db_manager = DBManager(db_path=db_path)
    
    try:
        # This will trigger schema initialization
        with db_manager.get_connection() as conn:
            logger.info("Initializing database schema...")
            db_manager._initialize_schema(conn)
            
            # Verify required tables exist
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"Created tables: {', '.join(tables)}")
            
            # Create some sample data if needed for testing
            if 'matches' in tables:
                # Check if matches table is empty
                cursor.execute("SELECT COUNT(*) FROM matches")
                count = cursor.fetchone()[0]
                if count == 0:
                    logger.info("Adding sample match data for testing...")
                    # Add a few sample matches for testing with the correct column names
                    sample_matches = [
                        ('m1', 'team123', 'team124', 'Arsenal', 'Chelsea', 'epl1', 'Premier League', '2025-05-25', 'SCHEDULED', None, None, None, None, None, None),
                        ('m2', 'team234', 'team235', 'Barcelona', 'Real Madrid', 'liga1', 'La Liga', '2025-05-26', 'SCHEDULED', None, None, None, None, None, None),
                        ('m3', 'team345', 'team346', 'Bayern Munich', 'Borussia Dortmund', 'bund1', 'Bundesliga', '2025-05-27', 'SCHEDULED', None, None, None, None, None, None),
                        ('m4', 'team456', 'team457', 'Inter Milan', 'Juventus', 'seriea1', 'Serie A', '2025-05-28', 'SCHEDULED', None, None, None, None, None, None),
                        ('m5', 'team567', 'team568', 'PSG', 'Marseille', 'ligue1', 'Ligue 1', '2025-05-29', 'SCHEDULED', None, None, None, None, None, None),
                        ('m6', 'team678', 'team679', 'Ajax', 'PSV', 'ered1', 'Eredivisie', '2025-05-30', 'SCHEDULED', None, None, None, None, None, None)
                    ]
                    cursor.executemany(
                        "INSERT INTO matches (id, home_team_id, away_team_id, competition_id, competition, match_date, status, home_score, away_score, total_goals, winner, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        sample_matches
                    )
                    conn.commit()
                    logger.info(f"Added {len(sample_matches)} sample matches")
                    
        logger.info("Database reset and initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error during database reset: {e}")
        return False
    finally:
        db_manager.close()

if __name__ == "__main__":
    # Run the async function in a proper event loop
    try:
        asyncio.run(reset_database())
    except Exception as e:
        logger.error(f"Uncaught exception: {e}")
        sys.exit(1)
