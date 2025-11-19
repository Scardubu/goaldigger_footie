"""
Database module initialization.
Provides initialization functions and access to the database manager.
"""
import logging
import os
from typing import Any, Dict, Optional

from database.db_manager import DatabaseManager
from database.schema import (
    League, Team, Match, MatchStats, TeamStats,
    Prediction, Odds, ValueBet
)
from utils.config import Config

logger = logging.getLogger(__name__)

def initialize_database() -> DatabaseManager: # Return the instance
    """
    Initialize the database with schema and create initial data if needed.
    This should be called during application startup.
    Returns the initialized DatabaseManager instance.
    """
    # Try to get database URI from environment variables first, then fall back to config
    db_uri = os.getenv('DATABASE_URI')
    if not db_uri:
        db_uri = Config.get('database.uri', 'sqlite:///data/football.db')
    
    # Ensure we have a valid URI - default to SQLite if nothing else works
    if not db_uri:
        # Provide a default URI that will definitely work
        data_dir = os.getenv('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
        os.makedirs(data_dir, exist_ok=True)
        db_uri = f"sqlite:///{os.path.join(data_dir, 'football.db')}"
        logger.warning(f"No database URI found in environment or config. Using default: {db_uri}")
    
    logger.info(f"Database URI set to: {db_uri}")

    # Ensure database directory exists for SQLite
    if db_uri.startswith('sqlite:///'):
        db_path = os.path.dirname(db_uri.replace('sqlite:///', ''))
        if db_path and not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
            logger.info(f"Created database directory: {db_path}")

    # Create and initialize the DatabaseManager instance
    local_db_manager = DatabaseManager(db_uri=db_uri)
    local_db_manager.create_tables() # Initialize the database schema
    logger.info("Database schema initialized successfully.")

    # Initialize default data if needed
    with local_db_manager.session_scope() as session:
        # Check if we need to initialize leagues
        if session.query(League).count() == 0:
            _initialize_default_leagues(session)
            logger.info("Initialized default leagues.")

        # Check if we need to populate historical data
        if session.query(Match).count() == 0:
            logger.info("No historical match data found. Triggering data population...")
            _populate_historical_data()

    logger.info("Database initialization completed.")
    return local_db_manager

def _initialize_default_leagues(session: Any):
    """
    Initialize default leagues in the database.
    """
    default_leagues = [
        League(id="premier_league", name="Premier League", country="England", tier=1, api_id="PL"),
        League(id="la_liga", name="La Liga", country="Spain", tier=1, api_id="PD"),
        League(id="bundesliga", name="Bundesliga", country="Germany", tier=1, api_id="BL1"),
        League(id="serie_a", name="Serie A", country="Italy", tier=1, api_id="SA"),
        League(id="ligue_1", name="Ligue 1", country="France", tier=1, api_id="FL1"),
        League(id="eredivisie", name="Eredivisie", country="Netherlands", tier=1, api_id="DED"),
        League(id="primeira_liga", name="Primeira Liga", country="Portugal", tier=1, api_id="PPL"),
        League(id="championship", name="Championship", country="England", tier=2, api_id="ELC"),
        League(id="bundesliga2", name="2. Bundesliga", country="Germany", tier=2, api_id="BL2"),
        League(id="serie_b", name="Serie B", country="Italy", tier=2, api_id="SB"),
    ]
    for league in default_leagues:
        session.add(league)
    session.commit()
    logger.info(f"Added {len(default_leagues)} default leagues to the database.")

def _populate_historical_data():
    """
    Populate the database with historical football data.
    This is called automatically when no match data exists.
    """
    try:
        import subprocess
        import sys
        from pathlib import Path

        # Get the project root directory
        project_root = Path(__file__).parent.parent
        populate_script = project_root / "populate_historical_data.py"

        if not populate_script.exists():
            logger.warning(f"Historical data population script not found at {populate_script}")
            return

        logger.info("Starting historical data population...")

        # Run the population script
        result = subprocess.run(
            [sys.executable, str(populate_script)],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=project_root
        )

        if result.returncode == 0:
            logger.info("Historical data population completed successfully")
            if result.stdout:
                logger.info(f"Population output: {result.stdout}")
        else:
            logger.error(f"Historical data population failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Population error: {result.stderr}")

    except subprocess.TimeoutExpired:
        logger.error("Historical data population timed out after 10 minutes")
    except Exception as e:
        logger.error(f"Error during historical data population: {e}")

# Export commonly used items
# Note: db_manager is no longer exported as a global instance from here.
# Consumers should call initialize_database() to get an instance.
__all__ = [
    'initialize_database', 'DatabaseManager', # Export the function and class
    'League', 'Team', 'Match', 'MatchStats', 'TeamStats',
    'Prediction', 'Odds', 'ValueBet'
]