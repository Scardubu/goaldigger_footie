"""
Database update utility for the football betting insights platform.
Provides a simple interface to update all database data.
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from database.db_manager import DatabaseManager
from database.schema import Match
from scripts.data_pipeline.db_integrator import DataIntegrator

logger = logging.getLogger(__name__)

async def _run_full_update(days_ahead: int = 14, force_refresh: bool = False) -> bool:
    """Internal async function to run a full database update."""
    try:
        # Create data integrator
        integrator = DataIntegrator()
        
        # Update leagues and teams
        league_results = await integrator.update_all_leagues()
        logger.info(f"League update results: {league_results}")
        
        # Update upcoming matches
        match_results = await integrator.update_upcoming_matches_all_leagues(days_ahead)
        logger.info(f"Match update results: {match_results}")
        
        # Calculate success rate
        league_success = sum(1 for success in league_results.values() if success) / len(league_results) if league_results else 0
        match_success = sum(1 for success in match_results.values() if success) / len(match_results) if match_results else 0
        
        # Consider update successful if at least 70% of operations succeeded
        return (league_success + match_success) / 2 >= 0.7
    
    except Exception as e:
        logger.error(f"Error during full database update: {e}")
        return False


def update_all_data(days_ahead: int = 14, force_refresh: bool = False) -> bool:
    """Entry point to update all data in the database.
    
    Args:
        days_ahead: Number of days ahead to fetch data for
        force_refresh: Whether to force update regardless of last update time
        
    Returns:
        True if update was successful, False otherwise
    """
    try:
        # Check if update is needed (unless force_refresh is True)
        if not force_refresh:
            db = DatabaseManager()
            
            with db.session_scope() as session:
                # Check when the last update was performed
                last_match = session.query(Match).order_by(Match.updated_at.desc()).first()
                if last_match and last_match.updated_at:
                    # If updated in the last 6 hours, skip update
                    hours_since_update = (datetime.now() - last_match.updated_at).total_seconds() / 3600
                    if hours_since_update < 6:
                        logger.info(f"Skipping update, last update was {hours_since_update:.2f} hours ago")
                        return True
        
        # Run the update asynchronously
        return asyncio.run(_run_full_update(days_ahead, force_refresh))
    
    except Exception as e:
        logger.error(f"Error in update_all_data: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the update
    logger.info("Starting database update...")
    success = update_all_data(force_refresh=True)
    logger.info(f"Database update {'successful' if success else 'failed'}")
