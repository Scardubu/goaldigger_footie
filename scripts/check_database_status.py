#!/usr/bin/env python
"""
Script to check the status of the database population.
Displays counts of leagues, teams, matches, and other entities.
"""
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to check database status."""
    # Import here to ensure paths are set up correctly
    from database.db_manager import DatabaseManager
    from database.schema import (League, Match, MatchStats, Odds, Prediction,
                                 Team)

    logger.info("Checking database status...")
    
    # Initialize database manager
    db = DatabaseManager()
    
    try:
        with db.get_session() as session:
            # Count entities
            league_count = session.query(League).count()
            team_count = session.query(Team).count()
            match_count = session.query(Match).count()
            match_stats_count = session.query(MatchStats).count()
            prediction_count = session.query(Prediction).count()
            odds_count = session.query(Odds).count()
            
            logger.info(f"Database status:")
            logger.info(f"- Leagues: {league_count}")
            logger.info(f"- Teams: {team_count}")
            logger.info(f"- Matches: {match_count}")
            logger.info(f"- Match Stats: {match_stats_count}")
            logger.info(f"- Predictions: {prediction_count}")
            logger.info(f"- Odds: {odds_count}")
            
            # Get sample of teams if any exist
            if team_count > 0:
                teams = session.query(Team).limit(5).all()
                logger.info("\nSample Teams:")
                for team in teams:
                    logger.info(f"  - {team.name} (ID: {team.id}, League: {team.league_id}, TLA: {team.tla})")
            
            # Get sample of leagues if any exist
            if league_count > 0:
                leagues = session.query(League).limit(5).all()
                logger.info("\nSample Leagues:")
                for league in leagues:
                    logger.info(f"  - {league.name} (ID: {league.id}, Country: {league.country})")
            
            return 0
    except Exception as e:
        logger.error(f"Error checking database status: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
