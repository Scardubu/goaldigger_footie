"""
Data pipeline runner for the football betting insights platform.
Executes data collection, integration, and preprocessing tasks.
"""
import argparse
import asyncio
import logging
import os
from datetime import datetime, timedelta

from database.db_manager import DatabaseManager
from scripts.data_pipeline.db_integrator import DataIntegrator
from scripts.scrapers.scraper_factory import ScraperFactory
from utils.config import Config
from utils.logging_config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)

async def run_initial_data_load():
    """
    Run initial data load to populate the database with leagues, teams, and upcoming matches.
    """
    logger.info("Starting initial data load...")
    
    # Initialize components
    db_manager = DatabaseManager()
    scraper_factory = ScraperFactory()
    data_integrator = DataIntegrator(db_manager, scraper_factory)
    
    try:
        # Update all supported leagues
        logger.info("Updating leagues...")
        league_results = await data_integrator.update_all_leagues()
        
        success_count = sum(1 for result in league_results.values() if result)
        logger.info(f"Updated {success_count}/{len(league_results)} leagues")
        
        # Update upcoming matches for all leagues
        logger.info("Updating upcoming matches...")
        days_ahead = Config.get("data_pipeline.upcoming_matches_days", 14)
        match_results = await data_integrator.update_upcoming_matches_all_leagues(days_ahead=days_ahead)
        
        success_count = sum(1 for result in match_results.values() if result)
        logger.info(f"Updated matches for {success_count}/{len(match_results)} leagues")
        
        # Fetch odds for all upcoming matches
        logger.info("Fetching odds for upcoming matches...")
        
        # Get all upcoming matches from database
        with db_manager.session_scope() as session:
            from datetime import timezone

            from database.schema import Match
            upcoming_matches = session.query(Match).filter(
                Match.status == "scheduled",
                Match.match_date >= datetime.now(timezone.utc),
                Match.match_date <= datetime.now(timezone.utc) + timedelta(days=days_ahead)
            ).all()
            
            logger.info(f"Found {len(upcoming_matches)} upcoming matches")
            
            # Concurrent odds integration with bounded concurrency
            semaphore = asyncio.Semaphore(10)
            odds_success = 0

            async def _fetch_odds(match_id: str):
                nonlocal odds_success
                async with semaphore:
                    try:
                        if await data_integrator.integrate_odds(match_id):
                            odds_success += 1
                    except Exception as e:
                        logger.debug(f"Odds fetch failed for {match_id}: {e}")

            await asyncio.gather(*[_fetch_odds(m.id) for m in upcoming_matches])
            logger.info(f"Updated odds for {odds_success}/{len(upcoming_matches)} matches (concurrent)")
        
        logger.info("Initial data load completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during initial data load: {e}")
        return False

async def update_match_data(match_id: str):
    """
    Update data for a specific match, including details and odds.
    
    Args:
        match_id: Match ID to update
    """
    logger.info(f"Updating data for match: {match_id}")
    
    # Initialize components
    db_manager = DatabaseManager()
    scraper_factory = ScraperFactory()
    data_integrator = DataIntegrator(db_manager, scraper_factory)
    
    try:
        # Update match details
        details_success = await data_integrator.integrate_match_details(match_id)
        logger.info(f"Updated match details: {details_success}")
        
        # Update odds
        odds_success = await data_integrator.integrate_odds(match_id)
        logger.info(f"Updated match odds: {odds_success}")
        
        return details_success and odds_success
        
    except Exception as e:
        logger.error(f"Error updating match data: {e}")
        return False

async def update_league_data(league_code: str, days_ahead: int = 7):
    """
    Update data for a specific league, including teams and upcoming matches.
    
    Args:
        league_code: League code to update
        days_ahead: Number of days ahead to fetch matches for
    """
    logger.info(f"Updating data for league: {league_code}")
    
    # Initialize components
    db_manager = DatabaseManager()
    scraper_factory = ScraperFactory()
    data_integrator = DataIntegrator(db_manager, scraper_factory)
    
    try:
        # Update league data
        league_success = await data_integrator.integrate_league_data(league_code)
        logger.info(f"Updated league data: {league_success}")
        
        # Update upcoming matches
        matches_success = await data_integrator.integrate_upcoming_matches(league_code, days_ahead)
        logger.info(f"Updated upcoming matches: {matches_success}")
        
        return league_success and matches_success
        
    except Exception as e:
        logger.error(f"Error updating league data: {e}")
        return False

async def main():
    """Main entry point for the data pipeline runner."""
    parser = argparse.ArgumentParser(description="Football Betting Insights Data Pipeline Runner")
    parser.add_argument("--initial-load", action="store_true", help="Run initial data load")
    parser.add_argument("--update-match", help="Update data for a specific match")
    parser.add_argument("--update-league", help="Update data for a specific league")
    parser.add_argument("--days-ahead", type=int, default=7, help="Number of days ahead to fetch matches for")
    
    args = parser.parse_args()
    
    if args.initial_load:
        await run_initial_data_load()
    elif args.update_match:
        await update_match_data(args.update_match)
    elif args.update_league:
        await update_league_data(args.update_league, args.days_ahead)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
