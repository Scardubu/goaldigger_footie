"""
Data backfill utility for identifying and recovering missing scraped data.
Works with the existing scraping infrastructure to target only matches with missing data.
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

# Add project root to path to allow for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.storage.database import DBManager
from utils.config import Config
from utils.logging_config import setup_logging
from dashboard.error_log import log_error
from scripts.run_bulk_scrape import generate_mcp_request, gather_mcp_requests_async, process_mcp_results_and_save

# Configure logging
logger = logging.getLogger(__name__)

def identify_missing_scraped_data(db_manager: DBManager, days_lookback: int = 30) -> List[str]:
    """
    Identify matches with missing scraped data entries.
    
    Args:
        db_manager: Database manager instance
        days_lookback: Number of days to look back for matches
        
    Returns:
        List of match IDs that are missing scraped data
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_lookback)
        
        logger.info(f"Identifying matches without scraped data from {start_date.date()} to {end_date.date()}")
        
        # Get all matches in the date range
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Query matches in the date range
            cursor.execute(
                """
                SELECT id FROM matches 
                WHERE match_date BETWEEN ? AND ?
                """,
                (start_date.isoformat(), end_date.isoformat())
            )
            all_matches = set(row[0] for row in cursor.fetchall())
            
            # Query matches with scraped data
            cursor.execute(
                """
                SELECT match_id FROM scraped_data
                """
            )
            scraped_matches = set(row[0] for row in cursor.fetchall())
            
            # Find matches without scraped data
            missing_matches = all_matches - scraped_matches
            
            logger.info(f"Found {len(missing_matches)} matches without scraped data out of {len(all_matches)} total matches")
            
            return list(missing_matches)
    
    except Exception as e:
        log_error(f"Error identifying matches with missing scraped data", e)
        return []

def get_match_details(db_manager: DBManager, match_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get match details for the specified match IDs.
    
    Args:
        db_manager: Database manager instance
        match_ids: List of match IDs to retrieve details for
        
    Returns:
        List of match detail dictionaries
    """
    if not match_ids:
        return []
    
    try:
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()

            # Prepare placeholders for SQL query
            placeholders = ', '.join(['?'] * len(match_ids))

            # Query match details with JOINs to get team names
            cursor.execute(
                f"""
                SELECT m.id, m.home_team_id, m.away_team_id, ht.name as home_team_name, at.name as away_team_name,
                       m.competition_id, m.competition, m.match_date, m.status
                FROM matches m
                LEFT JOIN teams ht ON m.home_team_id = ht.id
                LEFT JOIN teams at ON m.away_team_id = at.id
                WHERE m.id IN ({placeholders})
                """,
                match_ids
            )

            columns = [col[0] for col in cursor.description]
            matches = []

            for row in cursor.fetchall():
                match_dict = dict(zip(columns, row))
                # Ensure match_date is properly formatted
                if 'match_date' in match_dict and match_dict['match_date']:
                    try:
                        dt = datetime.fromisoformat(match_dict['match_date'])
                        match_dict['match_date'] = dt.isoformat()
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid date format for match {match_dict['id']}: {match_dict['match_date']}")
                matches.append(match_dict)

            logger.info(f"Retrieved details for {len(matches)} matches")
            return matches

    except Exception as e:
        log_error(f"Error retrieving match details", e)
        return []

async def backfill_scraped_data(match_ids: List[str], request_file: str = "backfill_mcp_requests.json", results_file: str = "backfill_mcp_results.json") -> bool:
    """
    Generate MCP requests for matches with missing scraped data and process the results.
    
    Args:
        match_ids: List of match IDs to backfill
        request_file: File to save MCP requests to
        results_file: File to save MCP results to
        
    Returns:
        True if the backfill was successful, False otherwise
    """
    try:
        # Initialize database manager
        db_manager = DBManager()
        
        # Get match details
        match_details = get_match_details(db_manager, match_ids)
        
        if not match_details:
            logger.warning("No match details retrieved, cannot proceed with backfill")
            return False
        
        logger.info(f"Generating MCP requests for {len(match_details)} matches")
        
        # Generate MCP requests for each match
        mcp_requests = []
        for match in match_details:
            # Create a request for each match
            mcp_request = generate_mcp_request(match)
            if mcp_request:
                mcp_requests.append(mcp_request)
        
        if not mcp_requests:
            logger.warning("No valid MCP requests generated, cannot proceed with backfill")
            return False
        
        # Save MCP requests to file
        with open(request_file, 'w') as f:
            json.dump(mcp_requests, f, indent=2)
        
        logger.info(f"Saved {len(mcp_requests)} MCP requests to {request_file}")
        logger.info("Please execute these requests using the MCP tool and save the results to {results_file}")
        
        # Here we would normally wait for the MCP results, but since that's a manual step,
        # we'll simply return and let the user know what to do next
        
        return True
    
    except Exception as e:
        log_error(f"Error during backfill process", e)
        return False

def check_and_process_results(results_file: str = "backfill_mcp_results.json") -> bool:
    """
    Check if MCP results file exists and process it.
    
    Args:
        results_file: File containing MCP results
        
    Returns:
        True if results were processed successfully, False otherwise
    """
    try:
        if not os.path.exists(results_file):
            logger.warning(f"MCP results file {results_file} not found. Cannot process.")
            return False
        
        logger.info(f"Processing MCP results from {results_file}")
        
        # Process MCP results and save to database
        process_mcp_results_and_save(results_file)
        
        return True
    
    except Exception as e:
        log_error(f"Error processing MCP results", e)
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Data backfill utility for missing scraped data")
    parser.add_argument("--mode", choices=["identify", "backfill", "process", "auto"], default="identify",
                        help="Mode of operation: identify (find missing data), backfill (generate MCP requests), process (process MCP results), auto (all steps)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to look back for matches")
    parser.add_argument("--request-file", default="backfill_mcp_requests.json", help="File to save MCP requests to")
    parser.add_argument("--results-file", default="backfill_mcp_results.json", help="File to save MCP results to")
    args = parser.parse_args()
    
    # Load configuration
    Config.load()
    
    # Initialize database manager
    db_manager = DBManager()
    
    # Execute based on mode
    if args.mode == "identify":
        missing_matches = identify_missing_scraped_data(db_manager, args.days)
        print(f"Found {len(missing_matches)} matches with missing scraped data")
        if missing_matches:
            print("Match IDs:")
            for match_id in missing_matches:
                print(f"  {match_id}")
    
    elif args.mode == "backfill":
        missing_matches = identify_missing_scraped_data(db_manager, args.days)
        if missing_matches:
            import asyncio
            asyncio.run(backfill_scraped_data(missing_matches, args.request_file, args.results_file))
            print(f"MCP requests generated and saved to {args.request_file}")
            print(f"Please execute these requests using the MCP tool and save the results to {args.results_file}")
            print(f"Then run this script with --mode process to process the results")
        else:
            print("No matches with missing scraped data found. Nothing to backfill.")
    
    elif args.mode == "process":
        success = check_and_process_results(args.results_file)
        if success:
            print(f"MCP results processed successfully")
        else:
            print(f"Error processing MCP results, please check the logs")
    
    elif args.mode == "auto":
        missing_matches = identify_missing_scraped_data(db_manager, args.days)
        if missing_matches:
            print(f"Found {len(missing_matches)} matches with missing scraped data")
            import asyncio
            asyncio.run(backfill_scraped_data(missing_matches, args.request_file, args.results_file))
            print(f"MCP requests generated and saved to {args.request_file}")
            print(f"Please execute these requests using the MCP tool and save the results to {args.results_file}")
            print(f"Then run this script with --mode process to process the results")
        else:
            print("No matches with missing scraped data found. Nothing to backfill.")

if __name__ == "__main__":
    main()
