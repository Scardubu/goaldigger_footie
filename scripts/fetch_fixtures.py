#!/usr/bin/env python
"""
Fetches football fixtures from Football-Data.org API and saves to database.
Handles both upcoming and historical fixtures with proper date range splitting.
"""
import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Optional

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import json  # Import json for storing raw data

from dashboard.error_log import log_error  # Import log_error
# Import necessary modules and config
from data.api_clients.football_data_api import FootballDataAPI
from data.storage.database import DBManager  # Use DBManager again
from utils.config import Config, ConfigError  # Import centralized config
from utils.env_validate import validate_env  # Import environment validator

# --- Environment Validation ---
# Call this first to ensure required env vars are set
validate_env()

# Basic logging setup if utils.logging_config is not used
# TODO: Integrate with a central logging config if available
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# Load defaults from centralized config
try:
    # Load config once (it's cached)
    Config.load()
    DEFAULT_LEAGUE_CODES = Config.get('fetching.default_leagues', ["PL", "PD", "BL1", "SA", "FL1", "DED"])
    HISTORICAL_CHUNK_DAYS = Config.get('fetching.historical_chunk_days', 9)
    logger.info(f"Loaded fetching config: Leagues={DEFAULT_LEAGUE_CODES}, ChunkDays={HISTORICAL_CHUNK_DAYS}")
except ConfigError as config_err:
    logger.error(f"CRITICAL CONFIG ERROR: {config_err}. Exiting.")
    sys.exit(1)
except Exception as config_e:
    log_error("Unexpected error loading fetching config", config_e)
    logger.error(f"Unexpected error loading fetching config: {config_e}. Using hardcoded defaults.")
    DEFAULT_LEAGUE_CODES = ["PL", "PD", "BL1", "SA", "FL1", "DED"]
    HISTORICAL_CHUNK_DAYS = 9


def standardize_date_format(date_str):
    """Convert various date formats to YYYY-MM-DD"""
    formats = [
        '%Y-%m-%d',  # 2024-04-13
        '%d-%m-%y',  # 13-04-24
        '%y-%m-%d',  # 24-04-13
        '%d-%m-%Y',  # 13-04-2024
        '%d/%m/%Y',  # 13/04/2024
        '%d/%m/%y',  # 13/04/24
        '%Y/%m/%d'   # 2024/04/13
    ]

    for fmt in formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            continue

    # If we get here, none of the formats matched
    logger.error(f"Unrecognized date format provided: {date_str}")
    raise ValueError(f"Unrecognized date format: {date_str}. Please use YYYY-MM-DD or DD-MM-YY format.")

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments and returns them with dates as datetime.date objects."""
    parser = argparse.ArgumentParser(description="Fetch fixtures (upcoming or historical) from Football-Data.org and save to database.")
    parser.add_argument(
        "--leagues", nargs='+', default=DEFAULT_LEAGUE_CODES,
        help=f"Space-separated list of Football-Data.org competition codes (e.g., PL BL1 SA). Defaults to: {' '.join(DEFAULT_LEAGUE_CODES)}"
    )
    parser.add_argument(
        "--date_from", type=str, default=None,
        help="Start date for fetching fixtures (YYYY-MM-DD or DD-MM-YY). If not provided with --date_to, fetches upcoming."
    )
    parser.add_argument(
        "--date_to", type=str, default=None,
        help="End date for fetching fixtures (YYYY-MM-DD or DD-MM-YY). If not provided with --date_from, fetches upcoming."
    )
    parser.add_argument(
        "--days", type=int, default=10,
        help="Number of days ahead to fetch if --date_from/--date_to are not specified (Max 10 for Football-Data API)."
    )
    parser.add_argument(
        "--status", type=str, default=None,
        help="Status of matches to fetch (e.g., SCHEDULED, FINISHED, LIVE). Fetches based on date range if None."
    )
    parser.add_argument(
        "--retry", type=int, default=3,
        help="Number of retry attempts for API requests that fail."
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between API requests in seconds (Note: Client might handle internal delays)."
    )
    # Note: Retry argument is removed as client handles retries
    args = parser.parse_args()

    # Standardize and convert date formats if provided
    try:
        if args.date_from:
            args.date_from = datetime.strptime(standardize_date_format(args.date_from), '%Y-%m-%d').date()
        if args.date_to:
            args.date_to = datetime.strptime(standardize_date_format(args.date_to), '%Y-%m-%d').date()
    except ValueError as date_err:
             log_error("Invalid date argument", date_err)
             sys.exit(1) # Exit if date parsing fails

    return args

# Removed fetch_data_with_retry helper function as API client handles retries internally

def fetch_and_save_fixtures(
    league_codes: list,
    date_from: Optional[date], # Expecting date objects now
    date_to: Optional[date], # Expecting date objects now
    days_ahead: int,
    status: Optional[str]
    # Removed retry_count and delay_seconds as client handles this
) -> dict:
    """
    Fetches fixtures from Football-Data.org API for the specified leagues/dates/status
    and saves them to the DuckDB database. Returns a summary dict.
    """
    summary = {"matches_fetched": 0, "matches_saved": 0, "errors": []}
    saved_count = 0 # Track saved matches separately
    # db = None # No need to initialize for finally block with DBManager's context manager
    try:
        # Only initialize after argument validation
        api_client = FootballDataAPI()  # Reads token from .env automatically
        db = DBManager() # Initialize DBManager

        # Determine date range and status for API call
        fetch_status = status
        if date_from and date_to:
            # Use specified date range
            if fetch_status is None:
                fetch_status = 'FINISHED'
            logger.info(f"Fetching fixtures for leagues {league_codes} from {date_from} to {date_to} with status: {fetch_status or 'Any'}")
        else:
            today = date.today()
            date_from = today
            date_to = today + timedelta(days=days_ahead)
            if fetch_status is None:
                fetch_status = 'SCHEDULED'
            logger.info(f"Fetching upcoming fixtures for leagues {league_codes} from {date_from} to {date_to} with status: {fetch_status}")

        all_matches = []
        # Determine if fetching historical data (requires chunking)
        is_historical = date_from and date_to and (date_from < date.today())

        if is_historical:
            # --- Historical Fetching (Chunking) ---
            start_date = date_from
            end_date = date_to
            current_start = start_date
            logger.info(f"Fetching historical data in chunks (Chunk size: {HISTORICAL_CHUNK_DAYS} days)...")
            while current_start <= end_date:
                # Use HISTORICAL_CHUNK_DAYS from config
                current_end = min(current_start + timedelta(days=HISTORICAL_CHUNK_DAYS), end_date)
                chunk_date_from_str = current_start.strftime('%Y-%m-%d')
                chunk_date_to_str = current_end.strftime('%Y-%m-%d')
                logger.info(f"Fetching chunk: {chunk_date_from_str} to {chunk_date_to_str}")

                # Call API client directly (it handles retries/delay)
                chunk_matches = api_client.get_matches_for_competitions(
                    competition_codes=league_codes,
                    date_from=chunk_date_from_str,
                    date_to=chunk_date_to_str,
                    status=fetch_status
                )
                # API client returns [] on failure after retries, or None if error was immediate (e.g., 404)
                if chunk_matches is not None:
                    all_matches.extend(chunk_matches)
                    logger.info(f"Fetched {len(chunk_matches)} matches for chunk.")
                else:
                    # Log error if client returned None (indicates non-retryable error or config issue)
                    logger.error(f"API client failed to return data for chunk {chunk_date_from_str} to {chunk_date_to_str}. Check API client logs.")
                    summary["errors"].append(f"API fetch failed for chunk {chunk_date_from_str}-{chunk_date_to_str}")
                    # Optional: break or continue depending on desired behavior on chunk failure

                # API client handles its own rate limiting delay, no extra sleep needed here
                current_start = current_end + timedelta(days=1)
            logger.info(f"Finished fetching historical data chunks. Total matches fetched: {len(all_matches)}")
        else: # Correctly indented else block (using 4 spaces)
            # --- Upcoming/Single Range Fetching ---
            logger.info(f"Fetching data for single range: {date_from.strftime('%Y-%m-%d')} to {date_to.strftime('%Y-%m-%d')}")
            all_matches = api_client.get_matches_for_competitions(
                competition_codes=league_codes,
                date_from=date_from.strftime('%Y-%m-%d'),
                date_to=date_to.strftime('%Y-%m-%d'),
                status=fetch_status
            )
            if all_matches is None: # Handle potential None return from client on error
                 logger.error("API client failed to return data for the specified range.")
                 summary["errors"].append(f"API fetch failed for range {date_from}-{date_to}")
                 all_matches = [] # Ensure all_matches is an empty list

        summary["matches_fetched"] = len(all_matches)
        if not all_matches:
            logger.warning("No matches fetched from the API for the specified criteria.")
            # No need to return early, just log and proceed (summary already reflects 0 fetched)
        else:
            logger.info(f"Fetched a total of {len(all_matches)} matches from the API.")
            # Save fetched matches to DB using DBManager.execute
            logger.info(f"Attempting to save {len(all_matches)} matches to the database...")
            for match in all_matches:
                try:
                    match_id = str(match.get("id", ""))
                    if not match_id:
                        logger.warning(f"Skipping match with missing ID: {match}")
                        continue

                    home_team = match.get("homeTeam", {})
                    away_team = match.get("awayTeam", {})
                    competition = match.get("competition", {})
                    score = match.get("score", {}).get("fullTime", {})

                    home_team_id = str(home_team.get("id")) if home_team else None
                    away_team_id = str(away_team.get("id")) if away_team else None
                    home_team_name = home_team.get("name") if home_team else None
                    away_team_name = away_team.get("name") if away_team else None
                    competition_id = str(competition.get("id")) if competition else None
                    competition_name = competition.get("name") if competition else None

                    try:
                        match_date_dt = (
                            datetime.fromisoformat(
                                match.get("utcDate", "").replace("Z", "+00:00")
                            )
                            if match.get("utcDate")
                            else None
                        )
                        # DuckDBStorage expects ISO format string or compatible type
                        match_date_str = match_date_dt.isoformat() if match_date_dt else None
                    except ValueError:
                        logger.warning(
                            f"Could not parse date {match.get('utcDate')} for match {match_id}. Setting to NULL."
                        )
                        match_date_str = None

                    # Check if 'raw_data' and 'updated_at' columns exist in the schema defined in DBManager
                    # Assuming they might not exist based on the schema I added earlier
                    # Adjust the query and params accordingly
                    query = """
                        INSERT OR REPLACE INTO matches (
                            id, home_team_id, away_team_id, home_team, away_team,
                            competition_id, competition, match_date, status,
                            home_score, away_score
                            -- Removed raw_data, updated_at assuming they aren't in the SQLite schema
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    params = (
                        match_id,
                        home_team_id,
                        away_team_id,
                        home_team_name,
                        away_team_name,
                        competition_id,
                        competition_name,
                        match_date_str, # Use ISO string
                        match.get("status"),
                        score.get("home"),
                        score.get("away")
                        # Removed json.dumps(match) and datetime.now().isoformat()
                    )
                    # Use the execute method from DBManager
                    db.execute(query, params)
                    saved_count += 1
                except Exception as save_err:
                    log_error(f"Error saving individual match {match.get('id', 'N/A')} to database", save_err)
                    summary["errors"].append(f"DB save error for match {match.get('id', 'N/A')}: {save_err}") # Keep existing logic
                    # Continue trying to save other matches

            summary["matches_saved"] = saved_count
            logger.info(f"Successfully saved {saved_count} matches.")

        logger.info("Fixture fetching and saving process completed.")
    except ValueError as val_err: # Catch specific errors like API key missing
         log_error("Configuration or Value Error during fixture fetch", val_err) # Keep existing log_error call
         summary["errors"].append(str(val_err))
    except Exception as e:
        log_error("An unexpected error occurred during the fetch/save process", e)
        summary["errors"].append(f"Unexpected error: {e}")
    # No finally block needed as DBManager uses 'with self.connect()'

    return summary

if __name__ == "__main__":
    args = parse_arguments()
    # Pass only necessary args, retry/delay handled by client
    result = fetch_and_save_fixtures(
        league_codes=args.leagues,
        date_from=args.date_from,
        date_to=args.date_to,
        days_ahead=args.days,
        status=args.status
    )
    logger.info(f"Script finished. Summary: {result}")
    # Exit with error code if errors occurred
    if result["errors"]:
         logger.error("Process completed with errors.")
         sys.exit(1)
    else:
         logger.info("Process completed successfully.")
         sys.exit(0)
