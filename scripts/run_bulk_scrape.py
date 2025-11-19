import argparse
import asyncio
import json
import logging
import os
import platform
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Removed BeautifulSoup import as parsing is now internal to MCP tool's Python script execution
# from bs4 import BeautifulSoup, Comment
# Removed urllib imports as URL finding is internal to MCP tool's Python script execution
# from urllib.parse import quote_plus, urljoin

# Fix for Playwright on Windows: Set asyncio event loop policy
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from data.storage.database import DBManager
from scripts.core.ai_validator import AIDataValidator
# Import functions directly - consolidation might still be needed if MCP tool returns structured data
from scripts.core.enhanced_scraper import \
    _consolidate_results as consolidate_scraper_results
# Removed parse_html_content import as MCP tool handles parsing internally now
# from scripts.core.enhanced_scraper import parse_html_content
from scripts.core.scrapers.proxy_manager import ProxyManager
from utils.config import Config
from utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Load network config
NETWORK_CONFIG = Config.get('network', {})
MAX_CONCURRENT_GATHER = NETWORK_CONFIG.get('max_concurrent_requests', 10)

# --- MCP Constants ---
# Note: We now primarily use the new tool
MCP_SERVER_NAME = "fircrawl-scraper"
# MCP_TOOL_NAME_SCRAPE = "scrape_dynamic_website" # Old tool, might remove later
MCP_TOOL_NAME_FIND_AND_SCRAPE = "find_and_scrape_match_data" # New tool

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run bulk data scraping request generation and processing.")
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days ahead to gather requests for."
    )
    parser.add_argument(
        "--leagues", nargs='+', default=None, help="Specific league IDs/names to gather requests for."
    )
    parser.add_argument(
        "--delay", type=float, default=0.05, help="Small delay added within async gather worker." # Reduced default delay
    )
    # Updated mode descriptions
    parser.add_argument("--mode", choices=['gather', 'process', 'auto'], default='auto', help="Execution mode: 'gather' generates MCP tool requests, 'process' processes results from executed requests, 'auto' runs gather then waits for processing.")
    # Renamed intermediate file
    parser.add_argument("--request_file", default="temp_mcp_requests.json", help="JSON file to store/load MCP tool requests (used by gather/process).")
    # Renamed results file
    parser.add_argument("--results_file", default="mcp_results.json", help="JSON file containing results from executed MCP requests (used by process mode).")
    return parser.parse_args()


# --- Removed URL Finding Helper Functions ---
# (_find_transfermarkt_url, _find_understat_url, _find_fbref_url are removed)
# The logic is now encapsulated within the Python script called by the find_and_scrape_match_data MCP tool


# --- Simplified Fetch Multi Source Logic ---
async def generate_mcp_requests_for_match(
    match_id: str,
    match_date_str: Optional[str],
    home_team: Optional[str],
    away_team: Optional[str],
    league_id: Optional[str],
    season: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Generates a list of MCP tool request arguments for desired sources for a single match.
    """
    logger.debug(f"Generating MCP requests for match {match_id}")
    mcp_requests = []
    target_sources = ["Transfermarkt", "Understat", "FBref"] # Define desired sources

    # Basic validation of necessary inputs for generating requests
    if not all([match_date_str, home_team, away_team, season, league_id]):
         logger.warning(f"Cannot generate MCP requests for match {match_id} due to missing team/date/season/league info.")
         return [] # Return empty list

    for source_site in target_sources:
        request_args = {
            "source_site": source_site,
            "match_id": match_id,
            "match_date": match_date_str,
            "home_team": home_team,
            "away_team": away_team,
            "league_id": league_id,
            "season": season,
        }
        # Structure for the MCP tool call
        mcp_request = {
            "tool_name": MCP_TOOL_NAME_FIND_AND_SCRAPE,
            "arguments": request_args,
            # Add context if needed, e.g., original match_id
            "context": {"original_match_id": match_id, "source_site": source_site}
        }
        mcp_requests.append(mcp_request)

    logger.info(f"Generated {len(mcp_requests)} MCP request arguments for match {match_id} for sources: {target_sources}")
    return mcp_requests


# --- Async Worker for Gather Mode (Modified) ---
async def _gather_worker(
    semaphore: asyncio.Semaphore,
    match_info: Tuple[str, datetime, Optional[str], Optional[str], Optional[str], Optional[str]],
    delay_seconds: float
) -> List[Dict[str, Any]]: # Return type is now List of MCP request dicts
    """
    Worker function to generate MCP tool request arguments for a single match.
    """
    match_id, match_date, home_team, away_team, league_id, season = match_info
    match_date_str = match_date.strftime('%Y-%m-%d') if pd.notna(match_date) else None

    async with semaphore:
        logger.info(f"Gather worker starting for match ID={match_id}, Date={match_date_str}, League={league_id}, Season={season}, Teams: {home_team} vs {away_team}")
        try:
            # Call the function to generate MCP request arguments
            mcp_requests = await generate_mcp_requests_for_match(
                match_id=match_id,
                match_date_str=match_date_str,
                home_team=home_team,
                away_team=away_team,
                league_id=league_id,
                season=season
            )

            # Optional small delay
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

            logger.info(f"Gather worker finished for match ID={match_id}, generated {len(mcp_requests)} requests.")
            return mcp_requests # Return the list of request arguments

        except Exception as e:
             logger.error(f"Error in gather worker for match {match_id}: {e}", exc_info=True)
             # Return an empty list or some error indicator if needed
             return []


# --- Gather Function (Modified) ---
async def gather_mcp_requests_async(
    days_ahead: int,
    target_leagues: Optional[list],
    delay_seconds: float,
    request_file: str # Changed from intermediate_file, now mandatory for saving
) -> bool: # Return True on success, False on failure
    """
    Asynchronously fetches scheduled matches and generates MCP tool request arguments,
    saving them to the specified file.
    """
    storage = None
    all_mcp_requests = [] # Initialize list to hold all requests

    try:
        storage = DBManager()

        # 1. Get scheduled matches
        today = date.today()
        date_from = datetime.combine(today, datetime.min.time())
        date_to = datetime.combine(today + timedelta(days=days_ahead), datetime.max.time())

        logger.info(f"Fetching scheduled matches from {date_from.date()} to {date_to.date()} for leagues: {target_leagues or 'All'}")
        matches_df = storage.get_matches_df(
            date_from=date_from,
            date_to=date_to,
            status="TIMED",
        )

        if matches_df.empty:
            logger.info("No scheduled matches found. No MCP requests generated.")
            # Save an empty list to the request file to indicate completion
            with open(request_file, 'w') as f:
                json.dump([], f)
            logger.info(f"Saved empty request list to {request_file}")
            return True # Successful run, just no matches

        logger.info(f"Found {len(matches_df)} matches to generate MCP requests for.")

        # 2. Prepare and run tasks
        tasks = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_GATHER)
        logger.info(f"Initializing gather tasks with concurrency limit: {MAX_CONCURRENT_GATHER}")

        for _, row in matches_df.iterrows():
            match_id = str(row['id'])
            match_date = row['match_date']
            home_team = row.get('home_team')
            away_team = row.get('away_team')
            league_id = str(row.get('competition_id')) if pd.notna(row.get('competition_id')) else None
            season = None
            if pd.notna(match_date):
                season = match_date.year if match_date.month >= 7 else match_date.year - 1
                season = str(season)

            match_info = (match_id, match_date, home_team, away_team, league_id, season)
            task = asyncio.create_task(_gather_worker(semaphore, match_info, delay_seconds))
            tasks.append(task)

        logger.info(f"Running {len(tasks)} gather tasks concurrently...")
        results_list_of_lists = await asyncio.gather(*tasks) # Returns list of lists
        logger.info("All gather tasks completed.")

        # 3. Flatten results and save
        for request_list in results_list_of_lists:
            if isinstance(request_list, list): # Ensure worker returned a list
                all_mcp_requests.extend(request_list)

        logger.info(f"Total MCP tool requests generated: {len(all_mcp_requests)}")

        try:
            with open(request_file, 'w') as f:
                json.dump(all_mcp_requests, f, indent=2, default=str)
            logger.info(f"Successfully saved {len(all_mcp_requests)} MCP requests to {request_file}")
            return True # Indicate success
        except Exception as e:
            logger.error(f"Failed to save MCP requests to {request_file}: {e}", exc_info=True)
            return False # Indicate failure

    except Exception as gather_err:
        logger.critical(f"Unhandled exception during gather phase: {gather_err}", exc_info=True)
        return False # Indicate failure
    finally:
         if storage:
              storage.close()


# --- Removed Async MCP Scraper Functions ---
# (_invoke_mcp_tool_via_cline, _mcp_scrape_worker, run_mcp_scrapes_async are removed)


# --- Processing Function (Adapted Placeholder) ---
def process_mcp_results_and_save(
    results_file: str # Input is now the file with results from executed MCP calls
):
    """
    Processes results from executed MCP tool calls, consolidates,
    validates, and saves valid results to the database.
    """
    storage = None
    validator = None
    flag_dir = os.path.join(project_root, "data", "status")
    flag_file = os.path.join(flag_dir, "data_ready.flag")
    all_parsed_data = [] # List to hold data parsed from MCP results

    try:
        # --- Load MCP Results ---
        logger.info(f"Loading MCP results from {results_file}...")
        try:
            with open(results_file, 'r') as f:
                # Expecting a list of result objects, where each object corresponds to one MCP call
                # Each result object should contain the original context and the tool's output content
                mcp_results_list = json.load(f)
        except FileNotFoundError:
            logger.error(f"MCP results file {results_file} not found. Cannot process.")
            return # Or raise error
        except json.JSONDecodeError:
             logger.error(f"Error decoding JSON from MCP results file {results_file}.")
             return # Or raise error

        logger.info(f"Loaded {len(mcp_results_list)} results from MCP execution.")
        if not mcp_results_list:
             logger.warning("MCP results file is empty. Nothing to process.")
             # Create flag file indicating no new data?
             # ... (add logic if needed) ...
             return

        # --- Extract Parsed Data from MCP Results ---
        logger.info("Extracting parsed data from MCP results...")
        extraction_errors = 0
        for mcp_result in mcp_results_list:
            # Assuming the result structure includes the original context and the content part
            context = mcp_result.get("context", {})
            content_parts = mcp_result.get("content", [])
            is_error = mcp_result.get("isError", False)
            original_match_id = context.get("original_match_id", "unknown")
            source_site = context.get("source_site", "unknown")

            if is_error or not content_parts:
                error_message = "MCP tool execution failed"
                if content_parts and isinstance(content_parts, list) and len(content_parts) > 0 and content_parts[0].get('type') == 'text':
                    error_message = content_parts[0]['text']
                logger.error(f"MCP tool failed for Match {original_match_id}, Source {source_site}. Error: {error_message}")
                extraction_errors += 1
                continue # Skip this result

            # Assuming the actual parsed data is JSON stringified in the first text part
            try:
                if content_parts[0].get('type') == 'text':
                    parsed_data_from_mcp = json.loads(content_parts[0]['text'])
                    # Add context back if needed for consolidation/validation
                    parsed_data_from_mcp['original_match_id'] = original_match_id
                    parsed_data_from_mcp['source_site'] = source_site # Ensure source is present
                    all_parsed_data.append(parsed_data_from_mcp)
                else:
                    logger.warning(f"Unexpected content type in MCP result for Match {original_match_id}, Source {source_site}. Expected 'text'. Content: {content_parts}")
                    extraction_errors += 1
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON content from MCP result for Match {original_match_id}, Source {source_site}. Content: {content_parts[0].get('text', 'N/A')}")
                extraction_errors += 1
            except Exception as e:
                 logger.error(f"Unexpected error processing MCP result content for Match {original_match_id}, Source {source_site}: {e}", exc_info=True)
                 extraction_errors += 1

        logger.info(f"Successfully extracted data for {len(all_parsed_data)} items. Extraction errors: {extraction_errors}")

        if not all_parsed_data:
             logger.warning("No valid data extracted from MCP results. Aborting processing.")
             return

        # --- Ensure status directory exists ---
        os.makedirs(flag_dir, exist_ok=True)
        if os.path.exists(flag_file):
            try: os.remove(flag_file); logger.info(f"Removed existing flag: {flag_file}")
            except OSError as e: logger.warning(f"Could not remove flag {flag_file}: {e}")

        # --- Initialize Validator & Storage ---
        logger.info("Initializing AIDataValidator and DBManager...")
        validator = AIDataValidator()
        storage = DBManager()
        logger.info("AIDataValidator and DBManager initialized.")

        # --- Consolidate Data (Group by original match_id) ---
        # This step might need adjustment depending on how data should be structured for validation/saving
        # For now, let's assume validation works on the list of individual source results
        logger.info("Consolidating results (currently placeholder - using extracted data directly)...")
        # TODO: Implement actual consolidation if needed (e.g., merging data from different sources for the same match)
        consolidated_data_for_validation = all_parsed_data # Using extracted data directly for now

        # --- Bulk Validation and Saving ---
        saved_count = 0
        failed_save_count = 0
        if not consolidated_data_for_validation:
            logger.warning("No consolidated data available for validation and saving.")
        else:
            logger.info(f"Attempting to validate {len(consolidated_data_for_validation)} data items...")
            try:
                # Convert list of dicts to DataFrame
                try:
                    # Use json_normalize, assuming the structure from find_and_scrape is relatively flat
                    # Need to know the exact structure returned by find_and_scrape JSON output
                    # Let's assume it includes keys like 'source', 'url', 'status', 'extracted_data', 'original_match_id' etc.
                    bulk_df = pd.json_normalize(consolidated_data_for_validation, sep='_')
                    logger.info(f"Created DataFrame for validation with shape: {bulk_df.shape}")
                    logger.debug(f"DataFrame columns: {bulk_df.columns.tolist()}")
                except Exception as norm_err:
                     logger.error(f"Failed to create/normalize DataFrame from consolidated data: {norm_err}", exc_info=True)
                     raise

                # Perform validation
                validated_df, report = validator.validate_dataset(bulk_df) # Pass the DataFrame
                logger.info(f"Validation Report: {json.dumps(report, indent=2, default=str)}")

                if report.get('validation_passed', False):
                    logger.info("Bulk validation passed. Proceeding with bulk save.")
                    try:
                        # Ensure the necessary ID column exists (e.g., 'original_match_id')
                        id_col = 'original_match_id' # Assuming this is the key
                        if id_col not in validated_df.columns:
                             logger.error(f"Critical error: '{id_col}' column missing from validated DataFrame. Cannot perform bulk save.")
                             failed_save_count = len(validated_df)
                        elif validated_df.empty:
                             logger.info("Validated DataFrame is empty. Nothing to bulk save.")
                        else:
                             # Call bulk save method (assuming it exists and handles this structure)
                             # Might need to adapt save_bulk_match_data or create a new method
                             # depending on how the validated_df columns map to the DB schema.
                             # This likely requires significant changes in DBManager.
                             logger.warning("Bulk saving logic needs implementation/adaptation in DBManager for the new data structure.")
                             # saved_count = storage.save_bulk_match_data(validated_df) # Placeholder call
                             saved_count = 0 # Simulate no save until DBManager is updated
                             failed_save_count = len(validated_df) # Assume failure for now
                             logger.info(f"Successfully bulk saved {saved_count} validated matches (SIMULATED).")

                    except Exception as bulk_save_err:
                         logger.error(f"Bulk save failed: {bulk_save_err}", exc_info=True)
                         failed_save_count = len(validated_df)
                else:
                    logger.error("Bulk validation failed! No data will be saved.")
                    failed_save_count = len(consolidated_data_for_validation)

            except Exception as val_err:
                logger.error(f"Error during bulk validation or saving process: {val_err}", exc_info=True)
                failed_save_count = len(consolidated_data_for_validation)

        logger.info(f"Processing finished. Items Validated: {len(consolidated_data_for_validation)}, Save Successful: {saved_count}, Save Failed: {failed_save_count}")

        # --- Create flag file ---
        validation_passed = report.get('validation_passed', False) if 'report' in locals() and report else False
        if validation_passed and saved_count > 0:
             try:
                  with open(flag_file, 'w') as f: f.write(f"Data processed and saved at: {datetime.now().isoformat()}")
                  logger.info(f"Created data readiness flag: {flag_file}")
             except Exception as flag_err: logger.error(f"Failed to create flag {flag_file}: {flag_err}", exc_info=True)
        elif validation_passed and saved_count == 0 and failed_save_count == 0 and len(consolidated_data_for_validation) == 0:
             try: # Handle case where no data needed processing
                  with open(flag_file, 'w') as f: f.write(f"No new data to process, run completed at: {datetime.now().isoformat()}")
                  logger.info(f"Created data readiness flag (no new data): {flag_file}")
             except Exception as flag_err: logger.error(f"Failed to create flag {flag_file}: {flag_err}", exc_info=True)
        else:
             logger.warning(f"Skipping flag file creation. Validation passed: {validation_passed}, Saved count: {saved_count}")

    finally:
         if storage:
              storage.close()


# --- Full Pipeline Orchestrator (Modified) ---
async def run_full_pipeline_async(days: int, leagues: Optional[list], delay: float, request_file: str, results_file: str):
    """
    Runs the gather phase, waits for MCP execution (manual step by Cline), then runs processing.
    """
    logger.info("Starting full data pipeline (auto mode)...")
    start_pipeline_time = time.time()

    # 1. Gather MCP Tool Requests
    logger.info("--- Stage 1: Gather MCP Tool Requests ---")
    gather_success = await gather_mcp_requests_async(days, leagues, delay, request_file)

    if not gather_success:
        logger.error("Gather phase failed. Aborting pipeline.")
        return

    # --- PAUSE ---
    # At this point, the `request_file` (e.g., temp_mcp_requests.json) contains the list of MCP tool calls.
    # Cline (the controlling agent) needs to:
    # 1. Read `request_file`.
    # 2. Iterate through the requests.
    # 3. Execute `use_mcp_tool` for each request using the specified tool_name and arguments.
    # 4. Collect the results (including errors) from each `use_mcp_tool` call.
    # 5. Save the collected results into `results_file` (e.g., mcp_results.json).
    # 6. Trigger the continuation of this script (or the 'process' mode manually).
    logger.info(f"--- Stage 2: Manual MCP Execution Required ---")
    logger.info(f"MCP requests generated and saved to: {request_file}")
    logger.info(f"Please execute these requests using the MCP tool and save the results to: {results_file}")
    logger.info("Once results are saved, run this script again in 'process' mode or trigger the processing stage.")
    # The script effectively stops here in 'auto' mode until manually restarted for processing.
    # We could add a loop/wait here, but manual trigger is likely safer.

    # --- Stage 3: Process Results (Executed in a separate run or after manual intervention) ---
    # This part would typically be run by calling: python scripts/run_bulk_scrape.py --mode process
    # logger.info("--- Stage 3: Process MCP Results ---")
    # try:
    #     process_mcp_results_and_save(results_file)
    #     logger.info("Processing and saving stage completed.")
    # except Exception as process_err:
    #     logger.error(f"Error during processing and saving stage: {process_err}", exc_info=True)

    end_pipeline_time = time.time()
    # Log duration only for the gather phase in this modified auto mode
    logger.info(f"Gather phase finished in {end_pipeline_time - start_pipeline_time:.2f} seconds. Waiting for MCP execution and processing.")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        Config.load()
        logger.info("Configuration loaded.")
        # env_validate() # Assuming validation happens elsewhere or is not needed here
    except Exception as e:
        logger.critical(f"CRITICAL ERROR during configuration loading: {e}", exc_info=True)
        sys.exit(1)

    args = parse_arguments()

    if args.mode == 'gather':
        logger.info("Running in 'gather' mode (async)...")
        asyncio.run(gather_mcp_requests_async(args.days, args.leagues, args.delay, args.request_file))
        logger.info("'gather' mode finished.")
    elif args.mode == 'process':
        logger.info("Running in 'process' mode (sync)...")
        process_mcp_results_and_save(args.results_file)
        logger.info("'process' mode finished.")
    elif args.mode == 'auto':
         logger.info("Running in 'auto' mode (gather phase only, async)...")
         # Run only the gather phase, then instruct user
         asyncio.run(gather_mcp_requests_async(args.days, args.leagues, args.delay, args.request_file))
         # Instructions are logged within run_full_pipeline_async or gather_mcp_requests_async
         logger.info("'auto' mode gather phase finished. Manual MCP execution needed.")
    else:
        logger.error(f"Invalid mode specified: {args.mode}")
        sys.exit(1)
