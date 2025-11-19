import logging  # Added logging
import subprocess
import sys  # Added sys
import time
from datetime import datetime

import schedule

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# Define constants for file paths (consider moving to config if complex)
INTERMEDIATE_FILE = "temp_scrape_data.json"
HTML_RESULTS_FILE = "fetched_html_results.json"

def run_daily_pipeline():
    """Runs the full data pipeline: fetch, scrape, process, insights."""
    start_time = datetime.now()
    logger.info(f"Starting daily pipeline run at {start_time}...")
    success = True

    try:
        # === Step 1: Fetch Fixtures ===
        logger.info("Step 1: Fetching latest fixtures...")
        # Fetch fixtures for the next few days (adjust days as needed)
        days_ahead_fetch = 7
        fetch_cmd = [
            sys.executable, # Use sys.executable for portability
            "scripts/fetch_fixtures.py",
            "--days", str(days_ahead_fetch)
            # Add --leagues if needed: "--leagues", "PL", "BL1", ...
        ]
        result = subprocess.run(fetch_cmd, check=False, capture_output=True, text=True) # Use check=False to handle errors manually
        if result.returncode != 0:
            logger.error(f"fetch_fixtures.py failed! Return Code: {result.returncode}")
            logger.error(f"Stderr: {result.stderr}")
            logger.error(f"Stdout: {result.stdout}")
            success = False
        else:
            logger.info("fetch_fixtures.py completed successfully.")
            logger.debug(f"Stdout: {result.stdout}") # Log stdout on success for info

        # === Step 2: Run Bulk Scrape (Gather Mode) ===
        if success: # Only proceed if previous step succeeded
            logger.info("Step 2: Running bulk scrape (gather mode)...")
            # Scrape for matches scheduled today (or adjust days as needed)
            days_ahead_scrape = 1
            gather_cmd = [
                sys.executable,
                "scripts/run_bulk_scrape.py",
                "--mode", "gather",
                "--days", str(days_ahead_scrape),
                "--intermediate_file", INTERMEDIATE_FILE
                # Add --leagues if needed
            ]
            result = subprocess.run(gather_cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"run_bulk_scrape.py (gather) failed! Return Code: {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
                logger.error(f"Stdout: {result.stdout}")
                success = False
            else:
                logger.info("run_bulk_scrape.py (gather) completed successfully.")
                logger.debug(f"Stdout: {result.stdout}")

        # === Step 3: External HTML Fetching (Placeholder) ===
        if success:
            logger.info("Step 3: Triggering external HTML fetching (Manual/External Step Required)...")
            # --- !!! IMPORTANT !!! ---
            # This is where the external process needs to run.
            # It should:
            # 1. Read the 'html_requests' list from INTERMEDIATE_FILE ("temp_scrape_data.json").
            # 2. For each request, call the fircrawl-scraper MCP tool:
            #    <use_mcp_tool>
            #    <server_name>fircrawl-scraper</server_name>
            #    <tool_name>scrape_dynamic_website</tool_name>
            #    <arguments>{"url": "...", "usePlaywright": ...}</arguments>
            #    </use_mcp_tool>
            # 3. Collect all results (HTML content or error messages).
            # 4. Write the results to HTML_RESULTS_FILE ("fetched_html_results.json")
            #    in the format: {"url1": {"html": "...", "error": null}, "url2": {"html": null, "error": "..."}}
            #
            # This script assumes this external step completes successfully and creates the results file.
            # In a real system, you might add checks or waits here.
            logger.info(f"External process should now fetch HTML based on '{INTERMEDIATE_FILE}' and save results to '{HTML_RESULTS_FILE}'.")
            # For testing, you might manually create a dummy HTML_RESULTS_FILE here.

        # === Step 4: Run Bulk Scrape (Process Mode) ===
        if success:
            # Check if the expected HTML results file exists (basic check)
            if not os.path.exists(HTML_RESULTS_FILE):
                 logger.error(f"Cannot proceed to process mode: HTML results file '{HTML_RESULTS_FILE}' not found. Was the external fetching step successful?")
                 success = False
            else:
                logger.info("Step 4: Running bulk scrape (process mode)...")
                process_cmd = [
                    sys.executable,
                    "scripts/run_bulk_scrape.py",
                    "--mode", "process",
                    "--intermediate_file", INTERMEDIATE_FILE,
                    "--html_results_file", HTML_RESULTS_FILE
                ]
                result = subprocess.run(process_cmd, check=False, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"run_bulk_scrape.py (process) failed! Return Code: {result.returncode}")
                    logger.error(f"Stderr: {result.stderr}")
                    logger.error(f"Stdout: {result.stdout}")
                    success = False
                else:
                    logger.info("run_bulk_scrape.py (process) completed successfully.")
                    logger.debug(f"Stdout: {result.stdout}")

        # === Step 5: Generate Betting Insights ===
        if success:
            logger.info("Step 5: Generating betting insights...")
            # Generate insights for matches processed today
            insights_cmd = [
                sys.executable,
                "scripts/generate_betting_insights.py",
                "--date_from", datetime.now().strftime("%Y-%m-%d"),
                "--date_to", datetime.now().strftime("%Y-%m-%d")
                # Add --leagues or --teams if needed for the final report
            ]
            result = subprocess.run(insights_cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"generate_betting_insights.py failed! Return Code: {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
                logger.error(f"Stdout: {result.stdout}")
                success = False
            else:
                logger.info("generate_betting_insights.py completed successfully.")
                logger.debug(f"Stdout: {result.stdout}")

    except Exception as e:
        logger.exception(f"An unexpected error occurred in the daily pipeline: {e}")
        success = False

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Daily pipeline run finished at {end_time}. Duration: {duration}. Success: {success}")
    if not success:
        # Optionally send an alert about pipeline failure
        logger.error("PIPELINE FAILED.")
        # send_slack_message("Daily data pipeline failed. Check logs.") # Example alert

# Schedule to run every day at a specific time (e.g., 03:00 AM)
# Adjust time as needed based on when data is typically available and processing time
schedule.every().day.at("03:00").do(run_daily_pipeline)

if __name__ == "__main__":
    logger.info("Starting scheduled data pipeline automation...")
    # Optional: Run once immediately on start?
    # run_daily_pipeline()
    while True:
        schedule.run_pending()
        time.sleep(60)
