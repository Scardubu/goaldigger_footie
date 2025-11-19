#!/usr/bin/env python
"""
Football-Data API Diagnostic Test Script

This script tests the connection to the Football-Data.org API and verifies that
the API key is working properly. It also tests fetching match data to ensure
the fixes for the 400 Bad Request errors are working.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("football_data_api_test")

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from scripts.scrapers.football_data_scraper import FootballDataScraper
from utils.http_client_async import HttpClientAsync


async def test_api_key():
    """Test that the API key is properly configured and working."""
    logger.info("Testing API key configuration...")
    
    # Create a scraper instance
    scraper = FootballDataScraper()
    
    # Check if API key is configured
    if not scraper.api_key:
        logger.error("❌ No API key configured for Football-Data.org")
        logger.info("Please set the FOOTBALL_DATA_API_KEY environment variable or add it to config.yaml")
        return False
    
    # Check if API key is valid
    masked_key = scraper.api_key[:4] + "..." + scraper.api_key[-4:] if len(scraper.api_key) > 8 else "***"
    logger.info(f"API key found: {masked_key}")
    
    # Create an HTTP client for testing
    http_client = HttpClientAsync("https://api.football-data.org/v4")
    
    # Test API key with a simple request
    try:
        response = await http_client.get(
            "/competitions",
            headers={"X-Auth-Token": scraper.api_key, "Accept": "application/json"}
        )
        
        if response.status == 200:
            data = await response.json()
            logger.info(f"✅ API key is valid. Found {len(data.get('competitions', []))} competitions")
            return True
        elif response.status == 401:
            logger.error("❌ API key is invalid (Unauthorized)")
            return False
        else:
            logger.error(f"❌ Unexpected response: HTTP {response.status}")
            try:
                error_text = await response.text()
                logger.error(f"Error response: {error_text}")
            except:
                pass
            return False
    except Exception as e:
        logger.error(f"❌ Error testing API key: {e}")
        return False
    finally:
        await http_client.close()

async def test_get_matches():
    """Test fetching matches with date parameters."""
    logger.info("Testing match retrieval...")
    
    # Create scraper instance
    scraper = FootballDataScraper()
    
    # Check if API key is available
    if not scraper.api_key:
        logger.error("❌ No API key available for match retrieval test")
        return False
        
    masked_key = scraper.api_key[:4] + "..." + scraper.api_key[-4:] if len(scraper.api_key) > 8 else "***"
    logger.info(f"Using API key: {masked_key}")
    
    # Test with Premier League
    league_code = "premier_league"
    logger.info(f"Fetching matches for {league_code}...")
    
    # Test with date ranges - use dates that are guaranteed to have matches
    # Football season typically runs from August to May
    # Use a date range from the previous season
    date_from = "2024-08-01"  # Beginning of 2024-2025 season
    date_to = "2024-09-01"    # One month of matches
    
    logger.info(f"Date range: {date_from} to {date_to}")
    
    try:
        # Test competition endpoint first to check basic connectivity
        http_client = HttpClientAsync("https://api.football-data.org/v4")
        
        logger.info("Testing competitions endpoint first...")
        comp_response = await http_client.get(
            "/competitions/PL",
            headers={"X-Auth-Token": scraper.api_key, "Accept": "application/json"}
        )
        
        if comp_response.status != 200:
            logger.error(f"❌ Failed to get competition info: HTTP {comp_response.status}")
            error_data = await comp_response.text()
            logger.error(f"API Error: {error_data}")
            await http_client.close()
            return False
        else:
            logger.info("✅ Successfully connected to competition endpoint")
        
        # Now test matches endpoint
        logger.info("Now testing matches endpoint...")
        matches_df = await scraper.get_matches(
            league_code=league_code,
            date_from=date_from,
            date_to=date_to
        )
        
        if matches_df is not None and not matches_df.empty:
            logger.info(f"✅ Successfully retrieved {len(matches_df)} matches")
            logger.info(f"Sample matches: {matches_df.head(3)}")
            return True
        else:
            logger.warning("⚠️ No matches found or empty dataframe returned")
            # Try a different date range as fallback
            logger.info("Trying alternative date range as fallback...")
            matches_df = await scraper.get_matches(
                league_code=league_code,
                date_from="2024-01-01",
                date_to="2024-02-01"
            )
            
            if matches_df is not None and not matches_df.empty:
                logger.info(f"✅ Successfully retrieved {len(matches_df)} matches with fallback date range")
                return True
            else:
                logger.error("❌ Failed with alternative date range as well")
                return False
    except Exception as e:
        logger.error(f"❌ Error getting matches: {e}", exc_info=True)
        return False
    finally:
        if hasattr(scraper, 'http_client') and hasattr(scraper.http_client, "close"):
            await scraper.http_client.close()

async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("🔍 Football-Data API Diagnostic Test")
    logger.info("=" * 60)
    
    # Test API key
    api_key_valid = await test_api_key()
    
    if not api_key_valid:
        logger.error("API key test failed. Aborting further tests.")
        return
    
    # Test getting matches
    await test_get_matches()
    
    logger.info("=" * 60)
    logger.info("Tests completed")
    logger.info("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(f"Unhandled exception: {e}")
