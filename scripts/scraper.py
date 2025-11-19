import json
import logging
import os
import random
import time
from typing import Dict, List, Optional  # Added Optional

import requests
from bs4 import BeautifulSoup
# from core.scrapers.proxy_manager import ProxyManager # Removed unused import
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import config and utils
from data.storage.database import \
    UNIFIED_CONFIG  # Assuming config is loaded here
from scripts.utils.proxy_rotator import ProxyMaster

# Load scraper config
try:
    SCRAPER_CONFIG = UNIFIED_CONFIG.get('scraper', {})
    LOG_PATH = SCRAPER_CONFIG.get('log_path', 'logs/scraper.log') # Default log path
    REQUEST_PARAMS = SCRAPER_CONFIG.get('request_params', {'retries': 3, 'timeout': 15})
    DEFAULT_OUTPUT_PATH = SCRAPER_CONFIG.get('default_output_path', 'data/raw/scraped_fixtures.json')
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
except Exception as config_e:
    print(f"Warning: Error loading scraper config: {config_e}. Using defaults.")
    LOG_PATH = 'logs/scraper.log'
    REQUEST_PARAMS = {'retries': 3, 'timeout': 15}
    DEFAULT_OUTPUT_PATH = 'data/raw/scraped_fixtures.json'
    # Ensure default log directory exists
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


# Configure structured logging using config path
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class AdvancedFootballScraper:
    def __init__(self):
        logger.info("Initializing AdvancedFootballScraper...")
        self.proxy_master = ProxyMaster()  # Instantiate ProxyMaster (Assumed primary)
        self.ua = UserAgent()
        # Load request params from config
        self.request_retries = REQUEST_PARAMS.get('retries', 3)
        self.request_timeout = REQUEST_PARAMS.get('timeout', 15)
        logger.info(f"Scraper configured with Retries={self.request_retries}, Timeout={self.request_timeout}")
        self.session = self._create_resilient_session()
        # self.proxy_manager = ProxyManager() # Removed unused ProxyManager
        # self.proxy_manager.health_check()  # Removed call related to unused ProxyManager

    # Removed the old _refresh_proxies method entirely

    def _create_resilient_session(self) -> requests.Session:
        """Advanced retry logic with exponential backoff"""
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"]),
        )  # Added missing closing parenthesis
        session.mount("https://", HTTPAdapter(max_retries=retries))
        # Note: Session retries might differ from scrape_fixtures loop retries
        # Session retries handle network/server errors, loop retries handle proxy/parsing issues
        return session

    def _rotate_identity(self) -> Dict:
        """Generate fresh browser fingerprint."""
        return {
            "User-Agent": self.ua.random,
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": random.choice(["https://google.com", "https://bing.com"]),  # Removed angle brackets
        }

    def scrape_fixtures(self, url: str) -> List[Dict]:
        """Main scraping method with error containment."""
        # Use configured number of retries for the outer loop
        for attempt in range(self.request_retries):
            try:
                # Get proxy from ProxyMaster
                proxy_address: Optional[str] = self.proxy_master.get_proxy()
                if not proxy_address:
                    logger.error(
                        "Failed to get a working proxy. Aborting scrape attempt."
                    )
                    # Optionally wait and retry, or just fail this attempt
                    time.sleep(5)  # Wait before next attempt if no proxy
                    continue

                proxy_dict = {"http": proxy_address, "https": proxy_address}
                headers = self._rotate_identity()

                logger.info(f"Attempt {attempt+1} using proxy: {proxy_address}")
                response = self.session.get(
                    url,
                    headers=headers,
                    proxies=proxy_dict,
                    timeout=self.request_timeout, # Use configured timeout
                )
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

                # Parse response
                soup = BeautifulSoup(response.text, "lxml")
                matches = []

                for match_element in soup.select("div.match-container"): # Renamed variable for clarity
                    try:
                        teams = [t.text.strip() for t in match_element.select(".team-name")]
                        odds_elements = match_element.select(".odds")
                        # Attempt to convert odds, skip match if any conversion fails
                        odds = [float(o.text) for o in odds_elements]
                        if len(teams) < 2 or len(odds) == 0: # Basic validation
                             logger.warning(f"Skipping match due to incomplete data: {teams}")
                             continue
                        matches.append(
                            {"teams": teams, "odds": odds, "timestamp": int(time.time())}
                        )
                    except ValueError as ve:
                        logger.warning(f"Could not parse odds for a match, skipping. Error: {ve}. Text: {[o.text for o in odds_elements]}")
                        continue # Skip this match if odds parsing fails
                    except Exception as inner_ex: # Catch other potential errors during match processing
                        logger.error(f"Error processing a match element: {inner_ex}", exc_info=True)
                        continue # Skip this match

                logger.info(f"Successfully processed {len(matches)} fixtures")
                logger.info(f"Successfully processed {len(matches)} fixtures from {url}")
                return matches

            except requests.exceptions.Timeout as e:
                 logger.warning(f"Attempt {attempt+1}/{self.request_retries} failed (Timeout): {e}")
                 # Consider marking proxy as slow/bad in ProxyMaster if timeouts persist
                 time.sleep(2**attempt)
            except requests.exceptions.ConnectionError as e:
                 logger.warning(f"Attempt {attempt+1}/{self.request_retries} failed (Connection Error): {e}")
                 # Mark proxy as potentially bad? ProxyMaster might handle this.
                 time.sleep(2**attempt)
            except requests.exceptions.HTTPError as e:
                 logger.warning(f"Attempt {attempt+1}/{self.request_retries} failed (HTTP Error {e.response.status_code}): {e}")
                 # Specific handling for common blocking codes
                 if e.response.status_code in [403, 401]:
                      logger.error(f"Proxy {proxy_address} likely blocked (Status {e.response.status_code}).")
                      # Optionally tell ProxyMaster to discard/deprioritize this proxy
                 elif e.response.status_code == 404:
                      logger.error(f"URL not found (404): {url}. Aborting retries for this URL.")
                      return [] # Don't retry on 404
                 time.sleep(2**attempt)
            except requests.exceptions.RequestException as e: # Catch other request errors
                logger.warning(f"Attempt {attempt+1}/{self.request_retries} failed (RequestException): {e}")
                time.sleep(2**attempt)
            except Exception as e: # Catch other potential errors (e.g., parsing)
                logger.error(
                    f"Attempt {attempt+1}/{self.request_retries} failed (Unexpected Error): {e}",
                    exc_info=True,
                )
                time.sleep(2**attempt)

        logger.error(f"All scraping attempts failed for URL: {url}")
        return []  # Fail-safe empty return


if __name__ == "__main__":
    # Example usage: Scrape a default URL and save to configured default path
    # In a real scenario, URL and output path would likely come from args or another process
    target_url = "https://www.oddsportal.com/matches/soccer/" # Example URL (remove angle brackets if present)
    output_path = DEFAULT_OUTPUT_PATH # Use path from config

    logger.info(f"Starting example scrape for URL: {target_url}")
    scraper = AdvancedFootballScraper()
    fixtures = scraper.scrape_fixtures(target_url)

    if fixtures:
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(fixtures, f, indent=2) # Add indent for readability
            logger.info(f"Successfully scraped {len(fixtures)} fixtures and saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to write scraped data to {output_path}: {e}")
        except Exception as e:
             logger.exception(f"An unexpected error occurred during file writing: {e}")
    else:
        logger.warning(f"No fixtures were scraped from {target_url}.")
