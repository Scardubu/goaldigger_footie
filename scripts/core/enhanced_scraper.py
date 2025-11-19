import argparse  # Added for new entry point
import asyncio
import datetime  # Added for timestamping screenshots
import json
import logging
import os
import sys  # Added sys import
import time
from typing import Any, Dict, List, Optional, Set, Tuple  # Added Tuple
from urllib.parse import quote_plus, urljoin, urlparse  # Added quote_plus

from bs4 import BeautifulSoup, Comment  # Added Comment

# Determine the project root directory (assuming the script is in goaldiggers/scripts/core/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Insert the project root into sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- BEGIN FIX for Playwright NotImplementedError on Windows ---
# This must be done before any asyncio event loop is created or started by Playwright.
# Placing it at the top of the module ensures it's set when the module is imported.
if sys.platform == "win32":
    try:
        # Attempt to set the policy. This will raise an error if a loop is already running
        # and the policy has already been 'cemented' by other asyncio operations.
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception as e:
        # If this fails, it might be because an event loop is already running and
        # its policy cannot be changed. The issue might persist if the running loop
        # is not a Proactor loop. Ideally, this policy is set at the very
        # beginning of the application's main entry point.
        # For now, we log a warning if a logger is available, or just pass.
        # (Logger is not configured at this very top point of the script)
        # print(f"Note: Could not set WindowsProactorEventLoopPolicy at module import: {e}")
        pass
# --- END FIX ---

import aiohttp

# Import TimeoutError specifically
from playwright.async_api import TimeoutError, async_playwright

from database.db_manager import DatabaseManager
from utils.config import Config
from utils.logging_config import setup_logging
from utils.proxy_manager import ProxyManager

# Load configuration FIRST
config = Config.load()

# Setup logging AFTER loading config, passing the logging section
logging_config = config.get('logging', {}) # Get logging config dict
setup_logging(config_dict=logging_config) # Pass config to setup
logger = logging.getLogger(__name__)
logger.info("Logging setup complete using configuration.") # Confirm setup

# Continue loading other config sections
SCRAPER_CONFIG = Config.get('scraper', {}) # Use Config.get for safety
NETWORK_CONFIG = Config.get('network', {})
PLAYWRIGHT_CONFIG = config.get('playwright', {})
BASE_DATA_DIR = config.get('data_dir', 'data') # Use 'data' relative to project root

# --- Helper Functions ---

async def _fetch_with_playwright(url: str, proxy_config: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Fetches page content using Playwright for dynamic content.
    Accepts proxy configuration dictionary.
    """
    browser = None
    page = None
    launch_options = {
        'headless': PLAYWRIGHT_CONFIG.get('headless', True),
    }
    # Add proxy to launch options if provided
    if proxy_config:
        launch_options['proxy'] = proxy_config
        logger.info(f"Using proxy for Playwright: {proxy_config.get('server')}")
    else:
        logger.info("Proceeding with Playwright fetch without proxy.") # Changed from warning to info

    browser = None # Define browser outside try block for broader scope in finally
    page = None # Define page outside try block
    try:
        logger.info(f"Attempting Playwright launch for {url}...")
        async with async_playwright() as p:
            logger.debug("Async Playwright context entered.")
            # Use explicit executable path if configured or found
            # chrome_executable_path = PLAYWRIGHT_CONFIG.get('executable_path', r"C:\Program Files\Google\Chrome\Application\chrome.exe") # Default path added previously
            # if chrome_executable_path and os.path.exists(chrome_executable_path):
            #      launch_options['executable_path'] = chrome_executable_path
            #      logger.info(f"Using executable path: {chrome_executable_path}")
            # else:
            #      logger.info(f"Executable path '{chrome_executable_path}' not found or not configured. Relying on Playwright default.") # Changed from warning
            logger.info("Relying on Playwright default browser resolution (ensure 'python -m playwright install' has been run).")

            logger.info(f"Playwright launching chromium with options: {launch_options}")
            browser = await p.chromium.launch(**launch_options)
            logger.info("Playwright browser launched successfully.")

            # --- Add User-Agent Rotation ---
            user_agents = Config.get('scraping.request.user_agents', [])
            context_options = {}
            if user_agents:
                import random
                selected_user_agent = random.choice(user_agents)
                context_options['user_agent'] = selected_user_agent
                logger.info(f"Using Playwright User-Agent: {selected_user_agent}")
            else:
                logger.warning("No user agents found in config for Playwright rotation.")

            page = await browser.new_page(**context_options)
            # --- End User-Agent Rotation ---
            logger.info("Playwright page created successfully.")

            # --- Add Stealth Init Scripts ---
            logger.info("Adding Playwright stealth init scripts...")
            js_hide_webdriver = "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
            js_spoof_plugins = """
                const mockPlugins = [
                  { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format', mimeTypes: [{ type: 'application/pdf', suffixes: 'pdf' }] },
                  { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '', mimeTypes: [{ type: 'application/pdf', suffixes: 'pdf' }] },
                  { name: 'Native Client', filename: 'internal-nacl-plugin', description: '', mimeTypes: [{ type: 'application/x-nacl', suffixes: '' }, { type: 'application/x-pnacl', suffixes: '' }] }
                ];
                Object.defineProperty(navigator, 'plugins', { get: () => mockPlugins });
                Object.defineProperty(navigator, 'mimeTypes', {
                  get: () => {
                    const mimeTypes = {};
                    mockPlugins.forEach(plugin => {
                      plugin.mimeTypes.forEach(mime => { mimeTypes[mime.type] = mime; });
                    });
                    return mimeTypes;
                  },
                });
            """
            js_spoof_languages = "Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });"
            js_spoof_webgl = """
                try {
                  const getParameter = WebGLRenderingContext.prototype.getParameter;
                  WebGLRenderingContext.prototype.getParameter = function(parameter) {
                    if (parameter === 37445) return 'Intel Open Source Technology Center'; // VENDOR
                    if (parameter === 37446) return 'Mesa DRI Intel(R) Ivybridge Mobile '; // RENDERER
                    return getParameter(parameter);
                  };
                } catch (e) { console.error('WebGL spoofing failed:', e); }
            """
            js_override_permissions = """
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                  parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
                );
            """
            js_remove_chrome_props = """
                if (window.chrome) {
                  try { delete window.chrome.runtime; } catch(e) {}
                  try { delete window.chrome.csi; } catch(e) {}
                }
                if (window.__driver_evaluate) delete window.__driver_evaluate;
                if (window.__webdriver_evaluate) delete window.__webdriver_evaluate;
            """
            await page.add_init_script(js_hide_webdriver)
            await page.add_init_script(js_spoof_plugins)
            await page.add_init_script(js_spoof_languages)
            await page.add_init_script(js_spoof_webgl)
            await page.add_init_script(js_override_permissions)
            await page.add_init_script(js_remove_chrome_props)
            logger.info("Stealth init scripts added.")
            # --- End Stealth Init Scripts ---


            page.on("console", lambda msg: logger.warning(f"Browser Console ({msg.type}): {msg.text}"))

            # Use timeout from config, default to 120s if not found
            page_timeout = PLAYWRIGHT_CONFIG.get('timeout', 120000)
            wait_condition = PLAYWRIGHT_CONFIG.get('wait_until', 'load') # Default to 'load' instead of 'networkidle'
            logger.info(f"Playwright navigating to {url} with timeout {page_timeout}ms...")
            await page.goto(url, timeout=page_timeout)
            logger.info(f"Playwright navigation to {url} completed.")
            logger.info(f"Playwright waiting for load state: {wait_condition}")
            await page.wait_for_load_state(wait_condition, timeout=page_timeout) # Apply timeout here too
            logger.info(f"Playwright load state '{wait_condition}' reached for {url}.")
            content = await page.content()
            logger.info(f"Successfully fetched page content (Playwright): {url}")
            return content # Return content before closing browser in this path
    except playwright._impl._errors.TimeoutError as te:
        # Handle timeout specifically
        logger.error(f"Playwright TimeoutError for {url} after {page_timeout}ms. Error: {te}", exc_info=True)
        # Attempt screenshot
        if page:
            logger.info(f"Attempting screenshot due to Playwright TimeoutError for {url}...")
            try:
                screenshot_dir = os.path.join(project_root, BASE_DATA_DIR, 'debug_screenshots')
                os.makedirs(screenshot_dir, exist_ok=True)
                safe_filename = "".join(c if c.isalnum() else "_" for c in urlparse(url).netloc + urlparse(url).path)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(screenshot_dir, f"playwright_timeout_error_{timestamp}_{safe_filename[:100]}.png")
                await page.screenshot(path=screenshot_path, full_page=True)
                logger.info(f"Timeout screenshot successfully saved to: {screenshot_path}")
            except Exception as screenshot_err:
                logger.error(f"Failed to take screenshot on TimeoutError for {url}. Screenshot Error: {screenshot_err}", exc_info=True)
        else:
             logger.warning(f"Playwright page object not available, cannot take screenshot for TimeoutError on {url}.")
        return None # Return None as the fetch failed due to timeout
    except Exception as e:
        # Log other exceptions
        logger.error(f"Playwright fetch failed for {url}. Error Type: {type(e).__name__}, Message: {e}", exc_info=True)
        # Attempt screenshot
        if page:
            logger.info(f"Attempting screenshot due to Playwright error for {url}...")
            try:
                # Ensure data_dir is resolved correctly relative to project root
                screenshot_dir = os.path.join(project_root, BASE_DATA_DIR, 'debug_screenshots') # Use BASE_DATA_DIR consistently
                os.makedirs(screenshot_dir, exist_ok=True) # Ensure directory exists
                safe_filename = "".join(c if c.isalnum() else "_" for c in urlparse(url).netloc + urlparse(url).path)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(screenshot_dir, f"playwright_error_{timestamp}_{safe_filename[:100]}.png") # More specific filename
                await page.screenshot(path=screenshot_path, full_page=True)
                logger.info(f"Screenshot successfully saved on error to: {screenshot_path}")
            except Exception as screenshot_err:
                # Log the screenshot error specifically
                logger.error(f"Failed to take screenshot on error for {url}. Screenshot Error Type: {type(screenshot_err).__name__}, Message: {screenshot_err}", exc_info=True)
        else:
             logger.warning(f"Playwright page object not available, cannot take screenshot for error on {url}.")
        return None # Return None as the fetch failed
    finally:
        # Ensure browser is closed even if errors occurred after launch
        if page:
             logger.debug(f"Closing Playwright page for {url} in finally block.")
             try: await page.close()
             except Exception as page_close_err: logger.warning(f"Error closing page for {url}: {page_close_err}")
        if browser:
            logger.info(f"Closing Playwright browser for {url} in finally block.")
            try:
                await browser.close()
                logger.info(f"Playwright browser closed successfully for {url}.")
            except Exception as close_err:
                logger.error(f"Error closing Playwright browser in finally block for {url}: {close_err}")

async def _fetch_with_aiohttp(session: aiohttp.ClientSession, url: str, proxy: Optional[str] = None) -> Optional[str]:
    """Fetches page content using aiohttp for static content."""
    request_params = {
        "timeout": aiohttp.ClientTimeout(total=NETWORK_CONFIG.get('timeout', 30)),
        "headers": NETWORK_CONFIG.get('headers', {})
    }
    if proxy:
        request_params["proxy"] = proxy

    try:
        async with session.get(url, **request_params) as response:
            status = response.status
            if status == 200:
                content = await response.read()
                logger.debug(f"Successfully fetched (aiohttp): {url}")
                try:
                    return content.decode('utf-8')
                except UnicodeDecodeError:
                    logger.warning(f"Could not decode content from {url} as UTF-8, trying latin-1")
                    try:
                        return content.decode('latin-1')
                    except Exception:
                        logger.error(f"Failed to decode content from {url}")
                        return None
            else:
                logger.warning(f"aiohttp fetch failed for {url}, status: {status}")
                return None
    except Exception as e:
        logger.error(f"aiohttp fetch error for {url}: {e}", exc_info=True)
        return None

def _extract_links(html_content: str, base_url: str) -> Set[str]:
    """Extracts and normalizes links from HTML content."""
    links = set()
    soup = BeautifulSoup(html_content, 'html.parser')
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        url = urljoin(base_url, href)
        parsed_url = urlparse(url)
        if parsed_url.scheme in ['http', 'https'] and parsed_url.netloc:
             links.add(url)
    return links

# --- Source-Specific Parsers ---

def _parse_understat(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Parses data specifically from an Understat match page."""
    logger.info(f"Parsing Understat content from {url}")
    data = {"source": "Understat", "url": url, "status": "parsed"}
    try:
        scripts = soup.find_all('script')
        match_data_script = None
        # Look for script containing 'shotsData' or 'matchesData' etc.
        data_vars = ['shotsData', 'matchesData', 'teamsData', 'playersData']
        found_data = {}
        for script in scripts:
            if script.string:
                for var_name in data_vars:
                    if var_name in script.string:
                        # Extract JSON part (improved extraction)
                        try:
                            if f"JSON.parse('" in script.string:
                                start_index = script.string.find(f"{var_name} = JSON.parse('") + len(f"{var_name} = JSON.parse('")
                                end_index = script.string.find("');", start_index)
                                if start_index > -1 and end_index > -1:
                                    json_str_escaped = script.string[start_index:end_index]
                                    json_str = json_str_escaped.encode().decode('unicode_escape') # More robust decoding
                                    found_data[var_name] = json.loads(json_str)
                                    logger.debug(f"Extracted {var_name} from JSON.parse")
                            elif f"var {var_name} = " in script.string:
                                start_index = script.string.find(f"var {var_name} = ") + len(f"var {var_name} = ")
                                # Find the end: could be '];' or '};'
                                end_bracket = script.string.find("];", start_index)
                                end_curly = script.string.find("};", start_index)
                                end_index = -1
                                if end_bracket > -1 and end_curly > -1:
                                    end_index = min(end_bracket, end_curly) + 1
                                elif end_bracket > -1:
                                    end_index = end_bracket + 1
                                elif end_curly > -1:
                                    end_index = end_curly + 1

                                if start_index > -1 and end_index > -1:
                                    json_str = script.string[start_index:end_index]
                                    found_data[var_name] = json.loads(json_str)
                                    logger.debug(f"Extracted {var_name} from direct assignment")
                        except Exception as json_e:
                            logger.error(f"Error parsing JSON for {var_name} in {url}: {json_e}")

        if not found_data:
             data['status'] = 'error'
             data['error'] = "Could not find or parse expected script variables (e.g., shotsData)"
             logger.warning(f"Could not find expected script variables on Understat page: {url}")
        else:
             data['extracted_data'] = found_data # Store all extracted data
             logger.info(f"Successfully extracted data variables: {list(found_data.keys())} from {url}")

    except Exception as e:
        logger.error(f"Error parsing Understat page {url}: {e}", exc_info=True)
        data['status'] = 'error'
        data['error'] = str(e)
    return data

def _parse_fbref(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Parses data specifically from an FBref match page."""
    logger.info(f"Parsing FBref content from {url}")
    data = {"source": "FBref", "url": url, "status": "parsed"}
    try:
        # Example: Find lineup tables
        lineup_table = soup.find('div', {'class': 'lineup'})
        if lineup_table:
             data['lineup_status'] = "found_lineup_div" # Placeholder
             # TODO: Add actual parsing logic for lineup
        else:
             data['lineup_status'] = "lineup_div_not_found"

        # Example: Find main stats table (often commented out)
        stats_table = None
        stats_comment = soup.find(string=lambda text: isinstance(text, Comment) and 'id="stats_standard"' in text)
        if stats_comment:
           stats_soup = BeautifulSoup(stats_comment, 'html.parser')
           stats_table = stats_soup.find('table', {'id': 'stats_standard'})
           if stats_table:
                data['stats_table_status'] = "found_stats_table_in_comment"
           else:
                data['stats_table_status'] = "comment_found_but_no_table"
        else:
            # Try finding table directly if not in comment
            stats_table = soup.find('table', {'id': 'stats_standard'})
            if stats_table:
                 data['stats_table_status'] = "found_stats_table_directly"
            else:
                 data['stats_table_status'] = "stats_table_not_found"

        if stats_table:
             # TODO: Add actual parsing logic for stats table (e.g., using pandas.read_html)
             data['stats_placeholder'] = "Table found, parsing needed"
             pass

        # Add more specific parsing logic here based on FBref's structure

    except Exception as e:
        logger.error(f"Error parsing FBref page {url}: {e}", exc_info=True)
        data['status'] = 'error'
        data['error'] = str(e)
    return data

def _parse_transfermarkt(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Parses data specifically from a Transfermarkt match page."""
    logger.info(f"Parsing Transfermarkt content from {url}")
    data = {"source": "Transfermarkt", "url": url, "status": "parsed"}
    try:
        # Placeholder: Extract market values, lineups, events
        # Example: Find box score container
        box_score = soup.find('div', class_='box') # Very generic, needs refinement
        if box_score:
             data['box_score_status'] = "found_box_div"
             # TODO: Add parsing for lineups, goals, cards within the box score
        else:
             data['box_score_status'] = "box_div_not_found"

        # Add more specific parsing logic here based on Transfermarkt's structure

    except Exception as e:
        logger.error(f"Error parsing Transfermarkt page {url}: {e}", exc_info=True)
        data['status'] = 'error'
        data['error'] = str(e)
    return data

# --- Generic/Dispatcher Parser ---

def parse_html_content(html_content: str, source_name: str, url: str) -> Optional[Dict[str, Any]]:
    """
    Parses HTML content based on the source_name.
    Dispatches to source-specific parsers.
    """
    if not html_content:
        logger.warning(f"Cannot parse empty HTML content for source {source_name}, URL {url}")
        return {"source": source_name, "url": url, "status": "error", "error": "Empty HTML content provided"}

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        if source_name == "Understat":
            return _parse_understat(soup, url)
        elif source_name == "FBref":
            return _parse_fbref(soup, url)
        elif source_name == "Transfermarkt":
            return _parse_transfermarkt(soup, url)
        else:
            logger.warning(f"No specific parser implemented for source: {source_name}. URL: {url}")
            return {"source": source_name, "url": url, "status": "error", "error": f"No parser for source '{source_name}'"}

    except Exception as e:
        logger.error(f"Error creating BeautifulSoup object or dispatching parser for {url} (Source: {source_name}): {e}", exc_info=True)
        return {"source": source_name, "url": url, "status": "error", "error": f"Parsing failed: {e}"}


from urllib.parse import urlparse  # Ensure urlparse is imported at the top

# --- URL Finding Helper Functions ---

async def _find_transfermarkt_url(match_id, match_date_str, home_team, away_team, season, proxy_manager: ProxyManager) -> Optional[str]:
    """Finds the correct Transfermarkt match URL using search."""
    search_term = f'"{home_team}" "{away_team}" {match_date_str}'
    base_search_url = "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche"
    search_url = f"{base_search_url}?query={quote_plus(search_term)}"
    logger.info(f"Attempting Transfermarkt search for match {match_id} with URL: {search_url}")

    proxy_str = proxy_manager.get_proxy()
    proxy_config = None
    if proxy_str:
        parsed_proxy = urlparse(proxy_str)
        proxy_config = {
            "server": f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}"
        }
        if parsed_proxy.username:
            proxy_config["username"] = parsed_proxy.username
        if parsed_proxy.password:
            proxy_config["password"] = parsed_proxy.password

    html_content = await _fetch_with_playwright(search_url, proxy_config=proxy_config) # Pass parsed proxy config

    if not html_content:
        logger.error(f"Failed to fetch Transfermarkt search results for match {match_id}.")
        return None

    # Removed the HTML saving debug code from here

    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        match_links = soup.find_all('a', href=lambda href: href and '/spielbericht/index/spielbericht/' in href)
        logger.debug(f"Found {len(match_links)} potential Transfermarkt match links for '{search_term}'.")

        for link in match_links:
            link_text = link.get_text(strip=True).lower()
            parent_text = link.parent.get_text(strip=True).lower() if link.parent else ""
            combined_text = link_text + " " + parent_text

            if home_team.lower() in combined_text and away_team.lower() in combined_text:
                relative_url = link['href']
                absolute_url = urljoin("https://www.transfermarkt.com", relative_url)
                logger.info(f"Found likely Transfermarkt URL for match {match_id}: {absolute_url}")
                return absolute_url

        logger.warning(f"Could not find a specific matching Transfermarkt link for match {match_id} in search results.")
        return None
    except Exception as e:
        logger.error(f"Error parsing Transfermarkt search results for match {match_id}: {e}", exc_info=True)
        # Note: Proxy rotation on failure is removed here, should be handled by caller if needed
        # proxy_manager.rotate_proxy()
        return None

async def _find_understat_url(match_id, match_date_str, home_team, away_team, league_id, season, proxy_manager: ProxyManager) -> Optional[str]:
    """Finds the correct Understat match URL by parsing the league page."""
    league_mapping = {"PL": "EPL", "PD": "La_liga", "BL1": "Bundesliga", "SA": "Serie_A", "FL1": "Ligue_1", "DED": "Eredivisie"}
    understat_league_code = league_mapping.get(league_id)
    if not understat_league_code:
        logger.warning(f"No Understat league mapping for league_id: {league_id}. Cannot find URL for match {match_id}.")
        return None

    league_url = f"https://understat.com/league/{understat_league_code}/{season}"
    logger.info(f"Attempting Understat lookup for match {match_id} via league page: {league_url}")

    proxy_str = proxy_manager.get_proxy()
    proxy_config = None
    if proxy_str:
        parsed_proxy = urlparse(proxy_str)
        proxy_config = {
            "server": f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}"
        }
        if parsed_proxy.username:
            proxy_config["username"] = parsed_proxy.username
        if parsed_proxy.password:
            proxy_config["password"] = parsed_proxy.password

    # --- Modified Logic: Use Playwright to wait for the data request ---
    browser = None
    page = None
    json_part = None
    try:
        async with async_playwright() as p:
            logger.debug("Async Playwright context entered for Understat league page.")
            # chrome_executable_path = PLAYWRIGHT_CONFIG.get('executable_path', r"C:\Program Files\Google\Chrome\Application\chrome.exe")
            launch_options = {
                'headless': PLAYWRIGHT_CONFIG.get('headless', True),
                'proxy': proxy_config # Use the proxy config determined earlier
            }
            # if chrome_executable_path and os.path.exists(chrome_executable_path):
            #      launch_options['executable_path'] = chrome_executable_path
            #      logger.info(f"Using executable path: {chrome_executable_path}")
            # else:
            #      logger.info(f"Executable path '{chrome_executable_path}' not found or not configured. Relying on Playwright default.")
            logger.info("Relying on Playwright default browser resolution for Understat league page (ensure 'python -m playwright install' has been run).")

            logger.info(f"Playwright launching chromium for Understat league page: {league_url} with options: {launch_options}")
            browser = await p.chromium.launch(**launch_options)
            logger.info("Playwright browser launched successfully.")

            user_agents = Config.get('scraping.request.user_agents', [])
            context_options = {}
            if user_agents:
                import random
                selected_user_agent = random.choice(user_agents)
                context_options['user_agent'] = selected_user_agent
                logger.info(f"Using Playwright User-Agent: {selected_user_agent}")
            else:
                logger.warning("No user agents found in config for Playwright rotation.")

            page = await browser.new_page(**context_options)
            logger.info("Playwright page created successfully.")

            # Add stealth scripts (copied from _fetch_with_playwright for consistency)
            logger.info("Adding Playwright stealth init scripts...")
            js_hide_webdriver = "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
            js_spoof_plugins = """
                const mockPlugins = [
                  { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format', mimeTypes: [{ type: 'application/pdf', suffixes: 'pdf' }] },
                  { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '', mimeTypes: [{ type: 'application/pdf', suffixes: 'pdf' }] },
                  { name: 'Native Client', filename: 'internal-nacl-plugin', description: '', mimeTypes: [{ type: 'application/x-nacl', suffixes: '' }, { type: 'application/x-pnacl', suffixes: '' }] }
                ];
                Object.defineProperty(navigator, 'plugins', { get: () => mockPlugins });
                Object.defineProperty(navigator, 'mimeTypes', {
                  get: () => {
                    const mimeTypes = {};
                    mockPlugins.forEach(plugin => {
                      plugin.mimeTypes.forEach(mime => { mimeTypes[mime.type] = mime; });
                    });
                    return mimeTypes;
                  },
                });
            """
            js_spoof_languages = "Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });"
            js_spoof_webgl = """
                try {
                  const getParameter = WebGLRenderingContext.prototype.getParameter;
                  WebGLRenderingContext.prototype.getParameter = function(parameter) {
                    if (parameter === 37445) return 'Intel Open Source Technology Center'; // VENDOR
                    if (parameter === 37446) return 'Mesa DRI Intel(R) Ivybridge Mobile '; // RENDERER
                    return getParameter(parameter);
                  };
                } catch (e) { console.error('WebGL spoofing failed:', e); }
            """
            js_override_permissions = """
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                  parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
                );
            """
            js_remove_chrome_props = """
                if (window.chrome) {
                  try { delete window.chrome.runtime; } catch(e) {}
                  try { delete window.chrome.csi; } catch(e) {}
                }
                if (window.__driver_evaluate) delete window.__driver_evaluate;
                if (window.__webdriver_evaluate) delete window.__webdriver_evaluate;
            """
            await page.add_init_script(js_hide_webdriver)
            await page.add_init_script(js_spoof_plugins)
            await page.add_init_script(js_spoof_languages)
            await page.add_init_script(js_spoof_webgl)
            await page.add_init_script(js_override_permissions)
            await page.add_init_script(js_remove_chrome_props)
            logger.info("Stealth init scripts added.")

            page.on("console", lambda msg: logger.warning(f"Browser Console ({msg.type}): {msg.text}"))

            page_timeout = PLAYWRIGHT_CONFIG.get('timeout', 120000)
            # Change wait condition to potentially faster 'domcontentloaded'
            wait_condition = PLAYWRIGHT_CONFIG.get('wait_until', 'domcontentloaded')
            logger.info(f"Playwright navigating to {league_url} with timeout {page_timeout}ms and wait_until='{wait_condition}'...")

            # Define the selector for the match table container
            match_table_selector = "div.chemp.table" # Selector for the main table area
            logger.info(f"Will wait for selector: '{match_table_selector}'")

            # Navigate and wait for the selector
            await page.goto(league_url, timeout=page_timeout, wait_until=wait_condition)
            logger.info(f"Navigation to {league_url} complete ({wait_condition}). Now waiting for selector '{match_table_selector}'...")
            await page.wait_for_selector(match_table_selector, timeout=page_timeout/2) # Wait for table to appear
            logger.info(f"Selector '{match_table_selector}' found. Fetching page content...")

            # Get content *after* waiting for the table
            html_content_after_wait = await page.content()
            logger.info(f"Successfully fetched page content after waiting for selector.")

            # --- DEBUG: Save HTML after waiting ---
            try:
                debug_dir = os.path.join(project_root, BASE_DATA_DIR, 'debug_screenshots')
                os.makedirs(debug_dir, exist_ok=True)
                debug_filename = os.path.join(debug_dir, f"understat_league_after_selector_{league_id}_{season}_{time.strftime('%Y%m%d%H%M%S')}.html")
                with open(debug_filename, 'w', encoding='utf-8') as f:
                    f.write(html_content_after_wait)
                logger.info(f"Saved Understat league HTML after selector wait to {debug_filename}")
            except Exception as save_e:
                logger.error(f"Failed to save debug HTML after selector wait for Understat league {league_id} {season}: {save_e}")
            # --- END DEBUG ---

            # Now, parse the fetched HTML to find the script tag (reverting to original logic on updated HTML)
            logger.info("Parsing HTML content to find 'matchesData' script tag...")
            soup = BeautifulSoup(html_content_after_wait, 'html.parser')
            script_tag = soup.find('script', string=lambda t: t and 'matchesData' in t)

            if not script_tag or not script_tag.string:
                logger.error(f"Could not find 'matchesData' script tag in HTML after waiting for selector on {league_url}")
                # Log first few lines of HTML for context
                logger.debug(f"HTML start after wait:\n{html_content_after_wait[:500]}")
                return None

            script_content = script_tag.string
            logger.debug(f"Found script tag content containing 'matchesData':\n{script_content[:500]}...") # Log beginning of script

            # Extract JSON from the script tag
            try:
                if "JSON.parse('" in script_content:
                    start_index = script_content.find("JSON.parse('") + len("JSON.parse('")
                    end_index = script_content.find("');", start_index)
                    if start_index > -1 and end_index > -1:
                        json_str_escaped = script_content[start_index:end_index]
                        # Decode potential unicode escapes and backslash escapes
                        json_str = bytes(json_str_escaped, "utf-8").decode("unicode_escape").replace('\\\\', '\\')
                        json_part = json.loads(json_str)
                        logger.debug("Extracted matchesData from JSON.parse()")
                elif "var matchesData = " in script_content:
                    start_index = script_content.find("var matchesData = ") + len("var matchesData = ")
                    # Find the end: could be '];' or '};'
                    end_bracket = script_content.find("];", start_index)
                    end_curly = script_content.find("};", start_index)
                    end_index = -1
                    if end_bracket > -1 and end_curly > -1: end_index = min(end_bracket, end_curly) + 1
                    elif end_bracket > -1: end_index = end_bracket + 1
                    elif end_curly > -1: end_index = end_curly + 1

                    if start_index > -1 and end_index > -1:
                        json_str = script_content[start_index:end_index]
                        json_part = json.loads(json_str)
                        logger.debug("Extracted matchesData from direct assignment")
                else:
                     logger.error("Could not find known pattern (JSON.parse or var assignment) for matchesData.")
                     return None

            except Exception as extract_e:
                 logger.error(f"Error during JSON extraction from script tag: {extract_e}", exc_info=True)
                 logger.debug(f"Script content snippet: {script_content[:1000]}") # Log more script content on error
                 return None

    except TimeoutError as te: # Catch TimeoutError from goto or wait_for_selector
        # Determine stage based on whether the selector was found
        timeout_stage = "navigation/load" if 'page' not in locals() or not page.url == league_url else "selector_wait"
        logger.error(f"Playwright TimeoutError during '{timeout_stage}' stage for {league_url}. Error: {te}", exc_info=True)

        # Attempt screenshot
        if page:
            try:
                screenshot_dir = os.path.join(project_root, BASE_DATA_DIR, 'debug_screenshots')
                os.makedirs(screenshot_dir, exist_ok=True)
                safe_filename = "".join(c if c.isalnum() else "_" for c in urlparse(league_url).netloc + urlparse(league_url).path)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(screenshot_dir, f"playwright_understat_timeout_{timeout_stage}_{timestamp}_{safe_filename[:100]}.png")
                await page.screenshot(path=screenshot_path, full_page=True)
                logger.info(f"Timeout screenshot saved to: {screenshot_path}")
            except Exception as screenshot_err:
                logger.error(f"Failed to take screenshot on TimeoutError for {league_url}. Screenshot Error: {screenshot_err}", exc_info=True)
        # Return specific error structure for timeout
        # Note: The return inside the nested try/except for data_wait timeout handles that specific case.
        # This handles timeouts during navigation/load primarily.
        return {"source": "Understat", "url": league_url, "status": "error", "error": f"Playwright TimeoutError ({timeout_stage} stage): {te}"}

    except Exception as e:
        logger.error(f"Error during Playwright fetch or data wait for Understat league page {league_url}: {e}", exc_info=True)
        # Attempt screenshot on general error
        if page:
            try:
                screenshot_dir = os.path.join(project_root, BASE_DATA_DIR, 'debug_screenshots')
                os.makedirs(screenshot_dir, exist_ok=True)
                safe_filename = "".join(c if c.isalnum() else "_" for c in urlparse(league_url).netloc + urlparse(league_url).path)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = os.path.join(screenshot_dir, f"playwright_understat_error_{timestamp}_{safe_filename[:100]}.png")
                await page.screenshot(path=screenshot_path, full_page=True)
                logger.info(f"Error screenshot saved to: {screenshot_path}")
            except Exception as screenshot_err:
                logger.error(f"Failed to take screenshot on error for {league_url}. Screenshot Error: {screenshot_err}", exc_info=True)
        return None
    finally:
        if page:
             logger.debug(f"Closing Playwright page for {league_url} in finally block.")
             try: await page.close()
             except Exception as page_close_err: logger.warning(f"Error closing page for {league_url}: {page_close_err}")
        if browser:
            logger.info(f"Closing Playwright browser for {league_url} in finally block.")
            try:
                await browser.close()
                logger.info(f"Playwright browser closed successfully for {league_url}.")
            except Exception as close_err:
                logger.error(f"Error closing Playwright browser in finally block for {league_url}: {close_err}")

    # --- Process the extracted JSON data ---
    if not json_part or not isinstance(json_part, list):
        logger.error(f"Failed to extract or parse JSON data from network response for {league_url}")
        return None

    try:
        logger.debug(f"Processing {len(json_part)} matches from Understat network response for match {match_id} lookup.")
        home_team_lower = home_team.lower()
        away_team_lower = away_team.lower()
        match_date_simple = match_date_str

        for match_data in json_part:
            # Adjust keys based on actual JSON structure from network response if different
            if not all(k in match_data for k in ['datetime', 'h', 'a', 'id']):
                logger.warning(f"Skipping match data due to missing keys: {match_data}")
                continue
            match_dt_str = match_data['datetime']
            match_date_part = match_dt_str.split(' ')[0]
            h_team_name = match_data['h'].get('title', '').lower()
            a_team_name = match_data['a'].get('title', '').lower()

            if match_date_part == match_date_simple and h_team_name == home_team_lower and a_team_name == away_team_lower:
                found_match_id = match_data['id']
                logger.info(f"Found matching Understat match ID: {found_match_id} for original match {match_id}")
                absolute_url = f"https://understat.com/match/{found_match_id}"
                return absolute_url

        logger.warning(f"Could not find matching match in Understat league page data for match {match_id} ({home_team} vs {away_team} on {match_date_simple}).")
        return None
    except Exception as e:
        logger.error(f"Error parsing Understat league page for match {match_id}: {e}", exc_info=True)
        # Rotate proxy if fetch failed?
        proxy_manager.rotate_proxy()
        return None

async def _find_fbref_url(match_id, match_date_str, home_team, away_team, league_id, season, proxy_manager: ProxyManager) -> Optional[str]:
    """Finds the correct FBref match URL by parsing the schedule page."""
    league_mapping = {"PL": "9", "PD": "12", "BL1": "20", "SA": "11", "FL1": "13", "DED": "23"}
    fbref_league_id = league_mapping.get(league_id)
    if not fbref_league_id:
        logger.warning(f"No FBref league mapping for league_id: {league_id}. Cannot find URL for match {match_id}.")
        return None

    try:
        start_year = int(season)
        end_year = start_year + 1
        season_str = f"{start_year}-{end_year}"
    except ValueError:
        logger.error(f"Invalid season format '{season}' for FBref URL construction for match {match_id}.")
        return None

    schedule_url = f"https://fbref.com/en/comps/{fbref_league_id}/{season_str}/schedule/"
    logger.info(f"Attempting FBref lookup for match {match_id} via schedule page: {schedule_url}")

    proxy_str = proxy_manager.get_proxy()
    proxy_config = None
    if proxy_str:
        parsed_proxy = urlparse(proxy_str)
        proxy_config = {
            "server": f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}"
        }
        if parsed_proxy.username:
            proxy_config["username"] = parsed_proxy.username
        if parsed_proxy.password:
            proxy_config["password"] = parsed_proxy.password

    html_content = await _fetch_with_playwright(schedule_url, proxy_config=proxy_config) # Pass parsed proxy config

    if not html_content:
        logger.error(f"Failed to fetch FBref schedule page {schedule_url} for match {match_id}.")
        return None

    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        table_soup = None
        schedule_table_id_pattern = f'id="sched_{season_str}_'
        comment = soup.find(string=lambda text: isinstance(text, Comment) and schedule_table_id_pattern in text)

        if comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            schedule_table = comment_soup.find('table', id=lambda x: x and x.startswith(f'sched_{season_str}_'))
            if schedule_table: table_soup = schedule_table
            else: logger.error(f"Found comment but no table within it on FBref page: {schedule_url}")
        else:
            schedule_table = soup.find('table', id=lambda x: x and x.startswith(f'sched_{season_str}_'))
            if schedule_table:
                logger.warning(f"Found schedule table directly (not in comment) on {schedule_url}.")
                table_soup = schedule_table
            else:
                 logger.error(f"Could not find schedule table comment or table directly on FBref page: {schedule_url}")
                 return None

        if not table_soup or not table_soup.tbody:
             logger.error(f"Could not find table body for schedule on {schedule_url}")
             return None

        rows = table_soup.tbody.find_all('tr', class_=lambda x: x != 'spacer')
        logger.debug(f"Found {len(rows)} match rows in FBref schedule table for match {match_id} lookup.")

        home_team_lower = home_team.lower()
        away_team_lower = away_team.lower()
        match_date_simple = match_date_str

        for row in rows:
            date_cell = row.find('td', {'data-stat': 'date'})
            home_team_cell = row.find('td', {'data-stat': 'home_team'})
            away_team_cell = row.find('td', {'data-stat': 'away_team'})
            match_report_cell = row.find('td', {'data-stat': 'match_report'})

            if not all([date_cell, home_team_cell, away_team_cell, match_report_cell]): continue

            row_date_str = date_cell.get_text(strip=True)
            try:
                 row_date_simple = datetime.strptime(row_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
            except ValueError: continue

            row_home_team = home_team_cell.get_text(strip=True).lower()
            row_away_team = away_team_cell.get_text(strip=True).lower()

            if row_date_simple == match_date_simple and row_home_team == home_team_lower and row_away_team == away_team_lower:
                match_report_link = match_report_cell.find('a', href=True)
                if match_report_link:
                    relative_url = match_report_link['href']
                    absolute_url = urljoin("https://fbref.com", relative_url)
                    logger.info(f"Found matching FBref match report URL: {absolute_url} for original match {match_id}")
                    return absolute_url
                else: logger.warning(f"Found matching row for match {match_id} but no 'Match Report' link.")

        logger.warning(f"Could not find matching match in FBref schedule table for match {match_id} ({home_team} vs {away_team} on {match_date_simple}).")
        return None
    except Exception as e:
        logger.error(f"Error parsing FBref schedule page for match {match_id}: {e}", exc_info=True)
        # Rotate proxy if fetch failed?
        proxy_manager.rotate_proxy()
        return None


# --- Orchestration Function for Finding and Scraping ---
async def find_and_scrape(source_site: str, match_id: str, match_date_str: Optional[str], home_team: Optional[str], away_team: Optional[str], league_id: Optional[str], season: Optional[str]) -> Dict[str, Any]:
    """
    Finds the correct URL for the source site and scrapes/parses the data.
    """
    logger.info(f"Starting find_and_scrape for Match ID {match_id}, Source: {source_site}")
    final_url = None
    parsed_data = None

    # Initialize ProxyManager
    proxy_list = Config.get('scraping.proxy.providers', [])
    if not proxy_list:
        logger.warning("No proxies found in configuration (scraping.proxy.providers). Proceeding without proxies.")
    proxy_manager = ProxyManager(proxy_list=proxy_list)

    # Basic validation
    if not all([match_date_str, home_team, away_team, season, league_id]):
         logger.warning(f"Missing required details for find_and_scrape (Match ID: {match_id}, Source: {source_site}).")
         return {"source": source_site, "match_id": match_id, "status": "error", "error": "Missing required match details for URL finding."}

    # Find URL - Pass proxy_manager
    try:
        if source_site == "Transfermarkt":
            final_url = await _find_transfermarkt_url(match_id, match_date_str, home_team, away_team, season, proxy_manager)
        elif source_site == "Understat":
            final_url = await _find_understat_url(match_id, match_date_str, home_team, away_team, league_id, season, proxy_manager)
        elif source_site == "FBref":
            final_url = await _find_fbref_url(match_id, match_date_str, home_team, away_team, league_id, season, proxy_manager)
        else:
            logger.error(f"Unsupported source site '{source_site}' for find_and_scrape.")
            return {"source": source_site, "match_id": match_id, "status": "error", "error": f"Unsupported source site: {source_site}"}
    except Exception as find_e:
         logger.error(f"Error during URL finding for {source_site}, Match {match_id}: {find_e}", exc_info=True)
         return {"source": source_site, "match_id": match_id, "status": "error", "error": f"URL finding failed: {find_e}"}


    # Scrape and Parse if URL found
    if final_url:
        logger.info(f"Found URL for {source_site}, Match {match_id}: {final_url}. Fetching...")
        proxy_str = proxy_manager.get_proxy() # Get proxy for final fetch
        proxy_config = None
        if proxy_str:
            parsed_proxy = urlparse(proxy_str)
            proxy_config = {
                "server": f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}"
            }
            if parsed_proxy.username:
                proxy_config["username"] = parsed_proxy.username
            if parsed_proxy.password:
                proxy_config["password"] = parsed_proxy.password

        html_content = await _fetch_with_playwright(final_url, proxy_config=proxy_config) # Pass parsed proxy config
        if html_content:
            logger.info(f"Successfully fetched content for {final_url}. Parsing...")
            # Rotate proxy after successful fetch? Optional.
            # proxy_manager.rotate_proxy()
            parsed_data = parse_html_content(html_content, source_site, final_url)
            if not parsed_data: # Handle case where parse_html_content returns None
                 parsed_data = {"source": source_site, "url": final_url, "status": "error", "error": "Parsing function returned None."}
            # Add original match_id for context if not already present
            if 'match_id' not in parsed_data: parsed_data['match_id'] = match_id
            return parsed_data
        else:
            logger.error(f"Failed to fetch content from final URL: {final_url}")
            # Mark proxy as failed? Consider adding logic here or in ProxyManager
            # proxy_manager.mark_proxy_as_failed(proxy_str)
            return {"source": source_site, "match_id": match_id, "url": final_url, "status": "error", "error": "Failed to fetch content from found URL."}
    else:
        logger.warning(f"Could not find URL for {source_site}, Match {match_id}. No data scraped.")
        return {"source": source_site, "match_id": match_id, "status": "error", "error": "URL not found via search/lookup."}


# --- Fetching Logic (Standalone Crawler - kept for potential direct use) ---

async def _scrape_url(session: aiohttp.ClientSession, url: str, use_playwright: bool, proxy_manager: ProxyManager) -> Optional[Dict[str, Any]]:
    """
    Fetches a single URL using the appropriate method (aiohttp or Playwright).
    Includes basic parsing for standalone use.
    """
    proxy = proxy_manager.get_proxy()
    html_content = None
    fetch_method = "Playwright" if use_playwright else "aiohttp"
    source_name = "Unknown" # Determine source for parsing
    if "understat.com" in url: source_name = "Understat"
    elif "fbref.com" in url: source_name = "FBref"
    elif "transfermarkt.com" in url: source_name = "Transfermarkt"

    try:
        logger.info(f"Attempting to scrape {url} using {fetch_method}")
        if use_playwright:
            html_content = await _fetch_with_playwright(url, proxy)
        else:
            html_content = await _fetch_with_aiohttp(session, url, proxy)

        if html_content:
            parsed_data = parse_html_content(html_content, source_name, url)
            if parsed_data is None:
                parsed_data = {"source": source_name, "url": url, "status": "parsing_failed_or_no_parser"}

            parsed_data['raw_html'] = html_content # Include raw HTML for debugging

            if parsed_data.get('status') not in ['error', 'parsing_failed_or_no_parser']:
                 logger.info(f"Successfully parsed content from {url} (Source: {source_name})")
            else:
                 logger.error(f"Failed to parse content for {url} (Source: {source_name}). Parser returned: {parsed_data}")

            return parsed_data
        else:
            logger.warning(f"Failed to fetch content for {url} using {fetch_method}")
            return {"source": source_name, "url": url, "status": "fetch_failed", "fetch_method": fetch_method}
    except Exception as e:
        logger.error(f"Error scraping {url} using {fetch_method}: {e}", exc_info=True)
        return {"source": source_name, "url": url, "status": "scrape_exception", "error": str(e)}


async def _crawl_website(start_url: str, max_depth: int = 1, use_playwright: bool = True, data_dir: str = 'data'):
    """Crawls a website starting from a given URL (basic implementation)."""
    # ... (crawl logic remains largely the same, using _scrape_url) ...
    # Note: This crawl logic is separate from the find_and_scrape logic used by MCP
    results = []
    visited = set()
    queue = asyncio.Queue()
    await queue.put((start_url, 0))
    visited.add(start_url)

    proxy_list = Config.get('scraping.proxy.providers', [])
    if not proxy_list:
        logger.warning("No proxies found in configuration (scraping.proxy.providers). Proceeding without proxies.")
    proxy_manager = ProxyManager(proxy_list=proxy_list)

    # Explicit timeout for fetch context to ensure no indefinite hang
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=NETWORK_CONFIG.get('timeout', 30))) as session:
        active_tasks = set()
        max_concurrent = NETWORK_CONFIG.get('max_concurrent_requests', 10)

        while not queue.empty() or active_tasks:
            while not queue.empty() and len(active_tasks) < max_concurrent:
                url, depth = await queue.get()
                if depth > max_depth: continue
                logger.debug(f"Queueing crawl task for: {url} (depth {depth})")
                task = asyncio.create_task(_scrape_url(session, url, use_playwright, proxy_manager))
                active_tasks.add(task)
                task.add_done_callback(active_tasks.remove)

            if not active_tasks:
                 if queue.empty(): break
                 else: await asyncio.sleep(0.1); continue

            done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    data = task.result()
                    if data: results.append(data)
                    # Link extraction logic would go here if needed for deeper crawl
                except Exception as e:
                    logger.error(f"Crawl scraping task failed with error: {e}", exc_info=True)

    logger.info(f"Crawling finished for {start_url}. Found {len(results)} pages.")
    return results


def _consolidate_results(results: List[Dict[str, Any]], output_format: str = 'json', output_path: Optional[str] = None, data_dir: str = 'data') -> Optional[str]:
    """Consolidates scraping results into a specified format and file."""
    if not results:
        logger.warning("No results to consolidate.")
        return None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    default_filename = f"scraped_data_{timestamp}.{output_format}"
    # Resolve data_dir relative to project root
    absolute_data_dir = os.path.join(project_root, data_dir)
    scraped_data_dir = os.path.join(absolute_data_dir, 'scraped')
    # Handle output_path: if relative, join with scraped_data_dir, else use as is
    if output_path and not os.path.isabs(output_path):
         output_file = os.path.join(scraped_data_dir, output_path)
    elif output_path: # Absolute path provided
         output_file = output_path
    else: # Default filename
         output_file = os.path.join(scraped_data_dir, default_filename)


    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.info(f"Consolidating {len(results)} results to {output_file}")
        # Remove raw_html before saving if it exists and is large? Optional.
        results_to_save = []
        for res in results:
             # Create a copy to avoid modifying the original dict if needed elsewhere
             res_copy = res.copy()
             # Optionally remove large HTML content before saving JSON
             # if 'raw_html' in res_copy: del res_copy['raw_html']
             results_to_save.append(res_copy)

        if output_format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return None
        logger.info(f"Successfully saved results to {output_file}")
        return output_file
    except IOError as e:
        logger.error(f"Failed to write results to {output_file}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during consolidation: {e}", exc_info=True)
        return None

# --- Main Execution Logic ---

async def run_standalone_scrape(url: str, use_playwright: bool, output_format: str, output_path: Optional[str], max_depth: int, data_dir: str):
    """Main function for standalone scraping/crawling."""
    start_time = time.time()
    logger.info(f"Starting standalone scrape for URL: {url}")
    logger.info(f"Using Playwright: {use_playwright}")
    logger.info(f"Max crawl depth: {max_depth}")
    logger.info(f"Data directory (relative to project root): {data_dir}")

    absolute_data_dir = os.path.join(project_root, data_dir)
    os.makedirs(os.path.join(absolute_data_dir, 'scraped'), exist_ok=True)
    os.makedirs(os.path.join(absolute_data_dir, 'status'), exist_ok=True)

    results = await _crawl_website(url, max_depth=max_depth, use_playwright=use_playwright, data_dir=data_dir) # Pass relative data_dir

    consolidated_results_path = None
    if results:
        # Pass relative data_dir to consolidation as well
        consolidated_results_path = _consolidate_results(results, output_format, output_path, data_dir=data_dir)
    else:
        logger.warning(f"No data scraped from {url}.")

    # Flag file logic (optional for standalone)
    if consolidated_results_path:
        flag_dir = os.path.join(absolute_data_dir, 'status')
        flag_file = os.path.join(flag_dir, 'standalone_data_ready.flag')
        try:
            with open(flag_file, 'w', encoding='utf-8') as f:
                f.write(f"Data ready at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Consolidated file: {consolidated_results_path}\n")
            logger.info(f"Created data readiness flag: {flag_file}")
        except Exception as e:
            logger.error(f"Failed to create data readiness flag {flag_file}: {e}", exc_info=True)
    else:
        logger.warning("Skipping flag file creation as consolidation failed or produced no results.")

    end_time = time.time()
    logger.info(f"Standalone scrape finished in {end_time - start_time:.2f} seconds.")
    logger.info(f"Consolidated results saved to: {consolidated_results_path}" if consolidated_results_path else "No results saved.")


async def run_find_and_scrape(args):
    """Main function for the find_and_scrape mode called by MCP."""
    result = await find_and_scrape(
        source_site=args.source_site,
        match_id=args.match_id,
        match_date_str=args.match_date,
        home_team=args.home_team,
        away_team=args.away_team,
        league_id=args.league_id,
        season=args.season
    )
    # Print result as JSON to stdout for MCP server to capture
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Web Scraper & URL Finder")
    subparsers = parser.add_subparsers(dest='mode', help='Execution mode')

    # Subparser for standalone scraping/crawling
    parser_scrape = subparsers.add_parser('scrape', help='Run standalone scrape/crawl')
    parser_scrape.add_argument("url", help="The starting URL to scrape")
    parser_scrape.add_argument("--use-playwright", action='store_true', default=PLAYWRIGHT_CONFIG.get('enabled', True), help="Use Playwright for scraping")
    parser_scrape.add_argument("--no-playwright", action='store_false', dest='use_playwright', help="Do not use Playwright (use aiohttp)")
    parser_scrape.add_argument("--output-format", default="json", choices=['json'], help="Output format")
    parser_scrape.add_argument("--output-path", help="Specific path to save consolidated results (relative to data_dir/scraped if not absolute)")
    parser_scrape.add_argument("--max-depth", type=int, default=1, help="Maximum crawl depth")
    parser_scrape.add_argument("--data-dir", default=BASE_DATA_DIR, help=f"Base directory for data files (relative to project root). Default: '{BASE_DATA_DIR}'")

    # Subparser for find_and_scrape mode (called by MCP)
    parser_find = subparsers.add_parser('find_and_scrape', help='Find URL for a specific source and scrape it')
    parser_find.add_argument("--source-site", required=True, choices=['Transfermarkt', 'Understat', 'FBref'], help="Target data source site")
    parser_find.add_argument("--match-id", required=True, help="Original match ID (e.g., from Football-Data.org)")
    parser_find.add_argument("--match-date", required=True, help="Match date (YYYY-MM-DD)")
    parser_find.add_argument("--home-team", required=True, help="Home team name")
    parser_find.add_argument("--away-team", required=True, help="Away team name")
    parser_find.add_argument("--league-id", required=True, help="League ID/code (e.g., PL, PD)")
    parser_find.add_argument("--season", required=True, help="Season (starting year, e.g., 2023)")
    # Add --data-dir here too if find_and_scrape needs it (e.g., for screenshots)
    parser_find.add_argument("--data-dir", default=BASE_DATA_DIR, help=f"Base directory for data files (relative to project root). Default: '{BASE_DATA_DIR}'")


    args = parser.parse_args()

    # Update BASE_DATA_DIR if provided via args (affects screenshot path etc.)
    if hasattr(args, 'data_dir') and args.data_dir:
         BASE_DATA_DIR = args.data_dir

    if args.mode == 'scrape':
        # Adjust output path if relative
        final_output_path = args.output_path
        if final_output_path and not os.path.isabs(final_output_path):
             # Note: _consolidate_results now handles joining with data_dir/scraped
             pass # Path is handled within _consolidate_results

        asyncio.run(run_standalone_scrape(
            url=args.url,
            use_playwright=args.use_playwright,
            output_format=args.output_format,
            output_path=final_output_path, # Pass potentially relative path
            max_depth=args.max_depth,
            data_dir=args.data_dir # Pass relative data_dir
        ))
    elif args.mode == 'find_and_scrape':
         asyncio.run(run_find_and_scrape(args))
    else:
         # Should not happen if subparsers are defined correctly
         logger.error(f"Invalid or missing mode specified.")
         parser.print_help()
         sys.exit(1)

class EnhancedScraper:
    def __init__(self, db_manager: Optional[DatabaseManager] = None, use_playwright: bool = None):
        self.db_manager = db_manager or DatabaseManager()
        
        # Get proxy settings from configuration
        proxy_enabled = Config.get('scraping.proxy.enabled', False)
        proxy_providers = Config.get('scraping.proxy.providers', []) if proxy_enabled else []
        self.proxy_manager = ProxyManager(proxy_providers)
        
        # Playwright configuration with explicit toggle
        self.playwright_config = Config.get('scraping.playwright', {})
        # Allow explicitly overriding the configuration setting
        if use_playwright is not None:
            self.use_playwright = use_playwright
        else:
            self.use_playwright = self.playwright_config.get('enabled', False)
        self.request_timeout = self.playwright_config.get('timeout', 30000)
        self.max_retries = self.playwright_config.get('max_retries', 3)
        self.retry_delay = self.playwright_config.get('retry_delay', 2)
        self._browser = None
        self._context = None
        self._page = None
        # self._initialize_playwright() # Call removed from __init__

    async def initialize_playwright_components(self): # Renamed and will be called explicitly
        """Initialize Playwright browser and context with enhanced anti-detection measures."""
        # Skip initialization if use_playwright is False
        if not self.use_playwright:
            logger.info("Playwright initialization skipped as use_playwright=False")
            return True
            
        try:
            # Set PLAYWRIGHT_BROWSERS_PATH to a custom directory
            # project_root is defined at the top of the file. os is imported.
            custom_browsers_path = os.path.join(project_root, "pw_browsers")
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = custom_browsers_path
            logger.info(f"Setting PLAYWRIGHT_BROWSERS_PATH to: {custom_browsers_path}")

            # Diagnostic logging for event loop and policy
            logger.info(f"Attempting to initialize Playwright components...")
            logger.info(f"Current asyncio event loop policy: {type(asyncio.get_event_loop_policy())}")
            try:
                current_loop = asyncio.get_running_loop()
                logger.info(f"Current running asyncio event loop: {type(current_loop)}")
            except RuntimeError:
                logger.info("No asyncio event loop currently running in this context (before Playwright start).")

            # Add a timeout for Playwright initialization to prevent it from blocking the app indefinitely
            try:
                playwright = await asyncio.wait_for(async_playwright().start(), timeout=30.0)
                
                # Enhanced browser options for better anti-detection
                browser_options = {
                    'headless': self.playwright_config.get('headless', True),
                    'args': [
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-gpu',
                        '--disable-software-rasterizer',
                        '--disable-blink-features=AutomationControlled',
                        '--disable-features=IsolateOrigins,site-per-process',
                        '--disable-site-isolation-trials'
                    ]
                }
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"Failed to initialize Playwright within timeout: {e}")
                # Mark Playwright as disabled to prevent further attempts
                self.use_playwright = False
                return False
            
            # Add proxy if available
            proxy_str = self.proxy_manager.get_proxy()
            if proxy_str:
                parsed_proxy = urlparse(proxy_str)
                browser_options['proxy'] = {
                    'server': f"{parsed_proxy.scheme}://{parsed_proxy.hostname}:{parsed_proxy.port}"
                }
                if parsed_proxy.username:
                    browser_options['proxy']['username'] = parsed_proxy.username
                if parsed_proxy.password:
                    browser_options['proxy']['password'] = parsed_proxy.password

            self._browser = await playwright.chromium.launch(**browser_options)
            
            # Enhanced context options
            user_agents = Config.get('scraping.request.user_agents', [])
            context_options = {
                'viewport': {'width': 1920, 'height': 1080},
                'user_agent': user_agents[0] if user_agents else None,
                'locale': 'en-US',
                'timezone_id': 'America/New_York',
                'geolocation': {'latitude': 40.7128, 'longitude': -74.0060},
                'permissions': ['geolocation'],
                'color_scheme': 'dark',
                'reduced_motion': 'no-preference',
                'forced_colors': 'none'
            }
            
            self._context = await self._browser.new_context(**context_options)
            self._page = await self._context.new_page()
            
            # Add enhanced stealth scripts
            await self._add_enhanced_stealth_scripts()
            
            # Set up request interception for better control
            await self._setup_request_interception()
            
            logger.info("Playwright initialized successfully with enhanced anti-detection measures")
        except Exception as e:
            logger.error(f"Failed to initialize Playwright: {e}")
            raise

    async def _add_enhanced_stealth_scripts(self):
        """Add comprehensive stealth scripts to make browser behavior more human-like."""
        stealth_scripts = [
            # Hide automation
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            """,
            
            # Spoof Chrome properties
            """
            window.chrome = {
                runtime: {},
                loadTimes: function() {},
                csi: function() {},
                app: {}
            };
            """,
            
            # Add fake permissions
            """
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            """,
            
            # Add fake WebGL
            """
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                if (parameter === 37445) return 'Intel Open Source Technology Center';
                if (parameter === 37446) return 'Mesa DRI Intel(R) Ivybridge Mobile';
                return getParameter(parameter);
            };
            """,
            
            # Add fake media devices
            """
            Object.defineProperty(navigator, 'mediaDevices', {
                get: () => ({
                    enumerateDevices: () => Promise.resolve([
                        { kind: 'audioinput', deviceId: 'default', label: 'Default Microphone' },
                        { kind: 'videoinput', deviceId: 'default', label: 'Default Camera' }
                    ])
                })
            });
            """
        ]
        
        for script in stealth_scripts:
            await self._page.add_init_script(script)

    async def _setup_request_interception(self):
        """Set up request interception for better control over network requests."""
        await self._page.route("**/*", self._handle_route)

    async def _handle_route(self, route):
        """Handle intercepted requests with enhanced control."""
        request = route.request
        
        # Block unnecessary resources
        blocked_resources = ['image', 'stylesheet', 'font', 'media']
        if request.resource_type in blocked_resources:
            await route.abort()
            return
            
        # Add custom headers
        headers = {
            **request.headers,
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Continue with modified headers
        await route.continue_(headers=headers)

    async def _add_human_behavior(self):
        """Add human-like behavior to page interactions."""
        try:
            # Random mouse movements
            await self._page.mouse.move(
                random.randint(0, 1920),
                random.randint(0, 1080),
                steps=random.randint(5, 10)
            )
            
            # Random scrolling
            await self._page.evaluate("""
                window.scrollTo({
                    top: Math.random() * document.body.scrollHeight,
                    behavior: 'smooth'
                });
            """)
            
            # Random delays
            await asyncio.sleep(random.uniform(1, 3))
            
        except Exception as e:
            logger.warning(f"Error adding human behavior: {e}")

    async def fetch_multi_source(self, match_id: str, match_date_str: str, 
                               home_team: str, away_team: str, 
                               league_id: Optional[str] = None, 
                               season: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch data from multiple sources with enhanced error handling, retry logic, and fallback to secondary sources.
        Logs which sources succeed/fail for each match.
        """
        results = {}
        sources_config = Config.get('scraping.sources', {})
        primary_sources = sources_config.get('primary', [])
        secondary_sources = sources_config.get('secondary', [])
        all_sources = primary_sources + secondary_sources
        source_status = {}

        for source in all_sources:
            source_name = source['name']
            retry_count = 0
            success = False

            while retry_count < self.max_retries:
                try:
                    # Add human-like behavior before fetching
                    await self._add_human_behavior()

                    # Find and scrape data
                    data = await find_and_scrape(
                        source_site=source_name,
                        match_id=match_id,
                        match_date_str=match_date_str,
                        home_team=home_team,
                        away_team=away_team,
                        league_id=league_id,
                        season=season
                    )

                    # Validate data
                    if data and await self._validate_data(data, source_name):
                        results[source_name] = data
                        source_status[source_name] = 'success'
                        success = True
                        break
                    else:
                        logger.warning(f"Data validation failed for {source_name}, attempt {retry_count + 1}")
                        retry_count += 1
                        await asyncio.sleep(self.retry_delay * retry_count)

                except Exception as e:
                    logger.error(f"Error fetching from {source_name}: {e}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        await asyncio.sleep(self.retry_delay * retry_count)
                        # Rotate proxy on failure
                        self.proxy_manager.rotate_proxy()
                    else:
                        results[source_name] = {
                            'status': 'error',
                            'error': str(e),
                            'source': source_name
                        }
                        source_status[source_name] = 'fail'

            # If a primary source fails, try the next one; after all primaries, try secondaries
            if not success and source in primary_sources:
                logger.info(f"Primary source {source_name} failed for match {match_id}, will try secondary sources if available.")

        # Log the status of all sources for this match
        logger.info(f"Source status for match {match_id}: {source_status}")

        # If all sources fail, log a warning
        if not any(status == 'success' for status in source_status.values()):
            logger.warning(f"All sources failed for match {match_id}. No data available.")

        return results

    async def _validate_data(self, data, source_name):
        """
        Placeholder for data validation logic. Can be expanded to check schema, required fields, etc.
        """
        # Example: check for required fields from config
        sources_config = Config.get('scraping.sources', {})
        all_sources = sources_config.get('primary', []) + sources_config.get('secondary', [])
        required_fields = None
        for src in all_sources:
            if src['name'] == source_name:
                required_fields = src.get('required_fields', [])
                break
        if required_fields:
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Validation failed: {field} missing in data from {source_name}")
                    return False
        return True

    async def _solve_captcha(self, page):
        """
        Placeholder for CAPTCHA solving integration (e.g., 2Captcha, AntiCaptcha).
        """
        logger.info("CAPTCHA detected. Placeholder for CAPTCHA solving integration.")
        # TODO: Integrate with a CAPTCHA solving service
        await asyncio.sleep(2)  # Simulate delay
        return True

    async def close(self):
        """Clean up Playwright resources."""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            logger.info("Playwright resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up Playwright resources: {e}")
