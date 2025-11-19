import asyncio  # Keep asyncio fix
import logging
import os
import random
# Added performance optimizations and GPU integration
import sys
from typing import Any, Dict, Optional

import torch  # Added for GPU checks
from fake_useragent import UserAgent
from playwright.sync_api import Browser, BrowserContext
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Playwright, sync_playwright
from playwright_stealth import stealth

# Fix for Playwright NotImplementedError on Windows with asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logger = logging.getLogger(__name__)

# Import ProxyManager if needed for type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .proxy_manager import ProxyManager

class PlaywrightManager:
    def __init__(self, proxy_manager: Optional['ProxyManager'] = None):
        self.proxy_manager = proxy_manager # Keep proxy manager reference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"PlaywrightManager initialized. Device set to: {self.device}")
        # Initialize UserAgent for random user agents (keep this part)
        try:
            self.ua = UserAgent()
        except Exception as e:
            logger.warning(f"Failed to initialize UserAgent for Playwright, using fallback: {e}")
            self.ua = None
        # Keep browser/playwright as None initially, init on demand
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.use_fallback = False  # Add fallback flag
        self._init_browser() # Initialize browser on creation

    def _get_random_user_agent(self):
        """Returns a random user agent. Uses fake_useragent if available."""
        if self.ua:
            try:
                return self.ua.random
            except Exception as e:
                logger.warning(f"Failed to get random UA from fake-useragent: {e}. Using fallback.")
        # Fallback list if UserAgent failed or not initialized
        fallback_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        return random.choice(fallback_agents)

    def _init_browser(self):
        """Initialize browser with GPU acceleration if available"""
        if self.browser or self.use_fallback: # Also check if fallback is already active
            return # Already initialized or fallback mode

        logger.info(f"Initializing browser with device: {self.device}")
        try:
            # --- Outer Try Block for Overall Playwright Init ---
            # Explicitly catch NotImplementedError during Playwright start
            try:
                self.playwright = sync_playwright().start()
            except NotImplementedError:
                # Catch the error here, log, set fallback, and RETURN
                logger.warning("Playwright start() raised NotImplementedError. Enabling requests-based fallback.")
                self.playwright = None
                self.browser = None
                self.use_fallback = True
                return # Exit initialization, proceed in fallback mode

            if not self.playwright: # Check if start failed for other reasons
                 logger.error("sync_playwright().start() returned None or failed unexpectedly.")
                 # If start fails for other reasons, enable fallback and return
                 self.playwright = None
                 self.browser = None
                 self.use_fallback = True
                 logger.warning("Enabled requests-based fallback due to unexpected Playwright start failure.")
                 return

            browser_type = self.playwright.chromium

            # Set browser launch options with fallbacks and GPU support
            launch_options = {
                'headless': True,
                'args': [
                    '--disable-blink-features=AutomationControlled',
                    # Conditionally add GPU arg only if cuda is available
                    *(['--enable-gpu-rasterization'] if self.device == "cuda" else [])
                ]
            }
            
            # Add proxy if available
            if self.proxy_manager and self.proxy_manager.get_current_proxy():
                launch_options['proxy'] = {
                    'server': self.proxy_manager.get_current_proxy()
                }
                
            try:
                logger.debug(f"Launching Chromium with options: {launch_options}")
                self.browser = browser_type.launch(**launch_options)
            except Exception as browser_error:
                # Fallback without proxy if that was the issue
                logger.warning(f"Failed to launch with proxy: {str(browser_error)}")
                launch_options.pop('proxy', None)
                logger.debug(f"Retrying launch with options: {launch_options}")
                self.browser = browser_type.launch(**launch_options)
            
            logger.info("Playwright browser initialized successfully.")

        # Note: The original NotImplementedError except block is now handled within the inner try/except
        except PlaywrightError as e:
            logger.error(f"Failed to initialize Playwright browser: {e}")
            self.close() # Attempt cleanup
            raise # Re-raise the error
        except Exception as e: # Catch other potential errors during init
            logger.error(f"Unexpected error during Playwright initialization: {e}", exc_info=True)
            self.close() # Attempt cleanup if other errors occur
            # This outer block might be less likely to be hit now, but keep for safety
            logger.error(f"Failed to initialize Playwright browser: {e}")
            self.close() # Attempt cleanup
            # Enable fallback if PlaywrightError occurs during launch etc.
            self.use_fallback = True
            logger.warning("Enabled requests-based fallback due to PlaywrightError during initialization.")
            # raise # Optionally re-raise if critical

        except Exception as e: # Catch other potential errors during init
            logger.error(f"Unexpected error during Playwright initialization: {e}", exc_info=True)
            self.close() # Attempt cleanup
            # Consider setting fallback=True here too, or re-raising depending on desired behavior
            self.use_fallback = True # Enable fallback on any init error
            logger.warning("Enabled requests-based fallback due to unexpected initialization error.")
            # raise # Optionally re-raise if the error is critical

    # Added GPU-accelerated rendering support
    def _get_new_page(self) -> Page:
        """Creates and returns a new browser page with stealth and device-specific options."""
        if self.use_fallback:
             raise RuntimeError("Cannot get Playwright page when in fallback mode.")

        if not self.browser:
            self._init_browser() # Initialize if not already done

        # Check again after init attempt, including fallback status
        if self.use_fallback or not self.browser:
             raise RuntimeError("Playwright browser could not be initialized or is in fallback mode.")

        logger.debug("Creating new Playwright page with enhanced context...")

        context_options: Dict[str, Any] = {
            'device_scale_factor': 1.0 if self.device == "cpu" else 2.0, # Use device
            'user_agent': self._get_random_user_agent(), # Use the method
            'viewport': {'width': 1920, 'height': 1080}, # Standard viewport
            'java_script_enabled': True,
            'ignore_https_errors': True,
        }

        if self.device == "cuda":
            context_options.update({
                'screen': {'width': 3840, 'height': 2160}, # Higher res screen for GPU
                'has_touch': False # Typically false for desktop GPU setups
            })
            logger.debug("Applying CUDA specific context options.")

        # Integrate proxy manager logic (similar to original)
        if self.proxy_manager and self.proxy_manager.is_available():
            pw_proxy_dict = self.proxy_manager.get_playwright_proxy_dict()
            if pw_proxy_dict:
                 context_options["proxy"] = pw_proxy_dict
                 logger.info(f"Creating new browser context with proxy: {pw_proxy_dict.get('server')}")
            else:
                 logger.debug("Proxy manager available but returned no valid proxy dict for Playwright.")
        else:
             logger.debug("No proxy manager or proxies available. Creating context without proxy.")

        context = self.browser.new_context(**context_options)

        # Apply init scripts (keep this for stealth)
        try:
            context.add_init_script("""
                Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 4 });
                const originalQuery = navigator.permissions.query;
                navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
                if (navigator.userAgent.includes('Chrome')) {
                    window.chrome = { runtime: {}, app: {} };
                }
            """)
            logger.debug("Applied custom init scripts for fingerprinting.")
        except Exception as init_script_err:
            logger.warning(f"Failed to add custom init scripts: {init_script_err}")

        page = context.new_page()

        # Apply stealth to the page (keep this)
        try:
            # Apply stealth plugin to avoid detection
            stealth(page)
            logger.debug("Applied playwright-stealth to the page.")
        except Exception as stealth_err:
            logger.warning(f"Failed to apply playwright-stealth: {stealth_err}")

        return page

    def fetch_content(self, url: str, wait_for_selector: Optional[str] = None, timeout: int = 60000) -> str: # Increased default timeout
        """
        Fetches HTML content from a URL using Playwright with stealth (synchronous).

        Args:
            url (str): The URL to fetch.
            wait_for_selector (Optional[str]): A CSS selector to wait for before returning content.
            timeout (int): Navigation timeout in milliseconds.

        Returns:
            str: The fetched HTML content.

        Raises:
            PlaywrightError: If any Playwright operation fails.
            Exception: For other unexpected errors.
        """
        page = None # Initialize page to None
        context = None # Initialize context to None
        try:
            page = self._get_new_page()
            context = page.context # Get context for potential later use

            # Block unnecessary resources (images, css, fonts)
            try:
                page.route("**/*.{png,jpg,jpeg,gif,webp,svg,css,woff,woff2}", lambda route: route.abort())
                logger.debug(f"Set up resource blocking for {url}")
            except Exception as route_err:
                logger.warning(f"Could not set up resource blocking for {url}: {route_err}")


            logger.info(f"Navigating to {url} with timeout {timeout}ms...")
            # Add a common referer
            page.set_extra_http_headers({"Referer": "https://www.google.com/"})

            # Simulate some mouse movement before navigation
            page.mouse.move(random.randint(100, 500), random.randint(100, 500))

            # Use 'domcontentloaded' or 'load' instead of 'networkidle' which can be flaky
            page.goto(url, wait_until='domcontentloaded', timeout=timeout)
            logger.info(f"Navigation to {url} complete (DOM loaded).")

            # Add a small random delay after load
            page.wait_for_timeout(random.randint(1500, 4000))

            if wait_for_selector:
                logger.info(f"Waiting for selector: '{wait_for_selector}'...")
                # Use a reasonable default timeout for waiting for selector
                page.wait_for_selector(wait_for_selector, timeout=max(15000, timeout // 3)) # Wait at least 15s or 1/3 of nav timeout
                logger.info(f"Selector '{wait_for_selector}' found.")
                # Add another small delay after finding selector
                page.wait_for_timeout(random.randint(500, 1500))

            logger.info(f"Retrieving page content for {url}...")
            content = page.content()
            logger.info(f"Successfully retrieved content for {url} (length: {len(content)}).")
            return content
        except PlaywrightError as e:
             logger.error(f"Playwright error fetching {url}: {e}")
             raise # Re-raise Playwright specific errors
        except Exception as e:
             logger.error(f"Unexpected error fetching {url}: {e}", exc_info=True)
             raise # Re-raise other errors
        finally:
            # Close the specific context and its page
            if context:
                 try:
                      context.close() # Close context instead of just page
                      logger.debug(f"Closed Playwright context for {url}.")
                 except PlaywrightError as e:
                      logger.warning(f"Error closing Playwright context for {url}: {e}")
            elif page: # Fallback if context wasn't obtained
                 try:
                      page.close()
                      logger.debug(f"Closed Playwright page (context unavailable) for {url}.")
                 except PlaywrightError as e:
                      logger.warning(f"Error closing Playwright page for {url}: {e}")

    # Added batch processing capability
    @torch.inference_mode()
    def fetch_batch(self, urls: list, wait_for_selectors: Optional[list] = None) -> dict:
        """
        Process multiple URLs in batch with GPU optimization.

        Args:
            urls (list): A list of URLs to fetch.
            wait_for_selectors (Optional[list]): A list of CSS selectors corresponding to each URL.
                                                 If provided, must have the same length as urls.
                                                 Use None for URLs without a specific selector.

        Returns:
            dict: A dictionary where keys are URLs and values are the fetched HTML content or None on error.
        """
        results = {}
        if wait_for_selectors and len(urls) != len(wait_for_selectors):
            raise ValueError("Length of urls and wait_for_selectors must match if selectors are provided.")

        # Determine if selectors are provided for each URL
        selectors_to_use = wait_for_selectors if wait_for_selectors else [None] * len(urls)

        logger.info(f"Starting batch fetch for {len(urls)} URLs. Device: {self.device}")
        # Enable AMP autocasting if on CUDA for potential performance gains (though less relevant for I/O bound tasks)
        with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
            for i, url in enumerate(urls):
                selector = selectors_to_use[i]
                try:
                    # Assuming fetch_content handles errors internally and returns content or raises
                    # We might need to adjust fetch_content or handle exceptions here
                    logger.debug(f"Fetching URL {i+1}/{len(urls)}: {url} with selector '{selector}'")
                    results[url] = self.fetch_content(url, selector) # Use existing fetch_content
                except Exception as e:
                    logger.error(f"Error fetching {url} in batch: {e}")
                    results[url] = None # Indicate failure for this URL

        logger.info(f"Batch fetch completed. Results obtained for {sum(1 for r in results.values() if r is not None)}/{len(urls)} URLs.")
        return results

    def get_page_content(self, url: str) -> str:
        """
        Get content from a URL using Playwright with fallback to requests if needed.
        
        Args:
            url (str): The URL to fetch content from
            
        Returns:
            str: The page content
        """
        if not self.use_fallback:
            # Use Playwright if available
            try:
                logger.info(f"Fetching content with Playwright from {url}")
                page = self._get_new_page()  # Use our enhanced page creation
                try:
                    page.goto(url, wait_until="domcontentloaded")
                    content = page.content()
                    return content
                except Exception as e:
                    logger.error(f"Error using Playwright: {str(e)}")
                    # Fall through to fallback method
                finally:
                    if page:
                        try:
                            page.close()
                        except Exception as e:
                            logger.warning(f"Error closing page: {e}")
            except Exception as e:
                logger.error(f"Failed to create Playwright page: {e}")
                # Fall through to fallback
        
        # Fallback to requests
        logger.info(f"Using requests fallback for {url}")
        try:
            import requests
            headers = {
                'User-Agent': self._get_random_user_agent()  # Use our random UA method
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise for bad status codes
            return response.text
        except Exception as e:
            logger.error(f"Request fallback failed for {url}: {e}")
            raise  # Re-raise the error after logging

    def close(self):
        """Closes the Playwright browser and stops the Playwright instance (synchronous)."""
        if self.browser:
            try:
                logger.info("Closing Playwright browser...")
                self.browser.close()
                logger.info("Playwright browser closed.")
            except PlaywrightError as e:
                 logger.error(f"Error closing Playwright browser: {e}")
            finally:
                 self.browser = None

        if self.playwright:
            try:
                logger.info("Stopping Playwright instance...")
                self.playwright.stop()
                logger.info("Playwright instance stopped.")
            except Exception as e: # Catch broader exceptions during stop
                 logger.error(f"Error stopping Playwright instance: {e}")
            finally:
                 self.playwright = None
