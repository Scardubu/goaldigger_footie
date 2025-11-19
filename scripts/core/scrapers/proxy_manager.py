import logging
import os
import random
import time
from collections import deque  # Added for efficient rotation
from datetime import datetime, timedelta  # Added for failure tracking
from typing import Deque, Dict, List, Optional, Tuple  # Updated typing
from urllib.parse import urlparse

import requests

# Import default proxies
from scripts.core.scrapers.default_proxies import get_default_proxies

# Import the central performance monitor
from scripts.core.monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)

# Configuration for free proxy fetching (Consider moving to config.yaml later)
FREE_PROXY_API_URL = "https://api.proxyscrape.com/v2/?request=getproxies&protocol=http&timeout=10000&country=all&ssl=all&anonymity=elite"
FREE_PROXY_FETCH_TIMEOUT = 20 # Increased timeout slightly
DEFAULT_FAILURE_TIMEOUT_SECONDS = 300 # Avoid failed proxy for 5 minutes
DEFAULT_MAX_FAILURES = 3 # Temporarily remove after 3 failures

class ProxyManager:
    """
    Manages a list of proxies, providing round-robin selection,
    failure tracking, and optional fetching from free sources.
    """

    def __init__(
        self,
        proxy_list_str: Optional[str] = None,
        failure_timeout_seconds: int = DEFAULT_FAILURE_TIMEOUT_SECONDS,
        max_failures: int = DEFAULT_MAX_FAILURES
    ):
        """
        Initializes the ProxyManager.

        Args:
            proxy_list_str (Optional[str]): Comma-separated string of proxies.
                                            Loads from PROXY_LIST env var if None.
            failure_timeout_seconds (int): Duration to avoid a proxy after failure.
            max_failures (int): Number of failures before temporarily removing a proxy.
        """
        self.proxy_deque: Deque[str] = deque()
        # Stores (failure_count, last_failure_time)
        self.failed_proxies: Dict[str, Tuple[int, datetime]] = {}
        self.failure_timeout = timedelta(seconds=failure_timeout_seconds)
        self.max_failures = max_failures
        # Added performance metrics
        self.performance_metrics = {
            'success_count': 0,
            'failure_count': 0,
            'avg_response_time': 0.0 # Initialize with 0
        }
        self._load_proxies(proxy_list_str)
        self._perf_monitor = self._get_perf_monitor() # Initialize monitor instance

    def _get_perf_monitor(self):
        # Singleton pattern for PerformanceMonitor
        # Ensure this doesn't create circular dependencies if Monitor imports Manager
        # It seems okay as Monitor doesn't import Manager.
        if not hasattr(self, '_perf_monitor_instance'):
             # Check if an instance exists globally or create one
             # This simple approach assumes a single PerformanceMonitor instance is desired
             # A more robust singleton pattern might be needed in complex scenarios
             try:
                  # Attempt to get existing instance if available (e.g., passed in or global)
                  # For simplicity here, we create one if not found.
                  self._perf_monitor_instance = PerformanceMonitor()
             except Exception as e:
                  logger.error(f"Failed to get or create PerformanceMonitor instance: {e}")
                  # Fallback to a dummy object that does nothing?
                  class DummyMonitor:
                      def update(self, *args, **kwargs): pass
                  self._perf_monitor_instance = DummyMonitor()
        return self._perf_monitor_instance


    def _fetch_free_proxies(self) -> List[str]:
        """Fetches a list of free proxies from an online source."""
        logger.info(f"Attempting to fetch free proxies from {FREE_PROXY_API_URL}...")
        fetched_proxies = []
        try:
            response = requests.get(FREE_PROXY_API_URL, timeout=FREE_PROXY_FETCH_TIMEOUT)
            response.raise_for_status()
            # Proxies are typically newline-separated host:port
            raw_list = response.text.strip().split('\n')
            fetched_proxies = [f"http://{p.strip()}" for p in raw_list if p.strip()] # Assume http
            if fetched_proxies:
                logger.info(f"Successfully fetched {len(fetched_proxies)} free proxies.")
            else:
                logger.warning("Fetched free proxy list was empty.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch free proxies: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching free proxies: {e}", exc_info=True)
        return fetched_proxies

    def _parse_and_validate_proxies(self, raw_proxies: List[str], source: str) -> List[str]:
        """Parses a list of raw proxy strings and performs basic validation."""
        valid_proxies: List[str] = []
        seen_proxies = set() # Avoid duplicates

        for p in raw_proxies:
            proxy_str = p.strip()
            if not proxy_str:
                continue

            if "://" not in proxy_str:
                # logger.debug(f"Proxy '{proxy_str}' missing scheme. Assuming http://.")
                proxy_str = f"http://{proxy_str}"

            try:
                parsed = urlparse(proxy_str)
                # Basic validation: Check if it parses reasonably and has host/port
                if not parsed.hostname or not parsed.port or parsed.scheme not in ['http', 'https', 'socks4', 'socks5']:
                    logger.warning(f"Invalid or unsupported proxy format/scheme '{proxy_str}' from {source}. Skipping.")
                    continue
            except Exception as parse_err:
                logger.warning(f"Failed to parse proxy URL '{proxy_str}' from {source}: {parse_err}. Skipping.")
                continue # Skip if parsing fails

            if proxy_str not in seen_proxies:
                valid_proxies.append(proxy_str)
                seen_proxies.add(proxy_str)
            else:
                logger.debug(f"Skipping duplicate proxy: {proxy_str}")

        return valid_proxies

    def _load_proxies(self, proxy_list_str: Optional[str]):
        """
        Loads proxies from the provided string, environment variable,
        or fetches free proxies. Populates the internal deque.
        """
        # First try loading from the provided string
        if proxy_list_str:
            logger.info("Loading proxies from provided string")
            raw_proxies = proxy_list_str.split(",")
            validated_proxies = self._parse_and_validate_proxies(raw_proxies, "provided")
        else:
            # Then try environment variable
            env_proxies = os.getenv("PROXY_LIST")
            if env_proxies:
                logger.info("Loading proxies from PROXY_LIST environment variable")
                raw_proxies = env_proxies.split(",")
                validated_proxies = self._parse_and_validate_proxies(raw_proxies, "env")
            else:
                # Try fetching from free proxy source
                logger.info("No proxies provided. Attempting to fetch free proxies...")
                validated_proxies = self._fetch_free_proxies()
                
                # Use default proxies as fallback if free proxy fetching failed
                if not validated_proxies:
                    logger.warning("Free proxy fetching failed. Using default proxies as fallback...")
                    default_proxies = get_default_proxies()
                    validated_proxies = self._parse_and_validate_proxies(default_proxies, "default")
                
        if not validated_proxies:
            logger.error("Proxy pool is empty after initialization.")
            return
        
        # Populate the deque with validated proxies
        logger.info(f"Loaded {len(validated_proxies)} proxies successfully")
        self.proxy_deque = deque(validated_proxies)
        
        # Perform initial health check to verify proxies
        logger.info("Performing initial health check on proxy pool...")
        self.health_check()
        
        if not self.proxy_deque:
            logger.warning("Proxy pool is empty after health check. Adding default proxies...")
            default_proxies = get_default_proxies()
            validated_defaults = self._parse_and_validate_proxies(default_proxies, "default")
            if validated_defaults:
                self.proxy_deque = deque(validated_defaults)
                logger.info(f"Added {len(validated_defaults)} default proxies to pool")
            else:
                logger.error("Default proxy pool is empty after validation.")
                return
        
        # Add a direct connection (no proxy) as a fallback
        # This ensures that the proxy manager always has at least one option
        if not self.proxy_deque:
            # Add a special 'direct' proxy that the get_proxy function will recognize to skip proxy usage
            self.proxy_deque.append("direct://connection")
            logger.warning("No external proxies available. Added direct connection as fallback.")

        # Output result of proxy loading attempt
        if len(self.proxy_deque) <= 1 and "direct://connection" in self.proxy_deque:
            logger.warning("Using direct connections only (no proxies).")
        else:
            logger.info(f"Proxy manager initialized with {len(self.proxy_deque)} proxies")
            # For debugging only - don't log all proxies in production
            logger.debug(f"Available proxies: {list(self.proxy_deque)}")

    def _calculate_proxy_score(self, proxy: str) -> float:
        """Calculate proxy score based on historical performance"""
        failures, last_fail = self.failed_proxies.get(proxy, (0, None)) # Default to 0 failures, no last fail time

        # Avoid division by zero if failure_count is 0
        total_requests = self.performance_metrics['success_count'] + self.performance_metrics['failure_count']
        success_rate = self.performance_metrics['success_count'] / max(1, total_requests) # Overall success rate as base

        # Penalize based on individual proxy failures
        failure_penalty = failures * 0.2 # Simple penalty per failure

        # Factor in time since last failure (higher score if failed long ago or never)
        time_factor = 1.0
        if last_fail:
            time_since_fail = (datetime.now() - last_fail).total_seconds()
            # Normalize time factor: 0 if failed right now, 1 if timeout period has passed
            time_factor = min(1.0, time_since_fail / max(1, self.failure_timeout.total_seconds()))

        # Combine factors: Higher success rate and longer time since failure are better
        # Lower failure count is better
        score = (success_rate * 0.5) + (time_factor * 0.5) - failure_penalty
        # logger.debug(f"Score for {proxy}: SR={success_rate:.2f}, TF={time_factor:.2f}, FP={failure_penalty:.2f} -> Score={score:.3f}")
        return score

    def get_proxy(self) -> Optional[str]:
        """
        Get the next available proxy using round-robin and failure avoidance.
        Returns the proxy URL string or None if no suitable proxy is found.
        """
        # Return None if no proxies are loaded
        if not self.proxy_deque:
            logger.error("Proxy pool is empty after reset.")
            return None

        # Try to find a working proxy - may go through the entire deque
        for _ in range(len(self.proxy_deque)):
            # Get next proxy with round-robin
            proxy = self.proxy_deque.popleft()
            self.proxy_deque.append(proxy)  # Re-add to end for round-robin

            # If this is our special direct connection marker, return None to signal no proxy use
            if proxy == "direct://connection":
                return None

            # Check if proxy is in the failure list and if we should skip it
            failure_info = self.failed_proxies.get(proxy)
            if failure_info:
                failure_count, last_failure_time = failure_info
                time_since_failure = datetime.now() - last_failure_time

                # Skip if this proxy has failed recently and timeout hasn't expired
                if time_since_failure < self.failure_timeout:
                    continue

                # Skip if this proxy has failed too many times and doesn't get another chance
                if failure_count >= self.max_failures:
                    continue

            # This proxy seems good
            return proxy

        # If we've gone through every proxy and all are in the failure list
        # Return direct connection as absolute fallback
        return None

    def mark_failed(self, proxy_url: str):
        """Marks a proxy as failed, incrementing its failure count and timestamp."""
        if not proxy_url:
            return
        now = datetime.now()
        fail_count, _ = self.failed_proxies.get(proxy_url, (0, None))
        fail_count += 1
        self.failed_proxies[proxy_url] = (fail_count, now)
        logger.warning(f"Marked proxy {proxy_url} as failed ({fail_count} failures). Last failure: {now}.")
        # Optional: Could remove from deque here if fail_count >= self.max_failures
        # Note: The new get_proxy logic handles temporary removal based on timeout

    # Added latency monitoring
    def update_performance(self, proxy: str, success: bool, response_time: float):
        """Update proxy performance metrics"""
        if not proxy: # Cannot update performance for None proxy
            return

        total_requests_before = self.performance_metrics['success_count'] + self.performance_metrics['failure_count']

        if success:
            self.performance_metrics['success_count'] += 1
            # Update average response time using exponential moving average (EMA)
            # alpha = 0.1 # Smoothing factor
            # current_avg = self.performance_metrics['avg_response_time']
            # if total_requests_before == 0: # First successful request
            #     new_avg = response_time
            # else:
            #     new_avg = (alpha * response_time) + ((1 - alpha) * current_avg)
            # self.performance_metrics['avg_response_time'] = new_avg
            # logger.debug(f"Updated performance for {proxy}: SUCCESS, time={response_time:.3f}s. New avg time: {new_avg:.3f}s")

            # Simpler rolling average for now
            total_success = self.performance_metrics['success_count']
            current_total_time = self.performance_metrics['avg_response_time'] * (total_success -1)
            new_avg = (current_total_time + response_time) / total_success
            self.performance_metrics['avg_response_time'] = new_avg
            logger.debug(f"Updated performance for {proxy}: SUCCESS, time={response_time:.3f}s. New avg time: {new_avg:.3f}s")


            # Reset failure count for this specific proxy upon success
            if proxy in self.failed_proxies:
                _, last_fail_time = self.failed_proxies[proxy]
                self.failed_proxies[proxy] = (0, last_fail_time) # Reset count, keep last fail time
                logger.debug(f"Reset failure count for successful proxy {proxy}.")

        else:
            self.performance_metrics['failure_count'] += 1
            self.mark_failed(proxy) # Use existing method to handle failure count and timestamp
            logger.debug(f"Updated performance for {proxy}: FAILURE, time={response_time:.3f}s.")

        # --- Report to Central Performance Monitor ---
        if self._perf_monitor:
            try:
                self._perf_monitor.update('proxy', success, response_time)
            except Exception as e:
                logger.error(f"Failed to report proxy performance to central monitor: {e}")
        # --- End Reporting ---


    def refresh_proxies(self):
        """Reloads proxies from the original source (env var or free fetch)."""
        # Clear existing proxies to ensure a fresh load
        old_proxy_count = len(self.proxy_deque)
        self.proxy_deque.clear()
        
        # Try loading from env var or fetch free proxies
        self._load_proxies(None)  # Pass None to use env vars or fetch as needed
        
        # If proxy pool is still empty after refresh, fallback to default proxies
        if not self.is_available():
            logger.warning("Proxy pool is empty after initial refresh, falling back to default proxies")
            default_proxies = get_default_proxies()
            if default_proxies:
                parsed_defaults = self._parse_and_validate_proxies(default_proxies, "default_fallback")
                self.proxy_deque.extend(parsed_defaults)
                logger.info(f"Added {len(parsed_defaults)} default fallback proxies to the pool")
            else:
                logger.error("Proxy pool is empty after reset.")
        
        logger.info(f"Proxy refresh complete. Previous count: {old_proxy_count}, New count: {len(self.proxy_deque)}")
        return self.is_available()

    def get_requests_proxy_dict(self) -> Optional[Dict[str, str]]:
        """
        Returns the next available proxy in the format suitable for the requests library,
        e.g., {'http': 'http://...', 'https': 'http://...'}. Uses round-robin selection.
        """
        if not self.proxy_deque:
            logger.warning("No proxies available for requests. Using direct connection.")
            return None

        proxy_url = self.get_proxy()  # This implements round-robin and failure avoidance
        if not proxy_url:
            logger.warning("Using direct connection.")
            return None

        # Add http:// prefix if missing and not using a different scheme
        if not proxy_url.startswith(('http://', 'https://', 'socks4://', 'socks5://')):
            proxy_url = f"http://{proxy_url}"

        proxy_dict = {
            "http": proxy_url,
            "https": proxy_url  # Use same proxy for both HTTP/HTTPS
        }

        # Log detailed proxy info at debug level
        logger.debug(f"Selected proxy for request: {proxy_url}")
        # Track in the monitor - if all modules use ProxyManager, we'll see all proxy usage
        self._perf_monitor.update({
            'event_type': 'proxy_selection',
            'proxy_url': proxy_url,
            'timestamp': time.time()
        })

        return proxy_dict

    def get_playwright_proxy_dict(self) -> Optional[Dict[str, str]]:
        """
        Returns the next available proxy in the format suitable for Playwright's launch options.
        e.g., {'server': 'http://host:port', 'username': 'user', 'password': 'pass'}
              {'server': 'socks5://host:port', ...}. Uses round-robin selection.
        """
        proxy_url = self.get_proxy() # Gets next available proxy
        if not proxy_url:
            return None

        try:
            parsed = urlparse(proxy_url)
            scheme = parsed.scheme.lower()

            if scheme not in ['http', 'https', 'socks4', 'socks5']:
                logger.warning(f"Unsupported proxy scheme '{scheme}' for Playwright in proxy: {proxy_url}")
                return None

            # Server includes scheme, host, and port
            server = f"{scheme}://{parsed.hostname}:{parsed.port}"

            proxy_dict: Dict[str, str] = {"server": server}

            if parsed.username:
                proxy_dict["username"] = parsed.username
            if parsed.password:
                proxy_dict["password"] = parsed.password

            logger.debug(f"Providing Playwright proxy dict: {proxy_dict}")
            return proxy_dict

        except Exception as e:
            logger.error(f"Failed to parse proxy URL '{proxy_url}' for Playwright format: {e}")
            return None


    def is_available(self) -> bool:
        """Checks if the proxy deque is not empty."""
        return bool(self.proxy_deque)

    def health_check(self, test_url="https://api.football-data.org", timeout=5):
        """Checks all proxies in the deque and removes those that fail the test URL."""
        healthy = []
        for proxy in list(self.proxy_deque):
            try:
                resp = requests.get(test_url, proxies={"http": proxy, "https": proxy}, timeout=timeout)
                if resp.status_code == 200:
                    healthy.append(proxy)
                else:
                    self.mark_failed(proxy)
                    logger.warning(f"Proxy {proxy} failed health check: status {resp.status_code}")
            except Exception as e:
                self.mark_failed(proxy)
                logger.warning(f"Proxy {proxy} failed health check: {e}")
        self.proxy_deque = deque(healthy)
        logger.info(f"Proxy health check complete. {len(healthy)} healthy proxies remain.")
        return healthy

    def get_pool_status(self):
        """Returns a summary of the proxy pool for diagnostics or admin UI."""
        return {
            "active": list(self.proxy_deque),
            "failed": {k: v[0] for k, v in self.failed_proxies.items()},
            "failure_counts": {k: v[0] for k, v in self.failed_proxies.items()},
            "last_fail_times": {k: v[1] for k, v in self.failed_proxies.items()},
            "performance_metrics": dict(self.performance_metrics)
        }

# Example Usage (Illustrative)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for testing proxy logic
    # Example with env var: export PROXY_LIST="http://proxy1:8080,http://user:pass@proxy2:3128"
    # Or run without env var to test free proxy fetching
    os.environ.pop("PROXY_LIST", None) # Ensure env var isn't set for free proxy test if needed
    print("--- Initializing ProxyManager (will try env var then free source) ---")
    proxy_manager = ProxyManager()

    if proxy_manager.is_available():
        print("\n--- Proxies Loaded ---")
        print(f"Total proxies in deque: {len(proxy_manager.proxy_deque)}")

        print("\n--- Getting Proxies (Round-Robin) ---")
        for i in range(len(proxy_manager.proxy_deque) + 2): # Cycle through a bit more than available
            print(f"Attempt {i+1}:")
            req_proxy = proxy_manager.get_requests_proxy_dict()
            print(f"  Requests Format: {req_proxy}")
            if i == 1 and req_proxy: # Simulate first proxy failing
                failed_url = req_proxy.get('http')
                print(f"  *** Simulating failure for: {failed_url} ***")
                proxy_manager.mark_failed(failed_url)
            if i == 3 and req_proxy: # Simulate second proxy failing
                failed_url = req_proxy.get('http')
                print(f"  *** Simulating failure for: {failed_url} ***")
                proxy_manager.mark_failed(failed_url)
                proxy_manager.mark_failed(failed_url) # Fail it again
                proxy_manager.mark_failed(failed_url) # Fail it a third time (should hit max_failures)


            playwright_proxy = proxy_manager.get_playwright_proxy_dict() # Note: gets the *next* available proxy
            print(f"  Playwright Format: {playwright_proxy}")


        print("\n--- Checking Failed Proxies ---")
        print(proxy_manager.failed_proxies)

        print("\n--- Refreshing Proxies ---")
        proxy_manager.refresh_proxies()
        print(f"Total proxies after refresh: {len(proxy_manager.proxy_deque)}")
        print(f"Failed proxies after refresh: {proxy_manager.failed_proxies}")


    else:
        print("\n--- No proxies loaded. ---")
