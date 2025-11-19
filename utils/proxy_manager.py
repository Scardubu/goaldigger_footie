import logging
import random
import time
from collections import deque

from dashboard.error_log import log_error  # Import log_error


class ProxyManager:
    def __init__(self, proxy_list, max_failures=3, cooldown_seconds=600):
        self.proxies = deque(proxy_list)
        self.failed_proxies = set()
        self.proxy_failures = {}  # Track failure counts
        self.cooldown_proxies = {}  # proxy: cooldown_end_time
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds

    def get_proxy(self):
        self._prune_cooldown_proxies()
        if not self.proxies:
            self.reset_proxies()
        return self.proxies[0] if self.proxies else None

    def rotate_proxy(self):
        if self.proxies:
            self.proxies.rotate(-1)

    def mark_proxy_as_failed(self, proxy):
        now = time.time()
        self.proxy_failures[proxy] = self.proxy_failures.get(proxy, 0) + 1
        if self.proxy_failures[proxy] >= self.max_failures:
            self.failed_proxies.add(proxy)
            self.cooldown_proxies[proxy] = now + self.cooldown_seconds
            if proxy in self.proxies:
                self.proxies.remove(proxy)
            logging.warning(f"Proxy {proxy} marked for cooldown after {self.max_failures} failures.")
        else:
            if proxy in self.proxies:
                self.proxies.remove(proxy)
            self.proxies.append(proxy)  # Move to end for retry

    def reset_proxies(self):
        self._prune_cooldown_proxies()
        available = [p for p in self.failed_proxies if p not in self.cooldown_proxies]
        self.proxies.extend(available)
        self.failed_proxies.difference_update(available)
        
        # Shuffle if we have proxies
        if self.proxies:
            random.shuffle(list(self.proxies))
        else:
            # Try to recover by checking environment variables
            from utils.config import Config
            proxy_list = Config.get("scraping.proxy.providers", [])
            if proxy_list:
                self.proxies.extend(proxy_list)
                logging.info(f"Recovered proxy pool with {len(proxy_list)} proxies from config")
            else:
                logging.warning("Proxy pool is empty after reset. Using direct connection.")
                # Add a dummy direct connection proxy
                self.proxies.append("direct")
                
        # Always have at least the direct connection
        if not self.proxies or (len(self.proxies) == 0):
            self.proxies.append("direct")

    def _prune_cooldown_proxies(self):
        now = time.time()
        to_restore = [p for p, t in self.cooldown_proxies.items() if t <= now]
        for proxy in to_restore:
            self.failed_proxies.discard(proxy)
            self.proxies.append(proxy)
            del self.cooldown_proxies[proxy]
            self.proxy_failures[proxy] = 0
            logging.info(f"Proxy {proxy} restored to pool after cooldown.")

    def health_check(self, test_url="https://api.football-data.org", timeout=5):
        import requests
        healthy = []
        for proxy in list(self.proxies):
            try:
                resp = requests.get(test_url, proxies={"http": proxy, "https": proxy}, timeout=timeout)
                if resp.status_code == 200:
                    healthy.append(proxy)
                else:
                    self.mark_proxy_as_failed(proxy)
            except Exception as e:
                self.mark_proxy_as_failed(proxy)
                log_error(f"Proxy {proxy} failed health check", e) # Use log_error
        self.proxies = deque(healthy)
        return healthy

    def get_pool_status(self):
        return {
            "active": list(self.proxies),
            "failed": list(self.failed_proxies),
            "cooldown": list(self.cooldown_proxies.keys()),
            "failure_counts": dict(self.proxy_failures)
        }

    def export_status_json(self):
        """Return proxy pool status as a JSON-serializable dict for export/download."""
        return self.get_pool_status()

    def refresh_proxies(self, new_proxy_list=None):
        """Refresh the proxy pool, optionally with a new list. Remove dead proxies and reset failures."""
        if new_proxy_list is not None:
            self.proxies = deque(new_proxy_list)
        else:
            # Remove proxies that are in failed or cooldown state
            self.proxies = deque([p for p in self.proxies if p not in self.failed_proxies and p not in self.cooldown_proxies])
        self.failed_proxies.clear()
        self.proxy_failures.clear()
        self.cooldown_proxies.clear()
        logging.info("Proxy pool refreshed. All failures and cooldowns cleared.")

    def is_available(self):
        """Return True if there are any proxies available for use."""
        return bool(self.proxies)

class UserAgentManager:
    """Manages user agent rotation for scraping to avoid detection."""
    
    def __init__(self):
        self.user_agents = [
            # Modern browsers - Desktop
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            # Mobile browsers
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1",
        ]
        self.current_index = 0
        random.shuffle(self.user_agents)
        
    def get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        return random.choice(self.user_agents)
        
    def get_next_user_agent(self) -> str:
        """Get the next user agent in rotation."""
        ua = self.user_agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.user_agents)
        return ua

# Global instances for use throughout the application
# Initialize UserAgentManager
user_agent_manager = UserAgentManager()

# Proxy manager will be initialized when needed
# Usage example:
# from utils.proxy_manager import user_agent_manager, ProxyManager
# proxy_list = get_proxy_list_from_config()
# proxy_manager = ProxyManager(proxy_list)

def create_proxy_manager_from_env():
    """Create proxy manager from environment configuration."""
    import os
    import re

    from dotenv import load_dotenv

    # Ensure environment variables are loaded
    load_dotenv()
    
    # Get proxy list from environment
    proxy_str = os.environ.get("PROXY_LIST", "").strip('"\'')
    if not proxy_str:
        logging.warning("No proxies found in PROXY_LIST environment variable")
        return None
    
    # Parse comma-separated proxy list
    proxies = [p.strip() for p in proxy_str.split(',')]
    proxies = [p for p in proxies if p]  # Remove empty entries
    
    if not proxies:
        logging.warning("No valid proxies found in PROXY_LIST")
        return None
    
    logging.info(f"Creating proxy manager with {len(proxies)} proxies")
    return ProxyManager(proxies)
