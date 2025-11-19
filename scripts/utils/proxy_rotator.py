import logging
import random
from concurrent.futures import ThreadPoolExecutor

import requests
from goaldiggers.utils.fp import FreeProxy
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)
# Configure logging basic setup if not done elsewhere
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ProxyMaster:
    def __init__(self, test_url="https://httpbin.org/ip", max_workers=10):
        """
        Initializes the ProxyMaster.
        Args:
            test_url (str): URL to test proxy connectivity.
            max_workers (int): Max concurrent workers for testing proxies.
        """
        self.test_url = test_url
        self.max_workers = max_workers
        self.proxy_pool = []
        self._refresh_pool()

    def _test_proxy(self, proxy: str) -> bool:
        """Tests a single proxy string (e.g., 'http://ip:port')."""
        proxies = {
            "http": proxy,
            "https": proxy,  # Assume the proxy works for both, adjust if needed
        }
        try:
            # Use a reliable, simple endpoint for testing
            response = requests.get(
                self.test_url, proxies=proxies, timeout=5, verify=True
            )
            response.raise_for_status()
            logger.debug(f"Proxy {proxy} test successful.")
            return True
        except RequestException as e:
            logger.debug(f"Proxy {proxy} test failed: {e}")
            return False

    def _refresh_pool(self):
        """Fetches new proxies and tests them concurrently."""
        logger.info("Refreshing proxy pool...")
        try:
            raw_proxies = FreeProxy(timeout=1, rand=True).get_proxy_list(
                repeat=5
            )  # Get a few more to increase chances
            logger.info(f"Fetched {len(raw_proxies)} raw proxies. Testing...")
        except Exception as e:
            logger.error(f"Failed to fetch raw proxies using FreeProxy: {e}")
            raw_proxies = []  # Ensure raw_proxies is iterable

        valid_proxies = []
        # Use ThreadPoolExecutor for concurrent testing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self._test_proxy, raw_proxies)
            valid_proxies = [p for p, valid in zip(raw_proxies, results) if valid]

        logger.info(f"Found {len(valid_proxies)} working proxies.")
        self.proxy_pool = valid_proxies

    def get_proxy(self) -> str | None:
        """Returns a random working proxy from the pool."""
        if not self.proxy_pool:
            logger.warning("Proxy pool is empty. Attempting refresh.")
            self._refresh_pool()
            if not self.proxy_pool:
                logger.error("Proxy pool remains empty after refresh.")
                return None  # Return None if still empty

        # Refresh if pool is low, maybe less aggressively than before
        if len(self.proxy_pool) < 5:
            logger.info("Proxy pool low. Triggering refresh.")
            self._refresh_pool()
            if not self.proxy_pool:
                return None  # Check again after refresh

        return random.choice(self.proxy_pool)

    def get_proxies(self, count: int = 5) -> list[str]:
        """Returns a list of unique random working proxies."""
        if len(self.proxy_pool) < count:
            logger.warning(
                f"Requested {count} proxies, but only {len(self.proxy_pool)} available. Refreshing."
            )
            self._refresh_pool()
            # Adjust count if still not enough after refresh
            count = min(count, len(self.proxy_pool))

        if count == 0:
            return []

        return random.sample(self.proxy_pool, count)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    proxy_master = ProxyMaster()
    print("Single Proxy:", proxy_master.get_proxy())
    print("Multiple Proxies:", proxy_master.get_proxies(3))
