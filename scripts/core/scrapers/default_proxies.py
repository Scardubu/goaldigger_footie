"""
Default proxy configuration module for the betting insights platform.
Provides fallback proxies when external sources are unavailable.
"""
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# List of reliable free proxies that are regularly updated and tested
# These act as fallbacks when the proxy API fails or no proxies are provided
DEFAULT_PROXIES = [
    "http://51.79.50.31:9300",
    "http://51.79.50.22:9300",
    "http://51.79.50.46:9300",
    "http://51.195.137.59:3128",
    "http://51.159.115.233:3128",
    "http://193.187.242.242:8080",
    "http://82.180.163.163:80",
    "http://34.70.168.95:80"
]

def get_default_proxies() -> List[str]:
    """
    Returns a list of default proxies to use when no other source is available.
    
    Returns:
        List[str]: List of proxy URLs in standard format (e.g., http://host:port)
    """
    # Check if custom default proxies are provided via environment variable
    custom_proxies = os.getenv("DEFAULT_PROXIES")
    if custom_proxies:
        logger.info("Using custom default proxies from environment")
        return [proxy.strip() for proxy in custom_proxies.split(",") if proxy.strip()]
    
    logger.info(f"Using {len(DEFAULT_PROXIES)} built-in default proxies")
    return DEFAULT_PROXIES.copy()
