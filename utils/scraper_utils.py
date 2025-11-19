"""
Enhanced scraper utilities with anti-scraping measures and robust error handling.
Provides a unified interface for web scraping with configurable behavior.
"""
import logging
import random
import time
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import os
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import pandas as pd

from utils.config import Config
from utils.http_client import HttpClient
from dashboard.error_log import log_error

logger = logging.getLogger(__name__)

class ScraperUtils:
    """
    Utility class for web scraping with anti-scraping measures and robust error handling.
    Uses configuration from the central Config system.
    """
    
    def __init__(self):
        """Initialize the scraper utilities with configuration from Config."""
        # Load configuration
        self.config = self._load_config()
        
        # Initialize HTTP client
        self.http_client = HttpClient()
        
        # Initialize user agent rotation
        self.user_agents = self.config.get("anti_scraping", {}).get("user_agents", [])
        if not self.user_agents:
            self.user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            ]
            
        # Initialize request delay settings
        self.request_delay = self.config.get("anti_scraping", {}).get("request_delay", {"min": 1, "max": 3})
        
        # Initialize headers rotation
        self.rotate_headers = self.config.get("anti_scraping", {}).get("rotate_headers", True)
        
        # Initialize robots.txt respect
        self.respect_robots_txt = self.config.get("anti_scraping", {}).get("respect_robots_txt", True)
        
        # Last request timestamp for delay calculation
        self.last_request_time = 0
        
    def _load_config(self) -> Dict[str, Any]:
        """Load scraper configuration from Config."""
        # Get global HTTP client settings
        http_config = Config.get("http_client", {})
        
        # Get anti-scraping settings
        anti_scraping = Config.get("anti_scraping", {})
        
        # Get sources configuration
        sources = Config.get("sources", {})
        
        return {
            "http_client": http_config,
            "anti_scraping": anti_scraping,
            "sources": sources
        }
        
    def _get_random_user_agent(self) -> str:
        """Get a random user agent from the configured list with weighted selection."""
        # If using predefined list, implement weighted selection favoring newer browser versions
        user_agents_by_category = {
            # Desktop browsers (60% probability)
            "desktop": [
                # Chrome (most common)
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
                # Firefox
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 12.1; rv:96.0) Gecko/20100101 Firefox/96.0",
                # Edge
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36 Edg/96.0.1054.62",
            ],
            # Mobile browsers (30% probability)
            "mobile": [
                "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (Linux; Android 12; Pixel 6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.104 Mobile Safari/537.36",
                "Mozilla/5.0 (iPad; CPU OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/96.0.4664.116 Mobile/15E148 Safari/604.1",
            ],
            # Tablets and others (10% probability)
            "tablet": [
                "Mozilla/5.0 (Linux; Android 12; SM-T970) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.104 Safari/537.36",
                "Mozilla/5.0 (iPad; CPU OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Mobile/15E148 Safari/604.1",
            ]
        }
        
        # Choose category based on weighted probability
        category = random.choices(
            population=["desktop", "mobile", "tablet"],
            weights=[0.6, 0.3, 0.1],
            k=1
        )[0]
        
        # If custom user agents are configured, use those instead
        if self.user_agents and len(self.user_agents) > 0:
            return random.choice(self.user_agents)
        
        # Otherwise use our built-in categorized user agents
        return random.choice(user_agents_by_category[category])
        
    def _get_random_headers(self) -> Dict[str, str]:
        """Get random headers for anti-scraping with advanced browser fingerprinting evasion."""
        # Get a random user agent first - important for header consistency
        user_agent = self._get_random_user_agent()
        
        # Base headers every browser would have
        headers = {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
            "Accept-Language": random.choice(["en-US,en;q=0.9", "en-GB,en;q=0.9,en-US;q=0.8", "en-CA,en;q=0.9,fr-CA;q=0.8,fr;q=0.7"]),
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        }
        
        # Conditionally add other headers based on the user agent to maintain consistency
        # This prevents creating impossible header combinations that would flag as bot traffic
        
        # Chrome-specific headers
        if "Chrome" in user_agent and "Edg" not in user_agent:
            headers["sec-ch-ua"] = '" Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"'
            headers["sec-ch-ua-mobile"] = "?0" if "Mobile" not in user_agent else "?1"
            headers["sec-ch-ua-platform"] = '"Windows"' if "Windows" in user_agent else ('"macOS"' if "Mac" in user_agent else '"Linux"')
            
        # Add Accept-Encoding (gzip, deflate, br are common)
        headers["Accept-Encoding"] = "gzip, deflate, br"
        
        # Add random DNT (Do Not Track) header
        if random.random() > 0.7:  # 30% of browsers might have DNT enabled
            headers["DNT"] = "1"
            
        # Add random referer with higher sophistication
        if random.random() > 0.4:  # 60% chance of having a referer
            # Common referrers based on context
            search_engines = [
                "https://www.google.com/search?q=football+stats",
                "https://www.bing.com/search?q=premier+league+fixtures",
                "https://duckduckgo.com/?q=football+predictions",
                "https://www.google.co.uk/search?q=football+today"
            ]
            
            social_media = [
                "https://www.reddit.com/r/soccer",
                "https://twitter.com/premierleague",
                "https://www.facebook.com/"
            ]
            
            news_sites = [
                "https://www.bbc.co.uk/sport/football",
                "https://www.skysports.com/football",
                "https://www.espn.com/soccer/"
            ]
            
            # Weight categories based on typical browsing patterns
            referer_category = random.choices(
                population=[search_engines, social_media, news_sites],
                weights=[0.5, 0.3, 0.2],
                k=1
            )[0]
            
            headers["Referer"] = random.choice(referer_category)
            
        # Add random Cache-Control
        cache_controls = ["max-age=0", "no-cache", "max-age=0, private, must-revalidate"]
        headers["Cache-Control"] = random.choice(cache_controls)
        
        # Randomly add Connection header
        if random.random() > 0.2:  # 80% chance
            headers["Connection"] = random.choice(["keep-alive", "close"])
            
        # Mobile-specific headers
        if "Mobile" in user_agent:
            viewport_widths = [375, 390, 393, 412, 414, 428]
            viewport_heights = [667, 740, 781, 844, 846, 915, 926]
            pixel_ratio = [2, 3]
            
            headers["Viewport-Width"] = str(random.choice(viewport_widths))
            headers["Width"] = str(random.choice(viewport_widths))
            
        return headers
        
    def _apply_request_delay(self, source_name: Optional[str] = None):
        """
        Apply an adaptive delay between requests based on multiple factors:
        - Source reputation and rate limits
        - Human-like behavior patterns
        - Time of day variations
        - Previous success/failure rates
        
        Args:
            source_name: Optional name of the source being accessed for source-specific behavior
        """
        # Calculate time since last request
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Get source configuration if available
        source_config = self._get_source_config(source_name) if source_name else None
        
        # Determine base delay range from source config or defaults
        min_delay = self.request_delay["min"]
        max_delay = self.request_delay["max"]
        
        if source_config and "request_delay" in source_config:
            min_delay = source_config["request_delay"].get("min", min_delay)
            max_delay = source_config["request_delay"].get("max", max_delay)
        
        # If enough time has already passed naturally, we can proceed immediately
        if time_since_last >= max_delay:
            self.last_request_time = current_time
            return
        
        # --- Apply advanced delay calculation logic ---
        
        # 1. Adaptive rate limiting based on source reputation
        reputation_factor = 1.0
        if source_name:
            # Higher reputation factor means we're more cautious with this source
            reputation_mapping = {
                # High-tier sources with strict rate limits
                "api-football": 1.5,
                "football-data": 1.3,
                "whoscored": 1.8,  # Very sensitive to scraping
                "sofascore": 1.6,
                # Mid-tier sources
                "espn": 1.2,
                "bbc": 1.0,
                # Lower-tier sources with more lenient limits
                "wikipedia": 0.7,
                "reddit": 0.8
            }
            reputation_factor = reputation_mapping.get(source_name.lower(), 1.0)
        
        # 2. Time-of-day variation - be gentler during peak hours
        # Sites are often more heavily loaded during business hours
        hour_of_day = datetime.datetime.now().hour
        time_factor = 1.0
        
        # Simulate peak hours (9am-6pm) on weekdays when sites are most active/monitored
        if 9 <= hour_of_day <= 18 and datetime.datetime.now().weekday() < 5:  # Weekday daytime
            time_factor = 1.2
        elif hour_of_day >= 22 or hour_of_day <= 5:  # Late night/early morning
            time_factor = 0.8  # Less delay during off-hours
        
        # 3. Behavioral patterns - sometimes humans pause longer between actions
        # Add occasional longer pauses to seem more human-like
        behavioral_factor = 1.0
        random_chance = random.random()
        
        if random_chance > 0.95:  # 5% chance of a "distraction"
            behavioral_factor = 2.5  # Significantly longer pause
            logger.debug("Applying longer 'distraction' delay to mimic human behavior")
        elif random_chance > 0.85:  # 10% chance of slight hesitation
            behavioral_factor = 1.5  # Moderately longer pause
        
        # 4. Apply randomness with a skew toward longer delays (more conservative)
        # Use triangular distribution skewed toward the higher end
        skew_point = min_delay + (max_delay - min_delay) * 0.7  # 70% toward max
        base_delay = random.triangular(min_delay, max_delay, skew_point)
        
        # Combine all factors
        final_delay = base_delay * reputation_factor * time_factor * behavioral_factor
        
        # Apply a maximum cap to avoid excessive delays
        max_allowed_delay = 15.0  # Never delay more than 15 seconds
        final_delay = min(final_delay, max_allowed_delay)
        
        # Adjust for time already passed
        adjusted_delay = max(0, final_delay - time_since_last)
        
        # Apply the delay if needed
        if adjusted_delay > 0:
            if adjusted_delay > 2.0:
                logger.info(f"Applying longer anti-scraping delay of {adjusted_delay:.2f}s for {source_name or 'unknown source'} (reputation={reputation_factor:.1f}, time={time_factor:.1f}, behavior={behavioral_factor:.1f})")
            else:
                logger.debug(f"Applying anti-scraping delay of {adjusted_delay:.2f}s for {source_name or 'unknown source'}")
            
            # Split longer delays into smaller chunks with tiny variations to defeat timing pattern detection
            if adjusted_delay > 3.0:
                chunk_size = 1.0
                remaining = adjusted_delay
                while remaining > 0:
                    chunk = min(chunk_size, remaining)
                    # Add a tiny bit of noise to each chunk
                    chunk *= random.uniform(0.95, 1.05)
                    time.sleep(chunk)
                    remaining -= chunk
            else:
                time.sleep(adjusted_delay)
        
        # Update last request time
        self.last_request_time = time.time()
        
    def _get_source_config(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific data source."""
        # Check primary sources
        primary_sources = self.config.get("sources", {}).get("primary", [])
        for source in primary_sources:
            if source.get("name") == source_name:
                return source
                
        # Check fallback sources
        fallback_sources = self.config.get("sources", {}).get("fallback", [])
        for source in fallback_sources:
            if source.get("name") == source_name:
                return source
                
        # Check metadata sources
        metadata_sources = self.config.get("sources", {}).get("metadata", [])
        for source in metadata_sources:
            if source.get("name") == source_name:
                return source
                
        return None
        
    def fetch_html(
        self, 
        url: str, 
        source_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        proxy: Optional[str] = None,
        use_playwright: Optional[bool] = None
    ) -> Optional[str]:
        """
        Fetch HTML content from a URL with anti-scraping measures.
        
        Args:
            url: URL to fetch
            source_name: Name of the data source (to get source-specific config)
            params: Query parameters
            proxy: Proxy URL
            use_playwright: Whether to use Playwright for dynamic content
            
        Returns:
            HTML content as string, or None if fetch failed
        """
        # Get source configuration if provided
        source_config = self._get_source_config(source_name) if source_name else None
        
        # Determine if we should use Playwright
        should_use_playwright = use_playwright
        if should_use_playwright is None and source_config:
            should_use_playwright = source_config.get("use_playwright", False)
            
        # If using Playwright, delegate to MCP server
        if should_use_playwright:
            return self._fetch_html_with_playwright(url, source_config, params, proxy)
            
        # Otherwise, use regular HTTP client
        return self._fetch_html_with_http_client(url, source_config, params, proxy)
        
    def _fetch_html_with_http_client(
        self,
        url: str,
        source_config: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        proxy: Optional[str]
    ) -> Optional[str]:
        """Fetch HTML content using the HTTP client."""
        try:
            # Apply request delay for anti-scraping
            self._apply_request_delay()
            
            # Prepare headers
            headers = self._get_random_headers() if self.rotate_headers else None
            
            # Get timeout from source config or default
            timeout = source_config.get("timeout") if source_config else None
            
            # Make request
            response = self.http_client.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                proxy=proxy
            )
            
            # Return HTML content
            return response.text
        except Exception as e:
            log_error(f"Error fetching HTML from {url}", e)
            return None
            
    def _fetch_html_with_playwright(
        self,
        url: str,
        source_config: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]],
        proxy: Optional[str]
    ) -> Optional[str]:
        """
        Fetch HTML content using Playwright for dynamic content.
        
        Note: This method should be implemented to call the MCP server
        for Playwright-based scraping. For now, it returns a placeholder.
        """
        logger.info(f"Playwright scraping requested for {url} - this should be handled by MCP server")
        return None
        
    def parse_html(self, html: str) -> Optional[BeautifulSoup]:
        """
        Parse HTML content into a BeautifulSoup object.
        
        Args:
            html: HTML content as string
            
        Returns:
            BeautifulSoup object, or None if parsing failed
        """
        try:
            return BeautifulSoup(html, "html.parser")
        except Exception as e:
            log_error("Error parsing HTML", e)
            return None
            
    def extract_table(
        self,
        soup: BeautifulSoup,
        table_selector: str,
        header_selector: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Extract a table from HTML into a pandas DataFrame.
        
        Args:
            soup: BeautifulSoup object
            table_selector: CSS selector for the table
            header_selector: CSS selector for the table headers
            
        Returns:
            DataFrame containing the table data, or None if extraction failed
        """
        try:
            # Find the table
            table = soup.select_one(table_selector)
            if not table:
                logger.warning(f"Table not found with selector: {table_selector}")
                return None
                
            # Extract headers if selector provided
            headers = None
            if header_selector:
                header_elements = table.select(header_selector)
                if header_elements:
                    headers = [header.text.strip() for header in header_elements]
                    
            # Extract rows
            rows = []
            for row in table.select("tr"):
                cells = [cell.text.strip() for cell in row.select("td, th")]
                if cells:
                    rows.append(cells)
                    
            # Create DataFrame
            if not rows:
                logger.warning("No rows found in table")
                return None
                
            # If headers were found, use them
            if headers and len(headers) == len(rows[0]):
                df = pd.DataFrame(rows[1:], columns=headers)
            else:
                df = pd.DataFrame(rows)
                
            return df
        except Exception as e:
            log_error("Error extracting table from HTML", e)
            return None
            
    def fetch_json(
        self,
        url: str,
        source_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        proxy: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch JSON data from a URL with anti-scraping measures.
        
        Args:
            url: URL to fetch
            source_name: Name of the data source (to get source-specific config)
            params: Query parameters
            headers: Additional headers
            proxy: Proxy URL
            
        Returns:
            JSON data as dictionary, or None if fetch failed
        """
        try:
            # Get source configuration if provided
            source_config = self._get_source_config(source_name) if source_name else None
            
            # Apply request delay for anti-scraping
            self._apply_request_delay()
            
            # Prepare headers
            if self.rotate_headers and not headers:
                headers = self._get_random_headers()
                
            # Get timeout from source config or default
            timeout = source_config.get("timeout") if source_config else None
            
            # Make request
            response = self.http_client.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                proxy=proxy
            )
            
            # Return JSON data
            return response.json()
        except Exception as e:
            log_error(f"Error fetching JSON from {url}", e)
            return None
            
    def post_json(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        source_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        proxy: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Post data to a URL and receive JSON response with anti-scraping measures.
        
        Args:
            url: URL to post to
            data: Form data
            json_data: JSON data
            source_name: Name of the data source (to get source-specific config)
            headers: Additional headers
            proxy: Proxy URL
            
        Returns:
            JSON response as dictionary, or None if post failed
        """
        try:
            # Get source configuration if provided
            source_config = self._get_source_config(source_name) if source_name else None
            
            # Apply request delay for anti-scraping
            self._apply_request_delay()
            
            # Prepare headers
            if self.rotate_headers and not headers:
                headers = self._get_random_headers()
                
            # Get timeout from source config or default
            timeout = source_config.get("timeout") if source_config else None
            
            # Make request
            response = self.http_client.post(
                url,
                data=data,
                json=json_data,
                headers=headers,
                timeout=timeout,
                proxy=proxy
            )
            
            # Return JSON data
            return response.json()
        except Exception as e:
            log_error(f"Error posting to {url}", e)
            return None
            
    def get_endpoint_url(
        self,
        source_name: str,
        endpoint_name: str,
        path_params: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Get the full URL for a specific endpoint of a data source.
        
        Args:
            source_name: Name of the data source
            endpoint_name: Name of the endpoint
            path_params: Parameters to substitute in the endpoint path
            
        Returns:
            Full URL for the endpoint, or None if source or endpoint not found
        """
        # Get source configuration
        source_config = self._get_source_config(source_name)
        if not source_config:
            logger.warning(f"Source not found: {source_name}")
            return None
            
        # Get base URL
        base_url = source_config.get("url")
        if not base_url:
            logger.warning(f"Base URL not found for source: {source_name}")
            return None
            
        # Get endpoints configuration
        endpoints = source_config.get("endpoints", {})
        
        # Get endpoint path
        endpoint_path = endpoints.get(endpoint_name)
        if not endpoint_path:
            logger.warning(f"Endpoint not found: {endpoint_name} for source {source_name}")
            return None
            
        # Substitute path parameters
        if path_params:
            for param_name, param_value in path_params.items():
                endpoint_path = endpoint_path.replace(f"{{{param_name}}}", str(param_value))
                
        # Combine base URL and endpoint path
        if base_url.endswith("/") and endpoint_path.startswith("/"):
            url = f"{base_url}{endpoint_path[1:]}"
        elif not base_url.endswith("/") and not endpoint_path.startswith("/"):
            url = f"{base_url}/{endpoint_path}"
        else:
            url = f"{base_url}{endpoint_path}"
            
        return url
        
    def close(self):
        """Close the HTTP client."""
        self.http_client.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Create a singleton instance for easy access
scraper = ScraperUtils()

# Convenience functions for direct use without creating a scraper instance

def fetch_html(url: str, **kwargs) -> Optional[str]:
    """Fetch HTML content from a URL with anti-scraping measures."""
    return scraper.fetch_html(url, **kwargs)
    
def fetch_json(url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Fetch JSON data from a URL with anti-scraping measures."""
    return scraper.fetch_json(url, **kwargs)
    
def post_json(url: str, **kwargs) -> Optional[Dict[str, Any]]:
    """Post data to a URL and receive JSON response with anti-scraping measures."""
    return scraper.post_json(url, **kwargs)
    
def get_endpoint_url(source_name: str, endpoint_name: str, **kwargs) -> Optional[str]:
    """Get the full URL for a specific endpoint of a data source."""
    return scraper.get_endpoint_url(source_name, endpoint_name, **kwargs)
