"""
HTTP client utility with robust error handling and retry logic.
Provides a unified interface for making HTTP requests with configurable retry behavior.
"""
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from utils.config import Config

logger = logging.getLogger(__name__)

class HttpClient:
    """
    HTTP client with robust error handling, retry logic, and proxy support.
    Uses configuration from the central Config system.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize the HTTP client with configuration from Config.
        
        Args:
            base_url: Optional base URL for all requests
        """
        # Load configuration
        self.config = self._load_config()
        
        # Set base URL
        self.base_url = base_url
        
        # Create session with retry logic
        self.session = self._create_session()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load HTTP client configuration from Config."""
        http_config = Config.get("http_client", {})
        
        # Set defaults if not in config
        defaults = {
            "timeout": 30,
            "max_retries": 3,
            "backoff_factor": 0.5,
            "retry_statuses": [429, 500, 502, 503, 504],
            "user_agent": "GoalDiggers/1.0",
            "verify_ssl": True
        }
        
        # Merge defaults with config
        for key, value in defaults.items():
            if key not in http_config:
                http_config[key] = value
                
        return http_config
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        
        # Configure retry logic
        retry_strategy = Retry(
            total=self.config["max_retries"],
            backoff_factor=self.config["backoff_factor"],
            status_forcelist=self.config["retry_statuses"],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "User-Agent": self.config["user_agent"],
            "Accept": "application/json"
        })
        
        return session
        
    def _get_full_url(self, endpoint: str) -> str:
        """Get full URL by combining base URL and endpoint."""
        if self.base_url and not endpoint.startswith(("http://", "https://")):
            # Join base URL and endpoint, ensuring no double slashes
            if self.base_url.endswith("/") and endpoint.startswith("/"):
                return f"{self.base_url}{endpoint[1:]}"
            elif not self.base_url.endswith("/") and not endpoint.startswith("/"):
                return f"{self.base_url}/{endpoint}"
            else:
                return f"{self.base_url}{endpoint}"
        else:
            return endpoint
            
    def _add_jitter(self, delay: float) -> float:
        """Add random jitter to delay to prevent thundering herd problem."""
        return delay * (1 + random.random() * 0.1)
        
    def get(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        proxy: Optional[str] = None
    ) -> requests.Response:
        """
        Make a GET request with retry logic.
        
        Args:
            endpoint: API endpoint or full URL
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout in seconds (overrides config)
            proxy: Proxy URL (overrides session proxy)
            
        Returns:
            Response object
        """
        url = self._get_full_url(endpoint)
        timeout = timeout or self.config["timeout"]
        
        # Prepare request kwargs
        kwargs = {
            "params": params,
            "headers": headers,
            "timeout": timeout,
            "verify": self.config["verify_ssl"]
        }
        
        # Add proxy if provided
        if proxy:
            kwargs["proxies"] = {
                "http": proxy,
                "https": proxy
            }
            
        # Make request with retry logic
        try:
            response = self.session.get(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"GET request failed: {url} - {str(e)}")
            raise
            
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        proxy: Optional[str] = None
    ) -> requests.Response:
        """
        Make a POST request with retry logic.
        
        Args:
            endpoint: API endpoint or full URL
            data: Form data
            json: JSON data
            headers: Additional headers
            timeout: Request timeout in seconds (overrides config)
            proxy: Proxy URL (overrides session proxy)
            
        Returns:
            Response object
        """
        url = self._get_full_url(endpoint)
        timeout = timeout or self.config["timeout"]
        
        # Prepare request kwargs
        kwargs = {
            "headers": headers,
            "timeout": timeout,
            "verify": self.config["verify_ssl"]
        }
        
        # Add data or JSON
        if data is not None:
            kwargs["data"] = data
        if json is not None:
            kwargs["json"] = json
            
        # Add proxy if provided
        if proxy:
            kwargs["proxies"] = {
                "http": proxy,
                "https": proxy
            }
            
        # Make request with retry logic
        try:
            response = self.session.post(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request failed: {url} - {str(e)}")
            raise
            
    def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """
        Make a generic request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint or full URL
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response object
        """
        url = self._get_full_url(endpoint)
        
        # Set default timeout
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.config["timeout"]
            
        # Set SSL verification
        if "verify" not in kwargs:
            kwargs["verify"] = self.config["verify_ssl"]
            
        # Make request with retry logic
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"{method} request failed: {url} - {str(e)}")
            raise
            
    def close(self):
        """Close the session."""
        self.session.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions for direct use without creating a client instance

def get(url: str, **kwargs) -> requests.Response:
    """Make a GET request with retry logic."""
    with HttpClient() as client:
        return client.get(url, **kwargs)
        
def post(url: str, **kwargs) -> requests.Response:
    """Make a POST request with retry logic."""
    with HttpClient() as client:
        return client.post(url, **kwargs)
        
def request(method: str, url: str, **kwargs) -> requests.Response:
    """Make a generic request with retry logic."""
    with HttpClient() as client:
        return client.request(method, url, **kwargs)
