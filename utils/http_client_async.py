"""
Asynchronous HTTP client utility with robust error handling and retry logic.
Provides a unified interface for making HTTP requests with configurable retry behavior.
"""
import asyncio
import logging
import random
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import backoff
from aiohttp import ClientError, ClientResponse, ClientSession, TCPConnector
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.config import Config

logger = logging.getLogger(__name__)


class HttpClientAsync:
    """
    Asynchronous HTTP client with robust error handling, retry logic, and proxy support.
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
        self.session = None
        self._create_session_lock = asyncio.Lock()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load HTTP client configuration from Config."""
        http_config = Config.get("http_client_async", Config.get("http_client", {}))
        
        # Set defaults if not in config
        defaults = {
            "timeout": 30,
            "max_retries": 3,
            "backoff_factor": 0.5,
            "retry_statuses": [429, 500, 502, 503, 504],
            "user_agent": "GoalDiggers/1.0",
            "verify_ssl": True,
            "max_connections": 100,
            "request_rate_limit": 10,  # requests per second
            "limit_per_host": 10
        }
        
        # Merge defaults with config
        for key, value in defaults.items():
            if key not in http_config:
                http_config[key] = value
                
        return http_config
    
    async def _create_session(self) -> aiohttp.ClientSession:
        """Create an aiohttp client session with appropriate configuration."""
        timeout = aiohttp.ClientTimeout(total=self.config["timeout"])
        
        # Configure TCPConnector with connection limits
        connector = TCPConnector(
            limit=self.config["max_connections"],
            limit_per_host=self.config.get("limit_per_host", 0),
            ssl=self.config["verify_ssl"]
        )
        
        # Set default headers
        headers = {
            "User-Agent": self.config["user_agent"],
            "Accept": "application/json"
        }
        
        return aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
    
    async def get_session(self) -> aiohttp.ClientSession:
        """
        Get the current session or create a new one if needed.
        
        Returns:
            Active aiohttp ClientSession
        """
        if self.session is None or self.session.closed:
            async with self._create_session_lock:
                if self.session is None or self.session.closed:
                    self.session = await self._create_session()
        
        return self.session
    
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
    
    async def get(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        proxy: Optional[str] = None,
        retry_options: Optional[Dict[str, Any]] = None
    ) -> ClientResponse:
        """
        Make an asynchronous GET request with retry logic.
        
        Args:
            endpoint: API endpoint or full URL
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout in seconds (overrides config)
            proxy: Proxy URL
            retry_options: Custom retry options
            
        Returns:
            ClientResponse object
        """
        return await self.request("GET", endpoint, params=params, headers=headers, 
                                 timeout=timeout, proxy=proxy, retry_options=retry_options)
    
    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        proxy: Optional[str] = None,
        retry_options: Optional[Dict[str, Any]] = None
    ) -> ClientResponse:
        """
        Make an asynchronous POST request with retry logic.
        
        Args:
            endpoint: API endpoint or full URL
            data: Form data
            json: JSON data
            headers: Additional headers
            timeout: Request timeout in seconds (overrides config)
            proxy: Proxy URL
            retry_options: Custom retry options
            
        Returns:
            ClientResponse object
        """
        return await self.request("POST", endpoint, data=data, json=json, headers=headers,
                                 timeout=timeout, proxy=proxy, retry_options=retry_options)
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, aiohttp.ClientConnectionError),
        max_tries=3,
        jitter=backoff.full_jitter,
        on_backoff=lambda details: logger.warning(
            f"Retrying request after error. Attempt {details['tries']} of 3. "
            f"Backing off {details['wait']:.1f} seconds."
        )
    )
    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> ClientResponse:
        """
        Make a generic asynchronous request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint or full URL
            **kwargs: Additional arguments to pass to aiohttp
            
        Returns:
            ClientResponse object
        """
        url = self._get_full_url(endpoint)
        
        # Extract special kwargs
        timeout = kwargs.pop("timeout", None)
        proxy = kwargs.pop("proxy", None)
        retry_options = kwargs.pop("retry_options", {})
        
        # Set client timeout if provided
        if timeout:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=timeout)
        
        # Set proxy if provided
        if proxy:
            kwargs["proxy"] = proxy
        
        # Get the active session
        session = await self.get_session()
        
        # Make request with retry logic
        try:
            response = await session.request(method, url, **kwargs)
            # Some tests mock raise_for_status with an AsyncMock; aiohttp's real method is sync.
            try:
                r = response.raise_for_status()
                if asyncio.iscoroutine(r):
                    await r
            except AttributeError:
                # Fallback: if raise_for_status missing entirely just continue
                logger.debug("Response object missing raise_for_status; continuing without status validation")
            return response
        except aiohttp.ClientResponseError as e:
            status = getattr(e, 'status', None)
            logger.error(f"{method} request failed: {url} - Status: {status} - {str(e)}")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"{method} request failed: {url} - {str(e)}")
            raise
        except asyncio.TimeoutError:
            logger.error(f"{method} request timed out: {url}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {method} request: {url} - {str(e)}")
            raise
    
    async def stream(
        self,
        method: str,
        endpoint: str,
        chunk_size: int = 1024,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream response content for large responses.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint or full URL
            chunk_size: Size of chunks to yield
            **kwargs: Additional arguments to pass to aiohttp
            
        Yields:
            Content chunks
        """
        response = await self.request(method, endpoint, **kwargs)
        
        try:
            async for chunk in response.content.iter_chunked(chunk_size):
                yield chunk
        finally:
            response.release()
    
    async def close(self):
        """Close the session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Singleton instance for easy access
_http_client = None


async def get_http_client(base_url: str = None) -> HttpClientAsync:
    """Get the singleton HTTP client instance."""
    global _http_client
    if _http_client is None:
        _http_client = HttpClientAsync(base_url)
    return _http_client


# Convenience functions for direct use without creating a client instance

async def get(url: str, **kwargs) -> ClientResponse:
    """Make an asynchronous GET request with retry logic."""
    client = await get_http_client()
    return await client.get(url, **kwargs)


async def post(url: str, **kwargs) -> ClientResponse:
    """Make an asynchronous POST request with retry logic."""
    client = await get_http_client()
    return await client.post(url, **kwargs)


async def request(method: str, url: str, **kwargs) -> ClientResponse:
    """Make a generic asynchronous request with retry logic."""
    client = await get_http_client()
    return await client.request(method, url, **kwargs)


async def stream(method: str, url: str, **kwargs) -> AsyncGenerator[bytes, None]:
    """Stream response content for large responses."""
    client = await get_http_client()
    async for chunk in client.stream(method, url, **kwargs):
        yield chunk
