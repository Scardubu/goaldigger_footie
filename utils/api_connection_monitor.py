#!/usr/bin/env python3
"""
API Connection Monitor - Real-time Health Checking

Monitors API connectivity and provides early warning before cascading failures.
Implements circuit breaker pattern for resilient operations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional

import aiohttp

logger = logging.getLogger(__name__)


class APIStatus(Enum):
    """API connection status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class APIHealth:
    """Health status for a single API endpoint."""
    endpoint: str
    status: APIStatus
    response_time_ms: float
    last_check: datetime
    consecutive_failures: int
    last_error: Optional[str] = None


class APIConnectionMonitor:
    """Monitor API connection health with circuit breaker pattern."""
    
    def __init__(self):
        self.health_cache: Dict[str, APIHealth] = {}
        self.check_interval = 60  # seconds
        self._monitoring = False
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Circuit breaker thresholds
        self.failure_threshold = 3
        self.timeout_threshold_ms = 5000  # 5 seconds
        self.degraded_threshold_ms = 2000  # 2 seconds
    
    async def __aenter__(self):
        """Start monitoring session."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close monitoring session."""
        await self._close_session()
    
    async def _ensure_session(self):
        """Ensure HTTP session is active."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
    
    async def _close_session(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None
    
    async def check_endpoint(self, endpoint: str, headers: Optional[Dict] = None) -> APIHealth:
        """
        Check health of a single API endpoint.
        
        Args:
            endpoint: Full URL to check (e.g., https://api.football-data.org/v4/competitions)
            headers: Optional headers for authentication
            
        Returns:
            APIHealth object with current status
        """
        await self._ensure_session()
        
        start_time = time.time()
        status = APIStatus.UNKNOWN
        error_msg = None
        
        try:
            async with self._session.get(endpoint, headers=headers) as response:
                response_time_ms = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    if response_time_ms < self.degraded_threshold_ms:
                        status = APIStatus.HEALTHY
                    else:
                        status = APIStatus.DEGRADED
                        error_msg = f"Slow response: {response_time_ms:.0f}ms"
                elif response.status == 429:
                    status = APIStatus.DEGRADED
                    error_msg = "Rate limited"
                elif response.status >= 500:
                    status = APIStatus.DOWN
                    error_msg = f"Server error: {response.status}"
                else:
                    status = APIStatus.DEGRADED
                    error_msg = f"Unexpected status: {response.status}"
                    
        except asyncio.TimeoutError:
            response_time_ms = (time.time() - start_time) * 1000
            status = APIStatus.DOWN
            error_msg = "Request timeout"
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            status = APIStatus.DOWN
            error_msg = str(e)
        
        # Get previous health status
        previous = self.health_cache.get(endpoint)
        consecutive_failures = 0
        
        if status in (APIStatus.DOWN, APIStatus.DEGRADED):
            consecutive_failures = (previous.consecutive_failures + 1) if previous else 1
        
        # Create health object
        health = APIHealth(
            endpoint=endpoint,
            status=status,
            response_time_ms=response_time_ms,
            last_check=datetime.now(),
            consecutive_failures=consecutive_failures,
            last_error=error_msg
        )
        
        # Update cache
        self.health_cache[endpoint] = health
        
        # Log status changes
        if previous and previous.status != status:
            logger.warning(f"API status changed: {endpoint} {previous.status.value} → {status.value}")
        
        return health
    
    async def check_all_endpoints(self, endpoints: Dict[str, Dict]) -> Dict[str, APIHealth]:
        """
        Check health of multiple API endpoints concurrently.
        
        Args:
            endpoints: Dict mapping endpoint names to {url, headers} configs
            
        Returns:
            Dict mapping endpoint names to APIHealth objects
        """
        tasks = {}
        for name, config in endpoints.items():
            url = config['url']
            headers = config.get('headers')
            tasks[name] = self.check_endpoint(url, headers)
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        health_status = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Health check failed for {name}: {result}")
                health_status[name] = APIHealth(
                    endpoint=name,
                    status=APIStatus.DOWN,
                    response_time_ms=0,
                    last_check=datetime.now(),
                    consecutive_failures=999,
                    last_error=str(result)
                )
            else:
                health_status[name] = result
        
        return health_status
    
    def get_overall_status(self) -> APIStatus:
        """Get overall API health status."""
        if not self.health_cache:
            return APIStatus.UNKNOWN
        
        statuses = [h.status for h in self.health_cache.values()]
        
        # If any critical API is down, overall is down
        if any(s == APIStatus.DOWN for s in statuses):
            return APIStatus.DOWN
        
        # If majority are degraded, overall is degraded
        if sum(1 for s in statuses if s == APIStatus.DEGRADED) > len(statuses) / 2:
            return APIStatus.DEGRADED
        
        # If at least one is healthy, overall is healthy
        if any(s == APIStatus.HEALTHY for s in statuses):
            return APIStatus.HEALTHY
        
        return APIStatus.UNKNOWN
    
    def should_use_fallback(self, endpoint: str) -> bool:
        """
        Determine if fallback should be used for an endpoint.
        Implements circuit breaker pattern.
        
        Args:
            endpoint: Endpoint URL or name
            
        Returns:
            True if fallback should be used
        """
        health = self.health_cache.get(endpoint)
        
        if not health:
            return False  # No data yet, try the API
        
        # Circuit breaker: use fallback if too many consecutive failures
        if health.consecutive_failures >= self.failure_threshold:
            logger.warning(f"Circuit breaker OPEN for {endpoint}: {health.consecutive_failures} consecutive failures")
            return True
        
        # Use fallback if endpoint is down
        if health.status == APIStatus.DOWN:
            return True
        
        return False
    
    def get_health_report(self) -> Dict:
        """
        Get comprehensive health report.
        
        Returns:
            Dict with overall status and per-endpoint details
        """
        return {
            'overall_status': self.get_overall_status().value,
            'timestamp': datetime.now().isoformat(),
            'endpoints': {
                endpoint: {
                    'status': health.status.value,
                    'response_time_ms': health.response_time_ms,
                    'last_check': health.last_check.isoformat(),
                    'consecutive_failures': health.consecutive_failures,
                    'last_error': health.last_error
                }
                for endpoint, health in self.health_cache.items()
            }
        }


# Singleton instance
_monitor = None


def get_api_monitor() -> APIConnectionMonitor:
    """Get global API monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = APIConnectionMonitor()
    return _monitor


async def check_football_apis() -> Dict[str, APIHealth]:
    """
    Quick health check for all Football APIs.
    
    Returns:
        Dict mapping API names to health status
    """
    import os
    
    endpoints = {
        'football_data': {
            'url': 'https://api.football-data.org/v4/competitions',
            'headers': {'X-Auth-Token': os.getenv('FOOTBALL_DATA_API_KEY')}
        },
    }
    
    # Add API-Football if key available
    api_football_key = os.getenv('API_FOOTBALL_KEY')
    if api_football_key:
        endpoints['api_football'] = {
            'url': 'https://v3.football.api-sports.io/status',
            'headers': {
                'x-rapidapi-key': api_football_key,
                'x-rapidapi-host': 'v3.football.api-sports.io'
            }
        }
    
    async with APIConnectionMonitor() as monitor:
        return await monitor.check_all_endpoints(endpoints)


# CLI interface
if __name__ == "__main__":
    import sys
    
    async def main():
        print("🔍 Checking Football API Health...")
        print("=" * 60)
        
        health_status = await check_football_apis()
        
        for name, health in health_status.items():
            status_icon = {
                APIStatus.HEALTHY: "✅",
                APIStatus.DEGRADED: "⚠️",
                APIStatus.DOWN: "❌",
                APIStatus.UNKNOWN: "❓"
            }.get(health.status, "❓")
            
            print(f"\n{status_icon} {name.upper()}")
            print(f"   Status: {health.status.value}")
            print(f"   Response Time: {health.response_time_ms:.0f}ms")
            print(f"   Last Check: {health.last_check.strftime('%H:%M:%S')}")
            
            if health.last_error:
                print(f"   Error: {health.last_error}")
        
        print("\n" + "=" * 60)
        
        # Exit code based on overall health
        monitor = get_api_monitor()
        overall = monitor.get_overall_status()
        
        if overall == APIStatus.DOWN:
            print("❌ Overall Status: DOWN")
            sys.exit(1)
        elif overall == APIStatus.DEGRADED:
            print("⚠️ Overall Status: DEGRADED")
            sys.exit(0)
        else:
            print("✅ Overall Status: HEALTHY")
            sys.exit(0)
    
    asyncio.run(main())
