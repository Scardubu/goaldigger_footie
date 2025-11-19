"""
Fallback Manager for Data Sources

This module provides a robust fallback mechanism for data sources, ensuring
that the GoalDiggers platform can continue to function even when primary data 
sources are unavailable or rate-limited.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dashboard.error_log import ErrorLog, log_exceptions_decorator

logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="fallback_manager")

class DataSourceFallbackManager:
    """Manages fallback strategies for data sources."""
    
    def __init__(self, primary_sources: List[Any], secondary_sources: List[Any], fallback_sources: List[Any]):
        """
        Initialize the fallback manager with data sources in priority order.
        
        Args:
            primary_sources: List of primary data sources to try first
            secondary_sources: List of secondary sources to try if primaries fail
            fallback_sources: List of last-resort fallback sources
        """
        self.primary_sources = primary_sources
        self.secondary_sources = secondary_sources
        self.fallback_sources = fallback_sources
        self.source_status = {}
        self.source_success_counts = {}
        self.source_failure_counts = {}
        self.last_reset_time = datetime.now()
    
    @log_exceptions_decorator
    async def get_data(self, data_type: str, identifier: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
        """
        Try to get data with automatic fallback mechanism.
        
        Args:
            data_type: Type of data being requested (e.g., "match_odds", "team_stats")
            identifier: Unique identifier for the data (e.g., match_id)
            timeout: Timeout in seconds for each source attempt
            
        Returns:
            Data if available from any source, or None if all sources fail
        """
        # Start tracking the request
        request_id = f"{data_type}_{identifier}_{time.time()}"
        
        # Try primary sources first
        result = await self._try_sources(self.primary_sources, data_type, identifier, timeout, "primary")
        if result:
            return result
            
        # Try secondary sources
        result = await self._try_sources(self.secondary_sources, data_type, identifier, timeout, "secondary")
        if result:
            return result
            
        # Last resort: fallback sources
        result = await self._try_sources(self.fallback_sources, data_type, identifier, timeout, "fallback")
        if result:
            return result
        
        # If we get here, all sources failed
        logger.error(f"All data sources failed for {data_type}:{identifier}")
        error_log.error(
            f"Complete data source failure for {data_type}",
            details={"identifier": identifier, "request_id": request_id},
            err_type="data_source_complete_failure",
            suggestion="Check API status, network connectivity, and rate limits for all sources"
        )
        return None
    
    async def _try_sources(self, sources: List[Any], data_type: str, identifier: str, 
                           timeout: int, source_tier: str) -> Optional[Dict[str, Any]]:
        """Try a list of sources until one succeeds or all fail."""
        for source in sources:
            source_name = source.__class__.__name__
            
            # Skip sources that have failed too many times recently
            if self._should_skip_source(source_name):
                logger.info(f"Skipping {source_name} due to recent failures")
                continue
                
            try:
                # Try to get data with timeout
                result = await asyncio.wait_for(source.get_data(data_type, identifier), timeout)
                
                if result:
                    # Record success
                    self._record_source_success(source_name)
                    return result
                else:
                    # Record soft failure (source returned None)
                    self._record_source_failure(source_name, Exception("No data returned"))
                    
            except asyncio.TimeoutError:
                logger.warning(f"{source_tier.capitalize()} source {source_name} timed out after {timeout}s")
                self._record_source_failure(source_name, asyncio.TimeoutError(f"Timeout after {timeout}s"))
                
            except Exception as e:
                logger.warning(f"{source_tier.capitalize()} source {source_name} failed: {e}")
                self._record_source_failure(source_name, e)
                
        return None
    
    def _record_source_success(self, source_name: str) -> None:
        """Record a successful data retrieval for a source."""
        if source_name not in self.source_success_counts:
            self.source_success_counts[source_name] = 0
            
        self.source_success_counts[source_name] += 1
        
        # Reset failure count after success
        if source_name in self.source_failure_counts:
            self.source_failure_counts[source_name] = 0
            
        # Update status
        self.source_status[source_name] = {
            "status": "healthy",
            "last_success": datetime.now(),
            "consecutive_failures": 0
        }
    
    def _record_source_failure(self, source_name: str, exception: Exception) -> None:
        """Record a failure for a source."""
        if source_name not in self.source_failure_counts:
            self.source_failure_counts[source_name] = 0
            
        self.source_failure_counts[source_name] += 1
        
        # Update status
        consecutive_failures = self.source_status.get(source_name, {}).get("consecutive_failures", 0) + 1
        
        self.source_status[source_name] = {
            "status": "degraded" if consecutive_failures < 5 else "failing",
            "last_failure": datetime.now(),
            "last_error": str(exception),
            "consecutive_failures": consecutive_failures
        }
    
    def _should_skip_source(self, source_name: str) -> bool:
        """Determine if a source should be skipped due to repeated failures."""
        source_info = self.source_status.get(source_name, {})
        consecutive_failures = source_info.get("consecutive_failures", 0)
        
        # Implement exponential backoff for failing sources
        if consecutive_failures >= 3:
            last_failure = source_info.get("last_failure")
            if last_failure:
                # Calculate backoff time based on consecutive failures
                # 2^failures seconds (e.g., 8s for 3 failures, 16s for 4 failures)
                backoff_seconds = min(2 ** consecutive_failures, 3600)  # Cap at 1 hour
                
                time_since_failure = (datetime.now() - last_failure).total_seconds()
                if time_since_failure < backoff_seconds:
                    return True  # Skip this source
                    
        return False
    
    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status for all data sources.
        
        Returns:
            Dictionary with health status for each source
        """
        health_report = {}
        
        # Combine all sources
        all_sources = self.primary_sources + self.secondary_sources + self.fallback_sources
        
        for source in all_sources:
            source_name = source.__class__.__name__
            
            # Default status if not yet recorded
            if source_name not in self.source_status:
                health_report[source_name] = {
                    "status": "unknown",
                    "tier": self._get_source_tier(source),
                    "success_count": 0,
                    "failure_count": 0,
                    "consecutive_failures": 0
                }
                continue
                
            status_info = self.source_status[source_name]
            health_report[source_name] = {
                "status": status_info.get("status", "unknown"),
                "tier": self._get_source_tier(source),
                "last_success": status_info.get("last_success"),
                "last_failure": status_info.get("last_failure"),
                "last_error": status_info.get("last_error"),
                "success_count": self.source_success_counts.get(source_name, 0),
                "failure_count": self.source_failure_counts.get(source_name, 0),
                "consecutive_failures": status_info.get("consecutive_failures", 0)
            }
            
        return health_report
    
    def _get_source_tier(self, source: Any) -> str:
        """Determine which tier a source belongs to."""
        if source in self.primary_sources:
            return "primary"
        elif source in self.secondary_sources:
            return "secondary"
        elif source in self.fallback_sources:
            return "fallback"
        return "unknown"
    
    def reset_stats(self) -> None:
        """Reset all statistics and source status."""
        self.source_status = {}
        self.source_success_counts = {}
        self.source_failure_counts = {}
        self.last_reset_time = datetime.now()
        
    def mark_source_as_failed(self, source_name: str, error: Optional[str] = None) -> None:
        """
        Manually mark a source as failed.
        
        Args:
            source_name: Name of the source
            error: Optional error message
        """
        self._record_source_failure(source_name, Exception(error or "Manually marked as failed"))
        
    def mark_source_as_recovered(self, source_name: str) -> None:
        """
        Manually mark a source as recovered.
        
        Args:
            source_name: Name of the source
        """
        self._record_source_success(source_name)

# Create fallback cache for holding previously fetched data as final fallback
class FallbackCache:
    """Cache for previously fetched data to use as final fallback."""
    
    def __init__(self, max_age_hours: int = 24, max_entries: int = 1000):
        """
        Initialize the fallback cache.
        
        Args:
            max_age_hours: Maximum age of cached data in hours
            max_entries: Maximum number of entries to store
        """
        self.cache = {}
        self.max_age_hours = max_age_hours
        self.max_entries = max_entries
        
    def add(self, data_type: str, identifier: str, data: Dict[str, Any]) -> None:
        """
        Add data to the fallback cache.
        
        Args:
            data_type: Type of data
            identifier: Unique identifier
            data: Data to cache
        """
        cache_key = f"{data_type}:{identifier}"
        self.cache[cache_key] = {
            "data": data,
            "timestamp": datetime.now(),
            "data_type": data_type,
            "identifier": identifier
        }
        
        # Cleanup if cache gets too large
        if len(self.cache) > self.max_entries:
            self._cleanup()
            
    def get(self, data_type: str, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Get data from the fallback cache if available and not expired.
        
        Args:
            data_type: Type of data
            identifier: Unique identifier
            
        Returns:
            Cached data if available and not expired, otherwise None
        """
        cache_key = f"{data_type}:{identifier}"
        if cache_key not in self.cache:
            return None
            
        # Check if data is expired
        entry = self.cache[cache_key]
        age_hours = (datetime.now() - entry["timestamp"]).total_seconds() / 3600
        
        if age_hours > self.max_age_hours:
            # Data is expired, remove it
            del self.cache[cache_key]
            return None
            
        return entry["data"]
        
    def _cleanup(self) -> None:
        """Remove expired entries and trim cache to max size."""
        # Remove expired entries
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self.cache.items():
            age_hours = (current_time - entry["timestamp"]).total_seconds() / 3600
            if age_hours > self.max_age_hours:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.cache[key]
            
        # If still too many entries, remove oldest
        if len(self.cache) > self.max_entries:
            # Sort by timestamp
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1]["timestamp"]
            )
            
            # Remove oldest entries
            entries_to_remove = len(self.cache) - self.max_entries
            for i in range(entries_to_remove):
                key, _ = sorted_entries[i]
                del self.cache[key]
                
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache = {}
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        # Count entries by data type
        data_type_counts = {}
        for key, entry in self.cache.items():
            data_type = entry["data_type"]
            if data_type not in data_type_counts:
                data_type_counts[data_type] = 0
            data_type_counts[data_type] += 1
            
        return {
            "total_entries": len(self.cache),
            "max_entries": self.max_entries,
            "max_age_hours": self.max_age_hours,
            "data_types": data_type_counts
        }

# Create a global instance of the fallback cache
fallback_cache = FallbackCache()
