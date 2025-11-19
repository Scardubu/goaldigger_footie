#!/usr/bin/env python3
"""
API Monitoring System for GoalDiggers

This module provides real-time monitoring of API rate limits and data quality
for all external data sources used in the GoalDiggers platform. It includes:
1. Rate limit tracking for all API calls
2. Data quality metrics calculation
3. Alert system for rate limit warnings and data issues
4. Logging and reporting capabilities
"""

import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ApiSource(Enum):
    """Enumeration of all API sources used in the platform."""
    
    FOOTBALL_DATA = "football-data.org"
    UNDERSTAT = "understat.com"
    ODDS_API = "oddsapi.com"
    WEATHER_API = "openweathermap.org"
    CUSTOM = "custom"


class RateLimitStatus(Enum):
    """Enumeration of rate limit status levels."""
    
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    BLOCKED = "blocked"


class DataQuality(Enum):
    """Enumeration of data quality levels."""
    
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    INVALID = "invalid"


class ApiMonitor:
    """Central API monitoring system for tracking rate limits and data quality."""
    
    def __init__(self):
        """Initialize the API monitor."""
        self.rate_limits = {
            ApiSource.FOOTBALL_DATA: {"limit": 10, "window": 60, "remaining": 10},  # 10 calls per minute
            ApiSource.UNDERSTAT: {"limit": 20, "window": 60, "remaining": 20},      # 20 calls per minute
            ApiSource.ODDS_API: {"limit": 500, "window": 3600, "remaining": 500},   # 500 calls per hour
            ApiSource.WEATHER_API: {"limit": 60, "window": 60, "remaining": 60},    # 60 calls per minute
        }
        
        # Track API calls per source
        self.api_calls = {src: [] for src in ApiSource}
        
        # Track data quality metrics
        self.quality_metrics = {src: {} for src in ApiSource}
        
        # Initialize alert thresholds
        self.alert_thresholds = {
            "rate_limit_warning": 0.25,  # Alert when 25% of rate limit remains
            "rate_limit_critical": 0.1,  # Critical when 10% of rate limit remains
            "data_quality_threshold": 0.7  # Alert when data quality score falls below 70%
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        logger.info("API Monitor initialized")
    
    def register_alert_callback(self, callback: Callable[[str, str, Any], None]) -> None:
        """Register a callback function to be called when alerts are triggered.
        
        Args:
            callback: Function that takes (alert_type, message, data) parameters
        """
        self.alert_callbacks.append(callback)
        logger.debug(f"Alert callback registered: {callback.__name__ if hasattr(callback, '__name__') else 'unknown'}")
    
    def trigger_alert(self, alert_type: str, message: str, data: Optional[Any] = None) -> None:
        """Trigger alerts using all registered callbacks.
        
        Args:
            alert_type: Type of alert (rate_limit, data_quality, etc.)
            message: Alert message
            data: Additional data for the alert
        """
        logger.warning(f"ALERT [{alert_type}]: {message}")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message, data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def track_api_call(self, source: ApiSource, endpoint: str, status_code: int, 
                      response_time: float, response_size: int) -> None:
        """Track an API call and update rate limit information.
        
        Args:
            source: API source enum
            endpoint: API endpoint called
            status_code: HTTP status code received
            response_time: Time taken for the response in seconds
            response_size: Size of the response in bytes
        """
        timestamp = datetime.now()
        
        # Record call details
        call_info = {
            "timestamp": timestamp,
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time": response_time,
            "response_size": response_size
        }
        
        self.api_calls[source].append(call_info)
        
        # Clean up old calls outside the window
        self._clean_old_calls(source)
        
        # Update rate limit remaining
        if source in self.rate_limits:
            window = self.rate_limits[source]["window"]
            calls_in_window = [call for call in self.api_calls[source] 
                             if call["timestamp"] > timestamp - timedelta(seconds=window)]
            used = len(calls_in_window)
            limit = self.rate_limits[source]["limit"]
            self.rate_limits[source]["remaining"] = max(0, limit - used)
            
            # Check for rate limit warnings/alerts
            self._check_rate_limit_status(source)
        
        logger.debug(f"API call tracked: {source.value} - {endpoint} - {status_code}")
    
    def _clean_old_calls(self, source: ApiSource) -> None:
        """Clean up old API call records.
        
        Args:
            source: API source to clean up
        """
        if source not in self.rate_limits:
            return
            
        window = self.rate_limits[source]["window"]
        cutoff = datetime.now() - timedelta(seconds=window * 2)  # Keep 2x the window for analysis
        
        self.api_calls[source] = [call for call in self.api_calls[source] 
                                if call["timestamp"] > cutoff]
    
    def _check_rate_limit_status(self, source: ApiSource) -> RateLimitStatus:
        """Check the current rate limit status and trigger alerts if needed.
        
        Args:
            source: API source to check
            
        Returns:
            Current rate limit status
        """
        if source not in self.rate_limits:
            return RateLimitStatus.OK
            
        limit = self.rate_limits[source]["limit"]
        remaining = self.rate_limits[source]["remaining"]
        remaining_percent = remaining / limit if limit > 0 else 1.0
        
        # Determine status
        status = RateLimitStatus.OK
        if remaining <= 0:
            status = RateLimitStatus.BLOCKED
            self.trigger_alert(
                "rate_limit_blocked",
                f"API rate limit reached for {source.value}. API calls will be blocked.",
                {"source": source.value, "limit": limit, "remaining": 0}
            )
        elif remaining_percent <= self.alert_thresholds["rate_limit_critical"]:
            status = RateLimitStatus.CRITICAL
            self.trigger_alert(
                "rate_limit_critical",
                f"Critical rate limit for {source.value}: {remaining}/{limit} calls remaining",
                {"source": source.value, "limit": limit, "remaining": remaining}
            )
        elif remaining_percent <= self.alert_thresholds["rate_limit_warning"]:
            status = RateLimitStatus.WARNING
            self.trigger_alert(
                "rate_limit_warning",
                f"Rate limit warning for {source.value}: {remaining}/{limit} calls remaining",
                {"source": source.value, "limit": limit, "remaining": remaining}
            )
            
        return status
    
    def evaluate_data_quality(self, source: ApiSource, data: Any, 
                            expected_fields: Optional[List[str]] = None,
                            expected_size: Optional[int] = None) -> Tuple[DataQuality, float, Dict[str, Any]]:
        """Evaluate the quality of data received from an API.
        
        Args:
            source: API source
            data: Data to evaluate
            expected_fields: Expected fields in the data
            expected_size: Expected size of the data
            
        Returns:
            Tuple of (quality_level, quality_score, metrics)
        """
        metrics = {}
        
        # Initialize quality score
        quality_score = 1.0
        
        # Check for None or empty data
        if data is None:
            metrics["is_null"] = True
            return DataQuality.INVALID, 0.0, metrics
        
        # Different handling based on data type
        if isinstance(data, dict):
            metrics["data_type"] = "dict"
            metrics["size"] = len(data)
            
            # Check for expected fields
            if expected_fields:
                present_fields = [field for field in expected_fields if field in data]
                metrics["field_coverage"] = len(present_fields) / len(expected_fields)
                quality_score *= metrics["field_coverage"]
                metrics["missing_fields"] = [f for f in expected_fields if f not in data]
            
            # Check values (detect null values, etc.)
            null_values = sum(1 for v in data.values() if v is None)
            metrics["null_value_percentage"] = null_values / len(data) if data else 1.0
            quality_score *= (1 - metrics["null_value_percentage"])
            
        elif isinstance(data, list):
            metrics["data_type"] = "list"
            metrics["size"] = len(data)
            
            # Check size expectations
            if expected_size is not None:
                size_ratio = min(len(data) / expected_size, 1.0) if expected_size > 0 else 0.0
                metrics["size_ratio"] = size_ratio
                quality_score *= metrics["size_ratio"]
            
            # Check items if list is not empty
            if data:
                if all(isinstance(item, dict) for item in data):
                    # For list of dictionaries, check field consistency
                    if expected_fields:
                        field_coverage = []
                        for item in data:
                            present = sum(1 for field in expected_fields if field in item)
                            field_coverage.append(present / len(expected_fields))
                        
                        metrics["avg_field_coverage"] = sum(field_coverage) / len(field_coverage)
                        quality_score *= metrics["avg_field_coverage"]
        
        elif isinstance(data, pd.DataFrame):
            metrics["data_type"] = "dataframe"
            metrics["size"] = len(data)
            metrics["columns"] = list(data.columns)
            
            # Check for nulls
            null_percentage = data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) if data.size > 0 else 1.0
            metrics["null_percentage"] = null_percentage
            quality_score *= (1 - null_percentage)
            
            # Check for expected columns
            if expected_fields:
                column_coverage = sum(1 for col in expected_fields if col in data.columns) / len(expected_fields)
                metrics["column_coverage"] = column_coverage
                quality_score *= column_coverage
        
        else:
            metrics["data_type"] = str(type(data))
            quality_score = 0.5  # Default score for unhandled types
        
        # Determine quality level based on score
        quality_level = DataQuality.EXCELLENT
        if quality_score < 0.5:
            quality_level = DataQuality.POOR
        elif quality_score < 0.7:
            quality_level = DataQuality.FAIR
        elif quality_score < 0.9:
            quality_level = DataQuality.GOOD
        
        # Store metrics for this source
        self.quality_metrics[source] = {
            "last_score": quality_score,
            "last_level": quality_level,
            "last_metrics": metrics,
            "timestamp": datetime.now()
        }
        
        # Check for data quality alerts
        if quality_score < self.alert_thresholds["data_quality_threshold"]:
            self.trigger_alert(
                "data_quality",
                f"Data quality issue detected for {source.value}: score={quality_score:.2f}",
                {"source": source.value, "score": quality_score, "metrics": metrics}
            )
        
        return quality_level, quality_score, metrics
    
    def get_rate_limit_status(self, source: ApiSource) -> Dict[str, Any]:
        """Get the current rate limit status for an API source.
        
        Args:
            source: API source
            
        Returns:
            Dictionary with rate limit information
        """
        if source not in self.rate_limits:
            return {"status": RateLimitStatus.OK, "limit": None, "remaining": None, "used": 0}
        
        limit_info = self.rate_limits[source]
        window = limit_info["window"]
        calls_in_window = [call for call in self.api_calls[source] 
                         if call["timestamp"] > datetime.now() - timedelta(seconds=window)]
        used = len(calls_in_window)
        
        remaining = max(0, limit_info["limit"] - used)
        status = RateLimitStatus.OK
        
        remaining_percent = remaining / limit_info["limit"] if limit_info["limit"] > 0 else 1.0
        if remaining <= 0:
            status = RateLimitStatus.BLOCKED
        elif remaining_percent <= self.alert_thresholds["rate_limit_critical"]:
            status = RateLimitStatus.CRITICAL
        elif remaining_percent <= self.alert_thresholds["rate_limit_warning"]:
            status = RateLimitStatus.WARNING
        
        return {
            "status": status,
            "limit": limit_info["limit"],
            "window": limit_info["window"],
            "window_unit": "seconds",
            "remaining": remaining,
            "used": used
        }
    
    def get_api_stats(self, source: ApiSource) -> Dict[str, Any]:
        """Get API usage statistics for a source.
        
        Args:
            source: API source
            
        Returns:
            Dictionary with API statistics
        """
        if not self.api_calls.get(source):
            return {
                "total_calls": 0,
                "avg_response_time": None,
                "error_rate": 0,
                "last_call": None
            }
        
        calls = self.api_calls[source]
        
        # Calculate error rate (non-2xx responses)
        error_count = sum(1 for call in calls if call["status_code"] < 200 or call["status_code"] >= 300)
        error_rate = error_count / len(calls) if calls else 0
        
        # Calculate average response time
        avg_response_time = sum(call["response_time"] for call in calls) / len(calls) if calls else 0
        
        # Get last call time
        last_call = max(call["timestamp"] for call in calls) if calls else None
        
        return {
            "total_calls": len(calls),
            "avg_response_time": avg_response_time,
            "error_rate": error_rate,
            "last_call": last_call
        }
    
    def get_data_quality_metrics(self, source: ApiSource) -> Dict[str, Any]:
        """Get data quality metrics for a source.
        
        Args:
            source: API source
            
        Returns:
            Dictionary with data quality metrics
        """
        if source not in self.quality_metrics:
            return {"last_score": None, "last_level": None, "last_metrics": {}, "timestamp": None}
        
        return self.quality_metrics[source]
    
    def get_all_sources_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all API sources.
        
        Returns:
            Dictionary with status for each source
        """
        status = {}
        for source in ApiSource:
            rate_limit = self.get_rate_limit_status(source)
            api_stats = self.get_api_stats(source)
            quality_metrics = self.get_data_quality_metrics(source)
            
            status[source.value] = {
                "rate_limit": rate_limit,
                "api_stats": api_stats,
                "quality": quality_metrics
            }
        
        return status
    
    def update_rate_limit_config(self, source: ApiSource, limit: int, window: int) -> None:
        """Update rate limit configuration for a source.
        
        Args:
            source: API source
            limit: Maximum calls allowed
            window: Time window in seconds
        """
        if source in self.rate_limits:
            old_limit = self.rate_limits[source]["limit"]
            old_window = self.rate_limits[source]["window"]
            
            self.rate_limits[source]["limit"] = limit
            self.rate_limits[source]["window"] = window
            
            logger.info(f"Updated rate limit for {source.value}: {old_limit}/{old_window}s -> {limit}/{window}s")
        else:
            self.rate_limits[source] = {"limit": limit, "window": window, "remaining": limit}
            logger.info(f"Added new rate limit for {source.value}: {limit}/{window}s")


# API monitoring decorator
def monitor_api_call(source: ApiSource, endpoint: str = None):
    """Decorator to monitor API calls.
    
    Args:
        source: API source enum
        endpoint: API endpoint (optional, will be extracted from function name if not provided)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get API monitor instance
            api_monitor = _get_api_monitor_instance()
            
            # Get endpoint name if not provided
            nonlocal endpoint
            if endpoint is None:
                endpoint = func.__name__
            
            # Check rate limit before making call
            rate_limit_status = api_monitor.get_rate_limit_status(source)
            if rate_limit_status["status"] == RateLimitStatus.BLOCKED:
                logger.error(f"API call blocked due to rate limit: {source.value} - {endpoint}")
                raise RateLimitException(f"Rate limit exceeded for {source.value}")
            
            # Make the API call and measure performance
            start_time = time.time()
            response = None
            status_code = 500
            response_size = 0
            
            try:
                response = func(*args, **kwargs)
                
                # Try to get status code and size
                if hasattr(response, 'status_code'):
                    status_code = response.status_code
                elif isinstance(response, dict) and 'status_code' in response:
                    status_code = response['status_code']
                else:
                    status_code = 200  # Assume success
                
                if hasattr(response, 'content'):
                    response_size = len(response.content)
                elif isinstance(response, (dict, list)):
                    import json
                    try:
                        response_size = len(json.dumps(response))
                    except:
                        response_size = 0
                elif isinstance(response, str):
                    response_size = len(response)
                
            except Exception as e:
                status_code = 500  # Internal error
                logger.exception(f"Error in monitored API call to {source.value} - {endpoint}: {e}")
                raise
            finally:
                response_time = time.time() - start_time
                
                # Track the call
                api_monitor.track_api_call(
                    source=source,
                    endpoint=endpoint,
                    status_code=status_code,
                    response_time=response_time,
                    response_size=response_size
                )
            
            return response
        
        return wrapper
    
    return decorator


class RateLimitException(Exception):
    """Exception raised when an API rate limit is exceeded."""
    pass


# Singleton instance
_api_monitor_instance = None

def get_api_monitor() -> ApiMonitor:
    """Get the global API monitor instance.
    
    Returns:
        ApiMonitor instance
    """
    global _api_monitor_instance
    if _api_monitor_instance is None:
        _api_monitor_instance = ApiMonitor()
    return _api_monitor_instance

# Alias for internal use
_get_api_monitor_instance = get_api_monitor


# Sample usage of the API monitor
if __name__ == "__main__":
    # Setup logging
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from utils.logging_config import configure_logging  # type: ignore
            configure_logging()
        except Exception:
            logging.basicConfig(level=logging.INFO)
    
    # Get monitor instance
    monitor = get_api_monitor()
    
    # Register a simple alert callback
    def print_alert(alert_type, message, data):
        print(f"ALERT [{alert_type}]: {message}")
    
    monitor.register_alert_callback(print_alert)
    
    # Example API call function with monitoring
    @monitor_api_call(ApiSource.FOOTBALL_DATA, "get_matches")
    def get_football_data_matches():
        # Simulate API call
        time.sleep(0.5)  # 500ms response time
        return {"matches": ["match1", "match2", "match3"], "status_code": 200}
    
    # Make some API calls
    for _ in range(5):
        result = get_football_data_matches()
        print(f"API call result: {result}")
    
    # Test data quality evaluation
    test_data = [
        {"id": 1, "name": "Match 1", "date": "2025-01-01"},
        {"id": 2, "name": "Match 2", "date": None},
        {"id": 3, "name": None, "score": "2-1"}
    ]
    
    expected_fields = ["id", "name", "date", "score"]
    quality, score, metrics = monitor.evaluate_data_quality(
        ApiSource.FOOTBALL_DATA, 
        test_data,
        expected_fields=expected_fields
    )
    
    print(f"Data quality: {quality.value}, Score: {score:.2f}")
    print(f"Metrics: {metrics}")
    
    # Get overall status
    status = monitor.get_all_sources_status()
    for src, info in status.items():
        print(f"\nSource: {src}")
        print(f"Rate limit: {info['rate_limit']['remaining']}/{info['rate_limit']['limit']}")
        print(f"API stats: {info['api_stats']['total_calls']} calls, {info['api_stats']['avg_response_time']:.2f}s avg")
