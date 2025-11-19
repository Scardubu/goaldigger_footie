"""
Database Connection Pool Monitoring

This module provides advanced monitoring and management for database connection pools
to ensure optimal performance in production environments.
"""

import logging
import time
from datetime import datetime
from threading import Event, Thread
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class ConnectionPoolMonitor:
    """
    Monitors database connection pools and provides diagnostics and management.
    """
    
    def __init__(self, db_manager):
        """
        Initialize the connection pool monitor.
        
        Args:
            db_manager: DatabaseManager instance to monitor
        """
        self.db_manager = db_manager
        self.monitoring_thread: Optional[Thread] = None
        self.stop_event = Event()
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 100  # Keep the last 100 metrics readings
        self.monitoring_interval = 60  # Check every 60 seconds by default
    
    def start_monitoring(self, interval_seconds: int = 60):
        """
        Start monitoring the connection pool.
        
        Args:
            interval_seconds: Interval between checks in seconds
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.info("Monitoring thread is already running")
            return
            
        self.monitoring_interval = interval_seconds
        self.stop_event.clear()
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"Connection pool monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop monitoring the connection pool."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.stop_event.set()
            self.monitoring_thread.join(timeout=5.0)
            logger.info("Connection pool monitoring stopped")
    
    def _monitoring_loop(self):
        """Monitoring loop to periodically check pool status."""
        while not self.stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self._analyze_metrics(metrics)
                self._store_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Error in connection pool monitoring: {e}")
                
            # Wait for the next interval or until stop is requested
            self.stop_event.wait(self.monitoring_interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """
        Collect metrics about the connection pool.
        
        Returns:
            Dictionary with pool metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "pool": self.db_manager.get_pool_stats() if hasattr(self.db_manager, "get_pool_stats") else {},
        }
        
        # Add additional metrics
        try:
            # Check if database is responsive
            start_time = time.time()
            is_responsive = self.db_manager.test_connection()
            response_time = (time.time() - start_time) * 1000  # in ms
            
            metrics["database_responsive"] = is_responsive
            metrics["response_time_ms"] = response_time
            
            # Check if using fallback
            metrics["using_fallback"] = getattr(self.db_manager, "using_sqlite_fallback", False)
            
        except Exception as e:
            logger.error(f"Error collecting additional metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _analyze_metrics(self, metrics: Dict[str, Any]):
        """
        Analyze metrics and log warnings if needed.
        
        Args:
            metrics: Dictionary with pool metrics
        """
        pool = metrics.get("pool", {})
        
        # Check for high connection usage
        if pool:
            connections_in_use = pool.get("connections_in_use", 0)
            pool_size = pool.get("pool_size", 10)
            
            usage_ratio = connections_in_use / pool_size if pool_size > 0 else 0
            
            if usage_ratio > 0.8:
                logger.warning(f"Connection pool usage is high: {connections_in_use}/{pool_size} ({usage_ratio:.0%})")
            
            # Check for excessive overflow
            overflow = pool.get("overflow", 0)
            max_overflow = pool.get("max_overflow", 10)
            
            if overflow > 0:
                overflow_ratio = overflow / max_overflow if max_overflow > 0 else 0
                
                if overflow_ratio > 0.5:
                    logger.warning(f"Connection pool overflow is high: {overflow}/{max_overflow} ({overflow_ratio:.0%})")
        
        # Check for slow response time
        response_time = metrics.get("response_time_ms", 0)
        if response_time > 500:  # More than 500ms is slow
            logger.warning(f"Database response time is slow: {response_time:.2f}ms")
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        """
        Store metrics in history.
        
        Args:
            metrics: Dictionary with pool metrics
        """
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Get metrics history.
        
        Returns:
            List of metrics dictionaries
        """
        return self.metrics_history
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get the latest metrics.
        
        Returns:
            Dictionary with the latest metrics
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        return {}
    
    def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the connection pool.
        
        Returns:
            Dictionary with health check results
        """
        metrics = self._collect_metrics()
        self._store_metrics(metrics)
        
        pool = metrics.get("pool", {})
        response_time = metrics.get("response_time_ms", 0)
        is_responsive = metrics.get("database_responsive", False)
        
        # Calculate health status
        status = "healthy"
        warnings = []
        
        if not is_responsive:
            status = "unhealthy"
            warnings.append("Database is not responsive")
        elif response_time > 1000:  # More than 1 second
            status = "degraded"
            warnings.append(f"Slow database response time: {response_time:.2f}ms")
        
        if pool:
            connections_in_use = pool.get("connections_in_use", 0)
            pool_size = pool.get("pool_size", 10)
            
            if connections_in_use >= pool_size:
                status = "degraded"
                warnings.append(f"Connection pool is fully utilized: {connections_in_use}/{pool_size}")
                
            overflow = pool.get("overflow", 0)
            max_overflow = pool.get("max_overflow", 10)
            
            if overflow >= max_overflow:
                status = "degraded"
                warnings.append(f"Connection pool overflow is maxed out: {overflow}/{max_overflow}")
        
        return {
            "status": status,
            "warnings": warnings,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }


def create_pool_monitor(db_manager, auto_start: bool = True) -> ConnectionPoolMonitor:
    """
    Create a connection pool monitor for a database manager.
    
    Args:
        db_manager: DatabaseManager instance to monitor
        auto_start: Whether to automatically start monitoring
        
    Returns:
        ConnectionPoolMonitor instance
    """
    monitor = ConnectionPoolMonitor(db_manager)
    
    if auto_start:
        monitor.start_monitoring()
    
    return monitor