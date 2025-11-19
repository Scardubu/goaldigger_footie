"""
Comprehensive Monitoring System for GoalDiggers Platform
Provides centralized logging, performance monitoring, and metrics collection.
"""
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import psutil

from utils.comprehensive_error_handler import (
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    get_error_handler,
)

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemMetrics:
    """Container for system-level metrics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_connections: int = 0
    active_threads: int = 0
    active_processes: int = 0

class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, max_metrics: int = 10000):
        self._metrics: deque = deque(maxlen=max_metrics)
        self._aggregated_metrics: Dict[str, Dict[str, Union[int, float]]] = defaultdict(dict)
        self._lock = threading.Lock()

    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        with self._lock:
            self._metrics.append(metric)
            self._update_aggregated_metrics(metric)

    def _update_aggregated_metrics(self, metric: PerformanceMetrics):
        """Update aggregated metrics."""
        if metric.duration is None:
            return

        op_name = metric.operation_name

        # Initialize if not exists
        if op_name not in self._aggregated_metrics:
            self._aggregated_metrics[op_name] = {
                "count": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "min_duration": float('inf'),
                "max_duration": 0.0,
                "total_memory": 0.0,
                "avg_memory": 0.0
            }

        agg = self._aggregated_metrics[op_name]
        agg["count"] += 1
        agg["total_duration"] += metric.duration
        agg["avg_duration"] = agg["total_duration"] / agg["count"]
        agg["min_duration"] = min(agg["min_duration"], metric.duration)
        agg["max_duration"] = max(agg["max_duration"], metric.duration)

        if metric.memory_usage:
            agg["total_memory"] += metric.memory_usage
            agg["avg_memory"] = agg["total_memory"] / agg["count"]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self._lock:
            return {
                "total_metrics": len(self._metrics),
                "aggregated_metrics": dict(self._aggregated_metrics),
                "recent_metrics": [
                    {
                        "operation": m.operation_name,
                        "duration": m.duration,
                        "memory": m.memory_usage,
                        "timestamp": datetime.fromtimestamp(m.start_time).isoformat()
                    }
                    for m in list(self._metrics)[-10:]  # Last 10 metrics
                ]
            }

    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, Union[int, float]]]:
        """Get statistics for a specific operation."""
        with self._lock:
            return self._aggregated_metrics.get(operation_name)

class SystemMonitor:
    """Monitors system resources and performance."""

    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self._system_metrics: deque = deque(maxlen=1000)
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start_monitoring(self):
        """Start system monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                with self._lock:
                    self._system_metrics.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            # Network connections
            network_connections = len(psutil.net_connections())

            # Process information
            active_threads = threading.active_count()
            active_processes = len(psutil.pids())

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_connections=network_connections,
                active_threads=active_threads,
                active_processes=active_processes
            )

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics()

    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self._collect_system_metrics()

    def get_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Get system metrics history for the specified hours."""
        with self._lock:
            # Calculate how many metrics to return based on collection interval
            metrics_per_hour = 3600 // self.collection_interval
            num_metrics = hours * metrics_per_hour

            return list(self._system_metrics)[-num_metrics:]

    def get_system_health_score(self) -> float:
        """Calculate system health score (0-100)."""
        try:
            current = self.get_current_metrics()

            # Calculate health score based on various metrics
            cpu_score = max(0, 100 - current.cpu_percent)
            memory_score = max(0, 100 - current.memory_percent)
            disk_score = max(0, 100 - current.disk_usage_percent)

            # Weighted average
            health_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)

            return round(health_score, 2)

        except Exception as e:
            logger.error(f"Failed to calculate health score: {e}")
            return 0.0

class PerformanceMonitor:
    """Monitors application performance with timing and resource tracking."""

    def __init__(self):
        self._active_operations: Dict[str, PerformanceMetrics] = {}
        self._metrics_collector = MetricsCollector()
        self._system_monitor = SystemMonitor()
        self._lock = threading.Lock()

    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start monitoring an operation.

        Args:
            operation_name: Name of the operation
            metadata: Additional metadata

        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"

        with self._lock:
            self._active_operations[operation_id] = PerformanceMetrics(
                operation_name=operation_name,
                start_time=time.time(),
                metadata=metadata or {}
            )

        return operation_id

    def end_operation(self, operation_id: str) -> Optional[PerformanceMetrics]:
        """
        End monitoring an operation.

        Args:
            operation_id: Operation ID from start_operation

        Returns:
            PerformanceMetrics for the completed operation
        """
        with self._lock:
            if operation_id not in self._active_operations:
                logger.warning(f"Operation {operation_id} not found")
                return None

            metric = self._active_operations.pop(operation_id)
            metric.end_time = time.time()
            metric.duration = metric.end_time - metric.start_time

            # Collect resource usage
            try:
                process = psutil.Process()
                metric.memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                metric.cpu_usage = process.cpu_percent()
            except Exception as e:
                logger.debug(f"Failed to collect resource metrics: {e}")

            # Record the metric
            self._metrics_collector.record_metric(metric)

            return metric

    def monitor_function(self, operation_name: Optional[str] = None):
        """
        Decorator to monitor function performance.

        Args:
            operation_name: Custom operation name (defaults to function name)
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                operation_id = self.start_operation(op_name, {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                })

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    metric = self.end_operation(operation_id)
                    if metric and metric.duration:
                        logger.debug(f"Operation {op_name} completed in {metric.duration:.3f}s")

            return wrapper
        return decorator

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        from datetime import timezone
        return {
            "metrics_summary": self._metrics_collector.get_metrics_summary(),
            "system_health": {
                "current_health_score": self._system_monitor.get_system_health_score(),
                "current_metrics": self._system_monitor.get_current_metrics().__dict__,
                "metrics_history_count": len(self._system_monitor.get_metrics_history(1))
            },
            "active_operations": len(self._active_operations),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Add structured formatter if not already present
        if not any(isinstance(handler.formatter, StructuredFormatter)
                  for handler in self.logger.handlers):
            formatter = StructuredFormatter()
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_performance(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Log performance information."""
        self.logger.info("Performance metric", extra={
            "operation": operation,
            "duration": duration,
            "metadata": metadata or {},
            "log_type": "performance"
        })

    def log_error(self, error: Exception, context: Optional[ErrorContext] = None,
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Log error with context."""
        context = context or ErrorContext()

        self.logger.error("Application error", extra={
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "severity": severity.value,
            "context": {
                "component": context.component,
                "operation": context.operation,
                "user_id": context.user_id,
                "session_id": context.session_id,
                "metadata": context.metadata
            },
            "log_type": "error"
        })

    def log_api_call(self, endpoint: str, method: str, status_code: int,
                    duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Log API call information."""
        self.logger.info("API call", extra={
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration": duration,
            "metadata": metadata or {},
            "log_type": "api"
        })

    def log_user_action(self, user_id: str, action: str, metadata: Optional[Dict[str, Any]] = None):
        """Log user action."""
        self.logger.info("User action", extra={
            "user_id": user_id,
            "action": action,
            "metadata": metadata or {},
            "log_type": "user_action"
        })

    def log_system_event(self, event_type: str, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log system event."""
        self.logger.info("System event", extra={
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {},
            "log_type": "system"
        })

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        from datetime import timezone

        # Add timestamp if not present
        if not hasattr(record, 'timestamp'):
            record.timestamp = datetime.now(timezone.utc).isoformat()

        # Create structured log entry
        log_entry = {
            "timestamp": record.timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        # Add extra fields
        if hasattr(record, 'log_type'):
            log_entry["type"] = record.log_type

        # Add any extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)

class MonitoringDashboard:
    """Dashboard for monitoring system health and performance."""

    def __init__(self):
        self._performance_monitor = PerformanceMonitor()
        self._structured_logger = StructuredLogger("monitoring")
        self._alerts: List[Dict[str, Any]] = []
        self._alert_thresholds = {
            "cpu_usage_percent": 80.0,
            "memory_usage_percent": 85.0,
            "response_time_seconds": 5.0,
            "error_rate_percent": 5.0
        }

    def start_monitoring(self):
        """Start all monitoring systems."""
        self._performance_monitor._system_monitor.start_monitoring()
        self._structured_logger.log_system_event(
            "monitoring_started",
            "Comprehensive monitoring system activated"
        )

    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self._performance_monitor._system_monitor.stop_monitoring()
        self._structured_logger.log_system_event(
            "monitoring_stopped",
            "Comprehensive monitoring system deactivated"
        )

    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for system alerts based on thresholds."""
        alerts = []

        # System health alerts
        health_score = self._performance_monitor._system_monitor.get_system_health_score()
        if health_score < 70:
            alerts.append({
                "type": "system_health",
                "severity": "high",
                "message": f"System health score is low: {health_score}",
                "value": health_score,
                "threshold": 70
            })

        # Performance alerts
        metrics_summary = self._performance_monitor._metrics_collector.get_metrics_summary()
        for op_name, stats in metrics_summary.get("aggregated_metrics", {}).items():
            avg_duration = stats.get("avg_duration", 0)
            if avg_duration > self._alert_thresholds["response_time_seconds"]:
                alerts.append({
                    "type": "performance",
                    "severity": "medium",
                    "message": f"Operation {op_name} is slow: {avg_duration:.2f}s",
                    "value": avg_duration,
                    "threshold": self._alert_thresholds["response_time_seconds"]
                })

        # Log new alerts
        for alert in alerts:
            if alert not in self._alerts:
                self._structured_logger.logger.warning("System alert", extra={
                    "alert_type": alert["type"],
                    "severity": alert["severity"],
                    "message": alert["message"],
                    "log_type": "alert"
                })

        self._alerts = alerts
        return alerts

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "performance_report": self._performance_monitor.get_performance_report(),
            "alerts": self.check_alerts(),
            "system_info": {
                "python_version": __import__('sys').version,
                "platform": __import__('platform').platform(),
                "cpu_count": __import__('os').cpu_count(),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Global instances
_performance_monitor = PerformanceMonitor()
_structured_logger = StructuredLogger("goaldiggers")
_monitoring_dashboard = MonitoringDashboard()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _performance_monitor

def get_structured_logger(name: str = "goaldiggers") -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)

def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get the monitoring dashboard."""
    return _monitoring_dashboard

# Convenience functions
def monitor_function(operation_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    return _performance_monitor.monitor_function(operation_name)

def log_performance(operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
    """Log performance information."""
    _structured_logger.log_performance(operation, duration, metadata)

def log_error(error: Exception, context: Optional[ErrorContext] = None,
             severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Log error with context."""
    _structured_logger.log_error(error, context, severity)

# Initialize monitoring on import
_monitoring_dashboard.start_monitoring()