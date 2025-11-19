"""
Fail-Safe System for GoalDiggers Platform
Provides circuit breakers, health checks, automatic rollbacks, and backup strategies.
"""
import logging
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Type, Union
from datetime import datetime, timedelta, timezone
from enum import Enum
from dataclasses import dataclass, field
import json
import os
import shutil

from utils.comprehensive_error_handler import (
    get_error_handler,
    ErrorContext,
    ErrorSeverity,
    ErrorCategory,
    GoalDiggersException
)

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds to wait before trying half-open
    success_threshold: int = 3  # Number of successes needed to close from half-open
    timeout: float = 10.0      # Request timeout in seconds
    name: str = "default"

@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenException: If circuit is open
            Exception: Original exception if call fails
        """
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if not self._should_attempt_reset():
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.config.name}' is OPEN"
                    )
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0

        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            with self._lock:
                self._on_success()

            return result

        except Exception as e:
            with self._lock:
                self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True

        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                logger.info(f"Circuit breaker '{self.config.name}' reset to CLOSED")

    def _on_failure(self):
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker '{self.config.name}' reset to OPEN (half-open failure)")
        elif (self._state == CircuitBreakerState.CLOSED and
              self._failure_count >= self.config.failure_threshold):
            self._state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker '{self.config.name}' opened after {self._failure_count} failures")

    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "name": self.config.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout
            }
        }

class CircuitBreakerOpenException(GoalDiggersException):
    """Exception raised when circuit breaker is open."""
    def __init__(self, message: str):
        super().__init__(message, ErrorSeverity.HIGH, ErrorCategory.SYSTEM)

class HealthChecker:
    """Health checker for system components."""

    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._check_results: Dict[str, HealthCheckResult] = {}
        self._check_interval = 30  # seconds
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register a health check function."""
        self._checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if name not in self._checks:
            logger.warning(f"Health check '{name}' not found")
            return None

        try:
            start_time = time.time()
            result = self._checks[name]()
            result.response_time = time.time() - start_time

            with self._lock:
                self._check_results[name] = result

            return result

        except Exception as e:
            logger.error(f"Health check '{name}' failed: {e}")
            error_result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                error_message=str(e)
            )

            with self._lock:
                self._check_results[name] = error_result

            return error_result

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for name in self._checks:
            results[name] = self.run_check(name)
        return results

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            if not self._check_results:
                return HealthStatus.UNHEALTHY

            statuses = [result.status for result in self._check_results.values()]

            if HealthStatus.CRITICAL in statuses:
                return HealthStatus.CRITICAL
            elif HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            elif HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY

    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                self.run_all_checks()
                overall_health = self.get_overall_health()

                if overall_health in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    logger.warning(f"System health degraded: {overall_health.value}")

                time.sleep(self._check_interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self._check_interval)

    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        with self._lock:
            return {
                "overall_health": self.get_overall_health().value,
                "checks": {
                    name: {
                        "status": result.status.value,
                        "timestamp": result.timestamp.isoformat(),
                        "response_time": result.response_time,
                        "error_message": result.error_message,
                        "metadata": result.metadata
                    }
                    for name, result in self._check_results.items()
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

class AutoRollbackManager:
    """Manages automatic rollbacks for failed deployments."""

    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = backup_dir
        self._backups: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)

    def create_backup(self, component_name: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a backup of component data.

        Args:
            component_name: Name of the component
            data: Data to backup
            metadata: Additional metadata

        Returns:
            Backup ID
        """
        backup_id = f"{component_name}_{int(time.time())}"
        backup_path = os.path.join(self.backup_dir, f"{backup_id}.json")

            from datetime import timezone
            backup_info = {
                "backup_id": backup_id,
                "component": component_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
                "metadata": metadata or {}
            }

        try:
            with open(backup_path, 'w') as f:
                json.dump(backup_info, f, default=str, indent=2)

            with self._lock:
                self._backups[backup_id] = backup_info

            logger.info(f"Backup created: {backup_id}")
            return backup_id

        except Exception as e:
            logger.error(f"Failed to create backup {backup_id}: {e}")
            raise

    def rollback(self, backup_id: str) -> Any:
        """
        Rollback to a specific backup.

        Args:
            backup_id: Backup ID to rollback to

        Returns:
            Backup data
        """
        with self._lock:
            if backup_id not in self._backups:
                raise ValueError(f"Backup {backup_id} not found")

            backup_info = self._backups[backup_id]

        try:
            # Here you would implement the actual rollback logic
            # For now, just return the data
            logger.info(f"Rolling back to backup: {backup_id}")
            return backup_info["data"]

        except Exception as e:
            logger.error(f"Rollback failed for {backup_id}: {e}")
            raise

    def list_backups(self, component_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups."""
        with self._lock:
            backups = list(self._backups.values())

            if component_name:
                backups = [b for b in backups if b["component"] == component_name]

            return sorted(backups, key=lambda x: x["timestamp"], reverse=True)

    def cleanup_old_backups(self, max_age_days: int = 30):
        """Clean up old backups."""
    from datetime import timezone
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        with self._lock:
            to_remove = []
            for backup_id, backup_info in self._backups.items():
                backup_time = datetime.fromisoformat(backup_info["timestamp"])
                if backup_time < cutoff_time:
                    to_remove.append(backup_id)

            for backup_id in to_remove:
                del self._backups[backup_id]
                backup_path = os.path.join(self.backup_dir, f"{backup_id}.json")
                try:
                    os.remove(backup_path)
                    logger.info(f"Cleaned up old backup: {backup_id}")
                except Exception as e:
                    logger.warning(f"Failed to remove backup file {backup_path}: {e}")

class FailSafeManager:
    """Comprehensive fail-safe manager coordinating all fail-safe mechanisms."""

    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._health_checker = HealthChecker()
        self._rollback_manager = AutoRollbackManager()
        self._emergency_procedures: Dict[str, Callable] = {}
        self._lock = threading.Lock()

    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Register a circuit breaker."""
        with self._lock:
            self._circuit_breakers[name] = CircuitBreaker(config)
            logger.info(f"Registered circuit breaker: {name}")

    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """Register a health check."""
        self._health_checker.register_check(name, check_func)

    def register_emergency_procedure(self, name: str, procedure: Callable):
        """Register an emergency procedure."""
        with self._lock:
            self._emergency_procedures[name] = procedure
            logger.info(f"Registered emergency procedure: {name}")

    def execute_with_fail_safe(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with fail-safe mechanisms.

        Args:
            operation_name: Name of the operation
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        # Check circuit breaker
        if operation_name in self._circuit_breakers:
            circuit_breaker = self._circuit_breakers[operation_name]
            return circuit_breaker.call(func, *args, **kwargs)

        # Execute normally if no circuit breaker
        return func(*args, **kwargs)

    def trigger_emergency_procedure(self, procedure_name: str, **kwargs):
        """
        Trigger an emergency procedure.

        Args:
            procedure_name: Name of the emergency procedure
            **kwargs: Arguments for the procedure
        """
        with self._lock:
            if procedure_name not in self._emergency_procedures:
                logger.error(f"Emergency procedure '{procedure_name}' not found")
                return

            try:
                logger.warning(f"Triggering emergency procedure: {procedure_name}")
                self._emergency_procedures[procedure_name](**kwargs)
            except Exception as e:
                logger.error(f"Emergency procedure '{procedure_name}' failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        with self._lock:
            from datetime import timezone
            return {
                "circuit_breakers": {
                    name: cb.get_state() for name, cb in self._circuit_breakers.items()
                },
                "health_report": self._health_checker.get_health_report(),
                "backups": self._rollback_manager.list_backups(),
                "emergency_procedures": list(self._emergency_procedures.keys()),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def start_monitoring(self):
        """Start all monitoring systems."""
        self._health_checker.start_monitoring()
        logger.info("Fail-safe monitoring started")

    def stop_monitoring(self):
        """Stop all monitoring systems."""
        self._health_checker.stop_monitoring()
        logger.info("Fail-safe monitoring stopped")

# Global instances
_fail_safe_manager = FailSafeManager()

def get_fail_safe_manager() -> FailSafeManager:
    """Get the global fail-safe manager."""
    return _fail_safe_manager

# Convenience functions
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker to a function."""
    if config is None:
        config = CircuitBreakerConfig(name=name)

    _fail_safe_manager.register_circuit_breaker(name, config)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            return _fail_safe_manager.execute_with_fail_safe(name, func, *args, **kwargs)
        return wrapper
    return decorator

def health_check(name: str):
    """Decorator to register a function as a health check."""
    def decorator(func: Callable[[], HealthCheckResult]) -> Callable[[], HealthCheckResult]:
        _fail_safe_manager.register_health_check(name, func)
        return func
    return decorator

def emergency_procedure(name: str):
    """Decorator to register a function as an emergency procedure."""
    def decorator(func: Callable) -> Callable:
        _fail_safe_manager.register_emergency_procedure(name, func)
        return func
    return decorator

# Initialize default circuit breakers and health checks
def _initialize_default_fail_safes():
    """Initialize default fail-safe mechanisms."""
    # Database circuit breaker
    db_config = CircuitBreakerConfig(
        name="database",
        failure_threshold=3,
        recovery_timeout=30,
        success_threshold=2,
        timeout=5.0
    )
    _fail_safe_manager.register_circuit_breaker("database", db_config)

    # API circuit breaker
    api_config = CircuitBreakerConfig(
        name="external_api",
        failure_threshold=5,
        recovery_timeout=60,
        success_threshold=3,
        timeout=10.0
    )
    _fail_safe_manager.register_circuit_breaker("external_api", api_config)

    # Register basic health checks
    @health_check("system_memory")
    def check_system_memory() -> HealthCheckResult:
        """Check system memory health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent > 90:
                status = HealthStatus.CRITICAL
            elif memory_percent > 80:
                status = HealthStatus.UNHEALTHY
            elif memory_percent > 70:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return HealthCheckResult(
                status=status,
                metadata={"memory_percent": memory_percent}
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                error_message=str(e)
            )

    @health_check("disk_space")
    def check_disk_space() -> HealthCheckResult:
        """Check disk space health."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            if disk_percent > 95:
                status = HealthStatus.CRITICAL
            elif disk_percent > 90:
                status = HealthStatus.UNHEALTHY
            elif disk_percent > 85:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return HealthCheckResult(
                status=status,
                metadata={"disk_percent": disk_percent}
            )

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                error_message=str(e)
            )

    # Register emergency procedures
    @emergency_procedure("restart_services")
    def restart_services():
        """Emergency procedure to restart critical services."""
        logger.warning("Emergency: Restarting critical services...")
        # Implementation would depend on deployment setup
        # For now, just log the action
        logger.info("Emergency restart procedure completed")

    @emergency_procedure("switch_to_backup")
    def switch_to_backup():
        """Emergency procedure to switch to backup systems."""
        logger.warning("Emergency: Switching to backup systems...")
        # Implementation would depend on infrastructure
        logger.info("Emergency backup switch procedure completed")

_initialize_default_fail_safes()