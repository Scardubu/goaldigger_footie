"""Production-ready error handling and recovery utilities.

Provides centralized error handling patterns, graceful fallbacks, and monitoring
integration for the GoalDiggers platform. Includes retry logic, circuit breaker
pattern, and structured error reporting.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

# Add project root to Python path for robust imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    backoff_factor: float = 2.0
    max_delay: float = 60.0
    retryable_exceptions: tuple = (Exception,)


class CircuitBreakerState:
    """Simple circuit breaker implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open

    def should_allow_request(self) -> bool:
        """Check if request should be allowed through."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Global circuit breakers for different services
_circuit_breakers: Dict[str, CircuitBreakerState] = {}


def get_circuit_breaker(service_name: str) -> CircuitBreakerState:
    """Get or create circuit breaker for service."""
    if service_name not in _circuit_breakers:
        _circuit_breakers[service_name] = CircuitBreakerState()
    return _circuit_breakers[service_name]


def with_retry(config: RetryConfig = None, context: ErrorContext = None):
    """Decorator for automatic retry with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = config.base_delay
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    if attempt == config.max_attempts - 1:
                        # Last attempt failed
                        logger.error(
                            f"Operation failed after {config.max_attempts} attempts",
                            extra={
                                "operation": context.operation if context else func.__name__,
                                "component": context.component if context else "unknown",
                                "error": str(e),
                                "attempts": config.max_attempts
                            }
                        )
                        break
                    
                    # Log retry attempt
                    logger.warning(
                        f"Operation failed, retrying in {delay}s (attempt {attempt + 1}/{config.max_attempts})",
                        extra={
                            "operation": context.operation if context else func.__name__,
                            "error": str(e),
                            "delay": delay,
                            "attempt": attempt + 1
                        }
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * config.backoff_factor, config.max_delay)
                except Exception as e:
                    # Non-retryable exception
                    logger.error(
                        f"Non-retryable error in operation",
                        extra={
                            "operation": context.operation if context else func.__name__,
                            "component": context.component if context else "unknown", 
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                    )
                    raise
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator


def with_circuit_breaker(service_name: str, context: ErrorContext = None):
    """Decorator for circuit breaker pattern."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            breaker = get_circuit_breaker(service_name)
            
            if not breaker.should_allow_request():
                error_msg = f"Circuit breaker OPEN for {service_name}"
                logger.warning(
                    error_msg,
                    extra={
                        "service": service_name,
                        "state": breaker.state,
                        "failure_count": breaker.failure_count
                    }
                )
                raise ServiceUnavailableError(error_msg)
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                logger.error(
                    f"Operation failed in {service_name}",
                    extra={
                        "service": service_name,
                        "error": str(e),
                        "circuit_breaker_state": breaker.state,
                        "failure_count": breaker.failure_count
                    }
                )
                raise
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    fallback_value: Any = None,
    context: ErrorContext = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """Safely execute a function with optional fallback."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(
                f"Safe execution failed",
                extra={
                    "function": func.__name__,
                    "operation": context.operation if context else "unknown",
                    "component": context.component if context else "unknown", 
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_used": fallback_value is not None
                }
            )
        return fallback_value


def handle_prediction_error(
    error: Exception,
    home_team: str,
    away_team: str,
    operation: str = "prediction"
) -> Dict[str, Any]:
    """Handle prediction-specific errors with fallback response."""
    logger.error(
        f"Prediction error for {home_team} vs {away_team}",
        extra={
            "operation": operation,
            "component": "predictor",
            "home_team": home_team,
            "away_team": away_team,
            "error": str(error),
            "error_type": type(error).__name__
        }
    )
    
    # Return fallback prediction
    return {
        "home_win_probability": 0.33,
        "draw_probability": 0.34,
        "away_win_probability": 0.33,
        "confidence": 0.1,
        "error": True,
        "error_message": "Prediction service temporarily unavailable",
        "fallback": True
    }


def handle_data_error(
    error: Exception,
    data_source: str,
    operation: str = "data_fetch"
) -> Optional[Any]:
    """Handle data fetching errors with appropriate logging."""
    logger.error(
        f"Data error from {data_source}",
        extra={
            "operation": operation,
            "component": "data_integration",
            "data_source": data_source,
            "error": str(error),
            "error_type": type(error).__name__
        }
    )
    
    # Could implement fallback data loading here
    return None


class ServiceUnavailableError(Exception):
    """Exception raised when a service is unavailable due to circuit breaker."""
    pass


class PredictionError(Exception):
    """Exception raised during prediction operations."""
    pass


class DataIntegrityError(Exception):
    """Exception raised when data integrity checks fail."""
    pass


def monitor_health_check_errors(func: Callable) -> Callable:
    """Decorator to monitor and log health check errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            # Log successful health check
            logger.info(
                "Health check completed successfully",
                extra={
                    "component": "health_check",
                    "status": "healthy"
                }
            )
            return result
        except Exception as e:
            logger.error(
                "Health check failed",
                extra={
                    "component": "health_check",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "status": "unhealthy"
                }
            )
            # Return unhealthy status rather than raising
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    return wrapper


# Context manager for error context
class error_context:
    """Context manager for setting error context."""
    
    def __init__(self, operation: str, component: str, **kwargs):
        self.context = ErrorContext(
            operation=operation,
            component=component,
            **kwargs
        )
        self.old_context = getattr(error_context, '_current', None)
    
    def __enter__(self):
        error_context._current = self.context
        return self.context
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(
                f"Error in {self.context.operation}",
                extra={
                    "operation": self.context.operation,
                    "component": self.context.component,
                    "error": str(exc_val),
                    "error_type": exc_type.__name__ if exc_type else "unknown"
                }
            )
        error_context._current = self.old_context
        return False  # Don't suppress exceptions


__all__ = [
    "ErrorContext",
    "RetryConfig", 
    "CircuitBreakerState",
    "with_retry",
    "with_circuit_breaker",
    "safe_execute",
    "handle_prediction_error",
    "handle_data_error",
    "monitor_health_check_errors",
    "error_context",
    "ServiceUnavailableError",
    "PredictionError", 
    "DataIntegrityError"
]