"""
Comprehensive Error Handling System for GoalDiggers Platform
Provides centralized error handling, custom exceptions, and graceful degradation.
"""
import functools
import logging
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels for better classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for better classification."""
    DATABASE = "database"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"
    USER_INPUT = "user_input"

@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class GoalDiggersException(Exception):
    """Base exception class for GoalDiggers platform."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext()
        self.cause = cause
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "context": {
                "timestamp": self.context.timestamp.isoformat(),
                "user_id": self.context.user_id,
                "session_id": self.context.session_id,
                "request_id": self.context.request_id,
                "component": self.context.component,
                "operation": self.context.operation,
                "metadata": self.context.metadata
            },
            "traceback": self.traceback if self.traceback else None,
            "cause": str(self.cause) if self.cause else None
        }

# Database-related exceptions
class DatabaseException(GoalDiggersException):
    """Base class for database-related exceptions."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE, **kwargs)

class ConnectionError(DatabaseException):
    """Database connection errors."""
    pass

class QueryError(DatabaseException):
    """Database query execution errors."""
    pass

class MigrationError(DatabaseException):
    """Database migration errors."""
    pass

# Network and API exceptions
class NetworkException(GoalDiggersException):
    """Base class for network-related exceptions."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)

class APIException(NetworkException):
    """External API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code

class TimeoutException(NetworkException):
    """Network timeout errors."""
    pass

# Configuration exceptions
class ConfigurationException(GoalDiggersException):
    """Configuration-related errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)

class MissingConfigurationException(ConfigurationException):
    """Missing configuration errors."""
    pass

# Validation exceptions
class ValidationException(GoalDiggersException):
    """Data validation errors."""
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)
        self.field = field

class InvalidInputException(ValidationException):
    """Invalid user input errors."""
    pass

# Business logic exceptions
class BusinessLogicException(GoalDiggersException):
    """Business logic errors."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.BUSINESS_LOGIC, **kwargs)

class PredictionException(BusinessLogicException):
    """Prediction-related errors."""
    pass

class OddsException(BusinessLogicException):
    """Odds calculation errors."""
    pass

class ComprehensiveErrorHandler:
    """
    Comprehensive error handler for the GoalDiggers platform.
    Provides centralized error handling, logging, and recovery mechanisms.
    """

    def __init__(self):
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = {}
        self.recovery_strategies: Dict[Type[Exception], List[Callable]] = {}
        self.error_counts: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)

    def register_error_handler(self, category: ErrorCategory, handler: Callable):
        """Register an error handler for a specific category."""
        if category not in self.error_handlers:
            self.error_handlers[category] = []
        self.error_handlers[category].append(handler)

    def register_recovery_strategy(self, exception_type: Type[Exception], strategy: Callable):
        """Register a recovery strategy for a specific exception type."""
        if exception_type not in self.recovery_strategies:
            self.recovery_strategies[exception_type] = []
        self.recovery_strategies[exception_type].append(strategy)

    def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> bool:
        """
        Handle an error with appropriate logging and recovery.

        Args:
            error: The exception to handle
            context: Additional context information

        Returns:
            True if error was handled successfully, False otherwise
        """
        context = context or ErrorContext()

        # Log the error
        self._log_error(error, context)

        # Try recovery strategies
        if self._try_recovery(error, context):
            return True

        # Try category-specific handlers
        if isinstance(error, GoalDiggersException):
            category_handlers = self.error_handlers.get(error.category, [])
            for handler in category_handlers:
                try:
                    if handler(error, context):
                        return True
                except Exception as handler_error:
                    self.logger.error(f"Error handler failed: {handler_error}")

        return False

    def _log_error(self, error: Exception, context: ErrorContext):
        """Log an error with appropriate level and context."""
        error_key = f"{error.__class__.__name__}:{context.component or 'unknown'}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        log_data = {
            "error_type": error.__class__.__name__,
            "message": str(error),
            "component": context.component,
            "operation": context.operation,
            "user_id": context.user_id,
            "session_id": context.session_id,
            "request_id": context.request_id,
            "metadata": context.metadata,
            "count": self.error_counts[error_key]
        }

        if isinstance(error, GoalDiggersException):
            log_data.update(error.to_dict())

        # Determine log level based on severity
        if isinstance(error, GoalDiggersException):
            if error.severity == ErrorSeverity.CRITICAL:
                self.logger.critical("Critical error occurred", extra=log_data)
            elif error.severity == ErrorSeverity.HIGH:
                self.logger.error("High severity error occurred", extra=log_data)
            elif error.severity == ErrorSeverity.MEDIUM:
                self.logger.warning("Medium severity error occurred", extra=log_data)
            else:
                self.logger.info("Low severity error occurred", extra=log_data)
        else:
            self.logger.error("Unhandled error occurred", extra=log_data, exc_info=True)

    def _try_recovery(self, error: Exception, context: ErrorContext) -> bool:
        """Try to recover from an error using registered strategies."""
        for exception_type, strategies in self.recovery_strategies.items():
            if isinstance(error, exception_type):
                for strategy in strategies:
                    try:
                        if strategy(error, context):
                            self.logger.info(f"Successfully recovered from {error.__class__.__name__}")
                            return True
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery strategy failed: {recovery_error}")

        return False

def error_handler(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    re_raise: bool = True
):
    """
    Decorator for comprehensive error handling.

    Args:
        severity: Error severity level
        category: Error category
        re_raise: Whether to re-raise the exception after handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except GoalDiggersException:
                raise  # Re-raise our custom exceptions as-is
            except Exception as e:
                # Wrap unexpected exceptions
                context = ErrorContext(
                    component=getattr(func, '__module__', 'unknown'),
                    operation=func.__name__
                )

                wrapped_error = GoalDiggersException(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    severity=severity,
                    category=category,
                    context=context,
                    cause=e
                )

                # Get the global error handler
                handler = get_error_handler()
                if handler:
                    handler.handle_error(wrapped_error, context)

                if re_raise:
                    raise wrapped_error from e
                else:
                    return None

        return wrapper
    return decorator

# Global error handler instance
_global_error_handler: Optional[ComprehensiveErrorHandler] = None

def get_error_handler() -> Optional[ComprehensiveErrorHandler]:
    """Get the global error handler instance."""
    return _global_error_handler

def set_error_handler(handler: ComprehensiveErrorHandler):
    """Set the global error handler instance."""
    global _global_error_handler
    _global_error_handler = handler

def initialize_error_handler():
    """Initialize the global error handler with default configurations."""
    handler = ComprehensiveErrorHandler()

    # Register default recovery strategies
    def database_recovery(error: Exception, context: ErrorContext) -> bool:
        """Default database error recovery strategy."""
        if isinstance(error, ConnectionError):
            # Try to reconnect (simplified example)
            logger.info("Attempting database reconnection...")
            return False  # Placeholder - would implement actual reconnection logic
        return False

    def network_recovery(error: Exception, context: ErrorContext) -> bool:
        """Default network error recovery strategy."""
        if isinstance(error, TimeoutException):
            # Implement retry logic
            logger.info("Network timeout - could implement retry logic here")
            return False  # Placeholder
        return False

    handler.register_recovery_strategy(ConnectionError, database_recovery)
    handler.register_recovery_strategy(TimeoutException, network_recovery)

    set_error_handler(handler)
    logger.info("Comprehensive error handler initialized")

# Initialize on import
initialize_error_handler()