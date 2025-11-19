import functools
import inspect
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from utils.config import Config


# Configure logging
def setup_logging(
    log_level=logging.INFO,
    log_file=None,
    logger_name="goaldiggers",
    format_str='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    propagate: Optional[bool] = True,
):
    """
    Set up the logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Path to log file (default: logs/footie_dashboard.log)
        logger_name: Name of the logger (default: goaldiggers)
        format_str: Log format string
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    
    under_pytest = os.getenv("PYTEST_CURRENT_TEST") is not None

    # When running under pytest, remove existing handlers so caplog can capture emitted records
    if under_pytest and logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    # Only configure handlers when not running inside pytest (to avoid interfering with caplog)
    if not logger.handlers and not under_pytest:
        # Set log level
        logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(format_str)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file is None:
            log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "footie_dashboard.log")
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Ensure logger level honors requested configuration
    logger.setLevel(log_level)
        
    # Control propagation so external tooling (like pytest caplog) can capture records
    logger.propagate = propagate if propagate is not None else True
    
    return logger


# Create logger instance
logger = setup_logging()


class ErrorLog:
    def _send_external_alert(self, error_record):
        """
        Send an external alert for critical errors or health check failures.
        Extend this method to integrate with email, Slack, webhook, etc.
        """
        # Example: send to Slack webhook (pseudo-code, replace with real implementation)
        webhook_url = self.config.get("alert_webhook_url")
        if webhook_url and error_record['level'] in ["critical", "error"]:
            import requests
            try:
                payload = {
                    "text": f"[ALERT] {error_record['type']}: {error_record['message']}\nLevel: {error_record['level']}\nTime: {error_record['timestamp']}"
                }
                requests.post(webhook_url, json=payload, timeout=3)
            except Exception as e:
                self.logger.warning(f"Failed to send external alert: {e}")

    _streamlit_runtime_active = None  # Class variable to cache Streamlit status

    @staticmethod
    def _get_streamlit_runtime():
        """Checks if Streamlit is running and returns the module, otherwise False."""
        if ErrorLog._streamlit_runtime_active is None:
            try:
                import streamlit as st_module

                # Check if Streamlit is truly running a script via its runtime context
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                if get_script_run_ctx() is not None:
                    ErrorLog._streamlit_runtime_active = st_module
                else:
                    ErrorLog._streamlit_runtime_active = False # Streamlit importable but not in a run context
            except ImportError:
                ErrorLog._streamlit_runtime_active = False # Streamlit not importable
            except Exception: # Catch any other error during streamlit import/check
                ErrorLog._streamlit_runtime_active = False
        return ErrorLog._streamlit_runtime_active

    """
    A centralized error logging utility that combines:
    1. Standard Python logging to file/console
    2. Session-based error tracking for UI display (when using Streamlit)
    3. Interoperability with external logging systems
    """
    def __init__(self, component_name: str = None, logger=None):
        """
        Initialize the error logger.
        
        Args:
            component_name: Name of the component (optional, default None for root logger)
            logger: Existing logger instance to use (if None, creates a new one)
        """
        # Always use the root goaldiggers logger for the default instance
        if component_name:
            self.logger = logger or logging.getLogger(f"goaldiggers.{component_name}")
        else:
            self.logger = logger or logging.getLogger("goaldiggers")

        # Safely get config with fallback
        try:
            self.config = Config.get("logging", Config.get("error_handling", {}))
        except Exception as e:
            # Fallback to default config if Config system fails
            self.config = {
                "log_level": "INFO",
                "max_errors": 100,
                "error_retention_hours": 24
            }
            self.logger.warning(f"Failed to load error handling config, using defaults: {e}")
        
        # Track errors even without Streamlit
        self.recent_errors = []
    
    def log(self,
            err_type: str = "UNKNOWN",
            message: str = "",
            exception: Optional[Exception] = None,
            details: Optional[Dict[str, Any]] = None,
            suggestion: Optional[str] = None,
            level: str = "error",
            stack_info: bool = False,
            source: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None):
        """
        Log an error both to the logging system and to the session state.
        
        Args:
            err_type: Type/category of the error
            message: Error message
            exception: Exception object if available
            details: Additional error details
            suggestion: Suggested fix for the error
            level: Logging level (debug, info, warning, error, critical)
            stack_info: Include stack information
            extra: Additional data to pass to the logger
        """
        # Normalize err_type when not provided so logs reflect severity (e.g. [ERROR])
        if not err_type or err_type.upper() == "UNKNOWN":
            err_type = level.upper()

        # Format the log message
        log_msg = f"[{err_type}] {message}"
        if exception:
            log_msg += f": {str(exception)}"
            
        # Prepare extra data for logging
        log_extra = {
            'err_type': err_type,
            'details': details,
            'suggestion': suggestion
        }
        if extra:
            log_extra.update(extra)

        # Do not mutate handlers here; test harness (pytest/caplog) or
        # global logging configuration should control handler attachment.
        # Keep propagation enabled so records can bubble up to root handlers.
        try:
            self.logger.propagate = True
        except Exception:
            pass

        # Log to the Python logger.
        # IMPORTANT: logging expects exc_info to be a tuple (type, value, traceback) OR True.
        # Passing the raw exception object causes logging to call sys.exc_info(), which will
        # return (None, None, None) outside an except block, losing the real exception.
        if exception:
            exc_info = (type(exception), exception, exception.__traceback__)
        else:
            exc_info = None
        log_method = getattr(self.logger, level.lower())
        log_method(log_msg, exc_info=exc_info, stack_info=stack_info, extra=log_extra)

        # Create error record
        from datetime import timezone
        error_record = {
            'type': err_type,
            'message': message,
            'exception': str(exception) if exception else None,
            'details': details,
            'suggestion': suggestion,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': level
        }
        
        # Store in internal list
        self.recent_errors.append(error_record)
        self.recent_errors = self.recent_errors[-20:]  # Keep only last 20
        # Send external alert for critical errors/health failures
        if error_record['level'] in ["critical", "error"]:
            # Avoid external alert noise during pytest runs
            if not os.getenv("PYTEST_CURRENT_TEST"):
                self._send_external_alert(error_record)
        # Store in session state if using Streamlit
        # (Handler removal handled before emit)        

    def info(self, message: str, **kwargs):
        self.log(message=message, level="info", **kwargs)

    def warning(self, message: str, **kwargs):
        self.log(message=message, level="warning", **kwargs)

    def error(self, message: str, **kwargs):
        self.log(message=message, level="error", **kwargs)

    def critical(self, message: str, **kwargs):
        self.log(message=message, level="critical", **kwargs)

    def debug(self, message: str, **kwargs):
        self.log(message=message, level="debug", **kwargs)
        
# Health check/metrics alert hook
def alert_on_health_failure(component: str, message: str, level: str = "critical"):
    """
    Hook to alert on health check or performance failure.
    """
    error_log.log(
        err_type=f"health_check.{component}",
        message=message,
        level=level
    )


# Create a default instance for easy import, sharing the module-level logger so external
# consumers (and tests) capture the same log records.
error_log = ErrorLog(logger=logger)


# Enhanced error context tracking
class ErrorContext:
    """Collects and stores contextual information about errors."""
    
    @staticmethod
    def get_current_context(skip_frames: int = 1) -> Dict[str, Any]:
        """
        Get contextual information about the current execution frame.
        
        Args:
            skip_frames: Number of frames to skip in the stack
            
        Returns:
            Dictionary with contextual information
        """
        try:
            # Get the calling frame
            frame = inspect.currentframe()
            for _ in range(skip_frames):
                if frame.f_back:
                    frame = frame.f_back
            
            # Extract information from the frame
            filename = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            line_number = frame.f_lineno
            module_name = inspect.getmodule(frame).__name__ if inspect.getmodule(frame) else "unknown"
            
            # Get local variables (safely)
            local_vars = {}
            for var_name, var_val in frame.f_locals.items():
                try:
                    # Only include simple types to avoid circular references
                    if isinstance(var_val, (str, int, float, bool, list, dict)) and not var_name.startswith('_'):
                        if isinstance(var_val, (list, dict)) and len(str(var_val)) > 500:
                            # Truncate large structures
                            local_vars[var_name] = f"{str(var_val)[:500]}... (truncated)"
                        else:
                            local_vars[var_name] = var_val
                except:
                    pass
            
            return {
                "filename": filename,
                "function": function_name,
                "line": line_number,
                "module": module_name,
                "variables": local_vars
            }
        except Exception as e:
            # Fallback if context collection fails
            return {
                "error": f"Failed to collect context: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    @staticmethod
    def get_exception_context(exception: Exception) -> Dict[str, Any]:
        """
        Get contextual information about an exception.
        
        Args:
            exception: The exception to analyze
            
        Returns:
            Dictionary with exception context
        """
        try:
            return {
                "type": exception.__class__.__name__,
                "message": str(exception),
                "traceback": traceback.format_exception(type(exception), exception, exception.__traceback__),
                "module": exception.__class__.__module__
            }
        except Exception as e:
            # Fallback if context collection fails
            return {
                "error": f"Failed to collect exception context: {str(e)}",
                "type": exception.__class__.__name__,
                "message": str(exception)
            }


def log_exceptions_decorator(func):
    """Decorator to log exceptions occurring in the decorated function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"Exception in '{func.__name__}'"
            # Determine a relevant err_type, e.g., from the function or class context
            err_type = f"{func.__module__}.{func.__name__}.exception"
            error_log.error(
                message=error_message,
                exception=e,
                err_type=err_type,
                stack_info=True,  # Capturing stack info is useful for debugging
                details={'function_args': args, 'function_kwargs': kwargs}
            )
            raise # Re-raise the exception to ensure it's not swallowed
    return wrapper


def log_structured_error(context: str, error_type: str, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
    """
    Log detailed error with structured context for better debugging.
    
    Args:
        context: Error context (e.g., "data_loading", "api_call")
        error_type: Type of error
        message: Error message
        exception: Optional exception object
    """
    # Capture execution context
    frame_context = ErrorContext.get_current_context(skip_frames=2)  # Skip this function and caller
    
    # Get exception context if available
    exception_context = None
    if exception:
        exception_context = ErrorContext.get_exception_context(exception)
    
    # Add suggestion based on error type
    suggestion = _generate_error_suggestion(error_type, exception)
    
    # Log the structured error
    error_log.log(
        err_type=error_type,
        message=message,
        exception=exception,
        details={
            "context": context,
            "execution_context": frame_context,
            "exception_context": exception_context,
            **kwargs
        },
        suggestion=suggestion
    )

def _generate_error_suggestion(error_type: str, exception: Optional[Exception] = None) -> Optional[str]:
    """
    Generate a helpful suggestion based on error type and exception.
    
    Args:
        error_type: Type of error
        exception: Optional exception
        
    Returns:
        Suggestion string or None
    """
    suggestions = {
        "database_connection_error": "Check database connection string, credentials, and that the database server is running.",
        "data_parsing_error": "Verify data format matches expected schema. Check for recent API changes.",
        "api_timeout": "Check network connectivity and API status. Consider increasing timeout or implementing retry logic.",
        "scraper_blocked": "Your IP may be blocked. Use rotating proxies, reduce request frequency, or update user agents.",
        "data_source_failure": "Primary data source failed. Check the source status and consider fallback options.",
        "null_reference": "A required object was null. Check initialization sequence and error handling.",
        "key_error": "Dictionary key not found. Ensure the key exists or use .get() with a default value.",
        "index_error": "List index out of range. Check list length before accessing elements.",
        "value_error": "Invalid value provided. Check input validation and type conversion.",
        "type_error": "Type mismatch. Ensure variables have expected types or use proper type conversion.",
        "import_error": "Module not found. Check that the module is installed and the import path is correct.",
        "file_not_found": "File does not exist. Verify the file path and permissions.",
        "permission_error": "Insufficient permissions. Check file/directory permissions.",
        "memory_error": "Out of memory. Consider batch processing or optimizing memory usage.",
        "timeout_error": "Operation timed out. Increase timeout or check resource availability."
    }
    
    # Try to match based on error_type
    for key, suggestion in suggestions.items():
        if key in error_type.lower():
            return suggestion
    
    # Try to match based on exception type
    if exception:
        exception_type = exception.__class__.__name__.lower()
        for key, suggestion in suggestions.items():
            if key.lower() in exception_type:
                return suggestion
    
    # Generic suggestion
    return "Check logs for more details and consider adding specific error handling for this case."

# Convenience functions
def log_error(message, exception=None, **kwargs):
    """Convenience function to log errors."""
    error_log.error(message, exception=exception, **kwargs)

def log_warning(message, exception=None, **kwargs):
    """Convenience function to log warnings."""
    error_log.warning(message, exception=exception, **kwargs)

def log_info(message, **kwargs):
    """Convenience function to log info messages."""
    error_log.info(message, **kwargs)

def log_critical(message, exception=None, **kwargs):
    """Convenience function to log critical errors."""
    error_log.critical(message, exception=exception, **kwargs)

def log_debug(message, **kwargs):
    """Convenience function to log debug messages."""
    error_log.debug(message, **kwargs)

def get_logger(component_name="dashboard"):
    """
    Get a configured logger for a component.
    
    Args:
        component_name: Name of the component
        
    Returns:
        ErrorLog instance
    """
    return ErrorLog(component_name=component_name)
