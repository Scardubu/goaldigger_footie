#!/usr/bin/env python3
"""
Defensive Programming Utilities

Implements defensive programming patterns to improve error handling and
automatic recovery throughout the GoalDiggers platform.

Key Patterns:
1. Safe attribute access with fallbacks
2. Null-safe operations
3. Type validation and coercion
4. Safe data structure operations
5. Graceful degradation decorators
6. Input validation and sanitization
"""

import logging
import functools
import traceback
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar('T')

class SafeAccessor:
    """Safe accessor for nested data structures."""
    
    @staticmethod
    def safe_get(data: Any, path: str, default: Any = None, separator: str = '.') -> Any:
        """
        Safely get nested value from data structure.
        
        Args:
            data: Data structure to access
            path: Dot-separated path (e.g., 'user.profile.name')
            default: Default value if path not found
            separator: Path separator (default: '.')
        
        Returns:
            Value at path or default
        """
        try:
            if not data:
                return default
            
            keys = path.split(separator)
            current = data
            
            for key in keys:
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, (list, tuple)) and key.isdigit():
                    index = int(key)
                    current = current[index] if 0 <= index < len(current) else None
                elif hasattr(current, key):
                    current = getattr(current, key)
                else:
                    return default
                
                if current is None:
                    return default
            
            return current
            
        except (KeyError, IndexError, AttributeError, ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_set(data: Dict[str, Any], path: str, value: Any, separator: str = '.') -> bool:
        """
        Safely set nested value in dictionary.
        
        Args:
            data: Dictionary to modify
            path: Dot-separated path
            value: Value to set
            separator: Path separator
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not isinstance(data, dict):
                return False
            
            keys = path.split(separator)
            current = data
            
            # Navigate to parent of target key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    return False
                current = current[key]
            
            # Set the final value
            current[keys[-1]] = value
            return True
            
        except (KeyError, TypeError, AttributeError):
            return False

class TypeValidator:
    """Type validation and coercion utilities."""
    
    @staticmethod
    def safe_int(value: Any, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            if isinstance(value, int):
                return value
            elif isinstance(value, (float, str)):
                return int(float(value))
            else:
                return default
        except (ValueError, TypeError, OverflowError):
            return default
    
    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            else:
                return default
        except (ValueError, TypeError, OverflowError):
            return default
    
    @staticmethod
    def safe_str(value: Any, default: str = "") -> str:
        """Safely convert value to string."""
        try:
            if value is None:
                return default
            return str(value)
        except (TypeError, UnicodeError):
            return default
    
    @staticmethod
    def safe_bool(value: Any, default: bool = False) -> bool:
        """Safely convert value to boolean."""
        try:
            if isinstance(value, bool):
                return value
            elif isinstance(value, (int, float)):
                return bool(value)
            elif isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            else:
                return default
        except (TypeError, AttributeError):
            return default
    
    @staticmethod
    def safe_list(value: Any, default: Optional[List] = None) -> List:
        """Safely convert value to list."""
        if default is None:
            default = []
        
        try:
            if isinstance(value, list):
                return value
            elif isinstance(value, (tuple, set)):
                return list(value)
            elif value is None:
                return default
            else:
                return [value]
        except (TypeError, ValueError):
            return default
    
    @staticmethod
    def safe_dict(value: Any, default: Optional[Dict] = None) -> Dict:
        """Safely convert value to dictionary."""
        if default is None:
            default = {}
        
        try:
            if isinstance(value, dict):
                return value
            elif value is None:
                return default
            else:
                return default
        except (TypeError, ValueError):
            return default

def safe_execute(func: Callable[[], T], default: T = None, log_errors: bool = True) -> T:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        default: Default value on error
        log_errors: Whether to log errors
    
    Returns:
        Function result or default value
    """
    try:
        return func()
    except Exception as e:
        if log_errors:
            logger.warning(f"Safe execution failed: {e}")
        return default

def graceful_degradation(fallback_value: Any = None, log_errors: bool = True):
    """
    Decorator for graceful degradation on function failures.
    
    Args:
        fallback_value: Value to return on error
        log_errors: Whether to log errors
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"Function {func.__name__} failed gracefully: {e}")
                return fallback_value
        return wrapper
    return decorator

def null_safe(func: Callable) -> Callable:
    """
    Decorator to make functions null-safe.
    Returns None if any argument is None.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check for None in positional arguments
        if any(arg is None for arg in args):
            return None
        
        # Check for None in keyword arguments
        if any(value is None for value in kwargs.values()):
            return None
        
        return func(*args, **kwargs)
    return wrapper

def validate_input(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Keyword arguments mapping parameter names to validator functions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid value for parameter {param_name}: {value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)):
    """
    Decorator to retry function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        import time
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

class SafeDataProcessor:
    """Safe data processing utilities."""
    
    @staticmethod
    def safe_json_parse(json_str: str, default: Any = None) -> Any:
        """Safely parse JSON string."""
        try:
            import json
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError, ValueError):
            return default
    
    @staticmethod
    def safe_date_parse(date_str: str, format_str: str = "%Y-%m-%d", default: Optional[datetime] = None) -> Optional[datetime]:
        """Safely parse date string."""
        try:
            return datetime.strptime(date_str, format_str)
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def safe_division(numerator: Union[int, float], denominator: Union[int, float], default: float = 0.0) -> float:
        """Safely perform division with zero check."""
        try:
            if denominator == 0:
                return default
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return default
    
    @staticmethod
    def safe_percentage(part: Union[int, float], total: Union[int, float], default: float = 0.0) -> float:
        """Safely calculate percentage."""
        try:
            if total == 0:
                return default
            return (float(part) / float(total)) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            return default

class ErrorBoundary:
    """Error boundary for component isolation."""
    
    def __init__(self, component_name: str, fallback_value: Any = None):
        self.component_name = component_name
        self.fallback_value = fallback_value
        self.error_count = 0
        self.last_error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_count += 1
            self.last_error = exc_val
            logger.error(f"Error in component {self.component_name}: {exc_val}")
            return True  # Suppress the exception
        return False
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function within error boundary."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.error_count += 1
            self.last_error = e
            logger.error(f"Error in component {self.component_name}: {e}")
            return self.fallback_value

# Utility functions for common defensive patterns
def ensure_list(value: Any) -> List:
    """Ensure value is a list."""
    return TypeValidator.safe_list(value)

def ensure_dict(value: Any) -> Dict:
    """Ensure value is a dictionary."""
    return TypeValidator.safe_dict(value)

def ensure_string(value: Any) -> str:
    """Ensure value is a string."""
    return TypeValidator.safe_str(value)

def ensure_number(value: Any, default: Union[int, float] = 0) -> Union[int, float]:
    """Ensure value is a number."""
    if isinstance(default, int):
        return TypeValidator.safe_int(value, default)
    else:
        return TypeValidator.safe_float(value, default)

def safe_chain(*functions):
    """
    Chain functions safely, stopping on first None result.
    
    Args:
        *functions: Functions to chain
    
    Returns:
        Result of function chain or None if any function returns None
    """
    def chained_function(initial_value):
        result = initial_value
        for func in functions:
            if result is None:
                return None
            try:
                result = func(result)
            except Exception as e:
                logger.warning(f"Function chain broken at {func.__name__}: {e}")
                return None
        return result
    return chained_function
