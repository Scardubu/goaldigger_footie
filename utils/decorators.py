"""
Contains common utility decorators for the application.
"""
import functools
import logging
import traceback
from typing import Any, Callable, Dict, Optional, TypeVar, cast

# Type variables for better type hinting
F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)

def log_exceptions(logger_obj: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """
    Decorator to log exceptions from functions with detailed context.
    
    Args:
        logger_obj: Optional logger object. If None, uses module logger.
        
    Returns:
        Decorator function that wraps the target function and logs exceptions.
    """
    _logger = logger_obj or logger
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create a detailed error log
                arg_values = [str(arg) for arg in args]
                kwarg_values = {k: str(v) for k, v in kwargs.items()}
                
                error_context = {
                    'function': func.__qualname__,
                    'args': arg_values,
                    'kwargs': kwarg_values,
                    'exception_type': type(e).__name__,
                    'exception_msg': str(e),
                    'traceback': traceback.format_exc()
                }
                
                # Log the error with detailed context
                _logger.error(
                    f"Exception in {func.__qualname__}: {type(e).__name__}: {str(e)}\n"
                    f"Arguments: {arg_values}\n"
                    f"Keyword arguments: {kwarg_values}\n"
                    f"Traceback:\n{traceback.format_exc()}"
                )
                
                # Re-raise the exception to allow for exception handling up the call stack
                raise
                
        return cast(F, wrapper)
    
    return decorator

def performance_tracker(logger_obj: Optional[logging.Logger] = None) -> Callable[[F], F]:
    """
    Decorator to track performance of functions, logging execution time.
    
    Args:
        logger_obj: Optional logger object. If None, uses module logger.
        
    Returns:
        Decorator function that wraps the target function and tracks execution time.
    """
    _logger = logger_obj or logger
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import time
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                _logger.debug(f"Function {func.__qualname__} executed in {execution_time:.4f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                _logger.error(
                    f"Function {func.__qualname__} failed after {execution_time:.4f} seconds: {str(e)}"
                )
                raise
                
        return cast(F, wrapper)
    
    return decorator
