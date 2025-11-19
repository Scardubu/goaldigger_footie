"""
Global exception handler for the GoalDiggers platform.
Provides centralized error handling and reporting.
"""

import logging
import sys
import traceback
from typing import Any, Callable, Optional, Type

logger = logging.getLogger(__name__)

class ExceptionHandler:
    """
    Global exception handler for the GoalDiggers platform.
    """
    
    @staticmethod
    def handle_exception(exc_type: Type[BaseException], exc_value: BaseException, exc_traceback: Optional[Any]) -> None:
        """
        Global exception handler.
        
        Args:
            exc_type: Type of exception
            exc_value: Exception value
            exc_traceback: Exception traceback
        """
        # Don't print traceback for KeyboardInterrupt
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
            
        # Log the exception
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        
    @staticmethod
    def wrap_function(func: Callable) -> Callable:
        """
        Wrap a function with exception handling.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                # Re-raise the exception for the caller to handle
                raise
        return wrapper
        
    @staticmethod
    def install() -> None:
        """Install the global exception handler."""
        sys.excepthook = ExceptionHandler.handle_exception
        logger.info("Global exception handler installed")


# Install the global exception handler
ExceptionHandler.install()
