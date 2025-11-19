"""
Enhanced error handling module for the GoalDiggers platform.
Provides consistent error handling, user-friendly messages,
and detailed logging for better observability.
"""

import logging
import sys
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Error categories for better organization and handling
ERROR_CATEGORIES = {
    'DATABASE': {
        'title': 'Database Error',
        'icon': '🛢️',
        'color': '#dc3545',
        'keywords': ['database', 'db', 'sql', 'query', 'connection', 'sqlite']
    },
    'API': {
        'title': 'API Connection Error',
        'icon': '🌐',
        'color': '#fd7e14',
        'keywords': ['api', 'request', 'http', 'response', 'endpoint', 'url']
    },
    'DATA': {
        'title': 'Data Processing Error',
        'icon': '📊',
        'color': '#ffc107',
        'keywords': ['data', 'parse', 'json', 'csv', 'format', 'transform']
    },
    'MODEL': {
        'title': 'ML Model Error',
        'icon': '🧠',
        'color': '#17a2b8',
        'keywords': ['model', 'predict', 'inference', 'feature', 'train']
    },
    'UI': {
        'title': 'Interface Error',
        'icon': '🖥️',
        'color': '#6610f2',
        'keywords': ['ui', 'interface', 'display', 'render', 'streamlit']
    },
    'SYSTEM': {
        'title': 'System Error',
        'icon': '⚙️',
        'color': '#343a40',
        'keywords': ['system', 'file', 'memory', 'disk', 'os', 'path']
    }
}

class EnhancedError(Exception):
    """Enhanced error class with additional context for better error handling."""
    
    def __init__(
        self, 
        message: str, 
        category: str = 'SYSTEM',
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        """
        Initialize enhanced error with rich context.
        
        Args:
            message: Primary error message
            category: Error category (from ERROR_CATEGORIES)
            original_error: Original exception that was caught
            context: Additional context about the error (e.g., parameters, state)
            suggestions: List of suggested fixes or troubleshooting steps
        """
        self.message = message
        self.category = category if category in ERROR_CATEGORIES else 'SYSTEM'
        self.original_error = original_error
        self.context = context or {}
        self.suggestions = suggestions or []
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc() if original_error else None
        
        # Call parent constructor
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'message': self.message,
            'category': self.category,
            'original_error': str(self.original_error) if self.original_error else None,
            'context': self.context,
            'suggestions': self.suggestions,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback
        }
    
    def get_category_info(self) -> Dict[str, str]:
        """Get formatting information for this error category."""
        return ERROR_CATEGORIES[self.category]
    
    def display(self):
        """Display error in Streamlit UI with appropriate formatting."""
        category_info = self.get_category_info()
        
        # Main error message
        st.error(f"{category_info['icon']} {category_info['title']}: {self.message}")
        
        # Detailed error info in expander
        with st.expander("Error Details & Troubleshooting"):
            # Error context
            if self.context:
                st.markdown("#### Context")
                for key, value in self.context.items():
                    st.text(f"{key}: {value}")
            
            # Original error
            if self.original_error:
                st.markdown("#### Original Error")
                st.code(str(self.original_error), language="python")
            
            # Troubleshooting suggestions
            if self.suggestions:
                st.markdown("#### Suggested Actions")
                for i, suggestion in enumerate(self.suggestions, 1):
                    st.markdown(f"{i}. {suggestion}")
            else:
                # Default suggestions based on category
                self._display_default_suggestions()
    
    def _display_default_suggestions(self):
        """Display default suggestions based on error category."""
        st.markdown("#### Suggested Actions")
        
        if self.category == 'DATABASE':
            st.markdown("""
            1. Check database connection settings
            2. Verify the database exists and is accessible
            3. Ensure database schema is up to date
            4. Try running the data ingestion pipeline
            """)
        elif self.category == 'API':
            st.markdown("""
            1. Check your internet connection
            2. Verify API keys are valid
            3. Check API service status
            4. Try again in a few minutes
            """)
        elif self.category == 'DATA':
            st.markdown("""
            1. Check input data format and structure
            2. Verify required fields are present
            3. Run data validation tools
            4. Check for missing values or incorrect data types
            """)
        elif self.category == 'MODEL':
            st.markdown("""
            1. Check model input features are complete
            2. Verify model files are present and not corrupted
            3. Run model diagnostics
            4. Try a different model or fallback option
            """)
        elif self.category == 'UI':
            st.markdown("""
            1. Refresh the page
            2. Clear browser cache
            3. Try with a different browser
            4. Check for JavaScript console errors
            """)
        else:  # SYSTEM or default
            st.markdown("""
            1. Refresh the application
            2. Check system resources (memory, disk)
            3. Verify file permissions
            4. Check logs for more details
            """)

def handle_errors(
    category: str = 'SYSTEM',
    fallback_return: Any = None, 
    show_streamlit_error: bool = True,
    suggestions: Optional[List[str]] = None,
    log_level: int = logging.ERROR
):
    """
    Decorator to handle errors consistently across the application.
    
    Args:
        category: Error category (from ERROR_CATEGORIES)
        fallback_return: Value to return if an exception occurs
        show_streamlit_error: Whether to display error in Streamlit UI
        suggestions: Custom troubleshooting suggestions
        log_level: Logging level for errors
    
    Returns:
        Decorated function with error handling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Extract function context for better debugging
                context = {
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                # Create enhanced error
                error = EnhancedError(
                    message=f"Error in {func.__name__}: {str(e)}",
                    category=category,
                    original_error=e,
                    context=context,
                    suggestions=suggestions
                )
                
                # Log the error
                log_message = f"{error.get_category_info()['title']}: {error.message}"
                logger.log(log_level, log_message, exc_info=True)
                
                # Display in Streamlit if requested
                if show_streamlit_error:
                    error.display()
                
                # Return fallback value
                return fallback_return
        
        return wrapper
    
    return decorator

def determine_error_category(error: Exception) -> str:
    """
    Analyze an exception to determine its category.
    
    Args:
        error: The exception to analyze
        
    Returns:
        Error category from ERROR_CATEGORIES
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    for category, info in ERROR_CATEGORIES.items():
        # Check error type
        if any(keyword in error_type for keyword in info['keywords']):
            return category
        
        # Check error message
        if any(keyword in error_str for keyword in info['keywords']):
            return category
    
    # Default to SYSTEM if no match found
    return 'SYSTEM'

def log_error_to_database(error: EnhancedError):
    """
    Log error to database for analysis and monitoring.
    
    Args:
        error: Enhanced error to log
    """
    try:
        # This would normally insert into a database
        # Here we just log that we would do so
        logger.info(f"Would log error to database: {error.message} ({error.category})")
    except Exception as e:
        # Don't let error logging cause more errors
        logger.warning(f"Failed to log error to database: {e}")

def create_error_report(errors: List[EnhancedError]) -> pd.DataFrame:
    """
    Create a DataFrame report from a list of errors.
    
    Args:
        errors: List of enhanced errors
        
    Returns:
        DataFrame with error information
    """
    error_dicts = [err.to_dict() for err in errors]
    
    # Convert to DataFrame
    df = pd.DataFrame(error_dicts)
    
    # Add derived columns
    if not df.empty and 'timestamp' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        df['time'] = pd.to_datetime(df['timestamp']).dt.time
    
    return df

def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for uncaught exceptions.
    
    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Exception traceback
    """
    # Ignore KeyboardInterrupt
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Create enhanced error
    category = determine_error_category(exc_value)
    error = EnhancedError(
        message=str(exc_value),
        category=category,
        original_error=exc_value,
        context={'global_handler': True}
    )
    
    # Log error
    logger.critical(
        f"Uncaught {error.get_category_info()['title']}: {error.message}",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    
    # Log to database
    log_error_to_database(error)

# Set as global exception handler
sys.excepthook = handle_uncaught_exception

# Example usage of the decorator:
# @handle_errors(category='DATABASE', fallback_return=None)
# def fetch_data_from_db(query):
#     # Database access code here
#     pass
