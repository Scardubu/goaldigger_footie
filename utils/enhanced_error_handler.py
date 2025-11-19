#!/usr/bin/env python3
"""
Enhanced Error Handler for GoalDiggers Platform
Provides user-friendly error messages, recovery guidance, and automatic recovery mechanisms.
"""

import logging
import os
import time
import traceback
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

# Centralized logging (skip during pytest to avoid handler duplication)
if not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        from utils.logging_config import configure_logging  # type: ignore
        configure_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification."""
    NETWORK = "network"
    DATA = "data"
    PREDICTION = "prediction"
    UI = "ui"
    SYSTEM = "system"
    SECURITY = "security"


class EnhancedErrorHandler:
    """
    Enhanced error handler with user-friendly messages and recovery guidance.
    
    Features:
    - User-friendly error messages
    - Contextual recovery guidance
    - Automatic retry mechanisms
    - Error pattern detection
    - Recovery suggestions
    """
    
    def __init__(self):
        """Initialize enhanced error handler."""
        self.error_patterns = {}
        self.recovery_strategies = {}
        self.error_history = []
        self.retry_configs = {}
        
        # Initialize default error messages and recovery strategies
        self._initialize_error_messages()
        self._initialize_recovery_strategies()
        self._initialize_retry_configs()
        
        logger.info("🛡️ Enhanced Error Handler initialized")
    
    def _initialize_error_messages(self):
        """Initialize user-friendly error messages."""
        self.error_patterns = {
            # Network errors
            'connection_error': {
                'pattern': ['connection', 'network', 'timeout', 'unreachable'],
                'user_message': "We're having trouble connecting to our data sources. This is usually temporary.",
                'technical_message': "Network connection error occurred",
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.NETWORK
            },
            'api_rate_limit': {
                'pattern': ['rate limit', '429', 'too many requests'],
                'user_message': "We're making too many requests. Please wait a moment and try again.",
                'technical_message': "API rate limit exceeded",
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.NETWORK
            },
            
            # Data errors
            'data_not_found': {
                'pattern': ['not found', 'Page Not Found', 'missing data'],
                'user_message': "The requested match or team data isn't available right now. Please try a different selection.",
                'technical_message': "Requested data not found",
                'severity': ErrorSeverity.LOW,
                'category': ErrorCategory.DATA
            },
            'data_validation_error': {
                'pattern': ['validation', 'invalid data', 'corrupt'],
                'user_message': "The data we received doesn't look right. We're using our backup data instead.",
                'technical_message': "Data validation failed",
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.DATA
            },
            
            # Prediction errors
            'prediction_failed': {
                'pattern': ['prediction', 'model error', 'calculation failed'],
                'user_message': "We couldn't generate a prediction right now. Please try again in a moment.",
                'technical_message': "Prediction generation failed",
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.PREDICTION
            },
            'model_not_available': {
                'pattern': ['model not found', 'model unavailable'],
                'user_message': "Our prediction system is temporarily unavailable. We're working to restore it.",
                'technical_message': "ML model not available",
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.PREDICTION
            },
            
            # UI errors
            'ui_render_error': {
                'pattern': ['render', 'display', 'ui error'],
                'user_message': "There was a problem displaying this page. Please refresh and try again.",
                'technical_message': "UI rendering error",
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.UI
            },
            
            # System errors
            'memory_error': {
                'pattern': ['memory', 'out of memory', 'allocation'],
                'user_message': "The system is running low on resources. Please try again in a moment.",
                'technical_message': "Memory allocation error",
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.SYSTEM
            },
            'permission_error': {
                'pattern': ['permission', 'access denied', 'unauthorized'],
                'user_message': "You don't have permission to access this feature.",
                'technical_message': "Permission denied",
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.SECURITY
            }
        }
    
    def _initialize_recovery_strategies(self):
        """Initialize recovery strategies for different error types."""
        self.recovery_strategies = {
            ErrorCategory.NETWORK: [
                "Check your internet connection",
                "Try refreshing the page",
                "Wait a few moments and try again",
                "Contact support if the problem persists"
            ],
            ErrorCategory.DATA: [
                "Try selecting a different team or match",
                "Check if the match date is correct",
                "Refresh the page to get updated data",
                "Use the search function to find alternatives"
            ],
            ErrorCategory.PREDICTION: [
                "Try generating the prediction again",
                "Select different teams if available",
                "Check back in a few minutes",
                "Use our historical data in the meantime"
            ],
            ErrorCategory.UI: [
                "Refresh the page",
                "Clear your browser cache",
                "Try using a different browser",
                "Disable browser extensions temporarily"
            ],
            ErrorCategory.SYSTEM: [
                "Wait a moment and try again",
                "Close other browser tabs to free up memory",
                "Restart your browser if needed",
                "Contact support if the issue continues"
            ],
            ErrorCategory.SECURITY: [
                "Make sure you're logged in properly",
                "Check your account permissions",
                "Contact an administrator",
                "Try logging out and back in"
            ]
        }
    
    def _initialize_retry_configs(self):
        """Initialize retry configurations for different error types."""
        self.retry_configs = {
            ErrorCategory.NETWORK: {'max_retries': 3, 'delay': 2, 'backoff': 2},
            ErrorCategory.DATA: {'max_retries': 2, 'delay': 1, 'backoff': 1.5},
            ErrorCategory.PREDICTION: {'max_retries': 2, 'delay': 3, 'backoff': 2},
            ErrorCategory.UI: {'max_retries': 1, 'delay': 1, 'backoff': 1},
            ErrorCategory.SYSTEM: {'max_retries': 1, 'delay': 5, 'backoff': 1},
            ErrorCategory.SECURITY: {'max_retries': 0, 'delay': 0, 'backoff': 1}
        }
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle error with user-friendly messaging and recovery guidance."""
        try:
            error_str = str(error).lower()
            error_type = type(error).__name__
            
            # Classify the error
            error_info = self._classify_error(error_str, error_type)
            
            # Generate user-friendly response
            response = {
                'success': False,
                'error_id': self._generate_error_id(),
                'user_message': error_info['user_message'],
                'severity': error_info['severity'].value,
                'category': error_info['category'].value,
                'recovery_suggestions': self.recovery_strategies.get(error_info['category'], []),
                'can_retry': error_info['category'] in self.retry_configs,
                'timestamp': datetime.now().isoformat(),
                'context': context or {}
            }
            
            # Add technical details for logging
            technical_details = {
                'technical_message': error_info['technical_message'],
                'error_type': error_type,
                'error_details': str(error),
                'traceback': traceback.format_exc() if error_info['severity'] in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None
            }
            
            # Log the error
            self._log_error(response, technical_details)
            
            # Store in error history
            self._store_error_history(response, technical_details)
            
            return response
            
        except Exception as e:
            logger.error(f"Error handler failed: {e}")
            return {
                'success': False,
                'error_id': self._generate_error_id(),
                'user_message': "An unexpected error occurred. Please try again.",
                'severity': ErrorSeverity.HIGH.value,
                'category': ErrorCategory.SYSTEM.value,
                'recovery_suggestions': ["Refresh the page", "Try again in a moment"],
                'can_retry': True,
                'timestamp': datetime.now().isoformat()
            }
    
    def _classify_error(self, error_str: str, error_type: str) -> Dict[str, Any]:
        """Classify error based on patterns."""
        # Check each pattern
        for pattern_name, pattern_info in self.error_patterns.items():
            for pattern in pattern_info['pattern']:
                if pattern in error_str or pattern in error_type.lower():
                    return pattern_info
        
        # Default classification
        return {
            'user_message': "An unexpected error occurred. Please try again.",
            'technical_message': f"Unclassified error: {error_type}",
            'severity': ErrorSeverity.MEDIUM,
            'category': ErrorCategory.SYSTEM
        }
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        timestamp = int(time.time() * 1000)
        return f"ERR_{timestamp}"
    
    def _log_error(self, response: Dict[str, Any], technical_details: Dict[str, Any]):
        """Log error with appropriate level."""
        severity = ErrorSeverity(response['severity'])
        
        log_message = f"Error {response['error_id']}: {technical_details['technical_message']}"
        
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra=technical_details)
        elif severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra=technical_details)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _store_error_history(self, response: Dict[str, Any], technical_details: Dict[str, Any]):
        """Store error in history for pattern analysis."""
        error_record = {
            'error_id': response['error_id'],
            'timestamp': response['timestamp'],
            'category': response['category'],
            'severity': response['severity'],
            'technical_message': technical_details['technical_message'],
            'context': response['context']
        }
        
        self.error_history.append(error_record)
        
        # Keep only last 100 errors
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def retry_with_backoff(self, func: Callable, category: ErrorCategory, *args, **kwargs):
        """Retry function with exponential backoff."""
        if category not in self.retry_configs:
            # No retry for this category
            return func(*args, **kwargs)
        
        config = self.retry_configs[category]
        max_retries = config['max_retries']
        delay = config['delay']
        backoff = config['backoff']
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if attempt < max_retries:
                    wait_time = delay * (backoff ** attempt)
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries} in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for {func.__name__}")
        
        # All retries failed, raise the last error
        raise last_error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {'total_errors': 0}
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category = error['category']
            severity = error['severity']
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'recent_errors': self.error_history[-10:] if len(self.error_history) >= 10 else self.error_history
        }


# Global singleton instance
_error_handler_instance = None

def get_error_handler() -> EnhancedErrorHandler:
    """Get global error handler instance."""
    global _error_handler_instance
    if _error_handler_instance is None:
        _error_handler_instance = EnhancedErrorHandler()
    return _error_handler_instance


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Quick function to handle errors."""
    handler = get_error_handler()
    return handler.handle_error(error, context)


def error_handler_decorator(category: ErrorCategory = ErrorCategory.SYSTEM):
    """Decorator for automatic error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = get_error_handler()
                error_response = handler.handle_error(e, {'function': func.__name__})
                
                # For UI functions, we might want to return the error response
                # For other functions, we might want to raise or handle differently
                if category == ErrorCategory.UI:
                    return error_response
                else:
                    raise
        
        return wrapper
    return decorator


def retry_on_error(category: ErrorCategory):
    """Decorator for automatic retry with backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_error_handler()
            return handler.retry_with_backoff(func, category, *args, **kwargs)
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test error handler
    handler = get_error_handler()
    
    print("🛡️ Enhanced Error Handler Test")
    
    # Test different error types
    test_errors = [
        ConnectionError("Connection timeout"),
        ValueError("Invalid data format"),
        MemoryError("Out of memory"),
        PermissionError("Access denied")
    ]
    
    for error in test_errors:
        response = handler.handle_error(error)
        print(f"\nError: {error}")
        print(f"User Message: {response['user_message']}")
        print(f"Category: {response['category']}")
        print(f"Severity: {response['severity']}")
        print(f"Recovery: {response['recovery_suggestions'][:2]}")  # Show first 2 suggestions
    
    # Show statistics
    stats = handler.get_error_statistics()
    print(f"\nError Statistics: {stats}")
