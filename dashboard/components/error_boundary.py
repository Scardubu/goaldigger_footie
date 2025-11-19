#!/usr/bin/env python3
"""
Error Boundary Component for Streamlit Dashboard
Provides graceful error handling and recovery for dashboard components
"""

import logging
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)

class ErrorBoundary:
    """Context manager and decorator for error boundaries."""
    
    def __init__(
        self,
        component_name: str,
        fallback_message: Optional[str] = None,
        show_details: bool = False,
        log_errors: bool = True
    ):
        """
        Initialize error boundary.
        
        Args:
            component_name: Name of the component being protected
            fallback_message: Custom message to show on error
            show_details: Whether to show expandable error details
            log_errors: Whether to log errors
        """
        self.component_name = component_name
        self.fallback_message = fallback_message or f"{component_name} temporarily unavailable"
        self.show_details = show_details
        self.log_errors = log_errors
        self.error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            
            # Log the error
            if self.log_errors:
                logger.error(
                    f"Error in {self.component_name}: {exc_val}",
                    exc_info=True,
                    extra={
                        'component': self.component_name,
                        'error_type': exc_type.__name__,
                        'timestamp': datetime.now().isoformat()
                    }
                )
            
            # Display user-friendly error
            self._render_error_ui(exc_val, exc_tb)
            
            # Suppress the exception
            return True
        
        return False
    
    def _render_error_ui(self, error: Exception, traceback_obj):
        """Render error UI in Streamlit."""
        st.error(f"⚠️ {self.fallback_message}")
        
        if self.show_details:
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(f"{type(error).__name__}: {str(error)}")
                if traceback_obj:
                    st.code(''.join(traceback.format_tb(traceback_obj)))


def error_boundary(
    component_name: str,
    fallback_message: Optional[str] = None,
    show_details: bool = False,
    return_on_error: Any = None
):
    """
    Decorator to wrap functions with error boundaries.
    
    Args:
        component_name: Name of the component
        fallback_message: Message to show on error
        show_details: Whether to show error details
        return_on_error: Value to return if error occurs
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error
                logger.error(
                    f"Error in {component_name}: {e}",
                    exc_info=True,
                    extra={'component': component_name, 'function': func.__name__}
                )
                
                # Show error in UI
                st.error(f"⚠️ {fallback_message or f'{component_name} temporarily unavailable'}")
                
                if show_details and st.checkbox(
                    f"Show error details",
                    key=f"err_details_{component_name}_{func.__name__}"
                ):
                    st.code(f"{type(e).__name__}: {str(e)}")
                    st.code(traceback.format_exc())
                
                return return_on_error
        
        return wrapper
    return decorator


def safe_component_render(
    component_func: Callable,
    component_name: str,
    fallback_message: Optional[str] = None,
    show_retry: bool = True
) -> bool:
    """
    Safely render a component with error boundary and optional retry.
    
    Args:
        component_func: Function that renders the component
        component_name: Name of the component
        fallback_message: Error message to display
        show_retry: Whether to show a retry button
    
    Returns:
        True if component rendered successfully, False if error occurred
    """
    try:
        component_func()
        return True
    except Exception as e:
        logger.error(f"Error rendering {component_name}: {e}", exc_info=True)
        
        st.error(f"⚠️ {fallback_message or f'Unable to load {component_name}'}")
        
        # Show retry button
        if show_retry:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button(f"🔄 Retry {component_name}", key=f"retry_{component_name}"):
                    st.rerun()
        
        # Show details in expander
        with st.expander("🔍 Technical Details"):
            st.code(f"{type(e).__name__}: {str(e)}")
        
        return False


class ComponentRegistry:
    """Registry to track component health and errors."""
    
    def __init__(self):
        self.components: Dict[str, Dict] = {}
    
    def register_error(self, component_name: str, error: Exception):
        """Register an error for a component."""
        if component_name not in self.components:
            self.components[component_name] = {
                'errors': [],
                'last_success': None,
                'error_count': 0
            }
        
        self.components[component_name]['errors'].append({
            'error': str(error),
            'type': type(error).__name__,
            'timestamp': datetime.now()
        })
        self.components[component_name]['error_count'] += 1
    
    def register_success(self, component_name: str):
        """Register successful render for a component."""
        if component_name not in self.components:
            self.components[component_name] = {
                'errors': [],
                'last_success': None,
                'error_count': 0
            }
        
        self.components[component_name]['last_success'] = datetime.now()
    
    def get_component_health(self, component_name: str) -> Dict:
        """Get health status of a component."""
        if component_name not in self.components:
            return {'status': 'unknown', 'error_count': 0}
        
        component = self.components[component_name]
        
        # Component is healthy if:
        # 1. Last success was recent (< 5 minutes ago)
        # 2. Error count is low (< 3)
        last_success = component.get('last_success')
        error_count = component.get('error_count', 0)
        
        if last_success and (datetime.now() - last_success).seconds < 300 and error_count < 3:
            status = 'healthy'
        elif error_count < 5:
            status = 'degraded'
        else:
            status = 'critical'
        
        return {
            'status': status,
            'error_count': error_count,
            'last_success': last_success,
            'recent_errors': component['errors'][-5:] if component['errors'] else []
        }
    
    def render_health_panel(self):
        """Render a health status panel for all components."""
        if not self.components:
            return
        
        with st.expander("🏥 Component Health Status"):
            for component_name, health in [(name, self.get_component_health(name)) for name in self.components]:
                status_emoji = {
                    'healthy': '✅',
                    'degraded': '⚠️',
                    'critical': '🚨',
                    'unknown': '❓'
                }.get(health['status'], '❓')
                
                st.write(f"{status_emoji} **{component_name}**: {health['status']} (errors: {health['error_count']})")


# Global component registry
_component_registry = ComponentRegistry()

def get_component_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _component_registry


# Example usage:
if __name__ == "__main__":
    # Example 1: Using as context manager
    with ErrorBoundary("Example Component", show_details=True):
        # Component code that might fail
        st.write("This component works!")
        # raise ValueError("Example error")
    
    # Example 2: Using as decorator
    @error_boundary("Prediction Card", show_details=True)
    def render_prediction_card():
        st.write("Rendering prediction...")
        # Component code
    
    render_prediction_card()
    
    # Example 3: Using safe render helper
    def my_component():
        st.write("My component content")
    
    safe_component_render(
        my_component,
        "My Component",
        fallback_message="Could not load data",
        show_retry=True
    )
