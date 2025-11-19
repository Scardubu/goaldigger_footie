"""
GoalDiggers Dashboard Application Extension.

This module extends the main Streamlit application with advanced features like
performance monitoring, error recovery, and enhanced UI components.
"""

import asyncio
import functools
import logging
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import streamlit as st

from dashboard.components.ui_enhancements import (
    render_onboarding_guide,
    render_personalized_recommendations,
    render_system_status_indicator,
    user_interaction_tracker,
)
from dashboard.error_log import ErrorLog, log_exceptions_decorator

# Import project-specific modules
from dashboard.error_recovery import error_recovery_manager
from dashboard.optimizations.data_pipeline_monitor import DataPipelineMonitor
from dashboard.optimizations.performance_monitor import PerformanceTracker
from dashboard.optimizations.render_optimization import optimize_streamlit_rendering
from utils.fallback_manager import DataSourceFallbackManager

# Set up logging
logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="app_extension")

class AppExtension:
    """Extends the main Streamlit application with advanced features."""
    
    def __init__(self, app=None):
        """
        Initialize the application extension.
        
        Args:
            app: Optional reference to the main application
        """
        self.app = app
        self.performance_tracker = PerformanceTracker()
        self.data_pipeline_monitor = DataPipelineMonitor()
        
        # Initialize fallback manager
        self.fallback_manager = None
        self._init_fallback_manager()
        
        # Initialize session state variables
        self._init_session_state()
        
        # Track errors to attempt recovery
        self.recent_errors = []
        self.max_recent_errors = 10
    
    def _init_session_state(self):
        """Initialize required session state variables."""
        if "app_extension_initialized" not in st.session_state:
            st.session_state.app_extension_initialized = True
            st.session_state.show_performance_dashboard = False
            st.session_state.show_error_log = False
            st.session_state.system_status = self._get_default_system_status()
            st.session_state.recovery_attempts = {}
    
    def _init_fallback_manager(self):
        """Initialize the data source fallback manager."""
        try:
            # Define data sources (primary, secondary, fallback)
            primary_sources = ["api", "database"]
            secondary_sources = ["cached_api", "cached_database"]
            fallback_sources = ["local_files", "default_values"]
            
            self.fallback_manager = DataSourceFallbackManager(
                primary_sources=primary_sources,
                secondary_sources=secondary_sources,
                fallback_sources=fallback_sources
            )
            
            logger.info("Fallback manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize fallback manager: {e}")
            error_log.error(
                "Failed to initialize fallback manager",
                exception=e,
                err_type="initialization_error"
            )
    
    def _get_default_system_status(self) -> Dict[str, str]:
        """Get default system status for components."""
        return {
            "Data Pipeline": "healthy",
            "Database Connection": "healthy",
            "API Services": "healthy",
            "Model Predictions": "healthy",
            "UI Rendering": "healthy"
        }
    
    def update_system_status(self, component: str, status: str) -> None:
        """
        Update the status of a system component.
        
        Args:
            component: Name of the component
            status: Status value ('healthy', 'degraded', 'critical')
        """
        if component in st.session_state.system_status:
            st.session_state.system_status[component] = status
        else:
            st.session_state.system_status[component] = status
    
    def track_error(self, error_type: str, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Track an error for potential recovery.
        
        Args:
            error_type: Type of error
            exception: Exception object
            context: Optional context information
        """
        # Log the error
        error_log.error(
            f"Error encountered: {error_type}",
            exception=exception,
            err_type=error_type,
            details=context
        )
        
        # Add to recent errors
        self.recent_errors.append({
            "type": error_type,
            "exception": exception,
            "context": context or {},
            "timestamp": datetime.now()
        })
        
        # Limit the size of recent errors
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
        
        # Update system status based on error
        self._update_status_from_error(error_type)
    
    def _update_status_from_error(self, error_type: str) -> None:
        """Update system status based on error type."""
        if "database" in error_type.lower():
            self.update_system_status("Database Connection", "critical")
        elif "api" in error_type.lower():
            self.update_system_status("API Services", "degraded")
        elif "model" in error_type.lower() or "prediction" in error_type.lower():
            self.update_system_status("Model Predictions", "degraded")
        elif "data" in error_type.lower() or "pipeline" in error_type.lower():
            self.update_system_status("Data Pipeline", "degraded")
        elif "render" in error_type.lower() or "ui" in error_type.lower():
            self.update_system_status("UI Rendering", "degraded")
    
    async def attempt_error_recovery(self) -> None:
        """Attempt to recover from recent errors."""
        if not self.recent_errors:
            return
        
        # Start with the most recent error
        latest_error = self.recent_errors[-1]
        error_type = latest_error["type"]
        context = latest_error["context"]
        
        # Try to recover
        success = await error_recovery_manager.attempt_recovery(error_type, context)
        
        if success:
            # Log the successful recovery
            logger.info(f"Successfully recovered from {error_type}")
            
            # Update system status
            component = next((c for c, s in st.session_state.system_status.items() 
                             if s != "healthy" and c.lower() in error_type.lower()), None)
            
            if component:
                self.update_system_status(component, "healthy")
            
            # Remove the error from recent errors
            self.recent_errors = [e for e in self.recent_errors if e["type"] != error_type]
        else:
            # Log the failed recovery attempt
            logger.warning(f"Failed to recover from {error_type}")
    
    def wrap_streamlit_function(self, func: Callable) -> Callable:
        """
        Wrap a Streamlit function to add error handling and performance tracking.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            component_name = func.__name__
            
            # Start performance tracking
            start_time = time.time()
            
            try:
                # Execute the function
                with self.performance_tracker.track_component(component_name):
                    result = func(*args, **kwargs)
                
                # Record successful execution
                duration_ms = (time.time() - start_time) * 1000
                self.performance_tracker.record_component_render(component_name, duration_ms)
                
                return result
                
            except Exception as e:
                # Record error
                error_type = f"ui_render_error_{component_name}"
                self.track_error(error_type, e, {"component": component_name, "args": args})
                
                # Show error in UI
                st.error(f"Error rendering {component_name}: {str(e)}")
                
                # Attempt recovery asynchronously
                # Schedule async recovery using compatibility helper
                try:
                    from utils.asyncio_compat import ensure_loop
                    loop = ensure_loop()
                except Exception:
                    # Fallback to a brand new loop as last resort
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                try:
                    loop.create_task(self.attempt_error_recovery())
                except RuntimeError:
                    # If loop not running, run it synchronously
                    loop.run_until_complete(self.attempt_error_recovery())
        
        return wrapped_func
    
    def enhance_app(self) -> None:
        """Enhance the main application with extended features."""
        # Optimize Streamlit rendering
        optimize_streamlit_rendering()
        
        # Add the performance dashboard toggle to the sidebar
        with st.sidebar:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 🛠️ Advanced Tools")
            
            # Add button to toggle performance dashboard
            if st.sidebar.button("🚀 Performance Dashboard", key="toggle_perf_dashboard"):
                st.session_state.show_performance_dashboard = not st.session_state.show_performance_dashboard
            
            # Add button to toggle error log
            if st.sidebar.button("🔍 View Error Log", key="toggle_error_log"):
                st.session_state.show_error_log = not st.session_state.show_error_log
                
            # Add button to trigger recovery for all errors
            if st.sidebar.button("🔄 Attempt Recovery", key="trigger_recovery"):
                # Create a placeholder for the recovery message
                recovery_placeholder = st.sidebar.empty()
                recovery_placeholder.info("Attempting recovery...")
                
                # Attempt recovery
                try:
                    from utils.asyncio_compat import ensure_loop
                    loop = ensure_loop()
                except Exception:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                # If loop already running, create a task; else run directly
                if loop.is_running():
                    loop.create_task(self.attempt_error_recovery())
                else:
                    loop.run_until_complete(self.attempt_error_recovery())
                
                # Update placeholder with success message
                recovery_placeholder.success("Recovery attempt completed!")
        
        # Render the onboarding guide for new users
        render_onboarding_guide()
        
        # Render personalized recommendations
        render_personalized_recommendations()
        
        # Show the performance dashboard if enabled
        if st.session_state.show_performance_dashboard:
            self._render_performance_dashboard()
        
        # Show the error log if enabled
        if st.session_state.show_error_log:
            self._render_error_log()
    
    def _render_performance_dashboard(self) -> None:
        """Render a simplified performance dashboard."""
        st.subheader("🚀 System Performance")
        
        # Display system status
        render_system_status_indicator(st.session_state.system_status)
        
        # Show current performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get memory metrics
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            st.metric(
                label="Memory Usage",
                value=f"{memory_percent:.1f}%",
                delta=f"{memory_info.rss / (1024 * 1024):.1f} MB"
            )
        
        with col2:
            # Get data pipeline metrics
            pipeline_metrics = self.data_pipeline_monitor.get_metrics()
            
            processing_time = pipeline_metrics.get("avg_processing_time", 0)
            st.metric(
                label="Avg Processing Time",
                value=f"{processing_time:.2f} ms"
            )
        
        with col3:
            # Get UI rendering metrics
            component_metrics = self.performance_tracker.get_component_metrics()
            
            if component_metrics:
                avg_render_time = sum(component_metrics.values()) / len(component_metrics)
                st.metric(
                    label="Avg UI Render Time",
                    value=f"{avg_render_time:.2f} ms"
                )
            else:
                st.metric(
                    label="Avg UI Render Time",
                    value="N/A"
                )
        
        # Link to full dashboard
        st.markdown(
            """
            <div style="text-align:center">
                <a href="/performance_dashboard" target="_blank">Open Full Performance Dashboard</a>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _render_error_log(self) -> None:
        """Render the error log viewer."""
        st.subheader("🔍 Error Log")
        
        # Get recent errors from error_log
        recent_errors = error_log.get_recent_errors(10)
        
        if recent_errors:
            # Create a table of errors
            import pandas as pd

            # Convert to DataFrame
            error_df = pd.DataFrame(recent_errors)
            
            # Format timestamp
            error_df['timestamp'] = pd.to_datetime(error_df['timestamp'])
            error_df['formatted_time'] = error_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Show errors in a table
            st.table(error_df[['formatted_time', 'component', 'err_type', 'message']])
            
            # Add button to clear error log
            if st.button("Clear Error Log"):
                error_log.clear()
                st.success("Error log cleared!")
                st.rerun()
        else:
            st.info("No errors recorded. That's good news! 👍")
    
    def track_user_interaction(self, interaction_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Track a user interaction to improve user experience.
        
        Args:
            interaction_type: Type of interaction
            details: Optional details about the interaction
        """
        user_interaction_tracker.track_interaction(interaction_type, details)

# Function to create and configure the app extension
def create_app_extension(app=None):
    """
    Create and configure an application extension.
    
    Args:
        app: Optional reference to the main application
        
    Returns:
        Configured AppExtension instance
    """
    extension = AppExtension(app)
    return extension
