"""
Integration Manager for GoalDiggers optimizations.

This module provides integration points between the main application and
optimization components like error recovery, performance monitoring, and UI enhancements.
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

import streamlit as st

from dashboard.components.ui_enhancements import (
    render_onboarding_guide, render_personalized_recommendations,
    render_system_status_indicator, user_interaction_tracker)
from dashboard.error_log import ErrorLog, log_exceptions_decorator
# Import optimization components
from dashboard.error_recovery import error_recovery_manager
from dashboard.optimizations.data_pipeline_monitor import DataPipelineMonitor
from dashboard.optimizations.performance_monitor import PerformanceTracker
from dashboard.optimizations.render_optimization import \
    optimize_streamlit_rendering
from utils.fallback_manager import DataSourceFallbackManager

# Set up logging
logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="integration_manager")

class IntegrationManager:
    """Manages integration of optimization components with the main application."""
    
    def __init__(self, app=None):
        """
        Initialize the integration manager.
        
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
    
    def _init_session_state(self):
        """Initialize required session state variables."""
        if "optimization_initialized" not in st.session_state:
            st.session_state.optimization_initialized = True
            st.session_state.show_performance_metrics = False
            st.session_state.system_status = self._get_default_system_status()
    
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
    
    async def track_error(self, error_type: str, exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
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
        
        # Update system status based on error
        self._update_status_from_error(error_type)
        
        # Attempt recovery
        await self.attempt_error_recovery(error_type, context)
    
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
    
    async def attempt_error_recovery(self, error_type: str = None, context: Dict[str, Any] = None) -> bool:
        """
        Attempt to recover from an error.
        
        Args:
            error_type: Type of error to recover from
            context: Optional context information
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if not error_type:
            return False
            
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
                
            return True
        else:
            # Log the failed recovery attempt
            logger.warning(f"Failed to recover from {error_type}")
            return False
    
    def wrap_async_function(self, func: Callable) -> Callable:
        """
        Wrap an async function to add error handling and performance tracking.
        
        Args:
            func: Async function to wrap
            
        Returns:
            Wrapped async function
        """
        @log_exceptions_decorator
        async def wrapped_func(*args, **kwargs):
            component_name = func.__name__
            
            # Start performance tracking
            start_time = time.time()
            
            try:
                # Execute the function
                with self.performance_tracker.track_component(component_name):
                    result = await func(*args, **kwargs)
                
                # Record successful execution
                duration_ms = (time.time() - start_time) * 1000
                self.performance_tracker.record_component_render(component_name, duration_ms, func.__qualname__)
                
                return result
                
            except Exception as e:
                # Record error
                error_type = f"function_error_{component_name}"
                await self.track_error(error_type, e, {"component": component_name, "args": args})
                raise
        
        return wrapped_func
    
    def wrap_function(self, func: Callable) -> Callable:
        """
        Wrap a synchronous function to add error handling and performance tracking.
        
        Args:
            func: Function to wrap
            
        Returns:
            Wrapped function
        """
        @log_exceptions_decorator
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
                self.performance_tracker.record_component_render(component_name, duration_ms, func.__qualname__)
                
                return result
                
            except Exception as e:
                # Record error
                error_type = f"function_error_{component_name}"
                context = {"component": component_name, "args": args}
                
                # Log the error
                error_log.error(
                    f"Error in {component_name}",
                    exception=e,
                    err_type=error_type,
                    details=context
                )
                
                # Update system status
                self._update_status_from_error(error_type)
                
                # Attempt recovery in a separate task
                async def attempt_recovery_task():
                    await self.attempt_error_recovery(error_type, context)
                
                asyncio.create_task(attempt_recovery_task())
                
                # Re-raise the exception
                raise
        
        return wrapped_func
    
    def enhance_dashboard(self, dashboard):
        """
        Enhance the dashboard with optimizations and monitoring.
        
        Args:
            dashboard: DashboardApp instance to enhance
        """
        self.app = dashboard
        
        # Apply optimizations to DashboardApp methods
        if hasattr(dashboard, '_load_matches_async'):
            dashboard._load_matches_async = self.wrap_async_function(dashboard._load_matches_async)
        
        if hasattr(dashboard, 'initialize'):
            dashboard.initialize = self.wrap_async_function(dashboard.initialize)
        
        if hasattr(dashboard, 'render'):
            dashboard.render = self.wrap_async_function(dashboard.render)
            
        # Store reference to the integration manager in the dashboard
        dashboard.integration_manager = self
        
        # Optimize Streamlit rendering
        optimize_streamlit_rendering()
        
        # Add system status component to the dashboard
        self._add_system_status_component(dashboard)
        
        # Add user interaction tracking
        self._add_user_tracking(dashboard)
        
        logger.info("Dashboard enhanced with optimizations and monitoring")
    
    def _add_system_status_component(self, dashboard):
        """Add system status component to the dashboard."""
        # We'll monkey patch the render method to include our system status
        original_render = dashboard.render
        
        async def enhanced_render(self):
            # Call the original render method
            await original_render()
            
            # Add our system status component if performance metrics are enabled
            if st.session_state.get("show_performance_metrics", False):
                with st.sidebar.expander("🔍 System Status", expanded=False):
                    render_system_status_indicator(st.session_state.system_status)
                    
                    # Add button to toggle performance metrics
                    if st.button("Open Performance Dashboard"):
                        # This will be handled by app.py to open the performance dashboard
                        st.session_state.open_performance_dashboard = True
                        st.rerun()
        
        # Replace the render method
        dashboard.render = enhanced_render.__get__(dashboard, type(dashboard))
    
    def _add_user_tracking(self, dashboard):
        """Add user interaction tracking to the dashboard."""
        # Track when a match is selected
        original_handle_match_selection = dashboard.handle_match_selection
        
        def enhanced_handle_match_selection(self, match_id: str):
            # Call the original method
            original_handle_match_selection(match_id)
            
            # Track the interaction
            user_interaction_tracker.track_interaction(
                "view_match", 
                {"match_id": match_id}
            )
        
        # Replace the method
        dashboard.handle_match_selection = enhanced_handle_match_selection.__get__(dashboard, type(dashboard))
    
    def track_user_interaction(self, interaction_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Track a user interaction to improve user experience.
        
        Args:
            interaction_type: Type of interaction
            details: Optional details about the interaction
        """
        user_interaction_tracker.track_interaction(interaction_type, details)
    
    def render_dashboard_enhancements(self):
        """Render UI enhancements to the dashboard."""
        # Render onboarding guide for new users
        render_onboarding_guide()
        
        # Render personalized recommendations
        render_personalized_recommendations()

# Create a global instance
integration_manager = IntegrationManager()
