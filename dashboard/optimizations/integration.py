"""
Dashboard Performance Optimization Integration Module

This module serves as the central integration point for all performance optimizations
in the GoalDiggers dashboard. It provides:

1. Simplified imports for all optimization components
2. Dashboard-specific optimization configurations
3. Application initialization with optimizations enabled
4. Utility functions for monitoring performance
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import sys
import functools

import streamlit as st
import pandas as pd

# Import optimization modules
from dashboard.optimizations.caching import (
    optimized_data_cache,
    optimized_resource_cache,
    cached_dataframe_processing,
    lazy_load_visualizations,
    clear_all_caches,
    get_cache_stats,
    QueryCache
)

from dashboard.optimizations.lazy_loading import (
    lazy_load,
    LazyDataFrame,
    lazy_visualization,
    paginated_plotly,
    load_on_demand
)

from dashboard.optimizations.query_optimization import (
    optimized_query,
    BatchLoader,
    QueryOptimizer,
    LazyJoin,
    optimize_dataframe_memory
)

from dashboard.optimizations.render_optimization import (
    UIStateManager,
    optimize_container_hierarchy,
    progressive_render,
    render_only_if_visible,
    memoize_component,
    conditional_rendering,
    LazyTab,
    detect_mobile,
    adaptive_layout
)

from dashboard.optimizations.performance_monitor import (
    track_execution_time,
    track_query_time,
    track_component_render,
    get_performance_stats,
    render_performance_dashboard,
    start_memory_tracking,
    get_memory_snapshot
)

from dashboard.error_log import ErrorLog

logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="optimizations")

# Initialize global performance tracking
start_memory_tracking()

class DashboardOptimizer:
    """
    Central manager for dashboard optimizations.
    """
    def __init__(self):
        self.initialization_time = datetime.now()
        self.config = self._load_optimization_config()
        self.db_query_cache = QueryCache(
            max_size=self.config.get("db_query_cache_size", 200),
            default_ttl=self.config.get("db_query_cache_ttl", 300)
        )
    
    def _load_optimization_config(self) -> Dict[str, Any]:
        """Load optimization configuration."""
        # Default configuration
        default_config = {
            "enable_caching": True,
            "enable_lazy_loading": True,
            "enable_query_optimization": True,
            "enable_render_optimization": True,
            "enable_performance_monitoring": True,
            "db_query_cache_size": 200,
            "db_query_cache_ttl": 300,  # 5 minutes
            "data_cache_ttl": 3600,     # 1 hour
            "resource_cache_ttl": 86400,  # 24 hours
            "prefetch_top_leagues": True,
            "optimize_dataframe_memory": True,
            "lazy_load_visualizations": True,
            "progressive_rendering": True,
            "mobile_optimization": True
        }
        
        # Try to load from environment or config file
        try:
            from utils.config import Config
            optimization_config = Config.get("dashboard.optimizations", {})
            # Merge with defaults, prioritizing configured values
            for key, value in default_config.items():
                if key not in optimization_config:
                    optimization_config[key] = value
            return optimization_config
        except Exception as e:
            logger.warning(f"Failed to load optimization config from Config: {e}")
            return default_config
    
    def initialize_app_optimizations(self, app):
        """
        Initialize optimizations for the dashboard application.
        
        Args:
            app: DashboardApp instance to optimize
        """
        logger.info("Initializing dashboard optimizations")
        
        # Take initial memory snapshot
        get_memory_snapshot()
        
        # Apply monkey patches and optimizations based on config
        if self.config["enable_caching"]:
            self._initialize_caching(app)
        
        if self.config["enable_query_optimization"]:
            self._initialize_query_optimization(app)
        
        if self.config["enable_render_optimization"]:
            self._initialize_render_optimization(app)
        
        # Register optimization monitoring in session state
        if "optimization_metrics" not in st.session_state:
            st.session_state.optimization_metrics = {
                "initialization_time": self.initialization_time,
                "enabled_optimizations": [k for k, v in self.config.items() if v is True],
                "performance_snapshots": []
            }
        
        logger.info("Dashboard optimizations initialized successfully")
        return app
    
    def _initialize_caching(self, app):
        """Initialize caching optimizations."""
        logger.info("Initializing caching optimizations")
        
        # Apply function decorators to key methods
        if hasattr(app, "load_matches"):
            app.load_matches = optimized_data_cache(
                ttl=self.config["data_cache_ttl"]
            )(app.load_matches)
        
        if hasattr(app, "get_available_leagues"):
            app.get_available_leagues = optimized_data_cache(
                ttl=self.config["data_cache_ttl"]
            )(app.get_available_leagues)
        
        # Database connection is a resource that should be cached
        if hasattr(app, "initialize_database"):
            app.initialize_database = optimized_resource_cache()(app.initialize_database)
        
        # Cache prediction models
        if hasattr(app, "load_prediction_models"):
            app.load_prediction_models = optimized_resource_cache(
                ttl=self.config["resource_cache_ttl"]
            )(app.load_prediction_models)
        
        # Add cache stats method to app
        app.get_cache_stats = get_cache_stats
        app.clear_caches = clear_all_caches
    
    def _initialize_query_optimization(self, app):
        """Initialize database query optimizations."""
        logger.info("Initializing query optimizations")
        
        # Patch database methods with tracking and optimization
        if hasattr(app, "db_manager") and app.db_manager:
            # Add query time tracking
            original_fetchall = app.db_manager.fetchall
            app.db_manager.fetchall = track_query_time(original_fetchall)
            
            # Add batch loader for efficient querying
            app.batch_loader = BatchLoader(app.db_manager)
            
            # Add query optimizer for query analysis
            app.query_optimizer = QueryOptimizer()
    
    def _initialize_render_optimization(self, app):
        """Initialize rendering optimizations."""
        logger.info("Initializing render optimizations")
        
        # Track component rendering
        if hasattr(app, "render"):
            app.render = track_component_render("main_dashboard")(app.render)
        
        # Initialize UI state manager in session state
        if "ui_state" not in st.session_state:
            st.session_state.ui_state = {}
    
    def take_performance_snapshot(self):
        """Take a snapshot of current performance metrics."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "memory": get_memory_snapshot(),
            "cache_stats": get_cache_stats(),
            "performance_stats": get_performance_stats()
        }
        
        # Store in session state
        if "optimization_metrics" in st.session_state:
            st.session_state.optimization_metrics["performance_snapshots"].append(snapshot)
        
        return snapshot
    
    def render_optimization_dashboard(self):
        """Render the optimization monitoring dashboard."""
        render_performance_dashboard()
    
    @staticmethod
    def patch_data_loader(data_loader):
        """
        Apply optimizations to a DashboardDataLoader instance.
        
        Args:
            data_loader: DashboardDataLoader instance
            
        Returns:
            Optimized data_loader
        """
        logger.info("Patching data loader with optimizations")
        
        # Apply optimized data cache to key methods
        if hasattr(data_loader, "get_available_leagues"):
            data_loader.get_available_leagues = optimized_data_cache(
                ttl=3600,  # 1 hour
                show_spinner=False
            )(data_loader.get_available_leagues)
        
        if hasattr(data_loader, "load_matches"):
            data_loader.load_matches = optimized_data_cache(
                ttl=300,  # 5 minutes
                show_spinner=False
            )(data_loader.load_matches)
        
        if hasattr(data_loader, "load_match_details"):
            data_loader.load_match_details = optimized_data_cache(
                ttl=300,  # 5 minutes
                show_spinner=True
            )(data_loader.load_match_details)
        
        # Apply memory optimization for DataFrames
        original_load_matches = data_loader.load_matches
        
        @functools.wraps(original_load_matches)
        def optimized_load_matches(*args, **kwargs):
            df = original_load_matches(*args, **kwargs)
            if df is not None and not df.empty:
                return optimize_dataframe_memory(df)
            return df
        
        data_loader.load_matches = optimized_load_matches
        
        return data_loader
    
    @staticmethod
    def patch_db_manager(db_manager):
        """
        Apply optimizations to a DBManager instance.
        
        Args:
            db_manager: DBManager instance
            
        Returns:
            Optimized db_manager
        """
        logger.info("Patching DB manager with optimizations")
        
        # Apply query time tracking to database methods
        if hasattr(db_manager, "fetchall"):
            db_manager.fetchall = track_query_time(db_manager.fetchall)
        
        if hasattr(db_manager, "fetchone"):
            db_manager.fetchone = track_query_time(db_manager.fetchone)
        
        if hasattr(db_manager, "execute"):
            db_manager.execute = track_query_time(db_manager.execute)
        
        return db_manager
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply memory optimizations to a DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        return optimize_dataframe_memory(df)

# Initialize optimizer as a singleton
dashboard_optimizer = DashboardOptimizer()

# Convenience function for importing
def initialize_optimizations(app):
    """
    Initialize all optimizations for a dashboard app.
    
    Args:
        app: DashboardApp instance
        
    Returns:
        Optimized app
    """
    return dashboard_optimizer.initialize_app_optimizations(app)
