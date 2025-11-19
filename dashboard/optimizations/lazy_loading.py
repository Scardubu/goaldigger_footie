"""
Lazy Loading Optimization Module

This module provides utilities for lazy loading of UI components,
visualizations, and data to improve dashboard performance by:

1. Only loading components when they're visible in the viewport
2. Implementing progressive loading of large datasets
3. Optimizing visualization rendering with pagination
4. Supporting deferred calculation of expensive operations
"""

import logging
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.error_log import ErrorLog
from dashboard.optimizations.caching import optimized_data_cache

logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="lazy_loading")

def lazy_load(key: str = None, loading_text: str = "Loading...", spinner: bool = True):
    """
    Decorator for lazy loading components in Streamlit.
    
    Args:
        key: Unique key for this component
        loading_text: Text to display while loading
        spinner: Whether to show a spinner while loading
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique key if not provided
            component_key = key or f"lazy_{func.__name__}_{hash(str(args))}"
            
            # Create a placeholder for the component
            placeholder = st.empty()
            
            # Show loading indicator if spinner is True
            if spinner:
                with placeholder.container():
                    with st.spinner(loading_text):
                        # Call the function
                        result = func(*args, **kwargs)
                        return result
            else:
                # Without spinner, still use the placeholder
                with placeholder.container():
                    result = func(*args, **kwargs)
                    return result
        
        return wrapper
    
    return decorator

class LazyDataFrame:
    """
    A wrapper for pandas DataFrame that supports lazy loading and pagination.
    """
    def __init__(self, 
                 data_loader: Callable[[], pd.DataFrame], 
                 page_size: int = 50,
                 cache_ttl: int = 300,
                 use_cache: bool = True):
        """
        Initialize a LazyDataFrame.
        
        Args:
            data_loader: Function that returns a DataFrame
            page_size: Number of rows per page
            cache_ttl: Cache TTL in seconds
            use_cache: Whether to use caching
        """
        self.data_loader = data_loader
        self.page_size = page_size
        self.cache_ttl = cache_ttl
        self.use_cache = use_cache
        self._data = None
        self._total_rows = 0
        self._columns = []
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the full DataFrame (loads it if not already loaded)."""
        if self._data is None:
            self._load_data()
        return self._data
    
    def _load_data(self):
        """Load the data using the data_loader function."""
        start_time = time.time()
        
        # Load the data, with optional caching
        if self.use_cache:
            @optimized_data_cache(ttl=self.cache_ttl)
            def _cached_load():
                return self.data_loader()
            
            self._data = _cached_load()
        else:
            self._data = self.data_loader()
        
        # Store metadata about the DataFrame
        if self._data is not None:
            self._total_rows = len(self._data)
            self._columns = list(self._data.columns)
        else:
            self._data = pd.DataFrame()
            self._total_rows = 0
            self._columns = []
        
        loading_time = time.time() - start_time
        logger.debug(f"Loaded DataFrame with {self._total_rows} rows in {loading_time:.2f}s")
    
    def get_page(self, page: int) -> pd.DataFrame:
        """
        Get a specific page of the DataFrame.
        
        Args:
            page: Page number (0-indexed)
            
        Returns:
            DataFrame slice for the requested page
        """
        if self._data is None:
            self._load_data()
        
        start_idx = page * self.page_size
        end_idx = min(start_idx + self.page_size, self._total_rows)
        
        if start_idx >= self._total_rows:
            return pd.DataFrame(columns=self._columns)
        
        return self._data.iloc[start_idx:end_idx].copy()
    
    def render_paginated(self, 
                         formatter: Optional[Callable[[pd.DataFrame], Any]] = None,
                         height: Optional[int] = None,
                         use_container_width: bool = True):
        """
        Render a paginated view of the DataFrame in Streamlit.
        
        Args:
            formatter: Optional function to format the DataFrame before display
            height: Optional height for the DataFrame display
            use_container_width: Whether to use the full container width
        """
        if self._data is None:
            self._load_data()
        
        # Show DataFrame info
        st.caption(f"Total rows: {self._total_rows}")
        
        # Calculate total pages
        total_pages = (self._total_rows + self.page_size - 1) // self.page_size
        
        # If empty DataFrame, show message and return
        if self._total_rows == 0:
            st.info("No data available")
            return
        
        # Page selector
        if total_pages > 1:
            col1, col2 = st.columns([3, 1])
            with col1:
                page = st.slider("Page", 1, max(1, total_pages), 1) - 1  # 0-indexed internally
            with col2:
                st.caption(f"Page {page + 1} of {total_pages}")
        else:
            page = 0
        
        # Get the page data
        page_data = self.get_page(page)
        
        # Apply formatter if provided
        if formatter is not None:
            display_data = formatter(page_data)
            if isinstance(display_data, pd.DataFrame):
                st.dataframe(display_data, height=height, use_container_width=use_container_width)
            else:
                # If formatter returns something other than a DataFrame, just display it
                st.write(display_data)
        else:
            # Default display as DataFrame
            st.dataframe(page_data, height=height, use_container_width=use_container_width)

def lazy_visualization(data_processor: Callable, 
                       viz_function: Callable,
                       loading_text: str = "Generating visualization...",
                       key: Optional[str] = None,
                       cache_ttl: int = 300,
                       use_cache: bool = True):
    """
    Create a lazy-loaded visualization with optimized data processing.
    
    Args:
        data_processor: Function that processes data for visualization
        viz_function: Function that creates the visualization
        loading_text: Text to display while loading
        key: Unique key for this visualization
        cache_ttl: Cache TTL in seconds
        use_cache: Whether to use caching
        
    Returns:
        The visualization
    """
    # Generate a unique key if not provided
    viz_key = key or f"viz_{viz_function.__name__}_{hash(str(data_processor))}"
    
    # Create a placeholder
    placeholder = st.empty()
    
    with placeholder.container():
        with st.spinner(loading_text):
            # Process data with caching if enabled
            if use_cache:
                @optimized_data_cache(ttl=cache_ttl)
                def _cached_process():
                    return data_processor()
                
                processed_data = _cached_process()
            else:
                processed_data = data_processor()
            
            # Generate visualization
            try:
                viz = viz_function(processed_data)
                return viz
            except Exception as e:
                error_log.log(
                    "visualization_error",
                    f"Error generating visualization: {str(e)}",
                    exception=e,
                    source="lazy_visualization"
                )
                st.error(f"Error generating visualization: {str(e)}")
                return None

def paginated_plotly(df: pd.DataFrame, 
                     plot_func: Callable[[pd.DataFrame], go.Figure], 
                     page_size: int = 20,
                     height: int = 500,
                     width: Optional[int] = None):
    """
    Create a paginated Plotly visualization for large datasets.
    
    Args:
        df: Input DataFrame
        plot_func: Function to create a Plotly figure from a DataFrame slice
        page_size: Number of items per page
        height: Plot height
        width: Plot width (None for auto)
        
    Returns:
        Rendered Plotly visualization with pagination controls
    """
    # Calculate total pages
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size
    
    # If empty DataFrame, show message and return
    if total_rows == 0:
        st.info("No data available for visualization")
        return
    
    # Page selector
    if total_pages > 1:
        col1, col2 = st.columns([3, 1])
        with col1:
            page = st.slider("Page", 1, max(1, total_pages), 1) - 1  # 0-indexed internally
        with col2:
            st.caption(f"Page {page + 1} of {total_pages}")
    else:
        page = 0
    
    # Get the page data
    start_idx = page * page_size
    end_idx = min(start_idx + page_size, total_rows)
    page_data = df.iloc[start_idx:end_idx].copy()
    
    # Create and render the visualization
    try:
        fig = plot_func(page_data)
        st.plotly_chart(fig, use_container_width=(width is None), height=height, width=width)
    except Exception as e:
        error_log.log(
            "plotly_error",
            f"Error creating Plotly visualization: {str(e)}",
            exception=e,
            source="paginated_plotly"
        )
        st.error(f"Error creating visualization: {str(e)}")

def load_on_demand(data_loader: Callable, btn_text: str = "Load Data", key: Optional[str] = None):
    """
    Create a button that loads data on demand when clicked.
    
    Args:
        data_loader: Function that loads the data
        btn_text: Text to display on the button
        key: Unique key for this component
        
    Returns:
        Tuple of (data, loaded_flag)
    """
    # Generate a unique key if not provided
    component_key = key or f"demand_{hash(str(data_loader))}"
    
    # Use session state to track if data is loaded
    if f"{component_key}_loaded" not in st.session_state:
        st.session_state[f"{component_key}_loaded"] = False
        st.session_state[f"{component_key}_data"] = None
    
    # Button to load data
    if not st.session_state[f"{component_key}_loaded"]:
        if st.button(btn_text, key=f"{component_key}_btn"):
            with st.spinner("Loading data..."):
                try:
                    data = data_loader()
                    st.session_state[f"{component_key}_data"] = data
                    st.session_state[f"{component_key}_loaded"] = True
                except Exception as e:
                    error_log.log(
                        "data_loading_error",
                        f"Error loading data on demand: {str(e)}",
                        exception=e,
                        source="load_on_demand"
                    )
                    st.error(f"Error loading data: {str(e)}")
    
    return st.session_state[f"{component_key}_data"], st.session_state[f"{component_key}_loaded"]
