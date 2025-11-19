"""
Streamlit Rendering Optimization Module

This module provides utilities for optimizing Streamlit UI rendering:

1. Reducing unnecessary re-renders
2. Optimizing component hierarchies
3. Efficient state management
4. Progressive rendering of complex UIs
"""

import functools
import hashlib
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.error_log import ErrorLog
from dashboard.optimizations.caching import optimized_data_cache

logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="render_optimization")

class UIStateManager:
    """
    Manager for Streamlit UI state to prevent unnecessary re-renders.
    """
    @staticmethod
    def init_session_state(key: str, default_value: Any) -> Any:
        """
        Initialize a session state variable if it doesn't exist.
        
        Args:
            key: Session state key
            default_value: Default value if key doesn't exist
            
        Returns:
            Current value of the session state variable
        """
        if key not in st.session_state:
            st.session_state[key] = default_value
        return st.session_state[key]
    
    @staticmethod
    def update_session_state(key: str, value: Any) -> None:
        """
        Update a session state variable only if the value has changed.
        
        Args:
            key: Session state key
            value: New value
        """
        # Check if value has changed to avoid triggering a re-render
        if key not in st.session_state or st.session_state[key] != value:
            st.session_state[key] = value
    
    @staticmethod
    def toggle_session_state(key: str) -> bool:
        """
        Toggle a boolean session state variable.
        
        Args:
            key: Session state key
            
        Returns:
            New value after toggling
        """
        current = st.session_state.get(key, False)
        st.session_state[key] = not current
        return st.session_state[key]
    
    @staticmethod
    def track_component_render(component_key: str) -> None:
        """
        Track when a component is rendered to detect unnecessary re-renders.
        
        Args:
            component_key: Unique key for the component
        """
        # Initialize the render counter if it doesn't exist
        if "render_counters" not in st.session_state:
            st.session_state.render_counters = {}
        
        # Initialize counter for this component if it doesn't exist
        if component_key not in st.session_state.render_counters:
            st.session_state.render_counters[component_key] = 0
        
        # Increment the counter
        st.session_state.render_counters[component_key] += 1
        
        # Log excessive re-renders
        if st.session_state.render_counters[component_key] > 5:
            logger.warning(f"Component {component_key} has rendered {st.session_state.render_counters[component_key]} times")

def optimize_container_hierarchy(func: Callable) -> Callable:
    """
    Decorator to optimize Streamlit container hierarchy for better performance.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Generate a stable component key
        component_key = kwargs.get("key", f"{func.__name__}_{hash(str(args))}")
        
        # Track rendering of this component
        UIStateManager.track_component_render(component_key)
        
        # Create a container for this component
        with st.container():
            result = func(*args, **kwargs)
            return result
    
    return wrapper

def progressive_render(elements: List[Callable], 
                       delay_ms: int = 0, 
                       parallel: bool = False) -> None:
    """
    Render UI elements progressively to improve perceived performance.
    
    Args:
        elements: List of callable UI element functions
        delay_ms: Millisecond delay between elements (simulated)
        parallel: Whether to render elements in parallel
    """
    if parallel:
        # In parallel mode, render all elements without artificial delay
        for i, element_func in enumerate(elements):
            with st.container():
                element_func()
    else:
        # In sequential mode, render elements with optional delay
        for i, element_func in enumerate(elements):
            with st.container():
                element_func()
            
            # Simulate delay between renders (doesn't actually pause in Streamlit)
            if delay_ms > 0 and i < len(elements) - 1:
                st.empty()  # Empty placeholder for visual separation

def render_only_if_visible(component_func: Callable, 
                           condition_key: str, 
                           default_visible: bool = True) -> None:
    """
    Render a component only if it's marked as visible in session state.
    
    Args:
        component_func: Function that renders the component
        condition_key: Session state key that controls visibility
        default_visible: Default visibility if key doesn't exist
    """
    # Initialize visibility state if needed
    visible = UIStateManager.init_session_state(condition_key, default_visible)
    
    # Render toggle control
    toggle_label = "Hide" if visible else "Show"
    if st.button(f"{toggle_label} Section", key=f"toggle_{condition_key}"):
        UIStateManager.toggle_session_state(condition_key)
        st.rerun()  # Force rerun to update visibility
    
    # Render component if visible
    if visible:
        component_func()

def memoize_component(ttl: int = 300):
    """
    Decorator for memoizing UI components to prevent unnecessary re-renders.
    
    Args:
        ttl: Cache time-to-live in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        # Create a cache key prefix based on the function name
        cache_key_prefix = f"memoized_{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a stable hash for the arguments
            arg_hash = hashlib.md5(str(args).encode() + str(sorted(kwargs.items())).encode()).hexdigest()
            cache_key = f"{cache_key_prefix}_{arg_hash}"
            
            # Initialize cache in session state if needed
            if "component_cache" not in st.session_state:
                st.session_state.component_cache = {}
            
            # Check if we have a cached result
            if cache_key in st.session_state.component_cache:
                cache_entry = st.session_state.component_cache[cache_key]
                
                # Check if the cache is still valid
                if time.time() - cache_entry["timestamp"] < ttl:
                    # Use cached placeholder instead of re-rendering
                    return cache_entry["placeholder"]
            
            # Create a placeholder for this component
            placeholder = st.empty()
            
            # Render into the placeholder
            with placeholder.container():
                result = func(*args, **kwargs)
            
            # Cache the placeholder and result
            st.session_state.component_cache[cache_key] = {
                "placeholder": placeholder,
                "result": result,
                "timestamp": time.time()
            }
            
            return result
        
        return wrapper
    
    return decorator

def conditional_rendering(condition: bool, 
                          true_component: Callable, 
                          false_component: Optional[Callable] = None) -> None:
    """
    Conditionally render different components based on a condition.
    
    Args:
        condition: Boolean condition
        true_component: Component to render if condition is True
        false_component: Optional component to render if condition is False
    """
    if condition:
        true_component()
    elif false_component:
        false_component()

class LazyTab:
    """
    Utility for lazy-loading tabs in Streamlit to improve initial load time.
    """
    def __init__(self, tabs: List[str]):
        """
        Initialize lazy tabs.
        
        Args:
            tabs: List of tab names
        """
        self.tabs = tabs
        self.tab_objects = None
        self.active_tab = None
    
    def create(self):
        """Create the tab objects."""
        self.tab_objects = st.tabs(self.tabs)
        return self.tab_objects
    
    def __getitem__(self, index: int):
        """Get a specific tab by index."""
        if self.tab_objects is None:
            self.create()
        return self.tab_objects[index]
    
    def render_content(self, tab_index: int, content_func: Callable) -> None:
        """
        Render content for a specific tab only when that tab is active.
        
        Args:
            tab_index: Index of the tab
            content_func: Function that renders the tab content
        """
        if self.tab_objects is None:
            self.create()
        
        # Check if we need to update the active tab
        current_tab = UIStateManager.init_session_state("active_tab", 0)
        
        # Check if active tab has changed via a user click
        # This is a heuristic - Streamlit doesn't provide direct tab click events
        if current_tab != tab_index:
            # Use the tab context to check if it's active
            with self.tab_objects[tab_index]:
                placeholder = st.empty()
                with placeholder.container():
                    # If we can render something in this tab, it might be active
                    st.session_state.active_tab = tab_index
        
        # Only render content for the active tab
        if st.session_state.active_tab == tab_index:
            with self.tab_objects[tab_index]:
                content_func()

def detect_mobile():
    """
    Detect if the dashboard is being viewed on a mobile device.
    Uses a heuristic based on Streamlit's width.
    
    Returns:
        Boolean indicating if device is likely mobile
    """
    # This is a heuristic since Streamlit doesn't provide direct access to user agent
    try:
        # Check if a UI element renders with a small width
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            test_element = st.empty()
        mobile = test_element._element.width < 768  # Likely mobile if width is small
        test_element.empty()  # Remove the test element
        return mobile
    except:
        # Default to desktop if we can't detect
        return False

def adaptive_layout(mobile_layout: Callable, desktop_layout: Callable):
    """
    Render different layouts based on device type.
    
    Args:
        mobile_layout: Function to render mobile layout
        desktop_layout: Function to render desktop layout
    """
    is_mobile = detect_mobile()
    if is_mobile:
        mobile_layout()
    else:
        desktop_layout()

def optimize_streamlit_rendering():
    """
    Applies a set of Streamlit rendering optimizations.
    This function can be expanded to include various techniques from this module.
    For now, it serves as an entry point and logs its execution.
    """
    logger.info("Attempting to apply Streamlit rendering optimizations.")
    
    # Example: Could potentially initialize or configure some global settings here
    # UIStateManager.init_session_state("render_optimizations_active", True)
    
    # Future: Call specific optimization functions as needed
    # For example, one might want to apply a global patch or setup a global cache strategy here.
    # However, many optimizations are best applied as decorators or context managers directly
    # where components are defined or used.
    
    logger.info("Streamlit rendering optimization hook executed.")
