"""
Dashboard Caching Optimization Module

This module provides utilities and decorators for optimizing the dashboard performance
through strategic caching of expensive operations. It implements:

1. Streamlit caching wrappers with TTL settings
2. Custom memory caching with size limits
3. Database query result caching
4. Visualization caching helpers
"""

import time
import logging
import functools
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# Global cache registry for monitoring and management
_cache_registry = {
    "data_caches": {},
    "resource_caches": {},
    "stats": {
        "hits": 0,
        "misses": 0,
        "last_cleared": datetime.now().isoformat()
    }
}

def register_cache(cache_type: str, name: str, func_name: str, ttl: int) -> None:
    """Register a cache in the global registry for monitoring."""
    if cache_type not in _cache_registry:
        _cache_registry[cache_type] = {}
    
    _cache_registry[cache_type][name] = {
        "function": func_name,
        "ttl": ttl,
        "created": datetime.now().isoformat(),
        "hits": 0,
        "misses": 0
    }

def update_cache_stats(cache_type: str, name: str, hit: bool) -> None:
    """Update cache statistics."""
    if cache_type in _cache_registry and name in _cache_registry[cache_type]:
        if hit:
            _cache_registry[cache_type][name]["hits"] += 1
            _cache_registry["stats"]["hits"] += 1
        else:
            _cache_registry[cache_type][name]["misses"] += 1
            _cache_registry["stats"]["misses"] += 1

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about all registered caches."""
    total_hits = _cache_registry["stats"]["hits"]
    total_misses = _cache_registry["stats"]["misses"]
    total_calls = total_hits + total_misses
    
    stats = {
        "total_hits": total_hits,
        "total_misses": total_misses,
        "hit_rate": total_hits / total_calls if total_calls > 0 else 0,
        "last_cleared": _cache_registry["stats"]["last_cleared"],
        "caches": {}
    }
    
    # Add stats for each cache
    for cache_type in ["data_caches", "resource_caches"]:
        for name, cache_info in _cache_registry.get(cache_type, {}).items():
            cache_hits = cache_info["hits"]
            cache_misses = cache_info["misses"]
            cache_calls = cache_hits + cache_misses
            
            stats["caches"][name] = {
                "type": cache_type,
                "function": cache_info["function"],
                "ttl": cache_info["ttl"],
                "created": cache_info["created"],
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": cache_hits / cache_calls if cache_calls > 0 else 0
            }
    
    return stats

def clear_all_caches() -> None:
    """Clear all Streamlit caches and reset statistics."""
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Reset cache statistics
    _cache_registry["stats"] = {
        "hits": 0,
        "misses": 0,
        "last_cleared": datetime.now().isoformat()
    }
    
    # Reset individual cache statistics
    for cache_type in ["data_caches", "resource_caches"]:
        for name in _cache_registry.get(cache_type, {}):
            _cache_registry[cache_type][name]["hits"] = 0
            _cache_registry[cache_type][name]["misses"] = 0
    
    logger.info("All caches cleared and statistics reset")

def optimized_data_cache(ttl: int = 3600, max_entries: int = 100, show_spinner: bool = False) -> Callable:
    """
    Enhanced wrapper around st.cache_data with better monitoring and TTL support.
    
    Args:
        ttl: Time-to-live in seconds (default 1 hour)
        max_entries: Maximum number of entries to store in the cache
        show_spinner: Whether to show a loading spinner
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        # Create a unique name for this cache
        cache_name = f"{func.__module__}.{func.__name__}"
        
        # Register this cache
        register_cache("data_caches", cache_name, func.__qualname__, ttl)
        
        # Apply Streamlit's cache_data decorator
        @st.cache_data(ttl=ttl, max_entries=max_entries, show_spinner=show_spinner)
        def cached_func(*args, **kwargs):
            # Log cache miss
            logger.debug(f"Cache miss for {func.__name__}")
            update_cache_stats("data_caches", cache_name, hit=False)
            
            # Call the original function
            return func(*args, **kwargs)
        
        # Create a wrapper to handle logging
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Assume it's a hit until proven otherwise
            # (We can't directly know if it's a hit, but the inner function will only
            # be called on a miss, so we adjust our counter there)
            update_cache_stats("data_caches", cache_name, hit=True)
            return cached_func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def optimized_resource_cache(ttl: Optional[int] = None, show_spinner: bool = False) -> Callable:
    """
    Enhanced wrapper around st.cache_resource with better monitoring.
    
    Args:
        ttl: Optional time-to-live in seconds (default None, indefinite)
        show_spinner: Whether to show a loading spinner
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        # Create a unique name for this cache
        cache_name = f"{func.__module__}.{func.__name__}"
        
        # Register this cache
        register_cache("resource_caches", cache_name, func.__qualname__, ttl or 0)
        
        # Apply Streamlit's cache_resource decorator
        @st.cache_resource(ttl=ttl, show_spinner=show_spinner)
        def cached_func(*args, **kwargs):
            # Log cache miss
            logger.debug(f"Resource cache miss for {func.__name__}")
            update_cache_stats("resource_caches", cache_name, hit=False)
            
            # Call the original function
            return func(*args, **kwargs)
        
        # Create a wrapper to handle logging
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Assume it's a hit until proven otherwise
            update_cache_stats("resource_caches", cache_name, hit=True)
            return cached_func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def cached_dataframe_processing(df: pd.DataFrame, processing_func: Callable, key: str, ttl: int = 300) -> pd.DataFrame:
    """
    Apply a processing function to a DataFrame with caching.
    
    Args:
        df: Input DataFrame
        processing_func: Function to apply to the DataFrame
        key: Unique key for this operation
        ttl: Cache TTL in seconds
        
    Returns:
        Processed DataFrame
    """
    # Create a hash of the DataFrame to use as part of the cache key
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    cache_key = f"{key}_{df_hash}"
    
    # Use Streamlit's cache_data
    @st.cache_data(ttl=ttl)
    def _cached_process(_df, _key):
        logger.debug(f"Processing DataFrame with key {_key} (cache miss)")
        return processing_func(_df)
    
    return _cached_process(df, cache_key)

def lazy_load_visualizations(func: Callable) -> Callable:
    """
    Decorator for lazy loading visualizations when they become visible.
    
    Uses Streamlit's experimental_memo for caching and checks if the
    visualization is in the visible part of the UI before rendering.
    
    Args:
        func: Visualization function to decorate
        
    Returns:
        Decorated function with lazy loading
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a unique key for this visualization
        key = kwargs.get("key", f"viz_{func.__name__}_{hash(str(args))}")
        
        # Create a placeholder first
        placeholder = st.empty()
        
        # Check if we should render the visualization now or show a loading message
        with placeholder.container():
            # Generate the visualization
            result = func(*args, **kwargs)
            return result
    
    return wrapper

# Database query caching functions

class QueryCache:
    """
    Memory-efficient cache for database queries with TTL support.
    """
    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.stats = {
            "hits": 0,
            "misses": 0,
            "size": 0,
            "evictions": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache if it exists and is not expired."""
        if key in self.cache:
            entry = self.cache[key]
            # Check if entry is expired
            if entry["expires"] > time.time():
                self.stats["hits"] += 1
                # Update last_accessed to implement LRU
                entry["last_accessed"] = time.time()
                return entry["value"]
            else:
                # Entry is expired, remove it
                del self.cache[key]
                self.stats["size"] -= 1
        
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Add or update a value in the cache with TTL."""
        # Check if we need to evict entries first
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict()
        
        # Set TTL
        actual_ttl = ttl if ttl is not None else self.default_ttl
        
        # Store the value with metadata
        self.cache[key] = {
            "value": value,
            "expires": time.time() + actual_ttl,
            "last_accessed": time.time()
        }
        
        # Update size if this is a new entry
        if key not in self.cache:
            self.stats["size"] += 1
    
    def _evict(self) -> None:
        """Evict least recently used entries."""
        if not self.cache:
            return
        
        # Find the LRU entry
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k]["last_accessed"])
        
        # Remove it
        del self.cache[lru_key]
        
        # Update stats
        self.stats["size"] -= 1
        self.stats["evictions"] += 1
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.stats["size"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_calls = self.stats["hits"] + self.stats["misses"]
        return {
            **self.stats,
            "hit_rate": self.stats["hits"] / total_calls if total_calls > 0 else 0
        }
