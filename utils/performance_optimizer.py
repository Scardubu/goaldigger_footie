"""
Performance Optimizer for the football betting insights platform.
Provides caching, batching, and other optimization techniques to improve
system performance when dealing with large datasets.
"""
import functools
import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import hashlib
import json
import threading
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Thread-safe cache implementation
class DataCache:
    """
    Thread-safe cache implementation for expensive operations.
    
    Features:
    - Time-based expiration
    - Size limits
    - Key-based invalidation
    - Stats tracking
    """
    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items in the cache
            default_ttl: Default time-to-live in seconds
        """
        self._cache = {}  # {key: (value, timestamp)}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0,
            "invalidations": 0
        }
    
    def get(self, key: str, default=None) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                # Check if expired
                if datetime.now() - timestamp > timedelta(seconds=self._default_ttl):
                    # Expired
                    del self._cache[key]
                    self._stats["evictions"] += 1
                    self._stats["size"] = len(self._cache)
                    self._stats["misses"] += 1
                    return default
                
                # Valid cache hit
                self._stats["hits"] += 1
                return value
            
            # Cache miss
            self._stats["misses"] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (overrides default)
        """
        with self._lock:
            # Enforce size limit - remove least recently used
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Simple LRU implementation - remove oldest item
                oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
            
            # Add to cache with timestamp
            self._cache[key] = (value, datetime.now())
            self._stats["size"] = len(self._cache)
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a specific cache key.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if key was in cache, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["invalidations"] += 1
                self._stats["size"] = len(self._cache)
                return True
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys containing the pattern.
        
        Args:
            pattern: String pattern to match in keys
            
        Returns:
            Number of keys invalidated
        """
        invalidated = 0
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._cache[key]
                invalidated += 1
            
            self._stats["invalidations"] += invalidated
            self._stats["size"] = len(self._cache)
            return invalidated
    
    def clear(self) -> int:
        """
        Clear the entire cache.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats["invalidations"] += count
            self._stats["size"] = 0
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache stats
        """
        with self._lock:
            hit_ratio = 0
            total = self._stats["hits"] + self._stats["misses"]
            if total > 0:
                hit_ratio = self._stats["hits"] / total
            
            stats = {**self._stats, "hit_ratio": hit_ratio}
            return stats


# Create a global cache instance
_global_cache = DataCache()

def get_cache() -> DataCache:
    """Get the global cache instance."""
    global _global_cache
    return _global_cache


# Decorator for caching function results
def cached(namespace: str, ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        namespace: Cache namespace to prefix keys
        ttl: Time-to-live in seconds (uses default if None)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a cache key
            key_data = {
                "func": func.__name__,
                "args": [str(arg) for arg in args if not isinstance(arg, pd.DataFrame)],
                "kwargs": {k: str(v) for k, v in kwargs.items() if not isinstance(v, pd.DataFrame)}
            }
            
            # For DataFrames, use shape and hash of first few rows
            for i, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    if not arg.empty:
                        key_data[f"df_arg_{i}_shape"] = arg.shape
                        # Hash first few rows to identify the DataFrame
                        if len(arg) > 0:
                            subset = arg.head(2)
                            key_data[f"df_arg_{i}_hash"] = hashlib.md5(pd.util.hash_pandas_object(subset).values).hexdigest()
            
            for k, v in kwargs.items():
                if isinstance(v, pd.DataFrame):
                    if not v.empty:
                        key_data[f"df_kwarg_{k}_shape"] = v.shape
                        if len(v) > 0:
                            subset = v.head(2)
                            key_data[f"df_kwarg_{k}_hash"] = hashlib.md5(pd.util.hash_pandas_object(subset).values).hexdigest()
            
            # Create the cache key
            cache_key = f"{namespace}:{hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()}"
            
            # Check cache
            cache = get_cache()
            result = cache.get(cache_key)
            
            if result is not None:
                return result
            
            # Cache miss, call the function
            result = func(*args, **kwargs)
            
            # Cache the result
            cache.set(cache_key, result, ttl)
            
            return result
        
        # Add function to clear this function's cache
        def clear_cache():
            """Clear all cached results for this function."""
            cache = get_cache()
            return cache.invalidate_pattern(f"{namespace}:{func.__name__}")
        
        wrapper.clear_cache = clear_cache
        return wrapper
    
    return decorator


# Performance monitoring decorator
def monitor_performance(threshold: float = 1.0, log_level: str = "warning"):
    """
    Decorator to monitor function performance.
    
    Args:
        threshold: Time threshold in seconds to log as warning
        log_level: Log level for slow operations ("debug", "info", "warning", "error")
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            if duration > threshold:
                message = f"Performance alert: {func.__name__} took {duration:.2f}s (threshold: {threshold:.2f}s)"
                
                if log_level == "debug":
                    logger.debug(message)
                elif log_level == "info":
                    logger.info(message)
                elif log_level == "error":
                    logger.error(message)
                else:  # default to warning
                    logger.warning(message)
            
            return result
        
        return wrapper
    
    return decorator


# Batch processing helper
def process_in_batches(data: pd.DataFrame, batch_size: int, process_func: Callable[[pd.DataFrame], Any]) -> List[Any]:
    """
    Process a large DataFrame in batches to reduce memory usage.
    
    Args:
        data: DataFrame to process
        batch_size: Number of rows per batch
        process_func: Function to process each batch
        
    Returns:
        List of results from each batch
    """
    results = []
    
    # Process in batches
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size].copy()
        batch_result = process_func(batch)
        results.append(batch_result)
    
    return results


# DataFrame optimization
def optimize_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Optimize a DataFrame's memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        verbose: Whether to print memory savings
        
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If fewer than 50% unique values
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem
    
    if verbose:
        logger.info(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1%})")
    
    return df


# Parallel data processing
def parallel_process(data_list: List[Any], process_func: Callable[[Any], Any], 
                    max_workers: int = 4) -> List[Any]:
    """
    Process items in parallel using thread pool.
    
    Args:
        data_list: List of data items to process
        process_func: Function to process each item
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of processed results
    """
    try:
        from concurrent.futures import ThreadPoolExecutor
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_func, data_list))
        return results
    
    except Exception as e:
        logger.error(f"Error in parallel processing: {str(e)}")
        # Fallback to sequential processing
        return [process_func(item) for item in data_list]
