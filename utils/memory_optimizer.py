#!/usr/bin/env python3
"""
Memory Optimizer for GoalDiggers Platform

Optimizes memory management for long-running dashboard sessions
with intelligent garbage collection and resource monitoring.
"""

import gc
import psutil
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """
    Advanced memory optimization system for long-running sessions.
    """
    
    def __init__(self, 
                 memory_threshold_mb: int = 1000,
                 cleanup_interval: int = 300):
        """
        Initialize the memory optimizer.
        
        Args:
            memory_threshold_mb: Memory threshold in MB to trigger cleanup
            cleanup_interval: Interval in seconds between automatic cleanups
        """
        self.memory_threshold_bytes = memory_threshold_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval
        
        self.memory_stats = {}
        self.cleanup_callbacks = []
        self.monitoring_active = False
        self.cleanup_thread = None
        
        # Memory tracking
        self.peak_memory = 0
        self.cleanup_count = 0
        self.last_cleanup_time = 0
        
        logger.info(f"Memory optimizer initialized with {memory_threshold_mb}MB threshold")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024
        }
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        if df.empty:
            return df
        
        original_memory = df.memory_usage(deep=True).sum()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if df[col].dtype == 'int64':
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
            
            elif df[col].dtype == 'float64':
                if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype('float32')
        
        # Optimize object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Optimize datetime columns
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='ignore')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        
        logger.info(f"DataFrame optimized: {reduction:.1f}% memory reduction")
        return df
    
    def memory_efficient_operation(self, chunk_size: int = 10000):
        """
        Decorator for memory-efficient operations on large datasets.
        
        Args:
            chunk_size: Size of chunks to process
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(data, *args, **kwargs):
                if isinstance(data, pd.DataFrame) and len(data) > chunk_size:
                    # Process in chunks
                    results = []
                    for i in range(0, len(data), chunk_size):
                        chunk = data.iloc[i:i + chunk_size]
                        result = func(chunk, *args, **kwargs)
                        results.append(result)
                        
                        # Force garbage collection after each chunk
                        gc.collect()
                    
                    # Combine results
                    if isinstance(results[0], pd.DataFrame):
                        return pd.concat(results, ignore_index=True)
                    else:
                        return results
                else:
                    return func(data, *args, **kwargs)
            
            return wrapper
        return decorator
    
    def register_cleanup_callback(self, callback: Callable):
        """
        Register a callback function to be called during cleanup.
        
        Args:
            callback: Function to call during cleanup
        """
        self.cleanup_callbacks.append(callback)
        logger.info(f"Registered cleanup callback: {callback.__name__}")
    
    def force_cleanup(self) -> Dict[str, Any]:
        """
        Force immediate memory cleanup.
        
        Returns:
            Cleanup statistics
        """
        start_time = time.time()
        memory_before = self.get_memory_usage()
        
        # Call registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear unnecessary caches
        if hasattr(pd, 'core'):
            if hasattr(pd.core, 'common'):
                if hasattr(pd.core.common, 'clear_cache'):
                    pd.core.common.clear_cache()
        
        memory_after = self.get_memory_usage()
        cleanup_time = time.time() - start_time
        
        self.cleanup_count += 1
        self.last_cleanup_time = time.time()
        
        stats = {
            'memory_before_mb': memory_before['rss_mb'],
            'memory_after_mb': memory_after['rss_mb'],
            'memory_freed_mb': memory_before['rss_mb'] - memory_after['rss_mb'],
            'objects_collected': collected,
            'cleanup_time_seconds': cleanup_time,
            'cleanup_count': self.cleanup_count
        }
        
        logger.info(f"Memory cleanup completed: {stats['memory_freed_mb']:.1f}MB freed")
        return stats
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                memory_usage = self.get_memory_usage()
                current_memory_bytes = memory_usage['rss_mb'] * 1024 * 1024
                
                # Update peak memory
                if current_memory_bytes > self.peak_memory:
                    self.peak_memory = current_memory_bytes
                
                # Check if cleanup is needed
                if current_memory_bytes > self.memory_threshold_bytes:
                    logger.warning(f"Memory threshold exceeded: {memory_usage['rss_mb']:.1f}MB")
                    self.force_cleanup()
                
                # Store memory stats
                self.memory_stats[time.time()] = memory_usage
                
                # Keep only last hour of stats
                cutoff_time = time.time() - 3600
                self.memory_stats = {
                    t: stats for t, stats in self.memory_stats.items()
                    if t > cutoff_time
                }
                
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.cleanup_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring_active = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        logger.info("Memory monitoring stopped")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_memory = self.get_memory_usage()
        
        # Calculate memory trends
        if len(self.memory_stats) > 1:
            times = sorted(self.memory_stats.keys())
            recent_memory = [self.memory_stats[t]['rss_mb'] for t in times[-10:]]
            memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
        else:
            memory_trend = 0
        
        return {
            'current_memory_mb': current_memory['rss_mb'],
            'peak_memory_mb': self.peak_memory / 1024 / 1024,
            'memory_threshold_mb': self.memory_threshold_bytes / 1024 / 1024,
            'memory_percent': current_memory['percent'],
            'available_memory_mb': current_memory['available_mb'],
            'memory_trend_mb_per_sample': memory_trend,
            'cleanup_count': self.cleanup_count,
            'last_cleanup_ago_seconds': time.time() - self.last_cleanup_time if self.last_cleanup_time else None,
            'monitoring_active': self.monitoring_active,
            'gc_stats': {
                'counts': gc.get_count(),
                'thresholds': gc.get_threshold()
            }
        }
    
    def optimize_session_state(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize Streamlit session state for memory efficiency.
        
        Args:
            session_state: Streamlit session state dictionary
            
        Returns:
            Optimization statistics
        """
        original_size = 0
        optimized_size = 0
        optimized_items = 0
        
        for key, value in session_state.items():
            try:
                if isinstance(value, pd.DataFrame):
                    original_size += value.memory_usage(deep=True).sum()
                    session_state[key] = self.optimize_dataframe(value)
                    optimized_size += session_state[key].memory_usage(deep=True).sum()
                    optimized_items += 1
                
                elif isinstance(value, (list, tuple)) and len(value) > 1000:
                    # Convert large lists to numpy arrays if possible
                    try:
                        if all(isinstance(x, (int, float)) for x in value[:100]):  # Sample check
                            session_state[key] = np.array(value)
                            optimized_items += 1
                    except Exception:
                        pass
                
            except Exception as e:
                logger.error(f"Failed to optimize session state item '{key}': {e}")
        
        return {
            'optimized_items': optimized_items,
            'memory_saved_mb': (original_size - optimized_size) / 1024 / 1024,
            'optimization_ratio': (original_size - optimized_size) / original_size if original_size > 0 else 0
        }

# Global instance for easy access
memory_optimizer = MemoryOptimizer()
