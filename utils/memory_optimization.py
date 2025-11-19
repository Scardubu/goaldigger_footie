"""
Memory optimization module for the GoalDiggers platform.
Provides utilities for monitoring and optimizing memory usage.
"""

import gc
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """
    Memory optimization utilities for the GoalDiggers platform.
    Monitors and optimizes memory usage to improve performance.
    """
    
    def __init__(self, threshold_mb: float = 500.0):
        """
        Initialize the memory optimizer.
        
        Args:
            threshold_mb: Memory threshold in MB to trigger optimization
        """
        self.threshold_mb = threshold_mb
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self._get_memory_usage()
        self._last_check_time = time.time()
        self._check_interval = 60  # Check every 60 seconds
        
        # Keep track of large objects
        self._object_sizes: Dict[str, int] = {}
        
        logger.info(f"Memory optimizer initialized. Current usage: {self.initial_memory:.2f} MB")
        
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        return self.process.memory_info().rss / 1024 / 1024
        
    def check_and_optimize(self) -> None:
        """
        Check memory usage and optimize if necessary.
        """
        current_time = time.time()
        
        # Only check periodically to avoid overhead
        if current_time - self._last_check_time < self._check_interval:
            return
            
        self._last_check_time = current_time
        
        current_memory = self._get_memory_usage()
        delta = current_memory - self.initial_memory
        
        if delta > self.threshold_mb:
            logger.warning(f"Memory usage increased by {delta:.2f} MB. Optimizing...")
            self.optimize()
        else:
            logger.debug(f"Memory usage stable at {current_memory:.2f} MB")
            
    def optimize(self) -> None:
        """
        Optimize memory usage.
        """
        # Run garbage collection
        gc.collect()
        
        # Log memory usage after optimization
        new_memory = self._get_memory_usage()
        logger.info(f"Memory optimized: {new_memory:.2f} MB")
        
    def track_object(self, obj: object, name: str) -> None:
        """
        Track a large object's memory usage.
        
        Args:
            obj: The object to track
            name: A name to identify the object
        """
        import sys
        size = sys.getsizeof(obj) / 1024 / 1024  # Size in MB
        self._object_sizes[name] = size
        logger.debug(f"Tracking object {name}: {size:.2f} MB")
        
    def get_memory_report(self) -> Dict[str, float]:
        """
        Get a report of memory usage.
        
        Returns:
            Dictionary with memory metrics
        """
        current_memory = self._get_memory_usage()
        
        return {
            "current_mb": current_memory,
            "initial_mb": self.initial_memory,
            "delta_mb": current_memory - self.initial_memory,
            "tracked_objects": self._object_sizes
        }
        
    @staticmethod
    def get_largest_objects(limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get the largest objects in memory.
        
        Args:
            limit: Maximum number of objects to return
            
        Returns:
            List of (object representation, size) tuples
        """
        import gc
        import sys

        # Get all objects
        objects = []
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj)
                objects.append((str(obj)[:100], size))
            except:
                pass
                
        # Sort by size (largest first)
        objects.sort(key=lambda x: x[1], reverse=True)
        
        # Return the largest objects
        return objects[:limit]


# Singleton instance
_memory_optimizer: Optional[MemoryOptimizer] = None

def get_optimizer() -> MemoryOptimizer:
    """
    Get or create the singleton memory optimizer instance.
    
    Returns:
        MemoryOptimizer instance
    """
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer
