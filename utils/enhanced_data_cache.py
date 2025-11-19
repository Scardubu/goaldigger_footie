#!/usr/bin/env python3
"""
Enhanced Data Cache System for GoalDiggers Platform

Implements aggressive caching strategies for frequently accessed data
with intelligent cache invalidation and memory management.
"""

import time
import pickle
import logging
import hashlib
import threading
from typing import Any, Dict, Optional, Callable, Union
from pathlib import Path
from functools import wraps
import pandas as pd

logger = logging.getLogger(__name__)

class EnhancedDataCache:
    """
    Advanced data caching system with aggressive caching strategies.
    """
    
    def __init__(self, cache_dir: str = "cache/data", max_memory_mb: int = 500):
        """
        Initialize the enhanced data cache.
        
        Args:
            cache_dir: Directory to store cached data
            max_memory_mb: Maximum memory cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_cache = {}
        self.cache_metadata = {}
        self.access_counts = {}
        self.cache_locks = {}
        
        # Cache configuration
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_usage = 0
        self.default_ttl = 1800  # 30 minutes
        self.high_frequency_ttl = 3600  # 1 hour for frequently accessed data
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        logger.info(f"Enhanced data cache initialized: {self.cache_dir}")
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a unique cache key."""
        key_data = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate the size of an object in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj[:100])  # Sample first 100
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in list(obj.items())[:100])  # Sample first 100
            else:
                return 1024  # Default 1KB estimate
    
    def _cleanup_memory_cache(self, required_space: int = 0):
        """Clean up memory cache based on LRU and access frequency."""
        if self.current_memory_usage + required_space <= self.max_memory_bytes:
            return
        
        # Sort by access frequency and recency
        cache_items = []
        for key, metadata in self.cache_metadata.items():
            if key in self.memory_cache:
                score = (
                    self.access_counts.get(key, 0) * 0.7 +  # Access frequency weight
                    (time.time() - metadata['last_accessed']) * -0.3  # Recency weight (negative for LRU)
                )
                cache_items.append((key, score, metadata['size']))
        
        # Sort by score (lower score = higher priority for removal)
        cache_items.sort(key=lambda x: x[1])
        
        # Remove items until we have enough space
        space_needed = self.current_memory_usage + required_space - self.max_memory_bytes
        space_freed = 0
        
        for key, _, size in cache_items:
            if space_freed >= space_needed:
                break
            
            if key in self.memory_cache:
                del self.memory_cache[key]
                self.current_memory_usage -= size
                space_freed += size
                logger.debug(f"Removed from memory cache: {key} ({size} bytes)")
    
    def _save_to_disk(self, cache_key: str, data: Any):
        """Save data to disk cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"Data saved to disk cache: {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to save data to disk cache: {e}")
    
    def _load_from_disk(self, cache_key: str) -> Optional[Any]:
        """Load data from disk cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            logger.debug(f"Data loaded from disk cache: {cache_key}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load data from disk cache: {e}")
            return None
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache_metadata:
            return False
        
        metadata = self.cache_metadata[cache_key]
        ttl = self.high_frequency_ttl if self.access_counts.get(cache_key, 0) > 10 else self.default_ttl
        
        return time.time() - metadata['timestamp'] < ttl
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get data from cache."""
        self.total_requests += 1
        
        # Check memory cache first
        if cache_key in self.memory_cache and self._is_cache_valid(cache_key):
            self.cache_hits += 1
            self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
            self.cache_metadata[cache_key]['last_accessed'] = time.time()
            return self.memory_cache[cache_key]
        
        # Check disk cache
        if self._is_cache_valid(cache_key):
            data = self._load_from_disk(cache_key)
            if data is not None:
                # Move to memory cache
                size = self._estimate_size(data)
                self._cleanup_memory_cache(size)
                
                self.memory_cache[cache_key] = data
                self.current_memory_usage += size
                self.cache_metadata[cache_key]['size'] = size
                self.cache_metadata[cache_key]['last_accessed'] = time.time()
                
                self.cache_hits += 1
                self.access_counts[cache_key] = self.access_counts.get(cache_key, 0) + 1
                return data
        
        self.cache_misses += 1
        return None
    
    def set(self, cache_key: str, data: Any, ttl: Optional[int] = None):
        """Set data in cache."""
        size = self._estimate_size(data)
        
        # Skip caching if data is too large
        if size > self.max_memory_bytes * 0.5:  # Don't cache items larger than 50% of cache
            logger.warning(f"Data too large to cache: {cache_key} ({size} bytes)")
            return
        
        # Cleanup memory cache if needed
        self._cleanup_memory_cache(size)
        
        # Store in memory cache
        self.memory_cache[cache_key] = data
        self.current_memory_usage += size
        
        # Update metadata
        self.cache_metadata[cache_key] = {
            'timestamp': time.time(),
            'last_accessed': time.time(),
            'size': size,
            'ttl': ttl or self.default_ttl
        }
        
        # Save to disk in background
        threading.Thread(
            target=self._save_to_disk,
            args=(cache_key, data),
            daemon=True
        ).start()
    
    def cached_function(self, ttl: Optional[int] = None, key_prefix: str = ""):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time to live in seconds
            key_prefix: Prefix for cache keys
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                func_name = f"{key_prefix}{func.__name__}" if key_prefix else func.__name__
                cache_key = self._generate_cache_key(func_name, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def invalidate(self, pattern: str = None, cache_key: str = None):
        """
        Invalidate cache entries.
        
        Args:
            pattern: Pattern to match cache keys (simple string matching)
            cache_key: Specific cache key to invalidate
        """
        if cache_key:
            # Invalidate specific key
            if cache_key in self.memory_cache:
                size = self.cache_metadata.get(cache_key, {}).get('size', 0)
                del self.memory_cache[cache_key]
                self.current_memory_usage -= size
            
            if cache_key in self.cache_metadata:
                del self.cache_metadata[cache_key]
            
            # Remove from disk
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            
            logger.info(f"Invalidated cache key: {cache_key}")
        
        elif pattern:
            # Invalidate by pattern
            keys_to_remove = [
                key for key in self.memory_cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                self.invalidate(cache_key=key)
            
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'memory_cache_count': len(self.memory_cache),
            'memory_usage_mb': self.current_memory_usage / (1024 * 1024),
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'total_requests': self.total_requests,
            'disk_cache_files': len(list(self.cache_dir.glob("*.pkl"))),
            'top_accessed_keys': sorted(
                self.access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    def clear_cache(self, cache_type: str = "all"):
        """Clear cache."""
        if cache_type in ['memory', 'all']:
            self.memory_cache.clear()
            self.cache_metadata.clear()
            self.access_counts.clear()
            self.current_memory_usage = 0
            logger.info("Memory cache cleared")
        
        if cache_type in ['disk', 'all']:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete cache file {cache_file}: {e}")
            logger.info("Disk cache cleared")
        
        if cache_type == 'all':
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_requests = 0

# Global instance for easy access
enhanced_data_cache = EnhancedDataCache()
