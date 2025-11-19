#!/usr/bin/env python3
"""
Advanced Lazy Loader for GoalDiggers Platform

Implements sophisticated lazy loading strategies to optimize performance:
- Component-level lazy loading
- Import optimization with caching
- Memory-efficient module loading
- Performance monitoring and metrics
"""

import asyncio
import importlib
import logging
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)

class AdvancedLazyLoader:
    """
    Advanced lazy loading system for optimal performance.
    """
    
    def __init__(self):
        """Initialize the lazy loader."""
        self._cache = {}
        self._loading_times = {}
        self._access_counts = {}
        self._lock = threading.RLock()
        self._weak_refs = weakref.WeakValueDictionary()
        
        logger.info("✅ Advanced Lazy Loader initialized")
    
    def lazy_import(self, module_name: str, attribute: str = None, alias: str = None):
        """
        Decorator for lazy importing modules.
        
        Args:
            module_name: Name of the module to import
            attribute: Specific attribute to import from module
            alias: Alias for the imported module/attribute
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = f"{module_name}.{attribute}" if attribute else module_name
                
                # Check cache first
                if cache_key in self._cache:
                    self._access_counts[cache_key] = self._access_counts.get(cache_key, 0) + 1
                    return func(*args, **kwargs)
                
                # Load module with timing
                start_time = time.time()
                
                try:
                    with self._lock:
                        # Double-check pattern
                        if cache_key not in self._cache:
                            logger.debug(f"Lazy loading: {cache_key}")
                            
                            module = importlib.import_module(module_name)
                            
                            if attribute:
                                imported_item = getattr(module, attribute)
                            else:
                                imported_item = module
                            
                            self._cache[cache_key] = imported_item
                            self._loading_times[cache_key] = time.time() - start_time
                            self._access_counts[cache_key] = 1
                            
                            logger.info(f"✅ Loaded {cache_key} in {self._loading_times[cache_key]:.3f}s")
                        
                        return func(*args, **kwargs)
                        
                except ImportError as e:
                    logger.warning(f"Failed to lazy load {cache_key}: {e}")
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_module(self, module_name: str, attribute: str = None) -> Any:
        """
        Get a module or attribute with lazy loading.
        
        Args:
            module_name: Name of the module
            attribute: Specific attribute to get
            
        Returns:
            The loaded module or attribute
        """
        cache_key = f"{module_name}.{attribute}" if attribute else module_name
        
        # Check cache
        if cache_key in self._cache:
            self._access_counts[cache_key] = self._access_counts.get(cache_key, 0) + 1
            return self._cache[cache_key]
        
        # Load with timing
        start_time = time.time()
        
        try:
            with self._lock:
                if cache_key not in self._cache:
                    logger.debug(f"Loading module: {cache_key}")
                    
                    module = importlib.import_module(module_name)
                    
                    if attribute:
                        item = getattr(module, attribute)
                    else:
                        item = module
                    
                    self._cache[cache_key] = item
                    self._loading_times[cache_key] = time.time() - start_time
                    self._access_counts[cache_key] = 1
                    
                    logger.info(f"✅ Loaded {cache_key} in {self._loading_times[cache_key]:.3f}s")
                
                return self._cache[cache_key]
                
        except ImportError as e:
            logger.error(f"Failed to load {cache_key}: {e}")
            return None
    
    def preload_modules(self, modules: Dict[str, str]):
        """
        Preload modules in background for faster access.
        
        Args:
            modules: Dict of {cache_key: module_name} to preload
        """
        def preload_worker():
            for cache_key, module_name in modules.items():
                if cache_key not in self._cache:
                    try:
                        self.get_module(module_name)
                        logger.debug(f"Preloaded: {cache_key}")
                    except Exception as e:
                        logger.warning(f"Failed to preload {cache_key}: {e}")
        
        # Run preloading in background thread
        thread = threading.Thread(target=preload_worker, daemon=True)
        thread.start()
        logger.info(f"🚀 Started preloading {len(modules)} modules")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for loaded modules."""
        return {
            'total_modules_loaded': len(self._cache),
            'total_loading_time': sum(self._loading_times.values()),
            'average_loading_time': sum(self._loading_times.values()) / len(self._loading_times) if self._loading_times else 0,
            'most_accessed': max(self._access_counts.items(), key=lambda x: x[1]) if self._access_counts else None,
            'slowest_loading': max(self._loading_times.items(), key=lambda x: x[1]) if self._loading_times else None,
            'cache_hit_rate': sum(self._access_counts.values()) / len(self._cache) if self._cache else 0,
            'modules_by_loading_time': sorted(self._loading_times.items(), key=lambda x: x[1], reverse=True)
        }
    
    def clear_cache(self, pattern: str = None):
        """
        Clear cache entries.
        
        Args:
            pattern: Optional pattern to match for selective clearing
        """
        with self._lock:
            if pattern:
                keys_to_remove = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self._cache[key]
                    self._loading_times.pop(key, None)
                    self._access_counts.pop(key, None)
                logger.info(f"Cleared {len(keys_to_remove)} cache entries matching '{pattern}'")
            else:
                self._cache.clear()
                self._loading_times.clear()
                self._access_counts.clear()
                logger.info("Cleared all cache entries")
    
    def optimize_imports(self):
        """Optimize imports based on usage patterns."""
        if not self._access_counts:
            return
        
        # Identify frequently accessed modules
        frequent_modules = {k: v for k, v in self._access_counts.items() if v > 5}
        
        # Identify slow-loading modules
        slow_modules = {k: v for k, v in self._loading_times.items() if v > 1.0}
        
        logger.info(f"📊 Import optimization analysis:")
        logger.info(f"   Frequently accessed: {len(frequent_modules)} modules")
        logger.info(f"   Slow loading: {len(slow_modules)} modules")
        
        # Suggest optimizations
        if slow_modules:
            logger.info("💡 Consider preloading slow modules:")
            for module, time_taken in sorted(slow_modules.items(), key=lambda x: x[1], reverse=True)[:3]:
                logger.info(f"   {module}: {time_taken:.3f}s")

# Global lazy loader instance
lazy_loader = AdvancedLazyLoader()

# Convenience decorators
def lazy_pandas():
    """Lazy load pandas."""
    return lazy_loader.get_module('pandas')

def lazy_plotly():
    """Lazy load plotly."""
    return lazy_loader.get_module('plotly.express')

def lazy_xgboost():
    """Lazy load XGBoost."""
    return lazy_loader.get_module('xgboost')

def lazy_streamlit():
    """Lazy load Streamlit (already loaded but for consistency)."""
    return lazy_loader.get_module('streamlit')

# Component lazy loaders
def lazy_prediction_engine():
    """Lazy load prediction engine."""
    return lazy_loader.get_module('enhanced_prediction_engine', 'EnhancedPredictionEngine')

def lazy_html_renderer():
    """Lazy load HTML renderer."""
    return lazy_loader.get_module('dashboard.components.html_renderer_fix', 'html_renderer')

def lazy_logo_system():
    """Lazy load logo system."""
    return lazy_loader.get_module('dashboard.components.logo_integration_system', 'logo_system')

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"⚡ {func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper

# Preload common modules for better performance
def initialize_preloading():
    """Initialize preloading of common modules."""
    common_modules = {
        'streamlit': 'streamlit',
        'datetime': 'datetime',
        'json': 'json',
        'logging': 'logging',
        'pathlib': 'pathlib',
        'typing': 'typing'
    }
    
    lazy_loader.preload_modules(common_modules)

# Auto-initialize preloading
initialize_preloading()

logger.info("✅ Advanced Lazy Loader module ready")
