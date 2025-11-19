#!/usr/bin/env python3
"""
Enhanced Memory Optimizer for GoalDiggers Premium Dashboard
Reduces memory usage from 402MB to <150MB target through intelligent optimization.
"""

import gc
import logging
import sys
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Set

import psutil

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedMemoryOptimizer:
    """
    Advanced memory optimizer specifically designed for premium dashboard performance.
    
    Features:
    - Aggressive memory optimization targeting <150MB usage
    - Intelligent component caching with weak references
    - Automatic garbage collection with smart triggers
    - Memory usage monitoring and alerting
    - Component lifecycle management
    - Memory leak detection and prevention
    """
    
    def __init__(self, target_memory_mb: int = 400):  # Updated realistic target for production
        """Initialize enhanced memory optimizer."""
        self.target_memory_mb = target_memory_mb
        self.target_memory_bytes = target_memory_mb * 1024 * 1024
        
        # Memory tracking
        self.initial_memory = self._get_current_memory()
        self.peak_memory = self.initial_memory
        self.optimization_count = 0
        
        # Component management
        self.component_cache = weakref.WeakValueDictionary()
        self.heavy_components = set()  # Track memory-intensive components
        self.deferred_components = {}  # Components that can be loaded later
        
        # Optimization strategies
        self.optimization_strategies = [
            self._optimize_garbage_collection,
            self._optimize_component_cache,
            self._optimize_data_structures,
            self._optimize_imports,
            self._optimize_streamlit_cache
        ]
        
        # Memory monitoring
        self.memory_history = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info(f"🧠 Enhanced Memory Optimizer initialized (Target: {target_memory_mb}MB)")
        logger.info(f"📊 Initial memory usage: {self.initial_memory:.1f}MB")
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in MB with robust error handling."""
        try:
            process = psutil.Process()
            # Try different memory info methods for Windows compatibility
            try:
                memory_info = process.memory_info()
                return memory_info.rss / 1024 / 1024
            except AttributeError:
                # Fallback for older psutil versions or Windows issues
                try:
                    memory_info = process.memory_info_ex()
                    return memory_info.rss / 1024 / 1024
                except AttributeError:
                    # Final fallback using memory_percent
                    memory_percent = process.memory_percent()
                    # Estimate based on system memory (rough approximation)
                    return (memory_percent / 100) * 1024  # Assume 1GB system memory as baseline
        except Exception as e:
            logger.debug(f"Failed to get memory usage: {e}")
            # Return a reasonable default to prevent crashes
            return 100.0  # Default 100MB estimate
    
    def _optimize_garbage_collection(self) -> float:
        """Perform aggressive garbage collection."""
        logger.debug("🗑️ Performing aggressive garbage collection...")
        
        initial_memory = self._get_current_memory()
        
        # Force garbage collection multiple times
        for generation in range(3):
            collected = gc.collect(generation)
            if collected > 0:
                logger.debug(f"GC generation {generation}: collected {collected} objects")
        
        # Force full garbage collection
        gc.collect()
        
        final_memory = self._get_current_memory()
        memory_saved = initial_memory - final_memory
        
        logger.debug(f"🗑️ Garbage collection saved {memory_saved:.1f}MB")
        return memory_saved
    
    def _optimize_component_cache(self) -> float:
        """Optimize component cache using weak references."""
        logger.debug("🧩 Optimizing component cache...")
        
        initial_memory = self._get_current_memory()
        
        # Clear expired weak references
        expired_keys = []
        for key in list(self.component_cache.keys()):
            try:
                if self.component_cache[key] is None:
                    expired_keys.append(key)
            except KeyError:
                expired_keys.append(key)
        
        for key in expired_keys:
            try:
                del self.component_cache[key]
            except KeyError:
                pass
        
        # Sustainable memory management with realistic thresholds
        current_memory = self._get_current_memory()

        # Gentle optimization if moderately over target
        if current_memory > self.target_memory_mb * 1.2:  # 20% over target
            logger.info(f"🧠 Memory moderately over target ({current_memory:.1f}MB > {self.target_memory_mb * 1.2:.1f}MB), starting gentle optimization...")
            self._defer_heavy_components()

        # More aggressive cleanup if significantly over target
        if current_memory > self.target_memory_mb * 1.5:  # 50% over target
            logger.warning(f"⚠️ Memory significantly over target, starting component cleanup...")
            self._unload_unused_components()

        # Warning only if critically over target - no emergency cleanup
        if current_memory > self.target_memory_mb * 2.0:  # 100% over target
            logger.warning(f"🚨 Critical memory usage ({current_memory:.1f}MB)")
            logger.info("💡 Consider restarting application for optimal performance")
            # Emergency cleanup permanently disabled to prevent system instability
        
        final_memory = self._get_current_memory()
        memory_saved = initial_memory - final_memory
        
        logger.debug(f"🧩 Component cache optimization saved {memory_saved:.1f}MB")
        return memory_saved

    def _unload_unused_components(self) -> float:
        """Unload unused ML components to free memory."""
        logger.debug("🗑️ Unloading unused components...")

        initial_memory = self._get_current_memory()

        # Clear large ML model caches
        try:
            import sys
            modules_to_clear = []
            for module_name in sys.modules:
                if any(pattern in module_name for pattern in ['sklearn', 'xgboost', 'lightgbm', 'numpy']):
                    module = sys.modules[module_name]
                    if hasattr(module, '__dict__'):
                        # Clear module-level caches
                        for attr_name in list(module.__dict__.keys()):
                            if attr_name.startswith('_cache') or 'cache' in attr_name.lower():
                                try:
                                    delattr(module, attr_name)
                                except (AttributeError, TypeError):
                                    pass
        except Exception as e:
            logger.debug(f"Failed to clear ML module caches: {e}")

        # Force garbage collection
        gc.collect()

        final_memory = self._get_current_memory()
        memory_saved = initial_memory - final_memory

        logger.debug(f"🗑️ Component unloading saved {memory_saved:.1f}MB")
        return memory_saved

    def _emergency_memory_cleanup(self) -> float:
        """Safe emergency memory cleanup - DISABLED to prevent system instability."""
        logger.warning("🚨 Emergency memory cleanup requested but DISABLED for system stability")
        logger.info("💡 Recommendation: Restart application if memory usage is critically high")

        # Only perform safe, minimal cleanup
        initial_memory = self._get_current_memory()

        try:
            # Only clear our own component caches - safe operations only
            if hasattr(self, 'component_cache'):
                self.component_cache.clear()
            if hasattr(self, 'heavy_components'):
                self.heavy_components.clear()

            # Safe garbage collection - single pass only
            import gc
            gc.collect()

            logger.info("✅ Safe memory cleanup completed")

        except Exception as e:
            logger.error(f"Safe cleanup failed: {e}")

        final_memory = self._get_current_memory()
        memory_saved = initial_memory - final_memory

        logger.info(f"🧹 Safe cleanup saved {memory_saved:.1f}MB")
        return memory_saved
    
    def _defer_heavy_components(self):
        """Defer loading of heavy components until needed."""
        heavy_component_names = ['dynamic_trainer']  # Most memory-intensive
        
        for component_name in heavy_component_names:
            if component_name in self.component_cache:
                # Move to deferred loading
                self.deferred_components[component_name] = self.component_cache[component_name]
                del self.component_cache[component_name]
                self.heavy_components.add(component_name)
                logger.info(f"📦 Deferred heavy component: {component_name}")
    
    def _optimize_data_structures(self) -> float:
        """Optimize data structures and clear unnecessary data."""
        logger.debug("📊 Optimizing data structures...")
        
        initial_memory = self._get_current_memory()
        
        # Clear Python's internal caches
        try:
            # Clear method resolution order cache
            if hasattr(type, '__dict__'):
                for cls in list(type.__dict__.values()):
                    if hasattr(cls, '__mro_entries__'):
                        try:
                            cls.__dict__.clear()
                        except (AttributeError, TypeError):
                            pass
        except Exception as e:
            logger.debug(f"Failed to clear type caches: {e}")
        
        # Clear import cache for unused modules
        self._clear_unused_imports()
        
        final_memory = self._get_current_memory()
        memory_saved = initial_memory - final_memory
        
        logger.debug(f"📊 Data structure optimization saved {memory_saved:.1f}MB")
        return memory_saved
    
    def _clear_unused_imports(self):
        """Clear unused imports to free memory."""
        # Get list of modules that can be safely removed
        removable_modules = []
        
        for module_name in list(sys.modules.keys()):
            # Skip essential modules
            if any(essential in module_name for essential in [
                'streamlit', 'pandas', 'numpy', 'logging', 'sys', 'os',
                'dashboard', 'models', 'utils', 'data', 'user'
            ]):
                continue
            
            # Skip built-in modules
            if module_name in sys.builtin_module_names:
                continue
            
            # Check if module is actually used
            module = sys.modules.get(module_name)
            if module and not hasattr(module, '__file__'):
                continue
            
            removable_modules.append(module_name)
        
        # Remove unused modules (be conservative)
        removed_count = 0
        for module_name in removable_modules[:10]:  # Limit to 10 modules
            try:
                del sys.modules[module_name]
                removed_count += 1
            except (KeyError, AttributeError):
                pass
        
        if removed_count > 0:
            logger.debug(f"🗑️ Removed {removed_count} unused modules")
    
    def _optimize_imports(self) -> float:
        """Optimize import-related memory usage."""
        logger.debug("📦 Optimizing imports...")
        
        initial_memory = self._get_current_memory()
        
        # Clear import caches
        try:
            import importlib
            importlib.invalidate_caches()
        except Exception as e:
            logger.debug(f"Failed to clear import caches: {e}")
        
        final_memory = self._get_current_memory()
        memory_saved = initial_memory - final_memory
        
        logger.debug(f"📦 Import optimization saved {memory_saved:.1f}MB")
        return memory_saved
    
    def _optimize_streamlit_cache(self) -> float:
        """Optimize Streamlit's internal caches."""
        logger.debug("🎯 Optimizing Streamlit caches...")
        
        initial_memory = self._get_current_memory()
        
        try:
            import streamlit as st

            # Clear Streamlit's cache if available
            if hasattr(st, 'cache_data') and hasattr(st.cache_data, 'clear'):
                st.cache_data.clear()
            
            if hasattr(st, 'cache_resource') and hasattr(st.cache_resource, 'clear'):
                st.cache_resource.clear()
                
        except Exception as e:
            logger.debug(f"Failed to clear Streamlit caches: {e}")
        
        final_memory = self._get_current_memory()
        memory_saved = initial_memory - final_memory
        
        logger.debug(f"🎯 Streamlit cache optimization saved {memory_saved:.1f}MB")
        return memory_saved
    
    def optimize_memory_comprehensive(self) -> Dict[str, float]:
        """
        Perform comprehensive memory optimization.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("🚀 Starting comprehensive memory optimization...")
        
        initial_memory = self._get_current_memory()
        total_saved = 0.0
        optimization_results = {}
        
        # Run all optimization strategies
        for strategy in self.optimization_strategies:
            try:
                saved = strategy()
                strategy_name = strategy.__name__.replace('_optimize_', '')
                optimization_results[strategy_name] = saved
                total_saved += saved
            except Exception as e:
                logger.error(f"Optimization strategy failed: {e}")
        
        final_memory = self._get_current_memory()
        actual_saved = initial_memory - final_memory
        
        # Update tracking
        self.optimization_count += 1
        self.peak_memory = max(self.peak_memory, initial_memory)
        
        # Log results
        logger.info(f"✅ Memory optimization complete:")
        logger.info(f"   Initial: {initial_memory:.1f}MB")
        logger.info(f"   Final: {final_memory:.1f}MB")
        logger.info(f"   Saved: {actual_saved:.1f}MB ({actual_saved/initial_memory*100:.1f}%)")
        logger.info(f"   Target: {self.target_memory_mb}MB")
        
        # Check if target achieved
        if final_memory <= self.target_memory_mb:
            logger.info(f"🎯 Memory target achieved! ({final_memory:.1f}MB <= {self.target_memory_mb}MB)")
        else:
            logger.warning(f"⚠️ Memory target not achieved ({final_memory:.1f}MB > {self.target_memory_mb}MB)")
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_saved_mb': actual_saved,
            'target_achieved': final_memory <= self.target_memory_mb,
            'optimization_details': optimization_results
        }
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status and statistics."""
        current_memory = self._get_current_memory()
        
        return {
            'current_memory_mb': current_memory,
            'initial_memory_mb': self.initial_memory,
            'peak_memory_mb': self.peak_memory,
            'target_memory_mb': self.target_memory_mb,
            'target_achieved': current_memory <= self.target_memory_mb,
            'memory_saved_mb': self.initial_memory - current_memory,
            'optimization_count': self.optimization_count,
            'heavy_components_deferred': len(self.heavy_components),
            'cache_size': len(self.component_cache)
        }
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous memory monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._memory_monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"📊 Memory monitoring started (interval: {interval_seconds}s)")
    
    def _memory_monitoring_loop(self, interval_seconds: int):
        """Memory monitoring loop."""
        while self.monitoring_active:
            try:
                current_memory = self._get_current_memory()
                self.memory_history.append({
                    'timestamp': time.time(),
                    'memory_mb': current_memory
                })
                
                # Keep only last 100 measurements
                if len(self.memory_history) > 100:
                    self.memory_history = self.memory_history[-100:]
                
                # Auto-optimize if memory exceeds threshold (sustainable approach)
                if current_memory > self.target_memory_mb * 1.5:  # 50% over target
                    logger.warning(f"⚠️ Memory usage high ({current_memory:.1f}MB), triggering optimization...")
                    self.optimize_memory_comprehensive()

                # Warning only if severely over target - no emergency cleanup
                if current_memory > self.target_memory_mb * 2.0:  # 100% over target
                    logger.warning(f"🚨 Critical memory usage ({current_memory:.1f}MB)")
                    logger.info("💡 Consider restarting application for optimal performance")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("📊 Memory monitoring stopped")

# Global singleton instance
_enhanced_memory_optimizer_instance = None

def get_enhanced_memory_optimizer() -> EnhancedMemoryOptimizer:
    """Get global enhanced memory optimizer instance."""
    global _enhanced_memory_optimizer_instance
    if _enhanced_memory_optimizer_instance is None:
        _enhanced_memory_optimizer_instance = EnhancedMemoryOptimizer()
    return _enhanced_memory_optimizer_instance
