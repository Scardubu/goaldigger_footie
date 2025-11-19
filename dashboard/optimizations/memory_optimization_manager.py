#!/usr/bin/env python3
"""
Memory Optimization Manager
Phase 3A: Technical Debt Resolution - Performance Optimization

This module implements comprehensive memory optimization and automatic cleanup
routines to maintain the 400MB memory target across all dashboard variants.
Addresses memory configuration inconsistencies identified in technical debt analysis.

Key Features:
- Real-time memory monitoring and alerting
- Automatic cleanup routines for unused components
- Memory-efficient caching strategies
- Component lifecycle management
- Memory leak detection and prevention
- Standardized 400MB memory target enforcement
"""

import gc
import logging
import psutil
import threading
import time
import weakref
import sys
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import tracemalloc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryPriority(Enum):
    """Memory priority levels for component management."""
    CRITICAL = 1    # Never cleanup
    HIGH = 2        # Cleanup only under pressure
    MEDIUM = 3      # Regular cleanup candidate
    LOW = 4         # Aggressive cleanup

class CleanupStrategy(Enum):
    """Memory cleanup strategies."""
    IMMEDIATE = "immediate"     # Clean up immediately
    DEFERRED = "deferred"      # Clean up after delay
    THRESHOLD = "threshold"    # Clean up when memory threshold reached
    PERIODIC = "periodic"      # Clean up on schedule

@dataclass
class MemoryConfig:
    """Memory optimization configuration."""
    target_memory_mb: float = 400.0
    warning_threshold_mb: float = 350.0
    critical_threshold_mb: float = 380.0
    cleanup_interval_s: float = 30.0
    enable_monitoring: bool = True
    enable_automatic_cleanup: bool = True
    enable_memory_profiling: bool = False

@dataclass
class ComponentMemoryInfo:
    """Memory information for a component."""
    name: str
    memory_usage_mb: float
    priority: MemoryPriority
    cleanup_strategy: CleanupStrategy
    last_accessed: float
    reference_count: int
    is_cached: bool = False

class MemoryOptimizationManager:
    """
    Comprehensive memory optimization manager for dashboard components.
    Maintains 400MB memory target through intelligent cleanup and monitoring.
    """
    
    def __init__(self, config: MemoryConfig = None):
        """Initialize memory optimization manager."""
        self.config = config or MemoryConfig()
        self.component_registry = weakref.WeakValueDictionary()
        self.component_memory_info = {}
        self.memory_history = []
        self.cleanup_callbacks = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Memory tracking
        self.peak_memory_mb = 0.0
        self.current_memory_mb = 0.0
        self.memory_warnings = 0
        self.cleanup_count = 0
        
        # Start memory profiling if enabled
        if self.config.enable_memory_profiling:
            tracemalloc.start()
        
        logger.info(f"🧠 Memory optimization manager initialized (Target: {self.config.target_memory_mb}MB)")
        
        # Start monitoring if enabled
        if self.config.enable_monitoring:
            self.start_monitoring()
    
    def register_component(self, name: str, component: Any, priority: MemoryPriority = MemoryPriority.MEDIUM,
                          cleanup_strategy: CleanupStrategy = CleanupStrategy.THRESHOLD,
                          cleanup_callback: Callable = None):
        """Register component for memory management."""
        try:
            # Store weak reference to component
            self.component_registry[name] = component
            
            # Estimate memory usage
            memory_usage = self._estimate_component_memory(component)
            
            # Create memory info
            self.component_memory_info[name] = ComponentMemoryInfo(
                name=name,
                memory_usage_mb=memory_usage,
                priority=priority,
                cleanup_strategy=cleanup_strategy,
                last_accessed=time.time(),
                reference_count=sys.getrefcount(component)
            )
            
            # Register cleanup callback if provided
            if cleanup_callback:
                self.cleanup_callbacks[name] = cleanup_callback
            
            logger.info(f"📝 Registered component: {name} ({memory_usage:.1f}MB, {priority.name} priority)")
            
        except Exception as e:
            logger.error(f"❌ Failed to register component {name}: {e}")
    
    def unregister_component(self, name: str):
        """Unregister component from memory management."""
        try:
            if name in self.component_registry:
                del self.component_registry[name]
            if name in self.component_memory_info:
                del self.component_memory_info[name]
            if name in self.cleanup_callbacks:
                del self.cleanup_callbacks[name]
            
            logger.info(f"🗑️ Unregistered component: {name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to unregister component {name}: {e}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get component and update access time."""
        if name in self.component_registry:
            component = self.component_registry[name]
            if component and name in self.component_memory_info:
                self.component_memory_info[name].last_accessed = time.time()
            return component
        return None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'total_memory_mb': memory_info.rss / 1024 / 1024,
                'virtual_memory_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'target_memory_mb': self.config.target_memory_mb,
                'available_memory_mb': self.config.target_memory_mb - (memory_info.rss / 1024 / 1024)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get memory usage: {e}")
            return {'total_memory_mb': 0.0}
    
    def check_memory_status(self) -> Dict[str, Any]:
        """Check current memory status and return detailed information."""
        memory_usage = self.get_memory_usage()
        current_memory = memory_usage['total_memory_mb']
        
        # Update tracking
        self.current_memory_mb = current_memory
        if current_memory > self.peak_memory_mb:
            self.peak_memory_mb = current_memory
        
        # Add to history
        self.memory_history.append({
            'timestamp': time.time(),
            'memory_mb': current_memory,
            'component_count': len(self.component_registry)
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.memory_history) > 100:
            self.memory_history = self.memory_history[-100:]
        
        # Determine status
        if current_memory >= self.config.critical_threshold_mb:
            status = 'CRITICAL'
            self.memory_warnings += 1
        elif current_memory >= self.config.warning_threshold_mb:
            status = 'WARNING'
        elif current_memory <= self.config.target_memory_mb:
            status = 'OPTIMAL'
        else:
            status = 'GOOD'
        
        return {
            'status': status,
            'current_memory_mb': current_memory,
            'target_memory_mb': self.config.target_memory_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_utilization': (current_memory / self.config.target_memory_mb) * 100,
            'component_count': len(self.component_registry),
            'cleanup_count': self.cleanup_count,
            'memory_warnings': self.memory_warnings
        }
    
    def cleanup_components(self, force: bool = False) -> Dict[str, Any]:
        """Clean up components based on priority and strategy."""
        logger.info("🧹 Starting component cleanup")
        
        cleanup_results = {
            'components_cleaned': 0,
            'memory_freed_mb': 0.0,
            'cleanup_details': []
        }
        
        memory_before = self.get_memory_usage()['total_memory_mb']
        
        # Get cleanup candidates
        candidates = self._get_cleanup_candidates(force)
        
        for name, info in candidates:
            try:
                # Call custom cleanup callback if available
                if name in self.cleanup_callbacks:
                    self.cleanup_callbacks[name]()
                
                # Remove from registry
                if name in self.component_registry:
                    del self.component_registry[name]
                
                cleanup_results['components_cleaned'] += 1
                cleanup_results['cleanup_details'].append({
                    'component': name,
                    'memory_mb': info.memory_usage_mb,
                    'priority': info.priority.name
                })
                
                logger.info(f"🗑️ Cleaned up component: {name} ({info.memory_usage_mb:.1f}MB)")
                
            except Exception as e:
                logger.error(f"❌ Failed to cleanup component {name}: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Calculate memory freed
        memory_after = self.get_memory_usage()['total_memory_mb']
        cleanup_results['memory_freed_mb'] = memory_before - memory_after
        
        self.cleanup_count += 1
        
        logger.info(f"✅ Cleanup completed: {cleanup_results['components_cleaned']} components, {cleanup_results['memory_freed_mb']:.1f}MB freed")
        
        return cleanup_results
    
    def _get_cleanup_candidates(self, force: bool = False) -> List[tuple]:
        """Get list of components that are candidates for cleanup."""
        candidates = []
        current_time = time.time()
        
        for name, info in self.component_memory_info.items():
            # Skip critical components unless forced
            if info.priority == MemoryPriority.CRITICAL and not force:
                continue
            
            # Check if component is still referenced
            if name not in self.component_registry:
                candidates.append((name, info))
                continue
            
            # Apply cleanup strategy
            should_cleanup = False
            
            if info.cleanup_strategy == CleanupStrategy.IMMEDIATE:
                should_cleanup = True
            elif info.cleanup_strategy == CleanupStrategy.THRESHOLD:
                memory_status = self.check_memory_status()
                should_cleanup = memory_status['status'] in ['WARNING', 'CRITICAL']
            elif info.cleanup_strategy == CleanupStrategy.PERIODIC:
                time_since_access = current_time - info.last_accessed
                should_cleanup = time_since_access > self.config.cleanup_interval_s
            elif info.cleanup_strategy == CleanupStrategy.DEFERRED:
                time_since_access = current_time - info.last_accessed
                should_cleanup = time_since_access > (self.config.cleanup_interval_s * 2)
            
            if should_cleanup or force:
                candidates.append((name, info))
        
        # Sort by priority (lowest priority first) and last access time
        candidates.sort(key=lambda x: (x[1].priority.value, -x[1].last_accessed))
        
        return candidates
    
    def _estimate_component_memory(self, component: Any) -> float:
        """Estimate memory usage of a component."""
        try:
            # Try to get actual memory usage if possible
            if hasattr(component, '__sizeof__'):
                size_bytes = sys.getsizeof(component)
                return size_bytes / 1024 / 1024
            
            # Estimate based on component type
            component_type = type(component).__name__
            
            # Default estimates for common component types
            estimates = {
                'EnhancedPredictionEngine': 50.0,
                'CrossLeagueHandler': 25.0,
                'EnhancedTeamDataManager': 30.0,
                'PreferenceEngine': 20.0,
                'DataLoader': 15.0,
                'ResponsiveVisualizationEngine': 35.0
            }
            
            return estimates.get(component_type, 10.0)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to estimate memory for component: {e}")
            return 10.0  # Default estimate
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_memory():
            while self.monitoring_active:
                try:
                    status = self.check_memory_status()
                    
                    # Log status if not optimal
                    if status['status'] != 'OPTIMAL':
                        logger.warning(f"⚠️ Memory status: {status['status']} ({status['current_memory_mb']:.1f}MB / {status['target_memory_mb']:.1f}MB)")
                    
                    # Trigger automatic cleanup if enabled and needed
                    if (self.config.enable_automatic_cleanup and 
                        status['status'] in ['WARNING', 'CRITICAL']):
                        self.cleanup_components()
                    
                    time.sleep(self.config.cleanup_interval_s)
                    
                except Exception as e:
                    logger.error(f"❌ Memory monitoring error: {e}")
                    time.sleep(5)  # Short delay before retry
        
        self.monitoring_thread = threading.Thread(target=monitor_memory, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("🔍 Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("🛑 Memory monitoring stopped")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report."""
        status = self.check_memory_status()
        
        # Component breakdown
        component_breakdown = []
        for name, info in self.component_memory_info.items():
            component_breakdown.append({
                'name': name,
                'memory_mb': info.memory_usage_mb,
                'priority': info.priority.name,
                'last_accessed': info.last_accessed,
                'is_active': name in self.component_registry
            })
        
        # Sort by memory usage
        component_breakdown.sort(key=lambda x: x['memory_mb'], reverse=True)
        
        return {
            'memory_status': status,
            'component_breakdown': component_breakdown,
            'memory_history': self.memory_history[-10:],  # Last 10 entries
            'configuration': {
                'target_memory_mb': self.config.target_memory_mb,
                'warning_threshold_mb': self.config.warning_threshold_mb,
                'critical_threshold_mb': self.config.critical_threshold_mb,
                'monitoring_enabled': self.config.enable_monitoring,
                'automatic_cleanup_enabled': self.config.enable_automatic_cleanup
            },
            'recommendations': self._generate_memory_recommendations(status)
        }
    
    def _generate_memory_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if status['status'] == 'CRITICAL':
            recommendations.append("🚨 Critical memory usage - immediate cleanup required")
            recommendations.append("Consider reducing component count or increasing memory limit")
        elif status['status'] == 'WARNING':
            recommendations.append("⚠️ High memory usage - monitor closely")
            recommendations.append("Consider enabling automatic cleanup")
        elif status['status'] == 'OPTIMAL':
            recommendations.append("✅ Memory usage is optimal")
        
        # Component-specific recommendations
        if len(self.component_registry) > 10:
            recommendations.append("Consider reducing number of active components")
        
        if self.memory_warnings > 5:
            recommendations.append("Frequent memory warnings - consider increasing memory target")
        
        return recommendations
    
    def force_memory_optimization(self) -> Dict[str, Any]:
        """Force comprehensive memory optimization."""
        logger.info("🚀 Starting forced memory optimization")
        
        # 1. Clean up all non-critical components
        cleanup_results = self.cleanup_components(force=False)
        
        # 2. Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # 3. Clear memory history to free space
        self.memory_history = self.memory_history[-10:]
        
        # 4. Get final status
        final_status = self.check_memory_status()
        
        return {
            'cleanup_results': cleanup_results,
            'final_memory_status': final_status,
            'optimization_successful': final_status['status'] in ['OPTIMAL', 'GOOD']
        }
    
    def __del__(self):
        """Cleanup when manager is destroyed."""
        self.stop_monitoring()

# Singleton instance for global access
_memory_manager_instance = None

def get_memory_manager(config: MemoryConfig = None) -> MemoryOptimizationManager:
    """Get singleton memory optimization manager instance."""
    global _memory_manager_instance
    
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryOptimizationManager(config)
    
    return _memory_manager_instance

# Convenience functions
def register_component_for_memory_management(name: str, component: Any, 
                                           priority: MemoryPriority = MemoryPriority.MEDIUM):
    """Register component for memory management."""
    manager = get_memory_manager()
    manager.register_component(name, component, priority)

def check_memory_status() -> Dict[str, Any]:
    """Check current memory status."""
    manager = get_memory_manager()
    return manager.check_memory_status()

def cleanup_memory() -> Dict[str, Any]:
    """Trigger memory cleanup."""
    manager = get_memory_manager()
    return manager.cleanup_components()
