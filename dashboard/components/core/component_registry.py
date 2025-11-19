#!/usr/bin/env python3
"""
Component Registry System
Component Architecture Enhancement - Phase 3A Technical Debt Resolution

This module provides a centralized registry for all GoalDiggers dashboard components,
enabling efficient component management, dependency resolution, and lifecycle control.

Key Features:
- Centralized component registration and discovery
- Dependency resolution and injection
- Component lifecycle management
- Performance monitoring and health checks
- Feature flag integration
- Memory optimization and cleanup
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Type, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import weakref
import gc

from .unified_component_base import (
    UnifiedComponentBase, ComponentConfig, ComponentType, ComponentPriority
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegistryStatus(Enum):
    """Component registry status."""
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"

@dataclass
class ComponentRegistration:
    """Component registration information."""
    component_id: str
    component_class: Type[UnifiedComponentBase]
    config: ComponentConfig
    dependencies: List[str] = field(default_factory=list)
    instance: Optional[UnifiedComponentBase] = None
    registration_time: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_singleton: bool = True

class ComponentRegistry:
    """
    Centralized registry for all GoalDiggers dashboard components.
    Manages component lifecycle, dependencies, and performance.
    """
    
    def __init__(self):
        """Initialize component registry."""
        self.logger = logging.getLogger(__name__)
        
        # Registry state
        self.status = RegistryStatus.INITIALIZING
        self.registrations: Dict[str, ComponentRegistration] = {}
        self.instances: Dict[str, weakref.ref] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Performance tracking
        self.start_time = time.time()
        self.total_registrations = 0
        self.total_instantiations = 0
        self.cleanup_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Feature flags
        self.feature_flags: Dict[str, bool] = {}
        
        self.status = RegistryStatus.READY
        self.logger.info("🚀 Component registry initialized")
    
    def register_component(
        self,
        component_id: str,
        component_class: Type[UnifiedComponentBase],
        config: ComponentConfig = None,
        dependencies: List[str] = None,
        is_singleton: bool = True
    ) -> bool:
        """Register a component in the registry."""
        with self.lock:
            try:
                # Create default config if not provided
                if config is None:
                    config = ComponentConfig(
                        component_id=component_id,
                        component_type=ComponentType.UI_ELEMENT
                    )
                
                # Validate component class
                if not issubclass(component_class, UnifiedComponentBase):
                    raise ValueError(f"Component {component_id} must inherit from UnifiedComponentBase")
                
                # Check for duplicate registration
                if component_id in self.registrations:
                    self.logger.warning(f"⚠️ Component {component_id} already registered, updating")
                
                # Create registration
                registration = ComponentRegistration(
                    component_id=component_id,
                    component_class=component_class,
                    config=config,
                    dependencies=dependencies or [],
                    is_singleton=is_singleton
                )
                
                # Store registration
                self.registrations[component_id] = registration
                
                # Update dependency graph
                self._update_dependency_graph(component_id, dependencies or [])
                
                self.total_registrations += 1
                
                self.logger.info(f"✅ Registered component: {component_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Failed to register component {component_id}: {e}")
                return False
    
    def get_component(
        self,
        component_id: str,
        feature_flags: Dict[str, bool] = None,
        force_new: bool = False
    ) -> Optional[UnifiedComponentBase]:
        """Get component instance from registry."""
        with self.lock:
            try:
                # Check if component is registered
                if component_id not in self.registrations:
                    self.logger.warning(f"⚠️ Component {component_id} not registered")
                    return None
                
                registration = self.registrations[component_id]
                
                # Update access tracking
                registration.last_accessed = time.time()
                registration.access_count += 1
                
                # Check for existing singleton instance
                if registration.is_singleton and not force_new:
                    if component_id in self.instances:
                        instance_ref = self.instances[component_id]
                        instance = instance_ref()
                        if instance is not None:
                            return instance
                
                # Resolve dependencies
                dependencies = self._resolve_dependencies(component_id)
                if dependencies is None:
                    self.logger.error(f"❌ Failed to resolve dependencies for {component_id}")
                    return None
                
                # Update config with feature flags
                config = registration.config
                if feature_flags:
                    config.feature_flags.update(feature_flags)
                
                # Create new instance
                instance = registration.component_class(config)
                
                # Store singleton instance
                if registration.is_singleton:
                    self.instances[component_id] = weakref.ref(instance)
                    registration.instance = instance
                
                self.total_instantiations += 1
                
                self.logger.debug(f"🔧 Created instance of {component_id}")
                return instance
                
            except Exception as e:
                self.logger.error(f"❌ Failed to get component {component_id}: {e}")
                return None
    
    def _resolve_dependencies(self, component_id: str) -> Optional[Dict[str, UnifiedComponentBase]]:
        """Resolve component dependencies."""
        try:
            registration = self.registrations[component_id]
            dependencies = {}
            
            for dep_id in registration.dependencies:
                if dep_id not in self.registrations:
                    self.logger.error(f"❌ Dependency {dep_id} not registered for {component_id}")
                    return None
                
                # Check for circular dependencies
                if self._has_circular_dependency(component_id, dep_id):
                    self.logger.error(f"❌ Circular dependency detected: {component_id} -> {dep_id}")
                    return None
                
                # Get dependency instance
                dep_instance = self.get_component(dep_id)
                if dep_instance is None:
                    self.logger.error(f"❌ Failed to instantiate dependency {dep_id} for {component_id}")
                    return None
                
                dependencies[dep_id] = dep_instance
            
            return dependencies
            
        except Exception as e:
            self.logger.error(f"❌ Dependency resolution failed for {component_id}: {e}")
            return None
    
    def _update_dependency_graph(self, component_id: str, dependencies: List[str]):
        """Update dependency graph for circular dependency detection."""
        self.dependency_graph[component_id] = set(dependencies)
    
    def _has_circular_dependency(self, component_id: str, dependency_id: str) -> bool:
        """Check for circular dependencies using DFS."""
        visited = set()
        
        def dfs(current_id: str, target_id: str) -> bool:
            if current_id == target_id:
                return True
            
            if current_id in visited:
                return False
            
            visited.add(current_id)
            
            for dep in self.dependency_graph.get(current_id, set()):
                if dfs(dep, target_id):
                    return True
            
            return False
        
        return dfs(dependency_id, component_id)
    
    def unregister_component(self, component_id: str) -> bool:
        """Unregister component from registry."""
        with self.lock:
            try:
                if component_id not in self.registrations:
                    self.logger.warning(f"⚠️ Component {component_id} not registered")
                    return False
                
                # Clean up instance
                if component_id in self.instances:
                    instance_ref = self.instances[component_id]
                    instance = instance_ref()
                    if instance and hasattr(instance, 'cleanup'):
                        instance.cleanup()
                    del self.instances[component_id]
                
                # Remove registration
                del self.registrations[component_id]
                
                # Update dependency graph
                if component_id in self.dependency_graph:
                    del self.dependency_graph[component_id]
                
                self.logger.info(f"🗑️ Unregistered component: {component_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Failed to unregister component {component_id}: {e}")
                return False
    
    def list_components(self, component_type: ComponentType = None) -> List[str]:
        """List registered components, optionally filtered by type."""
        with self.lock:
            if component_type is None:
                return list(self.registrations.keys())
            
            return [
                comp_id for comp_id, reg in self.registrations.items()
                if reg.config.component_type == component_type
            ]
    
    def get_component_info(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a component."""
        with self.lock:
            if component_id not in self.registrations:
                return None
            
            registration = self.registrations[component_id]
            
            # Get instance info if available
            instance_info = None
            if component_id in self.instances:
                instance_ref = self.instances[component_id]
                instance = instance_ref()
                if instance:
                    instance_info = instance.get_performance_metrics()
            
            return {
                'component_id': component_id,
                'component_type': registration.config.component_type.value,
                'priority': registration.config.priority.value,
                'dependencies': registration.dependencies,
                'is_singleton': registration.is_singleton,
                'registration_time': registration.registration_time,
                'last_accessed': registration.last_accessed,
                'access_count': registration.access_count,
                'instance_active': component_id in self.instances,
                'instance_info': instance_info
            }
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status."""
        with self.lock:
            active_instances = len([
                ref for ref in self.instances.values()
                if ref() is not None
            ])
            
            component_types = {}
            for registration in self.registrations.values():
                comp_type = registration.config.component_type.value
                component_types[comp_type] = component_types.get(comp_type, 0) + 1
            
            return {
                'status': self.status.value,
                'uptime': time.time() - self.start_time,
                'total_registrations': self.total_registrations,
                'active_registrations': len(self.registrations),
                'total_instantiations': self.total_instantiations,
                'active_instances': active_instances,
                'cleanup_count': self.cleanup_count,
                'component_types': component_types,
                'memory_usage': self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate registry memory usage in MB."""
        try:
            import sys
            
            total_size = 0
            total_size += sys.getsizeof(self.registrations)
            total_size += sys.getsizeof(self.instances)
            total_size += sys.getsizeof(self.dependency_graph)
            
            return total_size / 1024 / 1024  # Convert to MB
            
        except Exception:
            return 0.0
    
    def cleanup_inactive_instances(self) -> int:
        """Clean up inactive component instances."""
        with self.lock:
            cleaned_count = 0
            
            # Find inactive instances
            inactive_ids = []
            for comp_id, instance_ref in self.instances.items():
                if instance_ref() is None:
                    inactive_ids.append(comp_id)
            
            # Remove inactive instances
            for comp_id in inactive_ids:
                del self.instances[comp_id]
                if comp_id in self.registrations:
                    self.registrations[comp_id].instance = None
                cleaned_count += 1
            
            # Force garbage collection
            gc.collect()
            
            self.cleanup_count += cleaned_count
            
            if cleaned_count > 0:
                self.logger.info(f"🧹 Cleaned up {cleaned_count} inactive component instances")
            
            return cleaned_count
    
    def update_feature_flags(self, feature_flags: Dict[str, bool]):
        """Update global feature flags for all components."""
        with self.lock:
            self.feature_flags.update(feature_flags)
            
            # Update existing instances
            for instance_ref in self.instances.values():
                instance = instance_ref()
                if instance:
                    instance.feature_flags.update(feature_flags)
            
            self.logger.info(f"🚩 Updated feature flags: {list(feature_flags.keys())}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of registry and components."""
        with self.lock:
            health_status = {
                'registry_healthy': True,
                'component_health': {},
                'issues': [],
                'recommendations': []
            }
            
            # Check registry status
            if self.status != RegistryStatus.READY:
                health_status['registry_healthy'] = False
                health_status['issues'].append(f"Registry status: {self.status.value}")
            
            # Check component health
            for comp_id, registration in self.registrations.items():
                if comp_id in self.instances:
                    instance_ref = self.instances[comp_id]
                    instance = instance_ref()
                    if instance:
                        metrics = instance.get_performance_metrics()
                        health_status['component_health'][comp_id] = {
                            'healthy': metrics['is_healthy'],
                            'error_count': metrics['error_count'],
                            'uptime': metrics['uptime']
                        }
                        
                        if not metrics['is_healthy']:
                            health_status['issues'].append(f"Component {comp_id} has errors")
            
            # Generate recommendations
            if len(self.instances) > 20:
                health_status['recommendations'].append("Consider cleaning up inactive instances")
            
            if self._estimate_memory_usage() > 100:  # MB
                health_status['recommendations'].append("Registry memory usage is high")
            
            return health_status

# Global registry instance
_global_registry: Optional[ComponentRegistry] = None

def get_component_registry() -> ComponentRegistry:
    """Get global component registry instance."""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ComponentRegistry()
    
    return _global_registry

def register_component(
    component_id: str,
    component_class: Type[UnifiedComponentBase],
    config: ComponentConfig = None,
    dependencies: List[str] = None,
    is_singleton: bool = True
) -> bool:
    """Register component in global registry."""
    registry = get_component_registry()
    return registry.register_component(
        component_id, component_class, config, dependencies, is_singleton
    )

def get_component(
    component_id: str,
    feature_flags: Dict[str, bool] = None,
    force_new: bool = False
) -> Optional[UnifiedComponentBase]:
    """Get component from global registry."""
    registry = get_component_registry()
    return registry.get_component(component_id, feature_flags, force_new)

def cleanup_registry():
    """Clean up global registry."""
    registry = get_component_registry()
    return registry.cleanup_inactive_instances()
