#!/usr/bin/env python3
"""
Dependency Resolution System for GoalDiggers Platform

Implements dependency injection and circular dependency detection for:
- ML components (enhanced_prediction_engine, adaptive_ensemble, dynamic_trainer)
- Configuration components
- Dashboard components
- Data processing components

Features:
- Dependency mapping and validation
- Circular dependency detection
- Initialization order optimization
- Component registry management
- Error handling and fallback mechanisms
"""

import logging
import time
import weakref
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

class DependencyResolver:
    """
    Dependency resolution system with circular dependency detection.
    
    Manages component dependencies and ensures proper initialization order
    while detecting and preventing circular dependencies.
    """
    
    def __init__(self):
        """Initialize dependency resolver."""
        self.component_registry = {}
        self.dependency_graph = defaultdict(list)
        self.reverse_dependency_graph = defaultdict(list)
        self.component_factories = {}
        self.initialized_components = weakref.WeakValueDictionary()
        
        # Component status tracking
        self.component_status = {}
        self.initialization_order = []
        
        # Performance tracking
        self.resolution_times = {}
        
        # Define component dependencies
        self._define_component_dependencies()
        
        logger.info("🔗 Dependency Resolver initialized")
    
    def _define_component_dependencies(self):
        """Define component dependencies for the GoalDiggers platform."""
        # ML Component Dependencies
        self.add_dependency('enhanced_prediction_engine', [])  # No dependencies
        self.add_dependency('adaptive_ensemble', [])          # No dependencies
        self.add_dependency('dynamic_trainer', [])            # No dependencies
        
        # Configuration Dependencies
        self.add_dependency('unified_config', [])             # No dependencies
        self.add_dependency('production_memory_optimizer', ['unified_config'])
        
        # Dashboard Dependencies
        self.add_dependency('optimized_dashboard', ['unified_config'])
        self.add_dependency('premium_dashboard', ['unified_config', 'enhanced_prediction_engine'])

        # Enhanced Component Dependencies
        self.add_dependency('Interface', [])
        self.add_dependency('enhanced_team_data_manager', [])
        self.add_dependency('enhanced_prediction_display', ['Interface'])

        # Data Processing Dependencies
        self.add_dependency('data_loader', ['unified_config'])
        self.add_dependency('live_data_processor', ['unified_config'])
        self.add_dependency('odds_aggregator', ['unified_config'])
        
        logger.info("📋 Component dependencies defined")
    
    def add_dependency(self, component: str, dependencies: List[str]):
        """
        Add component dependency mapping.
        
        Args:
            component: Component name
            dependencies: List of component names this component depends on
        """
        self.dependency_graph[component] = dependencies
        
        # Build reverse dependency graph
        for dep in dependencies:
            self.reverse_dependency_graph[dep].append(component)
        
        # Initialize component status
        self.component_status[component] = 'not_initialized'
    
    def register_component_factory(self, component_name: str, factory: Callable[[], Any]):
        """
        Register a factory function for creating a component.
        
        Args:
            component_name: Name of the component
            factory: Factory function that creates the component
        """
        self.component_factories[component_name] = factory
        logger.debug(f"📝 Registered factory for component: {component_name}")
    
    def detect_circular_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Detect circular dependencies in the dependency graph.
        
        Returns:
            Tuple of (has_circular_deps, circular_path)
        """
        visited = set()
        rec_stack = set()
        circular_path = []
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found circular dependency
                cycle_start = path.index(node)
                circular_path.extend(path[cycle_start:] + [node])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.dependency_graph.get(node, []):
                if dfs(neighbor, path):
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        # Check all components
        for component in self.dependency_graph:
            if component not in visited:
                if dfs(component, []):
                    return True, circular_path
        
        return False, []
    
    def get_initialization_order(self) -> List[str]:
        """
        Get optimal initialization order using topological sort.
        
        Returns:
            List of components in initialization order
        """
        # Check for circular dependencies first
        has_circular, circular_path = self.detect_circular_dependencies()
        if has_circular:
            logger.error(f"❌ Circular dependency detected: {' -> '.join(circular_path)}")
            raise ValueError(f"Circular dependency detected: {' -> '.join(circular_path)}")
        
        # Perform topological sort
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for component in self.dependency_graph:
            for dependency in self.dependency_graph[component]:
                in_degree[dependency] += 1
        
        # Initialize queue with components having no dependencies
        queue = deque([comp for comp in self.dependency_graph if in_degree[comp] == 0])
        initialization_order = []
        
        while queue:
            current = queue.popleft()
            initialization_order.append(current)
            
            # Update in-degrees of dependent components
            for dependent in self.reverse_dependency_graph.get(current, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Verify all components are included
        if len(initialization_order) != len(self.dependency_graph):
            missing = set(self.dependency_graph.keys()) - set(initialization_order)
            logger.error(f"❌ Failed to resolve all dependencies. Missing: {missing}")
            raise ValueError(f"Failed to resolve dependencies for: {missing}")
        
        self.initialization_order = initialization_order
        logger.info(f"📋 Initialization order resolved: {' -> '.join(initialization_order)}")
        
        return initialization_order
    
    def resolve_component(self, component_name: str) -> Optional[Any]:
        """
        Resolve a component and its dependencies.
        
        Args:
            component_name: Name of the component to resolve
            
        Returns:
            Initialized component instance or None if failed
        """
        start_time = time.time()
        
        try:
            # Check if already initialized
            if component_name in self.initialized_components:
                return self.initialized_components[component_name]
            
            # Check if component is being initialized (prevent infinite recursion)
            if self.component_status.get(component_name) == 'initializing':
                logger.error(f"❌ Circular dependency detected during initialization: {component_name}")
                return None
            
            # Mark as initializing
            self.component_status[component_name] = 'initializing'
            
            # Resolve dependencies first
            dependencies = self.dependency_graph.get(component_name, [])
            resolved_dependencies = {}
            
            for dep_name in dependencies:
                dep_component = self.resolve_component(dep_name)
                if dep_component is None:
                    logger.error(f"❌ Failed to resolve dependency '{dep_name}' for '{component_name}'")
                    self.component_status[component_name] = 'failed'
                    return None
                resolved_dependencies[dep_name] = dep_component
            
            # Initialize the component
            component = self._initialize_component(component_name, resolved_dependencies)
            
            if component is not None:
                # Store in registry
                self.initialized_components[component_name] = component
                self.component_status[component_name] = 'initialized'
                
                resolution_time = time.time() - start_time
                self.resolution_times[component_name] = resolution_time
                
                logger.info(f"✅ Component '{component_name}' resolved in {resolution_time:.3f}s")
                return component
            else:
                self.component_status[component_name] = 'failed'
                logger.error(f"❌ Failed to initialize component: {component_name}")
                return None
                
        except Exception as e:
            self.component_status[component_name] = 'failed'
            logger.error(f"❌ Component resolution failed for '{component_name}': {e}")
            return None
    
    def _initialize_component(self, component_name: str, dependencies: Dict[str, Any]) -> Optional[Any]:
        """Initialize a component using its factory function."""
        try:
            # Check if factory is registered
            if component_name in self.component_factories:
                factory = self.component_factories[component_name]
                return factory()
            
            # Use default initialization for known components
            return self._default_component_initialization(component_name)
            
        except Exception as e:
            logger.error(f"Component initialization failed for '{component_name}': {e}")
            return None
    
    def _default_component_initialization(self, component_name: str) -> Optional[Any]:
        """Default initialization for known components."""
        try:
            if component_name == 'enhanced_prediction_engine':
                from enhanced_prediction_engine import \
                    get_enhanced_prediction_engine
                return get_enhanced_prediction_engine()
            
            elif component_name == 'adaptive_ensemble':
                from models.ensemble.adaptive_voting import \
                    get_adaptive_ensemble
                return get_adaptive_ensemble()
            
            elif component_name == 'dynamic_trainer':
                from models.realtime.dynamic_trainer import get_dynamic_trainer
                return get_dynamic_trainer()
            
            elif component_name == 'unified_config':
                from utils.unified_config_manager import get_unified_config
                return get_unified_config()
            
            elif component_name == 'production_memory_optimizer':
                from utils.production_memory_optimizer import \
                    get_production_memory_optimizer
                return get_production_memory_optimizer()
            
            elif component_name == 'data_loader':
                from dashboard.data_loader import DashboardDataLoader
                return DashboardDataLoader()

            elif component_name == 'Interface':
                from dashboard.components.unified_design_system import \
                    get_unified_design_system
                return get_unified_design_system()

            elif component_name == 'enhanced_team_data_manager':
                from utils.enhanced_team_data_manager import \
                    get_enhanced_team_data_manager
                return get_enhanced_team_data_manager()

            elif component_name == 'enhanced_prediction_display':
                from dashboard.components.enhanced_prediction_display import \
                    get_enhanced_prediction_display

                # This component requires the design system
                design_system = self.resolve_component('Interface')
                return get_enhanced_prediction_display(design_system)

            else:
                logger.warning(f"⚠️ No default initialization for component: {component_name}")
                return None
                
        except Exception as e:
            logger.error(f"Default initialization failed for '{component_name}': {e}")
            return None
    
    def resolve_all_components(self) -> Dict[str, Any]:
        """
        Resolve all components in optimal order.
        
        Returns:
            Dictionary of resolved components
        """
        logger.info("🔗 Resolving all components...")
        start_time = time.time()
        
        try:
            # Get initialization order
            initialization_order = self.get_initialization_order()
            
            resolved_components = {}
            failed_components = []
            
            # Resolve components in order
            for component_name in initialization_order:
                component = self.resolve_component(component_name)
                if component is not None:
                    resolved_components[component_name] = component
                else:
                    failed_components.append(component_name)
            
            total_time = time.time() - start_time
            
            logger.info(f"🔗 Component resolution completed in {total_time:.2f}s")
            logger.info(f"✅ Resolved: {len(resolved_components)}, ❌ Failed: {len(failed_components)}")
            
            if failed_components:
                logger.warning(f"⚠️ Failed components: {failed_components}")
            
            return resolved_components
            
        except Exception as e:
            logger.error(f"❌ Component resolution failed: {e}")
            return {}
    
    def get_dependency_status(self) -> Dict[str, Any]:
        """Get comprehensive dependency status."""
        has_circular, circular_path = self.detect_circular_dependencies()
        
        return {
            'has_circular_dependencies': has_circular,
            'circular_path': circular_path,
            'component_status': self.component_status.copy(),
            'initialization_order': self.initialization_order.copy(),
            'resolution_times': self.resolution_times.copy(),
            'total_components': len(self.dependency_graph),
            'initialized_components': len(self.initialized_components),
            'timestamp': datetime.now().isoformat()
        }

# Global dependency resolver instance
_dependency_resolver_instance = None

def get_dependency_resolver() -> DependencyResolver:
    """Get global dependency resolver instance."""
    global _dependency_resolver_instance
    if _dependency_resolver_instance is None:
        _dependency_resolver_instance = DependencyResolver()
    return _dependency_resolver_instance

def resolve_component(component_name: str) -> Optional[Any]:
    """Convenient function to resolve a component."""
    resolver = get_dependency_resolver()
    return resolver.resolve_component(component_name)

def resolve_all_components() -> Dict[str, Any]:
    """Convenient function to resolve all components."""
    resolver = get_dependency_resolver()
    return resolver.resolve_all_components()
