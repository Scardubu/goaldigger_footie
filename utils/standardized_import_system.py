"""
Standardized Import System for GoalDiggers Platform
Provides centralized import management to eliminate circular dependencies and ensure consistent module loading.
"""
import importlib
import logging
import sys
from typing import Any, Dict, List, Optional, Callable, Type
from pathlib import Path

logger = logging.getLogger(__name__)

class ImportManager:
    """
    Centralized import manager to handle module loading and prevent circular dependencies.
    """

    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}
        self._import_cache: Dict[str, Any] = {}
        self._dependency_graph: Dict[str, List[str]] = {}
        self._circular_dependencies: List[List[str]] = []

    def register_module(self, module_name: str, module_path: Optional[str] = None):
        """
        Register a module with the import manager.

        Args:
            module_name: Name of the module to register
            module_path: Optional path to the module file
        """
        if module_name not in self._loaded_modules:
            self._loaded_modules[module_name] = None
            self._dependency_graph[module_name] = []

    def add_dependency(self, module_name: str, dependency: str):
        """
        Add a dependency relationship between modules.

        Args:
            module_name: Module that depends on dependency
            dependency: Module that is depended upon
        """
        if module_name not in self._dependency_graph:
            self._dependency_graph[module_name] = []

        if dependency not in self._dependency_graph[module_name]:
            self._dependency_graph[module_name].append(dependency)

    def detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies in the module graph.

        Returns:
            List of circular dependency chains
        """
        def dfs(node: str, visited: set, path: List[str]) -> Optional[List[str]]:
            if node in visited:
                if node in path:
                    return path[path.index(node):] + [node]
                return None

            visited.add(node)
            path.append(node)

            for dependency in self._dependency_graph.get(node, []):
                cycle = dfs(dependency, visited, path)
                if cycle:
                    return cycle

            path.pop()
            return None

        visited = set()
        for module in self._dependency_graph:
            if module not in visited:
                cycle = dfs(module, visited, [])
                if cycle:
                    self._circular_dependencies.append(cycle)

        return self._circular_dependencies

    def safe_import(self, module_name: str, fallback: Any = None) -> Any:
        """
        Safely import a module with fallback handling.

        Args:
            module_name: Name of the module to import
            fallback: Fallback value if import fails

        Returns:
            Imported module or fallback value
        """
        if module_name in self._import_cache:
            return self._import_cache[module_name]

        try:
            module = importlib.import_module(module_name)
            self._import_cache[module_name] = module
            self._loaded_modules[module_name] = module
            logger.debug(f"Successfully imported module: {module_name}")
            return module
        except ImportError as e:
            logger.warning(f"Failed to import module {module_name}: {e}")
            self._import_cache[module_name] = fallback
            return fallback
        except Exception as e:
            logger.error(f"Unexpected error importing module {module_name}: {e}")
            self._import_cache[module_name] = fallback
            return fallback

    def lazy_import(self, module_name: str) -> Callable[[], Any]:
        """
        Create a lazy import function for a module.

        Args:
            module_name: Name of the module to lazy import

        Returns:
            Function that imports the module when called
        """
        def _import():
            if module_name not in self._import_cache:
                self._import_cache[module_name] = self.safe_import(module_name)
            return self._import_cache[module_name]

        return _import

    def get_module_info(self) -> Dict[str, Any]:
        """
        Get information about loaded modules and dependencies.

        Returns:
            Dictionary with module information
        """
        return {
            "loaded_modules": list(self._loaded_modules.keys()),
            "cached_modules": list(self._import_cache.keys()),
            "dependency_graph": self._dependency_graph.copy(),
            "circular_dependencies": self._circular_dependencies.copy(),
            "total_modules": len(self._loaded_modules),
            "total_cached": len(self._import_cache)
        }

class ComponentRegistry:
    """
    Registry for dashboard components to prevent circular imports and provide centralized access.
    """

    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._component_factories: Dict[str, Callable] = {}
        self._import_manager = ImportManager()

    def register_component(self, name: str, component_class: Type, module_path: Optional[str] = None):
        """
        Register a component class with the registry.

        Args:
            name: Name of the component
            component_class: Component class
            module_path: Optional module path for lazy loading
        """
        self._components[name] = component_class
        if module_path:
            self._import_manager.register_module(module_path)

    def register_factory(self, name: str, factory: Callable):
        """
        Register a component factory function.

        Args:
            name: Name of the component
            factory: Factory function that returns the component
        """
        self._component_factories[name] = factory

    def get_component(self, name: str, *args, **kwargs) -> Any:
        """
        Get a component instance.

        Args:
            name: Name of the component
            *args: Positional arguments for component initialization
            **kwargs: Keyword arguments for component initialization

        Returns:
            Component instance
        """
        # Try factory first
        if name in self._component_factories:
            try:
                return self._component_factories[name](*args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to create component {name} from factory: {e}")

        # Try direct component
        if name in self._components:
            try:
                component_class = self._components[name]
                return component_class(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to create component {name}: {e}")

        # Try lazy import
        try:
            module = self._import_manager.safe_import(f"dashboard.components.{name}")
            if module and hasattr(module, name.title()):
                component_class = getattr(module, name.title())
                return component_class(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to lazy import component {name}: {e}")

        logger.error(f"Component {name} not found in registry")
        return None

    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(set(self._components.keys()) | set(self._component_factories.keys()))

class StandardizedImporter:
    """
    Standardized importer that provides consistent import patterns across the application.
    """

    def __init__(self):
        self._import_manager = ImportManager()
        self._component_registry = ComponentRegistry()

        # Register common modules
        self._register_common_modules()

    def _register_common_modules(self):
        """Register commonly used modules for safe importing."""
        common_modules = [
            "streamlit",
            "pandas",
            "numpy",
            "plotly",
            "sklearn",
            "xgboost",
            "lightgbm",
            "catboost",
            "sqlalchemy",
            "fastapi",
            "uvicorn",
            "requests",
            "aiohttp"
        ]

        for module in common_modules:
            self._import_manager.register_module(module)

    def import_ml_libraries(self) -> Dict[str, Any]:
        """
        Safely import ML libraries with fallbacks.

        Returns:
            Dictionary of imported ML libraries
        """
        ml_libs = {}

        # Core ML libraries
        ml_libs["sklearn"] = self._import_manager.safe_import("sklearn")
        ml_libs["xgboost"] = self._import_manager.safe_import("xgboost")
        ml_libs["lightgbm"] = self._import_manager.safe_import("lightgbm")
        ml_libs["catboost"] = self._import_manager.safe_import("catboost")

        # Data processing
        ml_libs["pandas"] = self._import_manager.safe_import("pandas")
        ml_libs["numpy"] = self._import_manager.safe_import("numpy")

        # Visualization
        ml_libs["plotly"] = self._import_manager.safe_import("plotly")
        ml_libs["matplotlib"] = self._import_manager.safe_import("matplotlib")

        return ml_libs

    def import_dashboard_components(self) -> Dict[str, Any]:
        """
        Safely import dashboard components with fallbacks.

        Returns:
            Dictionary of imported dashboard components
        """
        components = {}

        # Core dashboard components
        component_imports = {
            "streamlit": "streamlit",
            "enhanced_prediction_ui": "dashboard.components.enhanced_prediction_ui",
            "prediction_ui": "dashboard.components.prediction_ui",
            "ui_elements": "dashboard.components.ui_elements",
            "enhanced_styling": "dashboard.components.enhanced_styling",
            "unified_design_system": "dashboard.components.unified_design_system",
            "personalization_integration": "dashboard.components.personalization_integration",
            "achievement_system": "dashboard.components.achievement_system"
        }

        for name, module_path in component_imports.items():
            components[name] = self._import_manager.safe_import(module_path)

        return components

    def import_database_modules(self) -> Dict[str, Any]:
        """
        Safely import database-related modules.

        Returns:
            Dictionary of imported database modules
        """
        db_modules = {}

        db_imports = {
            "sqlalchemy": "sqlalchemy",
            "database_manager": "database.db_manager",
            "database_schema": "database.schema",
            "db_config": "database.config"
        }

        for name, module_path in db_imports.items():
            db_modules[name] = self._import_manager.safe_import(module_path)

        return db_modules

    def get_import_status(self) -> Dict[str, Any]:
        """
        Get status of all imports.

        Returns:
            Dictionary with import status information
        """
        return {
            "module_info": self._import_manager.get_module_info(),
            "components": self._component_registry.list_components(),
            "ml_libraries": self.import_ml_libraries(),
            "dashboard_components": self.import_dashboard_components(),
            "database_modules": self.import_database_modules()
        }

# Global instances
_import_manager = ImportManager()
_component_registry = ComponentRegistry()
_standardized_importer = StandardizedImporter()

def get_import_manager() -> ImportManager:
    """Get the global import manager instance."""
    return _import_manager

def get_component_registry() -> ComponentRegistry:
    """Get the global component registry instance."""
    return _component_registry

def get_standardized_importer() -> StandardizedImporter:
    """Get the global standardized importer instance."""
    return _standardized_importer

# Convenience functions for common imports
def safe_import(module_name: str, fallback: Any = None) -> Any:
    """Safely import a module with fallback."""
    return _import_manager.safe_import(module_name, fallback)

def lazy_import(module_name: str) -> Callable[[], Any]:
    """Create a lazy import function."""
    return _import_manager.lazy_import(module_name)

def get_component(name: str, *args, **kwargs) -> Any:
    """Get a component from the registry."""
    return _component_registry.get_component(name, *args, **kwargs)

# Initialize common components on import
def _initialize_common_components():
    """Initialize commonly used components."""
    try:
        # Register core dashboard components
        _component_registry.register_factory("enhanced_prediction_ui",
            lambda: safe_import("dashboard.components.enhanced_prediction_ui"))
        _component_registry.register_factory("prediction_ui",
            lambda: safe_import("dashboard.components.prediction_ui"))
        _component_registry.register_factory("ui_elements",
            lambda: safe_import("dashboard.components.ui_elements"))

        logger.info("Common components initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize common components: {e}")

_initialize_common_components()