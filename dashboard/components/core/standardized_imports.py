#!/usr/bin/env python3
"""
Standardized Import System
Component Architecture Enhancement - Phase 3A Technical Debt Resolution

This module provides standardized import patterns and migration utilities
for the consolidated component architecture, ensuring clean and maintainable imports.

Key Features:
- Standardized import paths for all component types
- Legacy component import compatibility
- Automatic component discovery and registration
- Import optimization and dependency management
- Migration utilities for existing code
"""

import logging
import importlib
import sys
from typing import Dict, Any, List, Optional, Type, Union
from pathlib import Path

from .unified_component_base import (
    UnifiedComponentBase, ComponentConfig, ComponentType,
    UIComponentBase, PredictionComponentBase, AnalyticsComponentBase,
    VisualizationComponentBase, SystemComponentBase, IntegrationComponentBase,
    EnhancementComponentBase
)
from .component_registry import get_component_registry, register_component

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Component type mapping for standardized imports
COMPONENT_TYPE_MAPPING = {
    ComponentType.UI_ELEMENT: UIComponentBase,
    ComponentType.PREDICTION: PredictionComponentBase,
    ComponentType.ANALYTICS: AnalyticsComponentBase,
    ComponentType.VISUALIZATION: VisualizationComponentBase,
    ComponentType.SYSTEM: SystemComponentBase,
    ComponentType.INTEGRATION: IntegrationComponentBase,
    ComponentType.ENHANCEMENT: EnhancementComponentBase
}

# Legacy component mapping for backward compatibility
LEGACY_COMPONENT_MAPPING = {
    # UI Components
    'consistent_styling': 'dashboard.components.core.ui.styling_component',
    'ui_elements': 'dashboard.components.core.ui.elements_component',
    'ui_enhancements': 'dashboard.components.core.ui.enhancements_component',
    'visual_consistency_system': 'dashboard.components.core.ui.consistency_component',
    
    # Prediction Components
    'betting_insights': 'dashboard.components.core.prediction.insights_component',
    'enhanced_prediction_display': 'dashboard.components.core.prediction.display_component',
    'value_betting': 'dashboard.components.core.prediction.value_component',
    'value_betting_analyzer': 'dashboard.components.core.prediction.analyzer_component',
    
    # Analytics Components
    'betting_insights_dashboard': 'dashboard.components.core.analytics.insights_dashboard',
    'production_betting_dashboard': 'dashboard.components.core.analytics.production_dashboard',
    'advanced_analytics_dashboard': 'dashboard.components.core.analytics.advanced_dashboard',
    
    # System Components
    'system_status': 'dashboard.components.core.system.status_component',
    'realtime_system': 'dashboard.components.core.system.realtime_component',
    'data_integrity_monitor': 'dashboard.components.core.system.integrity_component',
    
    # Integration Components
    'phase2_integration': 'dashboard.components.core.integration.phase2_component',
    'personalization_integration': 'dashboard.components.core.integration.personalization_component',
    'pwa_implementation': 'dashboard.components.core.integration.pwa_component',
    
    # Enhancement Components
    'phase3_advanced_features': 'dashboard.components.core.enhancement.advanced_features',
    'universal_achievement_system': 'dashboard.components.core.enhancement.achievement_component',
    'universal_help_system': 'dashboard.components.core.enhancement.help_component',
    'universal_navigation_system': 'dashboard.components.core.enhancement.navigation_component',
    'universal_workflow_manager': 'dashboard.components.core.enhancement.workflow_component'
}

class StandardizedImportManager:
    """
    Manages standardized imports and component registration
    for the consolidated component architecture.
    """
    
    def __init__(self):
        """Initialize standardized import manager."""
        self.logger = logging.getLogger(__name__)
        self.registry = get_component_registry()
        
        # Import tracking
        self.imported_components: Dict[str, Type[UnifiedComponentBase]] = {}
        self.legacy_imports: Dict[str, str] = {}
        self.failed_imports: List[str] = []
        
        self.logger.info("🚀 Standardized import manager initialized")
    
    def import_component(
        self,
        component_name: str,
        component_type: ComponentType = None,
        legacy_fallback: bool = True
    ) -> Optional[Type[UnifiedComponentBase]]:
        """Import component with standardized path resolution."""
        try:
            # Check if already imported
            if component_name in self.imported_components:
                return self.imported_components[component_name]
            
            # Try standardized import path
            component_class = self._try_standardized_import(component_name, component_type)
            
            if component_class is None and legacy_fallback:
                # Try legacy import path
                component_class = self._try_legacy_import(component_name)
            
            if component_class is not None:
                self.imported_components[component_name] = component_class
                self.logger.debug(f"✅ Imported component: {component_name}")
                return component_class
            else:
                self.failed_imports.append(component_name)
                self.logger.warning(f"⚠️ Failed to import component: {component_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Import error for {component_name}: {e}")
            self.failed_imports.append(component_name)
            return None
    
    def _try_standardized_import(
        self,
        component_name: str,
        component_type: ComponentType = None
    ) -> Optional[Type[UnifiedComponentBase]]:
        """Try importing component using standardized path."""
        if component_type is None:
            # Try to infer component type from name
            component_type = self._infer_component_type(component_name)
        
        # Build standardized import path
        type_path = component_type.value if component_type else 'ui'
        module_path = f"dashboard.components.core.{type_path}.{component_name}"
        
        try:
            module = importlib.import_module(module_path)
            
            # Look for component class in module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, UnifiedComponentBase) and 
                    attr != UnifiedComponentBase):
                    return attr
            
            return None
            
        except ImportError:
            return None
    
    def _try_legacy_import(self, component_name: str) -> Optional[Type[UnifiedComponentBase]]:
        """Try importing component using legacy path."""
        if component_name in LEGACY_COMPONENT_MAPPING:
            legacy_path = LEGACY_COMPONENT_MAPPING[component_name]
            self.legacy_imports[component_name] = legacy_path
            
            try:
                module = importlib.import_module(legacy_path)
                
                # Look for component class in module
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, UnifiedComponentBase) and 
                        attr != UnifiedComponentBase):
                        return attr
                
                return None
                
            except ImportError:
                return None
        
        # Try direct import from components directory
        try:
            module_path = f"dashboard.components.{component_name}"
            module = importlib.import_module(module_path)
            
            # Look for component class in module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, 'render') and
                    attr_name != 'UnifiedComponentBase'):
                    # Wrap legacy component in unified base
                    return self._wrap_legacy_component(attr, component_name)
            
            return None
            
        except ImportError:
            return None
    
    def _infer_component_type(self, component_name: str) -> ComponentType:
        """Infer component type from component name."""
        name_lower = component_name.lower()
        
        if any(keyword in name_lower for keyword in ['prediction', 'betting', 'odds', 'value']):
            return ComponentType.PREDICTION
        elif any(keyword in name_lower for keyword in ['analytics', 'dashboard', 'insights']):
            return ComponentType.ANALYTICS
        elif any(keyword in name_lower for keyword in ['chart', 'plot', 'visualization', 'graph']):
            return ComponentType.VISUALIZATION
        elif any(keyword in name_lower for keyword in ['system', 'status', 'health', 'monitor']):
            return ComponentType.SYSTEM
        elif any(keyword in name_lower for keyword in ['integration', 'phase', 'pwa', 'personalization']):
            return ComponentType.INTEGRATION
        elif any(keyword in name_lower for keyword in ['enhancement', 'universal', 'advanced']):
            return ComponentType.ENHANCEMENT
        else:
            return ComponentType.UI_ELEMENT
    
    def _wrap_legacy_component(self, legacy_class: Type, component_name: str) -> Type[UnifiedComponentBase]:
        """Wrap legacy component in unified base class."""
        component_type = self._infer_component_type(component_name)
        base_class = COMPONENT_TYPE_MAPPING[component_type]
        
        class WrappedLegacyComponent(base_class):
            """Wrapped legacy component for compatibility."""
            
            def __init__(self, config: ComponentConfig = None):
                if config is None:
                    config = ComponentConfig(
                        component_id=component_name,
                        component_type=component_type
                    )
                super().__init__(config)
                self.legacy_instance = legacy_class()
            
            def _component_init(self):
                """Initialize wrapped legacy component."""
                pass
            
            def render(self, **kwargs):
                """Render using legacy component."""
                if hasattr(self.legacy_instance, 'render'):
                    return self.legacy_instance.render(**kwargs)
                elif hasattr(self.legacy_instance, 'display'):
                    return self.legacy_instance.display(**kwargs)
                elif hasattr(self.legacy_instance, 'show'):
                    return self.legacy_instance.show(**kwargs)
                else:
                    self.logger.warning(f"⚠️ No render method found for legacy component {component_name}")
                    return None
        
        return WrappedLegacyComponent
    
    def register_all_components(self) -> Dict[str, Any]:
        """Discover and register all available components."""
        self.logger.info("🔍 Discovering and registering all components")
        
        registration_results = {
            'registered': 0,
            'failed': 0,
            'legacy_wrapped': 0,
            'components': []
        }
        
        # Discover components in standardized structure
        components_dir = Path("dashboard/components")
        
        for component_file in components_dir.rglob("*.py"):
            if component_file.name.startswith("__") or "__pycache__" in str(component_file):
                continue
            
            component_name = component_file.stem
            
            # Skip core infrastructure files
            if component_name in ['unified_component_base', 'component_registry', 'component_consolidator', 'standardized_imports']:
                continue
            
            # Try to import and register component
            component_class = self.import_component(component_name)
            
            if component_class:
                # Create configuration
                component_type = self._infer_component_type(component_name)
                config = ComponentConfig(
                    component_id=component_name,
                    component_type=component_type
                )
                
                # Register component
                success = register_component(component_name, component_class, config)
                
                if success:
                    registration_results['registered'] += 1
                    registration_results['components'].append({
                        'name': component_name,
                        'type': component_type.value,
                        'legacy_wrapped': component_name in self.legacy_imports
                    })
                    
                    if component_name in self.legacy_imports:
                        registration_results['legacy_wrapped'] += 1
                else:
                    registration_results['failed'] += 1
            else:
                registration_results['failed'] += 1
        
        self.logger.info(f"✅ Registration complete: {registration_results['registered']} registered, {registration_results['failed']} failed")
        return registration_results
    
    def get_import_statistics(self) -> Dict[str, Any]:
        """Get import and registration statistics."""
        return {
            'imported_components': len(self.imported_components),
            'legacy_imports': len(self.legacy_imports),
            'failed_imports': len(self.failed_imports),
            'component_types': {
                comp_type.value: len([
                    name for name, cls in self.imported_components.items()
                    if self._infer_component_type(name) == comp_type
                ])
                for comp_type in ComponentType
            },
            'legacy_components': list(self.legacy_imports.keys()),
            'failed_components': self.failed_imports
        }
    
    def generate_migration_guide(self) -> str:
        """Generate migration guide for standardized imports."""
        guide = """
# Component Import Migration Guide

## Standardized Import Patterns

### UI Components
```python
from dashboard.components.core.ui import StylingComponent, ElementsComponent
```

### Prediction Components
```python
from dashboard.components.core.prediction import InsightsComponent, DisplayComponent
```

### Analytics Components
```python
from dashboard.components.core.analytics import AdvancedDashboard, ProductionDashboard
```

### System Components
```python
from dashboard.components.core.system import StatusComponent, RealtimeComponent
```

### Integration Components
```python
from dashboard.components.core.integration import PersonalizationComponent, PWAComponent
```

### Enhancement Components
```python
from dashboard.components.core.enhancement import AchievementComponent, NavigationComponent
```

## Legacy Import Compatibility

The following legacy imports are automatically mapped:

"""
        
        for legacy_name, new_path in LEGACY_COMPONENT_MAPPING.items():
            guide += f"- `{legacy_name}` → `{new_path}`\n"
        
        guide += """

## Component Registry Usage

```python
from dashboard.components.core import get_component_registry, get_component

# Get component from registry
registry = get_component_registry()
component = registry.get_component('component_name')

# Or use convenience function
component = get_component('component_name')
```

## Migration Steps

1. Update import statements to use standardized paths
2. Replace direct component instantiation with registry usage
3. Update component configurations to use ComponentConfig
4. Test all functionality with new import patterns
5. Remove legacy import statements

"""
        
        return guide

# Global import manager instance
_import_manager: Optional[StandardizedImportManager] = None

def get_import_manager() -> StandardizedImportManager:
    """Get global import manager instance."""
    global _import_manager
    
    if _import_manager is None:
        _import_manager = StandardizedImportManager()
    
    return _import_manager

def import_component(component_name: str, component_type: ComponentType = None) -> Optional[Type[UnifiedComponentBase]]:
    """Import component using standardized import manager."""
    manager = get_import_manager()
    return manager.import_component(component_name, component_type)

def register_all_components() -> Dict[str, Any]:
    """Register all available components."""
    manager = get_import_manager()
    return manager.register_all_components()

def get_import_statistics() -> Dict[str, Any]:
    """Get import statistics."""
    manager = get_import_manager()
    return manager.get_import_statistics()

# Convenience imports for common component types
def import_ui_component(component_name: str) -> Optional[Type[UIComponentBase]]:
    """Import UI component."""
    return import_component(component_name, ComponentType.UI_ELEMENT)

def import_prediction_component(component_name: str) -> Optional[Type[PredictionComponentBase]]:
    """Import prediction component."""
    return import_component(component_name, ComponentType.PREDICTION)

def import_analytics_component(component_name: str) -> Optional[Type[AnalyticsComponentBase]]:
    """Import analytics component."""
    return import_component(component_name, ComponentType.ANALYTICS)

def import_visualization_component(component_name: str) -> Optional[Type[VisualizationComponentBase]]:
    """Import visualization component."""
    return import_component(component_name, ComponentType.VISUALIZATION)

def import_system_component(component_name: str) -> Optional[Type[SystemComponentBase]]:
    """Import system component."""
    return import_component(component_name, ComponentType.SYSTEM)

def import_integration_component(component_name: str) -> Optional[Type[IntegrationComponentBase]]:
    """Import integration component."""
    return import_component(component_name, ComponentType.INTEGRATION)

def import_enhancement_component(component_name: str) -> Optional[Type[EnhancementComponentBase]]:
    """Import enhancement component."""
    return import_component(component_name, ComponentType.ENHANCEMENT)
