#!/usr/bin/env python3
"""
Component Registry for GoalDiggers Platform
Centralized access to all dashboard components
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class ComponentRegistry:
    """Registry for all GoalDiggers dashboard components"""
    
    def __init__(self):
        self._components = {}
        self._initialized = False
    
    def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        
        try:
            # Gamification Integration
            from dashboard.components.gamification_integration import \
                get_gamification_integration
            self._components['gamification'] = get_gamification_integration()
            logger.info("✅ Gamification component loaded")
        except Exception as e:
            logger.warning(f"⚠️ Gamification component failed to load: {e}")
        
        try:
            # Progressive Disclosure
            from dashboard.components.progressive_disclosure import \
                get_progressive_disclosure
            self._components['progressive_disclosure'] = get_progressive_disclosure()
            logger.info("✅ Progressive disclosure component loaded")
        except Exception as e:
            logger.warning(f"⚠️ Progressive disclosure component failed to load: {e}")
        
        try:
            # Real-time System
            from dashboard.components.realtime_system import realtime_system
            self._components['realtime'] = realtime_system
            logger.info("✅ Real-time system component loaded")
        except Exception as e:
            logger.warning(f"⚠️ Real-time system component failed to load: {e}")
        
        self._initialized = True
        logger.info(f"🎯 Component registry initialized with {len(self._components)} components")
    
    def get_component(self, component_name: str) -> Optional[Any]:
        """Get a component by name"""
        if not self._initialized:
            self.initialize()
        
        return self._components.get(component_name)
    
    def get_gamification(self):
        """Get gamification integration component"""
        return self.get_component('gamification')
    
    def get_progressive_disclosure(self):
        """Get progressive disclosure component"""
        return self.get_component('progressive_disclosure')
    
    def get_realtime_system(self):
        """Get real-time system component"""
        return self.get_component('realtime')
    
    def list_components(self) -> list:
        """List all available components"""
        if not self._initialized:
            self.initialize()
        return list(self._components.keys())
    
    def get_status(self) -> dict:
        """Get status of all components"""
        if not self._initialized:
            self.initialize()
        
        status = {
            'initialized': self._initialized,
            'component_count': len(self._components),
            'components': {}
        }
        
        for name, component in self._components.items():
            try:
                # Basic health check
                status['components'][name] = {
                    'loaded': component is not None,
                    'type': type(component).__name__
                }
                
                # Try to get component-specific status if available
                if hasattr(component, 'get_status'):
                    status['components'][name]['status'] = component.get_status()
                
            except Exception as e:
                status['components'][name] = {
                    'loaded': False,
                    'error': str(e)
                }
        
        return status

# Global registry instance
_registry = ComponentRegistry()

def get_registry():
    """Get the global component registry"""
    return _registry

def get_component(component_name: str):
    """Convenience function to get a component"""
    return _registry.get_component(component_name)

def get_gamification():
    """Convenience function to get gamification component"""
    return _registry.get_gamification()

def get_progressive_disclosure():
    """Convenience function to get progressive disclosure component"""
    return _registry.get_progressive_disclosure()

def get_realtime_system():
    """Convenience function to get realtime system component"""
    return _registry.get_realtime_system()
