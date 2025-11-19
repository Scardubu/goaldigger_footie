#!/usr/bin/env python3
"""
Unified Component Base Classes
Component Architecture Enhancement - Phase 3A Technical Debt Resolution

This module provides unified base classes for all GoalDiggers dashboard components,
establishing a consistent architecture and reducing code duplication across 40+ components.

Key Features:
- Abstract base classes for all component types
- Standardized initialization and configuration patterns
- Unified error handling and logging
- Performance tracking and optimization
- Feature flag integration
- Memory management and cleanup
"""

import logging
import time
import abc
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Component type enumeration."""
    UI_ELEMENT = "ui_element"
    PREDICTION = "prediction"
    ANALYTICS = "analytics"
    VISUALIZATION = "visualization"
    SYSTEM = "system"
    INTEGRATION = "integration"
    ENHANCEMENT = "enhancement"

class ComponentPriority(Enum):
    """Component loading priority."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class ComponentConfig:
    """Base configuration for all components."""
    component_id: str
    component_type: ComponentType
    priority: ComponentPriority = ComponentPriority.MEDIUM
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    performance_tracking: bool = True
    error_handling: bool = True
    memory_management: bool = True
    key_prefix: str = "component"

class UnifiedComponentBase(abc.ABC):
    """
    Abstract base class for all GoalDiggers dashboard components.
    Provides standardized functionality and patterns.
    """
    
    def __init__(self, config: ComponentConfig):
        """Initialize unified component base."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.component_id}")
        
        # Performance tracking
        self.start_time = time.time()
        self.render_count = 0
        self.performance_metrics = {
            'initialization_time': 0.0,
            'render_times': [],
            'error_count': 0,
            'memory_usage': 0.0
        }
        
        # Component state
        self.is_initialized = False
        self.is_rendered = False
        self.last_render_time = 0.0
        
        # Feature flags
        self.feature_flags = config.feature_flags
        
        # Initialize component
        self._initialize_component()
        
        self.logger.info(f"🚀 {config.component_id} component initialized")
    
    def _initialize_component(self):
        """Initialize component-specific functionality."""
        try:
            start_time = time.time()
            
            # Call component-specific initialization
            self._component_init()
            
            # Track initialization time
            init_time = time.time() - start_time
            self.performance_metrics['initialization_time'] = init_time
            
            self.is_initialized = True
            
            if self.config.performance_tracking:
                self.logger.debug(f"Component {self.config.component_id} initialized in {init_time:.3f}s")
                
        except Exception as e:
            self._handle_error(f"Component initialization failed: {e}")
    
    @abc.abstractmethod
    def _component_init(self):
        """Component-specific initialization logic."""
        pass
    
    @abc.abstractmethod
    def render(self, **kwargs) -> Any:
        """Render the component."""
        pass
    
    def safe_render(self, **kwargs) -> Any:
        """Safely render component with error handling."""
        if not self.is_initialized:
            self._handle_error("Component not initialized")
            return None
        
        try:
            start_time = time.time()
            
            # Call component render method
            result = self.render(**kwargs)
            
            # Track performance
            render_time = time.time() - start_time
            self.performance_metrics['render_times'].append(render_time)
            self.last_render_time = render_time
            self.render_count += 1
            self.is_rendered = True
            
            if self.config.performance_tracking:
                self.logger.debug(f"Component {self.config.component_id} rendered in {render_time:.3f}s")
            
            return result
            
        except Exception as e:
            self._handle_error(f"Component render failed: {e}")
            return self._render_error_fallback()
    
    def _render_error_fallback(self) -> Any:
        """Render error fallback UI."""
        if hasattr(st, 'error'):
            st.error(f"⚠️ Component {self.config.component_id} unavailable")
        return None
    
    def _handle_error(self, error_message: str):
        """Handle component errors consistently."""
        self.performance_metrics['error_count'] += 1
        
        if self.config.error_handling:
            self.logger.error(f"❌ {self.config.component_id}: {error_message}")
        else:
            raise RuntimeError(error_message)
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature flag is enabled."""
        return self.feature_flags.get(feature_name, False)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get component performance metrics."""
        avg_render_time = (
            sum(self.performance_metrics['render_times']) / len(self.performance_metrics['render_times'])
            if self.performance_metrics['render_times'] else 0.0
        )
        
        return {
            'component_id': self.config.component_id,
            'component_type': self.config.component_type.value,
            'initialization_time': self.performance_metrics['initialization_time'],
            'render_count': self.render_count,
            'average_render_time': avg_render_time,
            'last_render_time': self.last_render_time,
            'error_count': self.performance_metrics['error_count'],
            'uptime': time.time() - self.start_time,
            'is_healthy': self.performance_metrics['error_count'] == 0
        }
    
    def cleanup(self):
        """Clean up component resources."""
        if self.config.memory_management:
            # Clear performance metrics
            self.performance_metrics['render_times'] = []
            
            # Reset state
            self.is_rendered = False
            
            self.logger.debug(f"Component {self.config.component_id} cleaned up")

class UIComponentBase(UnifiedComponentBase):
    """Base class for UI components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize UI component."""
        config.component_type = ComponentType.UI_ELEMENT
        super().__init__(config)
    
    def apply_styling(self, custom_css: str = None):
        """Apply component styling."""
        try:
            if custom_css and hasattr(st, 'markdown'):
                st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
        except Exception as e:
            self._handle_error(f"Styling application failed: {e}")

class PredictionComponentBase(UnifiedComponentBase):
    """Base class for prediction components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize prediction component."""
        config.component_type = ComponentType.PREDICTION
        super().__init__(config)
    
    def validate_prediction_data(self, data: Dict[str, Any]) -> bool:
        """Validate prediction data structure."""
        required_fields = ['home_team', 'away_team', 'predictions']
        return all(field in data for field in required_fields)

class AnalyticsComponentBase(UnifiedComponentBase):
    """Base class for analytics components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize analytics component."""
        config.component_type = ComponentType.ANALYTICS
        super().__init__(config)
    
    def process_analytics_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process analytics data with standardized format."""
        return {
            'timestamp': time.time(),
            'component_id': self.config.component_id,
            'data': data,
            'metrics': self.get_performance_metrics()
        }

class VisualizationComponentBase(UnifiedComponentBase):
    """Base class for visualization components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize visualization component."""
        config.component_type = ComponentType.VISUALIZATION
        super().__init__(config)
    
    def create_chart_config(self, chart_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized chart configuration."""
        return {
            'type': chart_type,
            'data': data,
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'plugins': {
                    'legend': {'display': True},
                    'title': {'display': True, 'text': f"{self.config.component_id} Chart"}
                }
            }
        }

class SystemComponentBase(UnifiedComponentBase):
    """Base class for system components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize system component."""
        config.component_type = ComponentType.SYSTEM
        super().__init__(config)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system component status."""
        return {
            'component_id': self.config.component_id,
            'status': 'healthy' if self.performance_metrics['error_count'] == 0 else 'degraded',
            'uptime': time.time() - self.start_time,
            'performance': self.get_performance_metrics()
        }

class IntegrationComponentBase(UnifiedComponentBase):
    """Base class for integration components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize integration component."""
        config.component_type = ComponentType.INTEGRATION
        super().__init__(config)
    
    def validate_integration(self) -> bool:
        """Validate integration component connectivity."""
        return self.is_initialized and self.performance_metrics['error_count'] == 0

class EnhancementComponentBase(UnifiedComponentBase):
    """Base class for enhancement components."""
    
    def __init__(self, config: ComponentConfig):
        """Initialize enhancement component."""
        config.component_type = ComponentType.ENHANCEMENT
        super().__init__(config)
    
    def apply_enhancement(self, target_component: UnifiedComponentBase) -> bool:
        """Apply enhancement to target component."""
        try:
            # Enhancement logic would be implemented by subclasses
            return True
        except Exception as e:
            self._handle_error(f"Enhancement application failed: {e}")
            return False

# Factory functions for creating component configurations
def create_ui_config(component_id: str, **kwargs) -> ComponentConfig:
    """Create UI component configuration."""
    return ComponentConfig(
        component_id=component_id,
        component_type=ComponentType.UI_ELEMENT,
        **kwargs
    )

def create_prediction_config(component_id: str, **kwargs) -> ComponentConfig:
    """Create prediction component configuration."""
    return ComponentConfig(
        component_id=component_id,
        component_type=ComponentType.PREDICTION,
        priority=ComponentPriority.HIGH,
        **kwargs
    )

def create_analytics_config(component_id: str, **kwargs) -> ComponentConfig:
    """Create analytics component configuration."""
    return ComponentConfig(
        component_id=component_id,
        component_type=ComponentType.ANALYTICS,
        **kwargs
    )

def create_visualization_config(component_id: str, **kwargs) -> ComponentConfig:
    """Create visualization component configuration."""
    return ComponentConfig(
        component_id=component_id,
        component_type=ComponentType.VISUALIZATION,
        **kwargs
    )

def create_system_config(component_id: str, **kwargs) -> ComponentConfig:
    """Create system component configuration."""
    return ComponentConfig(
        component_id=component_id,
        component_type=ComponentType.SYSTEM,
        priority=ComponentPriority.CRITICAL,
        **kwargs
    )

def create_integration_config(component_id: str, **kwargs) -> ComponentConfig:
    """Create integration component configuration."""
    return ComponentConfig(
        component_id=component_id,
        component_type=ComponentType.INTEGRATION,
        **kwargs
    )

def create_enhancement_config(component_id: str, **kwargs) -> ComponentConfig:
    """Create enhancement component configuration."""
    return ComponentConfig(
        component_id=component_id,
        component_type=ComponentType.ENHANCEMENT,
        priority=ComponentPriority.LOW,
        **kwargs
    )
