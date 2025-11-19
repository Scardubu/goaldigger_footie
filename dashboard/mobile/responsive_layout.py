"""
Responsive layout components for GoalDiggers mobile optimization.

Provides responsive grid systems, mobile-first layouts, and adaptive
component rendering based on device characteristics.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple

import streamlit as st

from .mobile_detection import detect_mobile, get_device_info, is_tablet

logger = logging.getLogger(__name__)


class BreakPoint(Enum):
    """Responsive design breakpoints."""
    XS = 480   # Extra small devices
    SM = 768   # Small devices  
    MD = 1024  # Medium devices
    LG = 1200  # Large devices
    XL = 1440  # Extra large devices


class LayoutMode(Enum):
    """Layout rendering modes."""
    MOBILE_FIRST = "mobile_first"
    DESKTOP_FIRST = "desktop_first"
    ADAPTIVE = "adaptive"


class ResponsiveLayout:
    """Responsive layout manager for GoalDiggers platform."""
    
    def __init__(self, mode: LayoutMode = LayoutMode.ADAPTIVE):
        """Initialize responsive layout manager."""
        self.mode = mode
        self.device_info = get_device_info()
        self.is_mobile = self.device_info['is_mobile']
        self.is_tablet = self.device_info['is_tablet']
        self.is_desktop = self.device_info['is_desktop']
        
        # Layout configuration
        self.grid_config = self._get_grid_config()
        
        logger.debug(f"ResponsiveLayout initialized: {self.device_info}")
    
    def _get_grid_config(self) -> Dict[str, int]:
        """Get grid configuration based on device type."""
        if self.is_mobile:
            return {
                'columns': 1,
                'max_columns': 2,
                'spacing': 'small',
                'container_padding': '8px'
            }
        elif self.is_tablet:
            return {
                'columns': 2,
                'max_columns': 3,
                'spacing': 'medium',
                'container_padding': '16px'
            }
        else:  # desktop
            return {
                'columns': 3,
                'max_columns': 4,
                'spacing': 'large',
                'container_padding': '24px'
            }
    
    def create_responsive_columns(self, content_items: int, max_cols: Optional[int] = None) -> List[any]:
        """Create responsive column layout based on content and device."""
        if max_cols is None:
            max_cols = self.grid_config['max_columns']
        
        # Determine optimal number of columns
        if self.is_mobile:
            cols = min(content_items, 1)
        elif self.is_tablet:
            cols = min(content_items, 2)
        else:
            cols = min(content_items, max_cols)
        
        # Create Streamlit columns
        return st.columns(cols)
    
    def render_responsive_grid(self, items: List[dict], render_callback):
        """Render items in a responsive grid layout."""
        try:
            if not items:
                return
            
            # Get column configuration
            cols_per_row = self.grid_config['columns']
            
            # Process items in chunks
            for i in range(0, len(items), cols_per_row):
                chunk = items[i:i + cols_per_row]
                columns = st.columns(len(chunk))
                
                for j, item in enumerate(chunk):
                    with columns[j]:
                        render_callback(item, self.device_info)
                        
        except Exception as e:
            logger.error(f"Error rendering responsive grid: {e}")
            # Fallback to simple list
            for item in items:
                render_callback(item, self.device_info)
    
    def apply_mobile_optimizations(self):
        """Apply mobile-specific CSS optimizations."""
        if not self.is_mobile:
            return
        
        mobile_css = """
        <style>
        /* GoalDiggers Mobile Optimizations */
        .stApp {
            padding: 0.5rem !important;
        }
        
        /* Touch-friendly buttons */
        .stButton > button {
            min-height: 44px !important;
            min-width: 44px !important;
            padding: 12px 16px !important;
            font-size: 16px !important;
            border-radius: 8px !important;
            touch-action: manipulation !important;
        }
        
        /* Optimized select boxes */
        .stSelectbox > div > div > div {
            min-height: 44px !important;
            font-size: 16px !important;
        }
        
        /* Improved form inputs */
        .stTextInput > div > div > input {
            min-height: 44px !important;
            font-size: 16px !important;
            padding: 12px !important;
        }
        
        /* Card spacing */
        [data-testid="metric-container"] {
            margin-bottom: 12px !important;
            padding: 12px !important;
        }
        
        /* Responsive columns */
        .stColumn {
            padding: 0 4px !important;
        }
        
        /* Hide sidebar on mobile when collapsed */
        @media (max-width: 768px) {
            .css-1d391kg {
                width: 0 !important;
            }
        }
        
        /* Optimize tables for mobile */
        .stDataFrame {
            font-size: 14px !important;
        }
        
        /* Better mobile headers */
        h1, h2, h3 {
            font-size: clamp(1.2rem, 4vw, 2rem) !important;
            margin-bottom: 0.5rem !important;
        }
        </style>
        """
        
        st.markdown(mobile_css, unsafe_allow_html=True)
    
    def render_adaptive_container(self, content_func, mobile_func=None, tablet_func=None):
        """Render content with device-specific adaptations."""
        try:
            if self.is_mobile and mobile_func:
                mobile_func()
            elif self.is_tablet and tablet_func:
                tablet_func()
            else:
                content_func()
                
        except Exception as e:
            logger.error(f"Error in adaptive container: {e}")
            # Fallback to default content
            content_func()
    
    def get_optimal_chart_config(self) -> Dict[str, any]:
        """Get optimal chart configuration for current device."""
        if self.is_mobile:
            return {
                'height': 300,
                'width': 'container',
                'font_size': 12,
                'legend_position': 'bottom',
                'responsive': True,
                'toolbar': False
            }
        elif self.is_tablet:
            return {
                'height': 400,
                'width': 'container', 
                'font_size': 14,
                'legend_position': 'right',
                'responsive': True,
                'toolbar': True
            }
        else:  # desktop
            return {
                'height': 500,
                'width': 'container',
                'font_size': 16,
                'legend_position': 'right',
                'responsive': True,
                'toolbar': True
            }
    
    def create_responsive_metrics_layout(self, metrics: List[Dict]) -> None:
        """Create responsive layout for metrics display."""
        if not metrics:
            return
        
        # Determine layout based on device and number of metrics
        if self.is_mobile:
            # Stack metrics vertically on mobile
            for metric in metrics:
                st.metric(
                    label=metric['label'],
                    value=metric['value'],
                    delta=metric.get('delta'),
                    help=metric.get('help')
                )
        else:
            # Use columns for tablet/desktop
            cols_per_row = 2 if self.is_tablet else 4
            
            for i in range(0, len(metrics), cols_per_row):
                chunk = metrics[i:i + cols_per_row]
                columns = st.columns(len(chunk))
                
                for j, metric in enumerate(chunk):
                    with columns[j]:
                        st.metric(
                            label=metric['label'],
                            value=metric['value'],
                            delta=metric.get('delta'),
                            help=metric.get('help')
                        )
    
    def inject_responsive_meta_tags(self):
        """Inject responsive design meta tags."""
        meta_tags = """
        <script>
        (function() {
            // Ensure viewport meta tag exists
            let viewport = document.querySelector('meta[name="viewport"]');
            if (!viewport) {
                viewport = document.createElement('meta');
                viewport.name = 'viewport';
                viewport.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no';
                document.head.appendChild(viewport);
            }
            
            // Add mobile-web-app-capable for PWA
            let webAppCapable = document.createElement('meta');
            webAppCapable.name = 'mobile-web-app-capable';
            webAppCapable.content = 'yes';
            document.head.appendChild(webAppCapable);
            
            // Add apple-mobile-web-app-capable for iOS
            let appleWebAppCapable = document.createElement('meta');
            appleWebAppCapable.name = 'apple-mobile-web-app-capable'; 
            appleWebAppCapable.content = 'yes';
            document.head.appendChild(appleWebAppCapable);
            
            // Add apple-mobile-web-app-status-bar-style
            let statusBarStyle = document.createElement('meta');
            statusBarStyle.name = 'apple-mobile-web-app-status-bar-style';
            statusBarStyle.content = 'default';
            document.head.appendChild(statusBarStyle);
            
        })();
        </script>
        """
        
        st.markdown(meta_tags, unsafe_allow_html=True)
    
    def get_responsive_spacing(self) -> Dict[str, str]:
        """Get responsive spacing values."""
        if self.is_mobile:
            return {
                'container': '8px',
                'section': '16px',
                'element': '8px',
                'component': '12px'
            }
        elif self.is_tablet:
            return {
                'container': '16px',
                'section': '24px', 
                'element': '12px',
                'component': '16px'
            }
        else:  # desktop
            return {
                'container': '24px',
                'section': '32px',
                'element': '16px', 
                'component': '20px'
            }


# Utility functions for responsive design
def get_responsive_columns(item_count: int, device_type: str = None) -> int:
    """Get optimal number of columns for given item count and device."""
    if device_type is None:
        device_info = get_device_info()
        device_type = 'mobile' if device_info['is_mobile'] else 'tablet' if device_info['is_tablet'] else 'desktop'
    
    column_map = {
        'mobile': min(item_count, 1),
        'tablet': min(item_count, 2),
        'desktop': min(item_count, 4)
    }
    
    return column_map.get(device_type, 2)


def apply_responsive_styling():
    """Apply global responsive styling."""
    responsive_css = """
    <style>
    /* Global responsive utilities */
    .responsive-container {
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 16px;
    }
    
    .responsive-grid {
        display: grid;
        gap: 16px;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }
    
    .responsive-flex {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
    }
    
    .responsive-flex > * {
        flex: 1 1 300px;
    }
    
    /* Mobile-first media queries */
    @media (max-width: 768px) {
        .responsive-container {
            padding: 0 8px;
        }
        
        .responsive-grid {
            grid-template-columns: 1fr;
            gap: 12px;
        }
        
        .responsive-flex {
            flex-direction: column;
            gap: 12px;
        }
        
        .hide-on-mobile {
            display: none !important;
        }
    }
    
    @media (min-width: 769px) and (max-width: 1024px) {
        .responsive-grid {
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }
        
        .hide-on-tablet {
            display: none !important;
        }
    }
    
    @media (min-width: 1025px) {
        .hide-on-desktop {
            display: none !important;
        }
    }
    
    /* Touch optimizations */
    @media (hover: none) and (pointer: coarse) {
        button, .stButton > button {
            min-height: 44px !important;
            min-width: 44px !important;
        }
        
        select, input {
            font-size: 16px !important;
        }
    }
    </style>
    """
    
    st.markdown(responsive_css, unsafe_allow_html=True)


# Export main classes and functions
__all__ = [
    'ResponsiveLayout',
    'BreakPoint',
    'LayoutMode',
    'get_responsive_columns',
    'apply_responsive_styling'
]
