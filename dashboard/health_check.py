"""
GoalDiggers Dashboard Health Check

This module provides tools to check the health of the dashboard components
and ensure all systems are operational.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

class DashboardHealthCheck:
    """
    Dashboard health check system for monitoring component status
    and diagnosing issues in the GoalDiggers platform.
    """
    
    def __init__(self):
        """Initialize the health check system."""
        self.results = {}
        self.warnings = []
        self.errors = []
        self.start_time = time.time()
    
    def check_pwa_implementation(self) -> bool:
        """Check if PWA implementation is working correctly."""
        try:
            from dashboard.components.pwa_implementation import \
                PWAImplementation
            pwa = PWAImplementation()
            
            # Verify key methods exist
            required_methods = ['configure_page', 'render_pwa_interface']
            for method in required_methods:
                if not hasattr(pwa, method):
                    self.warnings.append(f"PWA implementation missing method: {method}")
                    return False
            
            self.results['pwa_implementation'] = "Operational"
            return True
        except Exception as e:
            self.errors.append(f"PWA implementation check failed: {e}")
            self.results['pwa_implementation'] = f"Error: {type(e).__name__}"
            return False
    
    def check_ui_elements(self) -> bool:
        """Check if UI elements are working correctly."""
        try:
            from dashboard.components.ui_elements import render_banner

            # Verify banner rendering
            banner_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     "dashboard", "static", "images", "GoalDiggers_banner.png")
            banner_exists = os.path.exists(banner_path)
            
            if not banner_exists:
                self.warnings.append(f"Banner image not found at: {banner_path}")
            
            self.results['ui_elements'] = "Operational" + (" (with fallback)" if not banner_exists else "")
            return True
        except Exception as e:
            self.errors.append(f"UI elements check failed: {e}")
            self.results['ui_elements'] = f"Error: {type(e).__name__}"
            return False
    
    def check_mobile_components(self) -> bool:
        """Check if mobile components are working correctly."""
        try:
            from dashboard.mobile.mobile_detection import (detect_mobile,
                                                           get_device_info)

            # Verify mobile detection returns expected type
            device_info = get_device_info()
            if not isinstance(device_info, dict):
                self.warnings.append(f"Mobile detection returned unexpected type: {type(device_info)}")
                return False
            
            self.results['mobile_components'] = "Operational"
            return True
        except Exception as e:
            self.errors.append(f"Mobile components check failed: {e}")
            self.results['mobile_components'] = f"Error: {type(e).__name__}"
            return False
    
    def check_data_loader(self) -> bool:
        """Check if data loader is working correctly."""
        try:
            from dashboard.data_loader import create_minimal_loader
            
            loader = create_minimal_loader()
            leagues = loader.get_available_leagues()
            
            if not leagues:
                self.warnings.append("Data loader returned no leagues")
            
            self.results['data_loader'] = f"Operational ({len(leagues) if leagues else 0} leagues)"
            return True
        except Exception as e:
            self.errors.append(f"Data loader check failed: {e}")
            self.results['data_loader'] = f"Error: {type(e).__name__}"
            return False
    
    def check_static_resources(self) -> bool:
        """Check if static resources are properly set up."""
        required_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "static"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "static", "images")
        ]
        
        required_files = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "static", "images", "GoalDiggers_banner.png"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "dashboard", "static", "images", "README.md")
        ]
        
        # Check directories
        missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
        if missing_dirs:
            for d in missing_dirs:
                self.errors.append(f"Missing required directory: {d}")
            self.results['static_resources'] = f"Error: Missing {len(missing_dirs)} directories"
            return False
        
        # Check files
        missing_files = [f for f in required_files if not os.path.isfile(f)]
        if missing_files:
            for f in missing_files:
                self.warnings.append(f"Missing recommended file: {f}")
            self.results['static_resources'] = f"Operational (missing {len(missing_files)} files)"
            return True
        
        self.results['static_resources'] = "Operational"
        return True
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        self.check_pwa_implementation()
        self.check_ui_elements()
        self.check_mobile_components()
        self.check_data_loader()
        self.check_static_resources()
        
        execution_time = time.time() - self.start_time
        
        return {
            'status': 'healthy' if not self.errors else 'degraded' if not self.warnings else 'warning',
            'execution_time': execution_time,
            'components': self.results,
            'warnings': self.warnings,
            'errors': self.errors,
            'timestamp': time.time()
        }

def render_health_dashboard():
    """Render interactive health check dashboard."""
    st.set_page_config(
        page_title="System Health | GoalDiggers",
        page_icon="🩺",
        layout="wide"
    )
    
    # Custom styling
    st.markdown("""
    <style>
    .healthy { color: #2ca02c; }
    .warning { color: #ff9800; }
    .error { color: #e53935; }
    
    .component-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: white;
    }
    
    .header-container {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 2rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 style="margin: 0;">System Health</h1>
        <p style="margin: 0.5rem 0 0 0;">Comprehensive platform status and diagnostics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run health checks
    if st.button("Run Health Check", use_container_width=True):
        with st.spinner("Running comprehensive health checks..."):
            health_check = DashboardHealthCheck()
            results = health_check.run_all_checks()
            
            # Overall status
            status_color = "healthy" if results['status'] == 'healthy' else "warning" if results['status'] == 'degraded' else "error"
            status_icon = "✅" if results['status'] == 'healthy' else "⚠️" if results['status'] == 'degraded' else "❌"
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <h2 style="margin: 0;">Overall Status: <span class="{status_color}">{status_icon} {results['status'].upper()}</span></h2>
                <div style="margin-left: auto; font-size: 0.9rem;">
                    Execution time: {results['execution_time']:.2f}s
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Component status
            st.subheader("Component Status")
            
            cols = st.columns(3)
            for i, (component, status) in enumerate(results['components'].items()):
                with cols[i % 3]:
                    status_class = "error" if status.startswith("Error") else "warning" if "missing" in status or "with fallback" in status else "healthy"
                    status_icon = "❌" if status.startswith("Error") else "⚠️" if "missing" in status or "with fallback" in status else "✅"
                    
                    st.markdown(f"""
                    <div class="component-card">
                        <h4>{component.replace('_', ' ').title()}</h4>
                        <p class="{status_class}">{status_icon} {status}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Warnings and errors
            if results['warnings']:
                st.subheader("⚠️ Warnings")
                for warning in results['warnings']:
                    st.warning(warning)
            
            if results['errors']:
                st.subheader("❌ Errors")
                for error in results['errors']:
                    st.error(error)
            
            if not results['warnings'] and not results['errors']:
                st.success("All systems operational! No warnings or errors detected.")
    else:
        # Initial state
        st.info("Click 'Run Health Check' to verify system status")
    
    # System information
    with st.expander("System Information"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Environment")
            st.text(f"Python: {sys.version.split()[0]}")
            st.text(f"Platform: {sys.platform}")
        
        with col2:
            st.markdown("### Versions")
            try:
                import streamlit
                st.text(f"Streamlit: {streamlit.__version__}")
            except ImportError:
                st.text("Streamlit: Not detected")
            
            try:
                from config.app_config import AppConfig
                app_config = AppConfig()
                st.text(f"GoalDiggers: {getattr(app_config, 'APP_VERSION', 'Unknown')}")
            except Exception:
                st.text("GoalDiggers: Version not detected")

if __name__ == "__main__":
    render_health_dashboard()
