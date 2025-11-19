#!/usr/bin/env python3
"""
Dashboard Router - Unified Dashboard Access Point
Phase 3A: Technical Debt Resolution - Backward Compatibility

This router provides backward compatibility for all existing dashboard variants
while routing to the unified dashboard implementation. Maintains all existing
URLs and entry points while consolidating the underlying implementation.

Legacy Dashboard Support:
- /premium → Premium UI Dashboard
- /integrated → Integrated Production Dashboard  
- /interactive → Interactive Cross-League Dashboard
- /optimized → Optimized Premium Dashboard
- /ultra_fast → Ultra Fast Premium Dashboard
- /classic → Classic Dashboard
- /fast_production → Fast Production Dashboard (Legacy)

Key Features:
- Seamless routing to unified dashboard with appropriate configuration
- URL parameter parsing for variant selection
- Graceful fallback to legacy dashboards if unified system fails
- Performance monitoring and error recovery
- Session state migration between variants
"""

import logging
import os
import sys
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

import streamlit as st

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DashboardRouter:
    """
    Router for unified dashboard system with backward compatibility.
    """
    
    def __init__(self):
        """Initialize dashboard router."""
        self.logger = logging.getLogger(__name__)
        
        # Legacy route mappings
        self.legacy_routes = {
            'premium': 'premium_ui',
            'integrated': 'integrated_production', 
            'interactive': 'interactive_cross_league',
            'optimized': 'optimized_premium',
            'ultra_fast': 'ultra_fast_premium',
            'classic': 'classic',
            'fast_production': 'fast_production'
        }
        
        # Default variant
        self.default_variant = 'premium_ui'
        
        self.logger.info("🚀 Dashboard router initialized")
    
    def route_dashboard(self) -> str:
        """
        Route to appropriate dashboard variant based on URL parameters or session state.
        Returns the variant name to load.
        """
        try:
            # Check URL parameters first
            variant = self._get_variant_from_url()
            
            if not variant:
                # Check session state
                variant = self._get_variant_from_session()
            
            if not variant:
                # Use default variant
                variant = self.default_variant
            
            # Validate variant
            variant = self._validate_variant(variant)
            
            # Store in session state for consistency
            st.session_state.dashboard_variant = variant
            
            self.logger.info(f"🎯 Routing to dashboard variant: {variant}")
            return variant
            
        except Exception as e:
            self.logger.error(f"Dashboard routing error: {e}")
            return self.default_variant
    
    def _get_variant_from_url(self) -> Optional[str]:
        """Extract dashboard variant from URL parameters."""
        try:
            # Get URL parameters from Streamlit
            query_params = st.query_params
            
            # Check for variant parameter
            if 'variant' in query_params:
                variant = query_params['variant'][0]
                return self._normalize_variant_name(variant)
            
            # Check for legacy route parameters
            if 'dashboard' in query_params:
                dashboard = query_params['dashboard'][0]
                return self.legacy_routes.get(dashboard, dashboard)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"URL parameter parsing error: {e}")
            return None
    
    def _get_variant_from_session(self) -> Optional[str]:
        """Get dashboard variant from session state."""
        return st.session_state.get('dashboard_variant', None)
    
    def _normalize_variant_name(self, variant: str) -> str:
        """Normalize variant name to standard format."""
        # Convert legacy names to standard names
        if variant in self.legacy_routes:
            return self.legacy_routes[variant]
        
        # Handle common variations
        variant_mappings = {
            'premium_ui_dashboard': 'premium_ui',
            'integrated_production_dashboard': 'integrated_production',
            'interactive_cross_league_dashboard': 'interactive_cross_league',
            'optimized_premium_dashboard': 'optimized_premium',
            'ultra_fast_premium_dashboard': 'ultra_fast_premium'
        }
        
        return variant_mappings.get(variant, variant)
    
    def _validate_variant(self, variant: str) -> str:
        """Validate variant name and return valid variant or default."""
        valid_variants = [
            'premium_ui',
            'integrated_production',
            'interactive_cross_league',
            'optimized_premium',
            'ultra_fast_premium',
            'classic',
            'fast_production'
        ]
        
        if variant in valid_variants:
            return variant
        else:
            self.logger.warning(f"Invalid variant '{variant}', using default")
            return self.default_variant
    
    def load_unified_dashboard(self, variant: str):
        """Load unified dashboard with specified variant configuration."""
        try:
            # Import unified dashboard
            from dashboard.unified_goaldiggers_dashboard import (
                DashboardConfig, DashboardVariant, UnifiedGoalDiggersDashboard)

            # Create configuration based on variant
            config = self._create_dashboard_config(variant)
            
            # Initialize and render unified dashboard
            dashboard = UnifiedGoalDiggersDashboard(config)
            dashboard.render_dashboard()
            
            self.logger.info(f"✅ Unified dashboard loaded successfully: {variant}")
            
        except Exception as e:
            self.logger.error(f"Unified dashboard loading failed: {e}")
            self._load_legacy_dashboard_fallback(variant)
    
    def _create_dashboard_config(self, variant: str):
        """Create dashboard configuration for specified variant."""
        from dashboard.unified_goaldiggers_dashboard import DashboardConfig
        
        config_mappings = {
            'premium_ui': DashboardConfig.get_premium_ui_config,
            'integrated_production': DashboardConfig.get_integrated_production_config,
            'interactive_cross_league': DashboardConfig.get_interactive_cross_league_config,
            'optimized_premium': DashboardConfig.get_optimized_premium_config
        }
        
        config_func = config_mappings.get(variant)
        if config_func:
            return config_func()
        else:
            # Default to premium UI config
            return DashboardConfig.get_premium_ui_config()
    
    def _load_legacy_dashboard_fallback(self, variant: str):
        """Load legacy dashboard as fallback if unified system fails."""
        self.logger.warning(f"Loading legacy dashboard fallback for: {variant}")
        
        try:
            if variant == 'premium_ui':
                self._load_premium_ui_legacy()
            elif variant == 'integrated_production':
                self._load_integrated_production_legacy()
            elif variant == 'interactive_cross_league':
                self._load_interactive_cross_league_legacy()
            elif variant == 'optimized_premium':
                self._load_optimized_premium_legacy()
            else:
                self._load_default_legacy()
                
        except Exception as e:
            self.logger.error(f"Legacy dashboard fallback failed: {e}")
            self._render_error_dashboard()
    
    def _load_premium_ui_legacy(self):
        """Load legacy premium UI dashboard."""
        try:
            from dashboard.enhanced_production_homepage import ProductionDashboardHomepage
            dashboard = ProductionDashboardProductionDashboardHomepage()
            dashboard.run()
        except ImportError:
            self._render_error_dashboard()
    
    def _load_integrated_production_legacy(self):
        """Load legacy integrated production dashboard."""
        try:
            from dashboard.integrated_production_dashboard import \
                IntegratedProductionDashboard
            dashboard = IntegratedProductionDashboardHomepage()
            dashboard.run()
        except ImportError:
            self._render_error_dashboard()
    
    def _load_interactive_cross_league_legacy(self):
        """Load legacy interactive cross-league dashboard."""
        try:
            from dashboard.interactive_cross_league_dashboard import \
                InteractiveCrossLeagueDashboard
            dashboard = ProductionDashboardHomepage()
            dashboard.render_dashboard()
        except ImportError:
            self._render_error_dashboard()
    
    def _load_optimized_premium_legacy(self):
        """Load legacy optimized premium dashboard."""
        try:
            from dashboard.optimized_premium_dashboard import \
                OptimizedPremiumDashboard
            dashboard = ProductionDashboardHomepage()
            dashboard.run()
        except ImportError:
            self._render_error_dashboard()
    
    def _load_default_legacy(self):
        """Load default legacy dashboard."""
        try:
            from dashboard.enhanced_production_homepage import ProductionDashboardHomepage
            dashboard = ProductionDashboardProductionDashboardHomepage()
            dashboard.run()
        except ImportError:
            self._render_error_dashboard()
    
    def _render_error_dashboard(self):
        """Render error dashboard when all else fails."""
        st.error("⚠️ Dashboard Loading Error")
        st.markdown("""
        ### 🚨 Dashboard System Error
        
        The dashboard system encountered an error and could not load any variant.
        
        **Possible Solutions:**
        1. Refresh the page
        2. Clear browser cache
        3. Check system logs for detailed error information
        4. Contact support if the issue persists
        
        **Error Details:**
        - Unified dashboard system failed to load
        - Legacy dashboard fallback failed
        - All dashboard variants unavailable
        """)
    
    def render_variant_selector(self):
        """Render variant selector for testing and debugging."""
        if st.sidebar.checkbox("🔧 Dashboard Variant Selector", key="variant_selector"):
            st.sidebar.markdown("### 🎛️ Select Dashboard Variant")
            
            variants = [
                ('premium_ui', '⚽ Premium UI'),
                ('integrated_production', '🎯 Integrated Production'),
                ('interactive_cross_league', '🎮 Interactive Cross-League'),
                ('optimized_premium', '⚡ Optimized Premium'),
                ('ultra_fast_premium', '🚀 Ultra Fast Premium'),
                ('classic', '📊 Classic'),
                ('fast_production', '🏃 Fast Production (Legacy)')
            ]
            
            current_variant = st.session_state.get('dashboard_variant', self.default_variant)
            
            selected_variant = st.sidebar.selectbox(
                "Choose Variant",
                options=[v[0] for v in variants],
                format_func=lambda x: next(v[1] for v in variants if v[0] == x),
                index=next((i for i, v in enumerate(variants) if v[0] == current_variant), 0)
            )
            
            if selected_variant != current_variant:
                st.session_state.dashboard_variant = selected_variant
                st.rerun()
    
    def render_debug_info(self):
        """Render debug information for dashboard routing."""
        if st.sidebar.checkbox("🔍 Debug Info", key="debug_info"):
            with st.sidebar.expander("🔧 Router Debug"):
                st.json({
                    'current_variant': st.session_state.get('dashboard_variant', 'None'),
                    'query_params': dict(st.query_params),
                    'session_state_keys': list(st.session_state.keys()),
                    'legacy_routes': self.legacy_routes
                })

def main():
    """Main entry point for dashboard router."""
    # Initialize router
    router = ProductionDashboardHomepage()
    
    # Add debug controls
    router.render_variant_selector()
    router.render_debug_info()
    
    # Route to appropriate dashboard
    variant = router.route_dashboard()
    
    # Load unified dashboard
    router.load_unified_dashboard(variant)

if __name__ == "__main__":
    main()
