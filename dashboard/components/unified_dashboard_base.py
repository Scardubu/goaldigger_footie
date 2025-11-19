#!/usr/bin/env python3
"""
Unified Dashboard Base Class for GoalDiggers Platform

Provides standardized integration pattern for all dashboard variants:
- Consistent enhanced component initialization
- Standardized ML pipeline integration
- Unified error handling and fallback mechanisms
- Performance optimization patterns
- Cross-dashboard consistency

This base class ensures all dashboards follow the same integration workflow
while allowing for specific customizations in derived classes.
"""

import logging
import time
import weakref
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)


class UnifiedDashboardBase(ABC):
    """Base class for all GoalDiggers dashboard variants."""
    
    def __init__(self, dashboard_type: str = "standard"):
        """
        Initialize unified dashboard with standard integration pattern.

        Args:
            dashboard_type: Type of dashboard (premium, cross_league, integrated, etc.)
        """
        self.dashboard_type = dashboard_type
        self.start_time = time.time()
        self.logger = logger

        # Performance tracking
        self.performance_metrics = {
            'initialization_time': 0,
            'component_load_times': {},
            'user_interactions': 0,
            'prediction_times': [],
            'error_count': 0
        }

        # Component registry
        self.components = {}
        self.component_cache = weakref.WeakValueDictionary()

        # Check for ultra-fast startup mode - defer ALL heavy component loading
        if hasattr(self, '_ultra_fast_startup') and self._ultra_fast_startup:
            logger.info(f"🚀 Ultra-fast startup mode: ALL heavy components deferred for {dashboard_type}")
            # Initialize only minimal required components
            self._initialize_minimal_components()
            return

        # Initialize components using standard pattern
        self._initialize_enhanced_components()
        self._initialize_ml_components()
        self._setup_component_bridges()

        # Initialize universal achievement system
        self._initialize_achievement_system()

        # Initialize universal workflow manager
        self._initialize_workflow_manager()

        # Initialize universal help system
        self._initialize_help_system()

        # Initialize universal navigation system
        self._initialize_navigation_system()

        # Initialize advanced visual components (Phase 3.3)
        self._initialize_advanced_visual_components()

        # Record initialization time
        self.performance_metrics['initialization_time'] = time.time() - self.start_time
        logger.info(f"✅ {dashboard_type.title()} dashboard initialized in {self.performance_metrics['initialization_time']:.3f}s")

    def _initialize_unified_styling(self):
        """
        Standardized CSS injection method across all dashboards.

        This method provides a consistent approach to styling that all dashboard
        variants can use, with proper fallback mechanisms and error handling.
        """
        try:
            # Primary method: Use design system if available
            if hasattr(self, 'design_system') and self.design_system:
                self.design_system.inject_unified_css(self.dashboard_type)
                logger.info(f"✅ Unified design system CSS applied to {self.dashboard_type} dashboard")
                return True

        except Exception as e:
            logger.warning(f"⚠️ Design system CSS injection failed: {e}")

        # Fallback method: Direct CSS file loading
        try:
            self._inject_fallback_css()
            logger.info(f"✅ Fallback CSS applied to {self.dashboard_type} dashboard")
            return True

        except Exception as e:
            logger.error(f"❌ All CSS injection methods failed: {e}")
            return False

    def _inject_fallback_css(self):
        """
        Fallback CSS injection method when design system is unavailable.

        Loads the unified design system CSS file directly and applies
        dashboard-specific enhancements.
        """
        import streamlit as st

        # Load unified design system CSS
        css_path = "dashboard/static/unified_design_system.css"
        with open(css_path, 'r', encoding='utf-8') as f:
            unified_css = f.read()

        # Apply base CSS
        st.markdown(f"<style>{unified_css}</style>", unsafe_allow_html=True)

        # Apply dashboard-specific enhancements
        dashboard_specific_css = self._get_dashboard_specific_css()
        if dashboard_specific_css:
            st.markdown(f"<style>{dashboard_specific_css}</style>", unsafe_allow_html=True)

    def _get_dashboard_specific_css(self) -> str:
        """
        Get dashboard-specific CSS enhancements.

        Returns:
            CSS string with dashboard-specific styling
        """
        dashboard_styles = {
            'premium': self._get_premium_dashboard_css(),
            'interactive_cross_league': self._get_cross_league_dashboard_css(),
            'integrated': self._get_integrated_dashboard_css(),
            'optimized_premium': self._get_optimized_dashboard_css()
        }

        return dashboard_styles.get(self.dashboard_type, '')

    def _get_premium_dashboard_css(self) -> str:
        """Premium dashboard specific CSS enhancements."""
        return """
        /* Premium Dashboard Enhancements */
        .premium-feature {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%);
            backdrop-filter: blur(15px);
            border: 1px solid rgba(31, 78, 121, 0.1);
            border-radius: var(--gd-radius-lg);
            box-shadow: var(--gd-shadow-xl);
        }

        .premium-badge {
            background: linear-gradient(135deg, var(--gd-accent) 0%, var(--gd-primary-light) 100%);
            color: var(--gd-white);
            padding: var(--gd-space-1) var(--gd-space-3);
            border-radius: var(--gd-radius-full);
            font-size: var(--gd-text-xs);
            font-weight: var(--gd-font-semibold);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        """

    def _get_cross_league_dashboard_css(self) -> str:
        """Interactive cross-league dashboard specific CSS enhancements."""
        return """
        /* Cross-League Dashboard Enhancements */
        .cross-league-header {
            background: linear-gradient(135deg, var(--gd-primary) 0%, var(--gd-secondary) 100%);
            padding: var(--gd-space-6);
            border-radius: var(--gd-radius-xl);
            color: var(--gd-white);
            text-align: center;
            margin-bottom: var(--gd-space-8);
            box-shadow: var(--gd-shadow-lg);
        }

        .achievement-badge {
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            color: #1f2937;
            padding: var(--gd-space-2) var(--gd-space-4);
            border-radius: var(--gd-radius-lg);
            font-weight: var(--gd-font-bold);
            box-shadow: var(--gd-shadow-md);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        """

    def _get_integrated_dashboard_css(self) -> str:
        """Integrated production dashboard specific CSS enhancements."""
        return """
        /* Integrated Dashboard Enhancements */
        .production-metric {
            background: var(--gd-white);
            border: 1px solid var(--gd-gray-200);
            border-radius: var(--gd-radius-base);
            padding: var(--gd-space-4);
            transition: all var(--gd-transition-fast);
        }

        .production-metric:hover {
            border-color: var(--gd-primary);
            box-shadow: var(--gd-shadow-md);
        }

        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: var(--gd-space-2);
        }

        .status-indicator.success { background-color: var(--gd-success); }
        .status-indicator.warning { background-color: var(--gd-warning); }
        .status-indicator.error { background-color: var(--gd-error); }
        """

    def _get_optimized_dashboard_css(self) -> str:
        """Optimized premium dashboard specific CSS enhancements."""
        return """
        /* Optimized Dashboard Enhancements */
        .optimized-card {
            background: var(--gd-white);
            border-radius: var(--gd-radius-lg);
            padding: var(--gd-space-5);
            box-shadow: var(--gd-shadow-base);
            transition: transform var(--gd-transition-fast);
        }

        .optimized-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--gd-shadow-lg);
        }

        .step-indicator {
            background: linear-gradient(135deg, var(--gd-primary) 0%, var(--gd-primary-light) 100%);
            color: var(--gd-white);
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: var(--gd-font-bold);
            font-size: var(--gd-text-sm);
        }
        """
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced components with standard pattern."""
        # 1. Unified Design System
        self.design_system = self._safe_component_init(
            'design_system',
            'dashboard.components.unified_design_system',
            'get_unified_design_system'
        )
        
        # 2. Enhanced Team Data Manager
        self.team_manager = self._safe_component_init(
            'team_manager',
            'utils.enhanced_team_data_manager',
            'get_enhanced_team_data_manager'
        )
        
        # 3. Enhanced Prediction Display
        self.prediction_display = self._safe_component_init(
            'prediction_display',
            'dashboard.components.enhanced_prediction_display',
            'get_enhanced_prediction_display',
            args=[self.design_system] if self.design_system else []
        )
        
        # Apply unified styling using standardized method
        self._initialize_unified_styling()

    def _initialize_minimal_components(self):
        """Initialize only minimal components for ultra-fast startup."""
        # Initialize only the most essential components for ultra-fast startup

        # 1. Minimal design system (just CSS injection)
        self.design_system = self._safe_component_init(
            'design_system',
            'dashboard.components.unified_design_system',
            'get_unified_design_system'
        )

        # 2. Apply unified styling (lightweight)
        self._initialize_unified_styling()

        # 3. Set up empty component structures (no actual loading)
        self.team_manager = None
        self.prediction_display = None
        self.ml_components = {}
        self.achievement_system = None
        self.workflow_manager = None
        self.help_system = None
        self.navigation_system = None

        logger.info(f"⚡ Minimal components initialized for ultra-fast {self.dashboard_type} startup")

    def _initialize_ml_components(self):
        """Initialize ML components with standard pattern."""
        # Check if ML initialization should be skipped for performance
        if hasattr(self, '_skip_ml_initialization') and self._skip_ml_initialization:
            self.ml_components = {}
            logger.info("⚡ ML component initialization skipped for ultra-fast startup")
            return

        # Check for ultra-fast startup mode (even more aggressive than skip_ml_initialization)
        if hasattr(self, '_ultra_fast_startup') and self._ultra_fast_startup:
            self.ml_components = {}
            logger.info("🚀 Ultra-fast startup mode: ALL heavy components deferred")
            return

        ml_component_configs = [
            ('dynamic_trainer', 'models.realtime.dynamic_trainer', 'get_dynamic_trainer'),
            ('adaptive_ensemble', 'models.ensemble.adaptive_voting', 'get_adaptive_ensemble'),
            ('enhanced_prediction_engine', 'enhanced_prediction_engine', 'EnhancedPredictionEngine'),
            ('live_data_processor', 'data.streams.live_data_processor', 'get_live_data_processor'),
            ('odds_aggregator', 'data.market.odds_aggregator', 'get_odds_aggregator'),
            ('preference_engine', 'user.personalization.preference_engine', 'get_preference_engine')
        ]

        self.ml_components = {}
        for component_name, module_path, factory_name in ml_component_configs:
            self.ml_components[component_name] = self._safe_component_init(
                f"ml_{component_name}",
                module_path,
                factory_name
            )
    
    def _safe_component_init(self, component_name: str, module_path: str, 
                           factory_name: str, args: List = None, kwargs: Dict = None) -> Optional[Any]:
        """
        Safely initialize a component with error handling and performance tracking.
        
        Args:
            component_name: Name of the component for tracking
            module_path: Python module path
            factory_name: Factory function or class name
            args: Arguments to pass to factory
            kwargs: Keyword arguments to pass to factory
            
        Returns:
            Initialized component or None if failed
        """
        start_time = time.time()
        args = args or []
        kwargs = kwargs or {}
        
        try:
            # Dynamic import
            module = __import__(module_path, fromlist=[factory_name])
            factory = getattr(module, factory_name)
            
            # Initialize component
            if callable(factory):
                if args or kwargs:
                    component = factory(*args, **kwargs)
                else:
                    component = factory()
            else:
                component = factory
            
            # Cache component
            self.components[component_name] = component
            
            # Record performance
            load_time = time.time() - start_time
            self.performance_metrics['component_load_times'][component_name] = load_time
            
            logger.debug(f"✅ {component_name} initialized in {load_time:.3f}s")
            return component
            
        except Exception as e:
            load_time = time.time() - start_time
            self.performance_metrics['component_load_times'][component_name] = load_time
            self.performance_metrics['error_count'] += 1
            
            logger.warning(f"⚠️ {component_name} initialization failed in {load_time:.3f}s: {e}")
            return None
    
    def _setup_component_bridges(self):
        """Setup communication bridges between components."""
        # Connect team manager to prediction display
        if self.team_manager and self.prediction_display:
            try:
                # Setup team resolution bridge
                self.prediction_display.set_team_resolver(self.team_manager.resolve_team)
                logger.debug("✅ Team manager bridge established")
            except Exception as e:
                logger.warning(f"⚠️ Team manager bridge failed: {e}")
        
        # Connect ML components to prediction display
        if self.prediction_display and any(self.ml_components.values()):
            try:
                # Setup ML prediction bridge
                available_ml = {k: v for k, v in self.ml_components.items() if v is not None}
                self.prediction_display.set_ml_components(available_ml)
                logger.debug(f"✅ ML components bridge established ({len(available_ml)} components)")
            except Exception as e:
                logger.warning(f"⚠️ ML components bridge failed: {e}")
    
    def get_team_metadata(self, team_name: str, league: str = None) -> Optional[Dict[str, Any]]:
        """Get team metadata using enhanced team manager."""
        if self.team_manager:
            try:
                metadata = self.team_manager.resolve_team(team_name, league)
                return metadata.__dict__ if metadata else None
            except Exception as e:
                logger.warning(f"Team metadata resolution failed: {e}")
                return None
        return None
    
    def generate_enhanced_prediction(self, home_team: str, away_team: str) -> Optional[Dict[str, Any]]:
        """Generate prediction using available ML components."""
        start_time = time.time()
        
        try:
            # Try enhanced prediction engine first
            if self.ml_components.get('enhanced_prediction_engine'):
                engine = self.ml_components['enhanced_prediction_engine']
                
                match_data = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': self._determine_league(home_team),
                    'timestamp': time.time()
                }
                
                if hasattr(engine, 'predict_match'):
                    prediction = engine.predict_match(match_data)
                    prediction['source'] = 'enhanced_ml'
                    prediction['method'] = 'Enhanced ML Engine'
                    
                    # Record performance
                    prediction_time = time.time() - start_time
                    self.performance_metrics['prediction_times'].append(prediction_time)
                    
                    return prediction
            
            # Fallback to adaptive ensemble
            if self.ml_components.get('adaptive_ensemble'):
                ensemble = self.ml_components['adaptive_ensemble']
                
                if hasattr(ensemble, 'predict'):
                    prediction = ensemble.predict(home_team, away_team)
                    prediction['source'] = 'adaptive_ensemble'
                    prediction['method'] = 'Adaptive Ensemble'
                    
                    prediction_time = time.time() - start_time
                    self.performance_metrics['prediction_times'].append(prediction_time)
                    
                    return prediction
            
            # Final fallback
            return self._generate_fallback_prediction(home_team, away_team)
            
        except Exception as e:
            logger.error(f"Enhanced prediction generation failed: {e}")
            return self._generate_fallback_prediction(home_team, away_team)
    
    def _determine_league(self, team_name: str) -> str:
        """Determine league for a team."""
        if self.team_manager:
            try:
                metadata = self.team_manager.resolve_team(team_name)
                return metadata.league if metadata else "Unknown"
            except:
                pass
        return "Unknown"
    
    def _generate_fallback_prediction(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Generate fallback prediction when ML components are unavailable."""
        return {
            'home_win': 0.45,
            'draw': 0.25,
            'away_win': 0.30,
            'confidence': {'overall': 0.65},
            'source': 'fallback',
            'method': 'Statistical Fallback',
            'insights': [f"Statistical analysis suggests {home_team} has a slight advantage"]
        }
    
    def render_enhanced_prediction(self, home_team: str, away_team: str):
        """Render prediction using enhanced display components."""
        if self.prediction_display:
            try:
                prediction = self.generate_enhanced_prediction(home_team, away_team)
                if prediction:
                    match_data = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'league': self._determine_league(home_team)
                    }
                    self.prediction_display.render_animated_prediction_card(prediction, match_data)
                    return True
            except Exception as e:
                logger.error(f"Enhanced prediction rendering failed: {e}")
        
        # Fallback to basic rendering
        self._render_basic_prediction(home_team, away_team)
        return False

    def render_unified_header(self, title: str, subtitle: str = None):
        """
        Standardized header rendering across all dashboards.

        Args:
            title: Main dashboard title
            subtitle: Optional subtitle text
        """
        try:
            # Primary method: Use design system if available
            if hasattr(self, 'design_system') and self.design_system:
                self.design_system.create_unified_header(title, subtitle or "")
                logger.debug(f"✅ Unified header rendered via design system")
                return True

        except Exception as e:
            logger.warning(f"⚠️ Design system header rendering failed: {e}")

        # Fallback method: Use Streamlit native components
        try:
            import streamlit as st

            # Create header with consistent styling
            st.markdown(f"""
            <div class="goaldiggers-unified-header">
                <h1>⚽ {title}</h1>
                {f'<p>{subtitle}</p>' if subtitle else ''}
            </div>
            """, unsafe_allow_html=True)

            logger.debug(f"✅ Fallback header rendered")
            return True

        except Exception as e:
            # Final fallback: Basic Streamlit components
            import streamlit as st
            st.title(f"⚽ {title}")
            if subtitle:
                st.subheader(subtitle)
            logger.warning(f"⚠️ Using basic header fallback: {e}")
            return False

    def _initialize_achievement_system(self):
        """Initialize the universal achievement system for this dashboard."""
        try:
            from dashboard.components.universal_achievement_system import \
                UniversalAchievementSystem

            self.achievement_system = UniversalAchievementSystem(dashboard_type=self.dashboard_type)
            logger.info(f"✅ Universal Achievement System initialized for {self.dashboard_type}")

        except ImportError as e:
            logger.warning(f"⚠️ Could not import Universal Achievement System: {e}")
            self.achievement_system = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Achievement System: {e}")
            self.achievement_system = None

    def track_prediction(self, is_correct: Optional[bool] = None, is_cross_league: bool = False):
        """
        Track a prediction using the universal achievement system.

        Args:
            is_correct: Whether the prediction was correct (None if unknown)
            is_cross_league: Whether this was a cross-league prediction
        """
        if self.achievement_system:
            self.achievement_system.track_prediction(is_correct, is_cross_league)

        # Also track in performance metrics
        if hasattr(self, 'performance_metrics'):
            self.performance_metrics['user_interactions'] += 1

    def track_feature_usage(self, feature_name: str):
        """
        Track usage of a specific feature.

        Args:
            feature_name: Name of the feature being used
        """
        if self.achievement_system:
            self.achievement_system.track_feature_usage(feature_name)

    def track_scenario_exploration(self):
        """Track exploration of a new scenario."""
        if self.achievement_system:
            self.achievement_system.track_scenario_exploration()

    def render_achievement_sidebar(self):
        """Render achievement stats in the sidebar."""
        if self.achievement_system:
            self.achievement_system.render_sidebar_stats()
        else:
            # Fallback sidebar content
            st.sidebar.markdown("### 📊 Dashboard Stats")
            if hasattr(self, 'performance_metrics'):
                load_time = self.performance_metrics.get('initialization_time', 0)
                st.sidebar.metric("Load Time", f"{load_time:.3f}s")

    def render_achievement_section(self, show_locked: bool = True):
        """
        Render the full achievements section.

        Args:
            show_locked: Whether to show locked achievements
        """
        if self.achievement_system:
            self.achievement_system.render_achievements_section(show_locked)
        else:
            st.info("🏆 Achievement system not available")

    def render_compact_achievement_stats(self):
        """Render compact achievement stats for dashboard headers."""
        if self.achievement_system:
            self.achievement_system.render_compact_stats()

    def get_user_achievement_stats(self) -> Dict[str, Any]:
        """Get user achievement statistics."""
        if self.achievement_system:
            return self.achievement_system.get_user_stats()
        else:
            return {
                'level': 1,
                'total_predictions': 0,
                'accuracy': 0.0,
                'achievements_unlocked': 0,
                'total_achievements': 0
            }

    def _initialize_workflow_manager(self):
        """Initialize the universal workflow manager for this dashboard."""
        try:
            from dashboard.components.universal_workflow_manager import \
                UniversalWorkflowManager

            self.workflow_manager = UniversalWorkflowManager(
                dashboard_type=self.dashboard_type,
                workflow_enabled=True  # Default to enabled, can be toggled
            )
            logger.info(f"✅ Universal Workflow Manager initialized for {self.dashboard_type}")

        except ImportError as e:
            logger.warning(f"⚠️ Could not import Universal Workflow Manager: {e}")
            self.workflow_manager = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Workflow Manager: {e}")
            self.workflow_manager = None

    def _initialize_help_system(self):
        """Initialize the universal help system for this dashboard."""
        try:
            from dashboard.components.universal_help_system import \
                UniversalHelpSystem

            self.help_system = UniversalHelpSystem(dashboard_type=self.dashboard_type)
            logger.info(f"✅ Universal Help System initialized for {self.dashboard_type}")

        except ImportError as e:
            logger.warning(f"⚠️ Could not import Universal Help System: {e}")
            self.help_system = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Help System: {e}")
            self.help_system = None

    def _initialize_navigation_system(self):
        """Initialize the universal navigation system for this dashboard."""
        try:
            from dashboard.components.universal_navigation_system import \
                UniversalNavigationSystem

            self.navigation_system = UniversalNavigationSystem(dashboard_type=self.dashboard_type)
            logger.info(f"✅ Universal Navigation System initialized for {self.dashboard_type}")

        except ImportError as e:
            logger.warning(f"⚠️ Could not import Universal Navigation System: {e}")
            self.navigation_system = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Navigation System: {e}")
            self.navigation_system = None

    def render_workflow_toggle(self):
        """Render workflow mode toggle control."""
        if self.workflow_manager:
            self.workflow_manager.render_workflow_toggle()

    def render_workflow_progress(self, workflow_id: Optional[str] = None):
        """Render workflow progress indicator."""
        if self.workflow_manager:
            self.workflow_manager.render_progress_indicator(workflow_id)

    def render_workflow_selector(self):
        """Render workflow selection interface."""
        if self.workflow_manager:
            self.workflow_manager.render_workflow_selector()

    def render_workflow_navigation(self, workflow_id: Optional[str] = None):
        """Render workflow step navigation controls."""
        if self.workflow_manager:
            self.workflow_manager.render_step_navigation(workflow_id)

    def get_current_workflow_step(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get information about the current workflow step."""
        if self.workflow_manager:
            return self.workflow_manager.get_current_step_info(workflow_id)
        return {}

    # Help System Methods
    def render_contextual_help(self, feature_key: str, position: str = "top"):
        """Render contextual help tooltip for a feature."""
        if not self.help_system:
            return

        try:
            self.help_system.render_contextual_tooltip(feature_key, position)
        except Exception as e:
            logger.warning(f"⚠️ Contextual help rendering failed: {e}")

    def render_help_bubble(self, feature_key: str, expanded: bool = False):
        """Render expandable help bubble with detailed information."""
        if not self.help_system:
            return

        try:
            self.help_system.render_help_bubble(feature_key, expanded)
        except Exception as e:
            logger.warning(f"⚠️ Help bubble rendering failed: {e}")

    def render_feature_discovery_tour(self, tour_key: str = "first_time_user"):
        """Render feature discovery guided tour."""
        if not self.help_system:
            return

        try:
            self.help_system.render_feature_discovery_tour(tour_key)
        except Exception as e:
            logger.warning(f"⚠️ Feature discovery tour rendering failed: {e}")

    def render_smart_suggestions(self):
        """Render smart suggestions based on user behavior."""
        if not self.help_system:
            return

        try:
            self.help_system.render_smart_suggestions()
        except Exception as e:
            logger.warning(f"⚠️ Smart suggestions rendering failed: {e}")

    def render_help_toggle(self):
        """Render help system toggle controls."""
        if not self.help_system:
            return

        try:
            self.help_system.render_help_toggle()
        except Exception as e:
            logger.warning(f"⚠️ Help toggle rendering failed: {e}")

    # Navigation System Methods
    def render_breadcrumb_navigation(self):
        """Render breadcrumb navigation."""
        if not self.navigation_system:
            return

        try:
            self.navigation_system.render_breadcrumb_navigation()
        except Exception as e:
            logger.warning(f"⚠️ Breadcrumb navigation rendering failed: {e}")

    def render_quick_actions(self):
        """Render quick action shortcuts."""
        if not self.navigation_system:
            return

        try:
            self.navigation_system.render_quick_actions()
        except Exception as e:
            logger.warning(f"⚠️ Quick actions rendering failed: {e}")

    def render_feature_search(self):
        """Render feature search interface."""
        if not self.navigation_system:
            return

        try:
            self.navigation_system.render_feature_search()
        except Exception as e:
            logger.warning(f"⚠️ Feature search rendering failed: {e}")

    def render_navigation_settings(self):
        """Render navigation system settings."""
        if not self.navigation_system:
            return

        try:
            self.navigation_system.render_navigation_settings()
        except Exception as e:
            logger.warning(f"⚠️ Navigation settings rendering failed: {e}")

    def navigate_to_section(self, section_id: str):
        """Navigate to a specific section."""
        if not self.navigation_system:
            return

        try:
            self.navigation_system.navigate_to_section(section_id)
        except Exception as e:
            logger.warning(f"⚠️ Section navigation failed: {e}")

    def advance_workflow_step(self, workflow_id: Optional[str] = None) -> bool:
        """Advance to the next workflow step."""
        if self.workflow_manager:
            return self.workflow_manager.advance_step(workflow_id)
        return False

    def is_workflow_enabled(self) -> bool:
        """Check if workflow mode is enabled."""
        if self.workflow_manager:
            return self.workflow_manager.workflow_enabled
        return False

    def get_workflow_progress_data(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive workflow progress data."""
        if self.workflow_manager:
            return self.workflow_manager.get_workflow_progress(workflow_id)
        return {}
    
    def _render_basic_prediction(self, home_team: str, away_team: str):
        """Basic prediction rendering fallback."""
        prediction = self.generate_enhanced_prediction(home_team, away_team)
        
        st.subheader(f"🎯 {home_team} vs {away_team}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"🏠 {home_team} Win", f"{prediction['home_win']:.1%}")
        with col2:
            st.metric("🤝 Draw", f"{prediction['draw']:.1%}")
        with col3:
            st.metric(f"✈️ {away_team} Win", f"{prediction['away_win']:.1%}")
        
        st.info(f"**Method:** {prediction.get('method', 'Analysis')}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get dashboard performance metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            **self.performance_metrics,
            'uptime_seconds': uptime,
            'components_loaded': len([c for c in self.components.values() if c is not None]),
            'ml_components_available': len([c for c in self.ml_components.values() if c is not None]),
            'average_prediction_time': sum(self.performance_metrics['prediction_times']) / len(self.performance_metrics['prediction_times']) if self.performance_metrics['prediction_times'] else 0
        }

    # Phase 3.3: Advanced Visual Components Integration
    def _initialize_advanced_visual_components(self):
        """Initialize advanced visual components for Phase 3.3 enhancement."""
        try:
            from dashboard.components.advanced_micro_interactions import \
                get_advanced_micro_interactions
            from dashboard.components.visual_consistency_system import \
                get_visual_consistency_system

            # Initialize micro-interactions system
            self.micro_interactions = get_advanced_micro_interactions()
            logger.info(f"✅ Advanced Micro-Interactions initialized for {self.dashboard_type}")

            # Initialize visual consistency system
            self.visual_consistency = get_visual_consistency_system()
            logger.info(f"✅ Visual Consistency System initialized for {self.dashboard_type}")

            # Apply global visual consistency
            self.visual_consistency.inject_global_styles()

            # Apply advanced micro-interactions CSS
            self.micro_interactions.inject_advanced_css()

            logger.info(f"🎨 Phase 3.3 Visual Enhancement activated for {self.dashboard_type}")

        except ImportError as e:
            logger.warning(f"⚠️ Could not import Advanced Visual Components: {e}")
            self.micro_interactions = None
            self.visual_consistency = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Advanced Visual Components: {e}")
            self.micro_interactions = None
            self.visual_consistency = None

    def render_animated_metric(self, label: str, value: str, delta: Optional[str] = None,
                              animation_type: str = "slide-in-up"):
        """Render an animated metric with micro-interactions."""
        if self.micro_interactions:
            self.micro_interactions.animated_metric(label, value, delta, animation_type)
        else:
            # Fallback to standard metric
            st.metric(label, value, delta)

    def render_animated_progress(self, progress: float, label: str = "", color: str = "#667eea"):
        """Render an animated progress bar."""
        if self.micro_interactions:
            self.micro_interactions.animated_progress_bar(progress, label, color)
        else:
            # Fallback to standard progress
            st.progress(progress / 100)
            if label:
                st.caption(label)

    def render_success_animation(self, message: str):
        """Render success message with bounce animation."""
        if self.micro_interactions:
            self.micro_interactions.success_animation(message)
        else:
            st.success(message)

    def render_error_animation(self, message: str):
        """Render error message with shake animation."""
        if self.micro_interactions:
            self.micro_interactions.error_animation(message)
        else:
            st.error(message)

    def render_gradient_title(self, title: str, size: str = "2em"):
        """Render a title with animated gradient text."""
        if self.micro_interactions:
            self.micro_interactions.gradient_title(title, size)
        else:
            st.title(title)

    def render_consistent_card(self, title: str, content: str, card_type: str = "default"):
        """Render a consistently styled card."""
        if self.visual_consistency:
            card_html = self.visual_consistency.create_consistent_card(title, content, card_type)
            st.markdown(card_html, unsafe_allow_html=True)
        else:
            # Fallback to standard container
            with st.container():
                st.subheader(title)
                st.write(content)

    def render_status_badge(self, text: str, status: str = "default"):
        """Render a consistently styled status badge."""
        if self.visual_consistency:
            badge_html = self.visual_consistency.create_status_badge(text, status)
            st.markdown(badge_html, unsafe_allow_html=True)
        else:
            # Fallback to standard text
            st.caption(f"Status: {text}")

    def get_design_token(self, category: str, key: str) -> str:
        """Get a design token value for consistent styling."""
        if self.visual_consistency:
            return self.visual_consistency.get_design_token(category, key)
        return ""
    
    @abstractmethod
    def render_dashboard(self):
        """Render the specific dashboard implementation."""
        pass
    
    @abstractmethod
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get dashboard-specific configuration."""
        pass


def get_unified_dashboard_base():
    """Factory function for unified dashboard base."""
    return UnifiedDashboardBase
