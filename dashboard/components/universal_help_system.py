#!/usr/bin/env python3
"""
Universal Help System for GoalDiggers Platform

Modular help and guidance system that can be integrated across all dashboard variants
to provide contextual assistance, feature discovery, and user onboarding.

Features:
- Contextual tooltips and help bubbles
- Feature discovery guided tours
- Smart suggestions based on user behavior
- Progressive disclosure of advanced features
- Performance-optimized with minimal memory footprint
- Extensible architecture for adding new help content
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

class UniversalHelpSystem:
    """
    Universal help system for GoalDiggers dashboards.
    
    Provides contextual assistance, feature discovery, and guided tours
    across all dashboard variants with consistent user experience.
    """
    
    def __init__(self, dashboard_type: str = "universal"):
        """
        Initialize the universal help system.
        
        Args:
            dashboard_type: Type of dashboard for customized help content
        """
        self.dashboard_type = dashboard_type
        self.help_content = self._initialize_help_content()
        self.tours = self._initialize_tours()
        self._initialize_session_state()
        
        logger.debug(f"✅ Universal Help System initialized for {dashboard_type}")
    
    def _initialize_session_state(self):
        """Initialize session state for help system."""
        if 'help_system_initialized' not in st.session_state:
            st.session_state.help_system_initialized = True
            st.session_state.help_tooltips_enabled = True
            st.session_state.tour_completed = {}
            st.session_state.help_suggestions_shown = []
            st.session_state.feature_discovery_progress = {}
            st.session_state.help_interactions = 0
            st.session_state.last_help_interaction = time.time()
    
    def _initialize_help_content(self) -> Dict[str, Dict[str, Any]]:
        """Initialize help content definitions."""
        base_content = {
            'team_selection': {
                'title': 'Team Selection',
                'tooltip': 'Choose two teams to analyze. Use the dropdown menus to select from top 6 leagues.',
                'detailed_help': 'Select home and away teams from Premier League, La Liga, Serie A, Bundesliga, Ligue 1, or Eredivisie. The AI will analyze their recent performance, head-to-head records, and current form.',
                'video_url': None,
                'category': 'basic'
            },
            'ai_analysis': {
                'title': 'AI Analysis',
                'tooltip': 'Our advanced ML models analyze team data to generate predictions.',
                'detailed_help': 'The AI engine processes over 50 data points including recent form, player injuries, historical matchups, and betting market trends to generate accurate predictions.',
                'video_url': None,
                'category': 'core'
            },
            'prediction_results': {
                'title': 'Prediction Results',
                'tooltip': 'View win probabilities and confidence levels for your selected match.',
                'detailed_help': 'Results show percentage probabilities for Home Win, Draw, and Away Win. Confidence levels indicate how certain the AI is about the prediction.',
                'video_url': None,
                'category': 'core'
            },
            'value_betting': {
                'title': 'Value Betting',
                'tooltip': 'Identify betting opportunities where odds exceed predicted probabilities.',
                'detailed_help': 'Value betting highlights when bookmaker odds are higher than our AI predictions suggest, indicating potential profitable betting opportunities.',
                'video_url': None,
                'category': 'advanced'
            },
            'achievements': {
                'title': 'Achievement System',
                'tooltip': 'Track your progress and unlock achievements as you use the platform.',
                'detailed_help': 'Earn points and unlock achievements by making predictions, exploring features, and improving your betting knowledge. Level up to access premium features.',
                'video_url': None,
                'category': 'engagement'
            },
            'workflow_mode': {
                'title': 'Workflow Mode',
                'tooltip': 'Enable step-by-step guided analysis for beginners.',
                'detailed_help': 'Workflow mode breaks down the prediction process into clear steps: Team Selection → AI Analysis → Results → Insights. Perfect for new users.',
                'video_url': None,
                'category': 'navigation'
            },
            'quick_actions': {
                'title': 'Quick Actions',
                'tooltip': 'Access frequently used features with one click.',
                'detailed_help': 'Quick Actions provide shortcuts to common tasks like starting new predictions, viewing results, and accessing help tours.',
                'video_url': None,
                'category': 'navigation'
            },
            'feature_search': {
                'title': 'Feature Search',
                'tooltip': 'Search for specific features and functionality.',
                'detailed_help': 'Use the search function to quickly find features, settings, or help content across the platform.',
                'video_url': None,
                'category': 'navigation'
            },
            'performance_metrics': {
                'title': 'Performance Metrics',
                'tooltip': 'Monitor dashboard performance and system health.',
                'detailed_help': 'View real-time performance metrics including load times, memory usage, and system status.',
                'video_url': None,
                'category': 'system'
            },
            'personalization': {
                'title': 'Personalization',
                'tooltip': 'Customize your dashboard experience.',
                'detailed_help': 'Personalize your interface with custom themes, preferred leagues, and adaptive recommendations.',
                'video_url': None,
                'category': 'user'
            }
        }
        
        # Add dashboard-specific content
        if self.dashboard_type == 'premium':
            base_content.update({
                'premium_features': {
                    'title': 'Premium Features',
                    'tooltip': 'Access advanced analytics, real-time data, and exclusive insights.',
                    'detailed_help': 'Premium features include live data streaming, advanced ML models, cross-league analysis, and personalized recommendations.',
                    'video_url': None,
                    'category': 'premium'
                }
            })
        elif self.dashboard_type == 'interactive':
            base_content.update({
                'cross_league': {
                    'title': 'Cross-League Analysis',
                    'tooltip': 'Compare teams and trends across different leagues.',
                    'detailed_help': 'Analyze performance patterns across Premier League, La Liga, Serie A, Bundesliga, Ligue 1, and Eredivisie to identify betting opportunities.',
                    'video_url': None,
                    'category': 'advanced'
                }
            })
        
        return base_content
    
    def _initialize_tours(self) -> Dict[str, Dict[str, Any]]:
        """Initialize guided tour definitions."""
        return {
            'first_time_user': {
                'name': 'Welcome to GoalDiggers',
                'description': 'Learn the basics of football prediction analysis',
                'steps': [
                    {
                        'target': 'team_selection',
                        'title': 'Step 1: Select Teams',
                        'content': 'Start by choosing two teams you want to analyze. Use the dropdown menus to select from top European leagues.',
                        'position': 'bottom'
                    },
                    {
                        'target': 'ai_analysis',
                        'title': 'Step 2: Generate Prediction',
                        'content': 'Click the prediction button to let our AI analyze the match. This takes about 30 seconds.',
                        'position': 'top'
                    },
                    {
                        'target': 'prediction_results',
                        'title': 'Step 3: Review Results',
                        'content': 'Examine the win probabilities and confidence levels. Higher confidence means more reliable predictions.',
                        'position': 'left'
                    },
                    {
                        'target': 'achievements',
                        'title': 'Step 4: Track Progress',
                        'content': 'Check your achievements and level up by using different features and making accurate predictions.',
                        'position': 'right'
                    }
                ],
                'category': 'onboarding'
            },
            'advanced_features': {
                'name': 'Advanced Features Tour',
                'description': 'Discover powerful analysis tools',
                'steps': [
                    {
                        'target': 'value_betting',
                        'title': 'Value Betting Analysis',
                        'content': 'Learn to identify profitable betting opportunities by comparing our predictions with bookmaker odds.',
                        'position': 'bottom'
                    },
                    {
                        'target': 'workflow_mode',
                        'title': 'Workflow Mode',
                        'content': 'Enable guided step-by-step analysis for a more structured approach to predictions.',
                        'position': 'top'
                    }
                ],
                'category': 'advanced'
            },
            'navigation_tour': {
                'name': 'Navigation & Quick Actions',
                'description': 'Master the navigation system',
                'steps': [
                    {
                        'target': 'quick_actions',
                        'title': 'Quick Actions',
                        'content': 'Access frequently used features with one-click shortcuts.',
                        'position': 'bottom'
                    },
                    {
                        'target': 'search_bar',
                        'title': 'Feature Search',
                        'content': 'Search for any feature or setting across the platform.',
                        'position': 'bottom'
                    },
                    {
                        'target': 'help_toggle',
                        'title': 'Help System',
                        'content': 'Get contextual help and guidance anywhere in the app.',
                        'position': 'left'
                    }
                ],
                'category': 'navigation'
            },
            'personalization_tour': {
                'name': 'Personalization Features',
                'description': 'Customize your experience',
                'steps': [
                    {
                        'target': 'dashboard_selector',
                        'title': 'Dashboard Variants',
                        'content': 'Choose from multiple dashboard variants optimized for different use cases.',
                        'position': 'top'
                    },
                    {
                        'target': 'league_preferences',
                        'title': 'Preferred Leagues',
                        'content': 'Set your favorite leagues for personalized recommendations.',
                        'position': 'right'
                    },
                    {
                        'target': 'adaptive_features',
                        'title': 'Adaptive Interface',
                        'content': 'The interface learns from your usage patterns and adapts accordingly.',
                        'position': 'center'
                    }
                ],
                'category': 'personalization'
            },
            'performance_tour': {
                'name': 'Performance & System Health',
                'description': 'Monitor system performance',
                'steps': [
                    {
                        'target': 'performance_metrics',
                        'title': 'Performance Metrics',
                        'content': 'Monitor real-time dashboard performance and system health.',
                        'position': 'top'
                    },
                    {
                        'target': 'memory_usage',
                        'title': 'Memory Usage',
                        'content': 'Track memory consumption to ensure optimal performance.',
                        'position': 'bottom'
                    },
                    {
                        'target': 'load_times',
                        'title': 'Load Times',
                        'content': 'Monitor dashboard loading times and optimization status.',
                        'position': 'left'
                    }
                ],
                'category': 'system'
            }
        }
    
    def render_contextual_tooltip(self, feature_key: str, position: str = "top") -> None:
        """
        Render a contextual tooltip for a specific feature.
        
        Args:
            feature_key: Key identifying the feature
            position: Tooltip position (top, bottom, left, right)
        """
        if not st.session_state.help_tooltips_enabled:
            return
        
        help_content = self.help_content.get(feature_key)
        if not help_content:
            return
        
        tooltip_text = help_content.get('tooltip', '')
        if tooltip_text:
            st.help(tooltip_text)
    
    def render_help_bubble(self, feature_key: str, expanded: bool = False) -> None:
        """
        Render an expandable help bubble with detailed information.
        
        Args:
            feature_key: Key identifying the feature
            expanded: Whether to show expanded view by default
        """
        help_content = self.help_content.get(feature_key)
        if not help_content:
            return
        
        title = help_content.get('title', 'Help')
        tooltip = help_content.get('tooltip', '')
        detailed_help = help_content.get('detailed_help', '')
        
        # Enhanced gradient help bubble instead of standard expander
        self._render_professional_help_bubble(title, tooltip, detailed_help, expanded)

        # Track help interaction
        self._track_help_interaction(feature_key)

    def _render_professional_help_bubble(self, title: str, tooltip: str, detailed_help: str, expanded: bool = False):
        """Render a professional gradient help bubble with enhanced styling."""
        # Create unique key for this help bubble
        help_key = f"help_{title.lower().replace(' ', '_')}"

        # Check if help bubble should be expanded
        if help_key not in st.session_state:
            st.session_state[help_key] = expanded

        # Toggle button with gradient styling
        if st.button(f"ℹ️ {title}", key=f"btn_{help_key}"):
            st.session_state[help_key] = not st.session_state[help_key]

        # Show help content if expanded
        if st.session_state[help_key]:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
                border-left: 4px solid #1f4e79;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(31, 78, 121, 0.1);
                border: 1px solid rgba(31, 78, 121, 0.2);
            ">
                {f'<div style="margin-bottom: 1rem;"><strong style="color: #1f4e79;">💡 Quick Tip:</strong> <span style="color: #374151;">{tooltip}</span></div>' if tooltip else ''}
                {f'<div><strong style="color: #1f4e79;">📖 Detailed Guide:</strong> <span style="color: #374151;">{detailed_help}</span></div>' if detailed_help else ''}
            </div>
            """, unsafe_allow_html=True)

    def render_feature_discovery_tour(self, tour_key: str = "first_time_user") -> None:
        """
        Render a feature discovery guided tour.
        
        Args:
            tour_key: Key identifying the tour to display
        """
        tour = self.tours.get(tour_key)
        if not tour:
            return
        
        # Check if tour was already completed
        if st.session_state.tour_completed.get(tour_key, False):
            return
        
        tour_name = tour.get('name', 'Guided Tour')
        tour_description = tour.get('description', '')
        
        # Show tour introduction
        with st.container():
            st.info(f"🎯 **{tour_name}**\n\n{tour_description}")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                if st.button("Start Tour", key=f"start_tour_{tour_key}"):
                    st.session_state[f"tour_active_{tour_key}"] = True
                    st.rerun()
            
            with col2:
                if st.button("Skip Tour", key=f"skip_tour_{tour_key}"):
                    st.session_state.tour_completed[tour_key] = True
                    st.rerun()
            
            with col3:
                if st.button("Remind Later", key=f"remind_tour_{tour_key}"):
                    st.session_state[f"tour_remind_later_{tour_key}"] = True
                    st.rerun()
    
    def render_smart_suggestions(self) -> None:
        """Render smart suggestions based on user behavior."""
        suggestions = self._generate_smart_suggestions()
        
        if suggestions:
            st.sidebar.markdown("### 💡 Smart Suggestions")
            
            for suggestion in suggestions[:3]:  # Show max 3 suggestions
                suggestion_id = suggestion.get('id', '')
                
                if suggestion_id not in st.session_state.help_suggestions_shown:
                    with st.sidebar.container():
                        st.info(f"**{suggestion['title']}**\n\n{suggestion['content']}")
                        
                        if st.button(f"Try it", key=f"suggestion_{suggestion_id}"):
                            self._handle_suggestion_action(suggestion)
                            st.session_state.help_suggestions_shown.append(suggestion_id)
                            st.rerun()
    
    def _generate_smart_suggestions(self) -> List[Dict[str, Any]]:
        """Generate smart suggestions based on user behavior."""
        suggestions = []
        
        # Suggestion based on usage patterns
        if st.session_state.help_interactions < 3:
            suggestions.append({
                'id': 'explore_help',
                'title': 'Explore Help Features',
                'content': 'Click on ℹ️ icons to learn more about each feature.',
                'action': 'show_help',
                'priority': 1
            })
        
        # Suggestion for workflow mode
        if not st.session_state.get('workflow_mode_tried', False):
            suggestions.append({
                'id': 'try_workflow',
                'title': 'Try Workflow Mode',
                'content': 'Enable step-by-step guidance for easier navigation.',
                'action': 'enable_workflow',
                'priority': 2
            })
        
        # Suggestion for achievements
        if not st.session_state.get('achievements_viewed', False):
            suggestions.append({
                'id': 'check_achievements',
                'title': 'Check Your Progress',
                'content': 'View your achievements and see how to level up.',
                'action': 'show_achievements',
                'priority': 3
            })
        
        return sorted(suggestions, key=lambda x: x.get('priority', 999))
    
    def _handle_suggestion_action(self, suggestion: Dict[str, Any]) -> None:
        """Handle suggestion action."""
        action = suggestion.get('action', '')
        
        if action == 'show_help':
            st.session_state.help_tooltips_enabled = True
        elif action == 'enable_workflow':
            st.session_state.workflow_mode_enabled = True
            st.session_state.workflow_mode_tried = True
        elif action == 'show_achievements':
            st.session_state.achievements_viewed = True
        
        # Track suggestion interaction
        self._track_help_interaction(f"suggestion_{suggestion.get('id', '')}")
    
    def _track_help_interaction(self, interaction_type: str) -> None:
        """Track help system interactions for analytics."""
        st.session_state.help_interactions += 1
        st.session_state.last_help_interaction = time.time()
        
        logger.debug(f"Help interaction tracked: {interaction_type}")
    
    def render_help_toggle(self) -> None:
        """Render help system toggle controls."""
        with st.sidebar:
            st.markdown("### 🆘 Help Settings")
            
            # Tooltip toggle
            st.session_state.help_tooltips_enabled = st.checkbox(
                "Show Tooltips",
                value=st.session_state.get('help_tooltips_enabled', True),
                help="Enable contextual tooltips throughout the interface"
            )
            
            # Tour reset button
            if st.button("Reset Tours", help="Reset all guided tours to show again"):
                st.session_state.tour_completed = {}
                st.success("Tours reset! They will appear again on next visit.")
    
    def get_help_analytics(self) -> Dict[str, Any]:
        """Get help system usage analytics."""
        return {
            'total_interactions': st.session_state.get('help_interactions', 0),
            'last_interaction': st.session_state.get('last_help_interaction', 0),
            'tours_completed': len(st.session_state.get('tour_completed', {})),
            'suggestions_shown': len(st.session_state.get('help_suggestions_shown', [])),
            'tooltips_enabled': st.session_state.get('help_tooltips_enabled', True)
        }
