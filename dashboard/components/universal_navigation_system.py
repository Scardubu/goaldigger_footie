#!/usr/bin/env python3
"""
Universal Navigation System for GoalDiggers Platform

Modular navigation system that can be integrated across all dashboard variants
to provide breadcrumb navigation, quick action shortcuts, and feature search.

Features:
- Breadcrumb navigation with step tracking
- Quick action shortcuts for common tasks
- Feature search and discovery
- Smart navigation suggestions
- Performance-optimized with minimal memory footprint
- Extensible architecture for adding new navigation patterns
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

class UniversalNavigationSystem:
    """
    Universal navigation system for GoalDiggers dashboards.
    
    Provides consistent navigation patterns, breadcrumbs, quick actions,
    and feature discovery across all dashboard variants.
    """
    
    def __init__(self, dashboard_type: str = "universal"):
        """
        Initialize the universal navigation system.
        
        Args:
            dashboard_type: Type of dashboard for customized navigation
        """
        self.dashboard_type = dashboard_type
        self.navigation_structure = self._initialize_navigation_structure()
        self.quick_actions = self._initialize_quick_actions()
        self.search_index = self._build_search_index()
        self._initialize_session_state()
        
        logger.debug(f"✅ Universal Navigation System initialized for {dashboard_type}")
    
    def _initialize_session_state(self):
        """Initialize session state for navigation system."""
        if 'navigation_initialized' not in st.session_state:
            st.session_state.navigation_initialized = True
            st.session_state.navigation_breadcrumbs = []
            st.session_state.current_section = 'home'
            st.session_state.navigation_history = []
            st.session_state.quick_actions_used = []
            st.session_state.search_queries = []
            st.session_state.navigation_preferences = {
                'show_breadcrumbs': True,
                'show_quick_actions': True,
                'compact_mode': False
            }
    
    def _initialize_navigation_structure(self) -> Dict[str, Dict[str, Any]]:
        """Initialize navigation structure definitions."""
        base_structure = {
            'home': {
                'title': 'Dashboard Home',
                'icon': '🏠',
                'description': 'Main dashboard overview',
                'children': ['team_selection', 'analysis', 'results', 'insights'],
                'level': 0
            },
            'team_selection': {
                'title': 'Team Selection',
                'icon': '🎯',
                'description': 'Choose teams for analysis',
                'parent': 'home',
                'children': ['analysis'],
                'level': 1
            },
            'analysis': {
                'title': 'AI Analysis',
                'icon': '🤖',
                'description': 'Generate predictions',
                'parent': 'team_selection',
                'children': ['results'],
                'level': 2
            },
            'results': {
                'title': 'Results',
                'icon': '📊',
                'description': 'View predictions',
                'parent': 'analysis',
                'children': ['insights'],
                'level': 3
            },
            'insights': {
                'title': 'Insights',
                'icon': '💡',
                'description': 'Actionable recommendations',
                'parent': 'results',
                'children': [],
                'level': 4
            }
        }
        
        # Add dashboard-specific sections
        if self.dashboard_type == 'premium':
            base_structure.update({
                'value_betting': {
                    'title': 'Value Betting',
                    'icon': '💰',
                    'description': 'Betting opportunities',
                    'parent': 'results',
                    'children': [],
                    'level': 4
                },
                'achievements': {
                    'title': 'Achievements',
                    'icon': '🏆',
                    'description': 'Progress tracking',
                    'parent': 'home',
                    'children': [],
                    'level': 1
                }
            })
        elif self.dashboard_type == 'interactive':
            base_structure.update({
                'cross_league': {
                    'title': 'Cross-League Analysis',
                    'icon': '🌍',
                    'description': 'Compare across leagues',
                    'parent': 'analysis',
                    'children': [],
                    'level': 3
                }
            })
        
        return base_structure
    
    def _initialize_quick_actions(self) -> List[Dict[str, Any]]:
        """Initialize quick action definitions."""
        base_actions = [
            {
                'id': 'new_prediction',
                'title': 'New Prediction',
                'icon': '⚡',
                'description': 'Start a new prediction analysis',
                'shortcut': 'Ctrl+N',
                'action': self._action_new_prediction,
                'category': 'primary'
            },
            {
                'id': 'view_results',
                'title': 'View Results',
                'icon': '📊',
                'description': 'Go to latest results',
                'shortcut': 'Ctrl+R',
                'action': self._action_view_results,
                'category': 'navigation'
            },
            {
                'id': 'help_tour',
                'title': 'Help Tour',
                'icon': '🎯',
                'description': 'Start guided tour',
                'shortcut': 'F1',
                'action': self._action_help_tour,
                'category': 'help'
            },
            {
                'id': 'reset_session',
                'title': 'Reset Session',
                'icon': '🔄',
                'description': 'Clear all data and start fresh',
                'shortcut': 'Ctrl+Shift+R',
                'action': self._action_reset_session,
                'category': 'utility'
            },
            {
                'id': 'performance_stats',
                'title': 'Performance Stats',
                'icon': '⚡',
                'description': 'View system performance metrics',
                'shortcut': 'Ctrl+P',
                'action': self._action_performance_stats,
                'category': 'utility'
            },
            {
                'id': 'personalization',
                'title': 'Personalize',
                'icon': '👤',
                'description': 'Customize dashboard experience',
                'shortcut': 'Ctrl+U',
                'action': self._action_personalization,
                'category': 'utility'
            },
            {
                'id': 'export_results',
                'title': 'Export Results',
                'icon': '📤',
                'description': 'Export prediction data',
                'shortcut': 'Ctrl+E',
                'action': self._action_export_results,
                'category': 'utility'
            },
            {
                'id': 'search_features',
                'title': 'Search Features',
                'icon': '🔍',
                'description': 'Find features and settings',
                'shortcut': 'Ctrl+F',
                'action': self._action_search_features,
                'category': 'navigation'
            }
        ]
        
        # Add dashboard-specific actions
        if self.dashboard_type == 'premium':
            base_actions.extend([
                {
                    'id': 'value_betting',
                    'title': 'Value Betting',
                    'icon': '💰',
                    'description': 'Find betting opportunities',
                    'shortcut': 'Ctrl+V',
                    'action': self._action_value_betting,
                    'category': 'analysis'
                },
                {
                    'id': 'achievements',
                    'title': 'Achievements',
                    'icon': '🏆',
                    'description': 'View progress',
                    'shortcut': 'Ctrl+A',
                    'action': self._action_achievements,
                    'category': 'engagement'
                }
            ])
        
        return base_actions
    
    def _build_search_index(self) -> Dict[str, List[str]]:
        """Build search index for features and content."""
        index = {}
        
        # Index navigation sections
        for section_id, section in self.navigation_structure.items():
            keywords = [
                section['title'].lower(),
                section['description'].lower(),
                section_id.lower()
            ]
            index[section_id] = keywords
        
        # Index quick actions
        for action in self.quick_actions:
            keywords = [
                action['title'].lower(),
                action['description'].lower(),
                action['id'].lower(),
                action['category'].lower()
            ]
            index[f"action_{action['id']}"] = keywords
        
        return index
    
    def render_breadcrumb_navigation(self):
        """Render breadcrumb navigation."""
        if not st.session_state.navigation_preferences['show_breadcrumbs']:
            return
        
        breadcrumbs = st.session_state.navigation_breadcrumbs
        if not breadcrumbs:
            return
        
        # Create breadcrumb display
        breadcrumb_items = []
        for i, crumb in enumerate(breadcrumbs):
            section = self.navigation_structure.get(crumb, {})
            icon = section.get('icon', '📍')
            title = section.get('title', crumb.title())
            
            if i == len(breadcrumbs) - 1:
                # Current page - no link
                breadcrumb_items.append(f"{icon} **{title}**")
            else:
                # Clickable breadcrumb
                breadcrumb_items.append(f"{icon} {title}")
        
        breadcrumb_text = " → ".join(breadcrumb_items)

        # Enhanced gradient breadcrumb display
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 4px solid #1f4e79;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(31, 78, 121, 0.1);
            border: 1px solid rgba(31, 78, 121, 0.1);
        ">
            <span style="color: #1f4e79; font-weight: 600;">
                🧭 {breadcrumb_text}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    def render_quick_actions(self):
        """Render quick action shortcuts."""
        if not st.session_state.navigation_preferences['show_quick_actions']:
            return
        
        st.sidebar.markdown("### ⚡ Quick Actions")
        
        # Group actions by category
        categories = {}
        for action in self.quick_actions:
            category = action.get('category', 'other')
            if category not in categories:
                categories[category] = []
            categories[category].append(action)
        
        # Render actions by category
        for category, actions in categories.items():
            if category != 'other':
                st.sidebar.markdown(f"**{category.title()}**")
            
            for action in actions:
                if st.sidebar.button(
                    f"{action['icon']} {action['title']}",
                    key=f"quick_action_{action['id']}",
                    help=f"{action['description']} ({action.get('shortcut', '')})"
                ):
                    self._execute_quick_action(action)
    
    def render_feature_search(self):
        """Render feature search interface."""
        st.sidebar.markdown("### 🔍 Feature Search")
        
        search_query = st.sidebar.text_input(
            "Search features...",
            placeholder="Type to search",
            key="navigation_search"
        )
        
        if search_query:
            results = self._search_features(search_query)
            
            if results:
                st.sidebar.markdown("**Search Results:**")
                for result in results[:5]:  # Show top 5 results
                    result_type = result['type']
                    result_data = result['data']
                    
                    if result_type == 'section':
                        section = self.navigation_structure[result_data]
                        if st.sidebar.button(
                            f"{section['icon']} {section['title']}",
                            key=f"search_result_{result_data}",
                            help=section['description']
                        ):
                            self.navigate_to_section(result_data)
                    
                    elif result_type == 'action':
                        action = next(a for a in self.quick_actions if a['id'] == result_data.replace('action_', ''))
                        if st.sidebar.button(
                            f"{action['icon']} {action['title']}",
                            key=f"search_action_{action['id']}",
                            help=action['description']
                        ):
                            self._execute_quick_action(action)
            else:
                st.sidebar.info("No results found")
    
    def _search_features(self, query: str) -> List[Dict[str, Any]]:
        """Search for features matching the query."""
        query_lower = query.lower()
        results = []
        
        for item_id, keywords in self.search_index.items():
            score = 0
            for keyword in keywords:
                if query_lower in keyword:
                    score += 1
                    if keyword.startswith(query_lower):
                        score += 2  # Boost for prefix matches
            
            if score > 0:
                if item_id.startswith('action_'):
                    results.append({
                        'type': 'action',
                        'data': item_id,
                        'score': score
                    })
                else:
                    results.append({
                        'type': 'section',
                        'data': item_id,
                        'score': score
                    })
        
        # Sort by score (descending)
        return sorted(results, key=lambda x: x['score'], reverse=True)
    
    def navigate_to_section(self, section_id: str):
        """Navigate to a specific section."""
        if section_id not in self.navigation_structure:
            return
        
        # Update current section
        st.session_state.current_section = section_id
        
        # Build breadcrumb trail
        breadcrumbs = self._build_breadcrumb_trail(section_id)
        st.session_state.navigation_breadcrumbs = breadcrumbs
        
        # Add to navigation history
        if section_id not in st.session_state.navigation_history:
            st.session_state.navigation_history.append(section_id)
        
        # Trigger rerun to update display
        st.rerun()
    
    def _build_breadcrumb_trail(self, section_id: str) -> List[str]:
        """Build breadcrumb trail for a section."""
        trail = []
        current = section_id
        
        while current:
            trail.insert(0, current)
            section = self.navigation_structure.get(current, {})
            current = section.get('parent')
        
        return trail
    
    def _execute_quick_action(self, action: Dict[str, Any]):
        """Execute a quick action."""
        try:
            action_func = action.get('action')
            if action_func and callable(action_func):
                action_func()
                
                # Track action usage
                st.session_state.quick_actions_used.append({
                    'action_id': action['id'],
                    'timestamp': time.time()
                })
                
                logger.debug(f"Quick action executed: {action['id']}")
        except Exception as e:
            logger.error(f"Failed to execute quick action {action['id']}: {e}")
            st.error(f"Action failed: {e}")
    
    # Quick Action Implementations
    def _action_new_prediction(self):
        """Start a new prediction analysis."""
        # Reset prediction-related session state
        for key in list(st.session_state.keys()):
            if 'prediction' in key.lower() or 'team' in key.lower():
                del st.session_state[key]
        
        self.navigate_to_section('team_selection')
        st.success("🎯 Ready for new prediction!")
    
    def _action_view_results(self):
        """Navigate to results section."""
        self.navigate_to_section('results')
    
    def _action_help_tour(self):
        """Start help tour."""
        st.session_state.tour_completed = {}
        st.success("🎯 Help tour will start on next interaction!")
    
    def _action_reset_session(self):
        """Reset the entire session."""
        # Clear all session state except navigation preferences
        prefs = st.session_state.navigation_preferences.copy()
        st.session_state.clear()
        st.session_state.navigation_preferences = prefs
        st.success("🔄 Session reset successfully!")

    def _action_performance_stats(self):
        """Show performance statistics."""
        self.navigate_to_section('performance')
        st.success("⚡ Performance stats displayed!")

    def _action_personalization(self):
        """Open personalization settings."""
        self.navigate_to_section('personalization')
        st.success("👤 Personalization options available!")

    def _action_export_results(self):
        """Export prediction results."""
        st.success("📤 Export functionality activated!")
        # Implementation would depend on available results

    def _action_search_features(self):
        """Activate feature search."""
        st.session_state.search_active = True
        st.success("🔍 Feature search activated!")
    
    def _action_value_betting(self):
        """Navigate to value betting section."""
        self.navigate_to_section('value_betting')
    
    def _action_achievements(self):
        """Navigate to achievements section."""
        self.navigate_to_section('achievements')
    
    def render_navigation_settings(self):
        """Render navigation system settings with enhanced gradient styling."""
        # Enhanced gradient header for navigation settings
        st.sidebar.markdown("""
        <div style="
            background: linear-gradient(135deg, #fd7e14 0%, #ffc107 100%);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            color: white;
            text-align: center;
            box-shadow: 0 3px 12px rgba(253, 126, 20, 0.3);
        ">
            <h4 style="margin: 0; font-weight: 600;">⚙️ Navigation Settings</h4>
        </div>
        """, unsafe_allow_html=True)

        with st.sidebar.container():
            st.session_state.navigation_preferences['show_breadcrumbs'] = st.checkbox(
                "Show Breadcrumbs",
                value=st.session_state.navigation_preferences['show_breadcrumbs']
            )
            
            st.session_state.navigation_preferences['show_quick_actions'] = st.checkbox(
                "Show Quick Actions",
                value=st.session_state.navigation_preferences['show_quick_actions']
            )
            
            st.session_state.navigation_preferences['compact_mode'] = st.checkbox(
                "Compact Mode",
                value=st.session_state.navigation_preferences['compact_mode']
            )
    
    def get_navigation_analytics(self) -> Dict[str, Any]:
        """Get navigation system usage analytics."""
        return {
            'current_section': st.session_state.get('current_section', 'home'),
            'breadcrumb_depth': len(st.session_state.get('navigation_breadcrumbs', [])),
            'history_length': len(st.session_state.get('navigation_history', [])),
            'quick_actions_used': len(st.session_state.get('quick_actions_used', [])),
            'search_queries': len(st.session_state.get('search_queries', [])),
            'preferences': st.session_state.get('navigation_preferences', {})
        }
