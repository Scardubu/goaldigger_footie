#!/usr/bin/env python3
"""
Enhanced Intuitive Sidebar for GoalDiggers Platform
Provides user-centric navigation with improved labels and organization
"""

from typing import Any, Dict, List

import streamlit as st


class EnhancedIntuitiveNavigation:
    """Enhanced navigation system with user-centric labels."""

    def __init__(self):
        self.navigation_items = {
            'Home': {
                'page': 'home',
                'description': 'Unified overview: ingestion, calibration, fixtures, insights',
                'icon': '🏠',
                'priority': 0
            },
            'Dashboard': {
                'page': 'dashboard',
                'description': 'Main dashboard with key insights and predictions',
                'icon': '📊',
                'priority': 1
            },
            'Match Predictions': {
                'page': 'predictions',
                'description': 'AI-powered match predictions and analysis',
                'icon': '🔮',
                'priority': 2
            },
            'Value Betting': {
                'page': 'value_bets',
                'description': 'Discover bets with positive expected value',
                'icon': '💰',
                'priority': 3
            },
            'Advanced Analytics': {
                'page': 'analytics',
                'description': 'Deep dive into team statistics and trends',
                'icon': '📊',
                'priority': 4
            },
            'Performance Tracking': {
                'page': 'performance',
                'description': 'Track your prediction accuracy and ROI',
                'icon': '📈',
                'priority': 5
            },
            'Live Odds': {
                'page': 'live_odds',
                'description': 'Real-time odds comparison across bookmakers',
                'icon': '⚡',
                'priority': 6
            },
            'Bankroll Management': {
                'page': 'bankroll',
                'description': 'Professional bankroll and stake management',
                'icon': '💼',
                'priority': 7
            },
            'Prediction History': {
                'page': 'history',
                'description': 'Review your past predictions and outcomes',
                'icon': '📋',
                'priority': 8
            },
            'Achievements': {
                'page': 'achievements',
                'description': 'Track progress and unlock achievements',
                'icon': '🏆',
                'priority': 9
            },
            'Settings': {
                'page': 'settings',
                'description': 'Customize your GoalDiggers experience',
                'icon': '⚙️',
                'priority': 10
            },
            'System Status': {
                'page': 'status',
                'description': 'System status and diagnostics',
                'icon': '🔧',
                'priority': 11
            }
        }

    def render_sidebar(self) -> str:
        """Render the enhanced sidebar with user-centric navigation."""
        with st.sidebar:
            # Header
            st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h2 style='color: #1e90ff; margin: 0;'>🎯 GoalDiggers</h2>
                <p style='font-size: 0.9rem; color: #666; margin: 0.5rem 0 0 0;'>
                    AI Football Intelligence
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Navigation items
            selected_page = None

            for name, config in self.navigation_items.items():
                if st.button(
                    f"{config['icon']} {name}",
                    key=f"nav_{config['page']}",
                    help=config['description'],
                    use_container_width=True
                ):
                    selected_page = config['page']
                    st.session_state.current_page = config['page']

            # Footer
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; padding: 1rem 0; font-size: 0.8rem; color: #666;'>
                <p>GoalDiggers Platform v3.0</p>
                <p>⚽ AI-Powered Football Intelligence</p>
            </div>
            """, unsafe_allow_html=True)

        # Return current page from session state or default
        return st.session_state.get('current_page', 'dashboard')

def render_enhanced_intuitive_sidebar() -> str:
    """Render the enhanced intuitive sidebar and return selected page."""
    navigator = EnhancedIntuitiveNavigation()
    return navigator.render_sidebar()

# For backward compatibility
def get_enhanced_navigation():
    """Get enhanced navigation instance."""
    return EnhancedIntuitiveNavigation()
