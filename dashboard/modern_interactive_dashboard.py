#!/usr/bin/env python3
"""
Modern Interactive Dashboard for GoalDiggers Platform
Optimized for clear betting insights and user engagement.

This dashboard focuses on:
- Intuitive match selection and filtering
- Clear betting insights presentation
- Entertaining micro-interactions
- Mobile-first responsive design
- Actionable betting recommendations
"""

import importlib
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Import our enhanced layout system
from dashboard.components.enhanced_layout_system import enhanced_layout
from dashboard.components.ui_elements import render_banner
from dashboard.data_loader import DashboardDataLoader, create_minimal_loader
from dashboard.error_log import ErrorLog
from utils.exception_handler import ExceptionHandler
# Import memory optimization
from utils.memory_optimization import get_optimizer

logger = logging.getLogger(__name__)

class ModernInteractiveDashboard:
    """
    Modern, interactive dashboard optimized for betting insights.
    """
    
    def __init__(self):
        """Initialize the modern dashboard."""
        self.start_time = time.time()
        self.error_log = ErrorLog()
        self.layout = enhanced_layout
        
        # Initialize memory optimizer
        self.memory_optimizer = get_optimizer()
        
        # Initialize data loader
        try:
            self.data_loader = create_minimal_loader()
        except Exception as e:
            logger.error(f"Failed to initialize data loader: {e}")
            self.data_loader = None
        
        # Dashboard state
        self.current_view = "matches"
        self.selected_matches = []
        self.user_preferences = self._load_user_preferences()
        
        # Component cache for lazy loading
        self._loaded_components = {}
        
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences from session state."""
        return {
            'preferred_leagues': st.session_state.get('preferred_leagues', ['Premier League', 'La Liga']),
            'risk_tolerance': st.session_state.get('risk_tolerance', 'Medium'),
            'bet_amount_range': st.session_state.get('bet_amount_range', (10, 100)),
            'notification_preferences': st.session_state.get('notification_preferences', {
                'value_bets': True,
                'match_updates': False,
                'daily_summary': True
            })
        }
    
    def _save_user_preferences(self, preferences: Dict[str, Any]):
        """Save user preferences to session state."""
        for key, value in preferences.items():
            st.session_state[f'pref_{key}'] = value
    
    def lazy_load(self, component_name, default=None):
        """Lazily load a component only when needed."""
        if component_name not in self._loaded_components:
            try:
                logger.debug(f"Lazy loading component: {component_name}")
                start_time = time.time()
                
                # Import the component
                module_path, class_name = component_name.rsplit('.', 1)
                module = importlib.import_module(module_path)
                component_class = getattr(module, class_name)
                
                # Initialize the component
                self._loaded_components[component_name] = component_class()
                
                logger.debug(f"Loaded component {component_name} in {time.time() - start_time:.2f}s")
            except Exception as e:
                logger.error(f"Error lazy loading component {component_name}: {e}")
                return default
                
        return self._loaded_components.get(component_name, default)
        
    def optimize_initialization(self):
        """Optimize the dashboard initialization."""
        # Preload essential components
        essential_components = [
            "dashboard.components.ui_elements.UIElements",
            "dashboard.components.enhanced_renderer.HTMLRenderer"
        ]
        
        for component_name in essential_components:
            self.lazy_load(component_name)
            
        # Defer loading of heavy components
        logger.debug("Heavy components will be lazy-loaded when needed")
            
    def render_dashboard(self):
        """Render the complete modern dashboard."""
        try:
            # Optimize initialization
            self.optimize_initialization()
            
            # Check memory usage periodically
            self.memory_optimizer.check_and_optimize()
            
            # Configure page
            self._configure_page()
            
            # Load and apply CSS
            self.layout.load_enhanced_css()
            
            # Render hero header
            self.layout.render_hero_header(
                title="GoalDiggers",
                subtitle="AI-Powered Football Betting Intelligence"
            )
            
            # Render main navigation
            self._render_navigation()
            
            # Render main content based on current view
            if self.current_view == "matches":
                self._render_matches_view()
            elif self.current_view == "insights":
                self._render_insights_view()
            elif self.current_view == "analytics":
                self._render_analytics_view()
            elif self.current_view == "settings":
                self._render_settings_view()
            
            # Render footer
            self._render_footer()
            
            # Track memory usage of large objects
            self.memory_optimizer.track_object(self.selected_matches, "selected_matches")
            
        except Exception as e:
            self.error_log.error("Dashboard rendering error", exception=e)
            st.error("❌ Dashboard loading error. Please refresh the page.")
    
    def _configure_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="GoalDiggers - Football Betting Intelligence",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="collapsed",
            menu_items={
                'Get Help': 'https://goaldiggers.help',
                'Report a bug': 'https://goaldiggers.support',
                'About': "GoalDiggers - AI-Powered Football Betting Intelligence"
            }
        )
    
    def _render_navigation(self):
        """Render the main navigation bar."""
        st.markdown("""
        <div class="gd-layout-container">
            <div style="background: white; padding: 1rem 2rem; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 2rem;">
                <div style="display: flex; justify-content: center; gap: 2rem; align-items: center;">
        """, unsafe_allow_html=True)
        
        # Navigation tabs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🏆 Matches", use_container_width=True, type="primary" if self.current_view == "matches" else "secondary"):
                self.current_view = "matches"
                st.rerun()
        
        with col2:
            if st.button("💡 Insights", use_container_width=True, type="primary" if self.current_view == "insights" else "secondary"):
                self.current_view = "insights"
                st.rerun()
        
        with col3:
            if st.button("📊 Analytics", use_container_width=True, type="primary" if self.current_view == "analytics" else "secondary"):
                self.current_view = "analytics"
                st.rerun()
        
        with col4:
            if st.button("⚙️ Settings", use_container_width=True, type="primary" if self.current_view == "settings" else "secondary"):
                self.current_view = "settings"
                st.rerun()
        
        st.markdown("</div></div></div>", unsafe_allow_html=True)
    
    def _render_matches_view(self):
        """Render the matches view with enhanced filtering."""
        self.layout.render_section_header("🏆 Upcoming Matches", "Select matches for betting analysis")
        
        # Render filter bar
        filters = self.layout.render_interactive_filter_bar({
            'value_bets_only': False,
            'min_confidence': 0.7,
            'max_risk': 'Medium',
            'sort_by': 'Expected Value'
        })
        
        # Load matches data
        matches_data = self._load_matches_data(filters)
        
        if not matches_data:
            self._render_no_matches_state()
            return
        
        # Render matches in a responsive grid
        self._render_matches_grid(matches_data)
        
        # Render quick action buttons
        self._render_quick_actions()
    
    def _render_insights_view(self):
        """Render the betting insights view."""
        self.layout.render_section_header("💡 Betting Insights", "AI-powered recommendations for your selected matches")
        
        # Get selected matches
        if not self.selected_matches:
            st.info("👆 Please select matches from the Matches tab to see betting insights.")
            return
        
        # Generate insights for selected matches
        insights_data = self._generate_insights(self.selected_matches)
        
        # Render insights dashboard
        self.layout.render_insights_dashboard(insights_data)
        
        # Render recommended actions
        self._render_recommended_actions(insights_data)
    
    def _render_analytics_view(self):
        """Render the analytics view with performance metrics."""
        self.layout.render_section_header("📊 Performance Analytics", "Track your betting performance and AI accuracy")
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_performance_metrics()
        
        with col2:
            self._render_accuracy_trends()
        
        # League performance comparison
        league_data = self._get_league_performance_data()
        if not league_data.empty:
            self.layout.render_league_performance_chart(league_data)
    
    def _render_settings_view(self):
        """Render the settings and preferences view."""
        self.layout.render_section_header("⚙️ Settings & Preferences", "Customize your GoalDiggers experience")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Betting Preferences")
            
            # Risk tolerance
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["Conservative", "Moderate", "Aggressive"],
                index=["Conservative", "Moderate", "Aggressive"].index(self.user_preferences.get('risk_tolerance', 'Moderate'))
            )
            
            # Bet amount range
            bet_range = st.slider(
                "Bet Amount Range ($)",
                min_value=5,
                max_value=500,
                value=self.user_preferences.get('bet_amount_range', (10, 100)),
                step=5
            )
            
            # Preferred leagues
            all_leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Eredivisie"]
            preferred_leagues = st.multiselect(
                "Preferred Leagues",
                all_leagues,
                default=self.user_preferences.get('preferred_leagues', ['Premier League', 'La Liga'])
            )
        
        with col2:
            st.subheader("🔔 Notifications")
            
            # Notification preferences
            notify_value_bets = st.checkbox(
                "Value Bet Alerts",
                value=self.user_preferences.get('notification_preferences', {}).get('value_bets', True)
            )
            
            notify_match_updates = st.checkbox(
                "Match Updates",
                value=self.user_preferences.get('notification_preferences', {}).get('match_updates', False)
            )
            
            notify_daily_summary = st.checkbox(
                "Daily Summary",
                value=self.user_preferences.get('notification_preferences', {}).get('daily_summary', True)
            )
            
            st.subheader("🎨 Display")
            
            # Theme selection
            theme = st.selectbox(
                "Theme",
                ["Auto", "Light", "Dark"],
                index=0
            )
            
            # Dashboard layout
            layout_style = st.selectbox(
                "Layout Style",
                ["Compact", "Comfortable", "Spacious"],
                index=1
            )
        
        # Save preferences button
        if st.button("💾 Save Preferences", type="primary", use_container_width=True):
            new_preferences = {
                'risk_tolerance': risk_tolerance,
                'bet_amount_range': bet_range,
                'preferred_leagues': preferred_leagues,
                'notification_preferences': {
                    'value_bets': notify_value_bets,
                    'match_updates': notify_match_updates,
                    'daily_summary': notify_daily_summary
                },
                'theme': theme,
                'layout_style': layout_style
            }
            self._save_user_preferences(new_preferences)
            st.success("✅ Preferences saved successfully!")
            time.sleep(1)
            st.rerun()
    
    def _load_matches_data(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load and filter matches data."""
        try:
            if not self.data_loader:
                return self._get_mock_matches_data()
            
            # Load real data if available
            today = datetime.now().date()
            end_date = today + timedelta(days=7)
            
            matches_df = self.data_loader.load_matches(
                league_names=self.user_preferences.get('preferred_leagues', ['Premier League']),
                date_range=(today, end_date)
            )
            
            if matches_df.empty:
                return self._get_mock_matches_data()
            
            # Convert DataFrame to list of dicts
            matches_list = []
            for _, match in matches_df.iterrows():
                match_dict = match.to_dict()
                # Add betting insights
                match_dict.update(self._generate_match_insights(match_dict))
                matches_list.append(match_dict)
            
            # Apply filters
            filtered_matches = self._apply_filters(matches_list, filters)
            
            return filtered_matches
            
        except Exception as e:
            logger.error(f"Error loading matches data: {e}")
            return self._get_mock_matches_data()
    
    def _get_mock_matches_data(self) -> List[Dict[str, Any]]:
        """Generate mock matches data for demonstration."""
        mock_matches = [
            {
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'match_time': 'Today 15:00',
                'league': 'Premier League',
                'home_win_probability': 0.45,
                'draw_probability': 0.25,
                'away_win_probability': 0.30,
                'confidence': 0.82,
                'expected_value': 0.12,
                'is_value_bet': True,
                'risk_level': 'Medium',
                'home_form': ['W', 'W', 'D', 'W', 'L'],
                'away_form': ['L', 'W', 'D', 'W', 'W']
            },
            {
                'home_team': 'Barcelona',
                'away_team': 'Real Madrid',
                'match_time': 'Tomorrow 16:30',
                'league': 'La Liga',
                'home_win_probability': 0.38,
                'draw_probability': 0.32,
                'away_win_probability': 0.30,
                'confidence': 0.75,
                'expected_value': 0.08,
                'is_value_bet': False,
                'risk_level': 'High',
                'home_form': ['W', 'D', 'W', 'W', 'L'],
                'away_form': ['W', 'W', 'W', 'D', 'W']
            },
            {
                'home_team': 'Bayern Munich',
                'away_team': 'Borussia Dortmund',
                'match_time': 'Sunday 14:30',
                'league': 'Bundesliga',
                'home_win_probability': 0.52,
                'draw_probability': 0.28,
                'away_win_probability': 0.20,
                'confidence': 0.88,
                'expected_value': 0.15,
                'is_value_bet': True,
                'risk_level': 'Low',
                'home_form': ['W', 'W', 'W', 'D', 'W'],
                'away_form': ['L', 'D', 'W', 'L', 'W']
            }
        ]
        
        return mock_matches
    
    def _apply_filters(self, matches: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to matches data."""
        filtered = matches.copy()
        
        # Value bets only filter
        if filters.get('value_bets_only', False):
            filtered = [m for m in filtered if m.get('is_value_bet', False)]
        
        # Minimum confidence filter
        min_confidence = filters.get('min_confidence', 0.0)
        filtered = [m for m in filtered if m.get('confidence', 0) >= min_confidence]
        
        # Risk level filter
        max_risk = filters.get('max_risk', 'High')
        risk_order = ['Low', 'Medium', 'High']
        max_risk_level = risk_order.index(max_risk)
        filtered = [m for m in filtered if risk_order.index(m.get('risk_level', 'Medium')) <= max_risk_level]
        
        # Sort by selected criteria
        sort_by = filters.get('sort_by', 'Expected Value')
        if sort_by == 'Expected Value':
            filtered.sort(key=lambda x: x.get('expected_value', 0), reverse=True)
        elif sort_by == 'Confidence':
            filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        elif sort_by == 'Match Time':
            # For simplicity, just maintain original order
            pass
        
        return filtered
    
    def _generate_match_insights(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate betting insights for a match."""
        # Mock insight generation - in real implementation, this would use ML models
        home_prob = match_data.get('home_win_probability', 0.33)
        
        # Simple logic for demonstration
        confidence = 0.6 + (abs(home_prob - 0.5) * 0.4)  # Higher confidence for more decisive predictions
        expected_value = max(0, (confidence - 0.7) * 0.3)  # EV based on confidence
        is_value_bet = expected_value > 0.05
        
        risk_level = 'Low' if confidence > 0.8 else 'Medium' if confidence > 0.65 else 'High'
        
        return {
            'confidence': confidence,
            'expected_value': expected_value,
            'is_value_bet': is_value_bet,
            'risk_level': risk_level
        }
    
    def _render_matches_grid(self, matches_data: List[Dict[str, Any]]):
        """Render matches in a responsive grid."""
        if not matches_data:
            return
        
        # Create responsive columns
        cols_per_row = 2 if len(matches_data) > 1 else 1
        cols = st.columns(cols_per_row)
        
        for i, match in enumerate(matches_data):
            col_idx = i % cols_per_row
            with cols[col_idx]:
                self.layout.render_match_card(match)
                
                # Add selection button
                match_key = f"{match['home_team']}_vs_{match['away_team']}"
                if st.button(f"Select Match", key=f"select_{match_key}", use_container_width=True):
                    if match not in self.selected_matches:
                        self.selected_matches.append(match)
                        st.success(f"✅ Added {match['home_team']} vs {match['away_team']} to analysis")
                    time.sleep(0.5)
                    st.rerun()
    
    def _render_no_matches_state(self):
        """Render state when no matches are available."""
        st.markdown("""
        <div class="gd-layout-container">
            <div style="text-align: center; padding: 3rem; color: #6b7280;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">⚽</div>
                <h3 style="color: #374151; margin-bottom: 1rem;">No matches found</h3>
                <p>Try adjusting your filters or check back later for upcoming fixtures.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_quick_actions(self):
        """Render quick action buttons."""
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Matches", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button(f"📋 View Selected ({len(self.selected_matches)})", use_container_width=True):
                if self.selected_matches:
                    self.current_view = "insights"
                    st.rerun()
                else:
                    st.warning("No matches selected yet!")
        
        with col3:
            if st.button("🎯 Auto-Select Value Bets", use_container_width=True):
                # Auto-select high-value bets
                matches_data = self._load_matches_data({'value_bets_only': True, 'min_confidence': 0.75})
                self.selected_matches = matches_data[:3]  # Select top 3
                st.success(f"✅ Auto-selected {len(self.selected_matches)} value bets!")
                time.sleep(1)
                st.rerun()
    
    def _generate_insights(self, selected_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate comprehensive insights for selected matches."""
        # This would typically involve ML model predictions
        # For now, return the selected matches with enhanced analysis
        return selected_matches
    
    def _render_recommended_actions(self, insights_data: List[Dict[str, Any]]):
        """Render recommended betting actions."""
        if not insights_data:
            return
        
        self.layout.render_section_header("🎯 Recommended Actions", "AI-suggested betting strategies")
        
        value_bets = [insight for insight in insights_data if insight.get('is_value_bet', False)]
        
        if value_bets:
            st.success(f"🎉 Found {len(value_bets)} value betting opportunities!")
            
            for bet in value_bets:
                with st.expander(f"💰 {bet['home_team']} vs {bet['away_team']} - Value Bet"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Expected Value", f"{bet['expected_value']*100:.1f}%")
                        st.metric("Confidence", f"{bet['confidence']*100:.0f}%")
                    
                    with col2:
                        st.metric("Risk Level", bet['risk_level'])
                        recommended_stake = bet.get('recommended_stake', 50)
                        st.metric("Suggested Stake", f"${recommended_stake}")
                    
                    st.info(f"💡 **Strategy**: Bet on {bet['home_team']} to win with {bet['confidence']*100:.0f}% confidence.")
        else:
            st.info("No high-value betting opportunities found in your selection. Consider adjusting your criteria.")
    
    def _render_performance_metrics(self):
        """Render user performance metrics."""
        st.subheader("📈 Your Performance")
        
        # Mock performance data
        total_bets = 45
        winning_bets = 28
        win_rate = (winning_bets / total_bets) * 100 if total_bets > 0 else 0
        
        # Render metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Bets", total_bets)
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col2:
            st.metric("Winning Bets", winning_bets)
            st.metric("ROI", "+12.5%")
        
        # Progress indicators
        self.layout.render_progress_indicator(win_rate, 100, "Win Rate Progress", "success")
        self.layout.render_progress_indicator(75, 100, "AI Accuracy", "primary")
    
    def _render_accuracy_trends(self):
        """Render AI accuracy trends."""
        st.subheader("🎯 AI Accuracy Trends")
        
        # Mock trend data
        accuracy_data = {
            'Week': ['W1', 'W2', 'W3', 'W4'],
            'Accuracy': [72, 78, 85, 82]
        }
        
        df = pd.DataFrame(accuracy_data)
        st.line_chart(df.set_index('Week'))
        
        st.info("📊 AI prediction accuracy has improved by 15% over the last month!")
    
    def _get_league_performance_data(self) -> pd.DataFrame:
        """Get league performance data for visualization."""
        # Mock data for demonstration
        data = {
            'league': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
            'accuracy': [85.2, 82.7, 88.1, 79.3, 81.5]
        }
        return pd.DataFrame(data)
    
    def _render_footer(self):
        """Render dashboard footer."""
        st.markdown("---")
        st.markdown("""
        <div class="gd-layout-container">
            <div style="text-align: center; color: #6b7280; font-size: 0.875rem; padding: 2rem 0;">
                <p>🏆 GoalDiggers Football Betting Intelligence v2.0</p>
                <p>Powered by AI • Built for Winners • © 2025</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main function to run the modern interactive dashboard."""
    dashboard = ModernInteractiveDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()
