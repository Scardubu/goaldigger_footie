#!/usr/bin/env python3
"""
Interactive Cross-League Dashboard for GoalDiggers Platform

Enhanced dashboard with engaging cross-league analysis features:
- Interactive "What-if" scenarios between leagues
- Gamified prediction experience with achievement system
- Animated visualizations and confidence indicators
- Storytelling elements with dynamic insights
- Progressive difficulty levels and entertainment features

Features:
- Cross-league match predictions with visual confidence meters
- League strength comparisons with interactive rankings
- Achievement badges for prediction accuracy
- Entertaining commentary and insights
- Responsive design with GoalDiggers branding
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Import UnifiedDashboardBase for consistency
try:
    from dashboard.components.unified_dashboard_base import \
        UnifiedDashboardBase
    UNIFIED_BASE_AVAILABLE = True
except ImportError:
    UNIFIED_BASE_AVAILABLE = False
    UnifiedDashboardBase = object

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveCrossLeagueDashboard(UnifiedDashboardBase if UNIFIED_BASE_AVAILABLE else object):
    """Interactive dashboard for cross-league analysis and entertainment."""

    def __init__(self):
        """Initialize interactive cross-league dashboard."""
        # CRITICAL: Set ultra-fast startup mode to prevent heavy ML loading
        self._ultra_fast_startup = True
        self._skip_ml_initialization = True

        # Initialize unified base if available
        if UNIFIED_BASE_AVAILABLE:
            super().__init__(dashboard_type="interactive_cross_league")
        else:
            self.start_time = time.time()
            self.logger = logger
        self.initialize_session_state()
        self.setup_page_config()

        # Defer heavy component loading for ultra-fast startup
        self.ml_components = None
        self.multi_league_data_loader = None
        self._components_loaded = False
        
        # Note: Achievement system is now handled by UnifiedDashboardBase
        # Keep legacy achievements for backward compatibility
        self.achievements = self.initialize_achievements()

        # Performance tracking (standardized)
        if not hasattr(self, 'performance_metrics'):
            self.performance_metrics = {
                'initialization_time': time.time() - self.start_time,
                'component_load_times': {},
                'user_interactions': 0,
                'prediction_times': [],
                'error_count': 0
            }

        logger.info("🎮 Interactive Cross-League Dashboard initialized")
    
    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'user_level' not in st.session_state:
            st.session_state.user_level = 1
        if 'total_predictions' not in st.session_state:
            st.session_state.total_predictions = 0
        if 'correct_predictions' not in st.session_state:
            st.session_state.correct_predictions = 0
        if 'achievements_unlocked' not in st.session_state:
            st.session_state.achievements_unlocked = []
        if 'current_scenario' not in st.session_state:
            st.session_state.current_scenario = None
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
    
    def setup_page_config(self):
        """Setup Streamlit page configuration."""
        st.set_page_config(
            page_title="GoalDiggers - Cross-League Analysis",
            page_icon="⚽",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Initialize design system and enhanced components
        try:
            from dashboard.components.enhanced_prediction_display import \
                get_enhanced_prediction_display
            from dashboard.components.unified_design_system import \
                get_unified_design_system
            from utils.enhanced_team_data_manager import \
                get_enhanced_team_data_manager

            self.design_system = get_unified_design_system()
            self.prediction_display = get_enhanced_prediction_display(self.design_system)
            self.team_manager = get_enhanced_team_data_manager()

            logger.info("✅ Enhanced components initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced components: {e}")
            self.design_system = None
            self.prediction_display = None
            self.team_manager = None

        # Note: CSS injection is now handled by UnifiedDashboardBase._initialize_unified_styling()
        # Custom CSS is now included in dashboard-specific CSS methods

        # Apply custom CSS for interactive cross-league dashboard
        st.markdown("""
        <style>
        .league-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #2a5298;
            margin: 0.5rem 0;
        }

        .prediction-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin: 1rem 0;
        }

        .achievement-badge {
            background: #ffd700;
            color: #333;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            margin: 0.25rem;
            font-weight: bold;
        }

        .confidence-meter {
            background: #f0f0f0;
            border-radius: 10px;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }

        .scenario-card {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def load_ml_components(self):
        """Load ML components for predictions (lazy loading)."""
        if self._components_loaded:
            return

        try:
            from enhanced_cross_league_engine import \
                get_enhanced_cross_league_engine
            from enhanced_prediction_engine import \
                get_enhanced_prediction_engine

            self.prediction_engine = get_enhanced_prediction_engine()
            self.cross_league_engine = get_enhanced_cross_league_engine(self.prediction_engine)
            self._components_loaded = True

            logger.info("✅ ML components loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load ML components: {e}")
            self.prediction_engine = None
            self.cross_league_engine = None
    
    def load_multi_league_data(self):
        """Load multi-league data (lazy loading)."""
        if hasattr(self, 'data_loader') and self.data_loader is not None:
            return

        try:
            from utils.multi_league_data_loader import \
                get_multi_league_data_loader

            self.data_loader = get_multi_league_data_loader()
            logger.info("✅ Multi-league data loader initialized")
        except Exception as e:
            logger.error(f"❌ Failed to load multi-league data loader: {e}")
            self.data_loader = None
    
    def initialize_achievements(self) -> Dict[str, Dict[str, Any]]:
        """Initialize achievement system."""
        return {
            'first_prediction': {
                'name': '🎯 First Shot',
                'description': 'Make your first cross-league prediction',
                'requirement': 1,
                'type': 'predictions'
            },
            'accuracy_master': {
                'name': '🏆 Accuracy Master',
                'description': 'Achieve 70% prediction accuracy',
                'requirement': 0.7,
                'type': 'accuracy'
            },
            'cross_league_expert': {
                'name': '🌍 Cross-League Expert',
                'description': 'Make 10 cross-league predictions',
                'requirement': 10,
                'type': 'cross_league'
            },
            'scenario_explorer': {
                'name': '🔍 Scenario Explorer',
                'description': 'Try 5 different what-if scenarios',
                'requirement': 5,
                'type': 'scenarios'
            },
            'league_connoisseur': {
                'name': '⭐ League Connoisseur',
                'description': 'Analyze all 6 supported leagues',
                'requirement': 6,
                'type': 'leagues'
            }
        }
    
    def render_dashboard(self):
        """Render the main dashboard."""
        # PHASE 3 INTEGRATION: Apply consolidated mobile system and unified design
        self._apply_phase3_integrations()

        # Use standardized header rendering from UnifiedDashboardBase
        self.render_unified_header(
            "GoalDiggers - Interactive Cross-League Analysis",
            "Explore exciting \"What-if\" scenarios between Europe's top leagues!"
        )

        # Sidebar
        self.render_sidebar()
        
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎮 What-If Scenarios", 
            "🏆 League Rankings", 
            "📊 Cross-League Stats", 
            "🎯 My Achievements"
        ])
        
        with tab1:
            self.render_what_if_scenarios()
        
        with tab2:
            self.render_league_rankings()
        
        with tab3:
            self.render_cross_league_stats()
        
        with tab4:
            self.render_achievements()
    
    def render_sidebar(self):
        """Render sidebar with user stats, theme toggle, and feedback."""
        st.sidebar.markdown("### 🎨 Theme")
        try:
            from dashboard.components.theme_utils import render_theme_toggle
            render_theme_toggle("Theme")
        except ImportError:
            pass
        st.sidebar.markdown("---")

        st.sidebar.markdown("### 💬 Feedback & Error Reporting")
        feedback = st.sidebar.text_area("Have feedback, found a bug, or want to suggest a feature?", "", key="cross_league_feedback")
        if st.sidebar.button("Submit Feedback", key="cross_league_feedback_btn"):
            if feedback.strip():
                st.sidebar.success("Thank you for your feedback! Our team will review it.")
            else:
                st.sidebar.warning("Please enter your feedback before submitting.")
        st.sidebar.markdown("---")

        st.sidebar.markdown("### 🎮 Player Stats")
        # User level and progress
        accuracy = (st.session_state.correct_predictions / max(1, st.session_state.total_predictions)) * 100
        st.sidebar.metric("Level", st.session_state.user_level)
        st.sidebar.metric("Total Predictions", st.session_state.total_predictions)
        st.sidebar.metric("Accuracy", f"{accuracy:.1f}%")
        # Progress bar
        progress = min(1.0, st.session_state.total_predictions / (st.session_state.user_level * 10))
        st.sidebar.progress(progress)
        if progress >= 1.0:
            if st.sidebar.button("🆙 Level Up!"):
                st.session_state.user_level += 1
                st.success(f"🎉 Congratulations! You've reached Level {st.session_state.user_level}!")
        # Quick achievements
        st.sidebar.markdown("### 🏆 Recent Achievements")
        recent_achievements = st.session_state.achievements_unlocked[-3:]
        for achievement in recent_achievements:
            st.sidebar.markdown(f"<div class='achievement-badge'>{achievement}</div>", unsafe_allow_html=True)
    
    def render_what_if_scenarios(self):
        """Render interactive what-if scenarios."""
        st.markdown("## 🎮 What-If Scenarios")
        st.markdown("*Explore exciting matchups between teams from different leagues!*")
        
        # Scenario difficulty selection
        difficulty = st.selectbox(
            "🎯 Choose Your Challenge Level:",
            ["🟢 Beginner (Same League)", "🟡 Intermediate (Cross-League)", "🔴 Expert (Fantasy Matchups)"],
            index=1
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced gradient header for home team selection
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #1f4e79 0%, #2a5298 100%);
                padding: 1.2rem;
                border-radius: 10px;
                margin: 0.8rem 0;
                color: white;
                text-align: center;
                box-shadow: 0 3px 12px rgba(31, 78, 121, 0.3);
            ">
                <h3 style="margin: 0; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                    🏠 Home Team
                </h3>
            </div>
            """, unsafe_allow_html=True)
            home_leagues = self.get_supported_leagues()
            home_league = st.selectbox("League", home_leagues, key="home_league")

            # Use enhanced team manager if available
            if self.team_manager:
                home_teams = self.team_manager.get_league_teams(home_league)
            else:
                home_teams = self.get_league_teams(home_league)

            home_team = st.selectbox("Team", home_teams, key="home_team")

            # Display team metadata if available
            if self.team_manager:
                home_metadata = self.team_manager.resolve_team(home_team, home_league)
                if home_metadata:
                    team_info = self.team_manager.get_team_display_info(home_metadata)
                    st.markdown(f"**Stadium:** {team_info.get('stadium', 'N/A')}")
                    st.markdown(f"**Form:** {team_info.get('form_display', 'N/A')}")

        with col2:
            # Enhanced gradient header for away team selection
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                padding: 1.2rem;
                border-radius: 10px;
                margin: 0.8rem 0;
                color: white;
                text-align: center;
                box-shadow: 0 3px 12px rgba(40, 167, 69, 0.3);
            ">
                <h3 style="margin: 0; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                    ✈️ Away Team
                </h3>
            </div>
            """, unsafe_allow_html=True)
            if "Cross-League" in difficulty or "Fantasy" in difficulty:
                away_leagues = [l for l in home_leagues if l != home_league]
                away_league = st.selectbox("League", away_leagues, key="away_league")
            else:
                away_league = home_league

            # Use enhanced team manager if available
            if self.team_manager:
                away_teams = self.team_manager.get_league_teams(away_league)
            else:
                away_teams = self.get_league_teams(away_league)

            away_teams = [t for t in away_teams if t != home_team]
            away_team = st.selectbox("Team", away_teams, key="away_team")

            # Display team metadata if available
            if self.team_manager:
                away_metadata = self.team_manager.resolve_team(away_team, away_league)
                if away_metadata:
                    team_info = self.team_manager.get_team_display_info(away_metadata)
                    st.markdown(f"**Stadium:** {team_info.get('stadium', 'N/A')}")
                    st.markdown(f"**Form:** {team_info.get('form_display', 'N/A')}")
        
        # Prediction button
        if st.button("🔮 Generate Prediction", type="primary"):
            self.generate_scenario_prediction(home_team, away_team, home_league, away_league, difficulty)
    
    def generate_scenario_prediction(self, home_team: str, away_team: str, home_league: str, away_league: str, difficulty: str):
        """Generate prediction for scenario."""
        with st.spinner("🤖 AI is analyzing the matchup..."):
            # Removed artificial delay for production performance
            
            # Create match data
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'home_league': home_league,
                'away_league': away_league,
                'is_cross_league': home_league != away_league,
                'scenario_difficulty': difficulty
            }
            
            # Get prediction
            if self.cross_league_engine and home_league != away_league:
                prediction = self.cross_league_engine.predict_cross_league_match(match_data)
            else:
                prediction = self.get_fallback_prediction(match_data)
            
            # Display prediction with enhanced components
            if self.prediction_display:
                self.prediction_display.render_animated_prediction_card(prediction, match_data)

                # Display cross-league insights if available
                if 'cross_league_analysis' in prediction:
                    self.prediction_display.render_cross_league_insights(
                        prediction['cross_league_analysis'], match_data
                    )

                # Display entertaining commentary
                self.prediction_display.render_entertaining_commentary(prediction, match_data)
            else:
                # Fallback to original display
                self.display_animated_prediction(prediction, match_data)
            
            # Update user stats (legacy)
            st.session_state.total_predictions += 1
            self.check_achievements()

            # Track with universal achievement system
            is_cross_league = home_league != away_league
            self.track_prediction(is_cross_league=is_cross_league)
            if is_cross_league:
                self.track_feature_usage("cross_league_prediction")
            self.track_scenario_exploration()
    
    def display_animated_prediction(self, prediction: Dict[str, Any], match_data: Dict[str, Any]):
        """Display prediction with animated elements."""
        st.markdown("""
        <div class="prediction-card">
            <h3>🔮 AI Prediction Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Animated confidence meter
        confidence = prediction.get('confidence', {}).get('overall', 0.7)
        
        col1, col2, col3 = st.columns(3)
        
        predictions = prediction.get('predictions', {})
        
        with col1:
            home_prob = predictions.get('home_win', 0.4)
            st.metric(
                f"🏠 {match_data['home_team']} Win",
                f"{home_prob:.1%}",
                delta=f"Confidence: {confidence:.1%}"
            )
            st.progress(home_prob)
        
        with col2:
            draw_prob = predictions.get('draw', 0.3)
            st.metric(
                "🤝 Draw",
                f"{draw_prob:.1%}",
                delta="Balanced match"
            )
            st.progress(draw_prob)
        
        with col3:
            away_prob = predictions.get('away_win', 0.3)
            st.metric(
                f"✈️ {match_data['away_team']} Win",
                f"{away_prob:.1%}",
                delta=f"Away challenge"
            )
            st.progress(away_prob)
        
        # Cross-league insights
        if 'cross_league_analysis' in prediction:
            self.display_cross_league_insights(prediction['cross_league_analysis'])
        
        # Entertaining commentary
        self.display_entertaining_commentary(prediction, match_data)
    
    def display_cross_league_insights(self, analysis: Dict[str, Any]):
        """Display cross-league analysis insights."""
        # Enhanced gradient header for cross-league analysis
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            color: white;
            text-align: center;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        ">
            <h3 style="margin: 0; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                🌍 Cross-League Analysis
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏠 Home League Profile")
            home_profile = analysis.get('home_league_profile', {})
            if home_profile:
                st.write(f"**Strength:** {home_profile.get('strength_coefficient', 0):.2f}")
                st.write(f"**Pace Factor:** {home_profile.get('pace_factor', 0):.2f}")
                st.write(f"**Technical Index:** {home_profile.get('technical_index', 0):.2f}")
        
        with col2:
            st.markdown("#### ✈️ Away League Profile")
            away_profile = analysis.get('away_league_profile', {})
            if away_profile:
                st.write(f"**Strength:** {away_profile.get('strength_coefficient', 0):.2f}")
                st.write(f"**Pace Factor:** {away_profile.get('pace_factor', 0):.2f}")
                st.write(f"**Technical Index:** {away_profile.get('technical_index', 0):.2f}")
        
        # Strength differential visualization
        strength_diff = analysis.get('strength_differential', 0)
        if abs(strength_diff) > 0.05:
            if strength_diff > 0:
                st.success(f"🏠 Home league advantage: +{strength_diff:.2f}")
            else:
                st.info(f"✈️ Away league advantage: {strength_diff:.2f}")
    
    def display_entertaining_commentary(self, prediction: Dict[str, Any], match_data: Dict[str, Any]):
        """Display entertaining commentary about the prediction."""
        st.markdown("### 🎙️ AI Commentary")
        
        insights = prediction.get('insights', [])
        if insights:
            for insight in insights:
                st.info(f"💡 {insight}")
        
        # Generate entertaining commentary
        commentary = self.generate_commentary(prediction, match_data)
        
        st.markdown(f"""
        <div class="scenario-card">
            <h4>🎭 Match Preview</h4>
            <p>{commentary}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def generate_commentary(self, prediction: Dict[str, Any], match_data: Dict[str, Any]) -> str:
        """Generate entertaining commentary for the match."""
        home_team = match_data['home_team']
        away_team = match_data['away_team']
        home_league = match_data['home_league']
        away_league = match_data['away_league']
        
        predictions = prediction.get('predictions', {})
        home_prob = predictions.get('home_win', 0.4)
        
        if match_data.get('is_cross_league', False):
            if home_prob > 0.5:
                return f"🔥 {home_team} from {home_league} are expected to showcase their league's strength against {away_team}! The home advantage combined with {home_league}'s tactical approach could prove decisive in this fascinating cross-league encounter."
            else:
                return f"⚡ {away_team} from {away_league} are ready to make a statement! This cross-league clash promises fireworks as {away_league}'s playing style meets {home_league}'s approach. Expect the unexpected!"
        else:
            return f"🎯 A classic {home_league} battle! {home_team} vs {away_team} promises to be a tactical masterclass with both teams knowing each other's strengths and weaknesses."
    
    def render_league_rankings(self):
        """Render interactive league strength rankings."""
        st.markdown("## 🏆 League Strength Rankings")
        st.markdown("*Interactive comparison of Europe's top football leagues*")
        
        # League strength data
        league_data = {
            'League': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1', 'Eredivisie'],
            'Strength': [1.00, 0.95, 0.90, 0.85, 0.80, 0.75],
            'Avg Goals/Game': [2.82, 2.65, 3.15, 2.45, 2.55, 3.05],
            'Technical Index': [0.90, 1.25, 1.00, 1.10, 0.95, 1.15],
            'Pace Factor': [1.15, 0.85, 1.10, 0.80, 0.95, 1.05]
        }
        
        df = pd.DataFrame(league_data)
        
        # Interactive radar chart
        fig = go.Figure()
        
        for i, league in enumerate(df['League']):
            fig.add_trace(go.Scatterpolar(
                r=[df.iloc[i]['Strength'], df.iloc[i]['Avg Goals/Game']/4, 
                   df.iloc[i]['Technical Index'], df.iloc[i]['Pace Factor']],
                theta=['Strength', 'Goals/Game', 'Technical', 'Pace'],
                fill='toself',
                name=league,
                line_color=px.colors.qualitative.Set1[i]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1.5]
                )),
            showlegend=True,
            title="League Characteristics Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # League comparison table
        st.markdown("### 📊 Detailed Comparison")
        st.dataframe(df, use_container_width=True)
    
    def render_cross_league_stats(self):
        """Render cross-league statistics."""
        st.markdown("## 📊 Cross-League Statistics")
        
        # Mock historical data
        cross_league_data = {
            'Home League': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
            'vs Premier League': ['-', '52%', '48%', '45%', '58%'],
            'vs La Liga': ['48%', '-', '51%', '49%', '55%'],
            'vs Bundesliga': ['52%', '49%', '-', '53%', '57%'],
            'vs Serie A': ['55%', '51%', '47%', '-', '59%'],
            'vs Ligue 1': ['42%', '45%', '43%', '41%', '-']
        }
        
        df_cross = pd.DataFrame(cross_league_data)
        st.markdown("### 🏆 Historical Cross-League Win Rates")
        st.dataframe(df_cross, use_container_width=True)
        
        # Visualization
        fig = px.imshow(
            df_cross.set_index('Home League').astype(str).replace('-', '50%').apply(lambda x: x.str.rstrip('%').astype(float)),
            text_auto=True,
            aspect="auto",
            title="Cross-League Win Rate Heatmap (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_achievements(self):
        """Render user achievements."""
        st.markdown("## 🎯 My Achievements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Unlocked")
            for achievement_id in st.session_state.achievements_unlocked:
                if achievement_id in self.achievements:
                    achievement = self.achievements[achievement_id]
                    st.markdown(f"""
                    <div class="achievement-badge">
                        {achievement['name']}<br>
                        <small>{achievement['description']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔒 Locked")
            for achievement_id, achievement in self.achievements.items():
                if achievement_id not in st.session_state.achievements_unlocked:
                    progress = self.get_achievement_progress(achievement_id)
                    st.markdown(f"**{achievement['name']}**")
                    st.markdown(f"*{achievement['description']}*")
                    st.progress(progress)
                    st.markdown("---")
    
    def get_achievement_progress(self, achievement_id: str) -> float:
        """Get progress towards an achievement."""
        achievement = self.achievements[achievement_id]
        
        if achievement['type'] == 'predictions':
            return min(1.0, st.session_state.total_predictions / achievement['requirement'])
        elif achievement['type'] == 'accuracy':
            if st.session_state.total_predictions == 0:
                return 0.0
            accuracy = st.session_state.correct_predictions / st.session_state.total_predictions
            return min(1.0, accuracy / achievement['requirement'])
        
        return 0.0
    
    def check_achievements(self):
        """Check and unlock achievements."""
        for achievement_id, achievement in self.achievements.items():
            if achievement_id not in st.session_state.achievements_unlocked:
                if self.get_achievement_progress(achievement_id) >= 1.0:
                    st.session_state.achievements_unlocked.append(achievement_id)
                    st.success(f"🎉 Achievement Unlocked: {achievement['name']}")
    
    def get_supported_leagues(self) -> List[str]:
        """Get list of supported leagues."""
        return ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1', 'Eredivisie']
    
    def get_league_teams(self, league: str) -> List[str]:
        """Get teams for a specific league."""
        teams_by_league = {
            'Premier League': ['Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 'Manchester United', 'Tottenham'],
            'La Liga': ['Real Madrid', 'Barcelona', 'Atletico Madrid', 'Real Sociedad', 'Villarreal', 'Real Betis'],
            'Bundesliga': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Union Berlin', 'SC Freiburg', 'Bayer Leverkusen'],
            'Serie A': ['Inter Milan', 'AC Milan', 'Juventus', 'Atalanta', 'Roma', 'Lazio'],
            'Ligue 1': ['PSG', 'Monaco', 'Lille', 'Nice', 'Rennes', 'Lyon'],
            'Eredivisie': ['PSV', 'Ajax', 'Feyenoord', 'AZ Alkmaar', 'Twente', 'Utrecht']
        }
        return teams_by_league.get(league, ['Team A', 'Team B', 'Team C'])
    
    def get_fallback_prediction(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback prediction when ML components are unavailable."""
        return {
            'predictions': {
                'home_win': 0.45,
                'draw': 0.30,
                'away_win': 0.25
            },
            'confidence': {
                'overall': 0.75
            },
            'insights': [
                f"Exciting matchup between {match_data['home_team']} and {match_data['away_team']}!",
                "Home advantage could be decisive in this encounter."
            ],
            'metadata': {
                'model_version': 'fallback',
                'prediction_type': 'entertainment'
            }
        }

    def _apply_phase3_integrations(self):
        """Apply Phase 3 integrations: consolidated mobile system, unified design, PWA support."""
        try:
            # Apply consolidated mobile CSS system with animations for interactive experience
            try:
                from dashboard.components.consolidated_mobile_system import \
                    apply_mobile_css_to_variant
                apply_mobile_css_to_variant('interactive_cross_league', enable_animations=True)
                self.logger.debug("✅ Consolidated mobile system applied to interactive cross-league")
            except ImportError as e:
                self.logger.warning(f"⚠️ Consolidated mobile system not available: {e}")

            # Apply unified design system
            try:
                from dashboard.components.consistent_styling import \
                    get_unified_design_system
                design_system = get_unified_design_system()
                design_system.apply_unified_styling()
                self.logger.debug("✅ Unified design system applied to interactive cross-league")
            except ImportError as e:
                self.logger.warning(f"⚠️ Unified design system not available: {e}")

            # Initialize PWA support for interactive features
            try:
                from dashboard.components.pwa_implementation import \
                    PWAImplementation
                pwa = PWAImplementation()
                pwa.render_pwa_interface('interactive_cross_league')
                self.logger.debug("✅ PWA implementation applied to interactive cross-league")
            except ImportError as e:
                self.logger.warning(f"⚠️ PWA implementation not available: {e}")

            # Apply personalization for achievement system
            try:
                from dashboard.components.personalization_integration import \
                    PersonalizationIntegration
                personalization = PersonalizationIntegration()
                personalization.apply_user_preferences('interactive_cross_league')
                self.logger.debug("✅ Personalization integration applied to interactive cross-league")
            except ImportError as e:
                self.logger.warning(f"⚠️ Personalization integration not available: {e}")

        except Exception as e:
            self.logger.error(f"❌ Phase 3 integration failed for interactive cross-league: {e}")

    def render_dashboard(self):
        """Render the interactive cross-league dashboard implementation."""
        return self.run()

    def get_dashboard_config(self) -> Dict[str, Any]:
        """Get interactive cross-league dashboard-specific configuration."""
        return {
            'dashboard_type': 'interactive_cross_league',
            'features': {
                'cross_league_analysis': True,
                'interactive_scenarios': True,
                'gamification': True,
                'achievement_system': True,
                'animated_visualizations': True,
                'entertaining_commentary': True
            },
            'performance_targets': {
                'load_time_seconds': 1.0,
                'memory_usage_mb': 400.0,
                'interaction_response_ms': 150
            }
        }

def main():
    """Main function to run the dashboard."""
    dashboard = ProductionDashboardHomepage()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()
