#!/usr/bin/env python3
"""
Personalization Integration Component
Phase 3B: Advanced Features Implementation - User Personalization Engine Activation

This component activates the existing preference engine across all dashboard variants,
implementing adaptive interfaces, personalized recommendations, and user preference learning.
Integrates seamlessly with the unified dashboard architecture from Phase 3A.

Key Features:
- Activation of existing preference engine across all dashboard variants
- Adaptive interface configuration based on user behavior
- Personalized team and league recommendations
- Risk tolerance analysis and betting style adaptation
- User onboarding flow and preference collection
- Privacy-preserving preference storage and management
"""

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Import existing personalization engine with enhanced integrations
try:
    from user.personalization.preference_engine import (
        PersonalizedRecommendation, PreferenceEngine, UserBehavior,
        UserPreference)
    from utils.html_sanitizer import sanitize_for_html
    PERSONALIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Personalization engine not available: {e}")
    PERSONALIZATION_AVAILABLE = False
    def sanitize_for_html(text): return str(text)

# Enhanced integrations with ML and achievement systems
try:
    from dashboard.components.achievement_system import AchievementSystem
    from dashboard.components.advanced_analytics_dashboard import \
        AdvancedAnalyticsDashboard
    from enhanced_prediction_engine import EnhancedPredictionEngine
    ENHANCED_INTEGRATIONS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Enhanced integrations not available: {e}")
    ENHANCED_INTEGRATIONS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonalizationLevel(Enum):
    """Personalization levels."""
    MINIMAL = "minimal"
    BASIC = "basic"
    ENHANCED = "enhanced"
    ADVANCED = "advanced"

class OnboardingStep(Enum):
    """User onboarding steps."""
    WELCOME = "welcome"
    PREFERENCES = "preferences"
    TEAMS = "teams"
    LEAGUES = "leagues"
    BETTING_STYLE = "betting_style"
    COMPLETE = "complete"

@dataclass
class PersonalizationConfig:
    """Configuration for personalization integration."""
    enable_personalization: bool = True
    enable_adaptive_interface: bool = True
    enable_recommendations: bool = True
    enable_onboarding: bool = True
    enable_risk_analysis: bool = True
    personalization_level: PersonalizationLevel = PersonalizationLevel.ENHANCED
    max_recommendations: int = 5
    key_prefix: str = "personalization"

class PersonalizationIntegration:
    """
    Personalization integration component activating preference engine
    across all dashboard variants with adaptive interfaces.
    """
    
    def __init__(self, config: PersonalizationConfig = None):
        """Initialize personalization integration with enhanced ML and achievement system integration."""
        self.config = config or PersonalizationConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize preference engine if available
        self.preference_engine = None
        if PERSONALIZATION_AVAILABLE and self.config.enable_personalization:
            try:
                self.preference_engine = PreferenceEngine()
                self.logger.info("✅ Preference engine initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize preference engine: {e}")

        # Initialize enhanced integrations
        self.prediction_engine = None
        self.achievement_system = None
        self.analytics_dashboard = None

        if ENHANCED_INTEGRATIONS_AVAILABLE:
            try:
                self.prediction_engine = EnhancedPredictionEngine()
                self.achievement_system = AchievementSystem()
                self.analytics_dashboard = AdvancedAnalyticsDashboard()
                self.logger.info("✅ Enhanced ML and achievement integrations initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize enhanced integrations: {e}")

        # Session management
        self.session_id = self._get_or_create_session_id()

        # Onboarding state
        self.onboarding_complete = False

        # Enhanced personalization data
        self.user_prediction_history = []
        self.personalized_insights = {}

        self.logger.info("🚀 Enhanced personalization integration initialized")
    
    def render_personalization_interface(self, dashboard_variant: str = "premium_ui"):
        """Main personalization interface rendering method."""
        try:
            if not self.config.enable_personalization or not self.preference_engine:
                return
            
            # Check if user needs onboarding
            if self.config.enable_onboarding and not self._is_onboarding_complete():
                self._render_onboarding_flow()
                return
            
            # Render personalization sidebar
            self._render_personalization_sidebar(dashboard_variant)
            
            # Apply adaptive interface if enabled
            if self.config.enable_adaptive_interface:
                self._apply_adaptive_interface(dashboard_variant)
                
            # Quick personalization controls
            if st.button("⚙️ Update Preferences", key="update_prefs"):
                st.session_state.onboarding_step = OnboardingStep.PREFERENCES.value
                st.rerun()
            
        except Exception as e:
            self.logger.error(f"Personalization interface rendering error: {e}")
    
    def get_personalized_recommendations(self, context: str = "general") -> List[Dict[str, Any]]:
        """Get personalized recommendations for current user."""
        try:
            if not self.preference_engine:
                return []
            
            recommendations = self.preference_engine.get_personalized_recommendations(
                self.session_id, context
            )
            
            return [self._format_recommendation(rec) for rec in recommendations[:self.config.max_recommendations]]
            
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return []
    
    def get_adaptive_interface_config(self, dashboard_variant: str) -> Dict[str, Any]:
        """Get adaptive interface configuration for current user."""
        try:
            if not self.preference_engine:
                return self._get_default_interface_config(dashboard_variant)
            
            config = self.preference_engine.get_adaptive_interface_config(self.session_id)
            
            # Enhance with variant-specific adaptations
            config.update(self._get_variant_specific_adaptations(dashboard_variant, config))
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to get adaptive interface config: {e}")
            return self._get_default_interface_config(dashboard_variant)
    
    def track_user_interaction(self, interaction_type: str, data: Dict[str, Any] = None):
        """Track user interaction for preference learning."""
        try:
            if not self.preference_engine:
                return
            
            self.preference_engine.track_user_behavior(
                session_id=self.session_id,
                action_type=interaction_type,
                target=str(interaction_type),
                metadata=data or {}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to track user interaction: {e}")

    def apply_user_preferences(self, dashboard_variant: str):
        """Apply user preferences to the specified dashboard variant."""
        try:
            if not self.config.enable_personalization or not self.preference_engine:
                self.logger.debug(f"Personalization not enabled or engine unavailable for {dashboard_variant}")
                return
            
            # Get user preferences
            user_prefs = self._get_user_preferences_summary()
            
            # Apply adaptive interface configuration
            if self.config.enable_adaptive_interface:
                interface_config = self.get_adaptive_interface_config(dashboard_variant)
                self._apply_interface_configuration(interface_config, dashboard_variant)
            
            # Apply personalized styling if enabled
            self._apply_personalized_styling(user_prefs, dashboard_variant)
            
            # Set up personalized recommendations context
            if self.config.enable_recommendations:
                self._setup_recommendations_context(dashboard_variant)
            
            # Track application of preferences
            self.track_user_interaction('preferences_applied', {
                'dashboard_variant': dashboard_variant,
                'preferences_available': bool(user_prefs),
                'timestamp': datetime.now().isoformat()
            })
            
            self.logger.debug(f"✅ User preferences applied to {dashboard_variant}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply user preferences to {dashboard_variant}: {e}")

    def get_personalized_prediction_context(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Get personalized context for prediction display."""
        try:
            context = {
                'user_experience_level': self._get_user_experience_level(),
                'relevant_achievements': self._get_relevant_achievements(home_team, away_team),
                'personalized_insights': self._generate_personalized_insights(home_team, away_team),
                'recommended_focus': self._get_recommended_focus_areas(),
                'prediction_history': self._get_user_prediction_history(home_team, away_team)
            }

            return context

        except Exception as e:
            self.logger.error(f"Failed to get personalized prediction context: {e}")
            return {}

    def integrate_with_prediction_workflow(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate personalization with prediction workflow."""
        try:
            if not self.preference_engine:
                return prediction_data

            # Get user preferences
            user_prefs = self._get_user_preferences_summary()

            # Enhance prediction data with personalization
            enhanced_data = prediction_data.copy()
            enhanced_data.update({
                'personalized_recommendations': self._get_personalized_betting_recommendations(prediction_data, user_prefs),
                'risk_assessment': self._assess_risk_for_user(prediction_data, user_prefs),
                'confidence_adjustment': self._adjust_confidence_for_user(prediction_data, user_prefs),
                'user_context': self.get_personalized_prediction_context(
                    prediction_data.get('home_team', ''),
                    prediction_data.get('away_team', '')
                )
            })

            # Track this prediction for learning
            self._track_prediction_for_learning(enhanced_data)

            return enhanced_data

        except Exception as e:
            self.logger.error(f"Failed to integrate with prediction workflow: {e}")
            return prediction_data

    def render_personalized_dashboard_layout(self, dashboard_variant: str) -> None:
        """Render personalized dashboard layout based on user preferences."""
        try:
            if not self.config.enable_personalization:
                return

            # Get adaptive interface config
            interface_config = self.get_adaptive_interface_config(dashboard_variant)

            # Apply personalized layout
            self._apply_personalized_layout(interface_config)

            # Render personalized widgets
            self._render_personalized_widgets(interface_config)

            # Show achievement progress if enabled
            if interface_config.get('show_achievements', True):
                self._render_achievement_integration()

        except Exception as e:
            self.logger.error(f"Failed to render personalized dashboard layout: {e}")

    def _render_onboarding_flow(self):
        """Render user onboarding flow with progress and fun feedback."""
        steps = [
            (OnboardingStep.WELCOME.value, "Welcome"),
            (OnboardingStep.PREFERENCES.value, "Preferences"),
            (OnboardingStep.TEAMS.value, "Favorite Teams"),
            (OnboardingStep.LEAGUES.value, "Leagues"),
            (OnboardingStep.BETTING_STYLE.value, "Betting Style"),
            (OnboardingStep.COMPLETE.value, "Complete")
        ]
        current_step = st.session_state.get('onboarding_step', OnboardingStep.WELCOME.value)
        step_idx = next((i for i, (val, _) in enumerate(steps) if val == current_step), 0)
        progress = (step_idx) / (len(steps)-1)
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:1rem;'>
            <span style='font-size:1.5rem;'>🚀</span>
            <span style='font-size:1.2rem;font-weight:600;'>Onboarding Progress: <span style='color:#764ba2'>{steps[step_idx][1]}</span></span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(progress)
        st.caption(f"Step {step_idx+1} of {len(steps)}")
        # Fun encouragement
        encouragements = [
            "Let's get started!",
            "Great! Now let's set your preferences.",
            "Pick your favorite teams for tailored insights! ⚽",
            "Choose your leagues for cross-league action! 🌍",
            "Tell us your betting style for smarter tips! 🎯",
            "You're all set! Enjoy your personalized experience! 🎉"
        ]
        st.info(encouragements[step_idx])
        # Render the current step
        if current_step == OnboardingStep.WELCOME.value:
            self._render_welcome_step()
        elif current_step == OnboardingStep.PREFERENCES.value:
            self._render_preferences_step()
        elif current_step == OnboardingStep.TEAMS.value:
            self._render_teams_step()
        elif current_step == OnboardingStep.LEAGUES.value:
            self._render_leagues_step()
        elif current_step == OnboardingStep.BETTING_STYLE.value:
            self._render_betting_style_step()
        elif current_step == OnboardingStep.COMPLETE.value:
            self._complete_onboarding()
    
    def _render_welcome_step(self):
        """Render welcome onboarding step."""
        st.markdown("""\
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        ">
            <h2 style="margin: 0; font-size: 2rem; font-weight: 700;">
                🎯 Personalize Your Experience
            </h2>
            <p style="margin: 1rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                Let's customize GoalDiggers to match your preferences and betting style
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **What we'll personalize:**
        - 🏆 Your favorite teams and leagues
        - 📊 Interface layout and features
        - 🎯 Betting recommendations and insights
        - ⚡ Risk tolerance and betting style
        """)
        
        if st.button("🚀 Get Started", key="onboarding_start", type="primary"):
            st.session_state.onboarding_step = OnboardingStep.PREFERENCES.value
            st.rerun()
    
    def _render_preferences_step(self):
        """Render preferences onboarding step with tooltips and fun UI."""
        st.markdown("#### 🎨 Interface Preferences")
        st.caption("Choose your favorite look and feel for the dashboard!")
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox(
                "Preferred Theme",
                ["Professional", "Dark", "Light", "Colorful"],
                key="onboarding_theme",
                help="Pick a theme that matches your vibe!"
            )
        with col2:
            layout = st.selectbox(
                "Preferred Layout",
                ["Compact", "Detailed", "Minimal", "Advanced"],
                key="onboarding_layout",
                help="How much info do you want on screen?"
            )
        info_density = st.slider(
            "Information Density",
            min_value=1,
            max_value=5,
            value=3,
            help="How much information do you want to see at once? 1 = Minimal, 5 = Max!",
            key="onboarding_density"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", key="prefs_back"):
                st.session_state.onboarding_step = OnboardingStep.WELCOME.value
                st.rerun()
        with col2:
            if st.button("Next →", key="prefs_next", type="primary"):
                self._save_onboarding_preferences({
                    'theme': theme.lower(),
                    'layout': layout.lower(),
                    'info_density': info_density
                })
                st.session_state.onboarding_step = OnboardingStep.TEAMS.value
                st.rerun()
    
    def _render_teams_step(self):
        """Render favorite teams onboarding step with dynamic feedback and summary."""
        st.markdown("#### ⚽ Favorite Teams")
        st.caption("Pick your favorite teams for tailored insights! You can select from multiple leagues.")
        # Team selection by league
        leagues = {
            "Premier League": ["Manchester City", "Arsenal", "Liverpool", "Chelsea", "Manchester United", "Tottenham"],
            "La Liga": ["Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Real Sociedad", "Villarreal"],
            "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Union Berlin"],
            "Serie A": ["AC Milan", "Inter Milan", "Juventus", "Napoli", "AS Roma", "Lazio"],
            "Ligue 1": ["PSG", "Marseille", "Monaco", "Lyon", "Lille", "Rennes"],
            "Eredivisie": ["Ajax", "PSV", "Feyenoord", "AZ Alkmaar", "FC Utrecht", "Vitesse"]
        }
        selected_teams = []
        for league, teams in leagues.items():
            with st.expander(f"🏆 {league}"):
                league_teams = st.multiselect(
                    f"Select teams from {league}",
                    teams,
                    key=f"onboarding_teams_{league.replace(' ', '_')}"
                )
                selected_teams.extend(league_teams)
        # Fun dynamic feedback
        if len(selected_teams) == 0:
            st.warning("No teams selected yet. Start picking your favorites! ⚽")
        elif len(selected_teams) == 1:
            st.success(f"{selected_teams[0]} is a great pick! 🎉")
        elif len(selected_teams) > 5:
            st.info("Wow, you have a broad taste! 🌍")
        # Summary card with team names (could add logos if available)
        if selected_teams:
            st.markdown("<div style='background:#f6f6ff;padding:1rem;border-radius:10px;margin:1rem 0;'>" +
                        "<b>Your picks:</b> " + ", ".join(selected_teams) + "</div>", unsafe_allow_html=True)
        # Randomize button
        if st.button("🎲 Randomize Teams", key="randomize_teams"):
            import random
            all_teams = [team for teams in leagues.values() for team in teams]
            random_picks = random.sample(all_teams, 3)
            for league, teams in leagues.items():
                st.session_state[f"onboarding_teams_{league.replace(' ', '_')}"] = [t for t in random_picks if t in teams]
            st.rerun()
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", key="teams_back"):
                st.session_state.onboarding_step = OnboardingStep.PREFERENCES.value
                st.rerun()
        with col2:
            if st.button("Next →", key="teams_next", type="primary"):
                self._save_onboarding_teams(selected_teams)
                st.session_state.onboarding_step = OnboardingStep.LEAGUES.value
                st.rerun()
    
    def _render_leagues_step(self):
        """Render favorite leagues onboarding step with drag-and-drop and playful tooltips."""
        st.markdown("#### 🌍 League Preferences")
        leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Eredivisie"]
        st.markdown("**Rank leagues by your interest (drag to reorder):** ", unsafe_allow_html=True)
        st.caption("Tip: Your top leagues will be prioritized in recommendations!")
        # Simulate drag-and-drop with selectbox order (Streamlit limitation)
        favorite_leagues = st.multiselect(
            "Select your favorite leagues",
            leagues,
            default=leagues[:3],
            key="onboarding_leagues",
            help="Choose the leagues you follow most."
        )
        # Show league badges/icons (text-based for now)
        if favorite_leagues:
            st.markdown("<div style='margin:0.5rem 0;'>" +
                        " ".join([f"<span style='padding:0.3em 0.7em;background:#e0e7ff;border-radius:8px;margin-right:0.3em;'>{l}</span>" for l in favorite_leagues]) +
                        "</div>", unsafe_allow_html=True)
        # Playful tooltip/info
        with st.expander("ℹ️ Why rank leagues?"):
            st.write("Ranking helps us tailor your experience and show you the most relevant matches and stats!")
        # Cross-league interest
        cross_league_interest = st.slider(
            "Interest in cross-league matches",
            min_value=1,
            max_value=5,
            value=4,
            help="How interested are you in matches between teams from different leagues? 1 = Not at all, 5 = Love it!",
            key="onboarding_cross_league"
        )
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", key="leagues_back"):
                st.session_state.onboarding_step = OnboardingStep.TEAMS.value
                st.rerun()
        with col2:
            if st.button("Next →", key="leagues_next", type="primary"):
                self._save_onboarding_leagues(favorite_leagues, cross_league_interest)
                st.session_state.onboarding_step = OnboardingStep.BETTING_STYLE.value
                st.rerun()
    
    def _render_betting_style_step(self):
        """Render betting style onboarding step with quiz and live preview."""
        st.markdown("#### 🎯 Betting Style & Risk Tolerance")
        st.caption("Let's find your betting personality! Answer a quick quiz:")
        # Quiz-style question
        quiz_q = st.radio(
            "When your team is losing at halftime, you...",
            [
                "Play it safe and avoid risky bets",
                "Look for value in comeback odds",
                "Double down for a big win!"
            ],
            key="onboarding_quiz1"
        )
        # Map quiz answer to style
        quiz_map = {
            0: "Conservative",
            1: "Balanced",
            2: "Aggressive"
        }
        betting_style_idx = [
            "Play it safe and avoid risky bets",
            "Look for value in comeback odds",
            "Double down for a big win!"
        ].index(quiz_q)
        betting_style = quiz_map[betting_style_idx]
        # Fun fact for each style
        fun_facts = {
            "Conservative": "Did you know? Conservative bettors often have the longest streaks!",
            "Balanced": "Balanced bettors get the best of both worlds: fun and safety.",
            "Aggressive": "Aggressive bettors chase the thrill and sometimes hit big jackpots!"
        }
        st.info(fun_facts[betting_style])
        # Risk tolerance
        risk_tolerance = st.slider(
            "Risk Tolerance",
            min_value=0.0,
            max_value=1.0,
            value=0.5 if betting_style=="Balanced" else (0.2 if betting_style=="Conservative" else 0.8),
            step=0.1,
            help="0.0 = Very Conservative, 1.0 = High Risk",
            key="onboarding_risk_tolerance"
        )
        # Preferred markets
        preferred_markets = st.multiselect(
            "Preferred Betting Markets",
            ["Match Result", "Over/Under Goals", "Both Teams to Score", "Handicap", "Correct Score"],
            default=["Match Result", "Over/Under Goals"],
            key="onboarding_markets"
        )
        # Experience level
        experience_level = st.selectbox(
            "Betting Experience",
            ["Beginner", "Intermediate", "Advanced", "Expert"],
            index=1,
            key="onboarding_experience"
        )
        # Live preview
        st.markdown(f"<div style='background:#f0fff4;padding:1rem;border-radius:10px;margin:1rem 0;'>"
                    f"<b>Preview:</b> <br>Style: <b>{betting_style}</b> | Risk: <b>{risk_tolerance:.1f}</b> | Experience: <b>{experience_level}</b>"
                    "</div>", unsafe_allow_html=True)
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back", key="betting_back"):
                st.session_state.onboarding_step = OnboardingStep.LEAGUES.value
                st.rerun()
        with col2:
            if st.button("Complete Setup 🎉", key="betting_complete", type="primary"):
                self._save_onboarding_betting_style({
                    'betting_style': betting_style.lower(),
                    'risk_tolerance': risk_tolerance,
                    'preferred_markets': preferred_markets,
                    'experience_level': experience_level.lower()
                })
                st.session_state.onboarding_step = OnboardingStep.COMPLETE.value
                st.rerun()
    
    def _complete_onboarding(self):
        """Complete onboarding process with celebration and summary."""
        # Celebratory animation (confetti)
        st.balloons()
        st.success("🎉 Personalization setup complete!")
        # Personalized summary card
        user_prefs = self._get_user_preferences_summary()
        st.markdown("""
        <div style='background:linear-gradient(90deg,#e0c3fc 0%,#8ec5fc 100%);padding:1.5rem;border-radius:15px;margin:1rem 0;'>
        <h3 style='margin:0 0 0.5em 0;'>Your GoalDiggers Experience is Ready!</h3>
        <ul style='font-size:1.1em;'>
        <li>✅ <b>Style:</b> {style}</li>
        <li>✅ <b>Risk:</b> {risk}</li>
        <li>✅ <b>Favorite Teams:</b> {teams}</li>
        <li>✅ <b>Leagues:</b> {leagues}</li>
        </ul>
        <div style='margin-top:1em;font-size:1.1em;'>
        <b>Motivation:</b> {motivation}
        </div>
        </div>
        """.format(
            style=user_prefs.get('betting_style','Balanced').title(),
            risk=f"{user_prefs.get('risk_tolerance',0.5):.1f}",
            teams=", ".join(user_prefs.get('favorite_teams',[]) or ["None"]),
            leagues=", ".join(user_prefs.get('favorite_leagues',[]) or ["None"]),
            motivation="You're ready to conquer the pitch with smart picks and bold moves! ⚽"
        ), unsafe_allow_html=True)
        # Mark onboarding as complete
        st.session_state.onboarding_complete = True
        # Share setup button (copy to clipboard)
        st.markdown("<button onclick=\"navigator.clipboard.writeText('I just personalized my GoalDiggers dashboard!')\" style='margin-top:1em;padding:0.5em 1em;border:none;background:#764ba2;color:white;border-radius:8px;cursor:pointer;'>Share your setup 🚀</button>", unsafe_allow_html=True)
        if st.button("🚀 Start Using GoalDiggers", key="onboarding_finish", type="primary"):
            st.rerun()
    
    def _render_personalization_sidebar(self, dashboard_variant: str):
        """Render personalization sidebar."""
        with st.sidebar:
            st.markdown("### 👤 Personalization")
            
            # User preferences summary
            user_prefs = self._get_user_preferences_summary()
            if user_prefs:
                st.markdown("#### 🎯 Your Preferences")
                st.markdown(f"**Style**: {user_prefs.get('betting_style', 'Balanced').title()}")
                st.markdown(f"**Risk**: {user_prefs.get('risk_tolerance', 0.5):.1f}")
                
                # Favorite teams
                fav_teams = user_prefs.get('favorite_teams', [])
                if fav_teams:
                    st.markdown("**Favorite Teams**:")
                    for team in fav_teams[:3]:
                        st.markdown(f"• {team}", unsafe_allow_html=True)
            
            # Personalized recommendations
            if self.config.enable_recommendations:
                recommendations = self.get_personalized_recommendations("sidebar")
                if recommendations:
                    st.markdown("#### 💡 Recommendations")
                    for rec in recommendations[:3]:
                        st.markdown(f"- {rec}", unsafe_allow_html=True)

    def _apply_interface_configuration(self, interface_config: Dict[str, Any], dashboard_variant: str):
        """Apply interface configuration to the dashboard."""
        try:
            # Apply theme-based styling
            theme = interface_config.get('theme', 'professional')
            if theme == 'dark':
                self._apply_dark_theme()
            elif theme == 'colorful':
                self._apply_colorful_theme()
            
            # Store configuration in session state for dashboard access
            st.session_state[f'{dashboard_variant}_interface_config'] = interface_config
            
            self.logger.debug(f"Interface configuration applied for {dashboard_variant}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply interface configuration: {e}")

    def _apply_personalized_styling(self, user_prefs: Dict[str, Any], dashboard_variant: str):
        """Apply personalized styling based on user preferences."""
        try:
            # Apply styling based on user preferences
            betting_style = user_prefs.get('betting_style', 'balanced')
            risk_tolerance = user_prefs.get('risk_tolerance', 0.5)
            
            # Store styling preferences in session state
            st.session_state[f'{dashboard_variant}_personalized_styling'] = {
                'betting_style': betting_style,
                'risk_tolerance': risk_tolerance,
                'emphasis_color': self._get_style_color(betting_style),
                'risk_indicators': self._get_risk_indicators(risk_tolerance)
            }
            
            self.logger.debug(f"Personalized styling applied for {dashboard_variant}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply personalized styling: {e}")

    def _setup_recommendations_context(self, dashboard_variant: str):
        """Set up recommendations context for the dashboard."""
        try:
            # Store recommendations context in session state
            st.session_state[f'{dashboard_variant}_recommendations_enabled'] = True
            st.session_state[f'{dashboard_variant}_max_recommendations'] = self.config.max_recommendations
            
            self.logger.debug(f"Recommendations context set up for {dashboard_variant}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup recommendations context: {e}")

    def _get_style_color(self, betting_style: str) -> str:
        """Get emphasis color based on betting style."""
        color_map = {
            'conservative': '#28a745',  # Green
            'balanced': '#007bff',      # Blue
            'aggressive': '#dc3545'     # Red
        }
        return color_map.get(betting_style, '#007bff')

    def _get_risk_indicators(self, risk_tolerance: float) -> Dict[str, bool]:
        """Get risk indicator settings based on risk tolerance."""
        return {
            'show_risk_warnings': risk_tolerance < 0.4,
            'highlight_safe_bets': risk_tolerance < 0.6,
            'show_aggressive_opportunities': risk_tolerance > 0.7
        }
    
    def _apply_adaptive_interface(self, dashboard_variant: str):
        """Apply adaptive interface based on user preferences."""
        try:
            config = self.get_adaptive_interface_config(dashboard_variant)
            
            # Apply theme adaptations
            if config.get('theme') == 'dark':
                self._apply_dark_theme()
            elif config.get('theme') == 'colorful':
                self._apply_colorful_theme()
            
            # Apply layout adaptations
            layout = config.get('layout', 'detailed')
            if layout == 'compact':
                st.session_state.layout_mode = 'compact'
            elif layout == 'minimal':
                st.session_state.layout_mode = 'minimal'
            
        except Exception as e:
            self.logger.error(f"Failed to apply adaptive interface: {e}")
    
    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        st.markdown("""
        <style>
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _apply_colorful_theme(self):
        """Apply colorful theme styling."""
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _get_or_create_session_id(self) -> str:
        """Get or create session ID for user tracking."""
        if 'user_session_id' not in st.session_state:
            st.session_state.user_session_id = str(uuid.uuid4())
        return st.session_state.user_session_id
    
    def _is_onboarding_complete(self) -> bool:
        """Check if user has completed onboarding."""
        return st.session_state.get('onboarding_complete', False)
    
    def _save_onboarding_preferences(self, preferences: Dict[str, Any]):
        """Save onboarding preferences."""
        if not self.preference_engine:
            return
        try:
            for key, value in preferences.items():
                self.preference_engine.update_user_preference(
                    self.session_id,
                    preference_key=key,
                    value=value
                )
        except Exception as e:
            self.logger.error(f"Failed to save onboarding preferences: {e}")
    
    def _save_onboarding_teams(self, teams: List[str]):
        """Save favorite teams from onboarding."""
        if not self.preference_engine:
            return
        try:
            self.preference_engine.update_user_preference(
                self.session_id,
                preference_key='favorite_teams',
                value=teams
            )
        except Exception as e:
            self.logger.error(f"Failed to save favorite teams: {e}")
    
    def _save_onboarding_leagues(self, leagues: List[str], cross_league_interest: int):
        """Save league preferences from onboarding."""
        if not self.preference_engine:
            return
        try:
            self.preference_engine.update_user_preference(
                self.session_id,
                preference_key='favorite_leagues',
                value=leagues
            )
            self.preference_engine.update_user_preference(
                self.session_id,
                preference_key='cross_league_interest',
                value=cross_league_interest / 5.0
            )
        except Exception as e:
            self.logger.error(f"Failed to save league preferences: {e}")
    
    def _save_onboarding_betting_style(self, betting_data: Dict[str, Any]):
        """Save betting style from onboarding."""
        if not self.preference_engine:
            return
        try:
            self.preference_engine.update_user_preference(
                self.session_id,
                preference_key='betting_style',
                value=betting_data['betting_style']
            )
            self.preference_engine.update_user_preference(
                self.session_id,
                preference_key='risk_tolerance',
                value=betting_data['risk_tolerance']
            )
            self.preference_engine.update_user_preference(
                self.session_id,
                preference_key='preferred_markets',
                value=betting_data['preferred_markets']
            )
        except Exception as e:
            self.logger.error(f"Failed to save betting style: {e}")
    
    def _get_user_preferences_summary(self) -> Dict[str, Any]:
        """Get user preferences summary."""
        if not self.preference_engine:
            return {}
        
        try:
            user_prefs = self.preference_engine.get_user_preferences(self.session_id)
            if user_prefs:
                return {
                    'betting_style': user_prefs.betting_style,
                    'risk_tolerance': user_prefs.risk_tolerance,
                    'favorite_teams': user_prefs.favorite_teams,
                    'favorite_leagues': user_prefs.favorite_leagues
                }
        except Exception as e:
            self.logger.error(f"Failed to get user preferences: {e}")
        
        return {}
    
    def _format_recommendation(self, rec) -> Dict[str, Any]:
        """Format recommendation for display."""
        return {
            'title': rec.title if hasattr(rec, 'title') else str(rec),
            'description': rec.description if hasattr(rec, 'description') else '',
            'confidence': rec.confidence if hasattr(rec, 'confidence') else 0.8,
            'type': rec.recommendation_type if hasattr(rec, 'recommendation_type') else 'general'
        }
    
    def _get_default_interface_config(self, dashboard_variant: str) -> Dict[str, Any]:
        """Get default interface configuration."""
        return {
            'theme': 'professional',
            'layout': 'detailed',
            'default_teams': [],
            'default_leagues': ['Premier League', 'La Liga', 'Bundesliga'],
            'risk_indicators': {
                'show_risk_warnings': True,
                'highlight_safe_bets': False,
                'show_aggressive_opportunities': False
            }
        }
    
    def _get_variant_specific_adaptations(self, dashboard_variant: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get variant-specific interface adaptations."""
        adaptations = {}
        
        if dashboard_variant == 'premium_ui':
            adaptations.update({
                'enhanced_styling': True,
                'premium_features': True
            })
        elif dashboard_variant == 'interactive_cross_league':
            adaptations.update({
                'gamification_enabled': True,
                'cross_league_focus': True
            })
        elif dashboard_variant == 'optimized_premium':
            adaptations.update({
                'performance_mode': True,
                'minimal_animations': True
            })
        
        return adaptations

    def _get_user_experience_level(self) -> str:
        """Get user experience level based on achievements and history."""
        if not self.achievement_system:
            return "intermediate"

        try:
            # Mock implementation - in real system, get from achievement system
            total_predictions = len(self.user_prediction_history)

            if total_predictions < 10:
                return "beginner"
            elif total_predictions < 50:
                return "intermediate"
            elif total_predictions < 200:
                return "advanced"
            else:
                return "expert"

        except Exception:
            return "intermediate"

    def _get_relevant_achievements(self, home_team: str, away_team: str) -> List[Dict[str, Any]]:
        """Get achievements relevant to current prediction."""
        if not self.achievement_system:
            return []

        try:
            # Mock relevant achievements
            return [
                {
                    'name': 'Cross-League Expert',
                    'progress': 7,
                    'total': 10,
                    'description': 'Make predictions across different leagues',
                    'relevance': 'high' if home_team and away_team else 'low'
                },
                {
                    'name': 'Prediction Streak',
                    'progress': 15,
                    'total': 20,
                    'description': 'Maintain a prediction streak',
                    'relevance': 'medium'
                }
            ]

        except Exception:
            return []

    def _generate_personalized_insights(self, home_team: str, away_team: str) -> List[str]:
        """Generate personalized insights based on user history and preferences."""
        insights = []

        try:
            user_prefs = self._get_user_preferences_summary()
            experience_level = self._get_user_experience_level()

            # Experience-based insights
            if experience_level == "beginner":
                insights.append("💡 Consider the home advantage factor in your prediction")
                insights.append("📊 Look at recent form - teams' last 5 matches matter")
            elif experience_level == "advanced":
                insights.append("🔍 Check for key player injuries that might affect the outcome")
                insights.append("📈 Consider the historical head-to-head record between these teams")

            # Preference-based insights
            betting_style = user_prefs.get('betting_style', 'balanced')
            if betting_style == 'conservative':
                insights.append("🛡️ This prediction aligns with your conservative betting style")
            elif betting_style == 'aggressive':
                insights.append("⚡ Consider the value bet opportunities in this match")

            return insights[:3]  # Limit to 3 insights

        except Exception as e:
            self.logger.error(f"Failed to generate personalized insights: {e}")
            return []

    def _get_recommended_focus_areas(self) -> List[str]:
        """Get recommended focus areas for user improvement."""
        try:
            experience_level = self._get_user_experience_level()

            if experience_level == "beginner":
                return ["Understanding odds", "Team form analysis", "Home advantage"]
            elif experience_level == "intermediate":
                return ["Advanced statistics", "Cross-league analysis", "Value betting"]
            else:
                return ["Market timing", "Advanced modeling", "Risk management"]

        except Exception:
            return ["General prediction skills"]

    def _get_user_prediction_history(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Get user's prediction history for these teams."""
        try:
            # Mock implementation - in real system, query user's history
            return {
                'total_predictions': len(self.user_prediction_history),
                'accuracy_rate': 0.67,
                'favorite_leagues': ['Premier League', 'La Liga'],
                'recent_performance': 'improving'
            }

        except Exception:
            return {}

# Factory functions for different personalization configurations
def create_enhanced_personalization_config() -> PersonalizationConfig:
    """Create enhanced personalization configuration."""
    return PersonalizationConfig(
        enable_personalization=True,
        enable_adaptive_interface=True,
        enable_recommendations=True,
        enable_onboarding=True,
        enable_risk_analysis=True,
        personalization_level=PersonalizationLevel.ENHANCED,
        max_recommendations=5
    )

def create_basic_personalization_config() -> PersonalizationConfig:
    """Create basic personalization configuration."""
    return PersonalizationConfig(
        enable_personalization=True,
        enable_adaptive_interface=False,
        enable_recommendations=True,
        enable_onboarding=True,
        enable_risk_analysis=False,
        personalization_level=PersonalizationLevel.BASIC,
        max_recommendations=3
    )
