import threading
#!/usr/bin/env python3
"""
Enhanced Match Selector Component - GoalDiggers Platform

Provides an interactive, responsive, and aesthetically cohesive user experience for:
- Team selection with intelligent search
- Real-time match predictions
- Historical head-to-head analysis
- Interactive prediction customization
"""

import html
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import streamlit as st

from dashboard.components.unified_design_system import get_unified_design_system
from database.db_manager import DatabaseManager
from database.schema import League, Match, Team
from models.enhanced_real_data_predictor import EnhancedRealDataPredictor

# CONSISTENCY FIX: Singleton predictor to avoid re-initialization
_predictor_singleton = None
_predictor_lock = threading.Lock()


def get_singleton_predictor() -> EnhancedRealDataPredictor:
    """Return a lazily constructed predictor shared across UI callbacks."""
    global _predictor_singleton
    if _predictor_singleton is None:
        with _predictor_lock:
            if _predictor_singleton is None:
                # Instantiate once to avoid recursive re-entry under Streamlit reruns
                _predictor_singleton = EnhancedRealDataPredictor()
                logger.info("✅ Initialized singleton predictor")
    return _predictor_singleton

from utils.team_assets_registry import get_team_assets_registry
from utils.team_data_enhancer import team_enhancer

logger = logging.getLogger(__name__)


def format_team_option(team_name: str) -> str:
    """Format team option with flag emoji if enabled."""
    # Check if team flags should be displayed
    if os.getenv('GOALDIGGERS_TEAM_FLAGS', 'false').lower() == 'true':
        try:
            # Get enhanced team data with flag emoji
            enhanced_data = team_enhancer.get_enhanced_team_data(team_name)
            flag = enhanced_data.get('flag', '⚽')
            display_name = enhanced_data.get('display_name', team_name)
            return f"{flag} {display_name}"
        except Exception as e:
            logger.warning(f"Could not enhance team {team_name}: {e}")
            return team_name
    return team_name


class EnhancedMatchSelector:
    """Enhanced interactive match selector with real-time predictions."""

    LEAGUE_CACHE_KEY = "match_selector_leagues_cache"
    LEAGUE_CACHE_TTL = timedelta(minutes=10)

    def __init__(self):
        """Initialize the enhanced match selector."""
        self.db_manager = DatabaseManager()
        self.logger = logger
        self._predictor: EnhancedRealDataPredictor | None = None
        self.design_system = get_unified_design_system()
        self._last_league_fetch_meta: dict[str, Any] = {}

        if "match_selector_css_injected" not in st.session_state:
            self.design_system.inject_unified_css("premium")
            st.session_state.match_selector_css_injected = True

    @property
    def predictor(self) -> EnhancedRealDataPredictor:
        """Lazily initialize and return the enhanced predictor."""
        if self._predictor is None:
            self.logger.info("Lazy-loading EnhancedRealDataPredictor for match selector")
            self._predictor = get_singleton_predictor()
        return self._predictor

    @predictor.setter
    def predictor(self, value: EnhancedRealDataPredictor) -> None:
        self._predictor = value

    def _render_header(self) -> None:
        """Render the primary hero header for the match selector."""
        self.design_system.create_unified_header(
            "⚽ Match Prediction Studio",
            "AI-powered insights with resilient data pipelines",
        )
        st.caption("Select teams, fine-tune assumptions, and publish predictions in seconds.")

    def _render_connection_status_banner(self, connection_info: dict[str, Any]) -> None:
        """Show a styled connection status banner using unified design chips."""
        using_fallback = connection_info.get("using_fallback", False)
        masked_uri = connection_info.get("masked_active_uri", "N/A")
        global_info = connection_info.get("global_fallback") or {}

        if using_fallback:
            icon = "🟠"
            label = "Fallback Mode"
            status_style = "background:rgba(253,126,20,0.16);color:#c35a05;"
        else:
            icon = "🟢"
            label = "Primary Database"
            status_style = "background:rgba(40,167,69,0.16);color:#1f4e79;"

        # Sanitize and truncate masked_uri to prevent display overflow
        display_uri = masked_uri
        if len(display_uri) > 60:
            display_uri = display_uri[:57] + "..."

        chips = [
            f'<span class="gd-chip" style="{status_style}">{icon} {label}</span>',
            f'<span class="gd-chip" style="background:rgba(31,78,121,0.08);color:#1f2a3d;">🔗 {html.escape(display_uri)}</span>',
        ]

        # Only show reason in expander to avoid cluttering the UI with long error messages
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:0.75rem;">' + "".join(chips) + "</div>",
            unsafe_allow_html=True,
        )

        if using_fallback:
            st.info(
                f"🛡️ Using resilient SQLite fallback. Connection: `{display_uri}`"
            )
            # Show detailed reason in expander
            reason = connection_info.get("fallback_reason") or "Primary database unavailable"
            if len(reason) > 100:
                with st.expander("📋 View connection details"):
                    st.code(reason, language="text")
        elif global_info:
            st.info(
                f"📦 Using cached fallback connection. Active: `{display_uri}`"
            )
        else:
            st.success("✅ Primary datastore connection verified and healthy.")

    def _format_relative_time(self, timestamp: datetime) -> str:
        """Return a small human-readable relative time string."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        delta = now_utc - timestamp
        seconds = max(int(delta.total_seconds()), 0)
        if seconds < 60:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = hours // 24
        return f"{days} day{'s' if days != 1 else ''} ago"

    def _render_data_refresh_meta(self, placeholder: Any | None = None) -> None:
        """Display metadata about when the team directory was last refreshed."""
        target = placeholder if placeholder is not None else st
        meta = self._last_league_fetch_meta or {}
        timestamp = meta.get("timestamp")
        if isinstance(timestamp, datetime):
            relative = self._format_relative_time(timestamp)
            source = "SQLite fallback" if meta.get("using_fallback") else "primary database"
            cache_minutes = int(self.LEAGUE_CACHE_TTL.total_seconds() // 60)
            caption_target = target if hasattr(target, "caption") else st
            caption_target.caption(f"Team directory refreshed {relative} • Source: {source} • Cache window {cache_minutes} min")
        else:
            caption_target = target if hasattr(target, "caption") else st
            caption_target.caption("Team directory will refresh automatically when outdated (≈10 min cache window).")
        
    def render_enhanced_match_selector(self) -> dict | None:
        """
        Render the complete enhanced match selector interface.
        
        Returns:
            Dict containing prediction results if teams are selected, None otherwise
        """
        self._render_header()
        
        # Initialize session state
        self._init_session_state()
        
        # Create main layout
        with st.container():
            # Team selection interface
            team_selection_result = self._render_team_selection_interface()
            
            if team_selection_result and team_selection_result['ready_for_prediction']:
                # Prediction customization panel
                prediction_config = self._render_prediction_customization_panel()
                
                # Generate prediction button and results
                return self._render_prediction_interface(
                    team_selection_result, prediction_config
                )
        
        return None
    
    def _init_session_state(self):
        """Initialize session state variables."""
        if 'enhanced_home_team' not in st.session_state:
            st.session_state.enhanced_home_team = None
        if 'enhanced_away_team' not in st.session_state:
            st.session_state.enhanced_away_team = None
        if 'selected_league' not in st.session_state:
            st.session_state.selected_league = 'All Leagues'
        if 'cross_league_enabled' not in st.session_state:
            st.session_state.cross_league_enabled = False
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
    
    def _render_team_selection_interface(self) -> dict | None:
        """Render the intelligent team selection interface."""
        st.markdown("### 🎯 Team Selection")

        connection_info = self.db_manager.connection_info()
        self._render_connection_status_banner(connection_info)

        meta_col, action_col = st.columns([5, 1])
        meta_placeholder = meta_col.empty()
        self._render_data_refresh_meta(meta_placeholder)
        with action_col:
            if st.button("🔄 Refresh teams", key="refresh_league_data", help="Invalidate cache and fetch the latest team directory"):
                st.session_state.pop(self.LEAGUE_CACHE_KEY, None)
                self._last_league_fetch_meta = {}
                st.rerun()

        use_spinner = not self._has_valid_league_cache()
        if use_spinner:
            with st.spinner("Loading team directory..."):
                leagues_data = self._get_leagues_and_teams()
        else:
            leagues_data = self._get_leagues_and_teams()

        if not leagues_data:
            st.error("❌ Unable to load team data. Please check your database connection.")
            return None

        self._render_data_refresh_meta(meta_placeholder)
        
        # League selection and cross-league toggle
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_league = st.selectbox(
                "Select League",
                options=['All Leagues'] + list(leagues_data.keys()),
                index=0 if st.session_state.selected_league == 'All Leagues' 
                else list(leagues_data.keys()).index(st.session_state.selected_league) + 1,
                key="league_selector"
            )
        
        with col2:
            # Use checkbox for compatibility (st.toggle is not a default Streamlit API)
            cross_league_enabled = st.checkbox(
                "Cross-League",
                value=st.session_state.cross_league_enabled,
                help="Enable matches between teams from different leagues"
            )
        
        # Update session state
        st.session_state.selected_league = selected_league
        st.session_state.cross_league_enabled = cross_league_enabled
        
        # Team selection interface
        return self._render_team_selection_widgets(leagues_data, selected_league, cross_league_enabled)
    
    def _render_team_selection_widgets(self, leagues_data: dict, selected_league: str, 
                                     cross_league_enabled: bool) -> dict | None:
        """Render team selection widgets."""
        # Get teams based on league selection
        if selected_league == 'All Leagues':
            # Use set to eliminate duplicates, then sort
            all_teams_set = set()
            for league_teams in leagues_data.values():
                all_teams_set.update(league_teams)
            all_teams = sorted(list(all_teams_set))
        else:
            # Ensure no duplicates in individual league teams
            league_teams = leagues_data.get(selected_league, [])
            all_teams = sorted(list(set(league_teams)))
        
        if not all_teams:
            st.warning("⚠️ No teams available for the selected league.")
            return None
        
        # Team selection columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏠 Home Team")
            # Build enhanced team options with flags for display
            home_display_options = ["🔍 Select home team..."] + [
                format_team_option(team)
                for team in all_teams
            ]
            home_selection_idx = st.selectbox(
                "Choose home team",
                options=range(len(home_display_options)),
                format_func=lambda idx: home_display_options[idx],
                key="home_team_selector",
                label_visibility="collapsed",
                help="Choose the home team with home field advantage"
            )
            home_team = all_teams[home_selection_idx - 1] if home_selection_idx > 0 else None
        
        with col2:
            st.markdown("#### ✈️ Away Team")
            # Filter away teams (exclude selected home team)
            away_team_options = [team for team in all_teams if team != home_team]
            away_display_options = ["🔍 Select away team..."] + [
                format_team_option(team)
                for team in away_team_options
            ]
            away_selection_idx = st.selectbox(
                "Choose away team",
                options=range(len(away_display_options)),
                format_func=lambda idx: away_display_options[idx],
                key="away_team_selector",
                label_visibility="collapsed",
                help="Choose the visiting team"
            )
            away_team = away_team_options[away_selection_idx - 1] if away_selection_idx > 0 else None
        
        # Update session state
        st.session_state.enhanced_home_team = home_team
        st.session_state.enhanced_away_team = away_team
        
        # Validation and ready state
        ready_for_prediction = bool(home_team and away_team and home_team != away_team)

        action_cols = st.columns([2, 1, 2])
        with action_cols[1]:
            if st.button("⇄ Swap teams", key="swap_match_selector", use_container_width=True, disabled=not (home_team and away_team)):
                st.session_state.home_team_selector = away_team
                st.session_state.away_team_selector = home_team
                st.rerun()

        if selected_league == 'All Leagues' and not cross_league_enabled:
            st.caption("Tip: enable Cross-League to compare teams from different competitions.")
        
        if home_team and away_team:
            if home_team == away_team:
                st.error("❌ Please select different teams for home and away.")
            else:
                # Show match preview
                self._render_match_preview(home_team, away_team)
                
                # Display team statistics comparison
                self._render_team_comparison(home_team, away_team)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'selected_league': selected_league,
            'cross_league_enabled': cross_league_enabled,
            'ready_for_prediction': ready_for_prediction
        }
    
    def _render_match_preview(self, home_team: str, away_team: str):
        """Render enhanced match preview using unified design system."""
        uds = self.design_system
        current_date = datetime.now()
        match_date = current_date + timedelta(days=1)
        home_enhanced = team_enhancer.get_enhanced_team_data(home_team)
        away_enhanced = team_enhancer.get_enhanced_team_data(away_team)
        home_meta = team_enhancer.get_team_enhancement(home_team)
        away_meta = team_enhancer.get_team_enhancement(away_team)
        match_title = team_enhancer.create_match_title(
            home_team,
            away_team,
            league=home_enhanced.get('league') if not home_enhanced.get('league_mismatch') else None,
            match_date=match_date,
        )
        hero_gradient = (
            f"linear-gradient(135deg, {home_meta['primary_color']}CC 0%, {away_meta['primary_color']}B3 100%)"
        )
        venue_name = home_enhanced.get('venue') or f"{home_enhanced.get('display_name', home_team)} Stadium"
        def card_content():
            hero_html = f"""
            <div style="background:{hero_gradient};padding:18px 24px;border-radius:22px;margin-bottom:18px;color:#0f172a;backdrop-filter: blur(12px);">
              <div style="font-size:0.85rem;opacity:0.75;letter-spacing:0.08em;text-transform:uppercase;">Featured Match</div>
              <div style="font-size:1.5rem;font-weight:700;margin-top:4px;">{match_title}</div>
              <div style="margin-top:8px;font-size:0.95rem;opacity:0.85;display:flex;gap:12px;flex-wrap:wrap;align-items:center;">
                <span>{home_meta['country_flag']} {home_meta.get('country', 'Unknown')}</span>
                <span>⚔️</span>
                <span>{away_meta['country_flag']} {away_meta.get('country', 'Unknown')}</span>
              </div>
            </div>
            """
            st.markdown(hero_html, unsafe_allow_html=True)
            st.markdown("### 🎫 Enhanced Match Preview")
            cols = st.columns([2, 1, 2])
            with cols[0]:
                st.markdown(f"<div style='text-align:center'><span style='font-size:2em'>{home_meta['flag']}</span><br><b>{home_meta['display_name']}</b><br><span style='font-size:0.9em;opacity:0.8'>🏠 Home Advantage</span></div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"<div style='text-align:center'><b>VS</b><br><span style='font-size:0.9em;opacity:0.8'>{match_date.strftime('%d/%m/%Y')}</span></div>", unsafe_allow_html=True)
            with cols[2]:
                st.markdown(f"<div style='text-align:center'><span style='font-size:2em'>{away_meta['flag']}</span><br><b>{away_meta['display_name']}</b><br><span style='font-size:0.9em;opacity:0.8'>✈️ Away Challenge</span></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='margin-top:1em;text-align:center;'>"
                f"<span style='background:rgba(15, 23, 42, 0.08);padding:6px 16px;border-radius:16px;margin:0 6px;'>⏰ {match_date.strftime('%H:%M')}</span>"
                f"<span style='background:rgba(15, 23, 42, 0.08);padding:6px 16px;border-radius:16px;margin:0 6px;'>🏟️ {venue_name}</span>"
                "<span style='background:rgba(15, 23, 42, 0.08);padding:6px 16px;border-radius:16px;margin:0 6px;'>🎯 AI Analysis Ready</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        uds.create_unified_card(card_content, card_class="goaldiggers-card")
    
    def _render_team_comparison(self, home_team: str, away_team: str):
        """Render quick team comparison stats with enhanced metadata."""
        try:
            home_meta = team_enhancer.get_team_enhancement(home_team)
            away_meta = team_enhancer.get_team_enhancement(away_team)
            home_stats = self._get_team_basic_stats(home_team)
            away_stats = self._get_team_basic_stats(away_team)
            st.markdown("### 📊 Quick Team Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    f"{home_meta['flag']} {home_team} Recent Form",
                    f"{home_stats.get('recent_form', 'N/A')}",
                    delta=f"{home_stats.get('form_trend', 0):.1f}%",
                    help=f"Primary color: {home_meta['primary_color']}"
                )
            with col2:
                st.metric(
                    "Head-to-Head",
                    "Loading...",
                    help="Historical matchup record"
                )
            with col3:
                st.metric(
                    f"{away_meta['flag']} {away_team} Recent Form",
                    f"{away_stats.get('recent_form', 'N/A')}",
                    delta=f"{away_stats.get('form_trend', 0):.1f}%",
                    help=f"Primary color: {away_meta['primary_color']}"
                )
        except Exception as e:
            self.logger.warning(f"Could not load team comparison: {e}")
    
    def _render_prediction_customization_panel(self) -> dict:
        """Render prediction customization options."""
        st.markdown("### ⚙️ Prediction Settings")
        
        with st.expander("🎛️ Customize Prediction Parameters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                use_real_data = st.checkbox(
                    "Use Real Data",
                    value=True,
                    help="Use real team statistics and form data"
                )
                
                include_weather = st.checkbox(
                    "Weather Impact",
                    value=False,
                    help="Consider weather conditions (premium feature)"
                )
            
            with col2:
                prediction_confidence = st.slider(
                    "Minimum Confidence",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.7,
                    step=0.05,
                    help="Only show predictions above this confidence level"
                )
                
                analysis_depth = st.selectbox(
                    "Analysis Depth",
                    options=["Quick", "Standard", "Deep"],
                    index=1,
                    help="Choose analysis complexity"
                )
        
        return {
            'use_real_data': use_real_data,
            'include_weather': include_weather,
            'prediction_confidence': prediction_confidence,
            'analysis_depth': analysis_depth
        }
    
    def _render_prediction_interface(self, team_selection: dict, config: dict) -> dict | None:
        """Render the prediction generation interface."""
        st.markdown("---")
        st.markdown("### 🔮 Generate Prediction")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            predict_button = st.button(
                "🚀 Generate AI Prediction",
                type="primary",
                use_container_width=True,
                help="Generate enhanced AI-powered match prediction"
            )
        
        if predict_button:
            return self._generate_and_display_prediction(team_selection, config)
        
        # Display prediction history
        self._render_prediction_history()
        
        return None
    
    def _generate_and_display_prediction(self, team_selection: dict, config: dict) -> dict:
        """Generate and display the match prediction."""
        home_team = team_selection['home_team']
        away_team = team_selection['away_team']
        
        with st.spinner("🤖 AI is analyzing teams and generating prediction..."):
            try:
                # Create match data for prediction
                match_data = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'match_date': datetime.now() + timedelta(days=1),
                    'league': team_selection.get('selected_league', 'Mixed'),
                    'venue': f"{home_team} Stadium",
                }
                
                # Generate enhanced prediction
                prediction_result = self.predictor.predict_match_enhanced(
                    home_team,
                    away_team,
                    match_data,
                    use_real_data=config['use_real_data']
                )
                
                if prediction_result:
                    # Display prediction results
                    self._display_prediction_results(prediction_result, team_selection, config)
                    
                    # Add to prediction history
                    self._add_to_prediction_history(prediction_result, team_selection)
                    
                    return {
                        'prediction': prediction_result,
                        'teams': team_selection,
                        'config': config,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    st.error("❌ Failed to generate prediction. Please try again.")
                    
            except Exception as e:
                self.logger.error(f"Prediction generation error: {e}")
                st.error(f"❌ Prediction failed: {str(e)}")
        
        return None
    
    def _display_prediction_results(self, prediction, team_selection: dict, config: dict):
        """Display comprehensive prediction results."""
        st.markdown("### 🎯 AI Prediction Results")
        
        # Main prediction metrics
        col1, col2, col3 = st.columns(3)
        
        home_prob = getattr(prediction, 'home_win_probability', 0) * 100
        draw_prob = getattr(prediction, 'draw_probability', 0) * 100
        away_prob = getattr(prediction, 'away_win_probability', 0) * 100
        
        with col1:
            st.metric(
                f"🏠 {team_selection['home_team']} Win",
                f"{home_prob:.1f}%",
                delta=f"+{home_prob - 33.3:.1f}%" if home_prob > 33.3 else f"{home_prob - 33.3:.1f}%"
            )
        
        with col2:
            st.metric(
                "🤝 Draw",
                f"{draw_prob:.1f}%",
                delta=f"+{draw_prob - 33.3:.1f}%" if draw_prob > 33.3 else f"{draw_prob - 33.3:.1f}%"
            )
        
        with col3:
            st.metric(
                f"✈️ {team_selection['away_team']} Win",
                f"{away_prob:.1f}%",
                delta=f"+{away_prob - 33.3:.1f}%" if away_prob > 33.3 else f"{away_prob - 33.3:.1f}%"
            )
        
        # Confidence and additional insights
        confidence = getattr(prediction, 'confidence', 0.7)
        st.progress(confidence, text=f"🎯 Prediction Confidence: {confidence:.1%}")
        
        # Key factors
        if hasattr(prediction, 'key_factors') and prediction.key_factors:
            st.markdown("#### 🔍 Key Factors")
            for i, factor in enumerate(prediction.key_factors[:3], 1):
                st.markdown(f"**{i}.** {factor}")
        
        # Expected value and betting insights
        if hasattr(prediction, 'expected_value'):
            expected_value = prediction.expected_value
            risk_level = getattr(prediction, 'risk_level', 'Medium')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💰 Expected Value", f"{expected_value:.2f}")
            with col2:
                risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "🟡")
                st.metric("⚠️ Risk Level", f"{risk_color} {risk_level}")
        
        # Real data usage indicator
        try:
            real_used = getattr(prediction, 'real_data_used', None)
            data_ts = getattr(prediction, 'data_timestamp', None)
            data_freshness = None
            if data_ts:
                try:
                    if isinstance(data_ts, (int, float)):
                        data_freshness = datetime.fromtimestamp(float(data_ts)).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        data_freshness = str(data_ts)
                except Exception:
                    data_freshness = str(data_ts)

            try:
                from utils.data_freshness import (
                    format_timestamp,
                    freshness_summary,
                    is_fresh,
                )

                if real_used is True:
                    formatted_ts = format_timestamp(data_ts) if data_ts else None
                    freshness_text = freshness_summary(data_ts) if data_ts else 'unknown'
                    fresh_6h = is_fresh(data_ts, max_age_hours=6.0)
                    fresh_24h = is_fresh(data_ts, max_age_hours=24.0)

                    badge_text = "✅ This prediction uses real-time team data and statistics"
                    if formatted_ts:
                        badge_text += f" — refreshed at {formatted_ts}"

                    if not fresh_6h:
                        st.warning(f"⚠️ Data is older ({freshness_text}). Predictions may be stale.")
                    else:
                        st.success(badge_text)

                    # Publish CTA enabled only when within 24h freshness window
                    if fresh_24h:
                        if st.button("📢 Publish Prediction", key=f"publish_{team_selection['home_team']}_{team_selection['away_team']}"):
                            try:
                                from utils.publish_prediction import publish_prediction
                                publish_prediction(prediction, meta={"source": "match_selector"})
                                st.success("✅ Prediction published to central log.")
                            except Exception as e:
                                self.logger.error(f"Publish failed: {e}")
                                st.error("❌ Publish failed; see logs")
                    else:
                        st.markdown("**Publish Prediction:** Disabled — data is too old to publish. Configure fresh data sources to enable publishing.")
                else:
                    st.warning("⚠️ This prediction was generated using fallback or simulated data. Publishing is disabled until verified real data is available.")
            except Exception as e:
                self.logger.debug(f"Freshness gating failed: {e}")
        except Exception as e:
            self.logger.debug(f"Real-data badge rendering failed: {e}")
    
    def _add_to_prediction_history(self, prediction, team_selection: dict):
        """Add prediction to session history."""
        history_entry = {
            'timestamp': datetime.now(),
            'home_team': team_selection['home_team'],
            'away_team': team_selection['away_team'],
            'prediction': prediction,
            'league': team_selection.get('selected_league', 'Mixed')
        }
        
        st.session_state.prediction_history.insert(0, history_entry)
        
        # Keep only last 10 predictions
        if len(st.session_state.prediction_history) > 10:
            st.session_state.prediction_history = st.session_state.prediction_history[:10]
    
    def _render_prediction_history(self):
        """Render prediction history panel."""
        if st.session_state.prediction_history:
            st.markdown("### 📚 Recent Predictions")
            
            with st.expander(f"View {len(st.session_state.prediction_history)} recent predictions"):
                for i, entry in enumerate(st.session_state.prediction_history[:5]):
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{entry['home_team']} vs {entry['away_team']}**")
                            st.caption(f"{entry['league']} • {entry['timestamp'].strftime('%d/%m/%Y %H:%M')}")
                        
                        with col2:
                            home_prob = getattr(entry['prediction'], 'home_win_probability', 0) * 100
                            st.metric("Home Win", f"{home_prob:.0f}%")
                        
                        with col3:
                            confidence = getattr(entry['prediction'], 'confidence', 0.7)
                            st.metric("Confidence", f"{confidence:.0%}")
                    
                    if i < len(st.session_state.prediction_history) - 1:
                        st.divider()
    
    def _get_fallback_leagues_and_teams(self) -> dict[str, list[str]]:
        """Provide fallback team data from registry/enhancer when database is unavailable."""
        try:
            registry = get_team_assets_registry()
            leagues_data: dict[str, list[str]] = {}
            for record in registry.all_team_records():
                # Use set to prevent duplicate team names in each league
                if record.league_name not in leagues_data:
                    leagues_data[record.league_name] = set()
                leagues_data[record.league_name].add(record.display_name)
            
            # Convert sets to sorted lists
            for league in leagues_data:
                leagues_data[league] = sorted(list(leagues_data[league]))
            
            if leagues_data:
                self.logger.info(f"Loaded {len(leagues_data)} leagues with fallback data from registry")
                return leagues_data
        except Exception as e:
            self.logger.error(f"Failed to load fallback team data: {e}")
        # Minimal fallback with essential teams
        return {
            "Premier League": ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Tottenham"],
            "La Liga": ["Barcelona", "Real Madrid", "Atletico Madrid", "Sevilla"],
            "Bundesliga": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig"],
            "Serie A": ["Juventus", "Inter Milan", "AC Milan", "AS Roma"],
        }

    def _has_valid_league_cache(self) -> bool:
        """Determine whether the cached league directory is still valid."""
        cache_meta = st.session_state.get(self.LEAGUE_CACHE_KEY)
        if not cache_meta:
            return False

        timestamp_str = cache_meta.get("timestamp")
        if not timestamp_str:
            return False
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            return False

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        reference_time = datetime.now(timezone.utc)
        if reference_time - timestamp > self.LEAGUE_CACHE_TTL:
            return False
        if cache_meta.get("using_fallback") != self.db_manager.using_sqlite_fallback:
            return False
        return bool(cache_meta.get("data"))

    def _load_cached_leagues(self) -> dict[str, list[str]] | None:
        """Return cached league directory if available and valid."""
        cache_meta = st.session_state.get(self.LEAGUE_CACHE_KEY)
        if not cache_meta:
            return None

        timestamp_str = cache_meta.get("timestamp")
        if not timestamp_str:
            return None
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            return None

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        reference_time = datetime.now(timezone.utc)
        if reference_time - timestamp > self.LEAGUE_CACHE_TTL:
            return None
        if cache_meta.get("using_fallback") != self.db_manager.using_sqlite_fallback:
            return None

        data = cache_meta.get("data")
        if not data:
            return None

        self._last_league_fetch_meta = {
            "timestamp": timestamp,
            "using_fallback": cache_meta.get("using_fallback", False),
            "from_cache": True,
        }
        return data

    def _store_cached_leagues(self, leagues_data: dict[str, list[str]], timestamp: datetime) -> None:
        """Persist the latest league directory in session state cache."""
        timestamp_utc = timestamp
        if timestamp_utc.tzinfo is None:
            timestamp_utc = timestamp_utc.replace(tzinfo=timezone.utc)
        else:
            timestamp_utc = timestamp_utc.astimezone(timezone.utc)

        st.session_state[self.LEAGUE_CACHE_KEY] = {
            "data": leagues_data,
            "timestamp": timestamp_utc.isoformat(),
            "using_fallback": self.db_manager.using_sqlite_fallback,
        }
        self._last_league_fetch_meta = {
            "timestamp": timestamp_utc,
            "using_fallback": self.db_manager.using_sqlite_fallback,
            "from_cache": False,
        }
    
    def _get_leagues_and_teams(self) -> dict[str, list[str]]:
        """Get available leagues and their teams."""
        try:
            cached = self._load_cached_leagues()
            if cached is not None:
                return cached

            with self.db_manager.session_scope() as session:
                leagues = session.query(League).all()
                leagues_data = {}
                
                for league in leagues:
                    teams = session.query(Team).filter(Team.league_id == league.id).all()
                    # Use set to eliminate duplicates, then sort
                    team_names = sorted(list(set(team.name for team in teams if team.name)))
                    
                    if team_names:  # Only include leagues with teams
                        leagues_data[league.name] = team_names
                
                # If no leagues found, provide fallback data
                if not leagues_data:
                    self.logger.warning("No leagues found in database, using fallback team data")
                    leagues_data = self._get_fallback_leagues_and_teams()

                timestamp = datetime.now(timezone.utc)
                self._store_cached_leagues(leagues_data, timestamp)
                
                return leagues_data
                
        except Exception as e:
            self.logger.error(f"Error loading leagues and teams: {e}")
            # Check if database fallback is active
            if hasattr(self.db_manager, 'using_sqlite_fallback') and self.db_manager.using_sqlite_fallback:
                st.info("ℹ️ Using local database fallback. Some features may be limited.")
            else:
                st.warning("⚠️ Database connection issue detected. Using fallback team data.")
            fallback_data = self._get_fallback_leagues_and_teams()
            timestamp = datetime.now(timezone.utc)
            self._store_cached_leagues(fallback_data, timestamp)
            return fallback_data
    
    def _get_team_basic_stats(self, team_name: str) -> dict:
        """Get basic team statistics."""
        try:
            with self.db_manager.session_scope() as session:
                team = session.query(Team).filter(Team.name == team_name).first()
                if not team:
                    return {}
                
                # Get recent matches
                recent_matches = session.query(Match).filter(
                    ((Match.home_team_id == team.id) | (Match.away_team_id == team.id)) &
                    (Match.status == 'FINISHED')
                ).order_by(Match.match_date.desc()).limit(5).all()
                
                if not recent_matches:
                    return {'recent_form': 'N/A', 'form_trend': 0}
                
                # Calculate recent form
                points = 0
                for match in recent_matches:
                    is_home = match.home_team_id == team.id
                    team_goals = match.home_score if is_home else match.away_score
                    opponent_goals = match.away_score if is_home else match.home_score
                    
                    if team_goals is not None and opponent_goals is not None:
                        if team_goals > opponent_goals:
                            points += 3
                        elif team_goals == opponent_goals:
                            points += 1
                
                form_percentage = (points / (len(recent_matches) * 3)) * 100
                
                return {
                    'recent_form': f"{points}/{len(recent_matches) * 3}",
                    'form_trend': form_percentage - 50  # Relative to 50% baseline
                }
                
        except Exception as e:
            self.logger.warning(f"Could not get team stats for {team_name}: {e}")
            return {'recent_form': 'N/A', 'form_trend': 0}


def render_enhanced_match_selector() -> dict | None:
    """
    Convenience function to render the enhanced match selector.
    
    Returns:
        Dict containing prediction results if available
    """
    selector = EnhancedMatchSelector()
    return selector.render_enhanced_match_selector()


if __name__ == "__main__":
    # For testing purposes
    st.set_page_config(
        page_title="Enhanced Match Selector",
        page_icon="⚽",
        layout="wide"
    )
    
    result = render_enhanced_match_selector()
    
    if result:
        st.success("✅ Prediction generated successfully!")
        st.json(result)
