"""Personalization Sidebar for GoalDiggers Dashboard

Provides personalized recommendations and adaptive interface elements.
"""
from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_personalization_sidebar():
    """Render personalized recommendations and preferences in sidebar."""
    try:
        from user.personalization.preference_engine import PreferenceEngine

        # Initialize preference engine if not in session state
        if "gd_preference_engine" not in st.session_state:
            st.session_state["gd_preference_engine"] = PreferenceEngine()
        
        engine = st.session_state["gd_preference_engine"]
        
        # Create or get session
        if "gd_user_session" not in st.session_state:
            st.session_state["gd_user_session"] = engine.create_user_session()
        
        session_id = st.session_state["gd_user_session"]
        
        with st.sidebar:
            st.markdown("### 🎯 Personalized For You")
            
            # Get recommendations
            recommendations = _get_user_recommendations(engine, session_id)
            
            if recommendations:
                _render_recommendations(recommendations)
            else:
                st.info("Building your profile... Interact with predictions to get personalized recommendations.")
            
            # Preference controls
            with st.expander("⚙️ Preferences", expanded=False):
                _render_preference_controls(engine, session_id)
                
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"Personalization sidebar unavailable: {e}")


def _get_user_recommendations(engine, session_id: str) -> List[Dict[str, Any]]:
    """Get personalized recommendations for user."""
    try:
        user_id = engine.user_sessions.get(session_id, "anonymous")
        recommendations = engine.get_personalized_recommendations(user_id)
        return recommendations[:5]  # Top 5 recommendations
    except Exception:
        return []


def _render_recommendations(recommendations: List[Dict[str, Any]]):
    """Render recommendation cards."""
    for rec in recommendations:
        rec_type = rec.get("recommendation_type", "").replace("_", " ").title()
        content = rec.get("content", {})
        confidence = rec.get("confidence", 0.0)
        reasoning = rec.get("reasoning", "")
        
        if rec_type == "Team":
            team_name = content.get("team_name", "Team")
            st.markdown(f"**🏆 {team_name}**")
            st.caption(f"Confidence: {confidence:.0%}")
            st.caption(reasoning)
        elif rec_type == "League":
            league_name = content.get("league_name", "League")
            st.markdown(f"**⚽ {league_name}**")
            st.caption(f"Match interest: {confidence:.0%}")
        elif rec_type == "Market":
            market_type = content.get("market_type", "Market")
            st.markdown(f"**📊 {market_type}**")
            st.caption(reasoning)
        else:
            st.markdown(f"**{rec_type}**")
            st.caption(reasoning)
        
        st.markdown("---")


def _render_preference_controls(engine, session_id: str):
    """Render preference control widgets."""
    user_id = engine.user_sessions.get(session_id, "anonymous")
    
    # Get current preferences
    prefs = engine.user_preferences.get(user_id)
    if not prefs:
        st.caption("No preferences set yet.")
        return
    
    # Betting style
    betting_style = st.selectbox(
        "Betting Style",
        options=["conservative", "balanced", "aggressive"],
        index=["conservative", "balanced", "aggressive"].index(prefs.betting_style),
        key="pref_betting_style"
    )
    
    # Risk tolerance
    risk_tolerance = st.slider(
        "Risk Tolerance",
        min_value=0.0,
        max_value=1.0,
        value=float(prefs.risk_tolerance),
        step=0.1,
        key="pref_risk_tolerance"
    )
    
    # Update button
    if st.button("Save Preferences", use_container_width=True):
        try:
            # Update preferences
            prefs.betting_style = betting_style
            prefs.risk_tolerance = risk_tolerance
            engine.user_preferences[user_id] = prefs
            st.success("✅ Preferences updated!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to update preferences: {e}")


def track_user_interaction(action_type: str, target: str, metadata: Dict[str, Any] | None = None):
    """Track user interaction for personalization learning."""
    try:
        if "gd_preference_engine" not in st.session_state:
            return
        
        engine = st.session_state["gd_preference_engine"]
        session_id = st.session_state.get("gd_user_session")
        
        if session_id:
            engine.track_user_behavior(
                session_id=session_id,
                action_type=action_type,
                target=target,
                metadata=metadata or {}
            )
    except Exception:
        pass  # Silent fail for tracking


def get_adaptive_interface_config() -> Dict[str, Any]:
    """Get adaptive interface configuration based on user preferences."""
    try:
        if "gd_preference_engine" not in st.session_state:
            return {}
        
        engine = st.session_state["gd_preference_engine"]
        session_id = st.session_state.get("gd_user_session")
        
        if not session_id:
            return {}
        
        user_id = engine.user_sessions.get(session_id, "anonymous")
        config = engine.get_adaptive_interface_config(user_id)
        
        return config
    except Exception:
        return {}
