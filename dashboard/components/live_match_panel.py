"""Live Match Panel for GoalDiggers Dashboard

Displays real-time match updates and live statistics.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st


def render_live_match_panel(matches: List[Dict[str, Any]] | None = None):
    """Render live match updates panel.
    
    Parameters
    ----------
    matches : list of dict, optional
        List of live matches. If None, loads from live data processor.
    """
    st.markdown("### ⚡ Live Matches")
    
    if matches is None or len(matches) == 0:
        st.info("No live matches at the moment. Check back during match hours.")
        return
    
    for match in matches[:5]:  # Show top 5 live matches
        _render_live_match_card(match)


def _render_live_match_card(match: Dict[str, Any]):
    """Render a single live match card."""
    home_team = match.get("home_team", "Home")
    away_team = match.get("away_team", "Away")
    score = match.get("score", {})
    home_score = score.get("home", 0)
    away_score = score.get("away", 0)
    minute = match.get("minute", 0)
    status = match.get("status", "live").upper()
    
    with st.container():
        cols = st.columns([2, 1, 2, 1])
        
        with cols[0]:
            st.markdown(f"**{home_team}**")
        
        with cols[1]:
            st.markdown(f"<div style='text-align:center;font-size:1.5rem;font-weight:700;'>{home_score}</div>", unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown(f"<div style='text-align:center;font-size:1.5rem;font-weight:700;'>{away_score}</div>", unsafe_allow_html=True)
        
        with cols[3]:
            st.markdown(f"**{away_team}**")
        
        if status == "LIVE":
            st.caption(f"🔴 LIVE - {minute}'")
        else:
            st.caption(f"Status: {status}")
        
        # Show recent events if available
        events = match.get("events", [])
        if events:
            with st.expander("Recent Events", expanded=False):
                for event in events[-3:]:  # Last 3 events
                    event_time = event.get("minute", "?")
                    event_type = event.get("event_type", "").title()
                    event_team = event.get("team", "")
                    event_desc = event.get("description", "")
                    st.caption(f"{event_time}' - {event_type} - {event_team}: {event_desc}")
        
        st.markdown("---")


def get_live_matches_async(limit: int = 10) -> List[Dict[str, Any]]:
    """Async helper to fetch live matches."""
    try:
        # Import here to avoid circular dependencies
        from data.streams.live_data_processor import LiveDataProcessor

        # Create processor instance
        processor = LiveDataProcessor()
        
        # Get live matches
        live_matches_dict = processor.live_matches or {}
        
        # Convert to list and limit
        matches = list(live_matches_dict.values())[:limit]
        
        return matches
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to fetch live matches: {e}")
        return []


def get_live_matches(limit: int = 10) -> List[Dict[str, Any]]:
    """Synchronous wrapper to get live matches."""
    try:
        return asyncio.run(asyncio.coroutine(lambda: get_live_matches_async(limit))())
    except RuntimeError:
        # If already in async context
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(asyncio.coroutine(lambda: get_live_matches_async(limit))())
        finally:
            loop.close()
    except Exception:
        return []
