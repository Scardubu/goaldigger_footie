"""
Sidebar component for the dashboard.
Handles view selection, league selection, date range, value betting settings, 
context toggles, AI analysis options, and system status.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

from dashboard.components.ui_elements import header
from dashboard.error_log import log_error

logger = logging.getLogger(__name__)

def get_default_sidebar_values(selected_view: str = "matches") -> Dict[str, Any]:
    """Return default sidebar values when leagues are not available or on error."""
    today = datetime.now().date()
    tomorrow = today + timedelta(days=1)
    return {
        "view": selected_view,
        "leagues": [],
        "date_range": (today, tomorrow),
        "context_toggles": {"h2h": True, "style": True, "form": True, "home": True, "weather": False},
        "ai_provider": "gemini",
        "ai_model": "gemini-pro",
        "value_bet_settings": {
            "min_edge": 0.05,
            "min_confidence": 0.65,
            "max_bets": 10
        },
        "quick_filters": {
            "value_bets": False,
            "live_matches": False,
            "top_leagues": False
        },
        "odds_format": "decimal",
        "theme": "auto"
    }

def render_league_selection(available_leagues: List[Dict[str, Any]]) -> List[str]:
    """Renders the league selection checkboxes and returns selected league IDs."""
    selected_leagues = []
    top_leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Eredivisie"]
    leagues_by_country = {}
    if not available_leagues:
        return []
    # Handle both list of strings and list of dicts for robustness
    if isinstance(available_leagues[0], str):
        for league_name in available_leagues:
            country = "Top Leagues" if league_name in top_leagues else "Other Leagues"
            leagues_by_country.setdefault(country, []).append({"name": league_name, "id": league_name})
    else:
        for league in available_leagues:
            if isinstance(league, dict):
                country = league.get("country", "Other")
                leagues_by_country.setdefault(country, []).append(league)
            else:
                leagues_by_country.setdefault("Other", []).append({"name": str(league), "id": str(league)})
    sorted_countries = sorted([c for c in leagues_by_country.keys() if c != "Top Leagues"])
    if "Top Leagues" in leagues_by_country:
        sorted_countries.insert(0, "Top Leagues")
    for country in sorted_countries:
        with st.expander(f"{country}", expanded=(country in ["Top Leagues", "England"])):
            for league in sorted(leagues_by_country[country], key=lambda x: x["name"]):
                league_name = league["name"]
                league_id = league["id"]
                is_top_league = league_name in top_leagues
                key = f"league_{league_id}"
                default = st.session_state.get(key, is_top_league)
                if st.checkbox(league_name, key=key, value=default):
                    selected_leagues.append(league_id)
    return selected_leagues

def render_date_range_selection() -> Tuple[datetime.date, datetime.date]:
    """Renders the date range selection and returns the selected range."""
    today = datetime.now().date()
    date_range_option = st.radio(
        "Select date range",
        options=["Next 7 days", "Next 14 days", "Next 30 days", "Custom"],
        index=st.session_state.get("date_range_option_index", 0),
        key="date_range_option",
        label_visibility="collapsed"
    )
    st.session_state["date_range_option_index"] = ["Next 7 days", "Next 14 days", "Next 30 days", "Custom"].index(date_range_option)
    if date_range_option == "Next 7 days":
        return (today, today + timedelta(days=7))
    elif date_range_option == "Next 14 days":
        return (today, today + timedelta(days=14))
    elif date_range_option == "Next 30 days":
        return (today, today + timedelta(days=30))
    else:  # Custom
        start_date = st.date_input("Start Date", value=st.session_state.get("start_date", today), min_value=today, key="start_date")
        end_date = st.date_input("End Date", value=st.session_state.get("end_date", today + timedelta(days=7)), min_value=start_date, key="end_date")
        return (start_date, end_date)

def render_value_betting_settings(selected_view: str) -> Dict[str, float]:
    """Renders value betting settings and returns them."""
    if selected_view == "value_betting":
        st.markdown(header("Value Betting", level=3, icon="💰", divider=False, accent_bar=False), unsafe_allow_html=True)
        min_edge = st.slider("Minimum Edge (%)", 1, 20, st.session_state.get("min_edge", 5), key="min_edge") / 100.0
        min_confidence = st.slider("Minimum Confidence (%)", 50, 95, st.session_state.get("min_confidence", 65), key="min_confidence") / 100.0
        max_bets = st.slider("Max Bets", 5, 30, st.session_state.get("max_bets", 10), key="max_bets")
        return {"min_edge": min_edge, "min_confidence": min_confidence, "max_bets": max_bets}
    return {"min_edge": 0.05, "min_confidence": 0.65, "max_bets": 10}

def render_context_toggles() -> Dict[str, bool]:
    """Renders the context toggles and returns their state."""
    h2h = st.checkbox("Head-to-Head History", value=st.session_state.get("h2h", True), key="h2h")
    form = st.checkbox("Recent Form", value=st.session_state.get("form", True), key="form")
    style = st.checkbox("Team Style", value=st.session_state.get("style", True), key="style")
    home = st.checkbox("Home Advantage", value=st.session_state.get("home", True), key="home")
    weather = st.checkbox("Weather Impact", value=st.session_state.get("weather", False), key="weather")
    return {"h2h": h2h, "form": form, "style": style, "home": home, "weather": weather}

def render_ai_selection() -> Tuple[str, str]:
    """Renders AI provider and model selection and returns them."""
    provider = st.selectbox("Provider", options=["gemini", "openai", "openrouter"], index=st.session_state.get("ai_provider_index", 0), key="ai_provider")
    model_options = {
        "gemini": ["gemini-pro"],
        "openai": ["gpt-4-turbo", "gpt-3.5-turbo"],
        "openrouter": ["meta/llama-3-70b", "anthropic/claude-3-opus"]
    }
    model = st.selectbox("Model", options=model_options.get(provider, ["gemini-pro"]), index=st.session_state.get("ai_model_index", 0), key="ai_model")
    st.session_state["ai_provider_index"] = ["gemini", "openai", "openrouter"].index(provider)
    st.session_state["ai_model_index"] = model_options.get(provider, ["gemini-pro"]).index(model)
    return provider, model

def render_theme_toggle() -> str:
    """Renders a theme toggle and returns the selected theme."""
    theme = st.selectbox(
        "Theme",
        options=["Auto", "Light", "Dark"],
        index=st.session_state.get("theme_index", 0),
        key="theme",
        help="Switch between light, dark, or auto theme."
    )
    st.session_state["theme_index"] = ["Auto", "Light", "Dark"].index(theme)
    return theme.lower()

def render_odds_format_selector() -> str:
    """Renders odds format selector and returns the selected format."""
    odds_format = st.selectbox(
        "Odds Format",
        options=["Decimal", "Fractional", "American"],
        index=st.session_state.get("odds_format_index", 0),
        key="odds_format",
        help="Choose your preferred odds format."
    )
    st.session_state["odds_format_index"] = ["Decimal", "Fractional", "American"].index(odds_format)
    return odds_format.lower()

def render_quick_filters() -> Dict[str, bool]:
    """Renders quick filter toggles and returns their state."""
    st.markdown("<div style='margin-top:0.5em;'></div>", unsafe_allow_html=True)
    st.markdown("**Quick Filters**", unsafe_allow_html=True)
    value_bets = st.checkbox("Show Only Value Bets", value=st.session_state.get("value_bets", False), key="value_bets", help="Show only bets with positive expected value.")
    live_matches = st.checkbox("Show Live Matches", value=st.session_state.get("live_matches", False), key="live_matches", help="Show only currently live matches.")
    top_leagues = st.checkbox("Show Top Leagues Only", value=st.session_state.get("top_leagues", False), key="top_leagues", help="Filter to top 6 European leagues.")
    return {"value_bets": value_bets, "live_matches": live_matches, "top_leagues": top_leagues}

def render_cache_management(get_cache_stats_fn: Callable[[], Dict[str, Any]], on_clear_cache_fn: Optional[Callable[[], None]]):
    """Renders cache statistics and clear button."""
    cache_stats = get_cache_stats_fn()
    cache_status = f"Cache: {cache_stats.get('hits', 0)} hits, {cache_stats.get('misses', 0)} misses"
    st.write(cache_status)
    if on_clear_cache_fn and st.button("Clear Cache"):
        on_clear_cache_fn()
        st.success("Cache cleared!")
        st.rerun()

def render_system_status(get_system_status_fn: Callable[[], Dict[str, str]]):
    """Renders the system status section."""
    st.markdown(header("System Status", level=4, icon="⚙️", divider=False, accent_bar=False), unsafe_allow_html=True)
    status = get_system_status_fn()
    status_colors = {"good": "🟢", "moderate": "🟡", "poor": "🔴"}
    for key, value in status.items():
        if isinstance(value, dict):
            status_value = value.get('status', 'unknown')
            status_icon = status_colors.get(status_value, "⚪️")
            st.write(f"{status_icon} {key.replace('_', ' ').title()}: {status_value.title()}")
        else:
            status_icon = status_colors.get(str(value), "⚪️")
            st.write(f"{status_icon} {key.replace('_', ' ').title()}: {str(value).title()}")

def render_reset_sidebar_button():
    """Renders a reset button to restore sidebar defaults."""
    if st.button("Reset Sidebar", help="Reset all sidebar controls to default values."):
        # Only clear sidebar-related keys
        sidebar_keys = [
            "theme", "theme_index", "odds_format", "odds_format_index", "date_range_option", "date_range_option_index",
            "start_date", "end_date", "min_edge", "min_confidence", "max_bets", "h2h", "form", "style", "home", "weather",
            "ai_provider", "ai_provider_index", "ai_model", "ai_model_index", "value_bets", "live_matches", "top_leagues"
        ]
        for k in list(st.session_state.keys()):
            if k.startswith("league_") or k in sidebar_keys:
                del st.session_state[k]
        st.rerun()

def render_sidebar(
    get_leagues_fn: Callable[[], List[Dict[str, Any]]],
    get_cache_stats_fn: Callable[[], Dict[str, Any]],
    get_system_status_fn: Optional[Callable[[], Dict[str, str]]] = None,
    on_clear_cache_fn: Optional[Callable[[], None]] = None,
    update_database_fn: Optional[Callable[[], bool]] = None
) -> Dict[str, Any]:
    """
    Renders the dashboard sidebar with all controls and returns selected values.
    """
    try:
        with st.sidebar:
            # --- Logo and Title ---
            logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "GoalDiggers_logo.png")
            st.image(logo_path, width=80)
            st.markdown("<h1 style='text-align:center; margin-bottom: 1em; font-family: var(--heading-font-family);'>GoalDiggers</h1>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:0.5em 0;'>", unsafe_allow_html=True)

            # --- View Selection ---
            st.markdown(header("Dashboard View", level=3, icon="📊", divider=False, accent_bar=False), unsafe_allow_html=True)
            view_options = ["Match Predictions", "Value Betting Opportunities"]
            selected_view = st.radio(
                "Select View",
                options=view_options,
                horizontal=True,
                index=0,
                label_visibility="collapsed"
            )
            selected_view = selected_view.lower().replace(" ", "_").replace("match_predictions", "matches").replace("value_betting_opportunities", "value_betting")

            # --- Match Selection ---
            st.markdown(header("Match Selection", level=3, icon="🏆", divider=False, accent_bar=False), unsafe_allow_html=True)
            available_leagues = get_leagues_fn()
            if not available_leagues:
                st.warning("⚠️ No leagues found.")
                if update_database_fn and st.button("🔄 Attempt Database Update"):
                    with st.spinner("Updating database..."):
                        if update_database_fn():
                            st.success("Database update complete. Please refresh.")
                            st.rerun()
                        else:
                            st.error("Database update failed.")
                return get_default_sidebar_values(selected_view)
            selected_leagues = render_league_selection(available_leagues)

            # --- Date Range ---
            st.markdown(header("Date Range", level=3, icon="📅", divider=False, accent_bar=False), unsafe_allow_html=True)
            selected_date_range = render_date_range_selection()

            # --- Value Betting Settings ---
            value_bet_settings = render_value_betting_settings(selected_view)

            # --- Quick Filters ---
            st.markdown("<hr style='margin:0.5em 0;'>", unsafe_allow_html=True)
            quick_filters = render_quick_filters()

            # --- Analysis Context ---
            st.markdown(header("Analysis Context", level=3, icon="🧠", divider=False, accent_bar=False), unsafe_allow_html=True)
            context_toggles = render_context_toggles()

            # --- AI Analysis ---
            st.markdown(header("AI Analysis", level=3, icon="🤖", divider=False, accent_bar=False), unsafe_allow_html=True)
            ai_provider, ai_model = render_ai_selection()

            # --- Odds Format & Theme ---
            st.markdown("<hr style='margin:0.5em 0;'>", unsafe_allow_html=True)
            odds_format = render_odds_format_selector()
            theme = render_theme_toggle()

            # --- System & Cache ---
            st.markdown("<hr style='margin:1.5em 0;'>", unsafe_allow_html=True)
            if get_system_status_fn:
                render_system_status(get_system_status_fn)
            render_cache_management(get_cache_stats_fn, on_clear_cache_fn)

            # --- Reset Sidebar ---
            render_reset_sidebar_button()

            # --- Footer ---
            st.markdown("""
            <div style='text-align:center; color:var(--color-text-secondary); font-size:0.8em; margin-top: 2em;'>
                GoalDiggers v1.0 &copy; 2025
            </div>
            """, unsafe_allow_html=True)

            return {
                "view": selected_view,
                "leagues": selected_leagues,
                "date_range": selected_date_range,
                "context_toggles": context_toggles,
                "ai_provider": ai_provider,
                "ai_model": ai_model,
                "value_bet_settings": value_bet_settings,
                "quick_filters": quick_filters,
                "odds_format": odds_format,
                "theme": theme
            }
    except Exception as e:
        log_error("Error rendering sidebar", e)
        st.sidebar.error(f"Error loading sidebar: {str(e)}")
        return get_default_sidebar_values("matches")