"""
Match details component for the dashboard.
Handles displaying match predictions, odds, value bets, and AI insights.
"""
import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

from dashboard.components.match_insights import render_match_insights
from dashboard.components.ui_elements import card, header, spinner_wrapper
from dashboard.error_log import log_error
from dashboard.visualizations.plots import (interactive_feature_importance,
                                            plot_probabilities_vs_odds)

logger = logging.getLogger(__name__)

def render_match_details(
    match_id: Optional[str] = None,
    match_details: Optional[Dict[str, Any]] = None,
    get_analysis_fn: Callable = None,
    ai_config: Optional[Dict[str, Any]] = None,
    selected_provider: Optional[str] = None,
    selected_model: Optional[str] = None,
    context_toggles: Dict[str, bool] = {},
    use_xgboost: bool = True
) -> None:
    """
    Render detailed match information with metrics, predictions, and odds.
    """
    if not match_id or not match_details:
        st.warning("⚠️ Please select a match from the list to see details.")
        return

    try:
        home_team = match_details.get("home_team", "Unknown Home")
        away_team = match_details.get("away_team", "Unknown Away")
        match_date = match_details.get("match_date", "Unknown Date")
        competition = match_details.get("fixture_details", {}).get("competition_name", "Unknown League")

        # --- Enhanced Match Header: use UnifiedDesignSystem when available ---
        try:
            from dashboard.components.unified_design_system import \
                get_unified_design_system
            uds = get_unified_design_system()
            uds.inject_unified_css('integrated')
            def header_card():
                st.markdown(f"<div style='display:flex; align-items:center; justify-content:center; text-align:center; margin-bottom: 2em;'>"
                            f"<div style='flex:1; text-align:right;'><h2>{home_team}</h2></div>"
                            f"<div style='padding: 0 2em;'><span style='font-size: 2em; font-weight: bold; color: var(--gd-primary);'>VS</span>"
                            f"<div style='font-size: 0.9em; color: var(--gd-medium-gray);'>{match_date}</div></div>"
                            f"<div style='flex:1; text-align:left;'><h2>{away_team}</h2></div></div>")
            uds.create_unified_card(header_card, card_class='featured-card')
        except Exception:
            uds = None
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:center; text-align:center; margin-bottom: 2em;">
                <div style="flex:1; text-align:right;">
                    <h2 style="font-family: var(--heading-font-family); font-weight: 700;">{home_team}</h2>
                </div>
                <div style="padding: 0 2em;">
                    <span style="font-size: 2em; font-weight: bold; color: var(--color-primary);">VS</span>
                    <div style="font-size: 0.9em; color: var(--color-text-secondary);">{match_date}</div>
                </div>
                <div style="flex:1; text-align:left;">
                    <h2 style="font-family: var(--heading-font-family); font-weight: 700;">{away_team}</h2>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # --- Main Content Tabs ---
        tab_titles = ["📊 Predictions", "🧠 AI Analysis", "📈 Feature Importance"]
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            render_predictions_tab(match_details, home_team, away_team)

        with tabs[1]:
            if get_analysis_fn:
                render_ai_analysis_tab(
                    home_team, away_team, match_details, get_analysis_fn, 
                    selected_provider, selected_model, ai_config
                )
            else:
                st.info("AI analysis is not configured for this view.")

        with tabs[2]:
            render_feature_importance_tab(match_details, context_toggles)

    except Exception as e:
        log_error("Error rendering match details", e)
        st.error(f"An error occurred while displaying match details: {e}")
