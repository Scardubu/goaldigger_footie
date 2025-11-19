"""
Value betting component for the GoalDiggers dashboard.
Displays value betting opportunities with interactive, themed visualizations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from dashboard.components.ui_elements import create_info_card
from dashboard.components.unified_design_system import \
    get_unified_design_system
from dashboard.data_loader import get_value_bets_df
from dashboard.error_log import log_error
from dashboard.visualizations.value_bets import (plot_kelly_stakes,
                                                 plot_probability_comparison,
                                                 plot_value_bet_edge,
                                                 plot_value_bets_overview)

logger = logging.getLogger(__name__)

def render_value_betting_dashboard(min_edge: float = 0.05, min_confidence: float = 0.6, max_bets: int = 20):
    """
    Render the value betting dashboard component using themed, centralized plots.
    """
    uds = get_unified_design_system()
    uds.inject_unified_css('integrated')
    uds.create_unified_header("🎯 Value Betting Opportunities")
    with st.expander("ℹ️ About Value Betting", expanded=False):
        st.markdown("""
        **Value betting** is a strategy where you place bets that have a positive expected value. 
        When our model's predicted probability of an outcome is higher than the probability implied by the bookmaker's odds, 
        a 'value' opportunity exists.
        
        - **Edge**: The percentage advantage you have over the bookmaker.
        - **Confidence**: How certain our model is about its prediction.
        - **Kelly Stake**: A formula-driven stake size to optimize long-term bankroll growth.
        """)
    # --- Filters ---
    cols = st.columns([2, 2, 2, 1])
    with cols[0]:
        min_edge = st.slider("Minimum Edge (%)", 1, 25, int(min_edge * 100), 1) / 100.0
    with cols[1]:
        min_confidence = st.slider("Minimum Confidence (%)", 50, 95, int(min_confidence * 100), 1) / 100.0
    with cols[2]:
        max_bets = st.slider("Max Bets to Show", 5, 50, max_bets, 5)
    with cols[3]:
        st.write("&nbsp;") # Spacer
        refresh = st.button("🔄 Refresh", key="refresh_value_bets_main")
    # --- Data Loading ---
    try:
        if refresh:
            st.cache_data.clear()
        df = get_value_bets_df(min_edge=min_edge, min_confidence=min_confidence, limit=max_bets)
        if df is None or df.empty:
            st.info("No value betting opportunities found with the current filters. Try adjusting the criteria.")
            return
        last_updated = df['created_at'].max() if 'created_at' in df.columns else datetime.now()
    except Exception as e:
        log_error("Failed to load value bets data", e)
        st.error("An error occurred while loading value betting data. Please try again later.")
        return
    # --- Overview Section ---
    uds.create_unified_header("📊 Opportunities Overview")
    overview_plot = plot_value_bets_overview(df)
    st.plotly_chart(overview_plot, use_container_width=True)
    # --- Top Opportunities Table ---
    uds.create_unified_header("🏅 Top Value Bets")
    try:
        display_df = df.copy()
        display_df["match"] = display_df["home_team"] + " vs " + display_df["away_team"]
        # Formatting for display
        for col, fmt in {"edge": ".2%", "kelly_stake": ".2%", "confidence": ".1%", "predicted_prob": ".1%", "implied_prob": ".1%"}.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: format(x, fmt))
        display_df["match_date"] = pd.to_datetime(display_df["match_date"]).dt.strftime("%Y-%m-%d %H:%M")
        # Column selection and renaming
        display_columns = {
            "match": "Match", "match_date": "Date", "league": "League", "bet_type": "Bet Type",
            "selection": "Selection", "odds": "Odds", "edge": "Edge", "kelly_stake": "Kelly Stake", "confidence": "Confidence"
        }
        st.dataframe(
            display_df[list(display_columns.keys())].rename(columns=display_columns),
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Data last updated: {pd.to_datetime(last_updated).strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        log_error("Failed to display value bets table", e)
        st.error("Could not display the value bets table.")

def render_match_value_bets(match_data: Dict[str, Any], value_bets: List[Dict[str, Any]]):
    """
    Render value betting opportunities for a specific match using themed plots.
    """
    uds = get_unified_design_system()
    uds.inject_unified_css('integrated')
    if not value_bets:
        st.info("No value betting opportunities found for this match.")
        return
    uds.create_unified_header("💰 Value Betting Analysis")
    try:
        # --- Key Metrics ---
        df = pd.DataFrame(value_bets)
        top_bet = df.loc[df['edge'].idxmax()]
        metrics = {
            "Top Bet Selection": f"{top_bet['selection']} ({top_bet['bet_type']})",
            "Highest Edge": f"{top_bet['edge']:.2%}",
            "Recommended Stake": f"{top_bet['kelly_stake']:.2%}"
        }
        uds.create_unified_metric_row(metrics)
        # --- Visualizations ---
        tab1, tab2 = st.tabs(["Betting Edge & Stake", "Probability Comparison"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                edge_fig = plot_value_bet_edge(value_bets)
                st.plotly_chart(edge_fig, use_container_width=True)
            with col2:
                kelly_fig = plot_kelly_stakes(value_bets)
                st.plotly_chart(kelly_fig, use_container_width=True)
        with tab2:
            prob_fig = plot_probability_comparison(value_bets)
            st.plotly_chart(prob_fig, use_container_width=True)
        # --- Detailed Table ---
        with st.expander("📊 Detailed Value Bets Data", expanded=False):
            display_df = df.copy()
            format_cols = {"edge": ".2%", "predicted_prob": ".1%", "implied_prob": ".1%", "kelly_stake": ".2%"}
            for col, fmt in format_cols.items():
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: format(x, fmt))
            rename_cols = {
                "bet_type": "Bet Type", "selection": "Selection", "odds": "Odds",
                "predicted_prob": "AI Prob", "implied_prob": "Implied Prob",
                "edge": "Edge", "kelly_stake": "Kelly Stake", "confidence": "Confidence"
            }
            st.dataframe(
                display_df[list(rename_cols.keys())].rename(columns=rename_cols),
                use_container_width=True,
                hide_index=True
            )
    except Exception as e:
        log_error(f"Error rendering match value bets for match_id: {match_data.get('match_id')}", e)
        st.error("An error occurred while displaying value betting details.")
