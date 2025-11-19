"""
Production-Ready Betting Insights Dashboard
A comprehensive, high-performance interface for actionable betting insights across top 6 football leagues.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.ui_elements import (create_metric_card, header,
                                              validate_match_data)
from dashboard.components.unified_design_system import \
    get_unified_design_system
from dashboard.error_log import log_error
from dashboard.visualizations.betting_insights import (
    plot_betting_opportunities_by_league, plot_value_rating_distribution)

logger = logging.getLogger(__name__)

# Top 6 leagues configuration with enhanced metadata
TOP_LEAGUES = {
    "Premier League": {
        "code": "PL", 
        "country": "England", 
        "priority": 1,
        "icon": "🏴",
        "color": "#37003C",
        "description": "English Premier League"
    },
    "La Liga": {
        "code": "PD", 
        "country": "Spain", 
        "priority": 2,
        "icon": "🇪🇸",
        "color": "#FF6B35",
        "description": "Spanish La Liga"
    },
    "Bundesliga": {
        "code": "BL1", 
        "country": "Germany", 
        "priority": 3,
        "icon": "🇩🇪",
        "color": "#D20515",
        "description": "German Bundesliga"
    },
    "Serie A": {
        "code": "SA", 
        "country": "Italy", 
        "priority": 4,
        "icon": "🇮🇹",
        "color": "#008FD7",
        "description": "Italian Serie A"
    },
    "Ligue 1": {
        "code": "FL1", 
        "country": "France", 
        "priority": 5,
        "icon": "🇫🇷",
        "color": "#091C3E",
        "description": "French Ligue 1"
    },
    "Eredivisie": {
        "code": "DED", 
        "country": "Netherlands", 
        "priority": 6,
        "icon": "🇳🇱",
        "color": "#FF6B35",
        "description": "Dutch Eredivisie"
    }
}

@dataclass
class BettingOpportunity:
    """Data class for betting opportunities."""
    match_id: str
    home_team: str
    away_team: str
    match_date: datetime
    competition: str
    outcome: str
    prediction_prob: float
    bookie_odds: float
    edge_pct: float
    kelly_stake: float
    value_rating: float
    confidence: str
    risk_level: str


def calculate_value_bet_metrics(prediction_prob: float, bookie_odds: float) -> Dict[str, float]:
    """
    Calculate comprehensive value betting metrics.
    """
    if bookie_odds <= 1:
        return {
            "edge": 0.0, "edge_pct": 0.0, "implied_prob": 1.0,
            "kelly_stake": 0.0, "value_rating": 0.0
        }
    
    implied_prob = 1 / bookie_odds
    edge = prediction_prob - implied_prob
    
    q = 1 - prediction_prob
    b = bookie_odds - 1
    kelly_stake = max(0, (b * prediction_prob - q) / b) if b > 0 else 0
    
    # Value rating: scale edge percentage to a 0-10 score. An edge of 20% gives a score of 10.
    value_rating = min(10, max(0, (edge * 100) / 2))
    
    return {
        "edge": edge,
        "edge_pct": edge * 100,
        "implied_prob": implied_prob,
        "kelly_stake": kelly_stake,
        "value_rating": value_rating,
    }


def analyze_match_betting_opportunities(
    match_data: Dict[str, Any],
    prediction_data: Dict[str, Any],
    odds_data: Dict[str, Any]
) -> List[BettingOpportunity]:
    """
    Analyze betting opportunities for a single match.
    """
    opportunities = []
    outcomes = {
        "Home Win": ("home_win_prob", "home_win"),
        "Draw": ("draw_prob", "draw"),
        "Away Win": ("away_win_prob", "away_win"),
    }

    for outcome_name, (pred_key, odds_key) in outcomes.items():
        pred_prob = prediction_data.get(pred_key, 0)
        bookie_odds = odds_data.get(odds_key, 0)

        if pred_prob == 0 or bookie_odds == 0:
            continue

        metrics = calculate_value_bet_metrics(pred_prob, bookie_odds)
        
        confidence = "High" if metrics["edge_pct"] > 10 else "Medium" if metrics["edge_pct"] > 5 else "Low"
        risk_level = "High" if metrics["kelly_stake"] > 0.1 else "Medium" if metrics["kelly_stake"] > 0.05 else "Low"
        
        opportunities.append(BettingOpportunity(
            match_id=match_data.get("id"),
            home_team=match_data.get("home_team", "N/A"),
            away_team=match_data.get("away_team", "N/A"),
            match_date=match_data.get("match_date"),
            competition=match_data.get("competition", "N/A"),
            outcome=outcome_name,
            prediction_prob=pred_prob,
            bookie_odds=bookie_odds,
            edge_pct=metrics["edge_pct"],
            kelly_stake=metrics["kelly_stake"],
            value_rating=metrics["value_rating"],
            confidence=confidence,
            risk_level=risk_level
        ))
    
    return opportunities

def render_sidebar_filters() -> Dict[str, Any]:
    """Render sidebar filters and return their values."""
    with st.sidebar:
        st.header("⚙️ Dashboard Filters")
        
        leagues = list(TOP_LEAGUES.keys())
        selected_leagues = st.multiselect("Leagues", leagues, default=leagues)
        
        today = datetime.now().date()
        # Limit date range to max 4 days
        date_range = st.date_input("Date Range (max 4 days)", [today, today + timedelta(days=1)])
        # Validate date range
        if len(date_range) == 2:
            delta = (date_range[1] - date_range[0]).days
            if delta > 4:
                st.warning(f"Date range cannot exceed 4 days. Please select a shorter range.")
                date_range = [date_range[0], date_range[0] + timedelta(days=4)]
        
        min_value_score = st.slider("Minimum Value Score", 0.0, 10.0, 3.0, 0.5)
        
        confidence_filter = st.multiselect("Confidence Level", ["High", "Medium", "Low"], default=["High", "Medium"])
        
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return {
        "leagues": selected_leagues,
        "date_range": date_range,
        "min_value": min_value_score,
        "confidence": confidence_filter
    }


def render_betting_opportunity_card(opportunity: BettingOpportunity):
    """
    Render a betting opportunity using themed components.
    """
    card_title = f"{opportunity.home_team} vs {opportunity.away_team}"
    card_subheader = f"{opportunity.competition} | {opportunity.match_date.strftime('%d %b %Y, %H:%M')}"
    
    content = f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 1.2em; font-weight: bold;">{opportunity.outcome}</div>
            <div style="font-size: 0.9em; color: #666;">Odds: @{opportunity.bookie_odds:.2f}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 1.1em; font-weight: bold; color: #2e7d32;">Edge: {opportunity.edge_pct:.2f}%</div>
            <div style="font-size: 0.9em; color: #1e88e5;">Kelly: {opportunity.kelly_stake:.2%}</div>
        </div>
    </div>
    """
    
    border_color = "#4CAF50" if opportunity.value_rating >= 7 else "#FFC107" if opportunity.value_rating >= 4 else "#FF9800"
    # Use UnifiedDesignSystem card with a content function to preserve HTML and behavior
    uds = get_unified_design_system()

    def _card_content():
        # Add a subtle left border to keep color semantics from the old themed card
        st.markdown(f"""
        <div style="border-left:4px solid {border_color}; padding-left:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size: 1.2em; font-weight: bold;">{opportunity.outcome}</div>
                    <div style="font-size: 0.9em; color: #666;">Odds: @{opportunity.bookie_odds:.2f}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.1em; font-weight: bold; color: #2e7d32;">Edge: {opportunity.edge_pct:.2f}%</div>
                    <div style="font-size: 0.9em; color: #1e88e5;">Kelly: {opportunity.kelly_stake:.2%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    try:
        uds.create_unified_card(_card_content, card_class='premium-card')
    except Exception:
        # Fallback to legacy themed render if unified card fails
        st.markdown(content, unsafe_allow_html=True)


def render_betting_insights_summary(opportunities: List[BettingOpportunity]):
    """
    Render summary statistics for betting opportunities.
    """
    if not opportunities:
        st.info("No betting opportunities match the selected criteria.")
        return

    st.subheader("📊 Key Metrics")
    total_ops = len(opportunities)
    avg_value = np.mean([o.value_rating for o in opportunities])
    avg_edge = np.mean([o.edge_pct for o in opportunities])
    premium_ops = sum(1 for o in opportunities if o.value_rating >= 7)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Total Opportunities", str(total_ops))
    with col2:
        create_metric_card("Avg. Value Score", f"{avg_value:.1f}/10")
    with col3:
        create_metric_card("Avg. Edge", f"{avg_edge:.2f}%")
    with col4:
        create_metric_card("Premium Bets", str(premium_ops), help_text="Opportunities with a value score of 7 or higher")

    st.subheader("Visual Analysis")
    fig_leagues = plot_betting_opportunities_by_league(opportunities)
    st.plotly_chart(fig_leagues, use_container_width=True)

    fig_dist = plot_value_rating_distribution(opportunities)
    st.plotly_chart(fig_dist, use_container_width=True)


def render_production_betting_dashboard(data_loader):
    """
    Render the main production betting insights dashboard.
    """
    # Prefer unified design system styling; fall back to legacy CSS loader
    try:
        uds = get_unified_design_system()
        try:
            uds.inject_unified_css('premium')
        except Exception as e:
            logger.warning(f"Unified CSS injection failed, falling back to legacy CSS: {e}")
            from dashboard.components.ui_elements import load_custom_css
            load_custom_css()

        # Use unified header when possible
        try:
            uds.create_unified_header("GoalDiggers Betting Insights", subtitle="Unified Platform for Advanced Betting Analysis")
            st.markdown(header("A production-ready dashboard for identifying high-value betting opportunities.", level=4, accent_bar=False, divider=False), unsafe_allow_html=True)
        except Exception:
            # Fallback to legacy header HTML
            st.markdown(header("GoalDiggers Betting Insights", level=2, icon="🎯", accent_bar=True, animation='slide'), unsafe_allow_html=True)
            st.markdown(header("Unified Platform for Advanced Betting Analysis", level=3, accent_bar=True, animation='fade'), unsafe_allow_html=True)
            st.markdown(header("A production-ready dashboard for identifying high-value betting opportunities.", level=4, accent_bar=False, divider=False), unsafe_allow_html=True)
    except Exception:
        # If design system isn't available for any reason, keep original behavior
        from dashboard.components.ui_elements import load_custom_css
        load_custom_css()
        st.markdown(header("GoalDiggers Betting Insights", level=2, icon="🎯", accent_bar=True, animation='slide'), unsafe_allow_html=True)
        st.markdown(header("Unified Platform for Advanced Betting Analysis", level=3, accent_bar=True, animation='fade'), unsafe_allow_html=True)
        st.markdown(header("A production-ready dashboard for identifying high-value betting opportunities.", level=4, accent_bar=False, divider=False), unsafe_allow_html=True)
    filters = render_sidebar_filters()
    if not filters["leagues"] or not filters["date_range"] or len(filters["date_range"]) != 2:
        st.warning("Please select valid filters in the sidebar to begin analysis.")
        return
    with st.spinner("Analyzing matches for betting opportunities..."):
        try:
            matches_df = data_loader.load_matches(filters["leagues"], filters["date_range"])
            if matches_df.empty:
                st.info("No matches found for the selected leagues and date range.")
                return
            all_opportunities = []
            missing_data_matches = []
            for _, match in matches_df.iterrows():
                match_details = data_loader.load_match_details(match['id'])
                predictions = match_details.get('prediction', {})
                odds = match_details.get('bookie_odds', {})
                if predictions and odds:
                    match_dict = validate_match_data(match.to_dict())
                    ops = analyze_match_betting_opportunities(match_dict, predictions, odds)
                    all_opportunities.extend(ops)
                else:
                    missing_data_matches.append({
                        'id': match.get('id', None),
                        'home_team': match.get('home_team', 'N/A'),
                        'away_team': match.get('away_team', 'N/A'),
                        'reason': f"Missing {'prediction' if not predictions else ''}{' and ' if not predictions and not odds else ''}{'odds' if not odds else ''}"
                    })
            filtered_ops = [
                op for op in all_opportunities
                if op.value_rating >= filters["min_value"] and op.confidence in filters["confidence"]
            ]
            filtered_ops.sort(key=lambda x: x.value_rating, reverse=True)
        except Exception as e:
            log_error(f"Error processing betting dashboard data: {e}", e)
            st.error("An error occurred while analyzing data. Please check the logs.")
            return
    # Use create_metric_card for summary metrics
    st.markdown(header("Production Dashboard", level=3, icon="📊", accent_bar=True, animation='fade'), unsafe_allow_html=True)
    total_ops = len(filtered_ops)
    avg_value = np.mean([o.value_rating for o in filtered_ops]) if filtered_ops else 0
    avg_edge = np.mean([o.edge_pct for o in filtered_ops]) if filtered_ops else 0
    premium_ops = sum(1 for o in filtered_ops if o.value_rating >= 7)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Total Opportunities", str(total_ops))
    with col2:
        create_metric_card("Avg. Value Score", f"{avg_value:.1f}/10")
    with col3:
        create_metric_card("Avg. Edge", f"{avg_edge:.2f}%")
    with col4:
        create_metric_card("Premium Bets", str(premium_ops), help_text="Opportunities with a value score of 7 or higher")

    st.subheader("Visual Analysis")
    fig_leagues = plot_betting_opportunities_by_league(filtered_ops)
    st.plotly_chart(fig_leagues, use_container_width=True)

    fig_dist = plot_value_rating_distribution(filtered_ops)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown(header("Betting Insights Dashboard", level=3, icon="🎯", accent_bar=True, animation='slide'), unsafe_allow_html=True)
    st.markdown(header("Actionable betting opportunities for the top 6 football leagues", level=4, accent_bar=True, animation='fade'), unsafe_allow_html=True)
    # Render top betting opportunities using unified cards (preserve HTML content)
    if not filtered_ops:
        st.info("No opportunities match the current filter settings.")
    else:
        uds = get_unified_design_system()
        for op in filtered_ops:
            card_html = f"<b>Outcome:</b> {op.outcome}<br><b>Odds:</b> {op.bookie_odds:.2f}<br><b>Edge:</b> {op.edge_pct:.2f}%<br><b>Kelly:</b> {op.kelly_stake:.2%}<br><b>Confidence:</b> {op.confidence}<br><b>Risk Level:</b> {op.risk_level}"

            def _op_content(card_html=card_html, op=op):
                st.markdown(card_html, unsafe_allow_html=True)

            try:
                uds.create_unified_card(_op_content, card_class='goaldiggers-card')
            except Exception:
                # Fallback to legacy themed card rendering
                st.markdown(card_html, unsafe_allow_html=True)

    with st.expander("📘 About This Dashboard"):
        st.markdown("""
        This dashboard leverages our proprietary prediction models to find **value bets**—where the likelihood of an outcome is higher than the bookmaker's odds suggest.
        - **Value Score**: A 0-10 rating of the opportunity's quality.
        - **Edge**: The percentage advantage over the bookmaker.
        - **Kelly Stake**: The recommended bet size as a percentage of your bankroll, calculated to maximize long-term growth.
        
        **Disclaimer**: This is a tool for analysis, not financial advice. Gamble responsibly.
        """, unsafe_allow_html=True)