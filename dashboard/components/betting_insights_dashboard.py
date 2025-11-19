"""
Comprehensive Betting Insights Dashboard Component
Provides actionable betting insights for the top 6 football leagues.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.ui_elements import (card, create_metric_card, header,
                                              validate_match_data)
from dashboard.components.unified_design_system import \
    get_unified_design_system

logger = logging.getLogger(__name__)

# Top 6 leagues configuration
TOP_LEAGUES = {
    "Premier League": {"code": "PL", "country": "England", "priority": 1},
    "La Liga": {"code": "PD", "country": "Spain", "priority": 2},
    "Bundesliga": {"code": "BL1", "country": "Germany", "priority": 3},
    "Serie A": {"code": "SA", "country": "Italy", "priority": 4},
    "Ligue 1": {"code": "FL1", "country": "France", "priority": 5},
    "Eredivisie": {"code": "DED", "country": "Netherlands", "priority": 6}
}

def calculate_value_bet_metrics(prediction_prob: float, bookie_odds: float) -> Dict[str, float]:
    """
    Calculate value betting metrics for a given prediction and odds.
    
    Args:
        prediction_prob: Model's predicted probability (0-1)
        bookie_odds: Bookmaker's decimal odds
        
    Returns:
        Dictionary with value betting metrics
    """
    if bookie_odds <= 1:
        return {
            "edge": 0.0,
            "edge_pct": 0.0,
            "implied_prob": 0.0,
            "kelly_stake": 0.0,
            "value_rating": 0.0
        }
    
    implied_prob = 1 / bookie_odds
    edge = prediction_prob - implied_prob
    edge_pct = edge * 100
    
    # Kelly Criterion calculation
    q = 1 - prediction_prob
    b = bookie_odds - 1
    kelly_stake = max(0, (b * prediction_prob - q) / b) if b > 0 else 0
    
    # Value rating (0-10 scale)
    value_rating = min(10, max(0, edge_pct * 2))  # Scale edge percentage to 0-10
    
    return {
        "edge": edge,
        "edge_pct": edge_pct,
        "implied_prob": implied_prob,
        "kelly_stake": kelly_stake,
        "value_rating": value_rating
    }

def analyze_match_betting_opportunities(
    match_data: Dict[str, Any],
    prediction_data: Dict[str, Any],
    odds_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze betting opportunities for a single match.
    
    Args:
        match_data: Match information
        prediction_data: Model predictions
        odds_data: Bookmaker odds
        
    Returns:
        Dictionary with betting analysis
    """
    analysis = {
        "match_id": match_data.get("id"),
        "home_team": match_data.get("home_team", "Unknown"),
        "away_team": match_data.get("away_team", "Unknown"),
        "match_date": match_data.get("match_date"),
        "competition": match_data.get("competition", "Unknown"),
        "betting_opportunities": {},
        "best_value_bet": None,
        "overall_value_score": 0.0
    }
    
    # Extract prediction probabilities
    predictions = {
        "home_win": prediction_data.get("home_win_prob", 0.33),
        "draw": prediction_data.get("draw_prob", 0.33),
        "away_win": prediction_data.get("away_win_prob", 0.34)
    }
    
    # Extract odds
    odds = {
        "home_win": odds_data.get("home_win", 2.0),
        "draw": odds_data.get("draw", 3.0),
        "away_win": odds_data.get("away_win", 2.0)
    }
    
    # Analyze each outcome
    best_value = None
    total_value_score = 0
    
    for outcome, pred_prob in predictions.items():
        bookie_odds = odds.get(outcome, 2.0)
        metrics = calculate_value_bet_metrics(pred_prob, bookie_odds)
        
        analysis["betting_opportunities"][outcome] = {
            "prediction_prob": pred_prob,
            "bookie_odds": bookie_odds,
            **metrics
        }
        
        # Track best value bet
        if metrics["edge_pct"] > 0 and (best_value is None or metrics["edge_pct"] > best_value["edge_pct"]):
            best_value = {
                "outcome": outcome,
                "edge_pct": metrics["edge_pct"],
                "kelly_stake": metrics["kelly_stake"],
                "value_rating": metrics["value_rating"]
            }
        
        total_value_score += metrics["value_rating"]
    
    analysis["best_value_bet"] = best_value
    analysis["overall_value_score"] = total_value_score / 3  # Average across all outcomes
    
    return analysis

def create_betting_insights_summary(matches_analysis: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary DataFrame of betting insights.
    
    Args:
        matches_analysis: List of match analysis dictionaries
        
    Returns:
        DataFrame with betting insights summary
    """
    summary_data = []
    
    for analysis in matches_analysis:
        match_id = analysis["match_id"]
        home_team = analysis["home_team"]
        away_team = analysis["away_team"]
        match_date = analysis["match_date"]
        competition = analysis["competition"]
        value_score = analysis["overall_value_score"]
        best_bet = analysis["best_value_bet"]
        
        # Format match date
        if isinstance(match_date, str):
            try:
                match_date = pd.to_datetime(match_date).strftime("%Y-%m-%d %H:%M")
            except:
                match_date = str(match_date)
        elif isinstance(match_date, datetime):
            match_date = match_date.strftime("%Y-%m-%d %H:%M")
        
        summary_data.append({
            "match_id": match_id,
            "date": match_date,
            "competition": competition,
            "match": f"{home_team} vs {away_team}",
            "value_score": value_score,
            "best_bet_outcome": best_bet["outcome"] if best_bet else "None",
            "best_bet_edge": best_bet["edge_pct"] if best_bet else 0.0,
            "best_bet_kelly": best_bet["kelly_stake"] if best_bet else 0.0,
            "best_bet_rating": best_bet["value_rating"] if best_bet else 0.0
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        return df.sort_values(by="value_score", ascending=False).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=[
            "match_id", "date", "competition", "match", "value_score",
            "best_bet_outcome", "best_bet_edge", "best_bet_kelly", "best_bet_rating"
        ])

def render_betting_insights_dashboard(
    data_loader,
    selected_leagues: List[str] = None,
    date_range: Tuple[datetime.date, datetime.date] = None,
    min_value_score: float = 3.0
):
    """
    Render the main betting insights dashboard.
    """
    st.markdown(header("Betting Insights Dashboard", level=2, icon="🎯", accent_bar=True, animation='slide'), unsafe_allow_html=True)
    st.markdown(header("Actionable betting opportunities for the top 6 football leagues", level=4, accent_bar=False, divider=False), unsafe_allow_html=True)
    
    # Default to top 6 leagues if none selected
    if not selected_leagues:
        selected_leagues = list(TOP_LEAGUES.keys())
    
    # Default date range (next 7 days)
    if not date_range:
        today = datetime.now().date()
        date_range = (today, today + timedelta(days=7))
    
    # Load matches for selected leagues and date range
    try:
        with st.spinner("Loading matches..."):
            # Fix: Pass league names directly to load_matches
            matches_df = data_loader.load_matches(selected_leagues, date_range)
            
            if matches_df.empty:
                st.warning("No matches found for the selected criteria. Try adjusting the date range or league selection.")
                return
            
            st.success(f"✅ Loaded {len(matches_df)} matches for analysis")
            
    except Exception as e:
        st.error(f"Error loading matches: {e}")
        logger.error(f"Error loading matches: {e}")
        return
    
    # Validate and sanitize match data before analysis
    matches_analysis = []
    with st.spinner("Analyzing betting opportunities..."):
        for _, match in matches_df.iterrows():
            match_id = match.get("id")
            try:
                match_details = data_loader.load_match_details(match_id)
                if not match_details or match_details.get("error"):
                    continue
                prediction_data = match_details.get("prediction", {})
                odds_data = match_details.get("bookie_odds", {})
                if not prediction_data or not odds_data:
                    continue
                # Validate match data
                match_dict = validate_match_data(match.to_dict())
                analysis = analyze_match_betting_opportunities(
                    match_dict,
                    prediction_data,
                    odds_data
                )
                matches_analysis.append(analysis)
            except Exception as e:
                logger.warning(f"Error analyzing match {match_id}: {e}")
                continue
    
    if not matches_analysis:
        st.warning("No betting analysis available. This could be due to missing prediction or odds data.")
        return
    
    # Create summary DataFrame
    summary_df = create_betting_insights_summary(matches_analysis)
    
    # Filter by minimum value score
    filtered_df = summary_df[summary_df["value_score"] >= min_value_score]
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card("Total Matches Analyzed", str(len(matches_analysis)), help_text="Number of matches with complete prediction and odds data")
    
    with col2:
        create_metric_card("Value Opportunities", str(len(filtered_df)), help_text="Matches with value score above threshold")
    
    with col3:
        avg_value = summary_df["value_score"].mean()
        create_metric_card("Average Value Score", f"{avg_value:.1f}/10", help_text="Average betting value across all matches")
    
    with col4:
        max_edge = summary_df["best_bet_edge"].max()
        create_metric_card("Best Edge Found", f"{max_edge:.1f}%", help_text="Highest edge percentage found")
    
    # Display betting insights table
    st.subheader("📊 Top Betting Opportunities")
    
    if not filtered_df.empty:
        # Format the display
        display_df = filtered_df.copy()
        display_df["value_score"] = display_df["value_score"].round(1)
        display_df["best_bet_edge"] = display_df["best_bet_edge"].round(1)
        display_df["best_bet_kelly"] = (display_df["best_bet_kelly"] * 100).round(1)
        display_df["best_bet_rating"] = display_df["best_bet_rating"].round(1)
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            "value_score": "Value Score",
            "best_bet_outcome": "Best Bet",
            "best_bet_edge": "Edge (%)",
            "best_bet_kelly": "Kelly (%)",
            "best_bet_rating": "Rating"
        })
        
        st.dataframe(
            display_df[["date", "competition", "match", "Value Score", "Best Bet", "Edge (%)", "Kelly (%)", "Rating"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No high-value betting opportunities found. Try lowering the minimum value score threshold.")
    
    # Value score distribution chart
    st.subheader("📈 Value Score Distribution")
    
    fig = px.histogram(
        summary_df,
        x="value_score",
        nbins=10,
        title="Distribution of Betting Value Scores",
        labels={"value_score": "Value Score", "count": "Number of Matches"}
    )
    
    fig.add_vline(
        x=min_value_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({min_value_score})"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # League performance analysis
    st.subheader("🏆 League Performance Analysis")
    
    league_stats = summary_df.groupby("competition").agg({
        "value_score": ["mean", "count"],
        "best_bet_edge": "max"
    }).round(2)
    
    league_stats.columns = ["Avg Value Score", "Match Count", "Max Edge (%)"]
    league_stats = league_stats.sort_values("Avg Value Score", ascending=False)
    
    st.dataframe(league_stats, use_container_width=True)
    
    # Detailed match analysis
    st.subheader("🔍 Detailed Match Analysis")
    
    if not filtered_df.empty:
        selected_match = st.selectbox(
            "Select a match for detailed analysis:",
            options=filtered_df["match"].tolist(),
            index=0
        )
        
        if selected_match:
            match_analysis = next(
                (analysis for analysis in matches_analysis 
                 if f"{analysis['home_team']} vs {analysis['away_team']}" == selected_match),
                None
            )
            
            if match_analysis:
                themed_match_card(match_analysis)

def render_detailed_match_analysis(match_analysis: Dict[str, Any]):
    """
    Render detailed analysis for a specific match.
    
    Args:
        match_analysis: Match analysis dictionary
    """
    st.markdown(f"### {match_analysis['home_team']} vs {match_analysis['away_team']}")
    st.markdown(f"**Competition:** {match_analysis['competition']} | **Date:** {match_analysis['match_date']}")
    
    # Overall value score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Value Score",
            f"{match_analysis['overall_value_score']:.1f}/10"
        )
    
    with col2:
        best_bet = match_analysis['best_value_bet']
        if best_bet:
            st.metric(
                "Best Bet",
                best_bet['outcome'].replace('_', ' ').title()
            )
    
    with col3:
        if best_bet:
            st.metric(
                "Edge",
                f"{best_bet['edge_pct']:.1f}%"
            )
    
    # Betting opportunities breakdown
    st.markdown("#### Betting Opportunities Breakdown")
    
    opportunities = match_analysis['betting_opportunities']
    
    for outcome, data in opportunities.items():
        outcome_name = outcome.replace('_', ' ').title()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Outcome", outcome_name)
        
        with col2:
            st.metric("Model Prob", f"{data['prediction_prob']:.1%}")
        
        with col3:
            st.metric("Bookie Odds", f"{data['bookie_odds']:.2f}")
        
        with col4:
            st.metric("Edge", f"{data['edge_pct']:.1f}%")
        
        with col5:
            st.metric("Kelly", f"{data['kelly_stake']:.1%}")
        
        # Color coding for value bets
        if data['edge_pct'] > 0:
            # Render as a unified card to preserve consistent UI
            uds = get_unified_design_system()
            def _val():
                st.success(f"✅ Value bet opportunity: {data['edge_pct']:.1f}% edge")
            try:
                uds.create_unified_card(_val, card_class='goaldiggers-card')
            except Exception:
                st.success(f"✅ Value bet opportunity: {data['edge_pct']:.1f}% edge")
        else:
            st.info(f"ℹ️ No value: {data['edge_pct']:.1f}% edge")
        
        st.divider()

def render_league_selector() -> List[str]:
    """
    Render league selection interface.
    
    Returns:
        List of selected league names
    """
    st.sidebar.markdown("### 🏆 League Selection")
    
    # Default to top 6 leagues
    default_selection = list(TOP_LEAGUES.keys())
    
    selected_leagues = st.sidebar.multiselect(
        "Select leagues to analyze:",
        options=list(TOP_LEAGUES.keys()),
        default=default_selection,
        help="Choose which leagues to include in the betting analysis"
    )
    
    return selected_leagues

def render_date_selector() -> Tuple[datetime.date, datetime.date]:
    """
    Render date range selection interface.
    
    Returns:
        Tuple of (start_date, end_date)
    """
    st.sidebar.markdown("### 📅 Date Range")
    
    today = datetime.now().date()
    
    # Default to next 7 days
    default_start = today
    default_end = today + timedelta(days=7)
    
    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=today,
        max_value=today + timedelta(days=30)
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=default_end,
        min_value=start_date,
        max_value=start_date + timedelta(days=30)
    )
    
    return start_date, end_date

def render_value_threshold_selector() -> float:
    """
    Render value threshold selection interface.
    
    Returns:
        Minimum value score threshold
    """
    st.sidebar.markdown("### 🎯 Value Threshold")
    
    min_value_score = st.sidebar.slider(
        "Minimum Value Score",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5,
        help="Only show matches with value score above this threshold"
    )
    
    return min_value_score