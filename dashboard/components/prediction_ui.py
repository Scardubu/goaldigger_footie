"""
Prediction UI components for the GoalDiggers dashboard.
"""


import logging
from typing import Dict, List, Union

import pandas as pd
import streamlit as st

from dashboard.components.unified_design_system import \
    get_unified_design_system

logger = logging.getLogger(__name__)

def create_prediction_results(predictions: Union[List[float], Dict]):
    """
    Display prediction results in a visually appealing format.
    
    Args:
        predictions: Either a list of 3 probabilities [home_win, draw, away_win]
                   or a dictionary with keys 'home_win', 'draw', 'away_win'
    
    Returns:
        None - displays the UI elements directly
    """
    try:
        uds = get_unified_design_system()
        uds.inject_unified_css('integrated')
        # Extract predictions
        if isinstance(predictions, list) and len(predictions) == 3:
            home_win_prob = predictions[0]
            draw_prob = predictions[1]
            away_win_prob = predictions[2]
        else:
            # Handle dictionary format
            home_win_prob = predictions.get('home_win', 0.33)
            draw_prob = predictions.get('draw', 0.33)
            away_win_prob = predictions.get('away_win', 0.33)
        uds.create_unified_header("⚽ Prediction Results")
        def card_content():
            st.markdown("<div style='text-align:center;font-size:1.1em;'>Model probabilities for each outcome</div>", unsafe_allow_html=True)
        uds.create_unified_card(card_content)
        metrics = {
            "🏠 Home Win": f"{home_win_prob:.1%}",
            "🤝 Draw": f"{draw_prob:.1%}",
            "✈️ Away Win": f"{away_win_prob:.1%}"
        }
        uds.create_unified_metric_row(metrics)
        st.markdown("#### Prediction Visualization")
        chart_data = pd.DataFrame({
            'Outcome': ['Home Win', 'Draw', 'Away Win'],
            'Probability': [home_win_prob, draw_prob, away_win_prob]
        })
        st.bar_chart(chart_data.set_index('Outcome'))
    except Exception as e:
        logger.error(f"Error displaying prediction results: {e}")
        st.error("Error displaying predictions. Please try again.")

def create_metrics_grid(metrics: Dict[str, str]):
    """
    Create a grid of metrics using st.columns and st.metric.
    
    Args:
        metrics: Dictionary of metric titles and values
    
    Returns:
        None - displays the UI elements directly
    """
    uds = get_unified_design_system()
    uds.inject_unified_css('integrated')
    uds.create_unified_metric_row(metrics)

def create_odds_comparison(bookmaker_odds: Dict[str, Dict[str, float]], 
                          our_prediction: List[float],
                          home_team: str,
                          away_team: str):
    """
    Create a comparison between bookmaker odds and our predictions.
    
    Args:
        bookmaker_odds: Dictionary of bookmakers with their odds
        our_prediction: Our prediction [home_win, draw, away_win]
        home_team: Home team name
        away_team: Away team name
    
    Returns:
        None - displays the UI elements directly
    """
    uds = get_unified_design_system()
    uds.inject_unified_css('integrated')
    uds.create_unified_header("🏆 Odds Comparison")
    # Extract our predictions
    home_win_prob, draw_prob, away_win_prob = our_prediction
    # Calculate implied probabilities for our predictions
    our_odds = {
        "home": 1 / home_win_prob if home_win_prob > 0 else 0,
        "draw": 1 / draw_prob if draw_prob > 0 else 0,
        "away": 1 / away_win_prob if away_win_prob > 0 else 0
    }
    # Prepare data for display
    bookmakers = list(bookmaker_odds.keys()) + ["Our Model"]
    # Create dataframe for odds comparison
    odds_data = []
    for bookie in bookmakers[:-1]:  # Exclude "Our Model" here
        bookie_odds = bookmaker_odds[bookie]
        odds_data.append({
            "Bookmaker": bookie,
            f"{home_team} Win": bookie_odds.get("home", 0),
            "Draw": bookie_odds.get("draw", 0),
            f"{away_team} Win": bookie_odds.get("away", 0)
        })
    # Add our model
    odds_data.append({
        "Bookmaker": "Our Model",
        f"{home_team} Win": our_odds["home"],
        "Draw": our_odds["draw"],
        f"{away_team} Win": our_odds["away"]
    })
    # Display as table
    odds_df = pd.DataFrame(odds_data)
    st.table(odds_df)
    # Find value bets
    value_threshold = 0.1  # 10% difference
    value_bets = []
    for bookie in bookmakers[:-1]:
        bookie_odds = bookmaker_odds[bookie]
        # Check home win value
        if bookie_odds.get("home", 0) > 0:
            bookie_implied_prob = 1 / bookie_odds["home"]
            if home_win_prob > bookie_implied_prob * (1 + value_threshold):
                value_bets.append(f"Value on {home_team} Win @ {bookie_odds['home']:.2f} with {bookie}")
        # Check draw value
        if bookie_odds.get("draw", 0) > 0:
            bookie_implied_prob = 1 / bookie_odds["draw"]
            if draw_prob > bookie_implied_prob * (1 + value_threshold):
                value_bets.append(f"Value on Draw @ {bookie_odds['draw']:.2f} with {bookie}")
        # Check away win value
        if bookie_odds.get("away", 0) > 0:
            bookie_implied_prob = 1 / bookie_odds["away"]
            if away_win_prob > bookie_implied_prob * (1 + value_threshold):
                value_bets.append(f"Value on {away_team} Win @ {bookie_odds['away']:.2f} with {bookie}")
    # Display value bets
    if value_bets:
        uds.create_unified_header("💎 Value Betting Opportunities")
        def card_content():
            for bet in value_bets:
                st.success(bet)
        uds.create_unified_card(card_content)
    else:
        st.info("No significant value betting opportunities identified")
