"""
Enhanced Prediction UI components for the GoalDiggers dashboard.
Improved visualization, actionable betting insights, and performance optimization.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

logger = logging.getLogger(__name__)

def create_enhanced_prediction_results(predictions: Union[List[float], Dict], 
                                      match_data: Optional[Dict] = None,
                                      home_team: Optional[str] = None, 
                                      away_team: Optional[str] = None):
    """
    Display enhanced prediction results with advanced visualizations and insights.
    
    Args:
        predictions: Either a list of 3 probabilities [home_win, draw, away_win]
                   or a dictionary with keys 'home_win', 'draw', 'away_win'
        match_data: Optional dictionary with additional match data
        home_team: Home team name
        away_team: Away team name
    
    Returns:
        None - displays the UI elements directly
    """
    try:
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
        
        # Create columns for displaying predictions with enhanced styling
        st.markdown("### 🎯 Match Outcome Prediction")
        
        # Create visually enhanced metrics with color coding
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color = _get_probability_color(home_win_prob)
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding-left: 10px;">
                <h4 style="margin: 0;">Home Win</h4>
                <p style="font-size: 24px; font-weight: bold; color: {color};">{home_win_prob:.1%}</p>
                <p style="color: #666; font-size: 14px;">{_get_probability_label(home_win_prob)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if home_team:
                st.caption(f"{home_team}")
        
        with col2:
            color = _get_probability_color(draw_prob)
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding-left: 10px;">
                <h4 style="margin: 0;">Draw</h4>
                <p style="font-size: 24px; font-weight: bold; color: {color};">{draw_prob:.1%}</p>
                <p style="color: #666; font-size: 14px;">{_get_probability_label(draw_prob)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            color = _get_probability_color(away_win_prob)
            st.markdown(f"""
            <div style="border-left: 5px solid {color}; padding-left: 10px;">
                <h4 style="margin: 0;">Away Win</h4>
                <p style="font-size: 24px; font-weight: bold; color: {color};">{away_win_prob:.1%}</p>
                <p style="color: #666; font-size: 14px;">{_get_probability_label(away_win_prob)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if away_team:
                st.caption(f"{away_team}")
        
        # Add an enhanced visualization using a horizontal bar chart with percentage labels
        st.markdown("#### 📊 Prediction Visualization")
        
        chart_data = pd.DataFrame({
            'Outcome': ['Home Win', 'Draw', 'Away Win'],
            'Probability': [home_win_prob, draw_prob, away_win_prob]
        })
        
        # Sort by probability for better visualization
        chart_data = chart_data.sort_values('Probability', ascending=False)
        
        # Enhanced horizontal bar chart with team names if available
        labels = chart_data['Outcome'].tolist()
        if home_team and away_team:
            labels = [l.replace('Home Win', f'{home_team} Win').replace('Away Win', f'{away_team} Win') for l in labels]
        
        fig, ax = plt.figure(figsize=(8, 3)), plt.axes()
        bars = ax.barh(labels, chart_data['Probability'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add percentage labels to the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                    va='center', color='black', fontweight='bold')
        
        st.pyplot(fig)
        
        # Display the most likely outcome with a callout
        most_likely_idx = np.argmax([home_win_prob, draw_prob, away_win_prob])
        outcomes = ['Home Win', 'Draw', 'Away Win']
        teams = [home_team, "Draw", away_team]
        probs = [home_win_prob, draw_prob, away_win_prob]
        
        most_likely = outcomes[most_likely_idx]
        confidence = probs[most_likely_idx]
        team_name = teams[most_likely_idx] if teams[most_likely_idx] else most_likely
        
        # Only show this if we have high confidence
        if confidence > 0.4:
            st.success(f"📈 **Most Likely Outcome**: {team_name} ({confidence:.1%} probability)")
        
        # Add expected goals prediction if available
        if match_data and 'expected_goals' in match_data:
            st.markdown("#### ⚽ Expected Goals (xG)")
            xg_home = match_data['expected_goals'].get('home', 1.2)
            xg_away = match_data['expected_goals'].get('away', 0.8)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{home_team} xG", f"{xg_home:.1f}")
            with col2:
                st.metric(f"{away_team} xG", f"{xg_away:.1f}")
            
            # Add most likely scoreline
            most_likely_score = _predict_most_likely_score(xg_home, xg_away)
            st.info(f"🎯 Most likely scoreline: **{most_likely_score[0]}-{most_likely_score[1]}**")
    
    except Exception as e:
        logger.error(f"Error displaying enhanced prediction results: {e}")
        st.error("Error displaying enhanced predictions. Please try again.")

def create_advanced_metrics_grid(metrics: Dict[str, Dict[str, Any]]):
    """
    Create a grid of advanced metrics with descriptions, trends, and styling.
    
    Args:
        metrics: Dictionary of metric categories with their values
    
    Returns:
        None - displays the UI elements directly
    """
    # Group metrics by category for better organization
    for category, category_metrics in metrics.items():
        st.markdown(f"#### {category}")
        
        # Calculate columns based on metrics count
        num_metrics = len(category_metrics)
        cols_per_row = min(3, num_metrics)
        
        # Create enough rows for all metrics
        rows_needed = (num_metrics + cols_per_row - 1) // cols_per_row
        metric_items = list(category_metrics.items())
        
        # Create grid layout
        for row in range(rows_needed):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                metric_idx = row * cols_per_row + col_idx
                
                # Check if we still have metrics to display
                if metric_idx < num_metrics:
                    title, metric_data = metric_items[metric_idx]
                    
                    # Extract metric values and metadata
                    if isinstance(metric_data, dict):
                        value = metric_data.get('value', 'N/A')
                        delta = metric_data.get('delta')
                        tooltip = metric_data.get('tooltip', '')
                        color = metric_data.get('color')
                    else:
                        value = metric_data
                        delta = None
                        tooltip = ''
                        color = None
                    
                    # Apply styling based on metadata
                    with cols[col_idx]:
                        if delta is not None:
                            st.metric(title, value, delta=delta, help=tooltip)
                        else:
                            st.metric(title, value, help=tooltip)
                        
                        # Add additional context if available
                        if isinstance(metric_data, dict) and 'context' in metric_data:
                            st.caption(metric_data['context'])

def create_enhanced_odds_comparison(bookmaker_odds: Dict[str, Dict[str, float]], 
                                   our_prediction: List[float],
                                   home_team: str,
                                   away_team: str,
                                   value_threshold: float = 0.1):
    """
    Create an enhanced comparison between bookmaker odds and our predictions
    with improved visualization and actionable betting insights.
    
    Args:
        bookmaker_odds: Dictionary of bookmakers with their odds
        our_prediction: Our prediction [home_win, draw, away_win]
        home_team: Home team name
        away_team: Away team name
        value_threshold: Threshold for identifying value bets (default 10%)
    
    Returns:
        None - displays the UI elements directly
    """
    st.markdown("### 🏆 Odds Comparison & Betting Value")
    
    # Extract our predictions
    home_win_prob, draw_prob, away_win_prob = our_prediction
    
    # Calculate implied probabilities and fair odds for our predictions
    our_implied_probs = {
        "home": home_win_prob,
        "draw": draw_prob,
        "away": away_win_prob
    }
    
    our_fair_odds = {
        "home": 1 / home_win_prob if home_win_prob > 0 else 0,
        "draw": 1 / draw_prob if draw_prob > 0 else 0,
        "away": 1 / away_win_prob if away_win_prob > 0 else 0
    }
    
    # Prepare data for display
    bookmakers = list(bookmaker_odds.keys())
    
    # Create dataframe for odds comparison with implied probabilities
    odds_data = []
    implied_probs_data = []
    
    for bookie in bookmakers:
        bookie_odds = bookmaker_odds[bookie]
        
        # Calculate implied probabilities (accounting for overround)
        home_odds = bookie_odds.get("home", 0)
        draw_odds = bookie_odds.get("draw", 0)
        away_odds = bookie_odds.get("away", 0)
        
        if home_odds > 0 and draw_odds > 0 and away_odds > 0:
            # Calculate raw implied probabilities
            raw_home_prob = 1 / home_odds
            raw_draw_prob = 1 / draw_odds
            raw_away_prob = 1 / away_odds
            
            # Calculate overround
            overround = raw_home_prob + raw_draw_prob + raw_away_prob
            
            # Normalize probabilities to account for overround
            implied_home_prob = raw_home_prob / overround
            implied_draw_prob = raw_draw_prob / overround
            implied_away_prob = raw_away_prob / overround
            
            # Add to odds data
            odds_data.append({
                "Bookmaker": bookie,
                f"{home_team} Win": f"{home_odds:.2f}",
                "Draw": f"{draw_odds:.2f}",
                f"{away_team} Win": f"{away_odds:.2f}",
                "Overround": f"{(overround-1)*100:.1f}%"
            })
            
            # Add to implied probabilities data
            implied_probs_data.append({
                "Bookmaker": bookie,
                f"{home_team} Win": f"{implied_home_prob:.1%}",
                "Draw": f"{implied_draw_prob:.1%}",
                f"{away_team} Win": f"{implied_away_prob:.1%}"
            })
    
    # Add our model
    odds_data.append({
        "Bookmaker": "Our Model (Fair Odds)",
        f"{home_team} Win": f"{our_fair_odds['home']:.2f}",
        "Draw": f"{our_fair_odds['draw']:.2f}",
        f"{away_team} Win": f"{our_fair_odds['away']:.2f}",
        "Overround": "0.0%"
    })
    
    implied_probs_data.append({
        "Bookmaker": "Our Model",
        f"{home_team} Win": f"{our_implied_probs['home']:.1%}",
        "Draw": f"{our_implied_probs['draw']:.1%}",
        f"{away_team} Win": f"{our_implied_probs['away']:.1%}"
    })
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Odds Comparison", "🔄 Implied Probabilities", "💰 Value Bets"])
    
    with tab1:
        # Display odds table
        st.markdown("#### 📊 Bookmaker Odds")
        st.caption("Lower odds mean lower potential payout but higher implied probability")
        odds_df = pd.DataFrame(odds_data)
        st.table(odds_df)
    
    with tab2:
        # Display implied probabilities table
        st.markdown("#### 🔄 Implied Probabilities")
        st.caption("Higher percentage means the bookmaker thinks this outcome is more likely")
        implied_probs_df = pd.DataFrame(implied_probs_data)
        st.table(implied_probs_df)
    
    with tab3:
        # Find value bets
        value_bets = _find_value_bets(
            bookmaker_odds, 
            our_prediction,
            home_team, 
            away_team,
            threshold=value_threshold
        )
        
        st.markdown("#### 💎 Value Betting Opportunities")
        st.caption("""Value bets are identified when our model's probability estimate 
                   is significantly higher than the bookmaker's implied probability.""")
        
        if value_bets:
            # Sort value bets by expected value
            sorted_value_bets = sorted(value_bets, key=lambda x: x['expected_value'], reverse=True)
            
            for i, bet in enumerate(sorted_value_bets):
                with st.container():
                    expected_value = bet['expected_value']
                    
                    # Color code based on expected value
                    if expected_value >= 0.2:  # Strong value (20%+)
                        color = "#28a745"  # Green
                        strength = "Strong"
                    elif expected_value >= 0.1:  # Good value (10-20%)
                        color = "#17a2b8"  # Blue
                        strength = "Good"
                    else:  # Moderate value (<10%)
                        color = "#ffc107"  # Yellow
                        strength = "Moderate"
                    
                    st.markdown(f"""
                    <div style="border-left: 5px solid {color}; padding-left: 10px; margin-bottom: 15px;">
                        <h4 style="margin: 0;">{strength} Value Bet ({i+1}/{len(sorted_value_bets)})</h4>
                        <p style="font-size: 16px; font-weight: bold;">{bet['description']}</p>
                        <p><strong>Expected Value:</strong> +{expected_value:.1%}</p>
                        <p><strong>Our Probability:</strong> {bet['our_probability']:.1%} vs. 
                        <strong>Implied Probability:</strong> {bet['implied_probability']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No significant value betting opportunities identified for this match")
            st.markdown("""
            **Why no value bets?**
            - Bookmaker odds align closely with our model's predictions
            - Bookmaker margins (overround) may absorb potential value
            - Try adjusting the value threshold or wait for odds movement
            """)

def create_betting_strategy_recommendations(
    predictions: Union[List[float], Dict],
    match_data: Dict,
    bankroll_size: float = 1000.0,
    risk_tolerance: str = 'moderate'
):
    """
    Create actionable betting strategy recommendations based on predictions and match data.
    
    Args:
        predictions: Model predictions as list [home_win, draw, away_win] or dictionary
        match_data: Match data including teams, odds, etc.
        bankroll_size: Total bankroll size (default 1000)
        risk_tolerance: Risk tolerance level (low, moderate, high)
        
    Returns:
        None - displays the UI elements directly
    """
    try:
        st.markdown("### 💰 Betting Strategy Recommendations")
        
        # Extract predictions
        if isinstance(predictions, list) and len(predictions) >= 3:
            home_win_prob = predictions[0]
            draw_prob = predictions[1]
            away_win_prob = predictions[2]
        else:
            home_win_prob = predictions.get('home_win', 0.33)
            draw_prob = predictions.get('draw', 0.33)
            away_win_prob = predictions.get('away_win', 0.33)
        
        # Extract team names and odds
        home_team = match_data.get('home_team', 'Home Team')
        away_team = match_data.get('away_team', 'Away Team')
        
        # Default odds if not provided
        default_odds = {'home': 2.0, 'draw': 3.5, 'away': 4.0}
        odds = match_data.get('odds', {}).get('best_odds', default_odds)
        
        # Calculate best betting opportunities
        bet_opportunities = []
        
        # Calculate expected value for each outcome
        home_ev = (home_win_prob * odds['home']) - 1
        draw_ev = (draw_prob * odds['draw']) - 1
        away_ev = (away_win_prob * odds['away']) - 1
        
        # Only recommend positive EV bets
        if home_ev > 0:
            bet_opportunities.append({
                'outcome': f"{home_team} Win",
                'odds': odds['home'],
                'probability': home_win_prob,
                'ev': home_ev,
                'stake': _calculate_recommended_stake(home_ev, bankroll_size, risk_tolerance)
            })
            
        if draw_ev > 0:
            bet_opportunities.append({
                'outcome': "Draw",
                'odds': odds['draw'],
                'probability': draw_prob,
                'ev': draw_ev,
                'stake': _calculate_recommended_stake(draw_ev, bankroll_size, risk_tolerance)
            })
            
        if away_ev > 0:
            bet_opportunities.append({
                'outcome': f"{away_team} Win",
                'odds': odds['away'],
                'probability': away_win_prob,
                'ev': away_ev,
                'stake': _calculate_recommended_stake(away_ev, bankroll_size, risk_tolerance)
            })
        
        # Sort by expected value
        bet_opportunities.sort(key=lambda x: x['ev'], reverse=True)
        
        if bet_opportunities:
            # Display betting recommendations
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.metric("Bankroll", f"£{bankroll_size:.2f}")
                st.markdown("**Risk Level:** " + risk_tolerance.capitalize())
                
                # Add option to change risk tolerance
                new_risk = st.radio(
                    "Adjust Risk Level:",
                    ["low", "moderate", "high"],
                    index=["low", "moderate", "high"].index(risk_tolerance)
                )
                
                if new_risk != risk_tolerance:
                    # Recalculate stakes with new risk level
                    for bet in bet_opportunities:
                        bet['stake'] = _calculate_recommended_stake(
                            bet['ev'], bankroll_size, new_risk
                        )
            
            with col2:
                st.markdown("#### 🎯 Recommended Bets")
                
                for i, bet in enumerate(bet_opportunities):
                    with st.container():
                        st.markdown(f"""
                        <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; border-radius: 5px;">
                            <div style="display: flex; justify-content: space-between;">
                                <h4 style="margin: 0;">{bet['outcome']}</h4>
                                <h4 style="margin: 0; color: #28a745;">@{bet['odds']:.2f}</h4>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                                <span><strong>Probability:</strong> {bet['probability']:.1%}</span>
                                <span><strong>Expected Value:</strong> +{bet['ev']:.1%}</span>
                            </div>
                            <div style="margin-top: 10px; text-align: center; background-color: #e9ecef; padding: 5px; border-radius: 3px;">
                                <span style="font-weight: bold;">Recommended Stake: £{bet['stake']:.2f}</span>
                                <span style="display: block; font-size: 0.8em; color: #666;">
                                    {bet['stake']/bankroll_size:.1%} of bankroll
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add betting strategy explanation
            with st.expander("Understanding These Recommendations"):
                st.markdown("""
                ### How We Calculate Bet Recommendations
                
                1. **Expected Value (EV)**: We calculate the expected value of each bet using our model's probability and the available odds.
                   - Formula: `EV = (Probability * Odds) - 1`
                   - Positive EV means the bet has long-term value
                
                2. **Stake Sizing**: We use the Kelly Criterion (adjusted for risk tolerance) to recommend optimal stake sizes.
                   - Conservative: 1/4 of Kelly recommendation
                   - Moderate: 1/2 of Kelly recommendation
                   - Aggressive: Full Kelly recommendation
                
                3. **Bankroll Management**: Recommendations are proportional to your total bankroll to ensure responsible betting.
                
                > Remember: These are recommendations based on mathematical models and historical data. 
                > Always bet responsibly and within your means.
                """)
        else:
            st.warning("No positive expected value bets identified for this match")
            st.markdown("""
            **Why no betting recommendations?**
            - All calculated expected values are negative
            - This suggests bookmaker odds are not favorable
            - Recommendation: Skip betting on this match or look for other markets
            """)
            
    except Exception as e:
        logger.error(f"Error creating betting strategy recommendations: {e}")
        st.error("Error generating betting strategy recommendations. Please try again.")

def create_confidence_meter(confidence: float, explanation: str = None):
    """
    Create a visual confidence meter showing the model's confidence in the prediction.
    
    Args:
        confidence: Confidence value (0-1)
        explanation: Optional explanation text
        
    Returns:
        None - displays the UI elements directly
    """
    st.markdown("### 🎯 Prediction Confidence")
    
    # Determine color and label based on confidence level
    if confidence >= 0.8:
        color = "#28a745"  # Green
        label = "Very High"
    elif confidence >= 0.65:
        color = "#17a2b8"  # Blue
        label = "High"
    elif confidence >= 0.5:
        color = "#ffc107"  # Yellow
        label = "Moderate"
    else:
        color = "#dc3545"  # Red
        label = "Low"
    
    # Create a visual meter
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center;">
            <h4 style="margin: 0; color: {color};">{label}</h4>
            <p style="font-size: 24px; font-weight: bold; color: {color};">{confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create a progress bar
        st.progress(confidence)
        
        # Add explanation if provided
        if explanation:
            st.caption(explanation)
        else:
            if confidence >= 0.8:
                st.caption("Very high confidence predictions are typically more reliable.")
            elif confidence >= 0.65:
                st.caption("High confidence predictions have strong statistical support.")
            elif confidence >= 0.5:
                st.caption("Moderate confidence suggests some uncertainty in the prediction.")
            else:
                st.caption("Low confidence indicates high uncertainty. Bet with caution.")

def create_historical_performance_summary(
    model_performance: Dict[str, Any],
    prediction_type: str = "match outcome"
):
    """
    Create a summary of the model's historical performance for context.
    
    Args:
        model_performance: Dictionary with performance metrics
        prediction_type: Type of prediction being evaluated
        
    Returns:
        None - displays the UI elements directly
    """
    st.markdown("### 📈 Model Performance History")
    
    # Extract key metrics
    accuracy = model_performance.get('accuracy', 0.65)
    precision = model_performance.get('precision', 0.62)
    recall = model_performance.get('recall', 0.64)
    f1_score = model_performance.get('f1_score', 0.63)
    
    # Display metrics in a grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
        st.caption("Correct predictions / Total predictions")
    
    with col2:
        st.metric("Precision", f"{precision:.1%}")
        st.caption("True positives / Predicted positives")
    
    with col3:
        st.metric("Recall", f"{recall:.1%}")
        st.caption("True positives / Actual positives")
    
    with col4:
        st.metric("F1 Score", f"{f1_score:.1%}")
        st.caption("Harmonic mean of precision and recall")
    
    # Create a performance breakdown by bet type if available
    if 'performance_by_type' in model_performance:
        performance_by_type = model_performance['performance_by_type']
        
        st.markdown("#### Performance by Bet Type")
        
        # Create dataframe for visualization
        perf_data = []
        for bet_type, metrics in performance_by_type.items():
            perf_data.append({
                'Bet Type': bet_type,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1 Score': metrics.get('f1_score', 0)
            })
        
        perf_df = pd.DataFrame(perf_data)
        
        # Display as a styled table
        st.dataframe(perf_df.style.format({
            'Accuracy': '{:.1%}',
            'Precision': '{:.1%}',
            'Recall': '{:.1%}',
            'F1 Score': '{:.1%}'
        }))
    
    # Add historical profitability if available
    if 'profitability' in model_performance:
        profit_data = model_performance['profitability']
        
        st.markdown("#### Historical Profitability")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            roi = profit_data.get('roi', 0.0)
            delta_color = "normal" if roi >= 0 else "inverse"
            st.metric("ROI", f"{roi:.1%}", delta=f"{roi:.1%}", delta_color=delta_color)
        
        with col2:
            total_bets = profit_data.get('total_bets', 0)
            winning_bets = profit_data.get('winning_bets', 0)
            win_rate = winning_bets / total_bets if total_bets > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1%}")
            st.caption(f"{winning_bets} / {total_bets} bets")
        
        with col3:
            profit_per_bet = profit_data.get('profit_per_bet', 0.0)
            delta_color = "normal" if profit_per_bet >= 0 else "inverse"
            st.metric("Avg Profit/Bet", f"£{profit_per_bet:.2f}", delta=f"{profit_per_bet:.2f}", delta_color=delta_color)
    
    # Add explanation
    with st.expander("Understanding Model Performance Metrics"):
        st.markdown("""
        ### How to Interpret These Metrics
        
        **Accuracy:** The proportion of predictions that were correct. While important, accuracy alone can be misleading for unbalanced outcome distributions.
        
        **Precision:** When the model predicts a specific outcome, how often is it correct? High precision means fewer false positives.
        
        **Recall:** Of all the actual instances of an outcome, how many did the model correctly identify? High recall means fewer false negatives.
        
        **F1 Score:** The harmonic mean of precision and recall, providing a balance between the two metrics.
        
        **ROI (Return on Investment):** The percentage return on each unit wagered when following the model's recommendations.
        
        **Win Rate:** The percentage of bets that resulted in a win.
        
        **Avg Profit/Bet:** The average profit (or loss) per bet placed.
        """)

# Helper functions for the enhanced components

def _get_probability_color(probability: float) -> str:
    """Get color based on probability value."""
    if probability >= 0.6:
        return "#28a745"  # Green
    elif probability >= 0.4:
        return "#17a2b8"  # Blue
    elif probability >= 0.25:
        return "#ffc107"  # Yellow
    else:
        return "#6c757d"  # Gray

def _get_probability_label(probability: float) -> str:
    """Get label based on probability value."""
    if probability >= 0.6:
        return "Strong Probability"
    elif probability >= 0.4:
        return "Moderate Probability"
    elif probability >= 0.25:
        return "Slight Probability"
    else:
        return "Low Probability"

def _predict_most_likely_score(xg_home: float, xg_away: float) -> Tuple[int, int]:
    """Predict the most likely score based on expected goals."""
    # This is a simplified Poisson distribution-based approach
    # In a real implementation, this would use a proper statistical model

    # Calculate probabilities for each score up to 5 goals
    max_goals = 5
    home_probs = [poisson.pmf(i, xg_home) for i in range(max_goals + 1)]
    away_probs = [poisson.pmf(i, xg_away) for i in range(max_goals + 1)]
    
    # Calculate joint probabilities for all score combinations
    score_probs = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            score_probs[(h, a)] = home_probs[h] * away_probs[a]
    
    # Find the most likely score
    most_likely_score = max(score_probs.items(), key=lambda x: x[1])[0]
    return most_likely_score

def _find_value_bets(bookmaker_odds: Dict[str, Dict[str, float]], 
                    our_prediction: List[float],
                    home_team: str,
                    away_team: str,
                    threshold: float = 0.1) -> List[Dict]:
    """
    Find value betting opportunities by comparing our predictions with bookmaker odds.
    
    Args:
        bookmaker_odds: Dictionary of bookmakers with their odds
        our_prediction: Our prediction [home_win, draw, away_win]
        home_team: Home team name
        away_team: Away team name
        threshold: Value threshold (default 10%)
        
    Returns:
        List of value betting opportunities
    """
    value_bets = []
    
    # Extract our predictions
    home_win_prob, draw_prob, away_win_prob = our_prediction
    
    for bookie, odds in bookmaker_odds.items():
        # Check home win value
        if odds.get("home", 0) > 0:
            bookie_implied_prob = 1 / odds["home"]
            if home_win_prob > bookie_implied_prob * (1 + threshold):
                value_bets.append({
                    'bookmaker': bookie,
                    'bet_type': 'home_win',
                    'outcome': f"{home_team} Win",
                    'odds': odds["home"],
                    'our_probability': home_win_prob,
                    'implied_probability': bookie_implied_prob,
                    'expected_value': home_win_prob - bookie_implied_prob,
                    'description': f"{home_team} Win @ {odds['home']:.2f} with {bookie}"
                })
        
        # Check draw value
        if odds.get("draw", 0) > 0:
            bookie_implied_prob = 1 / odds["draw"]
            if draw_prob > bookie_implied_prob * (1 + threshold):
                value_bets.append({
                    'bookmaker': bookie,
                    'bet_type': 'draw',
                    'outcome': "Draw",
                    'odds': odds["draw"],
                    'our_probability': draw_prob,
                    'implied_probability': bookie_implied_prob,
                    'expected_value': draw_prob - bookie_implied_prob,
                    'description': f"Draw @ {odds['draw']:.2f} with {bookie}"
                })
        
        # Check away win value
        if odds.get("away", 0) > 0:
            bookie_implied_prob = 1 / odds["away"]
            if away_win_prob > bookie_implied_prob * (1 + threshold):
                value_bets.append({
                    'bookmaker': bookie,
                    'bet_type': 'away_win',
                    'outcome': f"{away_team} Win",
                    'odds': odds["away"],
                    'our_probability': away_win_prob,
                    'implied_probability': bookie_implied_prob,
                    'expected_value': away_win_prob - bookie_implied_prob,
                    'description': f"{away_team} Win @ {odds['away']:.2f} with {bookie}"
                })
    
    return value_bets

def _calculate_recommended_stake(expected_value: float, 
                                bankroll: float,
                                risk_tolerance: str = 'moderate') -> float:
    """
    Calculate recommended stake using Kelly Criterion adjusted for risk tolerance.
    
    Args:
        expected_value: Expected value of the bet
        bankroll: Total bankroll
        risk_tolerance: Risk tolerance (low, moderate, high)
        
    Returns:
        Recommended stake amount
    """
    # Base Kelly calculation: f* = p - q/b where:
    # f* is the fraction of the bankroll to bet
    # p is the probability of winning
    # q is the probability of losing (1-p)
    # b is the odds received on the wager (odds - 1)
    
    # For this simplified version, we'll use expected value directly
    # Typical Kelly Criterion suggests betting f* = edge/odds of your bankroll
    
    if expected_value <= 0:
        return 0
    
    # Base Kelly stake
    kelly_stake = expected_value * bankroll
    
    # Adjust for risk tolerance
    if risk_tolerance == 'low':
        adjusted_stake = kelly_stake * 0.25  # Quarter Kelly
    elif risk_tolerance == 'moderate':
        adjusted_stake = kelly_stake * 0.5   # Half Kelly
    else:  # high risk tolerance
        adjusted_stake = kelly_stake         # Full Kelly
    
    # Cap at reasonable percentage of bankroll
    max_stake = bankroll * 0.1  # Never risk more than 10% of bankroll
    return min(adjusted_stake, max_stake)
