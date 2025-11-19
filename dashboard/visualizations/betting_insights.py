"""
Enhanced betting insights visualizations.
Creates interactive plots for match predictions, value bets, and feature importance.
"""
import logging
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.error_log import log_error
from dashboard.visualizations.plots import GOALDIGGERS_THEME, TEMPLATE_NAME

logger = logging.getLogger(__name__)


def create_value_bet_visualization(
    match_data: Dict[str, Any],
    prediction: Dict[str, float],
    odds: Dict[str, float],
    value_analysis: Dict[str, Any]
) -> go.Figure:
    """
    Create an interactive, themed visualization of value betting opportunities.
    """
    try:
        value_bets = value_analysis.get("value_bets", [])
        home_team = match_data.get("home_team", "Home")
        away_team = match_data.get("away_team", "Away")

        if not value_bets or not odds or not prediction:
            fig = go.Figure()
            fig.update_layout(
                title="No value betting opportunities identified",
                template=TEMPLATE_NAME,
                height=350
            )
            return fig

        outcomes, implied_probs, predicted_probs, edges, colors = [], [], [], [], []

        color_map = {
            "High": GOALDIGGERS_THEME["win_color"],
            "Medium": GOALDIGGERS_THEME["accent_color"],
            "Low": GOALDIGGERS_THEME["draw_color"]
        }

        for outcome in ["home_win", "draw", "away_win"]:
            if outcome in odds and outcome in prediction:
                display_name = outcome.replace("_", " ").title().replace("Home Win", home_team).replace("Away Win", away_team)
                
                odd = odds[outcome]
                implied_prob = 1.0 / odd if odd > 1.0 else 0
                predicted_prob = prediction[outcome]
                edge = (predicted_prob * odd) - 1.0 if odd > 1.0 else 0

                value_bet = next((vb for vb in value_bets if vb["outcome"] == outcome), None)
                confidence = value_bet["confidence"] if value_bet else "Low"

                outcomes.append(display_name)
                implied_probs.append(implied_prob * 100)
                predicted_probs.append(predicted_prob * 100)
                edges.append(edge * 100)
                colors.append(color_map.get(confidence, GOALDIGGERS_THEME["draw_color"]))
        
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{"type": "bar"}, {"type": "bar"}]],
            subplot_titles=["Probability Comparison", "Value Edge (%)"]
        )

        fig.add_trace(
            go.Bar(
                x=outcomes, y=implied_probs, name="Implied by Odds",
                marker_color=GOALDIGGERS_THEME["text_color"], opacity=0.7,
                hovertemplate='%{x}<br>Implied Probability: %{y:.1f}%<extra></extra>'
            ), row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=outcomes, y=predicted_probs, name="GoalDiggers AI",
                marker_color=GOALDIGGERS_THEME["accent_color"],
                hovertemplate='%{x}<br>AI Probability: %{y:.1f}%<extra></extra>'
            ), row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=outcomes, y=edges, name="Edge",
                marker_color=colors,
                hovertemplate='%{x}<br>Edge: %{y:+.1f}%<extra></extra>'
            ), row=1, col=2
        )

        fig.add_shape(
            type="line", x0=-0.5, x1=len(outcomes) - 0.5, y0=0, y1=0,
            line=dict(color=GOALDIGGERS_THEME["text_color"], width=1, dash="dash"),
            row=1, col=2
        )

        fig.update_layout(
            title="Value Betting Analysis",
            barmode="group",
            height=350,
            template=TEMPLATE_NAME,
            margin=dict(l=40, r=20, t=60, b=40),
            legend=dict(y=1.15)
        )
        fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
        fig.update_yaxes(title_text="Edge (%)", row=1, col=2)

        return fig

    except Exception as e:
        log_error("Error creating value bet visualization", e)
        return go.Figure().update_layout(title="Error: Could not create Value Bet plot", template=TEMPLATE_NAME)


def create_prediction_confidence_gauge(prediction: Dict[str, float], home_team: str, away_team: str) -> go.Figure:
    """
    Create a themed gauge chart showing prediction confidence.
    """
    try:
        if not prediction:
            return go.Figure(go.Indicator()).update_layout(title="No prediction data available", height=250, template=TEMPLATE_NAME)

        highest_prob = max(prediction.values())
        highest_outcome = max(prediction, key=prediction.get)

        display_name = highest_outcome.replace("_", " ").title().replace("Home Win", home_team).replace("Away Win", away_team)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=highest_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"<b>Most Likely Outcome:</b><br>{display_name}", 'font': {'size': 16}},
            number={'suffix': "%", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': GOALDIGGERS_THEME["text_color"]},
                'bar': {'color': GOALDIGGERS_THEME["accent_color"]},
                'bgcolor': GOALDIGGERS_THEME["secondary_color"],
                'borderwidth': 2,
                'bordercolor': GOALDIGGERS_THEME["grid_color"],
                'steps': [
                    {'range': [0, 40], 'color': GOALDIGGERS_THEME["plot_bg_color"]},
                    {'range': [40, 60], 'color': GOALDIGGERS_THEME["grid_color"]},
                ],
            }
        ))

        fig.update_layout(
            height=250,
            margin=dict(l=30, r=30, t=50, b=30),
            template=TEMPLATE_NAME
        )
        return fig

    except Exception as e:
        log_error("Error creating prediction confidence gauge", e)
        return go.Figure().update_layout(title="Error: Could not create Confidence Gauge", template=TEMPLATE_NAME)


def create_feature_importance_plot(feature_importance: Dict[str, float]) -> go.Figure:
    """
    This function is now delegated to the one in plots.py for consistency.
    """
    from dashboard.visualizations.plots import interactive_feature_importance
    logger.info("Delegating feature importance plot to dashboard.visualizations.plots.interactive_feature_importance")
    return interactive_feature_importance(feature_importance)


def create_odds_movement_chart(
    current_odds: Dict[str, float],
    historical_odds: List[Dict[str, float]],
    timestamps: List[str],
    home_team: str,
    away_team: str
) -> go.Figure:
    """
    Create a themed line chart showing odds movement over time.
    """
    try:
        if not current_odds or not historical_odds or not timestamps:
            return go.Figure().update_layout(title="No odds movement data available", height=300, template=TEMPLATE_NAME)

        if len(historical_odds) != len(timestamps):
            timestamps = timestamps[:len(historical_odds)]

        all_odds = historical_odds + [current_odds]
        all_timestamps = timestamps + ["Current"]

        fig = go.Figure()

        colors = {
            "home_win": GOALDIGGERS_THEME["win_color"],
            "draw": GOALDIGGERS_THEME["draw_color"],
            "away_win": GOALDIGGERS_THEME["loss_color"]
        }

        for outcome in ["home_win", "draw", "away_win"]:
            if outcome not in all_odds[0]: continue

            values = [odds.get(outcome) for odds in all_odds]
            
            for i in range(1, len(values)):
                if values[i] is None:
                    values[i] = values[i-1]

            display_name = outcome.replace("_", " ").title().replace("Home Win", home_team).replace("Away Win", away_team)

            fig.add_trace(go.Scatter(
                x=all_timestamps,
                y=values,
                mode="lines+markers",
                name=display_name,
                line=dict(color=colors.get(outcome, GOALDIGGERS_THEME["text_color"]), width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{display_name}</b><br>Odds: %{{y:.2f}}<br>Time: %{{x}}<extra></extra>'
            ))

        fig.update_layout(
            title="Odds Movement Over Time",
            yaxis_title="Decimal Odds",
            height=350,
            template=TEMPLATE_NAME,
            margin=dict(l=40, r=20, t=60, b=50),
            legend=dict(y=1.15)
        )
        return fig

    except Exception as e:
        log_error("Error creating odds movement chart", e)
        return go.Figure().update_layout(title="Error: Could not create Odds Movement chart", template=TEMPLATE_NAME)


def plot_match_outcome_probabilities(prediction: dict):
    """
    Plots the probabilities for match outcomes (Home Win, Draw, Away Win).

    Args:
        prediction (dict): A dictionary containing prediction probabilities.

    Returns:
        A Plotly figure object.
    """
    probas = prediction.get('probabilities', {})
    home_win = probas.get('home_win', 0)
    draw = probas.get('draw', 0)
    away_win = probas.get('away_win', 0)

    outcomes = ['Home Win', 'Draw', 'Away Win']
    probabilities = [home_win, draw, away_win]
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=outcomes,
        y=probabilities,
        text=[f"{p:.1%}" for p in probabilities],
        textposition='auto',
        marker_color=[GOALDIGGERS_THEME['primary_color'], GOALDIGGERS_THEME['secondary_color'], GOALDIGGERS_THEME['tertiary_color']],
        hovertemplate="<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>"
    ))

    fig.update_layout(
        title_text="<b>Match Outcome Probabilities</b>",
        xaxis_title="Result",
        yaxis_title="Probability",
        yaxis=dict(
            range=[0, 1],
            tickformat=".0%"
        ),
        height=350,
        template=TEMPLATE_NAME
    )
    
    return fig


def plot_betting_opportunities_by_league(opportunities: List[Any]) -> go.Figure:
    """
    Create a bar chart showing the distribution of betting opportunities by league.
    
    Args:
        opportunities: List of BettingOpportunity objects
        
    Returns:
        Plotly figure object with the league distribution
    """
    try:
        if not opportunities:
            fig = go.Figure()
            fig.update_layout(
                title="No betting opportunities available",
                template=TEMPLATE_NAME,
                height=350
            )
            return fig
        
        # Count opportunities by league
        league_counts = {}
        league_value_avg = {}
        
        for op in opportunities:
            league = op.competition
            if league not in league_counts:
                league_counts[league] = 0
                league_value_avg[league] = []
            
            league_counts[league] += 1
            league_value_avg[league].append(op.value_rating)
        
        # Calculate average value rating by league
        for league in league_value_avg:
            league_value_avg[league] = sum(league_value_avg[league]) / len(league_value_avg[league])
        
        # Sort leagues by count
        sorted_leagues = sorted(league_counts.keys(), key=lambda x: league_counts[x], reverse=True)
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for counts
        fig.add_trace(
            go.Bar(
                x=sorted_leagues,
                y=[league_counts[league] for league in sorted_leagues],
                name="Number of Opportunities",
                marker_color=GOALDIGGERS_THEME["primary_color"],
                text=[league_counts[league] for league in sorted_leagues],
                textposition="auto"
            ),
            secondary_y=False
        )
        
        # Add line chart for average value rating
        fig.add_trace(
            go.Scatter(
                x=sorted_leagues,
                y=[league_value_avg[league] for league in sorted_leagues],
                name="Avg Value Rating",
                mode="lines+markers",
                marker=dict(size=8),
                line=dict(color=GOALDIGGERS_THEME["secondary_color"], width=3)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Betting Opportunities by League",
            template=TEMPLATE_NAME,
            height=400,
            barmode="group",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Number of Opportunities", secondary_y=False)
        fig.update_yaxes(title_text="Avg Value Rating (0-10)", secondary_y=True)
        
        return fig
        
    except Exception as e:
        log_error(f"Error plotting betting opportunities by league: {e}")
        # Return a simple error figure
        fig = go.Figure()
        fig.update_layout(
            title="Error generating league distribution chart",
            template=TEMPLATE_NAME,
            height=350
        )
        return fig


def plot_value_rating_distribution(opportunities: List[Any]) -> go.Figure:
    """
    Create a histogram showing the distribution of value ratings.
    
    Args:
        opportunities: List of BettingOpportunity objects
        
    Returns:
        Plotly figure object with the value rating distribution
    """
    try:
        if not opportunities:
            fig = go.Figure()
            fig.update_layout(
                title="No betting opportunities available",
                template=TEMPLATE_NAME,
                height=350
            )
            return fig
        
        # Extract value ratings
        value_ratings = [op.value_rating for op in opportunities]
        
        # Create histogram with custom bins
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=value_ratings,
                nbinsx=10,
                marker_color=GOALDIGGERS_THEME["primary_color"],
                opacity=0.8,
                histnorm="probability",
                name="Value Rating"
            )
        )
        
        # Add a vertical line at value rating 7 (considered premium)
        fig.add_vline(
            x=7, 
            line_dash="dash", 
            line_color=GOALDIGGERS_THEME["accent_color"],
            annotation_text="Premium threshold",
            annotation_position="top right"
        )
        
        # Update layout
        fig.update_layout(
            title="Value Rating Distribution",
            xaxis_title="Value Rating (0-10)",
            yaxis_title="Probability",
            template=TEMPLATE_NAME,
            height=400,
            bargap=0.1,
            showlegend=False
        )
        
        # Add annotations
        average_rating = sum(value_ratings) / len(value_ratings)
        fig.add_annotation(
            x=average_rating,
            y=0.05,
            text=f"Avg: {average_rating:.2f}",
            showarrow=True,
            arrowhead=4,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=GOALDIGGERS_THEME["text_color"],
            font=dict(size=10, color=GOALDIGGERS_THEME["text_color"])
        )
        
        return fig
        
    except Exception as e:
        log_error(f"Error plotting value rating distribution: {e}")
        # Return a simple error figure
        fig = go.Figure()
        fig.update_layout(
            title="Error generating value rating distribution",
            template=TEMPLATE_NAME,
            height=350
        )
        return fig
