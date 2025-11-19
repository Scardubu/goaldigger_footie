"""
Visualization components for XGBoost predictions and SHAP explanations, themed for GoalDiggers.
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.error_log import log_error
from dashboard.visualizations.plots import GOALDIGGERS_THEME, TEMPLATE_NAME

logger = logging.getLogger(__name__)

def create_prediction_gauge(prediction: Dict[str, float], home_team: str, away_team: str) -> go.Figure:
    """
    Create a themed gauge-style pie chart for match outcome prediction.
    """
    fig = go.Figure().update_layout(template=TEMPLATE_NAME, height=300)
    if not prediction:
        return fig.update_layout(title="No Prediction Available")

    try:
        labels = [home_team, "Draw", away_team]
        probs = [prediction.get("home_win", 0), prediction.get("draw", 0), prediction.get("away_win", 0)]
        colors = [GOALDIGGERS_THEME["win_color"], GOALDIGGERS_THEME["draw_color"], GOALDIGGERS_THEME["loss_color"]]

        fig.add_trace(go.Pie(
            labels=labels, values=probs, hole=.4, marker_colors=colors,
            textinfo='label+percent', hoverinfo='label+percent', textfont_size=12
        ))

        max_prob_label = labels[np.argmax(probs)]

        fig.update_layout(
            title_text="Match Outcome Prediction",
            showlegend=False,
            annotations=[dict(text=f'<b>Most<br>Likely:</b><br>{max_prob_label}', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        return fig
    except Exception as e:
        log_error("Error creating prediction gauge", e)
        return fig.update_layout(title="Error Creating Prediction Gauge")

def create_feature_importance_chart(features: Dict[str, float], max_features: int = 10) -> go.Figure:
    """
    This function is deprecated. Use interactive_feature_importance from plots.py instead.
    """
    logger.warning("Deprecated: create_feature_importance_chart is called. Use interactive_feature_importance from plots.py.")
    from dashboard.visualizations.plots import interactive_feature_importance
    return interactive_feature_importance(features)

def create_value_bet_chart(prediction: Dict[str, float], odds: Dict[str, float], home_team: str, away_team: str) -> go.Figure:
    """
    This function is deprecated. Use plot_probabilities_vs_odds from plots.py instead.
    """
    logger.warning("Deprecated: create_value_bet_chart is called. Use plot_probabilities_vs_odds from plots.py.")
    from dashboard.visualizations.plots import plot_probabilities_vs_odds
    return plot_probabilities_vs_odds(prediction, odds, title=f"Value Analysis: {home_team} vs {away_team}")

def create_historical_performance_chart(results: List[Dict[str, Any]], rolling_window: int = 10) -> go.Figure:
    """
    Create a themed line chart showing model performance over time.
    """
    fig = go.Figure().update_layout(template=TEMPLATE_NAME, height=350)
    if not results or len(results) < 2:
        return fig.update_layout(title="Insufficient Historical Data")

    try:
        df = pd.DataFrame(results)
        if 'predicted_outcome' not in df.columns or 'actual_outcome' not in df.columns:
            return fig.update_layout(title="Missing required outcome columns")

        df['correct'] = (df['predicted_outcome'] == df['actual_outcome']).astype(int)
        df['date'] = pd.to_datetime(df.get('date', pd.date_range(end=pd.Timestamp.now(), periods=len(df))))
        df = df.sort_values('date')
        df['rolling_accuracy'] = df['correct'].rolling(window=rolling_window, min_periods=1).mean() * 100

        fig.add_trace(go.Scatter(
            x=df['date'], y=df['rolling_accuracy'], mode='lines+markers',
            name=f'Rolling {rolling_window}-Match Accuracy',
            line=dict(color=GOALDIGGERS_THEME["accent_color"], width=2),
            hovertemplate='%{x|%Y-%m-%d}<br>Accuracy: %{y:.1f}%<extra></extra>'
        ))

        fig.add_hline(
            y=33.3, line_dash="dash", line_color=GOALDIGGERS_THEME["grid_color"]
        )

        fig.add_annotation(
            y=33.3,
            text="Baseline (Random Guess)",
            xref="paper",
            x=0.98,
            yref="y",
            showarrow=False,
            font=dict(size=12, color=GOALDIGGERS_THEME["grid_color"]),
            align="right",
            yshift=10
        )

        fig.update_layout(
            title_text="Model Performance Over Time",
            xaxis_title="Date", yaxis_title="Accuracy (%)", yaxis_range=[0, 100],
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    except Exception as e:
        log_error("Error creating historical performance chart", e)
        return fig.update_layout(title="Error Creating Performance Chart")
