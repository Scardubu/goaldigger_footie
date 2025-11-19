import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from dashboard.error_log import log_error

logger = logging.getLogger(__name__)

# --- GoalDiggers Theme & Plotting Utilities ---

GOALDIGGERS_THEME = {
    "primary_color": "#0D1B2A",
    "secondary_color": "#1B263B",
    "accent_color": "#FFC107",  # Amber/Gold
    "text_color": "#E0E1DD",     # Light Gray
    "bg_color": "rgba(13, 27, 42, 0)", # Transparent BG for streamlit
    "plot_bg_color": "rgba(27, 38, 59, 0.6)",
    "grid_color": "#415A77",     # Muted Blue for grids
    "win_color": "#4CAF50",       # Green
    "draw_color": "#9E9E9E",      # Gray
    "loss_color": "#F44336",      # Red
    "font_family": "Roboto, Arial, sans-serif"
}

# Create a custom Plotly template
pio.templates["goaldiggers_dark"] = go.layout.Template(
    layout=dict(
        font=dict(
            family=GOALDIGGERS_THEME["font_family"],
            size=12,
            color=GOALDIGGERS_THEME["text_color"]
        ),
        title_font=dict(
            size=18,
            family=GOALDIGGERS_THEME["font_family"]
        ),
        paper_bgcolor=GOALDIGGERS_THEME["bg_color"],
        plot_bgcolor=GOALDIGGERS_THEME["plot_bg_color"],
        xaxis=dict(
            gridcolor=GOALDIGGERS_THEME["grid_color"],
            linecolor=GOALDIGGERS_THEME["grid_color"],
            zerolinecolor=GOALDIGGERS_THEME["grid_color"],
            showgrid=True,
            gridwidth=1,
        ),
        yaxis=dict(
            gridcolor=GOALDIGGERS_THEME["grid_color"],
            linecolor=GOALDIGGERS_THEME["grid_color"],
            zerolinecolor=GOALDIGGERS_THEME["grid_color"],
            showgrid=True,
            gridwidth=1,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        colorway=["#778DA9", "#E0E1DD", "#FFC107", "#415A77", "#1B263B"]
    )
)

TEMPLATE_NAME = "goaldiggers_dark"


def calculate_implied_probabilities(odds: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    Calculates implied probabilities from bookmaker odds, accounting for overround.

    Args:
        odds (Dict[str, float]): Dictionary with 'home_win', 'draw', 'away_win' odds.

    Returns:
        Optional[Dict[str, float]]: Dictionary with implied probabilities, or None if input is invalid.
    """
    required_keys = ["home_win", "draw", "away_win"]
    if not all(k in odds and isinstance(odds[k], (int, float)) and not np.isnan(odds[k]) and odds[k] > 0 for k in required_keys):
        logger.debug("Invalid odds data for calculating implied probabilities.")
        return None

    try:
        inv_odds_sum = (1 / odds["home_win"]) + (1 / odds["draw"]) + (1 / odds["away_win"])
        if inv_odds_sum <= 0: # Avoid division by zero or negative odds issues
             logger.warning(f"Sum of inverse odds is non-positive ({inv_odds_sum}), cannot calculate implied probabilities.")
             return None

        # Normalize to remove overround
        implied_home = (1 / odds["home_win"]) / inv_odds_sum
        implied_draw = (1 / odds["draw"]) / inv_odds_sum
        implied_away = (1 / odds["away_win"]) / inv_odds_sum

        return {"home_win": implied_home, "draw": implied_draw, "away_win": implied_away}
    except ZeroDivisionError:
        logger.error("Division by zero encountered while calculating implied probabilities (odds were likely 0).")
        return None
    except Exception as e:
        log_error("Error calculating implied probabilities", e) # Use log_error
        return None


# --- Enhancements for Production Readiness and Dashboard Integration ---

def plot_probabilities_vs_odds(
    model_pred: Optional[Dict[str, float]],
    bookie_odds: Optional[Dict[str, float]],
    title: str = "Model Prediction vs. Implied Odds"
) -> Optional[go.Figure]:
    """
    Generates a themed, grouped bar chart comparing model predictions and implied odds.
    Enhanced for dashboard integration: robust to missing data, clear tooltips, and color coding for value bets.
    """
    # --- Input Validation ---
    required_keys = ["home_win", "draw", "away_win"]
    if not model_pred or not isinstance(model_pred, dict) or not all(k in model_pred for k in required_keys):
        logger.warning(f"Invalid or incomplete model_pred data for plotting: {model_pred}")
        return None # Return None if data is invalid

    implied_probs = calculate_implied_probabilities(bookie_odds) if bookie_odds else None

    try: # Wrap core plotting logic
        outcomes = ["Home Win", "Draw", "Away Win"]
        model_values = [model_pred["home_win"], model_pred["draw"], model_pred["away_win"]]

        fig = go.Figure()

        # Add Model Probabilities Bar
        fig.add_trace(go.Bar(
            x=outcomes,
            y=model_values,
            name='GoalDiggers AI Probability',
            marker_color=GOALDIGGERS_THEME["accent_color"],
            hovertemplate='Model Probability: %{y:.1%}<extra></extra>'
        ))

        # Add Implied Odds Bar if available
        if implied_probs:
            implied_values = [implied_probs.get(k, 0) for k in required_keys]
            fig.add_trace(go.Bar(
                x=outcomes,
                y=implied_values,
                name='Implied Odds Probability',
                marker_color=GOALDIGGERS_THEME["text_color"],
                opacity=0.6,
                hovertemplate='Implied Odds: %{y:.1%}<extra></extra>'
            ))

            # Add difference text (optional)
            diffs = [m - i for m, i in zip(model_values, implied_values)]
            fig.add_trace(go.Scatter(
                x=outcomes,
                y=[max(m, i) + 0.02 for m, i in zip(model_values, implied_values)], # Position above bars
                mode='text',
                text=[f'{d:+.1%}' for d in diffs], # Format as signed percentage
                textposition='top center',
                name='Difference',
                showlegend=False,
                textfont=dict(
                    size=10,
                    color='rgb(150, 150, 150)'
                )
            ))

        # Highlight value bet bars if model probability > implied
        if implied_probs:
            highlight = [m > i for m, i in zip(model_values, implied_values)]
            for idx, is_value in enumerate(highlight):
                if is_value:
                    fig.add_shape(
                        type="rect",
                        x0=idx-0.4, x1=idx+0.4, y0=0, y1=max(model_values[idx], implied_values[idx]),
                        fillcolor="rgba(44, 160, 44, 0.15)", line_width=0, layer="below"
                    )

        fig.update_layout(
            title=title,
            yaxis=dict(
                title='Probability',
                range=[0, max(max(model_values) if model_values else [0], max(implied_values) if implied_probs else [0]) * 1.1 + 0.05]
            ),
            barmode='group',
            bargap=0.2,
            template=TEMPLATE_NAME,
            margin=dict(l=40, r=40, t=60, b=40),
            )
        return fig
    except Exception as e:
        log_error(f"Error generating plot_probabilities_vs_odds for title '{title}'", e)
        return None # Return None on plotting error

# --- Added Enhanced Visualizations ---

def interactive_feature_importance(feature_weights: dict) -> go.Figure:
    """
    Creates an interactive, themed horizontal bar plot for feature importance.
    """
    if not feature_weights or not isinstance(feature_weights, dict):
        logger.warning("Invalid feature_weights for plot.")
        return go.Figure(layout=go.Layout(title="Feature Importance Data Missing", template=TEMPLATE_NAME))

    valid_features = {k: v for k, v in feature_weights.items() if isinstance(v, (int, float)) and not np.isnan(v)}
    if not valid_features:
        return go.Figure(layout=go.Layout(title="No Valid Feature Importance Data", template=TEMPLATE_NAME))

    try:
        sorted_features = sorted(valid_features.items(), key=lambda item: item[1])
        top_features = sorted_features[-15:]
        feature_names = [item[0].replace("_", " ").title() for item in top_features]
        importance_values = [item[1] for item in top_features]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=feature_names,
            x=importance_values,
            orientation='h',
            marker=dict(
                color=importance_values,
                colorscale='Viridis',
                colorbar=dict(title="Importance", tickfont=dict(color=GOALDIGGERS_THEME["text_color"]))
            ),
            hovertemplate='Feature: %{y}<br>Importance: %{x:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title='Key Factors Influencing AI Prediction',
            xaxis_title='Feature Importance Score',
            yaxis_title='Feature',
            yaxis=dict(autorange="reversed"),
            template=TEMPLATE_NAME,
            height=400,
            margin=dict(l=150, r=40, t=60, b=40),
        )
        return fig
    except Exception as e:
        log_error("Error generating interactive feature importance plot", e)
        return go.Figure(layout=go.Layout(title="Error Generating Feature Importance Plot", template=TEMPLATE_NAME))


def create_team_performance_plot(
    team_data: Dict[str, Any],
    time_range: str = "last_10"
) -> go.Figure:
    """Create an interactive, themed team performance visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Goals Scored/Conceded", "Expected Goals (xG)", "Form Trend", "Performance Metrics"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    try:
        match_data = team_data.get("matches", {}).get(time_range, {})
        dates = match_data.get("dates", [])

        # Goals Scored/Conceded
        fig.add_trace(go.Bar(
            x=dates, y=match_data.get("goals_scored", []), name="Scored",
            marker_color=GOALDIGGERS_THEME["win_color"]), row=1, col=1)
        fig.add_trace(go.Bar(
            x=dates, y=match_data.get("goals_conceded", []), name="Conceded",
            marker_color=GOALDIGGERS_THEME["loss_color"]), row=1, col=1)

        # Expected Goals
        fig.add_trace(go.Scatter(
            x=dates, y=match_data.get("xg_for", []), name="xG For",
            line=dict(color=GOALDIGGERS_THEME["accent_color"])), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=dates, y=match_data.get("xg_against", []), name="xG Against",
            line=dict(color=GOALDIGGERS_THEME["text_color"], dash='dash')), row=1, col=2)

        # Form Trend
        fig.add_trace(go.Scatter(
            x=dates, y=match_data.get("form_points", []), name="Form Points",
            line=dict(color=GOALDIGGERS_THEME["accent_color"], width=3)), row=2, col=1)

        # Performance Metrics
        metrics = team_data.get("stats", {})
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        fig.add_trace(go.Bar(
            x=metric_names, y=metric_values, name="Metrics",
            marker_color=GOALDIGGERS_THEME["accent_color"]), row=2, col=2)

        fig.update_layout(
            title_text=f"Performance Analysis: {team_data.get('team_name', 'N/A')}",
            template=TEMPLATE_NAME,
            height=600,
            showlegend=False,
            barmode='overlay'
        )
        return fig
    except Exception as e:
        log_error(f"Error creating team performance plot for {team_data.get('team_name', 'N/A')}", e)
        return go.Figure(layout=go.Layout(title="Error Generating Team Performance Plot", template=TEMPLATE_NAME))


def create_league_standings_plot(
    standings_data: pd.DataFrame,
    highlight_teams: Optional[List[str]] = None
) -> go.Figure:
    """Create an interactive, themed league standings table."""
    if standings_data.empty:
        return go.Figure(layout=go.Layout(title="League Standings Data Not Available", template=TEMPLATE_NAME))

    try:
        highlight_teams = highlight_teams or []
        colors = [
            GOALDIGGERS_THEME["plot_bg_color"],
            GOALDIGGERS_THEME["secondary_color"]
        ] * (len(standings_data) // 2 + 1)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=[col.replace("_", " ").title() for col in standings_data.columns],
                fill_color=GOALDIGGERS_THEME["primary_color"],
                align='left',
                font=dict(color=GOALDIGGERS_THEME["text_color"], size=12)
            ),
            cells=dict(
                values=[standings_data[col] for col in standings_data.columns],
                fill_color=[[colors[i] if team not in highlight_teams else GOALDIGGERS_THEME["accent_color"] for i, team in enumerate(standings_data["team"])]] * len(standings_data.columns),
                align='left',
                font=dict(color=GOALDIGGERS_THEME["text_color"], size=11),
                height=30
            )
        )])

        fig.update_layout(
            title_text="League Standings",
            template=TEMPLATE_NAME,
            margin=dict(l=10, r=10, t=50, b=10),
            height=len(standings_data) * 30 + 60
        )
        return fig
    except Exception as e:
        log_error("Error creating league standings plot", e)
        return go.Figure(layout=go.Layout(title="Error Generating League Standings", template=TEMPLATE_NAME))
