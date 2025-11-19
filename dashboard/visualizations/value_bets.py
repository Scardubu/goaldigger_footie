"""
Value betting visualization component for the GoalDiggers platform.
Visualizes value betting opportunities with interactive, themed charts and metrics.
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.error_log import log_error
from dashboard.visualizations.plots import GOALDIGGERS_THEME, TEMPLATE_NAME

logger = logging.getLogger(__name__)


def _is_empty_input(obj) -> bool:
    """Return True when the provided input should be treated as empty.

    Handles lists, dicts, pandas DataFrame/Series, and None.
    """
    if obj is None:
        return True
    if isinstance(obj, (list, tuple, set)):
        return len(obj) == 0
    if isinstance(obj, dict):
        return len(obj) == 0
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.empty
    return False


def _normalize_value_bets_to_df(value_bets) -> pd.DataFrame:
    """Normalize input (list/Series/DataFrame) into a DataFrame with safe defaults.

    Ensures numeric columns exist and attempts to compute 'edge' when missing using
    predicted_prob/implied_prob or odds.
    """
    if isinstance(value_bets, pd.DataFrame):
        df = value_bets.copy()
    elif isinstance(value_bets, pd.Series):
        if value_bets.empty:
            return pd.DataFrame()
        df = pd.DataFrame([value_bets.to_dict()])
    elif isinstance(value_bets, dict):
        df = pd.DataFrame([value_bets])
    else:
        # assume iterable of dict-like
        try:
            df = pd.DataFrame(list(value_bets))
        except Exception:
            return pd.DataFrame()

    # Ensure numeric columns exist
    for col in ["edge", "predicted_prob", "implied_prob", "kelly_stake", "odds"]:
        if col not in df.columns:
            df[col] = np.nan
    # Try to compute implied_prob if missing and odds available
    if df["implied_prob"].isna().all() and not df["odds"].isna().all():
        with np.errstate(divide='ignore', invalid='ignore'):
            df["implied_prob"] = pd.to_numeric(df["odds"], errors='coerce').apply(lambda o: 1.0 / o if o and o > 0 else np.nan)

    # Try to compute edge if missing
    if df["edge"].isna().all():
        if not df["predicted_prob"].isna().all() and not df["implied_prob"].isna().all():
            df["edge"] = pd.to_numeric(df["predicted_prob"], errors='coerce') - pd.to_numeric(df["implied_prob"], errors='coerce')
        elif not df["predicted_prob"].isna().all() and df["implied_prob"].isna().all() and not df["odds"].isna().all():
            with np.errstate(divide='ignore', invalid='ignore'):
                implied = pd.to_numeric(df["odds"], errors='coerce').apply(lambda o: 1.0 / o if o and o > 0 else np.nan)
                df["edge"] = pd.to_numeric(df["predicted_prob"], errors='coerce') - implied

    # Fill NaNs for plotting convenience
    df["edge"] = pd.to_numeric(df["edge"], errors='coerce').fillna(0.0)
    df["kelly_stake"] = pd.to_numeric(df["kelly_stake"], errors='coerce').fillna(0.0)
    df["odds"] = pd.to_numeric(df["odds"], errors='coerce').fillna(0.0)

    # Confidence as string fallback
    if "confidence" not in df.columns:
        df["confidence"] = "Low"

    return df

def plot_value_bet_edge(value_bets: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a themed horizontal bar chart showing the edge for each value bet.
    """
    fig = go.Figure().update_layout(template=TEMPLATE_NAME, height=300)
    # Normalize input and handle empty input safely
    if _is_empty_input(value_bets):
        return fig.update_layout(title="No Value Betting Opportunities Found")

    df = _normalize_value_bets_to_df(value_bets)
    if df.empty:
        return fig.update_layout(title="No Value Betting Opportunities Found")

    try:
        # Ensure edge percentage computed
        if "edge_pct" not in df.columns:
            df["edge_pct"] = df["edge"] * 100

        # Safe bet label creation
        def _label(row):
            odds = row.get('odds', 0.0)
            try:
                return f"{row.get('bet_type', 'Selection')} ({float(odds):.2f})"
            except Exception:
                return f"{row.get('bet_type', 'Selection')}"

        df["bet_label"] = df.apply(_label, axis=1)
        df = df.sort_values("edge_pct", ascending=True)

        color_map = {"High": GOALDIGGERS_THEME["win_color"], "Medium": GOALDIGGERS_THEME["accent_color"], "Low": GOALDIGGERS_THEME["draw_color"]}
        colors = [color_map.get(str(conf), GOALDIGGERS_THEME["draw_color"]) for conf in df["confidence"]]

        fig.add_trace(go.Bar(
            y=df["bet_label"], x=df["edge_pct"], orientation="h", marker_color=colors,
            text=df["edge_pct"].apply(lambda x: f"{x:+.1f}%"), textposition="outside",
            hovertemplate="<b>%{y}</b><br>Edge: %{x:.1f}%<br>Kelly Stake: %{customdata[0]:.1f}%<extra></extra>",
            customdata=np.stack((df["kelly_stake"].fillna(0) * 100,), axis=-1)
        ))

        fig.update_layout(
            title="Value Betting Edge", xaxis_title="Edge (%)", yaxis_title="",
            margin=dict(l=10, r=20, t=40, b=10)
        )
        fig.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=len(df) - 0.5, line=dict(color=GOALDIGGERS_THEME["grid_color"], width=1, dash="dash"))
        return fig
    except Exception as e:
        log_error("Error creating value bet edge plot", e)
        return fig.update_layout(title="Error Creating Value Edge Plot")

def plot_kelly_stakes(value_bets: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a themed horizontal bar chart showing the Kelly criterion stake.
    """
    fig = go.Figure().update_layout(template=TEMPLATE_NAME, height=300)
    # Normalize input
    if _is_empty_input(value_bets):
        return fig.update_layout(title="No Value Betting Opportunities Found")

    try:
        df = _normalize_value_bets_to_df(value_bets)
        if df.empty:
            return fig.update_layout(title="No Value Betting Opportunities Found")

        df["kelly_pct"] = df["kelly_stake"] * 100
        df["bet_label"] = df.apply(lambda row: f"{row.get('bet_type','Selection')} ({row.get('odds',0.0):.2f})", axis=1)
        df = df.sort_values("kelly_pct", ascending=True)

        color_map = {"High": GOALDIGGERS_THEME["win_color"], "Medium": GOALDIGGERS_THEME["accent_color"], "Low": GOALDIGGERS_THEME["draw_color"]}
        colors = [color_map.get(str(conf), GOALDIGGERS_THEME["draw_color"]) for conf in df["confidence"]]

        fig.add_trace(go.Bar(
            y=df["bet_label"], x=df["kelly_pct"], orientation="h", marker_color=colors,
            text=df["kelly_pct"].apply(lambda x: f"{x:.1f}%"), textposition="outside",
            hovertemplate="<b>%{y}</b><br>Kelly Stake: %{x:.1f}%<br>Edge: %{customdata[0]:.1f}%<extra></extra>",
            customdata=np.stack((df["edge"].fillna(0) * 100,), axis=-1)
        ))

        fig.update_layout(
            title="Recommended Kelly Stakes", xaxis_title="Kelly Stake (% of Bankroll)", yaxis_title="",
            margin=dict(l=10, r=20, t=40, b=10)
        )
        return fig
    except Exception as e:
        log_error("Error creating Kelly stakes plot", e)
        return fig.update_layout(title="Error Creating Kelly Stakes Plot")

def plot_probability_comparison(value_bets: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a themed grouped bar chart comparing predicted vs. implied probabilities.
    """
    fig = go.Figure().update_layout(template=TEMPLATE_NAME, height=300, barmode="group")
    if _is_empty_input(value_bets):
        return fig.update_layout(title="No Value Betting Opportunities Found")

    try:
        df = _normalize_value_bets_to_df(value_bets).sort_values("bet_type")
        df["predicted_pct"] = df["predicted_prob"].fillna(0) * 100
        df["implied_pct"] = df["implied_prob"].fillna(0) * 100

        fig.add_trace(go.Bar(
            x=df["bet_type"], y=df["predicted_pct"], name="GoalDiggers AI",
            marker_color=GOALDIGGERS_THEME["accent_color"], hovertemplate="<b>%{x}</b><br>Predicted: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=df["bet_type"], y=df["implied_pct"], name="Implied by Odds",
            marker_color=GOALDIGGERS_THEME["text_color"], opacity=0.6, hovertemplate="<b>%{x}</b><br>Implied: %{y:.1f}%<extra></extra>"
        ))

        fig.update_layout(
            title="AI Prediction vs. Implied Odds", yaxis_title="Probability (%)",
            margin=dict(l=20, r=20, t=40, b=10), legend=dict(y=1.15)
        )
        return fig
    except Exception as e:
        log_error("Error creating probability comparison plot", e)
        return fig.update_layout(title="Error Creating Probability Plot")

def render_value_bets_dashboard(match_data: Dict[str, Any], value_bets: List[Dict[str, Any]]) -> None:
    """
    Render the value betting dashboard for a match using themed plots.
    """
    if _is_empty_input(match_data) or _is_empty_input(value_bets):
        st.info("No value betting opportunities found for this match.")
        return

    home_team = match_data.get("home_team", "Home")
    away_team = match_data.get("away_team", "Away")

    st.markdown(f"### Value Betting Analysis: {home_team} vs {away_team}")

    edge_fig = plot_value_bet_edge(value_bets)
    st.plotly_chart(edge_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        kelly_fig = plot_kelly_stakes(value_bets)
        st.plotly_chart(kelly_fig, use_container_width=True)
    with col2:
        prob_fig = plot_probability_comparison(value_bets)
        st.plotly_chart(prob_fig, use_container_width=True)

    st.markdown("#### Value Bet Details")
    try:
        df = _normalize_value_bets_to_df(value_bets)
        display_df = df.copy()
        # Safe formatting for percentage columns
        for col in ["edge", "predicted_prob", "implied_prob", "kelly_stake"]:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce').fillna(0.0)
                display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.1f}%")

        display_df = display_df.rename(columns={
            "bet_type": "Bet Type", "odds": "Odds", "predicted_prob": "AI Prob",
            "implied_prob": "Implied Prob", "edge": "Edge", "kelly_stake": "Kelly Stake", "confidence": "Confidence"
        })

        cols = [c for c in ["Bet Type", "Odds", "AI Prob", "Implied Prob", "Edge", "Kelly Stake", "Confidence"] if c in display_df.columns]
        st.dataframe(display_df[cols], use_container_width=True)
    except Exception as e:
        log_error("Error displaying value bet details table", e)
        st.error("Could not display value bet details.")

    st.markdown("#### Actionable Betting Insights")
    try:
        df_sorted = _normalize_value_bets_to_df(value_bets).sort_values("edge", ascending=False)
        if not df_sorted.empty:
            best_bet = df_sorted.iloc[0].to_dict()
            st.success(
                f"**Top Value Bet:** {best_bet.get('bet_type','Selection').replace('_', ' ').title()} @ {best_bet.get('odds',0.0):.2f} with an edge of **{best_bet.get('edge',0.0) * 100:.1f}%**. "
                f"Recommended Kelly stake: **{best_bet.get('kelly_stake',0.0) * 100:.1f}%** of bankroll."
            )
            # If international competition, highlight
            if 'competition' in match_data and match_data['competition'] not in [None, '', 'Domestic']:
                st.info(f"International Competition: {match_data['competition']}")
    except Exception:
        # Already logged earlier if normalization failed; silently continue
        pass
    st.info(
        "Edge = expected value of a bet. Positive edge = profitable opportunity. "
        "Kelly Stake = optimal bet size for long-term growth."
    )


def plot_value_bets_overview(df: pd.DataFrame) -> go.Figure:
    """
    Create a themed scatter plot showing the overview of value betting opportunities.
    """
    fig = go.Figure().update_layout(template=TEMPLATE_NAME, height=500)
    if df.empty:
        return fig.update_layout(title="No Value Betting Opportunities Found")

    try:
        # Ensure required columns exist, providing defaults if necessary
        for col in ['edge', 'kelly_stake', 'confidence', 'odds', 'selection', 'home_team', 'away_team', 'bet_type']:
            if col not in df.columns:
                df[col] = 0 if col in ['edge', 'kelly_stake', 'confidence', 'odds'] else 'N/A'
                
        fig = px.scatter(
            df,
            x="edge",
            y="kelly_stake",
            color="confidence",
            size="odds",
            hover_name="selection",
            custom_data=["home_team", "away_team", "bet_type", "odds"],
            color_continuous_scale=GOALDIGGERS_THEME.get('continuous_color_scale', 'viridis'),  # Fallback to viridis
            template=TEMPLATE_NAME,
            labels={
                "edge": "Edge",
                "kelly_stake": "Kelly Stake",
                "confidence": "Confidence",
                "odds": "Odds"
            }
        )

        fig.update_layout(
            title="Value Betting Opportunities Overview",
            xaxis_title="Edge (%)",
            yaxis_title="Kelly Stake (%)",
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%"),
            coloraxis_colorbar=dict(title="Confidence"),
            legend_title="Confidence"
        )

        fig.update_traces(
            hovertemplate="<b>%{customdata[2]}</b><br>" +
                         "Match: %{customdata[0]} vs %{customdata[1]}<br>" +
                         "Odds: %{customdata[3]:.2f}<br>" +
                         "Edge: %{x:.2%}<br>" +
                         "Kelly Stake: %{y:.2%}<br>" +
                         "Confidence: %{marker.color:.2f}<extra></extra>"
        )
        return fig
    except Exception as e:
        log_error("Error creating value bets overview plot", e)
        return go.Figure().update_layout(title="Error Creating Overview Plot", template=TEMPLATE_NAME)
