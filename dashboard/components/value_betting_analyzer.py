import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.ui_elements import create_metric_card
from dashboard.components.unified_design_system import get_unified_design_system
from dashboard.error_log import log_error
from dashboard.visualizations.value_bets import (
    plot_kelly_stakes,
    plot_probability_comparison,
    plot_value_bet_edge,
    plot_value_bets_overview,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_kelly_criterion(p: float, odds: float, fraction: float = 1.0) -> float:
    """
    Calculate the Kelly Criterion stake for a bet.
    
    Args:
        p: Probability of winning (0-1)
        odds: Decimal odds
        fraction: Fraction of Kelly to use (0-1, default=1.0)
        
    Returns:
        Recommended stake as a fraction of bankroll
    """
    if odds <= 1 or p <= 0:
        return 0
    
    b = odds - 1
    q = 1 - p
    
    try:
        kelly = (b * p - q) / b
        return max(0, kelly) * fraction
    except ZeroDivisionError:
        return 0

def analyze_value_opportunities(
    matches_data: List[Dict[str, Any]], 
    model_predictions: Dict[str, Dict[str, Any]],
    odds_data: Dict[str, Dict[str, Any]],
    min_edge_threshold: float = 0.05,
    max_stake_cap: float = 0.1
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze matches for value betting opportunities.
    """
    value_opportunities = {}
    
    for match in matches_data:
        match_id = match.get('id')
        if not match_id or match_id not in model_predictions or match_id not in odds_data:
            continue
            
        prediction = model_predictions[match_id]
        odds = odds_data[match_id]
        
        value_bets = {}
        for outcome in ['home_win', 'draw', 'away_win']:
            model_prob = prediction.get(outcome, 0)
            bookie_odds = odds.get(outcome, 0)
            
            if bookie_odds <= 1:
                continue
                
            implied_prob = 1 / bookie_odds
            edge = model_prob - implied_prob
            
            kelly = calculate_kelly_criterion(model_prob, bookie_odds)
            
            value_bets[outcome] = {
                'model_prob': model_prob,
                'bookie_odds': bookie_odds,
                'implied_prob': implied_prob,
                'edge': edge,
                'edge_pct': edge * 100,
                'kelly': kelly,
                'capped_kelly': min(kelly, max_stake_cap),
                'is_value': edge > min_edge_threshold
            }
        
        value_opportunities[match_id] = value_bets
    
    return value_opportunities

def get_value_bet_summary(
    value_opportunities: Dict[str, Dict[str, Dict[str, Any]]],
    matches_data: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a summary DataFrame of value betting opportunities.
    """
    summary_data = []
    match_map = {match['id']: match for match in matches_data}

    for match_id, value_bets in value_opportunities.items():
        match_info = match_map.get(match_id)
        if not match_info:
            continue

        home_team = match_info.get('home_team', 'N/A')
        away_team = match_info.get('away_team', 'N/A')

        for outcome, bet_info in value_bets.items():
            if bet_info['is_value']:
                outcome_map = {'home_win': f"{home_team} Win", 'draw': "Draw", 'away_win': f"{away_team} Win"}
                summary_data.append({
                    'match_id': match_id,
                    'date': match_info.get('match_date'),
                    'competition': match_info.get('competition', 'N/A'),
                    'match': f"{home_team} vs {away_team}",
                    'outcome': outcome_map.get(outcome, "N/A"),
                    'model_prob': bet_info['model_prob'],
                    'bookie_odds': bet_info['bookie_odds'],
                    'edge_pct': bet_info['edge_pct'],
                    'kelly_pct': bet_info['capped_kelly'] * 100
                })
    
    if not summary_data:
        return pd.DataFrame()

    return pd.DataFrame(summary_data).sort_values(by='edge_pct', ascending=False).reset_index(drop=True)


def render_value_betting_dashboard(
    value_opportunities: Dict[str, Dict[str, Dict[str, Any]]],
    matches_data: List[Dict[str, Any]]
):
    """
    Render a value betting dashboard using themed components.
    """
    try:
        uds = get_unified_design_system()
        try:
            uds.inject_unified_css('premium')
        except Exception:
            pass
        st.header("Value Betting Analyzer")
        st.markdown("Identifying profitable betting opportunities by comparing model predictions against bookmaker odds.")

        value_bets_df = get_value_bet_summary(value_opportunities, matches_data)

        if value_bets_df.empty:
            st.info("No value betting opportunities found based on the current data and criteria.")
            return

        total_value_bets = len(value_bets_df)
        avg_edge = value_bets_df['edge_pct'].mean()
        avg_kelly = value_bets_df['kelly_pct'].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("Value Bets Found", str(total_value_bets))
        with col2:
            create_metric_card("Average Edge", f"{avg_edge:.2f}%")
        with col3:
            create_metric_card("Average Kelly Stake", f"{avg_kelly:.2f}%")

        st.subheader("Top Value Opportunities")
        st.dataframe(value_bets_df.head(10), use_container_width=True)

        # Visualizations
        st.subheader("Analysis Visualizations")

        fig_overview = plot_value_bets_overview(value_bets_df)
        st.plotly_chart(fig_overview, use_container_width=True)

        selected_bet_index = st.selectbox(
            "Select a bet to analyze further:",
            options=value_bets_df.index,
            format_func=lambda i: f"{value_bets_df.loc[i, 'match']} - {value_bets_df.loc[i, 'outcome']}"
        )

        if selected_bet_index is not None:
            selected_bet = value_bets_df.loc[selected_bet_index]

            col1, col2 = st.columns(2)
            with col1:
                fig_edge = plot_value_bet_edge(selected_bet)
                st.plotly_chart(fig_edge, use_container_width=True)
            with col2:
                fig_kelly = plot_kelly_stakes(selected_bet)
                st.plotly_chart(fig_kelly, use_container_width=True)

            fig_prob_comp = plot_probability_comparison(selected_bet)
            st.plotly_chart(fig_prob_comp, use_container_width=True)

        with st.expander("📘 Value Betting Strategy Guide"):
            st.markdown("""
            #### Value Betting Strategy
            Value betting is a mathematical strategy focused on identifying odds that are more favorable than the true probability of an outcome suggests.
            - **Edge**: The percentage difference between our model's probability and the bookmaker's implied probability. A positive edge indicates a value bet.
            - **Kelly Criterion**: A formula to determine the optimal stake size to maximize long-term bankroll growth. We recommend using a fraction (e.g., 25-50%) of the suggested Kelly stake to reduce risk.
            - **Bankroll Management**: Never risk more than a small fraction (e.g., 1-5%) of your total bankroll on a single bet.
            """)

    except Exception as e:
        log_error(f"Error rendering value betting dashboard: {e}", e)
        st.error("An error occurred while rendering the value betting analysis.")

def render_value_bet_card(
    match_id: str,
    home_team: str,
    away_team: str,
    value_bets: Dict[str, Dict[str, Any]]
):
    """
    Render an individual value bet card for a specific match using themed components.
    """
    try:
        best_value = None
        best_edge = -1

        for outcome, bet_info in value_bets.items():
            if bet_info['is_value'] and bet_info['edge'] > best_edge:
                best_value = {'outcome': outcome, 'info': bet_info}
                best_edge = bet_info['edge']

        if not best_value:
            return

        outcome_map = {'home_win': f"{home_team} Win", 'draw': "Draw", 'away_win': f"{away_team} Win"}
        outcome_name = outcome_map.get(best_value['outcome'], "N/A")
        
        bet_info = best_value['info']
        edge_pct = bet_info['edge_pct']
        kelly_pct = bet_info['capped_kelly'] * 100
        odds = bet_info['bookie_odds']

        card_content = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
            <div>
                <span style=\"font-weight: 600; font-size: 1.1em;\">{outcome_name}</span>
                <span style=\"font-size: 0.9em; color: #666; margin-left: 10px;\">Odds: @{odds:.2f}</span>
            </div>
            <div>
                <span style=\"font-weight: 600; color: #2e7d32;\">Edge: {edge_pct:.2f}%</span>
                <br>
                <span style=\"font-weight: 600; color: #1e88e5;\">Kelly: {kelly_pct:.2f}%</span>
            </div>
        </div>
        """
        uds = get_unified_design_system()
        def _card():
            st.markdown(card_content, unsafe_allow_html=True)
        try:
            uds.create_unified_card(_card, card_class='premium-card')
        except Exception:
            st.markdown(card_content, unsafe_allow_html=True)
    except Exception as e:
        log_error(f"Error rendering value bet card for match {match_id}: {e}", e)


class ValueBettingAnalyzer:
    """
    Advanced value betting analyzer for identifying profitable betting opportunities.

    Analyzes market odds against ML predictions to identify value bets with positive
    expected value and appropriate risk assessment.
    """

    def __init__(self):
        """Initialize value betting analyzer."""
        self.logger = logger
        self._analysis_cache = {}
        self._performance_metrics = {
            'analyses_performed': 0,
            'value_opportunities_found': 0,
            'average_expected_value': 0.0
        }
        self._latest_market_snapshot: Dict[str, Dict[str, Any]] = {}
        self._latest_market_summary: Dict[str, Any] = {}

        # Value betting thresholds
        self._min_value_threshold = 0.02  # 2% minimum expected value
        self._high_value_threshold = 0.10  # 10% high value threshold

        self.logger.info("💰 Value Betting Analyzer initialized")

    def analyze_value_opportunities(self,
                                  home_team: str,
                                  away_team: str,
                                  ml_predictions: Dict[str, float],
                                  market_odds: Dict[str, float]) -> Dict[str, Any]:
        """Analyze value betting opportunities for a match."""
        try:
            analysis_start = time.time()

            # Extract probabilities and odds
            home_prob = ml_predictions.get('home_win', 0.33)
            draw_prob = ml_predictions.get('draw', 0.33)
            away_prob = ml_predictions.get('away_win', 0.33)

            home_odds = market_odds.get('home_odds', 2.0)
            draw_odds = market_odds.get('draw_odds', 3.0)
            away_odds = market_odds.get('away_odds', 3.0)

            # Calculate expected values
            home_ev = self.calculate_expected_value(home_prob, home_odds)
            draw_ev = self.calculate_expected_value(draw_prob, draw_odds)
            away_ev = self.calculate_expected_value(away_prob, away_odds)

            # Calculate market efficiency
            total_implied = (1/home_odds) + (1/draw_odds) + (1/away_odds)
            market_efficiency = 1.0 - ((total_implied - 1.0) / 2)

            analysis_time = time.time() - analysis_start
            self._performance_metrics['analyses_performed'] += 1

            return {
                'match': f"{home_team} vs {away_team}",
                'home_expected_value': home_ev,
                'draw_expected_value': draw_ev,
                'away_expected_value': away_ev,
                'market_efficiency': market_efficiency,
                'analysis_time': analysis_time,
                'best_opportunity': max([
                    ('home', home_ev, home_team),
                    ('draw', draw_ev, 'Draw'),
                    ('away', away_ev, away_team)
                ], key=lambda x: x[1])
            }

        except Exception as e:
            self.logger.error(f"❌ Value betting analysis failed: {e}")
            return {'error': str(e)}

    def calculate_expected_value(self, probability: float, odds: float) -> float:
        """Calculate expected value for a bet."""
        return (probability * (odds - 1)) - (1 - probability)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self._performance_metrics

    def ingest_market_snapshot(
        self,
        match_market: Dict[str, Dict[str, Any]],
        summary: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store latest live market snapshot for downstream analyses."""
        try:
            if not isinstance(match_market, dict):
                return

            self._latest_market_snapshot = {
                str(match_id): dict(data)
                for match_id, data in match_market.items()
                if isinstance(data, dict)
            }

            if isinstance(summary, dict):
                self._latest_market_summary = dict(summary)

            self.logger.debug(f"📈 Market snapshot ingested for {len(self._latest_market_snapshot)} matches")
        except Exception as e:
            self.logger.debug(f"Market snapshot ingestion skipped: {e}")

    def get_latest_market_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return the most recent market snapshot ingested."""
        return dict(self._latest_market_snapshot)

    def get_latest_market_summary(self) -> Dict[str, Any]:
        """Return the most recent market summary ingested."""
        return dict(self._latest_market_summary)
