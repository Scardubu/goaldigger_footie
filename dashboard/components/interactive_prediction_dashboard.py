#!/usr/bin/env python3
"""
Interactive Match Prediction Dashboard for GoalDiggers Platform
Provides an engaging, interactive experience for viewing and analyzing match predictions.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from models.enhanced_real_data_predictor import (
    MatchInsight,
    get_enhanced_match_prediction,
    get_enhanced_match_prediction_insight,
)

logger = logging.getLogger(__name__)

class InteractivePredictionDashboard:
    """Interactive dashboard for match predictions."""
    
    def __init__(self):
        """Initialize the interactive dashboard."""
        self.dashboard_name = "Interactive Prediction Dashboard"
        
    def create_probability_gauge(self, home_prob: float, draw_prob: float, away_prob: float, 
                               home_team: str, away_team: str) -> go.Figure:
        """Create an interactive probability gauge visualization."""
        
        # Create subplot with gauges
        fig = go.Figure()
        
        # Home team probability
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=home_prob * 100,
            domain={'row': 0, 'column': 0, 'x': [0, 0.33], 'y': [0, 1]},
            title={'text': f"🏠 {home_team}"},
            delta={'reference': 33.33, 'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 25], 'color': "#ffcccc"},
                    {'range': [25, 50], 'color': "#ffe6cc"}, 
                    {'range': [50, 75], 'color': "#ccffcc"},
                    {'range': [75, 100], 'color': "#ccffff"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        # Draw probability  
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=draw_prob * 100,
            domain={'row': 0, 'column': 1, 'x': [0.33, 0.66], 'y': [0, 1]},
            title={'text': "🤝 Draw"},
            delta={'reference': 33.33, 'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff7f0e"},
                'steps': [
                    {'range': [0, 25], 'color': "#ffcccc"},
                    {'range': [25, 50], 'color': "#ffe6cc"},
                    {'range': [50, 75], 'color': "#ccffcc"},
                    {'range': [75, 100], 'color': "#ccffff"}
                ],
            }
        ))
        
        # Away team probability
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta", 
            value=away_prob * 100,
            domain={'row': 0, 'column': 2, 'x': [0.66, 1], 'y': [0, 1]},
            title={'text': f"✈️ {away_team}"},
            delta={'reference': 33.33, 'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2ca02c"},
                'steps': [
                    {'range': [0, 25], 'color': "#ffcccc"},
                    {'range': [25, 50], 'color': "#ffe6cc"},
                    {'range': [50, 75], 'color': "#ccffcc"}, 
                    {'range': [75, 100], 'color': "#ccffff"}
                ],
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            title={
                'text': "Match Outcome Probabilities",
                'x': 0.5,
                'font': {'size': 18}
            }
        )
        
        return fig
    
    def create_confidence_radar(self, insight: MatchInsight) -> go.Figure:
        """Create a radar chart showing prediction confidence factors."""
        
        factors = [
            "Form Momentum",
            "Head-to-Head",
            "Venue Advantage", 
            "Squad Quality",
            "Injury Impact",
            "Motivation"
        ]
        
        values = [
            abs(insight.momentum_score) * 100,
            80 if 'strong' in insight.head_to_head_edge else (60 if 'slight' in insight.head_to_head_edge else 40),
            abs(insight.venue_advantage) * 400,  # Scale for visibility
            70,  # Placeholder - would use actual squad differential
            (1 - abs(insight.injury_impact)) * 100,
            abs(insight.motivation_factor) * 200  # Scale for visibility
        ]
        
        # Ensure values are in reasonable range
        values = [min(100, max(0, v)) for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=factors,
            fill='toself',
            name='Confidence Factors',
            line=dict(color='#1f77b4', width=2),
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title={
                'text': f"Prediction Confidence: {insight.confidence*100:.0f}%",
                'x': 0.5,
                'font': {'size': 16}
            },
            height=400
        )
        
        return fig
    
    def create_form_comparison(self, home_form: List[str], away_form: List[str], 
                             home_team: str, away_team: str) -> go.Figure:
        """Create form comparison visualization."""
        
        def form_to_points(form_list):
            points = []
            for result in form_list[-10:]:  # Last 10 matches
                if result.upper() == 'W':
                    points.append(3)
                elif result.upper() == 'D':
                    points.append(1)
                else:
                    points.append(0)
            return points
        
        home_points = form_to_points(home_form)
        away_points = form_to_points(away_form)
        
        # Create cumulative points
        home_cumulative = []
        away_cumulative = []
        
        home_total = 0
        away_total = 0
        
        for i in range(max(len(home_points), len(away_points))):
            if i < len(home_points):
                home_total += home_points[i]
            if i < len(away_points):
                away_total += away_points[i] 
                
            home_cumulative.append(home_total)
            away_cumulative.append(away_total)
        
        fig = go.Figure()
        
        matches = list(range(1, len(home_cumulative) + 1))
        
        fig.add_trace(go.Scatter(
            x=matches,
            y=home_cumulative,
            mode='lines+markers',
            name=f'{home_team} (Home)',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=matches,
            y=away_cumulative,
            mode='lines+markers',
            name=f'{away_team} (Away)',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Recent Form Comparison (Cumulative Points)",
            xaxis_title="Matches Ago",
            yaxis_title="Cumulative Points",
            hovermode='x unified',
            height=300
        )
        
        return fig
    
    def show_interactive_prediction(self, home_team: str, away_team: str, 
                                  match_data: Optional[Dict[str, Any]] = None) -> None:
        """Show interactive prediction dashboard for a match (UnifiedDesignSystem compliance)."""
        # Inject UnifiedDesignSystem CSS
        try:
            from dashboard.components.unified_design_system import (
                get_unified_design_system,
            )
            design_system = get_unified_design_system()
            design_system.inject_unified_css('premium')
        except Exception as css_e:
            logger.warning(f"UnifiedDesignSystem CSS injection failed: {css_e}")

        # Generate sample data if none provided
        if match_data is None:
            match_data = self._generate_sample_match_data(home_team, away_team)

        # Get enhanced prediction
        # Prefer structured insight object; fall back to raw dict if necessary
        try:
            insight = get_enhanced_match_prediction_insight(home_team, away_team, match_data)
        except Exception:
            raw = get_enhanced_match_prediction(home_team, away_team, match_data)
            # Minimal adapter (keeps existing visualization functional)
            class _RawAdapter:
                home_win_prob = raw.get('home_win_probability', 0.34)
                draw_prob = raw.get('draw_probability', 0.32)
                away_win_prob = raw.get('away_win_probability', 0.34)
                confidence = raw.get('confidence_score', 0.7)
                momentum_score = 0.0
                injury_impact = 0.0
                motivation_factor = 0.0
                expected_value = max(home_win_prob, draw_prob, away_win_prob) - (1/3)
                risk_level = 'Medium'
                head_to_head_edge = 'balanced'
                venue_advantage = home_win_prob - max(draw_prob, away_win_prob) / 2
                key_factors = raw.get('key_factors', [])
            insight = _RawAdapter()

        # Unified header
        try:
            design_system.create_unified_header(
                title=f"⚽ {home_team} vs {away_team}",
                subtitle="AI-Powered Match Analysis"
            )
        except Exception:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; color: white; margin: 1rem 0;'>
                <h1 style='margin: 0; font-size: 2.5rem;'>⚽ {home_team} vs {away_team}</h1>
                <p style='font-size: 1.1rem; margin: 0.5rem 0;'>AI-Powered Match Analysis</p>
            </div>
            """, unsafe_allow_html=True)

        # Main probability visualization
        prob_fig = self.create_probability_gauge(
            insight.home_win_prob, insight.draw_prob, insight.away_win_prob,
            home_team, away_team
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "🎯 Confidence", 
                f"{insight.confidence*100:.0f}%",
                delta=f"{(insight.confidence-0.7)*100:+.0f}%" if insight.confidence != 0.7 else None
            )

        with col2:
            st.metric(
                "💰 Expected Value", 
                f"{insight.expected_value*100:+.1f}%",
                delta="Positive" if insight.expected_value > 0 else "Negative"
            )

        with col3:
            risk_color = "normal" if insight.risk_level == "Medium" else ("inverse" if insight.risk_level == "High" else "off")
            st.metric("⚠️ Risk Level", insight.risk_level)

        with col4:
            # Most likely outcome
            probs = [insight.home_win_prob, insight.draw_prob, insight.away_win_prob]
            outcomes = ["Home Win", "Draw", "Away Win"]
            most_likely = outcomes[probs.index(max(probs))]
            st.metric("📊 Most Likely", most_likely)

        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "📈 Form", "🔍 Factors", "💡 Insights"])

        with tab1:
            # Confidence radar chart
            confidence_fig = self.create_confidence_radar(insight)
            st.plotly_chart(confidence_fig, use_container_width=True)

            # Value bet indicator
            if insight.expected_value > 0.1:
                st.success(f"💰 **Value Bet Opportunity!** Expected value: {insight.expected_value*100:+.1f}%")
            elif insight.expected_value > 0:
                st.info(f"💡 **Slight Edge Detected** Expected value: {insight.expected_value*100:+.1f}%")
            else:
                st.warning(f"⚠️ **No Clear Edge** Expected value: {insight.expected_value*100:+.1f}%")

        with tab2:
            # Form comparison
            home_form = match_data.get('home_form', ['W', 'W', 'D', 'L', 'W'])
            away_form = match_data.get('away_form', ['L', 'W', 'W', 'D', 'L'])

            form_fig = self.create_form_comparison(home_form, away_form, home_team, away_team)
            st.plotly_chart(form_fig, use_container_width=True)

            # Form summary
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{home_team} Recent Form:**")
                form_display = ""
                for result in home_form[-5:]:
                    if result.upper() == 'W':
                        form_display += "🟢 "
                    elif result.upper() == 'D':
                        form_display += "🟡 "
                    else:
                        form_display += "🔴 "
                st.markdown(form_display)

            with col2:
                st.markdown(f"**{away_team} Recent Form:**")
                form_display = ""
                for result in away_form[-5:]:
                    if result.upper() == 'W':
                        form_display += "🟢 "
                    elif result.upper() == 'D':
                        form_display += "🟡 "
                    else:
                        form_display += "🔴 "
                st.markdown(form_display)

        with tab3:
            # Key factors
            st.markdown("### 🔍 Key Factors")
            for i, factor in enumerate(insight.key_factors, 1):
                st.markdown(f"{i}. {factor}")

            # Detailed factor breakdown
            factor_data = {
                'Factor': ['Momentum', 'Venue', 'H2H Edge', 'Injuries', 'Motivation'],
                'Home Team': [
                    insight.momentum_score if insight.momentum_score > 0 else 0,
                    insight.venue_advantage,
                    0.1 if 'home' in insight.head_to_head_edge else 0,
                    -abs(insight.injury_impact) if insight.injury_impact < 0 else 0,
                    insight.motivation_factor if insight.motivation_factor > 0 else 0
                ],
                'Away Team': [
                    -insight.momentum_score if insight.momentum_score < 0 else 0,
                    0,
                    0.1 if 'away' in insight.head_to_head_edge else 0,
                    abs(insight.injury_impact) if insight.injury_impact > 0 else 0,
                    -insight.motivation_factor if insight.motivation_factor < 0 else 0
                ]
            }

            df_factors = pd.DataFrame(factor_data)

            # Create factor comparison chart
            fig_factors = go.Figure()

            fig_factors.add_trace(go.Bar(
                name=home_team,
                x=df_factors['Factor'],
                y=df_factors['Home Team'],
                marker_color='#1f77b4'
            ))

            fig_factors.add_trace(go.Bar(
                name=away_team,
                x=df_factors['Factor'],
                y=df_factors['Away Team'],
                marker_color='#ff7f0e'
            ))

            fig_factors.update_layout(
                title="Factor Advantage Comparison",
                barmode='group',
                yaxis_title="Advantage Score",
                height=300
            )

            st.plotly_chart(fig_factors, use_container_width=True)

        with tab4:
            # AI insights and recommendations
            st.markdown("### 💡 AI Insights")

            # Generate insights based on the prediction
            insights = self._generate_ai_insights(insight, home_team, away_team)

            for insight_text in insights:
                st.info(insight_text)

            # Betting recommendation
            st.markdown("### 💰 Betting Recommendation")

            if insight.expected_value > 0.15:
                st.success(f"🎯 **Strong Recommendation**: This match shows excellent value with {insight.confidence*100:.0f}% confidence.")
            elif insight.expected_value > 0.05:
                st.info(f"💡 **Moderate Recommendation**: Decent value opportunity with {insight.confidence*100:.0f}% confidence.")
            else:
                st.warning(f"⚠️ **Proceed with Caution**: Limited value detected. Consider smaller stakes.")
    
    def _generate_sample_match_data(self, home_team: str, away_team: str) -> Dict[str, Any]:
        """Generate realistic sample match data for demonstration."""
        return {
            'home_form': random.choices(['W', 'D', 'L'], weights=[0.4, 0.3, 0.3], k=8),
            'away_form': random.choices(['W', 'D', 'L'], weights=[0.35, 0.3, 0.35], k=8),
            'head_to_head': [
                {'result': random.choice(['home_win', 'draw', 'away_win'])} 
                for _ in range(6)
            ],
            'venue_stats': {
                'home_win_rate': random.uniform(0.4, 0.7),
                'avg_attendance': random.randint(25000, 60000),
                'years_at_venue': random.randint(3, 20)
            },
            'home_squad': {'market_value': random.randint(200, 800)},
            'away_squad': {'market_value': random.randint(200, 800)},
            'home_injuries': [
                {'importance': random.uniform(0.3, 1.0), 'position': random.choice(['striker', 'midfielder', 'defender']), 'weeks_out': random.randint(1, 8)}
                for _ in range(random.randint(0, 3))
            ],
            'away_injuries': [
                {'importance': random.uniform(0.3, 1.0), 'position': random.choice(['striker', 'midfielder', 'defender']), 'weeks_out': random.randint(1, 8)}
                for _ in range(random.randint(0, 3))
            ],
            'match_context': {
                'home_league_position': random.randint(1, 20),
                'away_league_position': random.randint(1, 20),
                'is_derby': random.choice([True, False]),
                'competition_importance': random.choice(['normal', 'high'])
            }
        }
    
    def _generate_ai_insights(self, insight: MatchInsight, home_team: str, away_team: str) -> List[str]:
        """Generate AI insights based on the prediction."""
        insights = []
        
        # Confidence-based insights
        if insight.confidence > 0.8:
            insights.append(f"🎯 High confidence prediction ({insight.confidence*100:.0f}%) suggests this is a reliable betting opportunity.")
        elif insight.confidence < 0.6:
            insights.append(f"⚠️ Lower confidence ({insight.confidence*100:.0f}%) indicates unpredictable match conditions.")
        
        # Probability insights
        max_prob = max(insight.home_win_prob, insight.draw_prob, insight.away_win_prob)
        if max_prob > 0.6:
            if insight.home_win_prob == max_prob:
                insights.append(f"🏠 Strong home advantage detected for {home_team} ({max_prob*100:.0f}% win probability).")
            elif insight.away_win_prob == max_prob:
                insights.append(f"✈️ {away_team} shows strong away form ({max_prob*100:.0f}% win probability).")
        
        # Form insights
        if abs(insight.momentum_score) > 0.3:
            better_team = home_team if insight.momentum_score > 0 else away_team
            insights.append(f"📈 {better_team} enters with significantly better recent form.")
        
        # Value insights
        if insight.expected_value > 0.1:
            insights.append(f"💰 Strong value bet opportunity with {insight.expected_value*100:.1f}% expected return.")
        
        # Risk insights
        if insight.risk_level == "Low":
            insights.append("🟢 Low risk match with predictable factors favoring the analysis.")
        elif insight.risk_level == "High":
            insights.append("🔴 High risk match due to unpredictable factors - consider reduced stakes.")
        
        return insights[:4]  # Return top 4 insights

# Global instance
interactive_dashboard = InteractivePredictionDashboard()

def show_interactive_match_prediction(home_team: str, away_team: str, match_data: Optional[Dict[str, Any]] = None):
    """Show interactive match prediction dashboard."""
    interactive_dashboard.show_interactive_prediction(home_team, away_team, match_data)
