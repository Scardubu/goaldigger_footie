"""
Enhanced Betting Insights Display for GoalDiggers Dashboard
Provides clear, actionable betting recommendations with professional presentation.
"""

import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

class EnhancedInsightsDisplay:
    """Enhanced display system for betting insights and recommendations."""
    
    def __init__(self):
        """Initialize the enhanced insights display."""
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        self.value_thresholds = {
            "excellent": 7.0,
            "good": 5.0,
            "fair": 3.0
        }
    
    def render_insights_dashboard(self, insights_data: List[Dict[str, Any]]) -> None:
        """Render the complete insights dashboard."""
        if not insights_data:
            self._render_no_insights_message()
            return
        
        # Render dashboard header with key metrics
        self._render_dashboard_header(insights_data)
        
        # Render insights grid
        self._render_insights_grid(insights_data)
        
        # Render summary analytics
        self._render_summary_analytics(insights_data)
    
    def _render_dashboard_header(self, insights_data: List[Dict[str, Any]]) -> None:
        """Render dashboard header with key metrics."""
        # Calculate key metrics
        total_matches = len(insights_data)
        value_bets = [i for i in insights_data if i.get("value_score", 0) >= 3.0]
        high_confidence = [i for i in insights_data if i.get("confidence", 0) >= 0.8]
        avg_value = np.mean([i.get("value_score", 0) for i in insights_data])
        
        # Render header metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_metric_card(
                title="Total Matches",
                value=str(total_matches),
                icon="⚽",
                color="primary"
            )
        
        with col2:
            self._render_metric_card(
                title="Value Bets",
                value=str(len(value_bets)),
                delta=f"{len(value_bets)/total_matches:.0%} of total",
                icon="💎",
                color="success"
            )
        
        with col3:
            self._render_metric_card(
                title="High Confidence",
                value=str(len(high_confidence)),
                delta=f"{len(high_confidence)/total_matches:.0%} of total",
                icon="🎯",
                color="info"
            )
        
        with col4:
            self._render_metric_card(
                title="Avg Value Score",
                value=f"{avg_value:.1f}",
                icon="📊",
                color="warning"
            )
    
    def _render_metric_card(self, title: str, value: str, icon: str, color: str, delta: Optional[str] = None) -> None:
        """Render an individual metric card."""
        delta_html = ""
        if delta:
            delta_html = f'<div class="metric-delta">{delta}</div>'
        
        st.markdown(f"""
        <div class="enhanced-metric-card {color}">
            <div class="metric-header">
                <span class="metric-icon">{icon}</span>
                <span class="metric-title">{title}</span>
            </div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def _render_insights_grid(self, insights_data: List[Dict[str, Any]]) -> None:
        """Render the main insights grid."""
        st.markdown("### 🎯 Betting Opportunities")
        
        # Sort insights by value score
        sorted_insights = sorted(insights_data, key=lambda x: x.get("value_score", 0), reverse=True)
        
        # Render insights in a responsive grid
        for i in range(0, len(sorted_insights), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                if i < len(sorted_insights):
                    self._render_insight_card(sorted_insights[i])
            
            with col2:
                if i + 1 < len(sorted_insights):
                    self._render_insight_card(sorted_insights[i + 1])
    
    def _render_insight_card(self, insight: Dict[str, Any]) -> None:
        """Render an individual insight card."""
        # Extract key data
        home_team = insight.get("home_team", "Unknown")
        away_team = insight.get("away_team", "Unknown")
        match_date = insight.get("match_date", "TBD")
        league = insight.get("league", "Unknown")
        confidence = insight.get("confidence", 0)
        value_score = insight.get("value_score", 0)
        prediction = insight.get("prediction", {})
        odds = insight.get("odds", {})
        
        # Determine confidence and value classes
        confidence_class = self._get_confidence_class(confidence)
        value_class = self._get_value_class(value_score)
        
        # Format match date
        if isinstance(match_date, str):
            try:
                match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                formatted_date = match_date.strftime("%b %d, %H:%M")
            except:
                formatted_date = match_date
        else:
            formatted_date = match_date.strftime("%b %d, %H:%M") if match_date else "TBD"
        
        # Get best bet recommendation
        best_bet = self._get_best_bet_recommendation(prediction, odds)
        
        st.markdown(f"""
        <div class="insight-card {value_class}">
            <div class="insight-header">
                <div class="match-info">
                    <div class="teams">
                        <span class="home-team">{home_team}</span>
                        <span class="vs">vs</span>
                        <span class="away-team">{away_team}</span>
                    </div>
                    <div class="match-meta">
                        <span class="league">{league}</span>
                        <span class="date">{formatted_date}</span>
                    </div>
                </div>
                <div class="confidence-badge {confidence_class}">
                    {confidence:.0%}
                </div>
            </div>
            
            <div class="insight-content">
                <div class="value-score">
                    <span class="value-label">Value Score</span>
                    <span class="value-number {value_class}">{value_score:.1f}/10</span>
                </div>
                
                <div class="best-bet">
                    <h4>💡 Recommended Bet</h4>
                    <div class="bet-details">
                        <span class="bet-type">{best_bet['type']}</span>
                        <span class="bet-odds">@ {best_bet['odds']:.2f}</span>
                        <span class="bet-edge">Edge: {best_bet['edge']:.1%}</span>
                    </div>
                </div>
                
                <div class="predictions">
                    <h5>🔮 Predictions</h5>
                    <div class="prediction-grid">
                        <div class="prediction-item">
                            <span class="pred-label">Home Win</span>
                            <span class="pred-value">{prediction.get('home_win', 0):.0%}</span>
                        </div>
                        <div class="prediction-item">
                            <span class="pred-label">Draw</span>
                            <span class="pred-value">{prediction.get('draw', 0):.0%}</span>
                        </div>
                        <div class="prediction-item">
                            <span class="pred-label">Away Win</span>
                            <span class="pred-value">{prediction.get('away_win', 0):.0%}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="insight-actions">
                <button class="action-btn primary" onclick="viewMatchDetails('{insight.get('match_id', '')}')">
                    📊 View Details
                </button>
                <button class="action-btn secondary" onclick="addToWatchlist('{insight.get('match_id', '')}')">
                    ⭐ Add to Watchlist
                </button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _get_confidence_class(self, confidence: float) -> str:
        """Get CSS class for confidence level."""
        if confidence >= self.confidence_thresholds["high"]:
            return "high"
        elif confidence >= self.confidence_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def _get_value_class(self, value_score: float) -> str:
        """Get CSS class for value score."""
        if value_score >= self.value_thresholds["excellent"]:
            return "excellent"
        elif value_score >= self.value_thresholds["good"]:
            return "good"
        elif value_score >= self.value_thresholds["fair"]:
            return "fair"
        else:
            return "poor"
    
    def _get_best_bet_recommendation(self, prediction: Dict[str, float], odds: Dict[str, float]) -> Dict[str, Any]:
        """Get the best betting recommendation."""
        # Calculate expected values for each outcome
        outcomes = ["home_win", "draw", "away_win"]
        best_bet = {"type": "No Value", "odds": 0, "edge": 0}
        best_edge = 0
        
        for outcome in outcomes:
            pred_prob = prediction.get(outcome, 0)
            odds_value = odds.get(outcome, 0)
            
            if odds_value > 0:
                implied_prob = 1 / odds_value
                edge = pred_prob - implied_prob
                
                if edge > best_edge:
                    best_edge = edge
                    outcome_names = {
                        "home_win": "Home Win",
                        "draw": "Draw", 
                        "away_win": "Away Win"
                    }
                    best_bet = {
                        "type": outcome_names[outcome],
                        "odds": odds_value,
                        "edge": edge
                    }
        
        return best_bet
    
    def _render_summary_analytics(self, insights_data: List[Dict[str, Any]]) -> None:
        """Render summary analytics and charts."""
        st.markdown("### 📈 Analytics Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_value_distribution_chart(insights_data)
        
        with col2:
            self._render_confidence_distribution_chart(insights_data)
    
    def _render_value_distribution_chart(self, insights_data: List[Dict[str, Any]]) -> None:
        """Render value score distribution chart."""
        value_scores = [insight.get("value_score", 0) for insight in insights_data]
        
        fig = go.Figure(data=[
            go.Histogram(
                x=value_scores,
                nbinsx=10,
                marker_color='rgba(102, 126, 234, 0.7)',
                marker_line_color='rgba(102, 126, 234, 1)',
                marker_line_width=1
            )
        ])
        
        fig.update_layout(
            title="Value Score Distribution",
            xaxis_title="Value Score",
            yaxis_title="Number of Matches",
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_confidence_distribution_chart(self, insights_data: List[Dict[str, Any]]) -> None:
        """Render confidence distribution chart."""
        confidences = [insight.get("confidence", 0) for insight in insights_data]
        
        # Create confidence bins
        high_conf = sum(1 for c in confidences if c >= 0.8)
        med_conf = sum(1 for c in confidences if 0.6 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.6)
        
        fig = go.Figure(data=[
            go.Bar(
                x=['High (≥80%)', 'Medium (60-80%)', 'Low (<60%)'],
                y=[high_conf, med_conf, low_conf],
                marker_color=['#28a745', '#ffc107', '#dc3545']
            )
        ])
        
        fig.update_layout(
            title="Confidence Distribution",
            xaxis_title="Confidence Level",
            yaxis_title="Number of Matches",
            showlegend=False,
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_no_insights_message(self) -> None:
        """Render message when no insights are available."""
        st.markdown("""
        <div class="no-insights-message">
            <div class="no-insights-icon">🔍</div>
            <h3>No Betting Insights Available</h3>
            <p>Try adjusting your filters or selecting different leagues and date ranges.</p>
            <div class="suggestions">
                <h4>Suggestions:</h4>
                <ul>
                    <li>Expand your date range</li>
                    <li>Lower the minimum value score threshold</li>
                    <li>Select additional leagues</li>
                    <li>Check if there are upcoming matches in your selected timeframe</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def apply_enhanced_styling(self) -> None:
        """Apply enhanced CSS styling for the insights display."""
        st.markdown("""
        <style>
        /* Enhanced Insights Display Styles */
        
        .enhanced-metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        
        .enhanced-metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }
        
        .enhanced-metric-card.success {
            border-left-color: #28a745;
        }
        
        .enhanced-metric-card.warning {
            border-left-color: #ffc107;
        }
        
        .enhanced-metric-card.info {
            border-left-color: #17a2b8;
        }
        
        .metric-header {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .metric-icon {
            font-size: 1.5rem;
        }
        
        .metric-title {
            color: #666;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .metric-value {
            color: #333;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        
        .metric-delta {
            color: #666;
            font-size: 0.8rem;
        }
        
        .insight-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
            border-left: 4px solid #ddd;
        }
        
        .insight-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }
        
        .insight-card.excellent {
            border-left-color: #28a745;
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.02) 0%, rgba(40, 167, 69, 0.05) 100%);
        }
        
        .insight-card.good {
            border-left-color: #17a2b8;
            background: linear-gradient(135deg, rgba(23, 162, 184, 0.02) 0%, rgba(23, 162, 184, 0.05) 100%);
        }
        
        .insight-card.fair {
            border-left-color: #ffc107;
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.02) 0%, rgba(255, 193, 7, 0.05) 100%);
        }
        
        .insight-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1rem;
        }
        
        .teams {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .home-team, .away-team {
            font-weight: 600;
            color: #333;
        }
        
        .vs {
            color: #999;
            font-size: 0.9rem;
        }
        
        .match-meta {
            display: flex;
            gap: 1rem;
            color: #666;
            font-size: 0.85rem;
        }
        
        .confidence-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .confidence-badge.high {
            background: rgba(40, 167, 69, 0.2);
            color: #28a745;
        }
        
        .confidence-badge.medium {
            background: rgba(255, 193, 7, 0.2);
            color: #ffc107;
        }
        
        .confidence-badge.low {
            background: rgba(220, 53, 69, 0.2);
            color: #dc3545;
        }
        
        .value-score {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem;
            background: rgba(102, 126, 234, 0.05);
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .value-label {
            color: #666;
            font-weight: 600;
        }
        
        .value-number {
            font-size: 1.25rem;
            font-weight: 700;
        }
        
        .value-number.excellent {
            color: #28a745;
        }
        
        .value-number.good {
            color: #17a2b8;
        }
        
        .value-number.fair {
            color: #ffc107;
        }
        
        .best-bet {
            margin-bottom: 1rem;
        }
        
        .best-bet h4 {
            margin: 0 0 0.5rem 0;
            color: #333;
            font-size: 1rem;
        }
        
        .bet-details {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .bet-type {
            background: #667eea;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        .bet-odds, .bet-edge {
            color: #666;
            font-size: 0.85rem;
            font-weight: 600;
        }
        
        .predictions h5 {
            margin: 0 0 0.5rem 0;
            color: #333;
            font-size: 0.9rem;
        }
        
        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
        }
        
        .prediction-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 0.5rem;
            background: rgba(102, 126, 234, 0.05);
            border-radius: 8px;
        }
        
        .pred-label {
            color: #666;
            font-size: 0.75rem;
            margin-bottom: 0.25rem;
        }
        
        .pred-value {
            color: #333;
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .insight-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        
        .action-btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.8rem;
        }
        
        .action-btn.primary {
            background: #667eea;
            color: white;
        }
        
        .action-btn.secondary {
            background: #f8f9fa;
            color: #667eea;
            border: 1px solid #667eea;
        }
        
        .action-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        
        .no-insights-message {
            text-align: center;
            padding: 3rem 2rem;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        
        .no-insights-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        .no-insights-message h3 {
            color: #333;
            margin-bottom: 1rem;
        }
        
        .no-insights-message p {
            color: #666;
            margin-bottom: 2rem;
        }
        
        .suggestions {
            text-align: left;
            max-width: 400px;
            margin: 0 auto;
        }
        
        .suggestions h4 {
            color: #333;
            margin-bottom: 0.5rem;
        }
        
        .suggestions ul {
            color: #666;
            line-height: 1.6;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .insight-header {
                flex-direction: column;
                gap: 1rem;
            }
            
            .bet-details {
                flex-direction: column;
                gap: 0.5rem;
            }
            
            .prediction-grid {
                grid-template-columns: 1fr;
            }
            
            .insight-actions {
                flex-direction: column;
            }
        }
        </style>
        """, unsafe_allow_html=True)

def render_enhanced_insights_display(insights_data: List[Dict[str, Any]]) -> None:
    """Render the complete enhanced insights display."""
    display = EnhancedInsightsDisplay()
    display.apply_enhanced_styling()
    display.render_insights_dashboard(insights_data)
