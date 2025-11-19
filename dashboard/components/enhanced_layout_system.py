#!/usr/bin/env python3
"""
Enhanced Layout System for GoalDiggers Platform
Modern, intuitive, and entertaining dashboard layout focusing on clear betting insights.

Key Features:
- Modern card-based layout with smart spacing
- Interactive match cards with hover effects
- Intelligent betting insights display
- Mobile-first responsive design
- Entertainment-focused micro-interactions
- Clear visual hierarchy for actionable insights
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class EnhancedLayoutSystem:
    """
    Modern layout system optimized for betting insights and user engagement.
    """
    
    def __init__(self):
        """Initialize the enhanced layout system."""
        self.theme_config = {
            'primary_gradient': 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)',
            'success_gradient': 'linear-gradient(135deg, #10b981 0%, #047857 100%)',
            'warning_gradient': 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
            'danger_gradient': 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
            'accent_gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'card_shadow': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            'card_shadow_hover': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
            'border_radius': '12px',
            'spacing_unit': '16px'
        }
        self.load_enhanced_css()
    
    def load_enhanced_css(self):
        """Load the enhanced CSS for modern layout."""
        enhanced_css = """
        <style>
        /* Enhanced GoalDiggers Layout System */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        .gd-layout-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #1f2937;
        }
        
        /* Modern Header Hero Section */
        .gd-hero-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            background-size: 400% 400%;
            animation: gradientShift 8s ease infinite;
            padding: 3rem 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
        }
        
        .gd-hero-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .gd-hero-title {
            font-size: 3rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            animation: titlePulse 3s ease-in-out infinite;
        }
        
        .gd-hero-subtitle {
            font-size: 1.25rem;
            margin: 1rem 0 0 0;
            opacity: 0.9;
            font-weight: 400;
        }
        
        /* Modern Match Cards */
        .gd-match-card {
            background: white;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #e5e7eb;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            cursor: pointer;
        }
        
        .gd-match-card:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            border-color: #3b82f6;
        }
        
        .gd-match-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
            transition: left 0.5s;
        }
        
        .gd-match-card:hover::before {
            left: 100%;
        }
        
        /* Team vs Team Layout */
        .gd-team-vs-layout {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 1.5rem 0;
        }
        
        .gd-team-info {
            flex: 1;
            text-align: center;
            padding: 1rem;
        }
        
        .gd-team-name {
            font-size: 1.25rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }
        
        .gd-team-form {
            font-size: 0.875rem;
            color: #6b7280;
            display: flex;
            justify-content: center;
            gap: 2px;
            margin-top: 0.5rem;
        }
        
        .gd-vs-separator {
            flex: 0 0 auto;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1.1rem;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        /* Betting Insights Cards */
        .gd-insight-card {
            background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .gd-insight-card.value-bet {
            border-color: #10b981;
            background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
        }
        
        .gd-insight-card.value-bet::before {
            content: '💰';
            position: absolute;
            top: -10px;
            right: -10px;
            background: #10b981;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        .gd-insight-card.high-confidence {
            border-color: #3b82f6;
            background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
        }
        
        /* Prediction Pills */
        .gd-prediction-pills {
            display: flex;
            gap: 0.75rem;
            margin: 1rem 0;
            flex-wrap: wrap;
        }
        
        .gd-prediction-pill {
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.875rem;
            font-weight: 500;
            color: white;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .gd-prediction-pill:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .gd-prediction-pill.home-win {
            background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        }
        
        .gd-prediction-pill.draw {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        }
        
        .gd-prediction-pill.away-win {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }
        
        /* Interactive Elements */
        .gd-interactive-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.375rem 0.75rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 50px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
        }
        
        .gd-interactive-badge:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        }
        
        /* Stats Grid */
        .gd-stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .gd-stat-item {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .gd-stat-item:hover {
            border-color: #3b82f6;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
        }
        
        .gd-stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.25rem;
        }
        
        .gd-stat-label {
            font-size: 0.875rem;
            color: #6b7280;
            font-weight: 500;
        }
        
        /* Progress Indicators */
        .gd-progress-bar {
            background: #e5e7eb;
            border-radius: 50px;
            height: 8px;
            overflow: hidden;
            position: relative;
        }
        
        .gd-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
            border-radius: 50px;
            transition: width 1s ease-out;
            position: relative;
        }
        
        .gd-progress-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            right: 0;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
            animation: progressShimmer 2s infinite;
        }
        
        /* Mobile Responsiveness */
        @media (max-width: 768px) {
            .gd-hero-title {
                font-size: 2rem;
            }
            
            .gd-team-vs-layout {
                flex-direction: column;
                gap: 1rem;
            }
            
            .gd-vs-separator {
                order: 2;
                margin: 0.5rem 0;
            }
            
            .gd-prediction-pills {
                justify-content: center;
            }
            
            .gd-stats-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 0.75rem;
            }
        }
        
        /* Animations */
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes titlePulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }
        
        @keyframes progressShimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .gd-fade-in {
            animation: fadeInUp 0.6s ease-out;
        }
        
        /* Form Elements and Betting Insights */
        .gd-form-result {
            display: inline-block;
            width: 24px;
            height: 24px;
            border-radius: 4px;
            text-align: center;
            line-height: 24px;
            font-weight: bold;
            font-size: 12px;
            color: white;
            margin: 1px;
        }
        
        .gd-form-result.win {
            background: #10b981;
        }
        
        .gd-form-result.draw {
            background: #f59e0b;
        }
        
        .gd-form-result.loss {
            background: #ef4444;
        }
        
        /* Value Betting Highlights */
        .gd-value-highlight {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 2px solid #f59e0b;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            position: relative;
        }
        
        .gd-value-highlight::before {
            content: '⭐';
            position: absolute;
            top: -8px;
            left: 16px;
            background: #f59e0b;
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
        }
        
        /* Enhanced Tooltips */
        .gd-tooltip {
            position: relative;
            cursor: help;
        }
        
        .gd-tooltip:hover::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #1f2937;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            white-space: nowrap;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .gd-tooltip:hover::before {
            content: '';
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%) translateY(100%);
            border: 4px solid transparent;
            border-top-color: #1f2937;
            z-index: 1000;
        }
        </style>
        """
        
        st.markdown(enhanced_css, unsafe_allow_html=True)
    
    def render_hero_header(self, title: str = "GoalDiggers", subtitle: str = "Football Betting Intelligence"):
        """Render an engaging hero header."""
        hero_html = f"""
        <div class="gd-layout-container">
            <div class="gd-hero-header gd-fade-in">
                <h1 class="gd-hero-title">⚽ {title}</h1>
                <p class="gd-hero-subtitle">{subtitle}</p>
            </div>
        </div>
        """
        st.markdown(hero_html, unsafe_allow_html=True)
    
    def render_match_card(self, match_data: Dict[str, Any], include_insights: bool = True) -> None:
        """Render an enhanced match card with betting insights."""
        try:
            home_team = match_data.get('home_team', 'Unknown')
            away_team = match_data.get('away_team', 'Unknown')
            match_time = match_data.get('match_time', 'TBD')
            league = match_data.get('league', 'Unknown League')
            
            # Predictions
            home_prob = match_data.get('home_win_probability', 0.33) * 100
            draw_prob = match_data.get('draw_probability', 0.33) * 100
            away_prob = match_data.get('away_win_probability', 0.33) * 100
            
            # Form data (mock if not available)
            home_form = match_data.get('home_form', ['W', 'W', 'D', 'L', 'W'])
            away_form = match_data.get('away_form', ['L', 'W', 'W', 'D', 'L'])
            
            # Value betting indicator
            is_value_bet = match_data.get('is_value_bet', False)
            confidence = match_data.get('confidence', 0.7)
            
            card_class = "gd-match-card"
            if is_value_bet:
                card_class += " gd-value-bet"
            
            # Generate form display
            def format_form(form_list):
                form_html = ""
                for result in form_list[-5:]:  # Last 5 matches
                    result_class = result.lower()
                    if result_class == 'w':
                        result_class = 'win'
                    elif result_class == 'd':
                        result_class = 'draw'
                    elif result_class == 'l':
                        result_class = 'loss'
                    form_html += f'<span class="gd-form-result {result_class}">{result}</span>'
                return form_html
            
            match_html = f"""
            <div class="gd-layout-container">
                <div class="{card_class} gd-fade-in">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                        <div style="color: #6b7280; font-size: 0.875rem; font-weight: 500;">
                            {league} • {match_time}
                        </div>
                        {f'<div class="gd-interactive-badge gd-tooltip" data-tooltip="High value betting opportunity!">💰 Value Bet</div>' if is_value_bet else ''}
                    </div>
                    
                    <div class="gd-team-vs-layout">
                        <div class="gd-team-info">
                            <div class="gd-team-name">{home_team}</div>
                            <div class="gd-team-form">{format_form(home_form)}</div>
                        </div>
                        
                        <div class="gd-vs-separator">VS</div>
                        
                        <div class="gd-team-info">
                            <div class="gd-team-name">{away_team}</div>
                            <div class="gd-team-form">{format_form(away_form)}</div>
                        </div>
                    </div>
                    
                    <div class="gd-prediction-pills">
                        <div class="gd-prediction-pill home-win gd-tooltip" data-tooltip="Home team win probability">
                            🏠 {home_prob:.1f}%
                        </div>
                        <div class="gd-prediction-pill draw gd-tooltip" data-tooltip="Draw probability">
                            🤝 {draw_prob:.1f}%
                        </div>
                        <div class="gd-prediction-pill away-win gd-tooltip" data-tooltip="Away team win probability">
                            ✈️ {away_prob:.1f}%
                        </div>
                    </div>
                    
                    <div class="gd-stats-grid">
                        <div class="gd-stat-item gd-tooltip" data-tooltip="AI confidence in prediction">
                            <div class="gd-stat-value">{confidence*100:.0f}%</div>
                            <div class="gd-stat-label">Confidence</div>
                        </div>
                        <div class="gd-stat-item gd-tooltip" data-tooltip="Expected value for betting">
                            <div class="gd-stat-value">{match_data.get('expected_value', 0.05)*100:+.1f}%</div>
                            <div class="gd-stat-label">Expected Value</div>
                        </div>
                        <div class="gd-stat-item gd-tooltip" data-tooltip="Risk assessment">
                            <div class="gd-stat-value">{match_data.get('risk_level', 'Medium')}</div>
                            <div class="gd-stat-label">Risk Level</div>
                        </div>
                    </div>
                </div>
            </div>
            """
            
            st.markdown(match_html, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering match card: {e}")
            # Fallback to basic display
            st.error(f"Error displaying match: {match_data.get('home_team', 'Unknown')} vs {match_data.get('away_team', 'Unknown')}")
    
    def render_insights_dashboard(self, insights_data: List[Dict[str, Any]]) -> None:
        """Render the betting insights dashboard."""
        try:
            if not insights_data:
                st.info("📊 No betting insights available for the selected criteria.")
                return
            
            # Header
            self.render_section_header("💡 Betting Insights Dashboard", "Actionable opportunities based on AI analysis")
            
            # Summary stats
            total_insights = len(insights_data)
            value_bets = len([i for i in insights_data if i.get('is_value_bet', False)])
            avg_confidence = sum(i.get('confidence', 0) for i in insights_data) / total_insights if total_insights > 0 else 0
            
            summary_html = f"""
            <div class="gd-layout-container">
                <div class="gd-stats-grid" style="margin: 2rem 0;">
                    <div class="gd-stat-item">
                        <div class="gd-stat-value">{total_insights}</div>
                        <div class="gd-stat-label">Total Matches</div>
                    </div>
                    <div class="gd-stat-item">
                        <div class="gd-stat-value">{value_bets}</div>
                        <div class="gd-stat-label">Value Bets</div>
                    </div>
                    <div class="gd-stat-item">
                        <div class="gd-stat-value">{avg_confidence*100:.0f}%</div>
                        <div class="gd-stat-label">Avg Confidence</div>
                    </div>
                    <div class="gd-stat-item">
                        <div class="gd-stat-value">{(value_bets/total_insights*100) if total_insights > 0 else 0:.0f}%</div>
                        <div class="gd-stat-label">Value Rate</div>
                    </div>
                </div>
            </div>
            """
            st.markdown(summary_html, unsafe_allow_html=True)
            
            # Render individual insights
            for insight in insights_data:
                self.render_match_card(insight, include_insights=True)
                
        except Exception as e:
            logger.error(f"Error rendering insights dashboard: {e}")
            st.error("Error loading betting insights. Please try again.")
    
    def render_section_header(self, title: str, subtitle: str = None, icon: str = "") -> None:
        """Render a modern section header."""
        header_html = f"""
        <div class="gd-layout-container">
            <div style="margin: 2rem 0 1.5rem 0;">
                <h2 style="font-size: 1.875rem; font-weight: 700; color: #1f2937; margin: 0; display: flex; align-items: center; gap: 0.5rem;">
                    {icon} {title}
                </h2>
                {f'<p style="color: #6b7280; margin: 0.5rem 0 0 0; font-size: 1rem;">{subtitle}</p>' if subtitle else ''}
                <div style="width: 60px; height: 4px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 2px; margin-top: 0.75rem;"></div>
            </div>
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    def render_progress_indicator(self, value: float, max_value: float = 100, label: str = "", color: str = "success") -> None:
        """Render an animated progress indicator."""
        percentage = min(100, (value / max_value) * 100) if max_value > 0 else 0
        
        color_map = {
            'success': 'linear-gradient(90deg, #10b981 0%, #34d399 100%)',
            'warning': 'linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%)',
            'danger': 'linear-gradient(90deg, #ef4444 0%, #f87171 100%)',
            'primary': 'linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%)'
        }
        
        gradient = color_map.get(color, color_map['success'])
        
        progress_html = f"""
        <div class="gd-layout-container">
            <div style="margin: 1rem 0;">
                {f'<div style="font-size: 0.875rem; font-weight: 500; color: #374151; margin-bottom: 0.5rem;">{label}</div>' if label else ''}
                <div class="gd-progress-bar">
                    <div class="gd-progress-fill" style="width: {percentage}%; background: {gradient};"></div>
                </div>
                <div style="text-align: right; font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;">
                    {percentage:.1f}%
                </div>
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    def render_interactive_filter_bar(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Render an interactive filter bar for match selection."""
        st.markdown('<div class="gd-layout-container">', unsafe_allow_html=True)
        
        # Create columns for filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            value_bets_only = st.checkbox("💰 Value Bets Only", value=filters.get('value_bets_only', False))
        
        with col2:
            min_confidence = st.slider("🎯 Min Confidence", 0.5, 1.0, filters.get('min_confidence', 0.7), 0.05)
        
        with col3:
            max_risk = st.selectbox("⚠️ Max Risk Level", ["Low", "Medium", "High"], index=["Low", "Medium", "High"].index(filters.get('max_risk', 'Medium')))
        
        with col4:
            sort_by = st.selectbox("📊 Sort By", ["Expected Value", "Confidence", "Match Time"], index=0)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return {
            'value_bets_only': value_bets_only,
            'min_confidence': min_confidence,
            'max_risk': max_risk,
            'sort_by': sort_by
        }
    
    def render_league_performance_chart(self, league_data: pd.DataFrame) -> None:
        """Render a league performance comparison chart."""
        try:
            if league_data.empty:
                st.info("📊 No league data available for visualization.")
                return
            
            self.render_section_header("🏆 League Performance Analysis", "AI prediction accuracy across leagues")
            
            # Create the chart
            fig = px.bar(
                league_data,
                x='league',
                y='accuracy',
                color='accuracy',
                color_continuous_scale='Viridis',
                title="Prediction Accuracy by League",
                labels={'accuracy': 'Accuracy (%)', 'league': 'League'},
                text='accuracy'
            )
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                height=400,
                showlegend=False,
                font=dict(family="Inter", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font=dict(size=16, color='#1f2937'),
                xaxis=dict(title_font=dict(color='#374151'), tickfont=dict(color='#6b7280')),
                yaxis=dict(title_font=dict(color='#374151'), tickfont=dict(color='#6b7280'))
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"Error rendering league performance chart: {e}")
            st.error("Error loading league performance data.")
    
    def render_mobile_optimized_layout(self, content_blocks: List[Dict[str, Any]]) -> None:
        """Render a mobile-optimized layout for smaller screens."""
        try:
            # Check if mobile
            is_mobile = st.session_state.get('is_mobile', False)
            
            if is_mobile:
                # Single column layout for mobile
                for block in content_blocks:
                    block_type = block.get('type', 'card')
                    block_data = block.get('data', {})
                    
                    if block_type == 'match_card':
                        self.render_match_card(block_data)
                    elif block_type == 'section_header':
                        self.render_section_header(block_data.get('title', ''), block_data.get('subtitle', ''))
                    elif block_type == 'progress':
                        self.render_progress_indicator(
                            block_data.get('value', 0),
                            block_data.get('max_value', 100),
                            block_data.get('label', '')
                        )
            else:
                # Desktop layout with columns
                cols = st.columns(2)
                for i, block in enumerate(content_blocks):
                    with cols[i % 2]:
                        block_type = block.get('type', 'card')
                        block_data = block.get('data', {})
                        
                        if block_type == 'match_card':
                            self.render_match_card(block_data)
                        elif block_type == 'section_header':
                            self.render_section_header(block_data.get('title', ''), block_data.get('subtitle', ''))
                        elif block_type == 'progress':
                            self.render_progress_indicator(
                                block_data.get('value', 0),
                                block_data.get('max_value', 100),
                                block_data.get('label', '')
                            )
                            
        except Exception as e:
            logger.error(f"Error rendering mobile optimized layout: {e}")
            # Fallback to simple layout
            for block in content_blocks:
                data = block.get('data', {})
                # If data is a dict, pretty-print as JSON, else as plain text
                import json
                if isinstance(data, dict):
                    st.markdown(f'<pre style="background:#f8fafc;padding:1em;border-radius:8px;overflow-x:auto;">{json.dumps(data, indent=2)}</pre>', unsafe_allow_html=True)
                else:
                    st.markdown(str(data))

# Global instance for easy access
enhanced_layout = EnhancedLayoutSystem()
