#!/usr/bin/env python3
"""
Enhanced Prediction Display Components for GoalDiggers Platform

Advanced visualization components for cross-league predictions:
- Animated confidence meters with smooth transitions
- Interactive probability distributions
- Enhanced insight integration with visual appeal
- Performance-optimized rendering (<1 second load time)
- Responsive design for all screen sizes
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedPredictionDisplay:
    """Enhanced prediction display with animated visualizations."""
    
    def __init__(self, design_system=None):
        """Initialize enhanced prediction display."""
        self.design_system = design_system
        self.animation_duration = 800  # milliseconds
        self.colors = self._get_colors()
        
        logger.info("📊 Enhanced Prediction Display initialized")
    
    def _get_colors(self) -> Dict[str, str]:
        """Get color scheme from design system or defaults."""
        if self.design_system:
            return self.design_system.brand_colors
        
        return {
            'primary': '#1f4e79',
            'secondary': '#28a745',
            'accent': '#fd7e14',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8'
        }
    
    def render_animated_prediction_card(self, prediction: Dict[str, Any], 
                                      match_data: Dict[str, Any]) -> None:
        """
        Render animated prediction card with confidence meters.
        
        Args:
            prediction: Prediction results from ML engine
            match_data: Match context data
        """
        start_time = time.time()
        
        try:
            # Extract prediction data
            predictions = prediction.get('predictions', {})
            confidence = prediction.get('confidence', {}).get('overall', 0.7)
            
            home_team = match_data.get('home_team', 'Home Team')
            away_team = match_data.get('away_team', 'Away Team')
            
            # Create main prediction card
            st.markdown("""
            <div class="prediction-card">
                <h3>🔮 AI Prediction Results</h3>
                <div class="prediction-subtitle">Advanced ML Analysis with Cross-League Intelligence</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Animated confidence header
            self._render_confidence_header(confidence)
            
            # Main prediction visualization
            col1, col2, col3 = st.columns(3)
            
            with col1:
                self._render_outcome_meter(
                    f"🏠 {home_team} Win",
                    predictions.get('home_win', 0.4),
                    self.colors['primary'],
                    "Home advantage and league strength considered"
                )
            
            with col2:
                self._render_outcome_meter(
                    "🤝 Draw",
                    predictions.get('draw', 0.3),
                    self.colors['warning'],
                    "Balanced match probability"
                )
            
            with col3:
                self._render_outcome_meter(
                    f"✈️ {away_team} Win",
                    predictions.get('away_win', 0.3),
                    self.colors['info'],
                    "Away team challenge factor"
                )
            
            # Interactive probability distribution
            self._render_probability_distribution(predictions, home_team, away_team)
            
            # Performance tracking
            render_time = time.time() - start_time
            if render_time > 1.0:
                logger.warning(f"Prediction display render time: {render_time:.2f}s (target: <1s)")
            else:
                logger.debug(f"Prediction display rendered in {render_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Failed to render prediction card: {e}")
            st.error("Unable to display prediction results")
    
    def _render_confidence_header(self, confidence: float) -> None:
        """Render animated confidence header."""
        confidence_pct = confidence * 100
        
        # Determine confidence level and color
        if confidence >= 0.8:
            level = "Very High"
            color = self.colors['success']
            icon = "🎯"
        elif confidence >= 0.6:
            level = "High"
            color = self.colors['primary']
            icon = "📊"
        elif confidence >= 0.4:
            level = "Moderate"
            color = self.colors['warning']
            icon = "⚖️"
        else:
            level = "Low"
            color = self.colors['danger']
            icon = "⚠️"
        
        # Animated confidence display
        st.markdown(f"""
        <div class="confidence-header" style="
            background: linear-gradient(90deg, {color}22 0%, {color}11 100%);
            border-left: 4px solid {color};
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        ">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: 1.2rem;">{icon}</span>
                    <strong style="margin-left: 0.5rem;">Prediction Confidence: {level}</strong>
                </div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {color};">
                    {confidence_pct:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Animated progress bar
        st.progress(confidence)
    
    def _render_outcome_meter(self, label: str, probability: float, 
                            color: str, description: str) -> None:
        """Render individual outcome meter with animation."""
        probability_pct = probability * 100
        
        # Create metric with custom styling
        st.markdown(f"""
        <div class="outcome-meter" style="
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            text-align: center;
            margin: 0.5rem 0;
            transition: transform 0.3s ease;
        ">
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; color: #333;">
                {label}
            </div>
            <div style="font-size: 2rem; font-weight: bold; color: {color}; margin: 0.5rem 0;">
                {probability_pct:.1f}%
            </div>
            <div style="font-size: 0.9rem; color: #666; margin-top: 0.5rem;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Animated progress bar
        st.progress(probability)
    
    def _render_probability_distribution(self, predictions: Dict[str, float], 
                                       home_team: str, away_team: str) -> None:
        """Render interactive probability distribution chart."""
        try:
            # Prepare data for visualization
            outcomes = ['Home Win', 'Draw', 'Away Win']
            probabilities = [
                predictions.get('home_win', 0.4),
                predictions.get('draw', 0.3),
                predictions.get('away_win', 0.3)
            ]
            colors_list = [self.colors['primary'], self.colors['warning'], self.colors['info']]
            
            # Create interactive bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=outcomes,
                    y=probabilities,
                    marker_color=colors_list,
                    text=[f'{p:.1%}' for p in probabilities],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': f'Match Outcome Probabilities: {home_team} vs {away_team}',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#333'}
                },
                xaxis_title='Outcome',
                yaxis_title='Probability',
                yaxis=dict(tickformat='.0%', range=[0, max(probabilities) * 1.2]),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial, sans-serif", size=12),
                height=400,
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # Add animation
            fig.update_traces(
                marker=dict(
                    line=dict(width=2, color='rgba(0,0,0,0.3)')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
        except Exception as e:
            logger.error(f"Failed to render probability distribution: {e}")
            st.warning("Probability chart temporarily unavailable")
    
    def render_cross_league_insights(self, analysis: Dict[str, Any], 
                                   match_data: Dict[str, Any]) -> None:
        """Render cross-league analysis insights with visual appeal."""
        if not analysis or 'home_league_profile' not in analysis:
            return
        
        st.markdown("### 🌍 Cross-League Analysis")
        
        # League comparison section
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_league_profile_card(
                "🏠 Home League",
                analysis.get('home_league_profile', {}),
                match_data.get('home_league', 'Unknown')
            )
        
        with col2:
            self._render_league_profile_card(
                "✈️ Away League", 
                analysis.get('away_league_profile', {}),
                match_data.get('away_league', 'Unknown')
            )
        
        # Strength differential visualization
        self._render_strength_differential(analysis)
        
        # Playing style compatibility
        self._render_style_compatibility(analysis)
    
    def _render_league_profile_card(self, title: str, profile: Dict[str, Any], 
                                  league_name: str) -> None:
        """Render individual league profile card."""
        if not profile:
            return
        
        strength = profile.get('strength_coefficient', 0)
        pace = profile.get('pace_factor', 0)
        technical = profile.get('technical_index', 0)
        
        st.markdown(f"""
        <div class="league-profile-card" style="
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            margin: 0.5rem 0;
        ">
            <h4 style="color: {self.colors['primary']}; margin-bottom: 1rem;">{title}</h4>
            <div style="font-weight: 600; font-size: 1.1rem; margin-bottom: 1rem;">{league_name}</div>
            
            <div style="margin: 0.5rem 0;">
                <strong>League Strength:</strong> 
                <span style="color: {self.colors['primary']};">{strength:.2f}</span>
            </div>
            <div style="margin: 0.5rem 0;">
                <strong>Pace Factor:</strong> 
                <span style="color: {self.colors['info']};">{pace:.2f}</span>
            </div>
            <div style="margin: 0.5rem 0;">
                <strong>Technical Index:</strong> 
                <span style="color: {self.colors['success']};">{technical:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_strength_differential(self, analysis: Dict[str, Any]) -> None:
        """Render league strength differential visualization."""
        strength_diff = analysis.get('strength_differential', 0)
        
        if abs(strength_diff) < 0.01:
            return
        
        # Determine advantage
        if strength_diff > 0:
            advantage_text = "🏠 Home league advantage"
            color = self.colors['success']
            icon = "📈"
        else:
            advantage_text = "✈️ Away league advantage"
            color = self.colors['info']
            icon = "📊"
        
        st.markdown(f"""
        <div class="strength-differential" style="
            background: linear-gradient(90deg, {color}22 0%, {color}11 100%);
            border-left: 4px solid {color};
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        ">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                <div>
                    <strong>{advantage_text}</strong>
                    <div style="font-size: 0.9rem; color: #666; margin-top: 0.25rem;">
                        Strength differential: {abs(strength_diff):.3f}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_style_compatibility(self, analysis: Dict[str, Any]) -> None:
        """Render playing style compatibility visualization."""
        compatibility = analysis.get('playing_style_compatibility', 0.5)
        
        if compatibility > 0.8:
            level = "Very High"
            color = self.colors['success']
            icon = "🎯"
            description = "Highly compatible playing styles - tactical battle expected"
        elif compatibility > 0.6:
            level = "High"
            color = self.colors['primary']
            icon = "⚖️"
            description = "Compatible styles - balanced encounter likely"
        elif compatibility > 0.4:
            level = "Moderate"
            color = self.colors['warning']
            icon = "🔄"
            description = "Moderately different styles - interesting dynamics"
        else:
            level = "Low"
            color = self.colors['accent']
            icon = "⚡"
            description = "Contrasting styles - unpredictable match expected"
        
        st.markdown(f"""
        <div class="style-compatibility" style="
            background: linear-gradient(90deg, {color}22 0%, {color}11 100%);
            border-left: 4px solid {color};
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        ">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
                    <div>
                        <strong>Playing Style Compatibility: {level}</strong>
                        <div style="font-size: 0.9rem; color: #666; margin-top: 0.25rem;">
                            {description}
                        </div>
                    </div>
                </div>
                <div style="font-size: 1.2rem; font-weight: bold; color: {color};">
                    {compatibility:.1%}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for compatibility
        st.progress(compatibility)
    
    def render_entertaining_commentary(self, prediction: Dict[str, Any], 
                                     match_data: Dict[str, Any]) -> None:
        """Render entertaining AI commentary with visual appeal."""
        st.markdown("### 🎙️ AI Match Preview")
        
        # Get insights from prediction
        insights = prediction.get('insights', [])
        
        # Generate commentary
        commentary = self._generate_enhanced_commentary(prediction, match_data)
        
        # Render commentary card
        st.markdown(f"""
        <div class="commentary-card" style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        ">
            <h4 style="color: white; margin-bottom: 1rem;">🎭 Match Preview</h4>
            <p style="font-size: 1.1rem; line-height: 1.6; margin: 0;">{commentary}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Render insights if available
        if insights:
            st.markdown("#### 💡 Key Insights")
            for i, insight in enumerate(insights[:3]):  # Limit to 3 insights
                st.info(f"**Insight {i+1}:** {insight}")
    
    def _generate_enhanced_commentary(self, prediction: Dict[str, Any], 
                                    match_data: Dict[str, Any]) -> str:
        """Generate enhanced entertaining commentary."""
        home_team = match_data.get('home_team', 'Home Team')
        away_team = match_data.get('away_team', 'Away Team')
        home_league = match_data.get('home_league', 'Unknown League')
        away_league = match_data.get('away_league', 'Unknown League')
        
        predictions = prediction.get('predictions', {})
        home_prob = predictions.get('home_win', 0.4)
        confidence = prediction.get('confidence', {}).get('overall', 0.7)
        
        is_cross_league = match_data.get('is_cross_league', False)
        
        if is_cross_league:
            if home_prob > 0.5:
                return f"🔥 {home_team} from {home_league} are expected to showcase their league's tactical superiority against {away_team}! With a {confidence:.0%} confidence level, our AI predicts the home advantage combined with {home_league}'s distinctive playing style could prove decisive in this fascinating cross-league encounter. Expect fireworks as two football philosophies collide!"
            else:
                return f"⚡ {away_team} from {away_league} are ready to make a statement on foreign soil! This cross-league clash promises to be a tactical masterpiece as {away_league}'s approach meets {home_league}'s style. With {confidence:.0%} confidence, our AI suggests this could be the upset of the season - expect the unexpected when leagues collide!"
        else:
            return f"🎯 A classic {home_league} battle awaits! {home_team} vs {away_team} promises to be a tactical chess match with both teams intimately familiar with each other's strengths and weaknesses. Our AI analysis shows {confidence:.0%} confidence in the prediction - this domestic encounter could go either way!"

# Global instance
_enhanced_prediction_display = None

def get_enhanced_prediction_display(design_system=None) -> EnhancedPredictionDisplay:
    """Get global enhanced prediction display instance."""
    global _enhanced_prediction_display
    if _enhanced_prediction_display is None:
        _enhanced_prediction_display = EnhancedPredictionDisplay(design_system)
    return _enhanced_prediction_display
