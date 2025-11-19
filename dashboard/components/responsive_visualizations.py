#!/usr/bin/env python3
"""
Responsive Visualizations for GoalDiggers Platform

Modern, interactive, and responsive data visualizations that provide:
- Clear and actionable betting insights
- Mobile-optimized responsive design
- Interactive elements with meaningful feedback
- Professional appearance aligned with modern design standards
- Purpose-driven visualizations for decision-making

All visualizations are optimized for performance and accessibility.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

logger = logging.getLogger(__name__)

class ResponsiveVisualizations:
    """
    Responsive visualization components for the GoalDiggers platform.
    """
    
    def __init__(self):
        """Initialize responsive visualizations."""
        self.color_palette = {
            'primary': '#1e40af',
            'secondary': '#059669',
            'accent': '#f59e0b',
            'success': '#10b981',
            'warning': '#f59e0b',
            'error': '#ef4444',
            'neutral': '#6b7280',
            'background': '#f9fafb',
            'surface': '#ffffff'
        }
        
        self.chart_config = {
            'displayModeBar': False,
            'responsive': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian'
            ]
        }
    
    def create_probability_chart(self, 
                               home_win: float, 
                               draw: float, 
                               away_win: float,
                               home_team: str = "Home",
                               away_team: str = "Away") -> go.Figure:
        """
        Create a responsive probability visualization chart.
        
        Args:
            home_win: Home win probability (0-1)
            draw: Draw probability (0-1)
            away_win: Away win probability (0-1)
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Plotly figure object
        """
        # Prepare data
        outcomes = ['Home Win', 'Draw', 'Away Win']
        probabilities = [home_win, draw, away_win]
        colors = [self.color_palette['primary'], self.color_palette['neutral'], self.color_palette['secondary']]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=outcomes,
                y=probabilities,
                marker_color=colors,
                text=[f'{p:.1%}' for p in probabilities],
                textposition='auto',
                textfont=dict(size=14, color='white', family='Inter'),
                hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>',
                name='Match Outcome Probabilities'
            )
        ])
        
        # Update layout for responsiveness
        fig.update_layout(
            title=dict(
                text=f'<b>{home_team} vs {away_team}</b><br><sub>Match Outcome Probabilities</sub>',
                x=0.5,
                font=dict(size=18, family='Inter', color=self.color_palette['neutral'])
            ),
            xaxis=dict(
                title='Outcome',
                titlefont=dict(size=14, family='Inter'),
                tickfont=dict(size=12, family='Inter'),
                showgrid=False
            ),
            yaxis=dict(
                title='Probability',
                titlefont=dict(size=14, family='Inter'),
                tickfont=dict(size=12, family='Inter'),
                tickformat='.0%',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                range=[0, max(probabilities) * 1.2]
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            margin=dict(l=50, r=50, t=80, b=50),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_confidence_gauge(self, confidence: float, title: str = "Prediction Confidence") -> go.Figure:
        """
        Create a responsive confidence gauge visualization.
        
        Args:
            confidence: Confidence value (0-1)
            title: Gauge title
            
        Returns:
            Plotly figure object
        """
        # Determine color based on confidence level
        if confidence >= 0.8:
            color = self.color_palette['success']
        elif confidence >= 0.6:
            color = self.color_palette['warning']
        else:
            color = self.color_palette['error']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 16, 'family': 'Inter'}},
            delta={'reference': 70, 'suffix': '%'},
            gauge={
                'axis': {'range': [None, 100], 'tickfont': {'size': 12, 'family': 'Inter'}},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                    {'range': [50, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_team_comparison_radar(self, 
                                   home_stats: Dict[str, float], 
                                   away_stats: Dict[str, float],
                                   home_team: str = "Home",
                                   away_team: str = "Away") -> go.Figure:
        """
        Create a responsive radar chart for team comparison.
        
        Args:
            home_stats: Home team statistics
            away_stats: Away team statistics
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Plotly figure object
        """
        # Default stats if not provided
        default_stats = {
            'Attack': 0.7,
            'Defense': 0.6,
            'Midfield': 0.8,
            'Form': 0.75,
            'Home Advantage': 0.65
        }
        
        home_stats = home_stats or default_stats
        away_stats = away_stats or {k: v * 0.9 for k, v in default_stats.items()}
        
        # Prepare data
        categories = list(home_stats.keys())
        home_values = list(home_stats.values())
        away_values = list(away_stats.values())
        
        fig = go.Figure()
        
        # Add home team trace
        fig.add_trace(go.Scatterpolar(
            r=home_values + [home_values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=f'rgba(30, 64, 175, 0.2)',
            line=dict(color=self.color_palette['primary'], width=2),
            name=home_team,
            hovertemplate='<b>%{theta}</b><br>%{r:.1%}<extra></extra>'
        ))
        
        # Add away team trace
        fig.add_trace(go.Scatterpolar(
            r=away_values + [away_values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=f'rgba(5, 150, 105, 0.2)',
            line=dict(color=self.color_palette['secondary'], width=2),
            name=away_team,
            hovertemplate='<b>%{theta}</b><br>%{r:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickformat='.0%',
                    tickfont=dict(size=10, family='Inter')
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, family='Inter')
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family='Inter')
            ),
            title=dict(
                text=f'<b>Team Comparison</b><br><sub>{home_team} vs {away_team}</sub>',
                x=0.5,
                font=dict(size=16, family='Inter', color=self.color_palette['neutral'])
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            height=400,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_historical_performance_chart(self, 
                                          home_form: List[str], 
                                          away_form: List[str],
                                          home_team: str = "Home",
                                          away_team: str = "Away") -> go.Figure:
        """
        Create a historical performance comparison chart.
        
        Args:
            home_form: Home team recent form ['W', 'L', 'D', 'W', 'W']
            away_form: Away team recent form ['L', 'W', 'D', 'L', 'W']
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Plotly figure object
        """
        # Default form if not provided
        if not home_form:
            home_form = ['W', 'W', 'D', 'L', 'W']
        if not away_form:
            away_form = ['L', 'W', 'W', 'D', 'L']
        
        # Convert form to points (W=3, D=1, L=0)
        form_to_points = {'W': 3, 'D': 1, 'L': 0}
        home_points = [form_to_points[result] for result in home_form]
        away_points = [form_to_points[result] for result in away_form]
        
        # Create match numbers
        matches = list(range(1, len(home_form) + 1))
        
        fig = go.Figure()
        
        # Add home team line
        fig.add_trace(go.Scatter(
            x=matches,
            y=home_points,
            mode='lines+markers',
            name=home_team,
            line=dict(color=self.color_palette['primary'], width=3),
            marker=dict(size=8, color=self.color_palette['primary']),
            hovertemplate='<b>%{fullData.name}</b><br>Match %{x}<br>Result: %{text}<br>Points: %{y}<extra></extra>',
            text=home_form
        ))
        
        # Add away team line
        fig.add_trace(go.Scatter(
            x=matches,
            y=away_points,
            mode='lines+markers',
            name=away_team,
            line=dict(color=self.color_palette['secondary'], width=3),
            marker=dict(size=8, color=self.color_palette['secondary']),
            hovertemplate='<b>%{fullData.name}</b><br>Match %{x}<br>Result: %{text}<br>Points: %{y}<extra></extra>',
            text=away_form
        ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>Recent Form Comparison</b><br><sub>Last {len(home_form)} matches</sub>',
                x=0.5,
                font=dict(size=16, family='Inter', color=self.color_palette['neutral'])
            ),
            xaxis=dict(
                title='Match',
                titlefont=dict(size=14, family='Inter'),
                tickfont=dict(size=12, family='Inter'),
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                dtick=1
            ),
            yaxis=dict(
                title='Points',
                titlefont=dict(size=14, family='Inter'),
                tickfont=dict(size=12, family='Inter'),
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                range=[-0.5, 3.5],
                tickvals=[0, 1, 3],
                ticktext=['Loss (0)', 'Draw (1)', 'Win (3)']
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family='Inter')
            ),
            height=350,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_betting_value_chart(self, 
                                 outcomes: List[str], 
                                 probabilities: List[float], 
                                 odds: List[float]) -> go.Figure:
        """
        Create a betting value analysis chart.
        
        Args:
            outcomes: List of outcomes ['Home Win', 'Draw', 'Away Win']
            probabilities: Model probabilities [0.45, 0.25, 0.30]
            odds: Bookmaker odds [2.2, 3.5, 2.8]
            
        Returns:
            Plotly figure object
        """
        # Calculate implied probabilities from odds
        implied_probs = [1/odd for odd in odds]
        
        # Calculate value (model_prob - implied_prob)
        values = [model - implied for model, implied in zip(probabilities, implied_probs)]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Probability Comparison', 'Betting Value'),
            vertical_spacing=0.15,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Add probability comparison
        fig.add_trace(
            go.Bar(
                x=outcomes,
                y=probabilities,
                name='Model Probability',
                marker_color=self.color_palette['primary'],
                text=[f'{p:.1%}' for p in probabilities],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=outcomes,
                y=implied_probs,
                name='Implied Probability',
                marker_color=self.color_palette['neutral'],
                text=[f'{p:.1%}' for p in implied_probs],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Add value chart
        colors = [self.color_palette['success'] if v > 0 else self.color_palette['error'] for v in values]
        fig.add_trace(
            go.Bar(
                x=outcomes,
                y=values,
                name='Betting Value',
                marker_color=colors,
                text=[f'{v:+.1%}' for v in values],
                textposition='auto',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(family='Inter')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Probability", tickformat='.0%', row=1, col=1)
        fig.update_yaxes(title_text="Value", tickformat='+.0%', row=2, col=1)
        
        return fig
