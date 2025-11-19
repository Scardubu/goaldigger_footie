#!/usr/bin/env python3
"""
Enhanced HTML Renderer
Provides enhanced rendering capabilities for the GoalDiggers dashboard
"""

import logging
from typing import Any, Dict, List, Optional, Union

import streamlit as st

logger = logging.getLogger(__name__)

class HTMLRenderer:
    """Enhanced HTML renderer for GoalDiggers dashboard components."""
    
    def __init__(self):
        """Initialize the HTML renderer."""
        self.theme_colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'accent': '#4facfe',
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545',
            'info': '#17a2b8',
            'bg_primary': '#f8f9fa',
            'bg_secondary': '#ffffff',
            'text_primary': '#2c3e50',
            'text_secondary': '#6c757d'
        }
    
    def render_status_badge(self, text: str, status: str = "info", tooltip: str = None):
        """Render a status badge with color coding."""
        try:
            color_map = {
                'success': self.theme_colors['success'],
                'warning': self.theme_colors['warning'],
                'error': self.theme_colors['error'],
                'info': self.theme_colors['info']
            }
            
            color = color_map.get(status, self.theme_colors['info'])
            
            badge_html = f"""
            <div style="
                background-color: {color};
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 0.375rem;
                font-size: 0.875rem;
                font-weight: 500;
                display: inline-block;
                margin: 0.25rem 0;
            " title="{tooltip or text}">
                {text}
            </div>
            """
            
            st.markdown(badge_html, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering status badge: {e}")
            # Fallback to simple text
            if status == 'success':
                st.success(text)
            elif status == 'warning':
                st.warning(text)
            elif status == 'error':
                st.error(text)
            else:
                st.info(text)
    
    def render_gradient_header(self, title: str, subtitle: str = None, icon: str = ""):
        """Render a gradient header."""
        try:
            gradient = f"linear-gradient(135deg, {self.theme_colors['primary']} 0%, {self.theme_colors['secondary']} 100%)"
            
            header_html = f"""
            <div style="
                background: {gradient};
                padding: 2rem;
                border-radius: 0.75rem;
                color: white;
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            ">
                <h1 style="
                    margin: 0;
                    font-size: 2.5rem;
                    font-weight: 700;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                ">
                    {icon} {title}
                </h1>
                {f'<p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">{subtitle}</p>' if subtitle else ''}
            </div>
            """
            
            st.markdown(header_html, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering gradient header: {e}")
            # Fallback to simple header
            st.title(f"{icon} {title}")
            if subtitle:
                st.subheader(subtitle)
    
    def render_match_card(self, home_team: str, away_team: str, match_date: str, 
                         league: str, match_id: str, on_click_callback=None):
        """Render a match card."""
        try:
            card_html = f"""
            <div style="
                background: {self.theme_colors['bg_secondary']};
                border: 1px solid #dee2e6;
                border-radius: 0.75rem;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                cursor: pointer;
            " onclick="handleMatchClick('{match_id}')">
                <div style="text-align: center;">
                    <h4 style="margin: 0 0 0.5rem 0; color: {self.theme_colors['text_primary']};">
                        {home_team} vs {away_team}
                    </h4>
                    <p style="margin: 0; color: {self.theme_colors['text_secondary']}; font-size: 0.9rem;">
                        📅 {match_date}
                    </p>
                    <p style="margin: 0.25rem 0 0 0; color: {self.theme_colors['text_secondary']}; font-size: 0.9rem;">
                        🏆 {league}
                    </p>
                </div>
            </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Handle click with button fallback
            if st.button(f"Analyze {home_team} vs {away_team}", key=f"btn_{match_id}"):
                if on_click_callback:
                    on_click_callback(match_id)
            
        except Exception as e:
            logger.error(f"Error rendering match card: {e}")
            # Fallback to simple display
            st.subheader(f"{home_team} vs {away_team}")
            st.write(f"📅 {match_date}")
            st.write(f"🏆 {league}")
            if st.button(f"Analyze", key=f"btn_fallback_{match_id}"):
                if on_click_callback:
                    on_click_callback(match_id)
    
    def render_metric_card(self, title: str, value: str, delta: str = None, 
                          help_text: str = None):
        """Render a metric card."""
        try:
            card_html = f"""
            <div style="
                background: {self.theme_colors['bg_secondary']};
                border: 1px solid #dee2e6;
                border-radius: 0.5rem;
                padding: 1rem;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            " title="{help_text or ''}">
                <h6 style="
                    margin: 0 0 0.5rem 0;
                    color: {self.theme_colors['text_secondary']};
                    font-size: 0.875rem;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                ">
                    {title}
                </h6>
                <div style="
                    font-size: 1.5rem;
                    font-weight: 700;
                    color: {self.theme_colors['text_primary']};
                    margin-bottom: 0.25rem;
                ">
                    {value}
                </div>
                {f'<div style="font-size: 0.75rem; color: {self.theme_colors["success"]};">{delta}</div>' if delta else ''}
            </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering metric card: {e}")
            # Fallback to streamlit metric
            st.metric(title, value, delta, help=help_text)
    
    def render_prediction_card(self, home_prob: float, draw_prob: float, away_prob: float,
                              home_team: str, away_team: str):
        """Render prediction results card."""
        try:
            # Determine most likely outcome
            probs = [home_prob, draw_prob, away_prob]
            outcomes = [f"{home_team} Win", "Draw", f"{away_team} Win"]
            max_idx = probs.index(max(probs))
            
            card_html = f"""
            <div style="
                background: linear-gradient(135deg, {self.theme_colors['primary']} 0%, {self.theme_colors['accent']} 100%);
                border-radius: 0.75rem;
                padding: 2rem;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            ">
                <h3 style="margin: 0 0 1rem 0; text-align: center;">🎯 AI Prediction</h3>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; text-align: center;">
                    <div style="{'border: 2px solid white; border-radius: 0.5rem; padding: 1rem;' if max_idx == 0 else 'padding: 1rem;'}">
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.5rem;">{home_team}</div>
                        <div style="font-size: 2rem; font-weight: 700;">{home_prob:.1%}</div>
                    </div>
                    
                    <div style="{'border: 2px solid white; border-radius: 0.5rem; padding: 1rem;' if max_idx == 1 else 'padding: 1rem;'}">
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.5rem;">Draw</div>
                        <div style="font-size: 2rem; font-weight: 700;">{draw_prob:.1%}</div>
                    </div>
                    
                    <div style="{'border: 2px solid white; border-radius: 0.5rem; padding: 1rem;' if max_idx == 2 else 'padding: 1rem;'}">
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 0.5rem;">{away_team}</div>
                        <div style="font-size: 2rem; font-weight: 700;">{away_prob:.1%}</div>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 0.5rem;">
                    <strong>Most Likely: {outcomes[max_idx]}</strong>
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-top: 0.25rem;">
                        Confidence: {max(probs):.1%}
                    </div>
                </div>
            </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering prediction card: {e}")
            # Fallback to simple columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{home_team} Win", f"{home_prob:.1%}")
            with col2:
                st.metric("Draw", f"{draw_prob:.1%}")
            with col3:
                st.metric(f"{away_team} Win", f"{away_prob:.1%}")
    
    def inject_custom_css(self):
        """Inject custom CSS for enhanced styling."""
        try:
            css = f"""
            <style>
            :root {{
                --primary-color: {self.theme_colors['primary']};
                --secondary-color: {self.theme_colors['secondary']};
                --accent-color: {self.theme_colors['accent']};
                --success-color: {self.theme_colors['success']};
                --warning-color: {self.theme_colors['warning']};
                --error-color: {self.theme_colors['error']};
                --info-color: {self.theme_colors['info']};
                --bg-primary: {self.theme_colors['bg_primary']};
                --bg-secondary: {self.theme_colors['bg_secondary']};
                --text-primary: {self.theme_colors['text_primary']};
                --text-secondary: {self.theme_colors['text_secondary']};
            }}
            
            .stApp {{
                background-color: var(--bg-primary);
            }}
            
            .main .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }}
            
            .enhanced-card {{
                background: var(--bg-secondary);
                border-radius: 0.75rem;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
            }}
            
            .enhanced-card:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 16px rgba(0,0,0,0.15);
            }}
            
            .metric-large {{
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--primary-color);
                margin: 0;
            }}
            
            .prediction-highlight {{
                background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
                color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                text-align: center;
                margin: 1rem 0;
            }}
            
            .status-online {{
                color: var(--success-color);
                font-weight: 600;
            }}
            
            .status-offline {{
                color: var(--error-color);
                font-weight: 600;
            }}
            
            /* Hide Streamlit branding */
            #MainMenu {{visibility: hidden;}}
            footer {{visibility: hidden;}}
            
            /* Custom button styling */
            .stButton > button {{
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                border: none;
                border-radius: 0.5rem;
                padding: 0.5rem 1rem;
                font-weight: 600;
                transition: all 0.3s ease;
            }}
            
            .stButton > button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            </style>
            """
            
            st.markdown(css, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error injecting custom CSS: {e}")
