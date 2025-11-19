#!/usr/bin/env python3
"""
HTML Display Fix for GoalDiggers Platform
Ensures HTML content is properly rendered instead of showing raw HTML code.
"""

import logging
from typing import Any, Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)

class HTMLDisplayFixer:
    """Fix HTML display issues by ensuring proper rendering."""
    
    def __init__(self):
        """Initialize the HTML display fixer."""
        self.applied_fixes = set()
        self.load_global_css_fix()
    
    def load_global_css_fix(self):
        """Apply global CSS to fix HTML rendering issues."""
        if 'html_display_fix' in self.applied_fixes:
            return
        
        fix_css = """
        <style>
        /* Global fix for raw HTML display */
        .stMarkdown pre, .stMarkdown code {
            display: none !important;
        }
        
        /* Ensure proper HTML content rendering */
        .stMarkdown [data-testid="stMarkdownContainer"] > div {
            display: block !important;
            width: 100% !important;
        }
        
        /* Force HTML content to render properly */
        .stMarkdown .gd-layout-container,
        .stMarkdown .gd-match-card,
        .stMarkdown .gd-hero-header,
        .stMarkdown .gd-team-vs-layout,
        .stMarkdown .gd-prediction-pills,
        .stMarkdown .gd-stats-grid {
            display: block !important;
            visibility: visible !important;
        }
        </style>
        """
        
        st.markdown(fix_css, unsafe_allow_html=True)
        self.applied_fixes.add('html_display_fix')
        logger.info("✅ HTML display fix applied")
    
    def render_safe_html(self, html_content: str, fallback_text: Optional[str] = None) -> bool:
        """Safely render HTML content with fallback."""
        try:
            if not html_content:
                if fallback_text:
                    st.markdown(fallback_text)
                return True
            
            # Apply the HTML directly with unsafe_allow_html=True
            st.markdown(html_content, unsafe_allow_html=True)
            return True
            
        except Exception as e:
            logger.error(f"Failed to render HTML content: {e}")
            if fallback_text:
                st.markdown(fallback_text)
            else:
                st.error("Content could not be displayed")
            return False
    
    def fix_match_card_display(self, match_data: Dict[str, Any]) -> None:
        """Fix match card display by using native Streamlit components."""
        try:
            home_team = match_data.get('home_team', 'Unknown')
            away_team = match_data.get('away_team', 'Unknown')
            
            # Use Streamlit's native components instead of raw HTML
            with st.container():
                # Header
                st.markdown(f"### ⚽ {home_team} vs {away_team}")
                
                # Team layout using columns
                col1, col2, col3 = st.columns([2, 1, 2])
                
                with col1:
                    st.markdown(f"**{home_team}**")
                    # Form using emoji badges
                    form = match_data.get('home_form', ['W', 'W', 'D', 'L', 'W'])
                    form_display = ""
                    for result in form[-5:]:
                        if result.upper() == 'W':
                            form_display += "🟢 "
                        elif result.upper() == 'D':
                            form_display += "🟡 "
                        else:
                            form_display += "🔴 "
                    st.markdown(f"Form: {form_display}")
                
                with col2:
                    st.markdown("**VS**", help="Match prediction")
                
                with col3:
                    st.markdown(f"**{away_team}**")
                    # Away form
                    form = match_data.get('away_form', ['L', 'W', 'W', 'D', 'L'])
                    form_display = ""
                    for result in form[-5:]:
                        if result.upper() == 'W':
                            form_display += "🟢 "
                        elif result.upper() == 'D':
                            form_display += "🟡 "
                        else:
                            form_display += "🔴 "
                    st.markdown(f"Form: {form_display}")
                
                # Predictions using metrics
                col1, col2, col3 = st.columns(3)
                home_prob = match_data.get('home_win_probability', 0.45) * 100
                draw_prob = match_data.get('draw_probability', 0.25) * 100
                away_prob = match_data.get('away_win_probability', 0.30) * 100
                
                with col1:
                    st.metric("🏠 Home Win", f"{home_prob:.1f}%", help="Home team win probability")
                
                with col2:
                    st.metric("🤝 Draw", f"{draw_prob:.1f}%", help="Draw probability")
                
                with col3:
                    st.metric("✈️ Away Win", f"{away_prob:.1f}%", help="Away team win probability")
                
                # Additional stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    confidence = match_data.get('confidence', 0.82)
                    st.metric("🎯 Confidence", f"{confidence*100:.0f}%")
                
                with col2:
                    expected_value = match_data.get('expected_value', 0.12)
                    st.metric("💰 Expected Value", f"{expected_value*100:+.1f}%")
                
                with col3:
                    risk_level = match_data.get('risk_level', 'Medium')
                    st.metric("⚠️ Risk Level", risk_level)
                
                # Value bet indicator
                if match_data.get('is_value_bet', False):
                    st.success("💰 **Value Bet Opportunity!** This match has positive expected value.")
        
        except Exception as e:
            logger.error(f"Error rendering match card: {e}")
            st.error(f"Error displaying match: {home_team} vs {away_team}")

# Global instance
html_fixer = HTMLDisplayFixer()

def render_match_card_fixed(match_data: Dict[str, Any]) -> None:
    """Render a match card using fixed display method."""
    html_fixer.fix_match_card_display(match_data)

def render_html_safely(html_content: str, fallback_text: Optional[str] = None) -> bool:
    """Safely render HTML content with fallback."""
    return html_fixer.render_safe_html(html_content, fallback_text)
