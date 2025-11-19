#!/usr/bin/env python3
"""
Visual Consistency System
Ensures consistent visual design across all dashboard variants
"""

import streamlit as st
from typing import Dict, Any, Optional, List
import json

class VisualConsistencySystem:
    """System for maintaining visual consistency across dashboard variants."""
    
    def __init__(self):
        """Initialize visual consistency system."""
        self.design_tokens = self._load_design_tokens()
        self.component_styles = {}
        
    def _load_design_tokens(self) -> Dict[str, Any]:
        """Load design tokens for consistent styling."""
        return {
            "colors": {
                "primary": "#667eea",
                "secondary": "#764ba2", 
                "success": "#4CAF50",
                "warning": "#FF9800",
                "error": "#f44336",
                "info": "#2196F3",
                "background": "#f8f9fa",
                "surface": "#ffffff",
                "text_primary": "#212529",
                "text_secondary": "#6c757d",
                "border": "#dee2e6"
            },
            "spacing": {
                "xs": "4px",
                "sm": "8px", 
                "md": "16px",
                "lg": "24px",
                "xl": "32px",
                "xxl": "48px"
            },
            "typography": {
                "font_family": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "font_sizes": {
                    "xs": "0.75rem",
                    "sm": "0.875rem",
                    "base": "1rem",
                    "lg": "1.125rem",
                    "xl": "1.25rem",
                    "2xl": "1.5rem",
                    "3xl": "1.875rem",
                    "4xl": "2.25rem"
                },
                "font_weights": {
                    "normal": "400",
                    "medium": "Server Error",
                    "semibold": "600",
                    "bold": "700"
                }
            },
            "shadows": {
                "sm": "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
                "base": "0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)",
                "md": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
                "lg": "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
                "xl": "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)"
            },
            "border_radius": {
                "sm": "4px",
                "base": "8px",
                "md": "12px",
                "lg": "16px",
                "xl": "24px",
                "full": "9999px"
            }
        }
    
    def inject_global_styles(self):
        """Inject global styles for visual consistency."""
        colors = self.design_tokens["colors"]
        typography = self.design_tokens["typography"]
        shadows = self.design_tokens["shadows"]
        spacing = self.design_tokens["spacing"]
        border_radius = self.design_tokens["border_radius"]
        
        css = f"""
        <style>
        /* Global Visual Consistency Styles */
        
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Root variables */
        :root {{
            --primary-color: {colors["primary"]};
            --secondary-color: {colors["secondary"]};
            --success-color: {colors["success"]};
            --warning-color: {colors["warning"]};
            --error-color: {colors["error"]};
            --info-color: {colors["info"]};
            --background-color: {colors["background"]};
            --surface-color: {colors["surface"]};
            --text-primary: {colors["text_primary"]};
            --text-secondary: {colors["text_secondary"]};
            --border-color: {colors["border"]};
            
            --font-family: {typography["font_family"]};
            --shadow-sm: {shadows["sm"]};
            --shadow-base: {shadows["base"]};
            --shadow-md: {shadows["md"]};
            --shadow-lg: {shadows["lg"]};
            --shadow-xl: {shadows["xl"]};
            
            --spacing-xs: {spacing["xs"]};
            --spacing-sm: {spacing["sm"]};
            --spacing-md: {spacing["md"]};
            --spacing-lg: {spacing["lg"]};
            --spacing-xl: {spacing["xl"]};
            --spacing-xxl: {spacing["xxl"]};
            
            --radius-sm: {border_radius["sm"]};
            --radius-base: {border_radius["base"]};
            --radius-md: {border_radius["md"]};
            --radius-lg: {border_radius["lg"]};
            --radius-xl: {border_radius["xl"]};
        }}
        
        /* Global font family */
        .stApp, .stApp * {{
            font-family: var(--font-family) !important;
        }}
        
        /* Consistent button styling */
        .stButton > button {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
            color: white !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            padding: var(--spacing-sm) var(--spacing-lg) !important;
            font-weight: 500 !important;
            box-shadow: var(--shadow-base) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-lg) !important;
        }}
        
        /* Consistent metric styling */
        .stMetric {{
            background: var(--surface-color) !important;
            border-radius: var(--radius-md) !important;
            padding: var(--spacing-lg) !important;
            box-shadow: var(--shadow-base) !important;
            border: 1px solid var(--border-color) !important;
        }}
        
        .stMetric:hover {{
            box-shadow: var(--shadow-md) !important;
            transform: translateY(-1px) !important;
        }}
        
        /* Consistent selectbox styling */
        .stSelectbox > div > div {{
            border-radius: var(--radius-base) !important;
            border: 2px solid var(--border-color) !important;
            box-shadow: var(--shadow-sm) !important;
        }}
        
        .stSelectbox > div > div:focus-within {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }}
        
        /* Consistent alert styling */
        .stAlert {{
            border-radius: var(--radius-md) !important;
            border: none !important;
            box-shadow: var(--shadow-base) !important;
        }}
        
        .stAlert[data-baseweb="notification"] {{
            background: var(--surface-color) !important;
        }}
        
        /* Success alerts */
        .stAlert[data-baseweb="notification"][kind="success"] {{
            background: linear-gradient(135deg, var(--success-color), #45a049) !important;
            color: white !important;
        }}
        
        /* Error alerts */
        .stAlert[data-baseweb="notification"][kind="error"] {{
            background: linear-gradient(135deg, var(--error-color), #d32f2f) !important;
            color: white !important;
        }}
        
        /* Warning alerts */
        .stAlert[data-baseweb="notification"][kind="warning"] {{
            background: linear-gradient(135deg, var(--warning-color), #f57c00) !important;
            color: white !important;
        }}
        
        /* Info alerts */
        .stAlert[data-baseweb="notification"][kind="info"] {{
            background: linear-gradient(135deg, var(--info-color), #1976d2) !important;
            color: white !important;
        }}
        
        /* Consistent sidebar styling */
        .css-1d391kg {{
            background: var(--surface-color) !important;
            border-right: 1px solid var(--border-color) !important;
        }}
        
        /* Consistent main content area */
        .main .block-container {{
            padding: var(--spacing-lg) !important;
            background: var(--background-color) !important;
        }}
        
        /* Consistent header styling */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }}
        
        /* Consistent text styling */
        p, span, div {{
            color: var(--text-primary) !important;
        }}
        
        /* Consistent card styling */
        .element-container {{
            background: var(--surface-color) !important;
            border-radius: var(--radius-base) !important;
            margin-bottom: var(--spacing-md) !important;
        }}
        
        /* Consistent input styling */
        .stTextInput > div > div > input {{
            border-radius: var(--radius-base) !important;
            border: 2px solid var(--border-color) !important;
            padding: var(--spacing-sm) var(--spacing-md) !important;
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }}
        
        /* Consistent progress bar styling */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color)) !important;
            border-radius: var(--radius-full) !important;
        }}
        
        /* Consistent table styling */
        .stDataFrame {{
            border-radius: var(--radius-md) !important;
            overflow: hidden !important;
            box-shadow: var(--shadow-base) !important;
        }}
        
        /* Consistent expander styling */
        .streamlit-expanderHeader {{
            background: var(--surface-color) !important;
            border-radius: var(--radius-base) !important;
            border: 1px solid var(--border-color) !important;
        }}
        
        /* Consistent tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: var(--spacing-sm) !important;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: var(--surface-color) !important;
            border-radius: var(--radius-base) !important;
            border: 1px solid var(--border-color) !important;
            padding: var(--spacing-sm) var(--spacing-md) !important;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color)) !important;
            color: white !important;
            border-color: var(--primary-color) !important;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: var(--spacing-md) !important;
            }}
            
            .stButton > button {{
                width: 100% !important;
                margin-bottom: var(--spacing-sm) !important;
            }}
        }}
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {{
            :root {{
                --background-color: #1a1a1a;
                --surface-color: #2d2d2d;
                --text-primary: #ffffff;
                --text-secondary: #b3b3b3;
                --border-color: #404040;
            }}
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    def create_consistent_card(self, title: str, content: str, 
                             card_type: str = "default") -> str:
        """Create a consistently styled card."""
        type_styles = {
            "default": "var(--surface-color)",
            "primary": "linear-gradient(135deg, var(--primary-color), var(--secondary-color))",
            "success": "linear-gradient(135deg, var(--success-color), #45a049)",
            "warning": "linear-gradient(135deg, var(--warning-color), #f57c00)",
            "error": "linear-gradient(135deg, var(--error-color), #d32f2f)"
        }
        
        text_color = "white" if card_type != "default" else "var(--text-primary)"
        background = type_styles.get(card_type, type_styles["default"])
        
        return f"""
        <div style="
            background: {background};
            color: {text_color};
            padding: var(--spacing-lg);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-base);
            margin: var(--spacing-md) 0;
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='var(--shadow-lg)'"
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='var(--shadow-base)'">
            <h3 style="margin: 0 0 var(--spacing-sm) 0; color: {text_color};">{title}</h3>
            <p style="margin: 0; color: {text_color};">{content}</p>
        </div>
        """
    
    def create_status_badge(self, text: str, status: str = "default") -> str:
        """Create a consistently styled status badge."""
        status_colors = {
            "default": "var(--text-secondary)",
            "success": "var(--success-color)",
            "warning": "var(--warning-color)",
            "error": "var(--error-color)",
            "info": "var(--info-color)",
            "primary": "var(--primary-color)"
        }
        
        color = status_colors.get(status, status_colors["default"])
        
        return f"""
        <span style="
            background: {color};
            color: white;
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: var(--radius-full);
            font-size: 0.875rem;
            font-weight: 500;
            display: inline-block;
            margin: var(--spacing-xs);
        ">
            {text}
        </span>
        """
    
    def get_design_token(self, category: str, key: str) -> str:
        """Get a specific design token value."""
        return self.design_tokens.get(category, {}).get(key, "")
    
    def apply_component_theme(self, component_type: str) -> Dict[str, str]:
        """Get theme configuration for a specific component type."""
        themes = {
            "metric": {
                "background": self.design_tokens["colors"]["surface"],
                "border_radius": self.design_tokens["border_radius"]["md"],
                "shadow": self.design_tokens["shadows"]["base"],
                "padding": self.design_tokens["spacing"]["lg"]
            },
            "button": {
                "background": f"linear-gradient(135deg, {self.design_tokens['colors']['primary']}, {self.design_tokens['colors']['secondary']})",
                "border_radius": self.design_tokens["border_radius"]["md"],
                "padding": f"{self.design_tokens['spacing']['sm']} {self.design_tokens['spacing']['lg']}"
            },
            "card": {
                "background": self.design_tokens["colors"]["surface"],
                "border_radius": self.design_tokens["border_radius"]["md"],
                "shadow": self.design_tokens["shadows"]["base"],
                "border": f"1px solid {self.design_tokens['colors']['border']}"
            }
        }
        
        return themes.get(component_type, {})

# Global instance
_visual_consistency = None

def get_visual_consistency_system():
    """Get the global VisualConsistencySystem instance."""
    global _visual_consistency
    if _visual_consistency is None:
        _visual_consistency = VisualConsistencySystem()
    return _visual_consistency
