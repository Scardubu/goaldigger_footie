#!/usr/bin/env python3
"""
Unified GoalDiggers Design System

Standardizes CSS styling, branding, and visual consistency across all dashboard variants:
- optimized_premium_dashboard.py
- interactive_cross_league_dashboard.py  
- premium_ui_dashboard.py

Features:
- Consistent GoalDiggers branding and color schemes
- Unified component styling (buttons, cards, metrics, progress bars)
- Responsive design validation
- Professional visual hierarchy
- Performance-optimized CSS delivery
"""

import logging
from typing import Any

# New: centralized theme tokens
try:
    from .theme_tokens import (
        BRAND_COLORS,
        COMPONENT_TOKENS,
        SPACING_TOKENS,
        TYPOGRAPHY_TOKENS,
        build_css_variable_block,
    )
except Exception:  # Fallback if file missing; preserve previous behavior
    BRAND_COLORS = None
    TYPOGRAPHY_TOKENS = None
    SPACING_TOKENS = None
    COMPONENT_TOKENS = None
    def build_css_variable_block(theme: str = 'light') -> str:  # type: ignore
        return ""  # minimal fallback

import streamlit as st

logger = logging.getLogger(__name__)

class UnifiedDesignSystem:
    """Unified design system for consistent GoalDiggers branding."""
    
    def __init__(self):
        """Initialize unified design system."""
        # If external token module available, use it; else fall back to legacy in-file definitions.
        self.brand_colors = BRAND_COLORS or self._define_brand_colors()
        self.typography = TYPOGRAPHY_TOKENS or self._define_typography()
        self.spacing = SPACING_TOKENS or self._define_spacing()
        self.components = COMPONENT_TOKENS or self._define_component_styles()
        # Backward compatibility alias (older code referenced self.colors)
        self.colors = self.brand_colors

        logger.info("🎨 Unified GoalDiggers Design System initialized")
    
    def _define_brand_colors(self) -> dict[str, str]:
        """Define consistent GoalDiggers brand colors."""
        return {
            # Primary Brand Colors
            'primary': '#1f4e79',           # GoalDiggers Blue
            'primary_light': '#2a5298',     # Lighter Blue
            'primary_dark': '#1a3d5f',      # Darker Blue
            
            # Secondary Colors
            'secondary': '#28a745',         # Success Green
            'accent': '#fd7e14',            # Warning Orange
            'danger': '#dc3545',            # Error Red
            'info': '#17a2b8',              # Info Cyan
            
            # Neutral Colors
            'white': '#ffffff',
            'light_gray': '#f8f9fa',
            'medium_gray': '#6c757d',
            'dark_gray': '#343a40',
            'black': '#000000',
            
            # Gradient Colors
            'gradient_primary': 'linear-gradient(135deg, #1f4e79 0%, #2a5298 100%)',
            'gradient_success': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
            'gradient_warning': 'linear-gradient(135deg, #fd7e14 0%, #ffc107 100%)',
            'gradient_cross_league': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'gradient_scenario': 'linear-gradient(45deg, #ff6b6b, #ee5a24)',
            
            # Achievement Colors
            'achievement_gold': '#ffd700',
            'achievement_silver': '#c0c0c0',
            'achievement_bronze': '#cd7f32',
            
            # Transparency
            'shadow': 'rgba(0, 0, 0, 0.1)',
            'overlay': 'rgba(31, 78, 121, 0.9)'
        }
    
    def _define_typography(self) -> dict[str, str]:
        """Define consistent typography system."""
        return {
            'font_family': '"Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif',
            'font_family_mono': '"SF Mono", "Monaco", "Inconsolata", "Roboto Mono", monospace',
            
            # Font Sizes
            'font_size_xs': '0.75rem',      # 12px
            'font_size_sm': '0.875rem',     # 14px
            'font_size_base': '1rem',       # 16px
            'font_size_lg': '1.125rem',     # 18px
            'font_size_xl': '1.25rem',      # 20px
            'font_size_2xl': '1.5rem',      # 24px
            'font_size_3xl': '1.875rem',    # 30px
            'font_size_4xl': '2.25rem',     # 36px
            
            # Font Weights
            'font_weight_light': '300',
            'font_weight_normal': '400',
            'font_weight_medium': '500',
            'font_weight_semibold': '600',
            'font_weight_bold': '700',
            'font_weight_extrabold': '800',
            
            # Line Heights
            'line_height_tight': '1.25',
            'line_height_normal': '1.5',
            'line_height_relaxed': '1.75'
        }
    
    def _define_spacing(self) -> dict[str, str]:
        """Define consistent spacing system."""
        return {
            'xs': '0.25rem',    # 4px
            'sm': '0.5rem',     # 8px
            'md': '1rem',       # 16px
            'lg': '1.5rem',     # 24px
            'xl': '2rem',       # 32px
            'xxl': '3rem',      # 48px
            'xxxl': '4rem'      # 64px
        }
    
    def _define_component_styles(self) -> dict[str, str]:
        """Define consistent component styling."""
        return {
            'border_radius_sm': '0.375rem',     # 6px
            'border_radius_md': '0.5rem',       # 8px
            'border_radius_lg': '0.75rem',      # 12px
            'border_radius_xl': '1rem',         # 16px
            'border_radius_full': '9999px',     # Full rounded
            
            'shadow_sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
            'shadow_md': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
            'shadow_lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
            'shadow_xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
            
            'transition_fast': '0.15s ease-in-out',
            'transition_normal': '0.3s ease-in-out',
            'transition_slow': '0.5s ease-in-out'
        }
    
    def inject_unified_css(self, dashboard_type: str = 'premium') -> None:
        """
        Inject unified CSS for consistent styling across all dashboards.
        Optimized for ultra-fast startup with caching and minimal I/O.

        Args:
            dashboard_type: Type of dashboard (premium, cross_league, integrated)
        """
        # Use cached CSS generation for ultra-fast startup
        css = _get_cached_unified_css(dashboard_type, self.brand_colors, self.typography, self.spacing, self.components)
        st.markdown(css, unsafe_allow_html=True)
        logger.info(f"✅ Unified design system CSS injected for {dashboard_type} dashboard")
    
    def _generate_unified_css(self, dashboard_type: str) -> str:
        """Generate unified CSS based on dashboard type."""
        colors = self.brand_colors
        typography = self.typography
        
        # Build css variables block (from external tokens if available)
        css_var_block = build_css_variable_block('light')

        # Helper lookups with safe defaults
        grad_primary = colors.get('gradient_primary', 'linear-gradient(135deg,#1f4e79,#2a5298)')
        grad_success = colors.get('gradient_success', 'linear-gradient(135deg,#28a745,#20c997)')
        grad_warning = colors.get('gradient_warning', 'linear-gradient(135deg,#fd7e14,#ffc107)')
        grad_cross = colors.get('gradient_cross_league', 'linear-gradient(135deg,#667eea,#764ba2)')
        grad_scenario = colors.get('gradient_scenario', 'linear-gradient(45deg,#ff6b6b,#ee5a24)')

        css_parts = [
            "<style>",
            "/* GoalDiggers Unified Design System - Ultra-Fast Startup */",
            "/* Token variables */",
            css_var_block,
            "/* Additional runtime-generated design tokens */",
            ":root {",
            "  --gd-elev-0: none;",
            "  --gd-elev-1: 0 1px 2px 0 rgba(0,0,0,0.06),0 1px 3px 1px rgba(0,0,0,0.04);",
            "  --gd-elev-2: 0 2px 4px -1px rgba(0,0,0,0.08),0 4px 6px -1px rgba(0,0,0,0.05);",
            "  --gd-elev-3: 0 4px 10px -2px rgba(0,0,0,0.10),0 2px 4px -1px rgba(0,0,0,0.04);",
            "  --gd-elev-4: 0 8px 18px -4px rgba(0,0,0,0.18),0 4px 12px -2px rgba(0,0,0,0.06);",
            "  --gd-elev-glass: 0 8px 32px -6px rgba(31,78,121,0.28),0 2px 12px rgba(0,0,0,0.08);",
            "  --gd-transition: 0.28s cubic-bezier(.4,0,.2,1);",
            "  --gd-transition-fast: 0.16s cubic-bezier(.4,0,.2,1);",
            "  --gd-focus-ring: 0 0 0 3px rgba(118,75,162,0.4);",
            "  --gd-glass-bg-1: rgba(255,255,255,0.82);",
            "  --gd-glass-bg-2: rgba(255,255,255,0.68);",
            "  --gd-glass-border: rgba(255,255,255,0.35);",
            "  --gd-glass-highlight: linear-gradient(135deg, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.15) 60%);",
            f"  --gd-state-success: {colors.get('secondary', colors.get('accent', '#28a745'))};",
            f"  --gd-state-danger: {colors.get('danger', '#dc3545')};",
            f"  --gd-state-warning: {colors.get('accent', '#fd7e14')};",
            f"  --gd-state-info: {colors.get('info', '#17a2b8')};",
            "}",
            "body[data-theme='dark'] { --gd-white:#11161d; --gd-light-gray:#1c232c; --gd-medium-gray:#586271; --gd-dark-gray:#e0e4e9; --gd-primary:#2a8bff; --gd-primary-dark:#1469c7; --gd-accent:#ff914d; color-scheme:dark; }",
            "body[data-theme='dark'] .gd-card, body[data-theme='dark'] .goaldiggers-card { background:linear-gradient(135deg,#161d26 0%,#1f2732 100%); border:1px solid rgba(255,255,255,0.06); box-shadow:0 6px 24px -4px rgba(0,0,0,0.6); }",
            "body[data-theme='dark'] h1, body[data-theme='dark'] h2, body[data-theme='dark'] h3 { color: var(--gd-dark-gray); }",
            "body[data-theme='dark'] .stButton > button { background: var(--gd-primary); }",
            "body[data-theme='dark'] .gd-featured-banner { background: linear-gradient(135deg, rgba(30,40,55,0.85) 0%, rgba(30,40,55,0.78) 100%); }",
            ".main .block-container { padding-top: var(--gd-spacing-lg); padding-bottom: var(--gd-spacing-lg); max-width:1200px; font-family: var(--gd-font-family); position: relative; z-index: 1; }",
            "h1, h2, h3, h4, h5, h6 { font-family: var(--gd-font-family); color: var(--gd-dark-gray); font-weight: var(--gd-font-weight-semibold); line-height: var(--gd-line-height-normal); margin-bottom: var(--gd-spacing-md); }",
            f"h1 {{ font-size: {typography['font_size_3xl']}; }}",
            f"h2 {{ font-size: {typography['font_size_2xl']}; }}",
            f"h3 {{ font-size: {typography['font_size_xl']}; }}",
            f"h4 {{ font-size: {typography['font_size_lg']}; }}",
            ".gd-glass { background: rgba(255,255,255,0.15); backdrop-filter: blur(14px) saturate(180%); -webkit-backdrop-filter: blur(14px) saturate(180%); border:1px solid rgba(255,255,255,0.25); box-shadow:0 8px 32px rgba(0,0,0,0.15); }",
            f".gd-chip {{ display:inline-flex; align-items:center; gap:6px; padding:6px 14px; border-radius:999px; font-size:{typography['font_size_sm']}; font-weight:600; background: var(--gd-light-gray); color: var(--gd-dark-gray); }}",
            f".gd-chip-primary {{ background:{grad_primary}; color:#fff; }}",
            f".gd-chip-success {{ background:{grad_success}; color:#fff; }}",
            f".gd-gradient-primary {{ background:{grad_primary}; color:#fff; }}",
            f".gd-gradient-success {{ background:{grad_success}; color:#fff; }}",
            f".gd-gradient-warning {{ background:{grad_warning}; color:#fff; }}",
            f".gd-gradient-cross {{ background:{grad_cross}; color:#fff; }}",
            f".gd-gradient-scenario {{ background:{grad_scenario}; color:#fff; }}",
            ".gd-match-card { background: linear-gradient(135deg, rgba(255,255,255,0.94) 0%, rgba(245,248,255,0.86) 100%); border:1px solid rgba(31,78,121,0.12); border-radius:22px; padding:20px 22px 26px; box-shadow:0 10px 34px -8px rgba(31,78,121,0.28),0 2px 10px rgba(0,0,0,0.06); position:relative; overflow:hidden; }",
            ".gd-match-card:before { content:''; position:absolute; inset:0; background:radial-gradient(circle at 18% 12%,rgba(118,75,162,0.25),transparent 60%),radial-gradient(circle at 82% 86%,rgba(31,78,121,0.25),transparent 65%); opacity:.65; pointer-events:none; }",
            ".gd-match-card:hover { transform: translateY(-4px); box-shadow:0 18px 48px -8px rgba(31,78,121,0.36),0 4px 16px rgba(0,0,0,0.10); }",
            ".gd-inline-metrics { display:flex; gap: var(--gd-spacing-md); flex-wrap:wrap; }",
            f".gd-section-title {{ font-size:{typography['font_size_2xl']}; font-weight:700; display:flex; align-items:center; gap:10px; background:linear-gradient(90deg,#1f4e79,#2a5298); -webkit-background-clip:text; color:transparent; }}",
            ".gd-divider { height:1px; background:linear-gradient(90deg, rgba(0,0,0,0) 0%, rgba(31,78,121,0.35) 50%, rgba(0,0,0,0) 100%); margin: var(--gd-spacing-lg) 0; }",
            f".cta-primary {{ background:{self.brand_colors.get('accent', '#fd7e14')}; color:{self.brand_colors.get('white', '#ffffff')}; padding:12px 18px; border-radius:12px; font-weight:700; box-shadow:0 6px 18px rgba(0,0,0,0.12); border:none; cursor:pointer; }}",
            f".goaldiggers-header {{ background:{grad_primary}; color:var(--gd-white); text-align:center; padding: var(--gd-spacing-xl) var(--gd-spacing-lg); margin: calc(-1 * var(--gd-spacing-lg)) calc(-1 * var(--gd-spacing-lg)) var(--gd-spacing-xl) calc(-1 * var(--gd-spacing-lg)); border-radius:0 0 var(--gd-border-radius-lg) var(--gd-border-radius-lg); box-shadow: var(--gd-shadow-lg); }}",
            ".goaldiggers-card { background: linear-gradient(135deg, rgba(255,255,255,0.92) 0%, rgba(245,245,255,0.82) 100%); border-radius: var(--gd-border-radius-lg); padding: var(--gd-spacing-lg); box-shadow: var(--gd-elev-2); border:1px solid rgba(31,78,121,0.12); margin-bottom: var(--gd-spacing-md); transition: box-shadow var(--gd-transition-fast), transform var(--gd-transition-fast); backdrop-filter: blur(12px) saturate(170%); }",
            ".goaldiggers-card:hover { box-shadow: var(--gd-elev-4); transform: translateY(-3px); }",
            f".stButton > button[kind='primary'] {{ background:{grad_primary}; }}",
            ".stProgress > div > div > div { background: linear-gradient(135deg,#1f4e79,#2a5298); border-radius: var(--gd-border-radius); }",
            "@media (max-width:768px){ .goaldiggers-card { padding: var(--gd-spacing-md);} }",
        ]

        css = "\n".join(css_parts)

        # Add dashboard-specific styles
        if dashboard_type == 'cross_league':
            css += self._get_cross_league_styles()
        elif dashboard_type == 'premium':
            css += self._get_premium_styles()
        elif dashboard_type == 'integrated':
            css += self._get_integrated_styles()

        css += "</style>"
        return css

    def _get_premium_styles(self) -> str:
        """Get premium dashboard specific styles with Material Design and glassmorphism."""
        # Use raw triple-quoted string to avoid any unintended interpolation that could
        # trigger NameError for CSS tokens like 'transform' inside braces.
        return r"""
        /* Premium Dashboard Specific Styles: Material + Glassmorphism */
    /* Enhanced sticky main header with safe layering */
    .main-header { background: linear-gradient(135deg, var(--gd-primary) 0%, var(--gd-secondary) 100%); padding: 2rem 1rem; border-radius: 0 0 20px 20px; margin-bottom: 2rem; color: #fff; text-align: center; box-shadow:0 4px 20px rgba(0,0,0,0.15); position:sticky; top:0; z-index: 999; backdrop-filter: blur(6px) saturate(160%); -webkit-backdrop-filter: blur(6px) saturate(160%); }
    /* Reserve top spacing so first section not hidden beneath sticky header */
    .main .block-container:first-of-type { padding-top: 0.5rem; }
    /* Fix for theme toggle positioning - ensure it doesn't appear above header */
    .stSelectbox[data-testid="stSelectbox"] > div, .stToggle[data-testid="stToggle"] > div { z-index: 1 !important; position: relative; }
    /* Ensure main content appears properly below theme controls */
    .main .block-container { position: relative; z-index: 1; }
    /* Fix theme toggle appearing above main header */
    div[data-testid="column"] { position: relative; z-index: 2; }
    section[data-testid="stSidebar"] { z-index: 100; }
        .hero-section { position:relative; background: linear-gradient(135deg, rgba(102,126,234,0.85) 0%, rgba(118,75,162,0.85) 100%); color:#fff; padding:3.2rem clamp(1rem,3vw,2.5rem); border-radius:28px; margin:2.5rem 0 2rem; text-align:center; box-shadow:0 10px 40px -4px rgba(0,0,0,0.35),0 4px 18px rgba(0,0,0,0.35); backdrop-filter:blur(22px) saturate(160%); -webkit-backdrop-filter:blur(22px) saturate(160%); overflow:hidden; }
        .hero-section:before, .hero-section:after { content:""; position:absolute; width:420px; height:420px; top:-140px; left:-140px; background:radial-gradient(circle at center, rgba(255,255,255,0.28), transparent 70%); animation:hero-pulse 9s ease-in-out infinite; pointer-events:none; }
        .hero-section:after { top:auto; left:auto; bottom:-160px; right:-160px; animation-delay:4.5s; }
        @keyframes hero-pulse { 0%,100% { transform:scale(1); opacity:.65;} 50% { transform:scale(1.25); opacity:.35;} }
        .hero-section h1 { font-size: clamp(2.4rem, 5vw, 3.2rem); margin:0 0 1rem; letter-spacing:.5px; }
        .hero-section h3 { font-weight:400; opacity:.92; margin:.25rem 0 1.25rem; }
        .hero-pill-group { display:flex; flex-wrap:wrap; justify-content:center; gap:10px; margin-top:1.75rem; }
        .hero-pill { background: rgba(255,255,255,0.16); padding:.55rem 1.05rem; border-radius:40px; display:inline-flex; gap:8px; align-items:center; font-weight:500; font-size:.82rem; letter-spacing:.5px; backdrop-filter: blur(10px) saturate(160%); border:1px solid rgba(255,255,255,0.25); }
        .hero-cta-row { margin-top:2.2rem; display:flex; justify-content:center; gap:1rem; flex-wrap:wrap; }
        .gd-btn-primary { background: linear-gradient(135deg,#1f4e79,#2a5298); color:#fff; padding:.9rem 1.4rem; font-weight:600; border-radius:14px; border:1px solid rgba(255,255,255,0.25); box-shadow:0 4px 18px -2px rgba(0,0,0,0.4); transition: all .25s ease; text-decoration:none; display:inline-flex; align-items:center; gap:8px; }
        .gd-btn-primary:hover { transform:translateY(-3px); box-shadow:0 8px 28px -2px rgba(0,0,0,0.55); }
        .gd-btn-outline { background: rgba(255,255,255,0.08); color:#fff; padding:.9rem 1.35rem; font-weight:500; border-radius:14px; border:1px solid rgba(255,255,255,0.35); backdrop-filter: blur(10px) saturate(150%); transition: all .25s ease; text-decoration:none; display:inline-flex; align-items:center; gap:8px; }
        .gd-btn-outline:hover { background: rgba(255,255,255,0.16); }
        .gd-auto-grid { display:grid; gap: clamp(.9rem, 1.8vw, 1.4rem); }
        @media (min-width: 700px){ .gd-auto-grid.cols-3 { grid-template-columns: repeat(3,minmax(0,1fr)); } }
        @media (min-width: 1100px){ .gd-auto-grid.cols-4 { grid-template-columns: repeat(4,minmax(0,1fr)); } }
        .feature-card { background: var(--gd-background, #ffffff); padding:2rem; border-radius:15px; box-shadow:0 4px 15px rgba(0,0,0,0.1); margin:1rem 0; border-left:4px solid #667eea; transition: transform 0.3s ease; }
        .feature-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }
        .quick-stat { background: linear-gradient(135deg,#f093fb 0%, #f5576c 100%); color:#fff; padding:1.5rem; border-radius:12px; text-align:center; margin:1rem; position:relative; overflow:hidden; }
        .quick-stat:before { content:""; position:absolute; inset:0; background: linear-gradient(130deg,rgba(255,255,255,0.22),transparent); opacity:0; transition:opacity .5s ease; }
        .quick-stat:hover:before { opacity:1; }
        .match-card { background:#fff; border:2px solid #e0e0e0; border-radius:12px; padding:1.5rem; margin:1rem 0; transition: all 0.3s ease; }
        .match-card:hover { border-color:#667eea; transform:scale(1.02); box-shadow:0 4px 20px rgba(102,126,234,0.2); }
        .prediction-badge { background: linear-gradient(135deg, var(--gd-primary) 0%, var(--gd-secondary) 100%); color:#fff; padding:.5rem 1rem; border-radius:20px; font-weight:700; text-align:center; margin:.5rem; }
        .confidence-meter { background: linear-gradient(90deg,#ff4757 0%, #ffa502 50%, #2ed573 100%); height:8px; border-radius:4px; margin:.5rem 0; }
        .team-logo { width:40px; height:40px; border-radius:50%; background: linear-gradient(135deg,#667eea 0%, #764ba2 100%); display:inline-flex; align-items:center; justify-content:center; color:#fff; font-weight:700; margin:0 10px; }
        .premium-metric {
            background: linear-gradient(135deg, rgba(255,255,255,0.85) 0%, rgba(245,245,255,0.7) 100%);
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: var(--gd-border-radius-lg);
            padding: var(--gd-spacing-lg);
            box-shadow: 0 8px 32px 0 rgba(31,78,121,0.18), 0 1.5px 6px 0 rgba(60,60,60,0.08);
            transition: box-shadow 0.3s cubic-bezier(0.4,0,0.2,1), transform 0.2s cubic-bezier(0.4,0,0.2,1);
        }
        .premium-metric:focus-within, .premium-metric:hover {
            box-shadow: 0 16px 40px 0 rgba(31,78,121,0.22), 0 3px 12px 0 rgba(60,60,60,0.12);
            outline: 2px solid var(--gd-accent);
            outline-offset: 2px;
            transform: translateY(-2px) scale(1.01);
        }
        .premium-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.92) 0%, rgba(245,245,255,0.82) 100%);
            backdrop-filter: blur(20px) saturate(180%);
            -webkit-backdrop-filter: blur(20px) saturate(180%);
            border: 1.5px solid rgba(31,78,121,0.12);
            border-radius: var(--gd-border-radius-lg);
            padding: var(--gd-spacing-xl);
            box-shadow: 0 12px 48px 0 rgba(31,78,121,0.18), 0 2px 8px 0 rgba(60,60,60,0.10);
            transition: box-shadow 0.3s cubic-bezier(0.4,0,0.2,1), transform 0.2s cubic-bezier(0.4,0,0.2,1);
        }
        .premium-card:focus-within, .premium-card:hover {
            box-shadow: 0 24px 64px 0 rgba(31,78,121,0.22), 0 4px 16px 0 rgba(60,60,60,0.14);
            outline: 2px solid var(--gd-accent);
            outline-offset: 2px;
            transform: translateY(-4px) scale(1.015);
        }
        /* Material Design micro-interactions */
        .stButton > button, .cta-primary {
            box-shadow: 0 2px 4px 0 rgba(31,78,121,0.10), 0 0.5px 2px 0 rgba(60,60,60,0.06);
            transition: box-shadow 0.2s cubic-bezier(0.4,0,0.2,1), background 0.2s cubic-bezier(0.4,0,0.2,1), transform 0.1s cubic-bezier(0.4,0,0.2,1);
        }
        .stButton > button:active, .cta-primary:active {
            box-shadow: 0 1px 2px 0 rgba(31,78,121,0.08);
            transform: scale(0.98);
        }
        /* Glassmorphic effect for cards */
        .goaldiggers-card, .featured-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.85) 0%, rgba(240,245,255,0.7) 100%);
            backdrop-filter: blur(14px) saturate(170%);
            -webkit-backdrop-filter: blur(14px) saturate(170%);
            border: 1px solid rgba(31,78,121,0.10);
            box-shadow: 0 6px 24px 0 rgba(31,78,121,0.12);
        }
        /* Accessibility: focus ring for all interactive elements */
        button:focus, .stButton > button:focus, .cta-primary:focus {
            outline: 2.5px solid var(--gd-accent);
            outline-offset: 2px;
        }
        /* Responsive tweaks for glassmorphic cards */
        @media (max-width: 768px) {
            .premium-card, .goaldiggers-card, .featured-card {
                padding: var(--gd-spacing-md);
            }
        }
        """
    
    def _get_integrated_styles(self) -> str:
        """Get integrated dashboard specific styles."""
        return """
        
        /* Integrated Dashboard Specific Styles */
        .integrated-section {{
            background: var(--gd-white);
            border-radius: var(--gd-border-radius-lg);
            padding: var(--gd-spacing-lg);
            margin-bottom: var(--gd-spacing-lg);
            border: 1px solid var(--gd-light-gray);
            box-shadow: var(--gd-shadow-md);
        }}
        
        .integrated-metric {{
            background: var(--gd-light-gray);
            border-radius: var(--gd-border-radius);
            padding: var(--gd-spacing-md);
            text-align: center;
            transition: var(--gd-transition);
        }}
        
        .integrated-metric:hover {{
            background: var(--gd-white);
            box-shadow: var(--gd-shadow-md);
        }}
        """
    
    def create_unified_header(self, title: str, subtitle: str = None) -> None:
        """Create unified header across all dashboards."""
        subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
        
        st.markdown(f"""
        <div class="goaldiggers-header">
            <h1>{title}</h1>
            {subtitle_html}
        </div>
        """, unsafe_allow_html=True)
    
    def create_unified_card(self, content_func, card_class: str = "goaldiggers-card") -> None:
        """Create unified card component."""
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        content_func()
        st.markdown('</div>', unsafe_allow_html=True)
    
    def create_unified_metric_row(self, metrics: dict[str, Any]) -> None:
        """Create unified metric row with consistent styling."""
        cols = st.columns(len(metrics))
        
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, dict):
                    st.metric(
                        label=label,
                        value=value.get('value', ''),
                        delta=value.get('delta', None)
                    )
                else:
                    st.metric(label=label, value=value)

    # === New Shared Component Helpers (for consolidation) ===
    def render_chip(self, label: str, kind: str = "default") -> None:
        """Render a chip with optional semantic kind."""
        kind_class = {
            'primary': 'gd-chip gd-chip-primary',
            'success': 'gd-chip gd-chip-success',
        }.get(kind, 'gd-chip')
        st.markdown(f"<span class='{kind_class}'>{label}</span>", unsafe_allow_html=True)

    def render_button(self, label: str, key: str | None = None, kind: str = 'primary') -> bool:
        """Render a unified styled button; returns True if clicked."""
        # For now we wrap streamlit button; future: custom JS hook.
        return st.button(label, key=key, type='primary' if kind == 'primary' else 'secondary')

    def render_card(self, body_func, accent: bool = False, class_name: str = 'gd-card') -> None:
        """Render a unified card wrapper.

        Args:
            body_func: callable to render inner Streamlit content
            accent: whether to add gradient border accent
            class_name: base class to apply
        """
        wrapper_open = f"<div class='{class_name}{' gd-card-accent' if accent else ''}'>"
        st.markdown(wrapper_open, unsafe_allow_html=True)
        try:
            body_func()
        finally:
            st.markdown('</div>', unsafe_allow_html=True)

    def render_metric_chip_row(self, data: dict[str, Any]) -> None:
        """Render a row of compact metric chips using inline metric container."""
        chips_html = "<div class='gd-inline-metrics'>" + "".join(
            f"<div class='metric'><strong>{k}</strong><br><span>{v}</span></div>" for k, v in data.items()
        ) + "</div>"
        st.markdown(chips_html, unsafe_allow_html=True)

# Global instance
_unified_design_system = None

@st.cache_data
def _get_cached_unified_css(dashboard_type: str, brand_colors: dict, typography: dict, spacing: dict, components: dict) -> str:
    """Get cached unified CSS for ultra-fast repeated access."""
    # Try to load from file first (for production)
    css_path = "dashboard/static/unified_design_system.css"
    try:
        with open(css_path, encoding='utf-8') as f:
            unified_css = f.read()

        # Add dashboard-specific enhancements
        dashboard_specific_css = _get_dashboard_specific_css(dashboard_type, brand_colors)
        if dashboard_specific_css:
            unified_css += f"\n{dashboard_specific_css}"

        return f"<style>{unified_css}</style>"

    except FileNotFoundError:
        # Fallback to generated CSS (cached for performance)
        logger.warning("Unified CSS file not found, using generated CSS")
        return _generate_unified_css_static(dashboard_type, brand_colors, typography, spacing, components)

def _get_dashboard_specific_css(dashboard_type: str, brand_colors: dict) -> str:
    """Get dashboard-specific CSS enhancements."""
    if dashboard_type == 'premium':
        return f"""
        /* Premium Dashboard Enhancements */
        .premium-gradient {{
            background: {brand_colors.get('gradient_primary', 'linear-gradient(135deg, #1f4e79 0%, #2a5298 100%)')};
        }}
        """
    return ""

def _generate_unified_css_static(dashboard_type: str, brand_colors: dict, typography: dict, spacing: dict, components: dict) -> str:
    """Generate unified CSS based on dashboard type (static version for caching)."""
    base_css = f"""
    <style>
    /* GoalDiggers Unified Design System - Ultra-Fast Startup */
    /* Use system fonts for ultra-fast loading */

    /* CSS Custom Properties */
    :root {{
        /* Brand Colors */
        --gd-primary: {brand_colors['primary']};
        --gd-secondary: {brand_colors['secondary']};
        --gd-accent: {brand_colors['accent']};
        --gd-white: {brand_colors['white']};
        --gd-light-gray: {brand_colors['light_gray']};

        /* Typography */
        --gd-font-family: {typography['font_family']};
        --gd-font-size-base: {typography['font_size_base']};

        /* Spacing */
        --gd-spacing-md: {spacing['md']};
        --gd-spacing-lg: {spacing['lg']};

        /* Components */
        --gd-border-radius: {components['border_radius_md']};
        --gd-shadow-md: {components['shadow_md']};
    }}

    /* Minimal essential styles for ultra-fast startup */
    .main .block-container {{
        font-family: var(--gd-font-family);
        color: var(--gd-primary);
    }}
    </style>
    """
    return base_css

def get_unified_design_system() -> UnifiedDesignSystem:
    """Get global unified design system instance with caching."""
    global _unified_design_system
    if _unified_design_system is None:
        # Use Streamlit caching for ultra-fast repeated access
        _unified_design_system = _create_cached_design_system()
    return _unified_design_system

def get_unified_styling(dashboard_type: str = 'premium') -> str:
    """
    Get unified CSS styling for dashboard integration.
    
    Args:
        dashboard_type: Type of dashboard (premium, cross_league, integrated)
    
    Returns:
        CSS styling string for injection
    """
    design_system = get_unified_design_system()
    return design_system._generate_unified_css(dashboard_type)

@st.cache_resource
def _create_cached_design_system() -> UnifiedDesignSystem:
    """Create cached design system instance for ultra-fast startup."""
    return UnifiedDesignSystem()


# ============================================================================
# COMPONENT HELPER FUNCTIONS (from design_components.py)
# Consolidated here for single source of truth
# ============================================================================

def ensure_tokens(theme: str | None = None, force: bool = False):
    """
    Ensure design tokens are injected once per run.
    Consolidated from design_components.py for single source of truth.
    
    Args:
        theme: Optional explicit theme override ('light'|'dark')
        force: Re-inject even if previously injected
    """
    import os
    key = '_gd_tokens_injected'
    if not force and st.session_state.get(key):
        return
    
    # Allow environment variable to set initial default theme
    if 'gd_theme' not in st.session_state:
        env_default = os.getenv('GOALDIGGERS_DEFAULT_THEME', '').lower()
        if env_default in ('light', 'dark'):
            st.session_state['gd_theme'] = env_default
    
    theme_pref = theme or st.session_state.get('gd_theme', 'light')
    try:
        # Use the unified design system CSS injection
        design_system = get_unified_design_system()
        design_system.inject_unified_css('premium')  # Default to premium styling
        st.session_state[key] = True
    except Exception as e:
        st.warning(f"Design token injection issue: {e}")
        logger.warning(f"Token injection failed: {e}")


def section_header(icon: str, title: str, subtitle: str | None = None):
    """Render a section header with icon and optional subtitle."""
    st.markdown(
        f"<h2 class='gd-section-title'>{icon} {title}</h2>",
        unsafe_allow_html=True
    )
    if subtitle:
        st.caption(subtitle)


def hero(title: str, description: str, chips: list[str] | None = None):
    """
    Render a prominent hero section using current token styling.
    
    Args:
        title: Hero title
        description: Hero description text
        chips: Optional list of feature chips/badges to display
    """
    ensure_tokens()
    
    chips_html = ''
    if chips:
        chips_html = (
            '<div class="gd-hero-chips" style="margin-top:1rem; display:flex; '
            'flex-wrap:wrap; gap:8px;">'
            + ''.join(
                f"<span class='gd-chip' role='note' aria-label='feature'>{c}</span>"
                for c in chips
            )
            + '</div>'
        )
    
    st.markdown(
        f"""
        <section class='gd-hero' role='banner' aria-label='Primary introduction'>
            <h2>{title}</h2>
            <p style='font-size:1.1rem; line-height:1.55; max-width:880px;'>
                {description}
            </p>
            {chips_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


from contextlib import contextmanager
from typing import Callable


def card(
    body_fn: Callable[[], None] | None = None,
    title: str | None = None,
    variant: str = 'solid',
    key: str | None = None
):
    """
    Render a styled card or return a context manager.
    
    Args:
        body_fn: Optional function to render inside card. If None, returns context manager
        title: Optional card title
        variant: Card variant ('solid' or 'glass' for glassmorphic effect)
        key: Optional unique key for Streamlit
    
    Returns:
        Context manager if body_fn is None, otherwise renders immediately
    
    Example:
        # As context manager
        with card(title="Example"):
            st.write("content")
        
        # As immediate render
        card(body_fn=lambda: st.write("content"), title="Example")
    """
    ensure_tokens()
    
    classes = 'gd-card'
    if variant == 'glass':
        classes += ' gd-glass'
    
    @contextmanager
    def _cm():
        # Use an explicit container so Streamlit places child widgets inside
        container = st.container()
        with container:
            container.markdown(f"<div class='{classes}' role='group'>", unsafe_allow_html=True)
            if title:
                container.markdown(
                    f"<h4 style='margin-top:0;'>{title}</h4>",
                    unsafe_allow_html=True
                )
            # Ensure all downstream Streamlit calls render within the card wrapper
            with container.container():
                yield
            container.markdown("</div>", unsafe_allow_html=True)
    
    if body_fn is None:
        return _cm()
    
    with _cm():
        body_fn()


def theme_toggle():
    """
    Render theme toggle buttons (light/dark).
    Thin wrapper delegating to newer theme_utils if available; fallback to legacy buttons.
    """
    try:
        from dashboard.components.theme_utils import (
            bootstrap_theme_from_storage,
            render_theme_toggle,
        )
        bootstrap_theme_from_storage()
        ensure_tokens()
        render_theme_toggle("Theme")
        return
    except Exception:
        pass
    
    # Legacy fallback
    ensure_tokens()
    _DEF_KEY_PREFIX = "gd_comp_"
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button('🌞 Light', key=_DEF_KEY_PREFIX + 'light'):
            if st.session_state.get('gd_theme') != 'light':
                st.session_state['gd_theme'] = 'light'
                st.session_state['_gd_tokens_injected'] = False
                st.rerun()
    with col2:
        if st.button('🌚 Dark', key=_DEF_KEY_PREFIX + 'dark'):
            if st.session_state.get('gd_theme') != 'dark':
                st.session_state['gd_theme'] = 'dark'
                st.session_state['_gd_tokens_injected'] = False
                st.rerun()


# ============================================================================
# CONVENIENCE FUNCTION (from unified_production_design_system.py)
# ============================================================================

def inject_production_css(theme: str = 'light'):
    """
    Inject unified production CSS with mobile-first responsive design.
    Convenience function matching unified_production_design_system API.
    
    Args:
        theme: Color theme ('light' or 'dark')
    """
    design_system = get_unified_design_system()
    design_system.inject_unified_css('premium')  # Use premium as default
    logger.info(f"✅ Unified production CSS injected (theme: {theme})")


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# For code importing from unified_production_design_system
get_unified_production_design_system = get_unified_design_system
get_unified_production_design_system = get_unified_design_system
