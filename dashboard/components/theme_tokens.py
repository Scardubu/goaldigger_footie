"""GoalDiggers Design System Tokens

Centralized theme tokens (Material-inspired with subtle glassmorphism support)
that can be consumed by the unified design system and any component-level styling
helpers. This separation allows future extension (e.g., dark mode, user themes)
without touching component logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Base brand color palette (light theme)
BRAND_COLORS: Dict[str, str] = {
    'primary': '#1f4e79',
    'primary_light': '#2a5298',
    'primary_dark': '#1a3d5f',
    'secondary': '#28a745',
    'accent': '#fd7e14',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'white': '#ffffff',
    'light_gray': '#f8f9fa',
    'medium_gray': '#6c757d',
    'dark_gray': '#343a40',
    'black': '#000000',
    'gradient_primary': 'linear-gradient(135deg, #1f4e79 0%, #2a5298 100%)',
    'gradient_success': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
    'gradient_warning': 'linear-gradient(135deg, #fd7e14 0%, #ffc107 100%)',
    'gradient_cross_league': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'gradient_scenario': 'linear-gradient(45deg, #ff6b6b, #ee5a24)',
    'achievement_gold': '#ffd700',
    'achievement_silver': '#c0c0c0',
    'achievement_bronze': '#cd7f32',
    'shadow': 'rgba(0,0,0,0.1)',
    'overlay': 'rgba(31,78,121,0.9)',
    # Glassmorphism & surfaces
    'glass_bg_1': 'rgba(255,255,255,0.82)',
    'glass_bg_2': 'rgba(255,255,255,0.68)',
    'glass_border': 'rgba(255,255,255,0.35)',
    'glass_highlight': 'linear-gradient(135deg, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.15) 60%)',
    # Semantic states / statuses
    'state_success': '#28a745',
    'state_warning': '#fd7e14',
    'state_danger': '#dc3545',
    'state_info': '#17a2b8',
    # Background layers
    'bg_default': '#ffffff',
    'bg_alt': '#f5f7fb',
    'bg_elevated': 'linear-gradient(135deg, #ffffff 0%, #f0f4fa 100%)'
}

# Typography scale (rem-based)
TYPOGRAPHY_TOKENS: Dict[str, str] = {
    'font_family': '"Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif',
    'font_family_mono': '"SF Mono", "Monaco", "Inconsolata", "Roboto Mono", monospace',
    'font_size_xs': '0.75rem',
    'font_size_sm': '0.875rem',
    'font_size_base': '1rem',
    'font_size_lg': '1.125rem',
    'font_size_xl': '1.25rem',
    'font_size_2xl': '1.5rem',
    'font_size_3xl': '1.875rem',
    'font_size_4xl': '2.25rem',
    'font_weight_light': '300',
    'font_weight_normal': '400',
    'font_weight_medium': '500',
    'font_weight_semibold': '600',
    'font_weight_bold': '700',
    'font_weight_extrabold': '800',
    'line_height_tight': '1.25',
    'line_height_normal': '1.5',
    'line_height_relaxed': '1.75'
}

# Spacing scale
SPACING_TOKENS: Dict[str, str] = {
    'xs': '0.25rem',
    'sm': '0.5rem',
    'md': '1rem',
    'lg': '1.5rem',
    'xl': '2rem',
    'xxl': '3rem',
    'xxxl': '4rem'
}

# Component primitives (radius, shadows, motion)
COMPONENT_TOKENS: Dict[str, str] = {
    'border_radius_sm': '0.375rem',
    'border_radius_md': '0.5rem',
    'border_radius_lg': '0.75rem',
    'border_radius_xl': '1rem',
    'border_radius_full': '9999px',
    'shadow_sm': '0 1px 2px 0 rgba(0,0,0,0.05)',
    'shadow_md': '0 4px 6px -1px rgba(0,0,0,0.1)',
    'shadow_lg': '0 10px 15px -3px rgba(0,0,0,0.1)',
    'shadow_xl': '0 20px 25px -5px rgba(0,0,0,0.1)',
    'transition_fast': '0.15s ease-in-out',
    'transition_normal': '0.3s ease-in-out',
    'transition_slow': '0.5s ease-in-out'
}

# Optional semantic elevations (Material-inspired)
ELEVATION_TOKENS: Dict[str, str] = {
    'elevation_0': 'none',
    'elevation_1': '0 1px 2px rgba(0,0,0,0.06)',
    'elevation_2': '0 3px 6px rgba(0,0,0,0.08)',
    'elevation_3': '0 6px 12px rgba(0,0,0,0.10)',
    'elevation_4': '0 10px 20px rgba(0,0,0,0.12)'
}

# Grouping for future theme switching
THEME_REGISTRY = {
    'light': {
        'colors': BRAND_COLORS,
        'typography': TYPOGRAPHY_TOKENS,
        'spacing': SPACING_TOKENS,
        'components': COMPONENT_TOKENS,
        'elevation': ELEVATION_TOKENS,
    },
    # Initial dark theme scaffold (can refine later)
    'dark': {
        'colors': {
            **BRAND_COLORS,
            'bg_default': '#11161d',
            'bg_alt': '#161d26',
            'bg_elevated': 'linear-gradient(135deg,#161d26 0%,#1f2732 100%)',
            'light_gray': '#1c232c',
            'medium_gray': '#586271',
            'dark_gray': '#e0e4e9',
            'white': '#ffffff',
            'glass_bg_1': 'rgba(25,32,42,0.82)',
            'glass_bg_2': 'rgba(25,32,42,0.68)',
            'glass_border': 'rgba(255,255,255,0.10)',
            'glass_highlight': 'linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.04) 60%)'
        },
        'typography': TYPOGRAPHY_TOKENS,
        'spacing': SPACING_TOKENS,
        'components': COMPONENT_TOKENS,
        'elevation': ELEVATION_TOKENS,
    }
}


def build_css_variable_block(theme: str = 'light') -> str:
    """Return a CSS variable block for the given theme.
    This keeps generation logic DRY across runtime vs static builds.
    """
    t = THEME_REGISTRY.get(theme, THEME_REGISTRY['light'])
    colors = t['colors']; typo = t['typography']; sp = t['spacing']; comp = t['components']
    lines = [":root {"]
    # Colors
    lines.extend([
        f"  --gd-primary: {colors['primary']};",
        f"  --gd-primary-light: {colors['primary_light']};",
        f"  --gd-primary-dark: {colors['primary_dark']};",
        f"  --gd-secondary: {colors['secondary']};",
        f"  --gd-accent: {colors['accent']};",
        f"  --gd-danger: {colors['danger']};",
        f"  --gd-info: {colors['info']};",
        f"  --gd-white: {colors['white']};",
        f"  --gd-light-gray: {colors['light_gray']};",
        f"  --gd-medium-gray: {colors['medium_gray']};",
        f"  --gd-dark-gray: {colors['dark_gray']};",
        f"  --gd-shadow: {colors['shadow']};",
        f"  --gd-bg-default: {colors.get('bg_default', '#ffffff')};",
        f"  --gd-bg-alt: {colors.get('bg_alt', '#f5f7fb')};",
        f"  --gd-bg-elevated: {colors.get('bg_elevated', '#ffffff')};",
        f"  --gd-glass-bg-1: {colors.get('glass_bg_1', 'rgba(255,255,255,0.82)')};",
        f"  --gd-glass-bg-2: {colors.get('glass_bg_2', 'rgba(255,255,255,0.68)')};",
        f"  --gd-glass-border: {colors.get('glass_border', 'rgba(255,255,255,0.35)')};",
        f"  --gd-glass-highlight: {colors.get('glass_highlight', 'linear-gradient(135deg, rgba(255,255,255,0.55) 0%, rgba(255,255,255,0.15) 60%)')};",
        f"  --gd-state-success: {colors.get('state_success', colors.get('secondary'))};",
        f"  --gd-state-warning: {colors.get('state_warning', colors.get('accent'))};",
        f"  --gd-state-danger: {colors.get('state_danger', colors.get('danger'))};",
        f"  --gd-state-info: {colors.get('state_info', colors.get('info'))};",
    ])
    # Typography
    lines.extend([
        f"  --gd-font-family: {typo['font_family']};",
        f"  --gd-font-size-base: {typo['font_size_base']};",
        f"  --gd-font-weight-normal: {typo['font_weight_normal']};",
        f"  --gd-font-weight-semibold: {typo['font_weight_semibold']};",
        f"  --gd-font-weight-bold: {typo['font_weight_bold']};",
        f"  --gd-line-height-normal: {typo['line_height_normal']};",
    ])
    # Spacing
    lines.extend([
        f"  --gd-spacing-sm: {sp['sm']};",
        f"  --gd-spacing-md: {sp['md']};",
        f"  --gd-spacing-lg: {sp['lg']};",
        f"  --gd-spacing-xl: {sp['xl']};",
    ])
    # Components
    lines.extend([
        f"  --gd-border-radius: {comp['border_radius_md']};",
        f"  --gd-border-radius-lg: {comp['border_radius_lg']};",
        f"  --gd-shadow-md: {comp['shadow_md']};",
        f"  --gd-shadow-lg: {comp['shadow_lg']};",
        f"  --gd-transition: {comp['transition_normal']};",
    ])
    lines.append("}")
    return "\n".join(lines)

__all__ = [
    'BRAND_COLORS', 'TYPOGRAPHY_TOKENS', 'SPACING_TOKENS', 'COMPONENT_TOKENS',
    'ELEVATION_TOKENS', 'THEME_REGISTRY', 'build_css_variable_block'
]
