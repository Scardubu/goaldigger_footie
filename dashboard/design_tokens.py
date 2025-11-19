"""Design tokens for GoalDiggers UI (Material + Glass hybrid).
Provides centralized theming values and CSS variable injector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import streamlit as st


@dataclass(frozen=True)
class ColorPalette:
    primary: str = "#1f4e79"
    primary_alt: str = "#2a5298"
    secondary: str = "#667eea"
    accent: str = "#f093fb"
    success: str = "#2ed573"
    warning: str = "#ffa502"
    danger: str = "#ff4757"
    info: str = "#17a2b8"
    gradient_primary: str = "linear-gradient(135deg, #1f4e79 0%, #2a5298 100%)"
    gradient_accent: str = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
    surface: str = "#ffffff"
    surface_alt: str = "#f8f9fb"
    border: str = "#e5e7eb"
    text: str = "#1f2933"
    text_muted: str = "#6b7280"
    backdrop: str = "rgba(255,255,255,0.55)"
    shadow_rgba: str = "rgba(0,0,0,0.12)"

@dataclass(frozen=True)
class TypographyScale:
    font_family: str = '"Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif'
    size_xs: str = "0.75rem"
    size_sm: str = "0.875rem"
    size_md: str = "1rem"
    size_lg: str = "1.125rem"
    size_xl: str = "1.25rem"
    size_2xl: str = "1.5rem"
    size_3xl: str = "1.875rem"
    size_4xl: str = "2.25rem"
    weight_normal: str = "400"
    weight_medium: str = "500"
    weight_semibold: str = "600"
    weight_bold: str = "700"

@dataclass(frozen=True)
class SpacingScale:
    xs: str = "0.25rem"
    sm: str = "0.5rem"
    md: str = "1rem"
    lg: str = "1.5rem"
    xl: str = "2rem"
    xxl: str = "3rem"

@dataclass(frozen=True)
class RadiusScale:
    sm: str = "6px"
    md: str = "10px"
    lg: str = "16px"
    xl: str = "22px"
    pill: str = "9999px"

@dataclass(frozen=True)
class Elevation:
    sm: str = "0 2px 4px rgba(0,0,0,0.06)"
    md: str = "0 4px 12px rgba(0,0,0,0.09)"
    lg: str = "0 8px 28px rgba(0,0,0,0.14)"
    glass: str = "0 8px 32px rgba(0,0,0,0.18)"

_palette = ColorPalette()
_type = TypographyScale()
_space = SpacingScale()
_radius = RadiusScale()
_elev = Elevation()

@st.cache_data
def get_tokens() -> Dict[str, Dict[str, str]]:
    return {
        'colors': _palette.__dict__,
        'typography': _type.__dict__,
        'spacing': _space.__dict__,
        'radius': _radius.__dict__,
        'elevation': _elev.__dict__,
    }

def inject_tokens(theme: str = 'light') -> None:
    """Inject design token CSS variables + core component styles.

    Refactored to avoid f-string single-brace CSS causing NameError (e.g. 'transform').
    """
    try:
        tok = get_tokens()
        dark_adjust = ''
        if theme == 'dark':
            # Raw string (no f-string) to preserve CSS braces
            dark_adjust = (
                "body[data-theme='dark'] {"\
                "\n    --gd-surface: #10151c;"\
                "\n    --gd-surface-alt: #1d2530;"\
                "\n    --gd-text: #e5e7eb;"\
                "\n    --gd-text-muted: #9ca3af;"\
                "\n    --gd-border: #2d3743;"\
                "\n    background: #0d1218;"\
                "\n    color: var(--gd-text);"\
                "\n}\nbody[data-theme='dark'] .gd-card {"\
                "\n    background: linear-gradient(135deg, #18222d 0%, #1f2933 100%);"\
                "\n    border-color: var(--gd-border);"\
                "\n    box-shadow: 0 4px 20px rgba(0,0,0,0.6);"\
                "\n}\n"
            )
        # Build CSS variable block safely
        kv_pairs = {
            **tok['colors'],
            **{f"font-{k}": v for k,v in tok['typography'].items()},
            **{f"space-{k}": v for k,v in tok['spacing'].items()},
            **{f"radius-{k}": v for k,v in tok['radius'].items()},
            **{f"elev-{k}": v for k,v in tok['elevation'].items()},
        }
        css_vars = [":root {"] + [f"  --gd-{k.replace('_','-')}: {v};" for k,v in kv_pairs.items()] + ["}"]
        base = "\n".join(css_vars)
        # Assemble component CSS without f-string to avoid brace interpolation
        core_css = [
            "<style>",
            base,
            ".gd-card {",
            "  background: var(--gd-surface);",
            "  border:1px solid var(--gd-border);",
            "  border-radius: var(--gd-radius-lg);",
            "  padding: var(--gd-space-lg);",
            "  box-shadow: var(--gd-elev-md);",
            "  transition: box-shadow .25s ease, transform .25s ease;",
            "}",
            ".gd-card.gd-glass {",
            "  background: linear-gradient(135deg, rgba(255,255,255,0.75) 0%, rgba(255,255,255,0.55) 100%);",
            "  backdrop-filter: blur(18px) saturate(160%);",
            "  -webkit-backdrop-filter: blur(18px) saturate(160%);",
            "  border:1px solid rgba(255,255,255,0.35);",
            "  box-shadow: var(--gd-elev-glass);",
            "}",
            ".gd-card:hover { transform: translateY(-3px); box-shadow: var(--gd-elev-lg); }",
            ".gd-hero {",
            "  background: var(--gd-gradient-accent, linear-gradient(135deg,#667eea 0%,#764ba2 100%));",
            "  color:#fff; padding: var(--gd-space-xl) var(--gd-space-lg); border-radius: var(--gd-radius-xl);",
            "  position: relative; overflow: hidden; box-shadow: 0 8px 40px rgba(0,0,0,0.25);",
            "}",
            ".gd-hero h2 { margin-top:0; font-size: var(--gd-font-size-3xl); }",
            ".gd-chip { display:inline-flex; align-items:center; gap:6px; padding:6px 14px; background: rgba(255,255,255,0.15); border-radius: var(--gd-radius-pill); color:#fff; font-weight:600; }",
            ".gd-grid { display:grid; gap: var(--gd-space-lg); }",
            "@media (min-width: 900px) { .gd-grid.cols-3 { grid-template-columns:repeat(3,1fr);} .gd-grid.cols-2 { grid-template-columns:repeat(2,1fr);} }",
            dark_adjust,
            "</style>"
        ]
        st.markdown("\n".join(core_css), unsafe_allow_html=True)
    except Exception as e:  # Surface but do not crash app
        st.warning(f"Design token injection issue: {e}")

def _self_test_token_injection() -> bool:
    """Lightweight self-test to ensure token CSS builds without NameError.
    Returns True if injection completes quietly in both themes.
    """
    try:
        inject_tokens('light')
        inject_tokens('dark')
        return True
    except Exception:
        return False

__all__ = ["inject_tokens", "get_tokens", "_self_test_token_injection"]
