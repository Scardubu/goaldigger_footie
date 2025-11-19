"""
Theme utilities for GoalDiggers Streamlit apps.
Provides: set_theme(data-theme on :root), theme toggle, and persistence via session_state.
"""
from __future__ import annotations

from typing import Literal

import streamlit as st

ThemeName = Literal["light", "dark"]

THEME_KEY = "gd_theme"
DEFAULT_THEME: ThemeName = "light"


def get_theme() -> ThemeName:
    return st.session_state.get(THEME_KEY, DEFAULT_THEME)  # type: ignore[return-value]


def set_theme(theme: ThemeName) -> None:
    st.session_state[THEME_KEY] = theme
    _inject_theme_attr(theme)


def render_theme_toggle(label: str = "Theme") -> ThemeName:
    """Render a theme toggle with proper fixed positioning."""
    current = get_theme()
    
    # Apply fixed positioning CSS for theme toggle
    st.markdown("""
    <style>
    /* Fixed theme toggle positioning */
    .gd-theme-toggle-container {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 0.5rem 1rem;
        border-radius: 24px;
        border: 1px solid rgba(148,163,184,0.2);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    [data-theme="dark"] .gd-theme-toggle-container {
        background: rgba(15,23,42,0.95);
        border-color: rgba(148,163,184,0.3);
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    
    /* Hide default streamlit padding around toggle */
    .gd-theme-toggle-container .stCheckbox {
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render toggle in container
    st.markdown('<div class="gd-theme-toggle-container">', unsafe_allow_html=True)
    choice = st.checkbox("🌙 Dark mode", value=(current == "dark"), key="gd_theme_toggle_checkbox")
    st.markdown('</div>', unsafe_allow_html=True)
    
    new_theme: ThemeName = "dark" if choice else "light"
    if new_theme != current:
        set_theme(new_theme)
    return new_theme


def _inject_theme_attr(theme: ThemeName) -> None:
    # Use a data-theme attribute on <html> root. Streamlit allows injecting CSS via markdown.
    css = f"""
    <script>
      try {{
        const root = window.parent?.document?.documentElement || document.documentElement;
        if (root) root.setAttribute('data-theme', '{theme}');
        window.localStorage.setItem('{THEME_KEY}', '{theme}');
      }} catch (e) {{ /* ignore */ }}
    </script>
    """
    st.markdown(css, unsafe_allow_html=True)


def bootstrap_theme_from_storage() -> None:
    # On first load, try to read from localStorage via JS and set attribute accordingly
    js = f"""
    <script>
      try {{
        const saved = window.localStorage.getItem('{THEME_KEY}');
        const theme = saved === 'dark' ? 'dark' : 'light';
        const root = window.parent?.document?.documentElement || document.documentElement;
        if (root) root.setAttribute('data-theme', theme);
      }} catch (e) {{ /* ignore */ }}
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)
    # Mirror to session_state (fallback if JS blocked)
    if THEME_KEY not in st.session_state:
        st.session_state[THEME_KEY] = DEFAULT_THEME

# Add enhanced CSS for theme toggle to ensure proper fixed positioning and visibility
css = """
<style>
  /* Ensure theme toggle button is fixed top-right with high z-index */
  #theme-toggle {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 4px;
    padding: 8px;
  }
</style>
"""

# Inject the CSS into the Streamlit app
import streamlit as st
st.markdown(css, unsafe_allow_html=True)
