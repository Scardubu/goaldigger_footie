"""Reusable component helpers leveraging design tokens."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional

import os
import streamlit as st

from .design_tokens import inject_tokens

_DEF_KEY_PREFIX = "gd_comp_"


def ensure_tokens(theme: Optional[str] = None, force: bool = False):
    """Ensure design tokens are injected once per run.

    Args:
        theme: Optional explicit theme override ('light'|'dark').
        force: Re-inject even if previously injected.
    """
    key = '_gd_tokens_injected'
    if not force and st.session_state.get(key):
        return
    # Allow environment variable to set initial default theme (light|dark)
    if 'gd_theme' not in st.session_state:
        env_default = os.getenv('GOALDIGGERS_DEFAULT_THEME', '').lower()
        if env_default in ('light','dark'):
            st.session_state['gd_theme'] = env_default
    theme_pref = theme or st.session_state.get('gd_theme', 'light')
    try:
        inject_tokens(theme_pref)
        st.session_state[key] = True
    except Exception as e:  # Non-fatal styling failure
        st.warning(f"Design token injection issue: {e}")


def section_header(icon: str, title: str, subtitle: Optional[str] = None):
    st.markdown(f"<h2 class='gd-section-title'>{icon} {title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def hero(title: str, description: str, chips: list[str] | None = None):
    """Render a prominent hero section using current token styling."""
    ensure_tokens()
    chips_html = ''
    if chips:
        chips_html = '<div class="gd-hero-chips" style="margin-top:1rem; display:flex; flex-wrap:wrap; gap:8px;">' + ''.join(
            f"<span class='gd-chip' role='note' aria-label='feature'>{c}</span>" for c in chips
        ) + '</div>'
    st.markdown(
        f"""
        <section class='gd-hero' role='banner' aria-label='Primary introduction'>
            <h2>{title}</h2>
            <p style='font-size:1.1rem; line-height:1.55; max-width:880px;'>{description}</p>
            {chips_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def card(body_fn: Callable[[], None] | None = None, title: Optional[str] = None, variant: str = 'solid', key: Optional[str] = None):
    """Render a styled card or return a context manager.

    If body_fn is provided, executes immediately. Otherwise can be used as:
        with card(title="Example"):
            st.write("content")
    """
    ensure_tokens()
    classes = 'gd-card'
    if variant == 'glass':
        classes += ' gd-glass'

    @contextmanager
    def _cm():
        st.markdown(f"<div class='{classes}' role='group'>", unsafe_allow_html=True)
        if title:
            st.markdown(f"<h4 style='margin-top:0;'>{title}</h4>", unsafe_allow_html=True)
        try:
            yield
        finally:
            st.markdown("</div>", unsafe_allow_html=True)

    if body_fn is None:
        return _cm()
    with _cm():
        body_fn()


def theme_toggle():
    """Thin wrapper delegating to newer theme_utils if available; fallback to legacy buttons."""
    try:
        from dashboard.components.theme_utils import render_theme_toggle, bootstrap_theme_from_storage
        bootstrap_theme_from_storage()
        ensure_tokens()
        render_theme_toggle("Theme")
        return
    except Exception:
        pass
    # Legacy fallback
    ensure_tokens()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button('🌞 Light', key=_DEF_KEY_PREFIX+'light'):
            if st.session_state.get('gd_theme') != 'light':
                st.session_state['gd_theme'] = 'light'
                st.session_state['_gd_tokens_injected'] = False
                st.rerun()
    with col2:
        if st.button('🌚 Dark', key=_DEF_KEY_PREFIX+'dark'):
            if st.session_state.get('gd_theme') != 'dark':
                st.session_state['gd_theme'] = 'dark'
                st.session_state['_gd_tokens_injected'] = False
                st.rerun()

__all__ = [
    'ensure_tokens', 'section_header', 'hero', 'card', 'theme_toggle'
]
