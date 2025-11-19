"""Reusable Streamlit error panel component for displaying recent errors.

Gracefully degrades (no-op) when Streamlit isn't available or not in a run context.
"""
from __future__ import annotations

from typing import List, Optional

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore

from dashboard.error_log import error_log

_DEF_LEVEL_ORDER = ["critical", "error", "warning", "info", "debug"]


def render_error_panel(
    title: str = "🛑 System Error Panel",
    max_errors: int = 15,
    include_levels: Optional[List[str]] = None,
    filter_type: Optional[str] = None,
    collapsed: bool = True,
    show_stack_toggle: bool = True,
):
    """Render recent errors with optional filtering.

    Args:
        title: Header / expander title
        max_errors: Max number of errors to display (most recent first)
        include_levels: Restrict to these log levels
        filter_type: Only include errors of this type
        collapsed: Whether expander is collapsed initially
        show_stack_toggle: Provide a button to reveal trace info when present
    """
    if st is None:
        return  # Streamlit not available

    # Get recent errors (already limited internally, we slice again defensively)
    errors = error_log.recent_errors[-max_errors:]
    if include_levels:
        include_levels_lower = {lvl.lower() for lvl in include_levels}
        errors = [e for e in errors if e.get("level", "").lower() in include_levels_lower]
    if filter_type:
        errors = [e for e in errors if e.get("type") == filter_type]

    with st.expander(title, expanded=not collapsed):
        if not errors:
            st.info("No recent errors.")
            return

        # Sort by severity order then reverse chronological
        def _severity_key(e):
            level = e.get("level", "error").lower()
            try:
                return _DEF_LEVEL_ORDER.index(level)
            except ValueError:
                return len(_DEF_LEVEL_ORDER)
        errors_sorted = sorted(errors, key=_severity_key)
        errors_sorted.reverse()  # highest severity first

        for idx, err in enumerate(errors_sorted):
            level = err.get("level", "error").upper()
            etype = err.get("type", "UNKNOWN")
            msg = err.get("message", "<no message>")
            suggestion = err.get("suggestion")
            exception_text = err.get("exception")
            details = err.get("details")

            badge_color = {
                "CRITICAL": "#8B0000",
                "ERROR": "#B22222",
                "WARNING": "#CD8500",
                "INFO": "#1E90FF",
                "DEBUG": "#708090",
            }.get(level, "#B22222")

            st.markdown(
                f"<div style='border:1px solid {badge_color};padding:8px;border-radius:6px;margin-bottom:6px'>"
                f"<span style='background:{badge_color};color:white;padding:2px 6px;border-radius:4px;font-size:12px;margin-right:6px'>{level}</span>"
                f"<strong>[{etype}]</strong> {msg}"
                "</div>",
                unsafe_allow_html=True,
            )
            if exception_text:
                st.caption(f"Exception: {exception_text}")
            if suggestion:
                st.caption(f"Suggestion: {suggestion}")

            if show_stack_toggle and details:
                if st.button(
                    f"Details #{idx+1}", key=f"err_detail_{idx}", help="View structured error context"
                ):
                    st.json(details)

__all__ = ["render_error_panel"]
