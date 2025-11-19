"""Accessible Stepper component for Streamlit using unified CSS classes."""
from __future__ import annotations

from typing import List, Optional

import streamlit as st


def render_stepper(steps: List[str], active_index: int = 0, completed: Optional[List[int]] = None, key: Optional[str] = None) -> None:
    completed = completed or []
    st.markdown('<nav class="gd-step-indicator" aria-label="Progress">', unsafe_allow_html=True)
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            is_active = i == active_index
            is_completed = i in completed
            btn_class = 'gd-btn gd-btn-sm rounded-full w-8 h-8'
            if is_completed:
                color = 'background: var(--gd-success); color: var(--gd-white);'
                label = '✓'
            elif is_active:
                color = 'background: var(--gd-primary); color: var(--gd-white);'
                label = str(i + 1)
            else:
                color = 'background: var(--gd-gray-200); color: var(--gd-gray-700);'
                label = str(i + 1)
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;'>"
                f"<div class='gd-btn gd-btn-sm' style='border-radius:9999px;width:32px;height:32px;display:flex;align-items:center;justify-content:center;{color}' aria-current={'step' if is_active else 'false'} aria-label='Step {i+1}: {step}'>"
                f"{label}</div>"
                f"<div><div class='text-sm' style='font-weight:600;{ 'color: var(--gd-primary);' if is_active else ''}'>{step}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown('</nav>', unsafe_allow_html=True)
