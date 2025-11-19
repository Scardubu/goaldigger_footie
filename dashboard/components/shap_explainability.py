"""SHAP Explainability rendering helpers for GoalDiggers dashboard.

Provides lightweight, defensive wrappers to display SHAP values (if enabled)
from the EnhancedRealDataPredictor. Falls back to pseudo-explanations or
status messages when SHAP is disabled or unavailable.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

try:  # Local import guarded to keep this module safe if predictor missing
    from models.enhanced_real_data_predictor import EnhancedRealDataPredictor
except Exception:  # pragma: no cover
    EnhancedRealDataPredictor = None  # type: ignore


def _format_importances(feature_names, shap_values):
    pairs = []
    try:
        for name, val in zip(feature_names, shap_values):
            pairs.append({
                'feature': name,
                'shap_value': float(val),
                'abs_value': float(abs(val))
            })
        pairs.sort(key=lambda x: x['abs_value'], reverse=True)
    except Exception:
        return []
    return pairs


def render_shap_panel(predictor: Optional[Any]) -> None:
    """Render an expandable SHAP explanation panel.

    predictor: instance of EnhancedRealDataPredictor (or compatible) or None.
    """
    if predictor is None or not hasattr(predictor, 'explain_last_prediction'):
        st.info("No predictor instance available for SHAP explanations.")
        return

    with st.expander("🔍 Model Explainability (SHAP)", expanded=False):
        try:
            explanation: Dict[str, Any] = predictor.explain_last_prediction()
        except Exception as e:  # pragma: no cover
            st.warning(f"Explainability unavailable: {e}")
            return

        if not explanation:
            st.write("No explanation data available.")
            return

        if explanation.get('is_mock'):
            reason = explanation.get('reason')
            st.caption(f"Mock explanation (reason: {reason}). Enable SHAP or generate a new prediction with ENABLE_SHAP=1.")
        else:
            st.caption("SHAP values reflect approximate feature contribution toward the model's output for the last prediction.")

        feature_names = explanation.get('feature_names') or []
        shap_values = explanation.get('shap_values') or []
        pairs = _format_importances(feature_names, shap_values)

        if not pairs:
            st.write("No feature contributions captured yet. Run a prediction.")
            return

        # Display top contributions
        top_n = pairs[:10]
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Top Features")
            for p in top_n:
                bar = abs(p['shap_value'])
                st.write(f"{p['feature']}: {p['shap_value']:+.4f}")
                st.progress(min(1.0, bar / (top_n[0]['abs_value'] or 1)), text="")
        with cols[1]:
            st.subheader("Raw Vector")
            st.json({p['feature']: p['shap_value'] for p in top_n})

        # Base value & meta
        meta_cols = st.columns(3)
        meta_cols[0].metric("Model Version", explanation.get('model_version'))
        meta_cols[1].metric("Base Value", f"{explanation.get('base_value')}")
        meta_cols[2].metric("Mode", "Mock" if explanation.get('is_mock') else "SHAP")

        # Offer raw dump toggle
        with st.expander("Raw Explanation Payload"):
            st.json(explanation)

__all__ = ["render_shap_panel"]
