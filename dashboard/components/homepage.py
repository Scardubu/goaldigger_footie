#!/usr/bin/env python3
"""Homepage component for GoalDiggers.

Lightweight, data-aware landing experience that stitches together:
- Hero system health + model calibration status
- Recent fixtures (from real_data_integrator if available) with quick prediction affordance
- Top betting insights preview (pulls from predictor insights util if accessible)
- Explainability teaser using last feature snapshot if present

All heavy operations are wrapped with safe fallbacks so homepage never blocks.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import streamlit as st

# Soft imports – homepage must remain resilient if subsystems unavailable
try:
    from real_data_integrator import (  # type: ignore
        get_real_matches_data,
        get_todays_fixtures,
    )
except Exception:  # pragma: no cover - resilience path
    get_real_matches_data = None  # type: ignore
    get_todays_fixtures = None  # type: ignore

try:
    from models.enhanced_real_data_predictor import (
        _get_singleton as _get_predictor_singleton,  # internal for last features
    )
    from models.enhanced_real_data_predictor import (
        get_enhanced_match_prediction,  # type: ignore
    )
except Exception:  # pragma: no cover
    get_enhanced_match_prediction = None  # type: ignore
    _get_predictor_singleton = None  # type: ignore

try:
    from explainability.explanation_service import explanation_service  # type: ignore
except Exception:  # pragma: no cover
    explanation_service = None  # type: ignore

try:
    from metrics.metrics_wrapper import snapshot as metrics_snapshot  # type: ignore
except Exception:  # pragma: no cover
    metrics_snapshot = None  # type: ignore

try:
    from utils.ingestion_freshness import compute_freshness  # type: ignore
except Exception:  # pragma: no cover
    compute_freshness = None  # type: ignore

# Calibration artifact (heuristic path) – adjust if config exposes explicit location later
CALIBRATION_STATE_PATH_CANDIDATES = [
    "calibration/calibration_state.json",
    "artifacts/calibration_state.json",
]

INGESTION_ARTIFACT_DIR_CANDIDATES = [
    "data/ingestion_runs",
    "artifacts/ingestion_runs",
]

LAST_FEATURE_SNAPSHOT_KEY = "last_prediction_features"


def _read_json_safe(path: str) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _find_first_existing(paths: List[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _load_latest_ingestion_metrics() -> Dict[str, Any]:
    base_dir = _find_first_existing(INGESTION_ARTIFACT_DIR_CANDIDATES)
    if not base_dir:
        return {}
    try:
        latest_file = None
        latest_mtime = -1
        for fname in os.listdir(base_dir):
            if fname.endswith('.json'):
                fpath = os.path.join(base_dir, fname)
                mtime = os.path.getmtime(fpath)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_file = fpath
        if latest_file:
            data = _read_json_safe(latest_file)
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


def _extract_ingestion_hero(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not metrics:
        return {}
    summary = metrics.get("summary") or {}
    steps = metrics.get("steps") or []
    success_steps = sum(1 for s in steps if s.get("status") == "success")
    freshness_minutes = None
    try:
        last_run_ts = summary.get("completed_at") or summary.get("ended_at")
        if last_run_ts:
            dt = datetime.fromisoformat(last_run_ts.replace("Z", "+00:00"))
            freshness_minutes = (datetime.utcnow() - dt).total_seconds() / 60.0
    except Exception:
        pass
    return {
        "total_steps": len(steps),
        "successful_steps": success_steps,
        "success_ratio": (success_steps / len(steps)) * 100 if steps else None,
        "freshness_minutes": freshness_minutes,
    }


def _load_calibration_status() -> Dict[str, Any]:
    path = _find_first_existing(CALIBRATION_STATE_PATH_CANDIDATES)
    if not path:
        return {}
    data = _read_json_safe(path) or {}
    if not isinstance(data, dict):
        return {}
    # Heuristic fields
    return {
        "fitted": bool(data.get("fitted") or data.get("is_fitted")),
        "method": data.get("method") or data.get("strategy"),
        "updated_at": data.get("updated_at") or data.get("timestamp"),
    }


def _get_recent_fixtures(limit: int = 6) -> List[Dict[str, Any]]:
    fixtures: List[Dict[str, Any]] = []
    if get_todays_fixtures:
        try:
            fixtures = get_todays_fixtures() or []  # type: ignore
        except Exception:
            fixtures = []
    # Normalize simple shape
    norm = []
    for fx in fixtures[:limit]:
        norm.append({
            "home": fx.get("homeTeam") or fx.get("home") or fx.get("home_team"),
            "away": fx.get("awayTeam") or fx.get("away") or fx.get("away_team"),
            "competition": fx.get("competition") or fx.get("league") or fx.get("competition_name"),
            "time": fx.get("utcDate") or fx.get("date") or fx.get("kickoff"),
        })
    return [f for f in norm if f.get("home") and f.get("away")]


def _preview_betting_insights(fixtures: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    if not get_enhanced_match_prediction:
        return []
    previews = []
    for fx in fixtures[:limit]:
        try:
            pred = get_enhanced_match_prediction(fx["home"], fx["away"], include_betting_insights=True)
            insights = (pred or {}).get("betting_insights") or []
            if insights:
                # take top first value insight
                top = insights[0]
                previews.append({
                    "match": f"{fx['home']} vs {fx['away']}",
                    "insight": top.get("recommendation") or top.get("text") or "Insight available",
                    "ev": top.get("expected_value") or top.get("ev")
                })
        except Exception:
            # Fail silently to keep homepage fast
            continue
    return previews


def _explainability_teaser() -> Dict[str, Any]:
    feats = st.session_state.get(LAST_FEATURE_SNAPSHOT_KEY) or {}
    if not isinstance(feats, dict) or not feats:
        return {}
    # Take top (first 3) feature keys deterministically
    items = list(feats.items())[:3]
    return {"top_features": items}


def render_homepage():
    """Render the homepage with resilient data sections."""
    st.markdown("""
    <div class='goaldiggers-header gd-fade-in'>
        <h1>🏠 Home</h1>
        <p>Your consolidated football intelligence at a glance</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data (fast, best-effort)
    ingestion_metrics = _load_latest_ingestion_metrics()
    ingestion_hero = _extract_ingestion_hero(ingestion_metrics)
    freshness = None
    if compute_freshness:
        try:
            freshness = compute_freshness()
        except Exception:
            freshness = None
    calibration_status = _load_calibration_status()
    fixtures = _get_recent_fixtures()
    insights_preview = _preview_betting_insights(fixtures)
    explainability = _explainability_teaser()

    # Hero metrics + calibration detail
    cols = st.columns(5)
    with cols[0]:
        sr = ingestion_hero.get("success_ratio")
        st.metric("Ingestion Success", f"{sr:.0f}%" if sr is not None else "—")
    with cols[1]:
        # Prefer new freshness util if available
        fr = None
        if freshness and freshness.age_minutes is not None:
            fr = freshness.age_minutes
        else:
            fr = ingestion_hero.get("freshness_minutes")
        st.metric("Data Freshness", f"{fr:.0f}m" if fr is not None else "—")
    if freshness and freshness.row_counts:
        with st.expander("📦 Ingestion Row Counts"):
            st.json(freshness.row_counts)
    with cols[2]:
        st.metric("Fixtures Today", len(fixtures) if fixtures else 0)
    with cols[3]:
        st.metric("Calibrated", "Yes" if calibration_status.get("fitted") else "No")
    with cols[4]:
        # Show calibration method or placeholder
        st.metric("Calib Method", calibration_status.get("method") or "—")

    # Detailed calibration widget (collapsed)
    with st.expander("📐 Calibration Status & Sample Counts"):
        calib_info = {}
        # Attempt to pull richer info from predictor if it stored any in session or accessible
        try:
            predictor_state = st.session_state.get("enhanced_predictor_state") or {}
            if isinstance(predictor_state, dict):
                calib_info = predictor_state.get("calibration_info") or {}
        except Exception:
            pass
        # Merge heuristic file status
        merged = {**calibration_status, **calib_info}
        if not merged:
            st.caption("No calibration info available. Model may be running uncalibrated.")
        else:
            cols_c = st.columns(4)
            with cols_c[0]:
                st.metric("Fitted", "Yes" if merged.get("fitted") else "No")
            with cols_c[1]:
                st.metric("Method", merged.get("method") or "—")
            with cols_c[2]:
                sc = merged.get("sample_counts") or {}
                total_samples = sum(int(v) for v in sc.values()) if sc else None
                st.metric("Samples", total_samples if total_samples else 0)
            with cols_c[3]:
                updated_at = merged.get("updated_at") or merged.get("timestamp")
                st.metric("Updated", updated_at.split("T")[0] if isinstance(updated_at, str) else "—")
            if merged.get("sample_counts"):
                st.json(merged.get("sample_counts"))

    st.markdown("---")

    # Recent Fixtures + Quick Predict
    st.markdown("### 📅 Today's Fixtures")
    if not fixtures:
        st.info("No fixtures available (API offline or none today).")
    else:
        for fx in fixtures:
            with st.expander(f"⚽ {fx['home']} vs {fx['away']}"):
                st.caption(f"{fx.get('competition') or 'Competition'} | {fx.get('time') or 'TBD'}")
                if get_enhanced_match_prediction and st.button("Run Prediction", key=f"pred_{fx['home']}_{fx['away']}"):
                    with st.spinner("Computing prediction..."):
                        try:
                            pred = get_enhanced_match_prediction(fx['home'], fx['away'], include_betting_insights=True)
                            st.json({k: v for k, v in pred.items() if k not in {"raw_features"}})
                            # Capture latest feature snapshot for explainability panel if predictor exposes it
                            try:
                                if _get_predictor_singleton:
                                    predictor_obj = _get_predictor_singleton()
                                    feats_df = predictor_obj.get_last_features()
                                    if feats_df is not None:
                                        st.session_state[LAST_FEATURE_SNAPSHOT_KEY] = feats_df.iloc[0].to_dict()
                                        st.session_state['enhanced_predictor_state'] = {
                                            'calibration_info': predictor_obj._calibration_status()  # internal OK for UI
                                        }
                            except Exception:
                                pass
                        except Exception as e:
                            st.error(f"Prediction error: {e}")

    # Betting Insights Preview
    st.markdown("### 💡 Betting Insights Preview")
    if not insights_preview:
        st.caption("No insights generated yet or predictor unavailable.")
    else:
        for p in insights_preview:
            st.success(f"{p['match']}: {p['insight']}" + (f" (EV: {p['ev']:+.1f}%)" if p.get('ev') else ""))

    # Explainability Teaser
    st.markdown("### 🧠 Explainability & Feature Insights")
    if not explainability:
        st.caption("Run a prediction above to capture feature snapshot.")
    else:
        feats = explainability['top_features']
        col_e1, col_e2 = st.columns([2,1])
        with col_e1:
            st.subheader("Top Features (Snapshot)")
            for name, value in feats:
                st.write(f"- **{name}**: {value}")
        with col_e2:
            # Offer explanation generation if service available
            if explanation_service and st.button("Generate Explanation", key="gen_expl"):
                try:
                    import pandas as _pd  # local import
                    feats_dict = st.session_state.get(LAST_FEATURE_SNAPSHOT_KEY) or {}
                    if feats_dict:
                        df = _pd.DataFrame([feats_dict])
                        with st.spinner("Computing explanation..."):
                            expl = explanation_service.explain(df)
                        st.session_state['last_explanation'] = expl
                except Exception as _e:
                    st.error(f"Explanation failed: {_e}")
            last_expl = st.session_state.get('last_explanation')
            if last_expl:
                st.caption("Explanation Summary")
                try:
                    # Show top absolute shap values (mock zeros tolerated)
                    shap_vals = last_expl.get('shap_values')
                    names = last_expl.get('feature_names', [])
                    if shap_vals is not None and names:
                        import numpy as _np
                        arr = _np.array(shap_vals)
                        if arr.ndim > 1:
                            arr = arr[0]
                        top_idx = _np.argsort(-_np.abs(arr))[:3]
                        for i in top_idx:
                            st.write(f"{names[i]}: {arr[i]:+.4f}")
                except Exception:
                    pass

    # Optional metrics snapshot
    if metrics_snapshot:
        with st.expander("📊 Internal Metrics Snapshot"):
            try:
                st.json(metrics_snapshot())
            except Exception:
                st.caption("Metrics snapshot unavailable.")

    st.markdown("---")
    st.caption("Homepage is lightweight & resilient – all sections degrade gracefully if subsystems are offline.")
