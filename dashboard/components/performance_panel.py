"""Lightweight Performance Panel Component

Provides real-time (interval-based) system + cache metrics for in-app visibility.
Designed to be optional: failures silently degrade to avoid breaking main UI.

Future extensions (Todo #10):
- Ingestion latency chart (pull from a shared latency registry)
- Model inference timing distribution
- Async queue depth visualization
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import streamlit as st

# Global state tracking
_REAL_DATA_PREDICTIONS = 0
_LAST_CALIB_STATUS = {}
_CALIB_APPLIED_COUNT = 0

def record_calibration_status(status: Dict[str, Any]):
    global _LAST_CALIB_STATUS, _CALIB_APPLIED_COUNT
    try:
        _LAST_CALIB_STATUS = status
        applied = status.get('applied_count') or status.get('applied')
        if isinstance(applied, int):
            _CALIB_APPLIED_COUNT = applied
    except Exception:
        pass

try:  # Optional dependency
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    from cached_data_utilities import cache_monitor
except Exception:  # pragma: no cover
    cache_monitor = None  # type: ignore

REFRESH_SECONDS = 15  # Update frequency

@st.cache_data(ttl=5)
def _get_process_metrics() -> Dict[str, Any]:
    """Collect system/process metrics (cached briefly to prevent flicker)."""
    if psutil is None:
        return {"cpu_percent": None, "memory_percent": None, "rss_mb": None}
    proc = psutil.Process()
    with proc.oneshot():
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        rss_mb = proc.memory_info().rss / (1024 * 1024)
    return {"cpu_percent": cpu, "memory_percent": mem, "rss_mb": f"{rss_mb:.1f}MB"}

@st.cache_data(ttl=5)
def _get_cache_metrics() -> Dict[str, Any]:
    if cache_monitor is None:
        return {}
    stats = cache_monitor.get_stats_summary()
    return {
        "entries": stats.get("total_entries"),
        "hit_rate": f"{stats.get('hit_rate', 0)*100:.1f}%",
        "memory_mb": f"{stats.get('approx_cache_size_mb', 0):.1f}MB",
    }

def render_performance_panel() -> None:
    """Render a collapsible performance panel if space permits."""
    with st.expander("📊 Performance Panel", expanded=False):
        col1, col2, col3 = st.columns(3)
        metrics = _get_process_metrics()
        cache_stats = _get_cache_metrics()
        ts = datetime.utcnow().strftime("%H:%M:%S")
        with col1:
            st.markdown("**System**")
            if metrics["cpu_percent"] is not None:
                st.metric("CPU %", metrics["cpu_percent"])
                st.metric("Mem %", metrics["memory_percent"])
                st.metric("Proc RSS", metrics["rss_mb"])
            else:
                st.caption("psutil unavailable")
        with col2:
            st.markdown("**Cache**")
            if cache_stats:
                st.metric("Entries", cache_stats.get("entries"))
                st.metric("Hit Rate", cache_stats.get("hit_rate"))
                st.metric("Size", cache_stats.get("memory_mb"))
            else:
                st.caption("Cache monitor unavailable")
        with col3:
            st.markdown("**Timing**")
            st.metric("Updated", ts)
            st.caption(f"Refresh every {REFRESH_SECONDS}s")
        st.progress(min(1.0, (time.time() % REFRESH_SECONDS)/REFRESH_SECONDS))

        # Advanced subsection: Latency & Calibration (collapsed UI style)
        with st.expander("Model & Calibration Diagnostics", expanded=False):
            sub1, sub2, sub3 = st.columns(3)
            # Latency stats
            if _LATENCY_SAMPLES:
                arr = list(_LATENCY_SAMPLES)
                latest = arr[-1]
                avg = sum(arr)/len(arr)
                p95 = sorted(arr)[int(0.95* (len(arr)-1))] if len(arr) > 5 else max(arr)
                with sub1:
                    st.markdown("**Prediction Latency (s)**")
                    st.metric("Latest", f"{latest:.3f}")
                    st.metric("Avg", f"{avg:.3f}")
                    st.metric("p95", f"{p95:.3f}")
            else:
                with sub1:
                    st.caption("No prediction latency samples yet")

            # Calibration status
            with sub2:
                st.markdown("**Calibration**")
                if _LAST_CALIB_STATUS:
                    st.metric("Enabled", str(_LAST_CALIB_STATUS.get('enabled')))
                    st.metric("Fitted", str(_LAST_CALIB_STATUS.get('fitted')))
                    st.metric("Applied", _LAST_CALIB_STATUS.get('applied_count', '0'))
                else:
                    st.caption("No calibration data")

            # Sample distribution quick chart (mini sparkline)
            with sub3:
                st.markdown("**Latency Trend**")
                if _LATENCY_SAMPLES:
                    st.line_chart(list(_LATENCY_SAMPLES))
                else:
                    st.caption("Pending samples")

            st.caption("Latency samples collected in-process; reset on app restart.")
            
            # Second row: Ingestion & Real Data + Perf Instrumentation
            ing1, ing2, ing3 = st.columns(3)
            
            with ing1:
                st.markdown("**Ingestion Latency (s)**")
                if _INGESTION_LATENCY_SAMPLES:
                    ing_arr = list(_INGESTION_LATENCY_SAMPLES)
                    ing_latest = ing_arr[-1]
                    ing_avg = sum(ing_arr)/len(ing_arr)
                    st.metric("Latest", f"{ing_latest:.3f}")
                    st.metric("Avg", f"{ing_avg:.3f}")
                else:
                    st.caption("No ingestion samples")
                    
            with ing2:
                st.markdown("**Real Data Usage**")
                if _TOTAL_PREDICTIONS > 0:
                    real_data_rate = (_REAL_DATA_PREDICTIONS / _TOTAL_PREDICTIONS) * 100
                    st.metric("Rate", f"{real_data_rate:.1f}%")
                    st.metric("Real/Total", f"{_REAL_DATA_PREDICTIONS}/{_TOTAL_PREDICTIONS}")
                else:
                    st.caption("No predictions tracked")
                    
            with ing3:
                st.markdown("**Perf Instrumentation**")
                try:
                    from utils import performance_instrumentation as _pi  # type: ignore
                    snap = _pi.snapshot()
                    # Show a compact subset (top 2 timed + first cache stat)
                    perf_items = list(snap.get('performance', {}).items())[:2]
                    cache_items = list(snap.get('cache', {}).items())[:1]
                    if not perf_items and not cache_items:
                        st.caption("No samples yet")
                    else:
                        for name, data in perf_items:
                            st.write(f"• {name.split('.')[-1]} avg: {data.get('avg_ms') and data['avg_ms']:.1f} ms")
                        for name, data in cache_items:
                            ratio = data.get('hit_ratio')
                            st.write(f"• cache {name.split('.')[-1]} hit: {ratio*100:.1f}%" if ratio is not None else f"• cache {name.split('.')[-1]} pending")
                except Exception:
                    st.caption("Instrumentation unavailable")

            # Persisted prediction trend (JSON lines log)
            with st.expander("Prediction Trend (Persisted)", expanded=False):
                try:
                    from monitoring.prediction_metrics_log import read_recent_events
                    events = read_recent_events(limit=120, minutes=180)
                    if not events:
                        st.caption("No persisted prediction events yet.")
                    else:
                        import pandas as _pd
                        df = _pd.DataFrame(events)
                        if 'latency_ms' in df.columns:
                            df['rolling_latency'] = df['latency_ms'].rolling(window=10, min_periods=1).mean()
                        if 'real_ratio' in df.columns:
                            df['rolling_real_ratio'] = df['real_ratio'].rolling(window=10, min_periods=1).mean()
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if 'rolling_latency' in df.columns:
                                st.line_chart(df[['rolling_latency']])
                                st.caption("Rolling (10) Latency ms")
                        with col_b:
                            if 'rolling_real_ratio' in df.columns:
                                st.line_chart(df[['rolling_real_ratio']])
                                st.caption("Rolling (10) Real Data Ratio")
                        st.caption(f"Events: {len(df)} (<=120, last 180m)")
                except Exception as e:
                    st.caption(f"Trend unavailable: {e}")
