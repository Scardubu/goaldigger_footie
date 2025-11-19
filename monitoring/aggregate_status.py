"""Aggregate monitoring utilities.

Provides a single function `collect_aggregate_status()` that bundles:
 - Core service uptime/context (injected by caller optionally)
 - Database availability check (lazy / lightweight; optional)
 - Predictor calibration & model version info (if predictor present)
 - Metrics wrapper snapshot (prediction/explanation latencies + counters)
 - Performance instrumentation snapshot (timing decorators + cache stats)
 - Latest data freshness artifact summary (worst age + status) if available
 - Ingestion pipeline last run duration samples (instrumented hooks can push)

This module is intentionally defensive: every optional dependency is wrapped in try/except
so that absence of a component never breaks the aggregate status endpoint.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# In‑process ring buffer for ingestion step / total durations.
_INGESTION_DURATION_SAMPLES = []  # list of (ts, step_name, seconds)
_MAX_SAMPLES = 200


def record_ingestion_duration(step_name: str, seconds: float):
    """Record an ingestion step duration (lightweight local buffer).

    This can be called from the structured pipeline controller or ad‑hoc steps.
    """
    try:
        _INGESTION_DURATION_SAMPLES.append((time.time(), step_name, float(seconds)))
        # Trim
        if len(_INGESTION_DURATION_SAMPLES) > _MAX_SAMPLES:
            del _INGESTION_DURATION_SAMPLES[: len(_INGESTION_DURATION_SAMPLES) - _MAX_SAMPLES]
    except Exception:
        pass


def _latest_freshness_summary() -> Optional[Dict[str, Any]]:
    """Return a compact freshness summary if artifacts exist.

    Looks under data/freshness_runs for newest freshness_*.json. Extracts worst age if present.
    """
    try:
        path_dir = os.path.join('data', 'freshness_runs')
        if not os.path.isdir(path_dir):
            return None
        import glob
        files = sorted(glob.glob(os.path.join(path_dir, 'freshness_*.json')), reverse=True)
        if not files:
            return None
        newest = files[0]
        with open(newest, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        # Expect structure with validations list or table entries; stay defensive
        worst_age = None
        status = payload.get('status')
        # Common patterns:
        # 1. payload['tables'][table]['freshness']['age_minutes']
        # 2. payload['validations'][...]['freshness']['age_minutes']
        tables = payload.get('tables') or {}
        for t, info in tables.items():
            age = (info.get('freshness') or {}).get('age_minutes')
            if age is not None:
                worst_age = max(worst_age, age) if worst_age is not None else age
        validations = payload.get('validations') or []
        for v in validations:
            age = (v.get('freshness') or {}).get('age_minutes')
            if age is not None:
                worst_age = max(worst_age, age) if worst_age is not None else age
        return {
            'artifact_file': os.path.basename(newest),
            'status': status,
            'worst_age_minutes': worst_age,
        }
    except Exception:
        return None


def _metrics_snapshot():
    try:
        from metrics import metrics_wrapper
        return metrics_wrapper.snapshot()
    except Exception:
        return None


def _performance_snapshot():
    try:
        from utils import performance_instrumentation as perf
        return perf.snapshot()
    except Exception:
        return None


def _predictor_status(predictor) -> Optional[Dict[str, Any]]:
    try:
        if predictor is None:
            return None
        status = {
            'model_version': getattr(predictor, 'model_version', 'unknown'),
            'calibration_enabled': bool(getattr(predictor, '_calibration_enabled', False)),
            'calibration_loaded': bool(getattr(predictor, '_calibration_loaded', False)),
            'calibration_applied_count': int(getattr(predictor, '_calibration_applied_count', 0)),
            'last_prediction_feature_shape': None,
        }
        try:
            lf = getattr(predictor, '_last_features', None)
            if lf is not None:
                status['last_prediction_feature_shape'] = list(lf.shape)
        except Exception:
            pass
        return status
    except Exception:
        return None


def _database_ping() -> Optional[Dict[str, Any]]:
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        import time as _t
        start = _t.time()
        with db.get_session() as session:
            session.execute("SELECT 1")
        latency = (_t.time() - start) * 1000.0
        return {'status': 'OK', 'latency_ms': round(latency, 2)}
    except Exception as e:
        return {'status': 'ERROR', 'error': str(e)}


def collect_aggregate_status(predictor=None, include_database: bool = True) -> Dict[str, Any]:
    """Collect and return aggregate monitoring data as a dictionary.

    Parameters
    ----------
    predictor : object, optional
        Predictor instance for calibration/model info.
    include_database : bool
        Whether to attempt a database connectivity check.
    """
    now = time.time()
    data: Dict[str, Any] = {
        'timestamp': now,
        'service': 'GoalDiggers Platform',
        'version': '1.0.0',
    }
    if include_database:
        data['database'] = _database_ping()
    data['predictor'] = _predictor_status(predictor)
    data['metrics'] = _metrics_snapshot()
    data['performance'] = _performance_snapshot()
    data['data_freshness'] = _latest_freshness_summary()

    # Ingestion durations (last 10 + summary)
    try:
        recent = _INGESTION_DURATION_SAMPLES[-10:]
        data['ingestion_durations'] = [
            {'ts': ts, 'step': step, 'seconds': secs} for ts, step, secs in recent
        ]
        if recent:
            secs_list = [s for _, _, s in recent]
            data['ingestion_duration_summary'] = {
                'count_last_10': len(recent),
                'avg_seconds_last_10': sum(secs_list)/len(secs_list),
                'max_seconds_last_10': max(secs_list),
            }
    except Exception:
        pass
    return data


__all__ = [
    'collect_aggregate_status',
    'record_ingestion_duration',
]
