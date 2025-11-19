"""Alert evaluation utilities for GoalDiggers runtime.

Derives higher-level alert objects from primitive snapshot sections:
  - real_data_coverage: ratio thresholds (error<warn<ok)
  - data_freshness: worst table age from data_pipeline (if provided)
  - cache_health: low hit rate or high miss burst

Each alert object shape:
  {
    'id': str,
    'severity': 'info'|'warning'|'error',
    'message': str,
    'metric': optional raw metric value,
    'thresholds': optional description
  }

Design: Pure + side-effect free. Consumers decide how to render.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _classify_real_data_ratio(ratio: Optional[float]) -> Optional[Dict[str, Any]]:
    if ratio is None:
        return None
    if ratio < 0.05:
        return {
            'id': 'real_data_coverage',
            'severity': 'error',
            'message': f"Real data enrichment critically low ({ratio*100:.1f}%); verify API keys & ingestion.",
            'metric': ratio,
            'thresholds': '<5% error, <25% warn'
        }
    if ratio < 0.25:
        return {
            'id': 'real_data_coverage',
            'severity': 'warning',
            'message': f"Real data enrichment low ({ratio*100:.1f}%); enrichment may be degraded.",
            'metric': ratio,
            'thresholds': '<5% error, <25% warn'
        }
    return {
        'id': 'real_data_coverage',
        'severity': 'info',
        'message': f"Real data enrichment healthy ({ratio*100:.1f}%).",
        'metric': ratio,
        'thresholds': '<5% error, <25% warn'
    }


def _classify_freshness(data_pipeline: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not data_pipeline:
        return None
    # Heuristic extraction: look for freshness fields or max age
    worst_age = None
    if isinstance(data_pipeline, dict):
        for key in ['worst_age_minutes', 'max_table_age_minutes', 'max_age_min']:
            val = data_pipeline.get(key)
            if isinstance(val, (int, float)):
                worst_age = float(val)
                break
        # Some integrators might supply 'tables': [{'age_minutes':..}, ...]
        if worst_age is None and isinstance(data_pipeline.get('tables'), list):
            ages = [t.get('age_minutes') for t in data_pipeline['tables'] if isinstance(t, dict) and isinstance(t.get('age_minutes'), (int, float))]
            if ages:
                worst_age = max(ages)
    if worst_age is None:
        return None
    if worst_age > 180:
        sev = 'error'; msg = f"Data freshness stale ({worst_age:.0f}m); ingestion likely stalled."
    elif worst_age > 90:
        sev = 'warning'; msg = f"Data freshness degrading ({worst_age:.0f}m)."
    else:
        sev = 'info'; msg = f"Data freshness OK ({worst_age:.0f}m)."
    return {
        'id': 'data_freshness',
        'severity': sev,
        'message': msg,
        'metric': worst_age,
        'thresholds': '>' '180m error, >90m warn'
    }


def _classify_cache(cache_metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cache_metrics:
        return None
    hit_rate = cache_metrics.get('hit_rate')
    if isinstance(hit_rate, (int, float)):
        if hit_rate < 0.1:
            return {
                'id': 'cache_health',
                'severity': 'warning',
                'message': f"Cache hit rate low ({hit_rate*100:.1f}%); cold start or ineffective keys.",
                'metric': hit_rate,
                'thresholds': '<10% warn'
            }
        if hit_rate < 0.5:
            return {
                'id': 'cache_health',
                'severity': 'info',
                'message': f"Cache warming ({hit_rate*100:.1f}%).",
                'metric': hit_rate,
                'thresholds': '<10% warn'
            }
        return {
            'id': 'cache_health',
            'severity': 'info',
            'message': f"Cache performing well ({hit_rate*100:.1f}%).",
            'metric': hit_rate,
            'thresholds': '<10% warn'
        }
    return None


def evaluate_alerts(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    predictor = snapshot.get('predictor') or {}
    ratio = predictor.get('real_data_usage_ratio')
    rd_alert = _classify_real_data_ratio(ratio)
    if rd_alert:
        alerts.append(rd_alert)
    freshness_alert = _classify_freshness(snapshot.get('data_pipeline'))
    if freshness_alert:
        alerts.append(freshness_alert)
    cache_alert = _classify_cache(snapshot.get('cache_metrics'))
    if cache_alert:
        alerts.append(cache_alert)
    return alerts

__all__ = ["evaluate_alerts"]
