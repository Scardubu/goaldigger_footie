"""Lightweight instrumentation utilities for timing & histogram aggregation.

Avoids external deps; can be extended later for Prometheus/OpenTelemetry.
"""
from __future__ import annotations

import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

_LOCK = threading.RLock()
_HISTOGRAMS: Dict[str, List[float]] = {}
_MAX_SAMPLES_PER_METRIC = 1000


def record_timing(metric: str, value: float):
    if value <= 0:
        return
    with _LOCK:
        bucket = _HISTOGRAMS.setdefault(metric, [])
        bucket.append(value)
        if len(bucket) > _MAX_SAMPLES_PER_METRIC:
            # Trim oldest half to bound memory
            del bucket[: len(bucket) // 2]


def summarize(metric: str) -> Dict[str, Any]:
    with _LOCK:
        values = list(_HISTOGRAMS.get(metric, []))
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None, "avg": None}
    values.sort()
    n = len(values)
    def pct(p: float):
        if n == 0:
            return None
        idx = min(max(int(n * p), 0), n - 1)
        return round(values[idx], 6)
    avg = round(sum(values) / n, 6)
    return {
        "count": n,
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "avg": avg,
    }


def summarize_all() -> Dict[str, Dict[str, Any]]:
    with _LOCK:
        metrics = list(_HISTOGRAMS.keys())
    return {m: summarize(m) for m in metrics}


def time_block(metric: str, tags: Optional[Dict[str, Any]] = None):
    """Context manager style helper (used as decorator factory)."""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                record_timing(metric, duration)
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                record_timing(metric, duration)
        # Preserve async nature
        return async_wrapper if hasattr(func, '__await__') or getattr(func, '__is_coroutine__', False) else sync_wrapper
    return decorator

__all__ = [
    'record_timing', 'summarize', 'summarize_all', 'time_block'
]
