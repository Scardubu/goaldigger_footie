"""Performance & Cache Instrumentation Utilities.

Provides decorators and a lightweight registry to track:
- Function execution timings (rolling window & aggregates)
- Cache hit/miss statistics for simple memoization wrappers

Intended for surfacing on dashboards (status / diagnostics panel) and
optionally emitting to Prometheus via metrics_wrapper snapshot if desired.
"""
from __future__ import annotations

import functools
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Optional, Tuple

# Thread safety for multi-threaded API contexts
_lock = threading.RLock()

_TIMINGS: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=200))
_STATS: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
    'count': 0,
    'total': 0.0,
    'min': None,
    'max': None
})
_CACHE_STATS: Dict[str, Dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0})


def timeit(name: Optional[str] = None):
    """Decorator to time a function and store rolling metrics.

    Args:
        name: Optional override for function identifier.
    """
    def _decorator(func: Callable):
        key = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                with _lock:
                    _TIMINGS[key].append(elapsed)
                    stat = _STS = _STATS[key]
                    stat['count'] += 1
                    stat['total'] += elapsed
                    stat['min'] = elapsed if stat['min'] is None or elapsed < stat['min'] else stat['min']
                    stat['max'] = elapsed if stat['max'] is None or elapsed > stat['max'] else stat['max']
        return _wrapper
    return _decorator


def simple_cache(maxsize: int = 128, stats_key: Optional[str] = None):
    """Simple in-process memoization with hit/miss tracking.

    Not intended to replace functools.lru_cache globally; used where we want
    explicit instrumentation separate from Python's internal cache stats.
    """
    def _decorator(func: Callable):
        cache: Dict[Tuple[Any,...,frozenset], Any] = {}
        key_name = stats_key or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            k = (args, frozenset(kwargs.items()))
            with _lock:
                if k in cache:
                    _CACHE_STATS[key_name]['hits'] += 1
                    return cache[k]
            result = func(*args, **kwargs)
            with _lock:
                if len(cache) >= maxsize:
                    # FIFO eviction: pop first key
                    try:
                        first_key = next(iter(cache))
                        cache.pop(first_key)
                    except Exception:
                        cache.clear()
                cache[k] = result
                _CACHE_STATS[key_name]['misses'] += 1
            return result
        return _wrapper
    return _decorator


def snapshot() -> Dict[str, Any]:
    """Return current performance & cache statistics."""
    with _lock:
        perf = {
            k: {
                'last_samples': list(v)[-5:],
                'avg_ms': (sum(v) / len(v) * 1000.0) if v else None,
                'min_ms': (_STATS[k]['min'] * 1000.0) if _STATS[k]['min'] is not None else None,
                'max_ms': (_STATS[k]['max'] * 1000.0) if _STATS[k]['max'] is not None else None,
                'count': _STATS[k]['count']
            } for k, v in _TIMINGS.items()
        }
        cache = {
            k: {
                'hits': v['hits'],
                'misses': v['misses'],
                'hit_ratio': (v['hits'] / (v['hits'] + v['misses'])) if (v['hits'] + v['misses']) else None
            } for k, v in _CACHE_STATS.items()
        }
        return {'performance': perf, 'cache': cache}

__all__ = ['timeit', 'simple_cache', 'snapshot']
