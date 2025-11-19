"""Prediction Metrics Logger (JSON Lines)

Lightweight, append-only telemetry for model predictions.

Writes newline-delimited JSON (NDJSON) events to:
  data/metrics/predictions.ndjson

Each event shape (fields may evolve additively):
  {
    "ts": float epoch seconds,
    "latency_ms": float|None,
    "real": bool,
    "total": int,                # cumulative total predictions
    "real_total": int,           # cumulative real-data enhanced predictions
    "real_ratio": float|None,    # ratio at time of event
    "model_version": str
  }

Design Goals:
  * Non-blocking best effort: failures never raise.
  * Minimal overhead: open->append->flush->close per event.
  * Simple rotation: if file exceeds MAX_BYTES, rename to .1 (overwriting existing) and start fresh.
  * Reading helper returns recent events for sparkline / trend charts.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

METRICS_DIR = os.path.join('data', 'metrics')
PREDICTIONS_FILE = os.path.join(METRICS_DIR, 'predictions.ndjson')
MAX_BYTES = 5 * 1024 * 1024  # ~5MB simple rotation threshold


def _ensure_dir() -> None:
    try:
        os.makedirs(METRICS_DIR, exist_ok=True)
    except Exception:
        pass


def append_prediction_event(event: Dict[str, Any]) -> None:
    """Append a single prediction telemetry event (best-effort)."""
    try:
        _ensure_dir()
        # Simple rotation
        try:
            if os.path.exists(PREDICTIONS_FILE) and os.path.getsize(PREDICTIONS_FILE) > MAX_BYTES:
                rotated = PREDICTIONS_FILE + '.1'
                try:
                    if os.path.exists(rotated):
                        os.remove(rotated)
                except Exception:
                    pass
                try:
                    os.replace(PREDICTIONS_FILE, rotated)
                except Exception:
                    pass
        except Exception:
            pass
        event.setdefault('ts', time.time())
        with open(PREDICTIONS_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, separators=(',', ':'), default=str) + '\n')
    except Exception:
        # Swallow all errors to avoid impacting prediction path
        pass


def read_recent_events(limit: int = 500, minutes: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read recent prediction events (best-effort).

    Args:
        limit: Max number of most-recent lines to return (after filtering).
        minutes: If provided, filter to only events newer than now - minutes.
    """
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return []
        # Read all lines (file expected to be small due to rotation). For very large files a tail
        # optimization could be added later.
        with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        events: List[Dict[str, Any]] = []
        cutoff = None
        if minutes is not None:
            cutoff = time.time() - (minutes * 60)
        for line in reversed(lines):  # iterate from newest
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if cutoff is not None and isinstance(obj.get('ts'), (int, float)) and obj['ts'] < cutoff:
                    # Since we are iterating newest-first, we can break once older than cutoff
                    break
                events.append(obj)
                if len(events) >= limit:
                    break
            except Exception:
                continue
        return list(reversed(events))  # chronological order
    except Exception:
        return []


__all__ = [
    'append_prediction_event',
    'read_recent_events',
    'PREDICTIONS_FILE'
]
