"""Ingestion freshness utilities.

Provides lightweight helpers to surface latest ingestion run recency
and basic row metrics for UI/monitoring without loading heavy deps.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

RUN_DIR_CANDIDATES = [
    "data/ingestion_runs",
    "artifacts/ingestion_runs",
    "ingestion_runs",
]

@dataclass
class IngestionFreshness:
    path: str
    completed_at: Optional[str]
    age_minutes: Optional[float]
    total_steps: Optional[int]
    success_steps: Optional[int]
    success_ratio: Optional[float]
    row_counts: Dict[str, int]


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _find_latest_run_dir() -> Optional[str]:
    latest = None
    latest_mtime = -1.0
    for base in RUN_DIR_CANDIDATES:
        if not os.path.isdir(base):
            continue
        for fname in os.listdir(base):
            if not fname.endswith('.json'):
                continue
            fpath = os.path.join(base, fname)
            try:
                mtime = os.path.getmtime(fpath)
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest = fpath
    return latest


def compute_freshness() -> IngestionFreshness:
    latest_path = _find_latest_run_dir()
    if not latest_path:
        return IngestionFreshness(
            path="",
            completed_at=None,
            age_minutes=None,
            total_steps=None,
            success_steps=None,
            success_ratio=None,
            row_counts={},
        )
    data = _read_json(latest_path) or {}
    summary = data.get("summary") or {}
    steps = data.get("steps") or []
    completed_at = summary.get("completed_at") or summary.get("ended_at")
    age_minutes = None
    if completed_at:
        try:
            dt = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            age_minutes = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 60.0
        except Exception:
            age_minutes = None
    success_steps = sum(1 for s in steps if s.get("status") == "success") if steps else None
    total_steps = len(steps) or None
    ratio = (success_steps / total_steps * 100.0) if success_steps is not None and total_steps else None
    # Row counts (if embedded)
    row_counts = {}
    for s in steps:
        if isinstance(s, dict):
            tbl = s.get("table") or s.get("name")
            rows = s.get("row_count") or s.get("rows")
            if tbl and isinstance(rows, int):
                row_counts[tbl] = rows
    return IngestionFreshness(
        path=latest_path,
        completed_at=completed_at,
        age_minutes=age_minutes,
        total_steps=total_steps,
        success_steps=success_steps,
        success_ratio=ratio,
        row_counts=row_counts,
    )


__all__ = ["compute_freshness", "IngestionFreshness"]
