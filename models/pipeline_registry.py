"""Singleton-ish registry for active ML pipeline instance.

Lightweight global to let runtime snapshot (and other monitors) fetch an
already-instantiated pipeline without forcing import side-effects or
creating a new heavy training object.

Design:
- register_pipeline(pipeline) stores weak reference style (direct ref ok here)
- get_pipeline() returns the instance or None
- Safe no-op if multiple registrations (last wins)
"""
from __future__ import annotations

from typing import Any, Optional

PIPELINE_INSTANCE: Optional[Any] = None

def register_pipeline(pipeline: Any) -> None:
    global PIPELINE_INSTANCE
    try:
        PIPELINE_INSTANCE = pipeline
    except Exception:  # pragma: no cover
        pass

def get_pipeline() -> Optional[Any]:
    return PIPELINE_INSTANCE

__all__ = ["register_pipeline", "get_pipeline"]
