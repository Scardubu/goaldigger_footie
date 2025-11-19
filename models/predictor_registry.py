"""Registry for active EnhancedRealDataPredictor instance.

Mirrors pipeline_registry to allow monitoring & dashboards to access
predictor state (e.g., calibration status, last feature vector) without
hard coupling imports or creating new predictor objects inadvertently.
"""
from __future__ import annotations

from typing import Any, Optional

PREDICTOR_INSTANCE: Optional[Any] = None

def register_predictor(predictor: Any) -> None:
    global PREDICTOR_INSTANCE
    try:
        PREDICTOR_INSTANCE = predictor
    except Exception:  # pragma: no cover
        pass

def get_predictor() -> Optional[Any]:
    return PREDICTOR_INSTANCE

__all__ = ["register_predictor", "get_predictor"]
