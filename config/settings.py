"""Centralized configuration settings for GoalDiggers platform.

Provides a single source of truth for environment-derived settings with
lightweight validation and defaults. All modules should import from here
instead of reading os.environ directly where practical.

Usage:
from config.settings import settings
print(settings.DB_PATH)

To override, set environment variables before process start.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# Default constants (can be overridden via env)
_DEFAULT_DB = os.path.join("data", "football.db")
_DEFAULT_ENV = "development"

@dataclass(frozen=True)
class Settings:
    ENV: str
    DB_PATH: str
    INGESTION_MODE: str
    LOG_LEVEL: str
    FRESHNESS_RUN_DIR: str
    METRICS_DIR: str
    ENABLE_CALIBRATION: bool
    MIN_CONFIDENCE_THRESHOLD: float
    FEATURE_FLAGS: frozenset
    ENABLE_PERSISTENT_FIXTURE_CACHE: bool
    FIXTURE_CACHE_TTL: int
    REDIS_URL: Optional[str]
    ENABLE_SHAP: bool

    # Convenience computed properties
    @property
    def is_prod(self) -> bool:  # noqa: D401
        return self.ENV.lower().startswith("prod")

    @property
    def is_dev(self) -> bool:
        return not self.is_prod

@lru_cache(maxsize=1)
def load_settings() -> Settings:
    env = os.getenv("APP_ENV", _DEFAULT_ENV)
    db_path = os.getenv("DATA_DB_PATH", _DEFAULT_DB)
    ingestion_mode = os.getenv("INGESTION_MODE", "legacy").lower()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    freshness_dir = os.getenv("FRESHNESS_RUN_DIR", os.path.join("data", "freshness_runs"))
    metrics_dir = os.getenv("METRICS_DIR", os.path.join("data", "metrics"))
    enable_calibration = os.getenv("ENABLE_CALIBRATION", "1").strip() not in ("0", "false", "False")
    min_confidence = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.50"))
    raw_flags = os.getenv("FEATURE_FLAGS", "")
    feature_flags = frozenset(f.strip() for f in raw_flags.split(",") if f.strip())

    return Settings(
        ENV=env,
        DB_PATH=db_path,
        INGESTION_MODE=ingestion_mode,
        LOG_LEVEL=log_level,
        FRESHNESS_RUN_DIR=freshness_dir,
        METRICS_DIR=metrics_dir,
        ENABLE_CALIBRATION=enable_calibration,
        MIN_CONFIDENCE_THRESHOLD=min_confidence,
        FEATURE_FLAGS=feature_flags,
        ENABLE_PERSISTENT_FIXTURE_CACHE=os.getenv("ENABLE_PERSISTENT_FIXTURE_CACHE","0").lower() in ("1","true","yes"),
        FIXTURE_CACHE_TTL=int(os.getenv("FIXTURE_CACHE_TTL","900")),  # 15 min default
        REDIS_URL=os.getenv("REDIS_URL"),
        ENABLE_SHAP=os.getenv("ENABLE_SHAP","0").lower() in ("1","true","yes"),
    )

# Public singleton-like accessor
settings: Settings = load_settings()

__all__ = ["settings", "Settings", "load_settings"]
