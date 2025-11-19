"""Utilities for silencing noisy runtime warnings and logs.

These helpers are intended for command-line validation scripts and launchers
that should present concise, user-friendly output while still allowing
critical errors to surface.
"""
from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable

_SUPPRESSED_LOGGERS: Iterable[str] = (
    "streamlit",
    "streamlit.runtime",
    "streamlit.runtime.caching",
    "streamlit.runtime.caching.cache_data_api",
)

_SUPPRESSED_WARNINGS: Iterable[tuple[str, type[Warning]]] = (
    (r"pkg_resources is deprecated as an API.*", DeprecationWarning),
    (r"No runtime found, using MemoryCacheStorageManager", UserWarning),
)


def apply_runtime_noise_filters() -> None:
    """Apply consistent warning and logging suppression."""
    for message, category in _SUPPRESSED_WARNINGS:
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=category,
        )

    for logger_name in _SUPPRESSED_LOGGERS:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

        if hasattr(logger, "disabled"):
            logger.disabled = True


def silence_pkg_resources_warning() -> None:
    """Suppress pkg_resources deprecation warnings without affecting others."""
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API.*",
        category=DeprecationWarning,
    )
