"""Asyncio compatibility helpers.

Centralizes safe loop acquisition and time retrieval to avoid deprecation warnings
with asyncio.get_event_loop() under Python 3.11+/pytest-asyncio modern mode.

Usage:
    from utils.asyncio_compat import ensure_loop, loop_time
    ts = loop_time()
"""
from __future__ import annotations

import asyncio
from typing import Optional

__all__ = ["ensure_loop", "loop_time", "get_running_loop_or_none"]


def get_running_loop_or_none() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def ensure_loop() -> asyncio.AbstractEventLoop:
    """Return the running loop or create & set a new event loop.

    This avoids deprecated implicit loop creation and unifies the pattern
    across sync wrappers and utilities.
    """
    loop = get_running_loop_or_none()
    if loop is not None:
        return loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def loop_time() -> float:
    """Safe monotonic loop time with fallback to ensure a loop exists."""
    loop = get_running_loop_or_none()
    if loop is not None:
        return loop.time()
    # create ephemeral loop just for time; we could also use time.perf_counter()
    loop = ensure_loop()
    return loop.time()
