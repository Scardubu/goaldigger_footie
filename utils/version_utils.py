"""Version utility helpers (pkg_resources replacement).

Provides lightweight helpers to retrieve package versions and resource text
without importing deprecated pkg_resources API.

Usage:
    from utils.version_utils import get_version_safe
    ver = get_version_safe('pandas')
"""
from __future__ import annotations
import importlib
import importlib.metadata as _md
from typing import Optional, Any

__all__ = [
    'get_version_safe',
    'require_min_version',
]

def get_version_safe(distribution: str) -> Optional[str]:
    """Return installed version of a distribution or None if not found."""
    try:
        return _md.version(distribution)
    except _md.PackageNotFoundError:  # pragma: no cover
        return None
    except Exception:  # pragma: no cover
        return None

def require_min_version(distribution: str, minimum: str) -> bool:
    """Check if installed version >= minimum (returns False if not installed)."""
    from packaging.version import Version, InvalidVersion
    ver = get_version_safe(distribution)
    if ver is None:
        return False
    try:
        return Version(ver) >= Version(minimum)
    except InvalidVersion:  # pragma: no cover
        return False

def import_optional(module: str) -> Optional[Any]:
    """Attempt to import a module returning None if unavailable."""
    try:
        return importlib.import_module(module)
    except Exception:  # pragma: no cover
        return None
