"""Integrity utilities for model & artifact validation.

Provides SHA256 checksum generation and verification helpers.
Avoids external dependencies; can be extended later with signature support.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional


def file_sha256(path: str, chunk_size: int = 65536) -> Optional[str]:
    """Compute SHA256 hash of a file. Returns hex digest or None if error."""
    if not os.path.exists(path) or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def write_checksum(path: str, checksum: str) -> Optional[str]:
    """Persist checksum sidecar (.sha256.json). Returns sidecar path."""
    sidecar = path + '.sha256.json'
    try:
        with open(sidecar, 'w', encoding='utf-8') as f:
            json.dump({"file": os.path.basename(path), "sha256": checksum}, f, indent=2)
        return sidecar
    except Exception:
        return None

def read_checksum(path: str) -> Optional[str]:
    sidecar = path + '.sha256.json'
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('sha256')
    except Exception:
        return None

def verify_checksum(path: str) -> Dict[str, Any]:
    """Verify checksum for file path against sidecar. Returns result dict."""
    recorded = read_checksum(path)
    current = file_sha256(path)
    status = 'missing' if recorded is None else ('mismatch' if recorded and recorded != current else 'ok')
    return {
        'file': path,
        'recorded': recorded,
        'current': current,
        'status': status
    }

__all__ = [
    'file_sha256', 'write_checksum', 'read_checksum', 'verify_checksum'
]