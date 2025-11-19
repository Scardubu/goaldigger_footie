"""Safe subprocess utilities.

Centralizes subprocess invocation with explicit command lists, timeout,
logging, and redaction of sensitive env vars.
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)

SENSITIVE_KEYS = {"API_KEY", "TOKEN", "SECRET", "PASSWORD"}


def _redact_env(env: dict) -> dict:
    redacted = {}
    for k, v in env.items():
        if any(s in k.upper() for s in SENSITIVE_KEYS):
            redacted[k] = "***"
        else:
            redacted[k] = v
    return redacted


def run_command(cmd: List[str], timeout: Optional[int] = None, env: Optional[dict] = None, check: bool = True):
    if not isinstance(cmd, list):  # enforce list for safety
        raise TypeError("cmd must be a list of arguments")
    display = " ".join(shlex.quote(c) for c in cmd)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    logger.debug("Executing command", extra={"cmd": display, "env": _redact_env(env or {})})
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=merged_env)
    if check and proc.returncode != 0:
        logger.error("Command failed", extra={"cmd": display, "code": proc.returncode, "stderr": proc.stderr[:500]})
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc

__all__ = ["run_command"]
