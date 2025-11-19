"""Lightweight lint runner.

Prefers ruff if installed for speed, falls back to flake8 which is already
in requirements. Exits non‑zero on issues (for CI gating) but can be run
locally with RUFF_FORMAT=github to get annotations.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print("→", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ruff = shutil.which("ruff")
    targets = [str(PROJECT_ROOT / "models"), str(PROJECT_ROOT / "monitoring"), str(PROJECT_ROOT / "scripts"), str(PROJECT_ROOT / "dashboard"), str(PROJECT_ROOT / "startup"), str(PROJECT_ROOT / "tests"), str(PROJECT_ROOT / "app.py")]  # minimal selection
    if ruff:
        # Run ruff (lint + format diff only)
        code = run([ruff, "check", *targets])
        if code != 0:
            return code
        # Optional formatting suggestion (won't fail build if differences exist)
        run([ruff, "format", "--check", *targets])
        return 0
    # Fallback: flake8
    flake8 = shutil.which("flake8")
    if flake8:
        return run([flake8, *targets])
    print("No linter available (install ruff or flake8).", file=sys.stderr)
    return 0  # do not block if tooling unavailable


if __name__ == "__main__":
    raise SystemExit(main())
