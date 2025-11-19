"""CI health gate script for GoalDiggers platform.

Runs the unified launcher health command and optionally persists the JSON
snapshot so CI pipelines can archive it as an artifact.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger("ci_health_gate")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified health check and enforce PASS result")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to invoke unified_launcher (default: current interpreter)",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing unified_launcher.py (default: current directory)",
    )
    parser.add_argument(
        "--with-prediction",
        action="store_true",
        help="Run health check with pre-flight sample prediction",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write parsed health snapshot JSON",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Treat warnings printed to stderr as failures",
    )
    return parser.parse_args(argv)


def _extract_snapshot(stdout: str) -> tuple[Optional[dict], str]:
    """Split stdout into JSON snapshot and trailing summary text."""
    if "Result:" not in stdout:
        return None, stdout
    json_blob, _, remainder = stdout.partition("Result:")
    start_index = json_blob.find("{")
    end_index = json_blob.rfind("}")
    if start_index == -1 or end_index == -1:
        LOGGER.warning("Unable to locate JSON payload in health output")
        return None, stdout
    try:
        payload = json.loads(json_blob[start_index : end_index + 1])
        return payload, remainder
    except json.JSONDecodeError:
        LOGGER.warning("Unable to parse health snapshot JSON; returning raw output")
        return None, stdout


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=os.getenv("CI_HEALTH_LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    project_root = args.project_root.resolve()
    os.chdir(project_root)

    cmd = [args.python, "unified_launcher.py", "health"]
    if args.with_prediction:
        cmd.append("--with-prediction")

    LOGGER.info("Running unified health command: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    snapshot, remainder = _extract_snapshot(stdout)
    if snapshot is not None and args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        LOGGER.info("Health snapshot JSON archived to %s", args.json_output)

    if result.returncode != 0:
        LOGGER.error("Health command exited with %s", result.returncode)
        return result.returncode

    if "Result: FAIL" in stdout or "Result: FAIL" in remainder:
        LOGGER.error("Health command reported FAIL state")
        return 1

    if args.fail_on_warn and stderr:
        LOGGER.error("Warnings detected in stderr while fail-on-warn enabled")
        return 2

    LOGGER.info("Health command passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
