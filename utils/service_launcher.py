"""Utilities for ensuring GoalDiggers background services are running.

This helper centralizes auto-start logic for the core runtime processes that the
health check and production launcher depend on:

- Enhanced startup orchestrator (scripts/enhanced_startup.py)
- FastAPI service (uvicorn fast_api_server:app)
- Realtime SSE server (uvicorn services.realtime.sse_server:app)
- Streamlit production dashboard (dashboard/enhanced_production_homepage.py)

The launcher keeps things lightweight: it only attempts to spawn a process if a
health probe indicates the service is unavailable. PIDs are persisted so that we
avoid launching duplicate instances across repeated checks.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import requests

try:  # psutil offers reliable process inspection if available
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore

LOGGER = logging.getLogger(__name__)

DEFAULT_API_PORT = 5000
DEFAULT_DASHBOARD_PORT = 8501
DEFAULT_SSE_PORT = 8079

PID_REGISTRY = Path("logs/service_pids.json")
SERVICE_LOG_DIR = Path("logs")
SERVICE_LOG_DIR.mkdir(exist_ok=True)


class ServiceLauncher:
    """Ensure GoalDiggers services are running, auto-starting if required."""

    def __init__(
        self,
        auto_start: Optional[bool] = None,
        retries: int = 20,
        delay_seconds: float = 1.5,
    ) -> None:
        self.auto_start = (
            auto_start
            if auto_start is not None
            else os.getenv("GOALDIGGERS_AUTO_START_SERVICES", "1").lower()
            not in {"0", "false", "off"}
        )
        self.retries = retries
        self.delay_seconds = delay_seconds
        self._service_registry = self._build_registry()
        self._pid_cache: Dict[str, int] = self._load_pid_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_service(self, name: str) -> Dict[str, Optional[str]]:
        """Ensure the named service is running, optionally auto-starting it.

        Returns a dict containing keys:
            running: "true" or "false" as strings for easy logging compat
            auto_started: "true" if a start attempt was triggered
            message: optional status message
        """
        if name not in self._service_registry:
            raise ValueError(f"Unknown service '{name}'")

        check_fn = self._service_registry[name]["check"]
        if check_fn():
            return {"running": "true", "auto_started": "false", "message": "Service already active"}

        if not self.auto_start:
            return {"running": "false", "auto_started": "false", "message": "Auto-start disabled"}

        start_fn = self._service_registry[name]["start"]
        started = start_fn()
        if not started:
            return {"running": "false", "auto_started": "true", "message": "Launch attempt failed"}

        if self._wait_for_service(name):
            return {"running": "true", "auto_started": "true", "message": "Service started"}
        return {"running": "false", "auto_started": "true", "message": "Service did not become healthy"}

    def check_only(self, name: str) -> bool:
        """Return True if the service appears healthy without attempting to start."""
        if name not in self._service_registry:
            raise ValueError(f"Unknown service '{name}'")
        return self._service_registry[name]["check"]()

    # ------------------------------------------------------------------
    # Registry + health helpers
    # ------------------------------------------------------------------
    def _build_registry(self) -> Dict[str, Dict[str, Callable[[], bool]]]:
        return {
            "enhanced_startup": {
                "check": self._check_enhanced_startup,
                "start": self._start_enhanced_startup,
            },
            "api": {
                "check": lambda: self._check_http(f"http://127.0.0.1:{DEFAULT_API_PORT}/api/v1/health"),
                "start": self._start_api_service,
            },
            "dashboard": {
                "check": self._check_dashboard,
                "start": self._start_dashboard,
            },
            "sse": {
                "check": lambda: self._check_http(f"http://127.0.0.1:{DEFAULT_SSE_PORT}/sse-health", expects_json=True),
                "start": self._start_sse_service,
            },
        }

    def _wait_for_service(self, name: str) -> bool:
        check_fn = self._service_registry[name]["check"]
        for _ in range(self.retries):
            if check_fn():
                return True
            time.sleep(self.delay_seconds)
        return False

    # ------------------------------------------------------------------
    # Start routines
    # ------------------------------------------------------------------
    def _start_enhanced_startup(self) -> bool:
        cmd = [sys.executable, "scripts/enhanced_startup.py"]
        return self._spawn_process("enhanced_startup", cmd)

    def _start_api_service(self) -> bool:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "fast_api_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(DEFAULT_API_PORT),
            "--log-level",
            "warning",
        ]
        return self._spawn_process("api", cmd)

    def _start_sse_service(self) -> bool:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "services.realtime.sse_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(DEFAULT_SSE_PORT),
            "--log-level",
            "warning",
        ]
        return self._spawn_process("sse", cmd)

    def _start_dashboard(self) -> bool:
        dashboard_path = Path("dashboard/enhanced_production_homepage.py")
        if not dashboard_path.exists():
            LOGGER.error("Dashboard entry point %s not found", dashboard_path)
            return False
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            f"--server.port={DEFAULT_DASHBOARD_PORT}",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ]
        env = os.environ.copy()
        env.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
        env.setdefault("GOALDIGGERS_MODE", "production")
        return self._spawn_process("dashboard", cmd, env=env)

    # ------------------------------------------------------------------
    # Health probes
    # ------------------------------------------------------------------
    def _check_http(self, url: str, expects_json: bool = False) -> bool:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code != 200:
                return False
            if expects_json:
                response.json()
            return True
        except Exception:
            return False

    def _check_dashboard(self) -> bool:
        if self._check_http(f"http://127.0.0.1:{DEFAULT_DASHBOARD_PORT}/_stcore/health"):
            return True
        return self._check_http(f"http://127.0.0.1:{DEFAULT_DASHBOARD_PORT}/")

    def _check_enhanced_startup(self) -> bool:
        if psutil is not None:
            try:
                for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                    cmdline = proc.info.get("cmdline") or []
                    if any("enhanced_startup.py" in part for part in cmdline):
                        self._record_pid("enhanced_startup", proc.pid)
                        return True
            except Exception:  # pragma: no cover
                LOGGER.debug("psutil process iteration failed; falling back to log check")
        log_path = Path("logs/enhanced_startup.log")
        if log_path.exists():
            try:
                if time.time() - log_path.stat().st_mtime < 300:
                    return True
            except OSError:
                pass
        return False

    # ------------------------------------------------------------------
    # Process helpers
    # ------------------------------------------------------------------
    def _spawn_process(self, name: str, cmd: list[str], env: Optional[Dict[str, str]] = None) -> bool:
        try:
            log_path = SERVICE_LOG_DIR / f"{name}_service.log"
            log_file = open(log_path, "a", encoding="utf-8")
            LOGGER.info("Starting %s service: %s", name, " ".join(cmd))
            creationflags = 0
            if platform.system() == "Windows":  # avoid opening extra console windows
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                env=env,
                start_new_session=True,
                creationflags=creationflags,
            )
            self._record_pid(name, proc.pid)
            return True
        except Exception as exc:
            LOGGER.error("Failed to start %s service: %s", name, exc)
            return False

    # ------------------------------------------------------------------
    # PID registry utilities
    # ------------------------------------------------------------------
    def _load_pid_registry(self) -> Dict[str, int]:
        if not PID_REGISTRY.exists():
            return {}
        try:
            with open(PID_REGISTRY, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return {k: int(v) for k, v in data.items() if isinstance(v, int)}
        except Exception:
            return {}

    def _record_pid(self, name: str, pid: int) -> None:
        self._pid_cache[name] = pid
        try:
            with open(PID_REGISTRY, "w", encoding="utf-8") as fh:
                json.dump(self._pid_cache, fh, indent=2)
        except Exception:
            pass


__all__ = ["ServiceLauncher"]
