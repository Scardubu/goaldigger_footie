"""Runtime diagnostics utilities.

Provides a single entrypoint `report_startup_environment` that gathers and logs structured
information about the running environment to aid production readiness and troubleshooting.

Focus areas:
- Python & platform details
- Key dependency versions (numpy, pandas, sqlalchemy, fastapi, streamlit if present)
- Application configuration highlights (selected keys only to avoid secrets)
- Database connectivity sanity check (optional, non-fatal)
- Model/predictor presence & version (if importable)
- Logging subsystem summary (root level, handler types)
- Safety: Never raises; returns dict; logs a concise summary line + debug detail

Design principles:
- Pure function (no global state mutation besides logging)
- Resilient to partial failures; each probe isolated
- Minimal import side-effects: perform imports inside probe functions
- Quick execution (<200ms typical) — avoid heavy computations
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("goaldiggers.diagnostics")

SAFE_CONFIG_KEYS = {
    "app.version",
    "environment.name",
    "logging.level",
    "data.storage_backend",
}


@dataclass
class DependencyVersion:
    name: str
    version: Optional[str]
    available: bool


@dataclass
class DiagnosticsReport:
    generated_at: float
    python_version: str
    platform: str
    executable: str
    cwd: str
    dependencies: List[DependencyVersion]
    config: Dict[str, Any]
    db: Dict[str, Any]
    model: Dict[str, Any]
    logging: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert dataclasses in list
        d["dependencies"] = [asdict(dep) for dep in self.dependencies]
        return d


def _probe_dependency(name: str) -> DependencyVersion:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        return DependencyVersion(name=name, version=version, available=True)
    except Exception:  # noqa: broad-except
        return DependencyVersion(name=name, version=None, available=False)


def _gather_dependencies() -> List[DependencyVersion]:
    core = ["numpy", "pandas", "sqlalchemy", "fastapi", "streamlit"]
    return [_probe_dependency(m) for m in core]


def _gather_config() -> Dict[str, Any]:
    try:
        from utils.config import Config  # type: ignore
    except Exception:  # noqa: broad-except
        return {"status": "unavailable"}
    cfg: Dict[str, Any] = {}
    for key in SAFE_CONFIG_KEYS:
        try:
            # Assuming Config.get(key, default) exists
            value = Config.get(key, "<unset>")  # type: ignore[attr-defined]
            cfg[key] = value
        except Exception:  # noqa: broad-except
            cfg[key] = "<error>"
    return cfg


def _check_db() -> Dict[str, Any]:
    # Lightweight connectivity import; avoid initiating full engine creation if heavy
    info: Dict[str, Any] = {"reachable": False, "error": None}
    try:
        from database import db_manager  # type: ignore
    except Exception as e:  # noqa: broad-except
        info["error"] = f"import_failed:{e.__class__.__name__}"
        return info
    try:
        if hasattr(db_manager, "get_engine"):
            engine = db_manager.get_engine()
            conn = engine.connect()
            conn.close()
            info["reachable"] = True
        else:
            info["error"] = "no_get_engine"
    except Exception as e:  # noqa: broad-except
        info["error"] = f"connect_failed:{e.__class__.__name__}"
    return info


def _check_model() -> Dict[str, Any]:
    info: Dict[str, Any] = {"available": False, "version": None}
    candidates = [
        ("models.enhanced_real_data_predictor", "EnhancedRealDataPredictor"),
        ("models.real_data_predictor", "RealDataPredictor"),
    }
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            predictor = None
            try:
                predictor = cls()  # type: ignore[call-arg]
            except Exception:
                # Constructor may require params; skip instantiation
                pass
            version = getattr(predictor, "model_version", None) if predictor else getattr(cls, "model_version", None)
            info.update({"available": True, "version": version, "class": f"{mod_name}.{cls_name}"})
            return info
        except Exception:
            continue
    return info


def _logging_summary() -> Dict[str, Any]:
    root = logging.getLogger()
    return {
        "root_level": logging.getLevelName(root.level),
        "handler_count": len(root.handlers),
        "handlers": [h.__class__.__name__ for h in root.handlers],
        "goaldiggers_level": logging.getLogger("goaldiggers").level,
    }


def report_startup_environment(log_json: bool = True, level: int = logging.INFO) -> DiagnosticsReport:
    """Collect and log a structured diagnostics snapshot.

    Args:
        log_json: If True emits the full report as a single JSON log line at DEBUG.
        level: Level at which to log the human concise summary (INFO default).
    Returns:
        DiagnosticsReport object (also logged).
    """
    start = time.time()
    deps = _gather_dependencies()
    report = DiagnosticsReport(
        generated_at=start,
        python_version=sys.version.replace("\n", " "),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        executable=sys.executable,
        cwd=os.getcwd(),
        dependencies=deps,
        config=_gather_config(),
        db=_check_db(),
        model=_check_model(),
        logging=_logging_summary(),
    )
    duration_ms = int((time.time() - start) * 1000)
    healthy_flags = [
        report.db.get("reachable"),
        any(d.name == "numpy" and d.available for d in deps),
    ]
    summary = (
        f"Diagnostics: py={report.python_version.split()[0]} platform={report.platform} "
        f"deps={sum(1 for d in deps if d.available)}/{len(deps)} "
        f"db={'ok' if report.db.get('reachable') else 'fail'} "
        f"model={'yes' if report.model.get('available') else 'no'} "
        f"handlers={report.logging['handler_count']} "
        f"time={duration_ms}ms"
    )
    logger.log(level, summary)
    if log_json:
        try:
            logger.debug("diagnostics_report=%s", json.dumps(report.to_dict(), sort_keys=True))
        except Exception:  # noqa: broad-except
            pass
    return report

__all__ = [
    "DependencyVersion",
    "DiagnosticsReport",
    "report_startup_environment",
]
