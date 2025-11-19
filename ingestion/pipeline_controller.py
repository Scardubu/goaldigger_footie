"""Structured ingestion pipeline controller.

Provides step orchestration with timing, retries, validation hooks, and
JSON artifact emission for observability.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

try:  # optional monitoring aggregator duration capture
    from monitoring.aggregate_status import record_ingestion_duration  # type: ignore
except Exception:  # pragma: no cover
    def record_ingestion_duration(step_name: str, seconds: float):  # type: ignore
        pass

# Optional metrics recorder (best-effort; no hard dependency)
try:
    from metrics.recorder import record_counter, record_gauge, record_metric
except Exception:  # pragma: no cover
    def record_counter(*a, **k):  # type: ignore
        pass
    def record_gauge(*a, **k):  # type: ignore
        pass
    def record_metric(*a, **k):  # type: ignore
        pass

DB_PATH = os.getenv("DATA_DB_PATH", os.path.join("data", "football.db"))
ARTIFACT_DIR = os.path.join("data", "ingestion_runs")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

@dataclass
class IngestionStep:
    name: str
    func: Callable[[], int]  # returns rows affected / processed (or -1 if N/A)
    retries: int = 1
    retry_delay_s: float = 2.0
    # Row count validation
    min_expected_rows: Optional[int] = None
    table: Optional[str] = None
    # Freshness validation
    freshness_timestamp_column: Optional[str] = None  # column holding last update ts/datetime
    freshness_max_age_minutes: Optional[int] = None   # max allowed age of newest record
    # Allow adapter to specify explicit connection pragma tweaks (future use)
    pragmas: Optional[Dict[str, Union[str,int]]] = None

@dataclass
class StepResult:
    name: str
    status: str
    start_ts: float
    end_ts: float
    duration_ms: int
    rows: Optional[int] = None
    error: Optional[str] = None
    retries_used: int = 0
    validation: Dict[str, Any] = field(default_factory=dict)

class PipelineController:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.results: List[StepResult] = []
        self.run_id = int(time.time())

    # --- Validation Helpers ---
    def _connect(self):
        return sqlite3.connect(self.db_path)

    def validate_table_min_rows(self, table: str, min_rows: int) -> Dict[str, Any]:
        out = {"table": table, "min_rows": min_rows, "actual": None, "ok": False}
        if not os.path.exists(self.db_path):
            out["error"] = "db_missing"
            return out
        try:
            with self._connect() as conn:
                cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
            out["actual"] = count
            out["ok"] = count >= min_rows
        except Exception as e:
            out["error"] = str(e)
        return out

    def validate_table_freshness(self, table: str, ts_column: str, max_age_minutes: int) -> Dict[str, Any]:
        """Validate that the most recent timestamp in `table.ts_column` is within max_age_minutes.

        Supports numeric epoch seconds or ISO8601-like text; falls back gracefully.
        """
        out: Dict[str, Any] = {
            "table": table,
            "ts_column": ts_column,
            "max_age_minutes": max_age_minutes,
            "latest": None,
            "age_minutes": None,
            "ok": False,
        }
        if not os.path.exists(self.db_path):
            out["error"] = "db_missing"
            return out
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    f"SELECT {ts_column} FROM {table} WHERE {ts_column} IS NOT NULL ORDER BY {ts_column} DESC LIMIT 1"
                )
                row = cur.fetchone()
            if not row:
                out["error"] = "no_rows_with_timestamp"
                return out
            latest = row[0]
            out["latest"] = latest
            now = time.time()
            # Attempt parsing
            ts_val: Optional[float] = None
            if isinstance(latest, (int, float)):
                # Heuristic: if it's far in the future in seconds treat as ms
                if latest > 1e12:  # likely ms
                    ts_val = latest / 1000.0
                elif latest > 1e10:  # maybe ms but smaller? still divide
                    ts_val = latest / 1000.0
                else:
                    ts_val = float(latest)
            elif isinstance(latest, str):
                # Try common formats
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                    try:
                        ts_val = time.mktime(time.strptime(latest[:19], fmt))
                        break
                    except Exception:  # noqa
                        continue
            if ts_val is None:
                out["error"] = "unparseable_timestamp"
                return out
            age_min = (now - ts_val) / 60.0
            out["age_minutes"] = round(age_min, 2)
            out["ok"] = age_min <= max_age_minutes
        except Exception as e:  # noqa
            out["error"] = str(e)
        return out

    # --- Execution ---
    def run(self, steps: List[IngestionStep]) -> None:
        run_start = time.time()
        for step in steps:
            start = time.time()
            attempt = 0
            rows = None
            status = "success"
            error_msg = None
            while True:
                try:
                    rows = step.func()
                    break
                except Exception as e:  # noqa
                    attempt += 1
                    error_msg = str(e)
                    if attempt > step.retries:
                        status = "failed"
                        break
                    time.sleep(step.retry_delay_s)
            end = time.time()
            validation: Dict[str, Any] = {}
            if status == "success":
                # Row count validation
                if step.table and step.min_expected_rows is not None:
                    row_val = self.validate_table_min_rows(step.table, step.min_expected_rows)
                    validation["row_count"] = row_val
                    if not row_val.get("ok"):
                        status = "validation_failed"
                # Freshness validation
                if (
                    step.table
                    and step.freshness_timestamp_column
                    and step.freshness_max_age_minutes is not None
                ):
                    fresh_val = self.validate_table_freshness(
                        step.table,
                        step.freshness_timestamp_column,
                        step.freshness_max_age_minutes,
                    )
                    validation["freshness"] = fresh_val
                    if not fresh_val.get("ok") and status == "success":
                        status = "freshness_failed"
            duration_ms = int((end - start) * 1000)
            self.results.append(
                StepResult(
                    name=step.name,
                    status=status,
                    start_ts=start,
                    end_ts=end,
                    duration_ms=duration_ms,
                    rows=rows,
                    error=error_msg,
                    retries_used=attempt,
                    validation=validation,
                )
            )
            # Record ingestion duration in aggregate monitoring (seconds)
            try:
                record_ingestion_duration(step.name, duration_ms / 1000.0)
            except Exception:
                pass
            # Metrics emission (non-fatal if recorder missing)
            try:
                record_counter("ingestion.step.total", meta={"step": step.name})
                record_counter(f"ingestion.step.status.{status}", meta={"step": step.name})
                record_metric("ingestion.step.duration_ms", duration_ms, mtype="timing", meta={"step": step.name})
                if rows is not None and rows >= 0:
                    record_gauge("ingestion.step.rows", rows, meta={"step": step.name})
                rv = validation.get("row_count")
                if rv and rv.get("actual") is not None:
                    record_gauge("ingestion.validation.row_count", rv.get("actual", 0), meta={"table": rv.get("table"), "step": step.name})
                fv = validation.get("freshness")
                if fv and fv.get("age_minutes") is not None:
                    record_gauge("ingestion.validation.freshness_age_minutes", fv.get("age_minutes"), meta={"table": fv.get("table"), "step": step.name})
                    if not fv.get("ok"):
                        record_counter("ingestion.validation.freshness_fail", 1, meta={"table": fv.get("table"), "step": step.name})
            except Exception:
                pass
        self._emit_artifact()
        run_end = time.time()
        # Overall run metrics
        try:
            total_ms = int((run_end - run_start) * 1000)
            record_metric("ingestion.run.duration_ms", total_ms, mtype="timing", meta={"run_id": self.run_id})
            success = sum(1 for r in self.results if r.status == "success")
            ratio = success / max(len(self.results), 1)
            record_gauge("ingestion.run.success_ratio", ratio, meta={"run_id": self.run_id})
        except Exception:
            pass

    # --- Artifact ---
    def _emit_artifact(self) -> None:
        artifact = {
            "run_id": self.run_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "db_path": self.db_path,
            "steps": [r.__dict__ for r in self.results],
        }
        path = os.path.join(ARTIFACT_DIR, f"run_{self.run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)

# --- Utility factory wrappers to adapt existing script files ---

def script_adapter(script_path: str) -> Callable[[], int]:
    """Return a callable that executes a script file and returns -1.
    Uses runpy semantics so we avoid spawning new processes for speed.
    Falls back to subprocess if import-style loading fails.
    """
    def _runner() -> int:
        if not os.path.exists(script_path):
            raise FileNotFoundError(script_path)
        # Try import-based execution for pure Python
        module_name = script_path.replace(os.sep, ".").rstrip(".py")
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore
        else:
            # Fallback: exec in file context
            with open(script_path, "r", encoding="utf-8") as f:
                code = f.read()
            exec(compile(code, script_path, "exec"), {"__name__": "__main__"})
        return -1
    return _runner

__all__ = [
    "IngestionStep", "PipelineController", "script_adapter"
]
