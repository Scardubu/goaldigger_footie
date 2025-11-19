"""Utility to schedule data ingestion and report archiving runs.

This script wires unified_launcher ingestion/report commands into a simple
APScheduler-based runner so production deployments can run continuous refreshes
without relying on external cron setup.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

LOGGER = logging.getLogger("data_pipeline_scheduler")
DEFAULT_INGEST_INTERVAL_MINUTES = 60
DEFAULT_REPORT_ARCHIVE_HOUR = 2
DEFAULT_REPORT_ARCHIVE_MINUTE = 15


def _ensure_report_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_command(args: List[str], label: str) -> subprocess.CompletedProcess:
    LOGGER.info("Running %s command: %s", label, " ".join(args))
    start_ts = time.monotonic()
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    duration = time.monotonic() - start_ts
    if result.returncode == 0:
        LOGGER.info("%s completed in %.2fs", label, duration)
    else:
        LOGGER.error("%s failed with exit code %s in %.2fs", label, result.returncode, duration)
        if result.stdout:
            LOGGER.error("%s stdout:\n%s", label, result.stdout.strip())
        if result.stderr:
            LOGGER.error("%s stderr:\n%s", label, result.stderr.strip())
    return result


class DataPipelineScheduler:
    def __init__(
        self,
        python_executable: str,
        project_root: Path,
        ingest_interval_minutes: int,
        ingest_days_back: int,
        ingest_days_ahead: int,
        report_hour: int,
        report_minute: int,
        report_directory: Path,
    ) -> None:
        self.python_executable = python_executable
        self.project_root = project_root
        self.ingest_interval_minutes = ingest_interval_minutes
        self.ingest_days_back = ingest_days_back
        self.ingest_days_ahead = ingest_days_ahead
        self.report_hour = report_hour
        self.report_minute = report_minute
        self.report_directory = _ensure_report_directory(report_directory)
        self.scheduler = BackgroundScheduler(timezone="UTC")

    def _ingest_job(self) -> None:
        args = [
            self.python_executable,
            "unified_launcher.py",
            "ingest",
            "--days-back",
            str(self.ingest_days_back),
            "--days-ahead",
            str(self.ingest_days_ahead),
        ]
        _run_command(args, label="ingest")

    def _archive_report_job(self) -> None:
        args = [
            self.python_executable,
            "unified_launcher.py",
            "report",
            "--output",
            "json",
        ]
        result = _run_command(args, label="report")
        if result.returncode != 0:
            return
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target_file = self.report_directory / f"launcher_report_{timestamp}.json"
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive path
            LOGGER.error("Failed to parse report JSON: %s", exc)
            return
        target_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOGGER.info("Archived launcher report to %s", target_file)
        latest_symlink = self.report_directory / "latest.json"
        try:
            if latest_symlink.exists() or latest_symlink.is_symlink():
                latest_symlink.unlink()
            latest_symlink.symlink_to(target_file.name)
        except OSError:  # pragma: no cover - Windows fallback
            latest_symlink.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def start(self, run_immediately: bool) -> None:
        if run_immediately:
            LOGGER.info("Running initial ingest before scheduling interval job")
            self._ingest_job()

        self.scheduler.add_job(
            self._ingest_job,
            IntervalTrigger(minutes=self.ingest_interval_minutes),
            name="hourly_ingest",
            next_run_time=datetime.now(timezone.utc) if run_immediately else None,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self._archive_report_job,
            CronTrigger(hour=self.report_hour, minute=self.report_minute),
            name="daily_report_archive",
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        LOGGER.info(
            "Scheduler started: ingest every %s min, report archive at %02d:%02d UTC",
            self.ingest_interval_minutes,
            self.report_hour,
            self.report_minute,
        )
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            LOGGER.info("Shutdown signal received; stopping scheduler…")
            self.scheduler.shutdown(wait=False)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule ingestion/report automation")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to invoke unified_launcher (default: current interpreter)",
    )
    parser.add_argument(
        "--project-root",
        default=Path.cwd(),
        type=Path,
        help="Project root containing unified_launcher.py (default: current directory)",
    )
    parser.add_argument(
        "--ingest-interval-minutes",
        type=int,
        default=int(os.getenv("INGEST_INTERVAL_MINUTES", DEFAULT_INGEST_INTERVAL_MINUTES)),
        help="Minutes between ingest runs (default: 60)",
    )
    parser.add_argument(
        "--ingest-days-back",
        type=int,
        default=int(os.getenv("INGEST_DAYS_BACK", "2")),
        help="Days back parameter for ingest (default: 2)",
    )
    parser.add_argument(
        "--ingest-days-ahead",
        type=int,
        default=int(os.getenv("INGEST_DAYS_AHEAD", "7")),
        help="Days ahead parameter for ingest (default: 7)",
    )
    parser.add_argument(
        "--report-hour",
        type=int,
        default=int(os.getenv("REPORT_ARCHIVE_HOUR", DEFAULT_REPORT_ARCHIVE_HOUR)),
        help="UTC hour to archive daily report (default: 2)",
    )
    parser.add_argument(
        "--report-minute",
        type=int,
        default=int(os.getenv("REPORT_ARCHIVE_MINUTE", DEFAULT_REPORT_ARCHIVE_MINUTE)),
        help="UTC minute to archive daily report (default: 15)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(os.getenv("REPORT_ARCHIVE_DIR", "data/reports")),
        help="Directory to store archived reports (default: data/reports)",
    )
    parser.add_argument(
        "--run-initial-ingest",
        action="store_true",
        help="Run a single ingest immediately before scheduling interval job",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=os.getenv("SCHEDULER_LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    project_root = args.project_root.resolve()
    os.chdir(project_root)
    LOGGER.info("Project root set to %s", project_root)
    scheduler = DataPipelineScheduler(
        python_executable=args.python,
        project_root=project_root,
        ingest_interval_minutes=max(1, args.ingest_interval_minutes),
        ingest_days_back=max(0, args.ingest_days_back),
        ingest_days_ahead=max(0, args.ingest_days_ahead),
        report_hour=args.report_hour % 24,
        report_minute=args.report_minute % 60,
        report_directory=args.report_dir,
    )
    scheduler.start(run_immediately=args.run_initial_ingest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
