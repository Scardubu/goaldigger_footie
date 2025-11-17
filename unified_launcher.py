#!/usr/bin/env python3
"""Unified GoalDiggers Platform Launcher

Single authoritative entry point consolidating previous variants:
- streamlined_production_launcher.py
- production_launcher.py
- enhanced_production_launcher.py
- ultimate_production_launcher.py

Usage examples:
  python unified_launcher.py dashboard            # launch default/enhanced dashboard
  python unified_launcher.py api                  # start FastAPI service only
  python unified_launcher.py all                  # dashboard + optional services
  python unified_launcher.py health               # emit runtime health snapshot & exit
  python unified_launcher.py ingest --days-back 2 --days-ahead 7
  python unified_launcher.py train                # (placeholder if future model training)

Exit codes: 0 success / 1 failure.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Reduce noisy third-party deprecation warnings surfaced during dashboard launch
warnings.filterwarnings(
    "once",
    message="pkg_resources is deprecated as an API",
    category=DeprecationWarning,
)
from utils.runtime_noise_filters import apply_runtime_noise_filters

apply_runtime_noise_filters()
from pathlib import Path

# Lazy imports inside functions to keep launcher light

LAUNCHER_VERSION = "1.1.0"
LOG = logging.getLogger("unified_launcher")

DEFAULT_DASHBOARD_PORT = 8501
DEFAULT_API_PORT = 5000
DEFAULT_SSE_PORT = 8079

BASIC_DASHBOARD_CANDIDATES = [
    'dashboard/enhanced_production_homepage.py',
    'enhanced_app.py',
    'main.py',
    'app.py'
]

def service_meta(name: str, port: int):
    if name == 'sse':
        return {
            'command': lambda py: f"{py} -m uvicorn services.realtime.sse_server:app --host 0.0.0.0 --port {port} --log-level warning",
            'port': port,
            'health': f'http://127.0.0.1:{port}/health'
        }
    if name == 'api':
        return {
            'command': lambda py: f"{py} -m uvicorn fast_api_server:app --host 0.0.0.0 --port {port} --log-level warning",
            'port': port,
            'health': f'http://127.0.0.1:{port}/health'
        }
    raise ValueError(f"Unknown service {name}")

OPTIONAL_SERVICES = {
    'sse': service_meta('sse', DEFAULT_SSE_PORT),
    'api': service_meta('api', DEFAULT_API_PORT)
}

REQUIRED_ENV_VARS: dict[str, str] = {
    # Core external data sources (warn if missing to highlight reduced freshness)
    'FOOTBALL_DATA_API_KEY': 'API key for football-data.org (improves match & standings freshness)',
}

OPTIONAL_ENV_VARS: dict[str, str] = {
    # Feature toggles / environment controls (auto-defaulted elsewhere)
    'GOALDIGGERS_MODE': 'Runtime mode (production|staging|dev) - auto-set if absent',
    # Future secrets placeholders (documented early for ops visibility)
    'UNDERSTAT_API_KEY': 'Reserved for potential Understat enhanced access (optional)',
}

SECRET_ENV_HINTS = ["KEY", "SECRET", "TOKEN", "PASSWORD", "PASS", "API_", "AUTH"]


def configure_logging():
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def mask_value(k: str, v: str) -> str:
    if not isinstance(v, str):
        return str(v)
    if any(hint in k.upper() for hint in SECRET_ENV_HINTS):
        if len(v) <= 6:
            return "***"
        return v[:3] + "***" + v[-2:]
    return v


def validate_environment() -> bool:
    ok = True
    for k, desc in REQUIRED_ENV_VARS.items():
        if k not in os.environ:
            LOG.warning("Missing env var %s (%s)", k, desc)
            ok = False
    for k, desc in OPTIONAL_ENV_VARS.items():
        if k not in os.environ:
            LOG.info("Optional env var %s not set (%s)", k, desc)
    # Log a few non-secret envs for diagnostics
    diagnostic = {k: mask_value(k, os.environ[k]) for k in os.environ if k.startswith('GOALDIGGERS_')}
    if diagnostic:
        LOG.info("Environment (masked): %s", diagnostic)
    return ok


def base_env_setup(enhanced: bool = True):
    os.environ.setdefault('GOALDIGGERS_MODE', 'production')
    os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
    os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
    if enhanced:
        os.environ.setdefault('GOALDIGGERS_UI_MODE', 'enhanced')
        os.environ.setdefault('GOALDIGGERS_TEAM_FLAGS', 'true')
        os.environ.setdefault('GOALDIGGERS_VISUAL_ENHANCEMENTS', 'true')
    # Quiet noisy libs
    for lib in ['urllib3', 'matplotlib', 'PIL', 'requests', 'asyncio']:
        logging.getLogger(lib).setLevel(logging.WARNING)
    LOG.info("Base environment configured (enhanced=%s)", enhanced)
    # Optional Prometheus metrics endpoint toggle env defaults
    os.environ.setdefault('ENABLE_PROMETHEUS', '0')
    os.environ.setdefault('PROMETHEUS_PORT', '9108')

_PROM_SERVER_STARTED = False

def maybe_start_prometheus():
    global _PROM_SERVER_STARTED
    if _PROM_SERVER_STARTED:
        return
    try:
        if os.getenv('ENABLE_PROMETHEUS','0') not in ('1','true','TRUE','yes','on'):
            return
        port = int(os.getenv('PROMETHEUS_PORT','9108'))
        from prometheus_client import start_http_server  # type: ignore
        start_http_server(port)
        LOG.info("Prometheus metrics server started on :%s", port)
        _PROM_SERVER_STARTED = True
    except Exception as e:  # pragma: no cover
        LOG.warning("Prometheus start failed: %s", e)


def initialize_runtime_eager():
    try:
        from startup.initialize_runtime import initialize_runtime
        summary = initialize_runtime(eager_warm=False)
        LOG.info("Runtime init summary: %s", {k: summary.get(k) for k in ['pipeline','predictor','pipeline_created','predictor_created']})
    except Exception as e:
        LOG.warning("Runtime initialization skipped: %s", e)


def pick_dashboard() -> str | None:
    root = Path(__file__).parent
    for candidate in BASIC_DASHBOARD_CANDIDATES:
        if (root / candidate).exists():
            return candidate
    return None


def launch_dashboard(port: int = DEFAULT_DASHBOARD_PORT, block: bool = True) -> int:
    dashboard = pick_dashboard()
    if not dashboard:
        LOG.error("No dashboard file found among candidates: %s", BASIC_DASHBOARD_CANDIDATES)
        return 1
    LOG.info("Using dashboard: %s", dashboard)

    try:
        import streamlit.web.cli as stcli
    except ImportError:  # pragma: no cover
        import streamlit.cli as stcli  # type: ignore

    sys.argv = [
        'streamlit', 'run', dashboard,
        f'--server.port={port}',
        '--server.address=0.0.0.0',
        '--server.headless=true',
        '--browser.gatherUsageStats=false'
    ]
    LOG.info("Launching Streamlit dashboard on :%s", port)
    stcli.main()
    return 0


def start_optional_services(which: list[str], api_port: int = DEFAULT_API_PORT, sse_port: int = DEFAULT_SSE_PORT) -> None:
    if not which:
        return
    import time

    import requests

    # Rebuild meta for ports if overridden
    dynamic_meta: dict[str, dict] = {}
    for name in which:
        if name == 'api':
            dynamic_meta[name] = service_meta('api', api_port)
        elif name == 'sse':
            dynamic_meta[name] = service_meta('sse', sse_port)
        else:
            LOG.warning("Unknown service requested: %s", name)
            continue
    for name, meta in dynamic_meta.items():
        if not meta:
            LOG.warning("Unknown service requested: %s", name)
            continue
        # Check already running
        try:
            r = requests.get(meta['health'], timeout=2)
            if r.status_code == 200:
                LOG.info("Service %s already running (%s)", name, meta['health'])
                continue
        except Exception:
            pass
        cmd = meta['command'](sys.executable)
        LOG.info("Starting service %s: %s", name, cmd)
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Wait briefly for health
        for _ in range(15):
            try:
                r = requests.get(meta['health'], timeout=1)
                if r.status_code == 200:
                    LOG.info("Service %s healthy", name)
                    break
            except Exception:
                pass
            time.sleep(1)


PRE_HEALTH_ERRORS: list[str] = []  # Collected errors before snapshot generation


def run_health_snapshot() -> int:
    try:
        from monitoring.runtime_snapshot import get_runtime_snapshot
        snap = get_runtime_snapshot()
        # Merge any pre-health errors (e.g., sample prediction failure) into errors list
        if PRE_HEALTH_ERRORS:
            existing = snap.get('errors') or []
            # Deduplicate while preserving order
            combined = existing + [e for e in PRE_HEALTH_ERRORS if e not in existing]
            snap['errors'] = combined
        # Print full JSON (avoid slicing that can corrupt structure)
        print(json.dumps(snap, indent=2))
        # Basic success heuristic reused
        data_status = (snap.get('data_pipeline') or {}).get('status')
        ok = all(snap.get(k) for k in ['metrics','data_pipeline','model_pipeline','predictor']) and data_status in ('healthy','degraded')
        print("Result:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    except Exception as e:
        LOG.error("Health snapshot failed: %s", e)
        return 1


def run_ingest(days_back: int, days_ahead: int) -> int:
    try:
        from ingestion.etl_pipeline import ingest_from_sources
        ingest_from_sources(days_back=days_back, days_ahead=days_ahead)
        LOG.info("Ingestion complete (back=%s ahead=%s)", days_back, days_ahead)
        return 0
    except Exception as e:
        LOG.error("Ingestion failed: %s", e)
        return 1


def _validate_port(port: int, label: str) -> bool:
    if not (1024 <= port <= 65535):
        LOG.error("%s port %s out of allowed range 1024-65535", label, port)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description="Unified GoalDiggers launcher")
    sub = parser.add_subparsers(dest='command', required=True)

    dash_p = sub.add_parser('dashboard', help='Launch Streamlit dashboard only')
    dash_p.add_argument('--no-enhanced', action='store_true', help='Disable enhanced UI flags')
    dash_p.add_argument('--port', type=int, default=DEFAULT_DASHBOARD_PORT, help='Dashboard port (default 8501)')

    api_p = sub.add_parser('api', help='Start API service only')
    api_p.add_argument('--api-port', type=int, default=DEFAULT_API_PORT, help='API port (default 5000)')

    sse_p = sub.add_parser('sse', help='Start SSE service only')
    sse_p.add_argument('--sse-port', type=int, default=DEFAULT_SSE_PORT, help='SSE port (default 8079)')

    all_p = sub.add_parser('all', help='Dashboard plus optional services')
    all_p.add_argument('--services', nargs='*', default=['api','sse'], help='Which optional services to start')
    all_p.add_argument('--port', type=int, default=DEFAULT_DASHBOARD_PORT, help='Dashboard port (default 8501)')
    all_p.add_argument('--api-port', type=int, default=DEFAULT_API_PORT, help='API port (default 5000)')
    all_p.add_argument('--sse-port', type=int, default=DEFAULT_SSE_PORT, help='SSE port (default 8079)')

    health_p = sub.add_parser('health', help='Print runtime health snapshot & exit')
    health_p.add_argument('--with-prediction', action='store_true', help='Run a sample prediction before snapshot to validate inference path')

    report_p = sub.add_parser('report', help='Print ingestion & model report and exit')
    report_p.add_argument('--output', choices=['json','pretty'], default='pretty', help='Output format')

    ingest_p = sub.add_parser('ingest', help='Run data ingestion once and exit')
    ingest_p.add_argument('--days-back', type=int, default=2)
    ingest_p.add_argument('--days-ahead', type=int, default=7)

    train_p = sub.add_parser('train', help='Run model training pipeline scaffold')
    train_p.add_argument('--notes', type=str, default=None, help='Optional notes to attach to training run')
    train_p.add_argument('--output', choices=['pretty','json'], default='pretty', help='Output format (default pretty)')

    args = parser.parse_args(argv)

    enhanced = not getattr(args, 'no_enhanced', False)
    base_env_setup(enhanced=enhanced)
    validate_environment()

    if args.command in {'dashboard','all','api','sse'}:
        initialize_runtime_eager()
        maybe_start_prometheus()

    if args.command == 'health':
        maybe_start_prometheus()
        if getattr(args, 'with_prediction', False):
            try:
                from models.enhanced_real_data_predictor import (
                    get_enhanced_match_prediction,
                )
                get_enhanced_match_prediction('Manchester City','Arsenal', league='Premier League')
                LOG.info("Sample prediction executed prior to health snapshot")
            except Exception as e:  # pragma: no cover
                PRE_HEALTH_ERRORS.append(f"sample_prediction_error:{e}")
                LOG.warning("Sample prediction failed (continuing health): %s", e)
        return run_health_snapshot()
    if args.command == 'report':
        maybe_start_prometheus()
        try:
            # Lazy imports only when needed
            from database.db_manager import DatabaseManager
            from models.enhanced_real_data_predictor import EnhancedRealDataPredictor
            from scripts.data_pipeline.db_integrator import DataIntegrator
            from scripts.scrapers.scraper_factory import ScraperFactory
            dbm = DatabaseManager()
            di = DataIntegrator(dbm, ScraperFactory())
            predictor = EnhancedRealDataPredictor()
            report = {
                'launcher_version': LAUNCHER_VERSION,
                'ingestion_report': di.generate_ingestion_report(),
                'predictor_snapshot': predictor.get_monitoring_snapshot() if hasattr(predictor,'get_monitoring_snapshot') else {},
                'calibration': predictor.get_calibration_status() if hasattr(predictor,'get_calibration_status') else {},
                'timestamp': time.time()
            }
            if args.output == 'json':
                print(json.dumps(report, default=str))
            else:
                print("\n=== GoalDiggers Platform Report ===")
                print(f"Launcher Version: {report['launcher_version']}")
                dq = report['ingestion_report'].get('data_quality', {})
                print(f"Data Quality: leagues={dq.get('leagues')} teams={dq.get('teams')} scheduled={dq.get('matches_scheduled')} finished={dq.get('matches_finished')}")
                print("Latest Match Timestamp:", dq.get('latest_match_timestamp'))
                calib_raw = report.get('calibration') or {}
                # Some historical code paths returned a bare bool for calibration; normalize
                calib = calib_raw if isinstance(calib_raw, dict) else {'enabled': bool(calib_raw)}
                print(f"Calibration: enabled={calib.get('enabled')} fitted={calib.get('fitted')} applied={calib.get('applied')} loaded={calib.get('loaded')}")
                ps = report.get('predictor_snapshot') or {}
                print(f"Predictor: inferences={ps.get('total_inferences')} avg_latency_ms={ps.get('avg_inference_ms')} last_infer_ms={ps.get('last_inference_ms')}")
                # Scraper summary (graceful if structure changes)
                scrapers_obj = report['ingestion_report'].get('scrapers', {}) or {}
                print("Scrapers:")
                scraper_errors = 0
                if isinstance(scrapers_obj, dict):
                    for name, meta in scrapers_obj.items():
                        if isinstance(meta, dict):
                            status = meta.get('status', '?')
                            err_flag = meta.get('error') or meta.get('exception')
                            if err_flag:
                                scraper_errors += 1
                            print(f"  - {name}: status={status}{' ERROR' if err_flag else ''}")
                        else:
                            # meta might be a simple status string/bool
                            print(f"  - {name}: status={meta}")
                if scraper_errors:
                    print(f"  (scraper_errors={scraper_errors})")
                print("System Monitor Ops:")
                sm = report['ingestion_report'].get('system_monitor', [])
                if isinstance(sm, list):
                    for op in sm[:5]:
                        print("  *", op)
                print("===================================\n")
            return 0
        except Exception as e:
            LOG.error("Report generation failed: %s", e)
            return 1
    if args.command == 'ingest':
        return run_ingest(args.days_back, args.days_ahead)
    if args.command == 'train':
        try:
            from models.training.training_pipeline import run_training
            record = run_training(notes=getattr(args, 'notes', None))
            if getattr(args, 'output', 'pretty') == 'json':
                print(json.dumps(record.__dict__, indent=2))
            else:
                print("\n=== Training Run Summary ===")
                print(f"Run ID: {record.run_id}")
                print(f"Status: {record.status}")
                print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.started_at))}")
                if record.finished_at:
                    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.finished_at))}")
                print(f"Model Hash: {record.model_hash}")
                print(f"Artifact: {record.artifact_path}")
                if record.notes:
                    print(f"Notes: {record.notes}")
                print("============================\n")
            return 0
        except Exception as e:
            LOG.error("Training run failed: %s", e)
            return 1

    if args.command == 'dashboard':
        if not _validate_port(args.port, 'Dashboard'):
            return 1
        LOG.info("Unified launcher v%s (mode=dashboard)", LAUNCHER_VERSION)
        return launch_dashboard(port=args.port)
    if args.command == 'api':
        api_port = getattr(args, 'api_port', DEFAULT_API_PORT)
        if not _validate_port(api_port, 'API'):
            return 1
        LOG.info("Unified launcher v%s (mode=api)", LAUNCHER_VERSION)
        start_optional_services(['api'], api_port=api_port)
        LOG.info("API service started (non-blocking, press Ctrl+C to exit)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            LOG.info("Exiting API mode")
            return 0
    if args.command == 'sse':
        sse_port = getattr(args, 'sse_port', DEFAULT_SSE_PORT)
        if not _validate_port(sse_port, 'SSE'):
            return 1
        LOG.info("Unified launcher v%s (mode=sse)", LAUNCHER_VERSION)
        start_optional_services(['sse'], sse_port=sse_port)
        LOG.info("SSE service started (non-blocking, press Ctrl+C to exit)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            LOG.info("Exiting SSE mode")
            return 0
    if args.command == 'all':
        if not (_validate_port(args.port, 'Dashboard') and _validate_port(args.api_port, 'API') and _validate_port(args.sse_port, 'SSE')):
            return 1
        LOG.info("Unified launcher v%s (mode=all services=%s)", LAUNCHER_VERSION, args.services)
        start_optional_services(args.services, api_port=args.api_port, sse_port=args.sse_port)
        return launch_dashboard(port=args.port)

    LOG.error("Unknown command: %s", args.command)
    return 1


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
