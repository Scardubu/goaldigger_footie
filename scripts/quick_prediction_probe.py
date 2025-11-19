#!/usr/bin/env python3
"""Quick Prediction Probe

Purpose:
    Simple CLI utility to exercise the batched prediction pathway and print
    real-data provenance fields (no Streamlit, minimal deps). Designed to
    avoid complex inline shell quoting on Windows PowerShell when testing
    async service functions.

Usage Examples:
    # Single fixture (defaults league if omitted)
    python scripts/quick_prediction_probe.py Arsenal Chelsea "Premier League" --pretty

    # Batch fixtures (mixed separator styles accepted)
    python scripts/quick_prediction_probe.py --batch "Arsenal vs Chelsea" "Liverpool vs Manchester City" --fields home,away,home_win_prob,real_data_used

    # Show all available keys for a single probe (diagnostic)
    python scripts/quick_prediction_probe.py Arsenal Chelsea --dump-all

Exit Codes:
    0 success, >0 failure (import errors, runtime exceptions, no predictions).

Outputs:
    JSON lines (one per prediction). Use --pretty for formatted JSON.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from typing import Iterable, List, Sequence

# --- Sys.path hardening (ensure project root present) ---
PROJECT_MARKERS = {'.git', 'pyproject.toml', 'README.md'}
ROOT_CANDIDATE = os.getcwd()
if not any(os.path.exists(os.path.join(ROOT_CANDIDATE, m)) for m in PROJECT_MARKERS):
    # If launched elsewhere, try script directory two levels up
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_CANDIDATE = os.path.abspath(os.path.join(_script_dir, '..'))
if ROOT_CANDIDATE not in sys.path:
    sys.path.insert(0, ROOT_CANDIDATE)

try:
    from prediction_ui.service import (  # type: ignore
        FixtureDescriptor,
        get_predictions_for_fixtures,
    )
except Exception as import_err:  # Provide rich diagnostic then exit
    # Avoid nested braces confusion in f-string by assembling pieces
    err_type = import_err.__class__.__name__
    err_msg = str(import_err).replace('"', '\"')
    print(f"{{\"error\": \"Failed to import prediction_ui.service\", \"exception\": \"{err_type}: {err_msg}\"}}")
    print("--- TRACEBACK START ---", file=sys.stderr)
    traceback.print_exc()
    print("--- TRACEBACK END ---", file=sys.stderr)
    sys.exit(2)


def _parse_batch_arg(batch_values: Sequence[str]) -> List[FixtureDescriptor]:
    fixtures: List[FixtureDescriptor] = []
    for raw in batch_values:
        candidate = raw.strip()
        if not candidate:
            continue
        # Accept formats: "Home vs Away" or "Home, Away" or "Home|Away" or "Home/Away"
        separators = [" vs ", " VS ", ",", "|", "/"]
        parts = None
        for sep in separators:
            if sep in candidate:
                maybe = [p.strip() for p in candidate.split(sep, 1)]
                if len(maybe) == 2 and all(maybe):
                    parts = maybe
                    break
        if not parts:
            print(f"[WARN] Skipping unparseable fixture spec: {raw}", file=sys.stderr)
            continue
        fixtures.append(FixtureDescriptor(parts[0], parts[1], None))
    return fixtures


def build_fixtures(args) -> List[FixtureDescriptor]:
    if args.batch:
        return _parse_batch_arg(args.batch)
    if args.home and args.away:
        return [FixtureDescriptor(args.home, args.away, args.league)]
    raise SystemExit("Must supply either --home/--away or --batch fixtures")


DEFAULT_FIELDS = [
    'home', 'away', 'league', 'league_code',
    'home_win_prob', 'draw_prob', 'away_win_prob', 'confidence',
    'real_data_used', 'real_data_timestamp'
]

EXTRA_FIELD_MAP = {
    'sources_keys': lambda p: list((p.get('real_data_sources') or {}).keys()),
    'sources_flags': lambda p: (p.get('real_data_sources') or {}).get('features'),
    'model_real_used': lambda p: (p.get('real_data_sources') or {}).get('model', {}).get('real_data_used'),
}


def _extract_record(p: dict, fields: Iterable[str], dump_all: bool) -> dict:
    if dump_all:
        return p
    rec = {}
    for f in fields:
        if f in EXTRA_FIELD_MAP:
            try:
                rec[f] = EXTRA_FIELD_MAP[f](p)
            except Exception as e:  # pragma: no cover
                rec[f] = f"<err:{e}>"
        elif f in ('home', 'away'):
            rec[f] = p.get('home_team') if f == 'home' else p.get('away_team')
        else:
            rec[f] = p.get(f)
    return rec


async def run_probe(fixtures: List[FixtureDescriptor], pretty: bool, fields: List[str], dump_all: bool) -> int:
    preds = await get_predictions_for_fixtures(fixtures)
    if not preds:
        print('{"error":"no_predictions"}')
        return 3
    for p in preds:
        record = _extract_record(p, fields, dump_all)
        print(json.dumps(record, indent=2 if pretty else None))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Quick prediction metadata probe")
    parser.add_argument('home', nargs='?', help='Home team (if not using --batch)')
    parser.add_argument('away', nargs='?', help='Away team (if not using --batch)')
    parser.add_argument('league', nargs='?', default='Premier League', help='League name (optional)')
    parser.add_argument('--batch', nargs='*', help='Batch fixtures like "Arsenal vs Chelsea"')
    parser.add_argument('--pretty', action='store_true', help='Pretty-print JSON output')
    parser.add_argument('--fields', help='Comma-separated subset of fields (default set shown in docs). Use "sources_keys" for provenance summary.', default=None)
    parser.add_argument('--dump-all', action='store_true', help='Dump the full raw prediction dict (diagnostic)')
    args = parser.parse_args()

    fixtures = build_fixtures(args)
    field_list = DEFAULT_FIELDS
    if args.fields:
        field_list = [f.strip() for f in args.fields.split(',') if f.strip()]
        if not field_list:
            print('[WARN] No valid fields parsed; using defaults', file=sys.stderr)
            field_list = DEFAULT_FIELDS
    try:
        exit_code = asyncio.run(run_probe(fixtures, pretty=args.pretty, fields=field_list, dump_all=args.dump_all))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:  # broad catch for CLI robustness
        print(json.dumps({'error': 'runtime_exception', 'message': str(e)}))
        traceback.print_exc()
        sys.exit(4)


if __name__ == '__main__':
    main()
