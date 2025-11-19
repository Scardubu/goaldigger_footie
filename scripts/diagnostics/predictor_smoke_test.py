#!/usr/bin/env python3
"""Predictor Smoke Test

Quick diagnostic to verify EnhancedRealDataPredictor end-to-end:
- Initializes singleton predictor
- Runs a sample prediction on a marquee fixture
- Prints key outputs (probabilities, confidence, real_data_used, calibration status)
- Exits 0 on success, 1 on failure

Usage:
  python scripts/diagnostics/predictor_smoke_test.py --home "Manchester City" --away Arsenal --league "Premier League"

Accepts optional --json flag for machine readable output.
"""
from __future__ import annotations
import argparse, json, sys, time

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.enhanced_real_data_predictor import get_enhanced_match_prediction

def run(home: str, away: str, league: str, as_json: bool=False) -> int:
    start = time.time()
    try:
        payload = get_enhanced_match_prediction(home, away, league=league, force_real=True)
        duration_ms = (time.time() - start) * 1000.0
        summary = {
            'fixture': f"{home} vs {away}",
            'league': league,
            'home_win': round(payload.get('home_win_probability', 0),4),
            'draw': round(payload.get('draw_probability', 0),4),
            'away_win': round(payload.get('away_win_probability', 0),4),
            'confidence': round(payload.get('confidence_score', 0),4),
            'real_data_used': payload.get('real_data_used'),
            'data_timestamp': payload.get('data_timestamp'),
            'calibration': payload.get('calibration', {}),
            'inference_ms': round(duration_ms, 2),
        }
        if as_json:
            print(json.dumps(summary))
        else:
            print("=== Predictor Smoke Test ===")
            for k,v in summary.items():
                print(f"{k}: {v}")
        return 0
    except Exception as e:
        if as_json:
            print(json.dumps({'error': str(e)}))
        else:
            print(f"Smoke test failed: {e}")
        return 1

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--home', default='Manchester City')
    p.add_argument('--away', default='Arsenal')
    p.add_argument('--league', default='Premier League')
    p.add_argument('--json', action='store_true')
    args = p.parse_args(argv)
    return run(args.home, args.away, args.league, args.json)

if __name__ == '__main__':
    sys.exit(main())
