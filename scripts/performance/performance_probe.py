"""Synthetic Performance Probe

Runs a controlled sequence of prediction batches to collect latency, cache,
and instrumentation metrics for baseline comparison.

Usage (example):
    python scripts/performance/performance_probe.py --batches 3 --batch-size 12 --output probe_results.json

Implementation Notes:
  - Ensures project root is on sys.path so that intra-project imports (e.g. prediction_ui.*) resolve
    even when executed via a deep script path ("python scripts/performance/performance_probe.py").
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Path bootstrap: add project root (two levels up from this file) to sys.path
# to avoid ModuleNotFoundError when running from the scripts/performance folder.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prediction_ui.metrics import collect_snapshot  # noqa: E402

# Local imports
from prediction_ui.service import (  # noqa
    _cache,
    get_fixtures,
    get_predictions_for_fixtures,
)
from utils.instrumentation import summarize_all


@dataclass
class BatchResult:
    batch_index: int
    fixture_count: int
    duration: float
    per_prediction_ms: float
    cache_hit_rate: float

async def run_probe(batches: int, batch_size: int, warm: bool) -> Dict[str, Any]:
    results: List[BatchResult] = []
    for i in range(batches):
        start = time.perf_counter()
        fixtures = await get_fixtures(limit=batch_size)
        preds_start = time.perf_counter()
        preds = await get_predictions_for_fixtures(fixtures)
        dur = time.perf_counter() - start
        pred_count = len(preds)
        per_pred = (time.perf_counter() - preds_start) / max(pred_count, 1)
        cache_stats = _cache.stats()
        results.append(BatchResult(
            batch_index=i,
            fixture_count=len(fixtures),
            duration=dur,
            per_prediction_ms=per_pred * 1000.0,
            cache_hit_rate=cache_stats.get('hit_rate', 0.0)
        ))
        # Optional small pause to simulate spaced usage
        await asyncio.sleep(0.25)
    # Aggregate
    durations = [r.duration for r in results]
    per_pred_ms = [r.per_prediction_ms for r in results]
    cache_rates = [r.cache_hit_rate for r in results]
    # Compute p95 safely (avoid negative index for very small samples)
    if durations:
        idx_p95 = min(max(int(len(durations) * 0.95) - 1, 0), len(durations) - 1)
        p95 = sorted(durations)[idx_p95]
    else:
        p95 = 0
    summary = {
        'batches': batches,
        'batch_size': batch_size,
        'initial_warm_run': warm,
        'duration_avg': statistics.fmean(durations) if durations else 0,
        'duration_p95': p95,
        'per_prediction_ms_avg': statistics.fmean(per_pred_ms) if per_pred_ms else 0,
        'cache_hit_rate_final': cache_rates[-1] if cache_rates else 0,
        'cache_hit_rate_series': cache_rates,
    }
    return {
        'runs': [r.__dict__ for r in results],
        'summary': summary,
        'instrumentation': summarize_all(),
        'metrics_snapshot': collect_snapshot(),
    }

def parse_args():
    p = argparse.ArgumentParser(description="Synthetic performance probe")
    p.add_argument('--batches', type=int, default=3, help='Number of batches to run')
    p.add_argument('--batch-size', type=int, default=10, help='Predictions per batch (target fixtures)')
    p.add_argument('--output', type=str, default='probe_results.json', help='Output JSON file')
    p.add_argument('--no-warm', action='store_true', help='Skip separate warm run')
    return p.parse_args()

async def main_async(args):
    # Optional warm run to populate cache before measured runs
    if not args.no_warm:
        await get_fixtures(limit=args.batch_size)
        await get_predictions_for_fixtures(await get_fixtures(limit=args.batch_size))
    data = await run_probe(args.batches, args.batch_size, warm=not args.no_warm)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Probe results written to {args.output}")

def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Probe cancelled.")

if __name__ == '__main__':
    main()
