"""Quick runtime health verifier.

Usage (from project root):
    python -m monitoring.verify_runtime_health

Outputs a JSON snapshot plus basic pass/fail signals for CI or manual checks.
Exit code 0 if core sections present; 1 otherwise.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict

from monitoring.runtime_snapshot import get_runtime_snapshot

REQUIRED_TOP_LEVEL = [
    'metrics', 'data_pipeline', 'model_pipeline', 'predictor'
]


def main() -> int:
    snap: Dict[str, Any] = get_runtime_snapshot()
    print(json.dumps(snap, indent=2)[:4000])

    missing = [k for k in REQUIRED_TOP_LEVEL if snap.get(k) is None]

    # Basic health heuristics
    data_status = (snap.get('data_pipeline') or {}).get('status')
    model_has_keys = isinstance(snap.get('model_pipeline'), dict)
    predictor_has_keys = isinstance(snap.get('predictor'), dict)

    passed = not missing and data_status in ('healthy', 'degraded') and model_has_keys and predictor_has_keys

    print("\nHealth Summary:")
    print(f"  Missing sections: {missing if missing else 'None'}")
    print(f"  Data status: {data_status}")
    print(f"  Model section OK: {model_has_keys}")
    print(f"  Predictor section OK: {predictor_has_keys}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
