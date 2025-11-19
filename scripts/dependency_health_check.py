"""Dependency & optional feature health check.

Runs lightweight introspection to list core + optional libraries and their availability.

Usage:
    python -m scripts.dependency_health_check
"""
from __future__ import annotations

import importlib
import json
import platform
import sys
from typing import Any, Dict

OPTIONAL_MODULES = {
    'xgboost': 'Advanced gradient boosting (model ensemble component)',
    'lightgbm': 'LightGBM boosting (model ensemble component)',
    'catboost': 'CatBoost boosting (model ensemble component)',
    'shap': 'Explainability (SHAP value computation)',
    'optuna': 'Hyperparameter optimization framework',
    'prometheus_client': 'Metrics exposition (Prometheus scraping)' ,
}

CORE_MODULES = {
    'numpy': 'Numerical computing',
    'pandas': 'Data manipulation',
    'sklearn': 'Model training & preprocessing',
}


def _probe(mod: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {'module': mod, 'available': False, 'version': None, 'description': OPTIONAL_MODULES.get(mod) or CORE_MODULES.get(mod)}
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, '__version__', None)
        info['available'] = True
        info['version'] = ver
    except Exception as e:  # noqa: BLE001
        info['error'] = str(e)
    return info


def main() -> int:
    report = {
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'core': [],
        'optional': [],
    }
    for mod in CORE_MODULES:
        report['core'].append(_probe(mod))
    for mod in OPTIONAL_MODULES:
        report['optional'].append(_probe(mod))

    print(json.dumps(report, indent=2))

    # Basic pass condition: all core modules available
    passed = all(entry['available'] for entry in report['core'])
    print("\nDependency Health:")
    print("  Core modules OK:" , passed)
    if not passed:
        missing = [c['module'] for c in report['core'] if not c['available']]
        print("  Missing core modules:", missing)
    return 0 if passed else 1


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
