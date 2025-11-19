"""Unified runtime snapshot aggregation for GoalDiggers platform.

Provides a single function get_runtime_snapshot() that attempts to gather:
  - Core metrics wrapper snapshot (prediction + explanation metrics)
  - Data pipeline snapshot (matches, teams, leagues, API health) from RealDataIntegrator
  - Model pipeline monitoring snapshot (if advanced pipeline loaded)
  - Predictor calibration / model status (if real data predictor available)

Design Principles:
  * Fail fast & soft: never raises; returns partial data and an errors[] list.
  * Lightweight: avoids triggering heavy training or network fetches.
  * Stable shape: keys remain present (value None) even if unavailable.

Intended Usage:
  from monitoring.runtime_snapshot import get_runtime_snapshot
  snapshot = get_runtime_snapshot()
  st.json(snapshot)  # Or log / expose via API
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

RUNTIME_SNAPSHOT_VERSION = "1.0"


def _try_import(name: str):
    try:
        module = __import__(name, fromlist=['*'])
        return module, None
    except Exception as e:  # pragma: no cover - defensive
        return None, str(e)


def get_runtime_snapshot(include_timestamps: bool = True) -> Dict[str, Any]:
    errors = []
    metrics_part: Optional[Dict[str, Any]] = None
    data_part: Optional[Dict[str, Any]] = None
    model_part: Optional[Dict[str, Any]] = None
    predictor_part: Optional[Dict[str, Any]] = None
    model_initialized = False
    predictor_initialized = False
    cache_metrics: Optional[Dict[str, Any]] = None

    # Metrics wrapper
    metrics_mod, err = _try_import('metrics.metrics_wrapper')
    if metrics_mod:
        try:  # snapshot() returns dict
            metrics_part = metrics_mod.snapshot()
        except Exception as e:  # pragma: no cover
            errors.append(f"metrics_snapshot_error:{e}")
    else:
        errors.append(f"metrics_import_error:{err}")

    # Data pipeline snapshot
    data_mod, err = _try_import('real_data_integrator')
    if data_mod:
        try:
            if hasattr(data_mod, 'get_data_pipeline_snapshot'):
                data_part = data_mod.get_data_pipeline_snapshot()
            elif hasattr(data_mod, 'real_data_integrator'):
                # Fallback: construct minimal if method absent
                integrator = getattr(data_mod, 'real_data_integrator')
                data_part = {
                    'status': 'legacy',
                    'matches_len': len(integrator.get_real_matches_data(days_ahead=7) or []),
                }
        except Exception as e:  # pragma: no cover
            errors.append(f"data_snapshot_error:{e}")
    else:
        errors.append(f"data_import_error:{err}")

    # Model pipeline snapshot via registry (preferred)
    try:
        reg_mod, _ = _try_import('models.pipeline_registry')
        pipeline_instance = None
        if reg_mod and hasattr(reg_mod, 'get_pipeline'):
            try:
                pipeline_instance = reg_mod.get_pipeline()
            except Exception:
                pipeline_instance = None
        if pipeline_instance is not None:
            try:
                if hasattr(pipeline_instance, 'get_monitoring_snapshot'):
                    model_part = pipeline_instance.get_monitoring_snapshot()
                else:
                    # Minimal structural info
                    model_part = {
                        'status': 'no_monitoring_method',
                        'class': pipeline_instance.__class__.__name__
                    }
            except Exception as e:  # pragma: no cover
                errors.append(f"model_snapshot_error:{e}")
        else:
            # Legacy class-level access (kept for backward compatibility)
            adv_pipeline_mod, _ = _try_import('models.predictive.enhanced_ml_pipeline')
            if adv_pipeline_mod and hasattr(adv_pipeline_mod, 'EnhancedMLPipeline'):
                try:
                    Enhanced = adv_pipeline_mod.EnhancedMLPipeline  # type: ignore
                    if hasattr(Enhanced, 'get_monitoring_snapshot'):
                        try:
                            model_part = Enhanced.get_monitoring_snapshot(Enhanced)
                        except TypeError:
                            model_part = {'status': 'unavailable_instance_required'}
                except Exception as e:
                    errors.append(f"model_snapshot_error:{e}")
            # Attempt a lightweight lazy instantiation so future snapshots have data
            if model_part is None or (isinstance(model_part, dict) and model_part.get('status') == 'unavailable_instance_required'):
                try:  # pragma: no cover - best effort
                    # Prefer advanced pipeline
                    pipeline_instance = None
                    if adv_pipeline_mod and hasattr(adv_pipeline_mod, 'EnhancedMLPipeline'):
                        try:
                            pipeline_instance = adv_pipeline_mod.EnhancedMLPipeline()
                        except Exception:
                            pipeline_instance = None
                    if pipeline_instance is None:
                        fb_mod, _ = _try_import('models.enhanced_ml_pipeline')
                        if fb_mod and hasattr(fb_mod, 'EnhancedMLPipeline'):
                            try:
                                pipeline_instance = fb_mod.EnhancedMLPipeline()
                            except Exception:
                                pipeline_instance = None
                    if pipeline_instance is not None:
                        try:
                            if reg_mod and hasattr(reg_mod, 'register_pipeline'):
                                reg_mod.register_pipeline(pipeline_instance)  # type: ignore
                        except Exception:
                            pass
                        try:
                            if hasattr(pipeline_instance, 'get_monitoring_snapshot'):
                                model_part = pipeline_instance.get_monitoring_snapshot()
                            else:
                                model_part = {
                                    'status': 'initialized_no_monitoring',
                                    'class': pipeline_instance.__class__.__name__
                                }
                            model_initialized = True
                        except Exception as e:
                            errors.append(f"model_lazy_init_snapshot_error:{e}")
                except Exception as e:
                    errors.append(f"model_lazy_init_error:{e}")
    except Exception as e:  # pragma: no cover
        errors.append(f"model_registry_error:{e}")

    # Real data predictor status via registry
    try:
        pred_reg_mod, _ = _try_import('models.predictor_registry')
        predictor_instance = None
        if pred_reg_mod and hasattr(pred_reg_mod, 'get_predictor'):
            try:
                predictor_instance = pred_reg_mod.get_predictor()
            except Exception:
                predictor_instance = None
        if predictor_instance is not None:
            try:
                if hasattr(predictor_instance, 'get_monitoring_snapshot'):
                    base = predictor_instance.get_monitoring_snapshot() or {}
                else:
                    base = {}
                base.update({
                    'class': predictor_instance.__class__.__name__,
                    'model_version': getattr(predictor_instance, 'model_version', None),
                })
                predictor_part = base
            except Exception as e:  # pragma: no cover
                errors.append(f"predictor_snapshot_error:{e}")
        else:
            # Legacy import-only detection (retain prior behavior)
            predictor_mod, err_pred = _try_import('models.enhanced_real_data_predictor')
            if predictor_mod:
                try:
                    fields = {}
                    for attr in ['is_calibrated', 'calibration_method', 'last_calibrated_at']:
                        if hasattr(predictor_mod, attr):
                            try:
                                val = getattr(predictor_mod, attr)
                                if callable(val):
                                    val = val()
                                fields[attr] = val
                            except Exception:
                                fields[attr] = None
                    predictor_part = fields or None
                except Exception as e:  # pragma: no cover
                    errors.append(f"predictor_snapshot_error:{e}")
            # Attempt lazy instantiation if still None
            if predictor_part is None:
                try:  # pragma: no cover - best effort
                    predictor_instance = None
                    if predictor_mod and hasattr(predictor_mod, 'EnhancedRealDataPredictor'):
                        try:
                            predictor_instance = predictor_mod.EnhancedRealDataPredictor()
                        except Exception:
                            predictor_instance = None
                    if predictor_instance is not None:
                        try:
                            if pred_reg_mod and hasattr(pred_reg_mod, 'register_predictor'):
                                pred_reg_mod.register_predictor(predictor_instance)  # type: ignore
                        except Exception:
                            pass
                        predictor_part = {
                            'class': predictor_instance.__class__.__name__,
                            'model_version': getattr(predictor_instance, 'model_version', None),
                            'calibration_enabled': getattr(predictor_instance, '_calibration_enabled', None),
                            'calibration_applied': getattr(predictor_instance, '_calibration_applied_count', None),
                            'lazy_initialized': True
                        }
                        predictor_initialized = True
                except Exception as e:
                    errors.append(f"predictor_lazy_init_error:{e}")
    except Exception as e:  # pragma: no cover
        errors.append(f"predictor_registry_error:{e}")

    snapshot: Dict[str, Any] = {
        'version': RUNTIME_SNAPSHOT_VERSION,
        'generated_at': time.time() if include_timestamps else None,
        'metrics': metrics_part,
        'data_pipeline': data_part,
        'model_pipeline': model_part,
        'predictor': predictor_part,
        'cache_metrics': None,  # replaced after optional import below
        'model_registered': model_part is not None,
        'predictor_registered': predictor_part is not None,
        'model_lazy_initialized': model_initialized or None,
        'predictor_lazy_initialized': predictor_initialized or None,
        'errors': errors or None
    }

    # Late-binding cache metrics to avoid interfering with core data collection
    try:
        cache_mod, _ = _try_import('cached_data_utilities')
        if cache_mod and hasattr(cache_mod, 'get_cache_metrics_snapshot'):
            cache_metrics = cache_mod.get_cache_metrics_snapshot()
            snapshot['cache_metrics'] = cache_metrics
    except Exception as e:  # pragma: no cover
        errors.append(f"cache_metrics_error:{e}")
        snapshot['cache_metrics'] = None

    # Derive alerts (best effort)
    try:  # pragma: no cover
        from monitoring.alert_evaluator import evaluate_alerts
        snapshot['alerts'] = evaluate_alerts(snapshot)
    except Exception:
        snapshot['alerts'] = None

    return snapshot

__all__ = ["get_runtime_snapshot", "RUNTIME_SNAPSHOT_VERSION"]
