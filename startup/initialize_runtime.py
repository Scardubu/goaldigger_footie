"""Runtime initializer to ensure pipeline & predictor singletons are instantiated and registered.

Idempotent and lightweight: avoids heavy training; only constructs objects if
registries are empty. Can be imported early by app/launcher before monitoring
snapshot collection so runtime health surfaces fully.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _safe_import(name: str):
    try:
        module = __import__(name, fromlist=['*'])
        return module, None
    except Exception as e:  # pragma: no cover
        return None, e


def initialize_runtime(eager_warm: bool = False) -> dict:
    """Initialize core runtime singletons.

    Args:
        eager_warm: If True, perform a trivial dummy prediction (if safe) to
                    populate feature vector shapes & inference timing.
    Returns:
        dict summary of actions taken
    """
    summary = {
        'pipeline': None,
        'predictor': None,
        'pipeline_created': False,
        'predictor_created': False,
        'warm_ran': False,
        'errors': []
    }

    # --- Pipeline ---
    pipe_reg, err = _safe_import('models.pipeline_registry')
    if not pipe_reg:
        summary['errors'].append(f'pipeline_registry_import:{err}')
    else:
        pipeline_instance = getattr(pipe_reg, 'get_pipeline', lambda: None)()
        if pipeline_instance is None:
            # Try advanced predictive pipeline first
            adv_mod, adv_err = _safe_import('models.predictive.enhanced_ml_pipeline')
            pipeline_instance = None
            if adv_mod and hasattr(adv_mod, 'EnhancedMLPipeline'):
                try:
                    pipeline_instance = adv_mod.EnhancedMLPipeline()
                    summary['pipeline_created'] = True
                except Exception as e:  # pragma: no cover
                    summary['errors'].append(f'advanced_pipeline_ctor:{e}')
            if pipeline_instance is None:
                # Fallback lightweight pipeline
                fb_mod, fb_err = _safe_import('models.enhanced_ml_pipeline')
                if fb_mod and hasattr(fb_mod, 'EnhancedMLPipeline'):
                    try:
                        pipeline_instance = fb_mod.EnhancedMLPipeline()
                        summary['pipeline_created'] = True
                    except Exception as e:  # pragma: no cover
                        summary['errors'].append(f'fallback_pipeline_ctor:{e}')
                else:
                    summary['errors'].append(f'fallback_pipeline_import:{fb_err}')
            if pipeline_instance is not None:
                try:  # already registered by ctor but ensure
                    getattr(pipe_reg, 'register_pipeline')(pipeline_instance)
                except Exception:
                    pass
        summary['pipeline'] = pipeline_instance.__class__.__name__ if pipeline_instance else None

    # --- Predictor ---
    pred_reg, err = _safe_import('models.predictor_registry')
    if not pred_reg:
        summary['errors'].append(f'predictor_registry_import:{err}')
    else:
        predictor_instance = getattr(pred_reg, 'get_predictor', lambda: None)()
        if predictor_instance is None:
            pred_mod, pred_err = _safe_import('models.enhanced_real_data_predictor')
            if pred_mod and hasattr(pred_mod, 'EnhancedRealDataPredictor'):
                try:
                    predictor_instance = pred_mod.EnhancedRealDataPredictor()
                    summary['predictor_created'] = True
                except Exception as e:  # pragma: no cover
                    summary['errors'].append(f'predictor_ctor:{e}')
            else:
                summary['errors'].append(f'predictor_import:{pred_err}')
            if predictor_instance is not None:
                try:
                    getattr(pred_reg, 'register_predictor')(predictor_instance)
                except Exception:
                    pass
        summary['predictor'] = predictor_instance.__class__.__name__ if predictor_instance else None

    # Optional warm inference (safe best-effort)
    if eager_warm:
        try:
            if predictor_instance and hasattr(predictor_instance, 'predict_match_enhanced'):
                # Pass obviously invalid IDs expecting graceful handling
                predictor_instance.predict_match_enhanced(-1, -1)
                summary['warm_ran'] = True
        except Exception:
            pass

    return summary

__all__ = ["initialize_runtime"]
