"""
Enhanced ML pipeline with calibration and advanced ensemble methods.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)


class CalibratedEnsemblePredictor:
    """Enhanced predictor with calibration and ensemble methods."""
    
    def __init__(self):
        self.models = {}
        self.calibrated_models = {}
        self.is_trained = False
        
    def _create_base_models(self) -> Dict[str, any]:
        """Create base models for ensemble."""
        models = {}
        
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
        # Always include logistic regression as baseline
        models['logistic'] = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        return models
        
    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):
        """Train ensemble of models with calibration."""
        logger.info("Training calibrated ensemble...")
        
        # Create base models
        self.models = self._create_base_models()
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
        # Create voting ensemble
        voting_models = [(name, model) for name, model in self.models.items()]
        self.ensemble = VotingClassifier(
            estimators=voting_models,
            voting='soft'  # Use predicted probabilities
        )
        self.ensemble.fit(X_train, y_train)
        
        # Calibrate the ensemble
        logger.info("Calibrating ensemble predictions...")
        self.calibrated_ensemble = CalibratedClassifierCV(
            self.ensemble,
            method='isotonic',  # Use isotonic regression for calibration
            cv=3
        )
        self.calibrated_ensemble.fit(X_train, y_train)
        
        self.is_trained = True
        logger.info("Calibrated ensemble training complete")
        
    def predict_proba(self, X):
        """Get calibrated probability predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        return self.calibrated_ensemble.predict_proba(X)
        
    def predict(self, X):
        """Get class predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        return self.calibrated_ensemble.predict(X)

# ---------------------------------------------------------------------------
# Backwards Compatibility Export
# Many modules import `EnhancedMLPipeline` from `models.enhanced_ml_pipeline`.
# The actual rich implementation lives in `models/predictive/enhanced_ml_pipeline.py`.
# Provide a safe alias so legacy imports succeed while keeping lightweight
# calibrated predictor available.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - defensive import
    from models.predictive.enhanced_ml_pipeline import (
        EnhancedMLPipeline as _RealEnhancedMLPipeline,  # type: ignore
    )
    EnhancedMLPipeline = _RealEnhancedMLPipeline  # noqa: N816 (keep expected class name)
except Exception:  # Fallback to simple calibrated ensemble
    class EnhancedMLPipeline(CalibratedEnsemblePredictor):  # type: ignore
        """Fallback EnhancedMLPipeline using simple calibrated ensemble.

        This lightweight substitute is used if the advanced predictive
        pipeline cannot be imported (missing heavy deps like xgboost, lightgbm).
        """
        def __init__(self):
            super().__init__()
            self._monitoring_enabled = True
            self._train_started_at: Optional[float] = None
            self._train_completed_at: Optional[float] = None
            self._last_infer_ms: Optional[float] = None
            self._total_inferences: int = 0
            self._last_log_loss: Optional[float] = None
            self._class_labels: Optional[List] = None
            # Register globally for monitoring snapshot discovery (best effort)
            try:  # pragma: no cover
                from models.pipeline_registry import register_pipeline as _reg_pipeline
                _reg_pipeline(self)
            except Exception:
                pass

        def train_ensemble(self, X_train, y_train, X_val=None, y_val=None):  # type: ignore[override]
            import time
            self._train_started_at = time.time()
            super().train_ensemble(X_train, y_train, X_val, y_val)
            self._train_completed_at = time.time()
            try:
                # Track class labels for later explainability helpers
                import numpy as _np
                self._class_labels = list(_np.unique(y_train))
                if X_val is not None and y_val is not None:
                    from sklearn.metrics import log_loss as _ll
                    try:
                        proba = self.calibrated_ensemble.predict_proba(X_val)
                        self._last_log_loss = float(_ll(y_val, proba))
                    except Exception:
                        self._last_log_loss = None
            except Exception:
                pass

        # ---- Monitoring / Metrics Helpers ----
        def get_monitoring_snapshot(self) -> Dict[str, any]:
            """Return a lightweight snapshot of pipeline health/metrics."""
            import time
            train_duration = None
            if self._train_started_at and self._train_completed_at:
                train_duration = self._train_completed_at - self._train_started_at
            return {
                'trained': self.is_trained,
                'models': list(self.models.keys()),
                'has_calibrated': hasattr(self, 'calibrated_ensemble'),
                'train_duration_sec': round(train_duration, 3) if train_duration else None,
                'last_infer_ms': self._last_infer_ms,
                'total_inferences': self._total_inferences,
                'log_loss_val': self._last_log_loss,
                'class_labels': self._class_labels,
                'timestamp': time.time(),
                'version': 'fallback-calibrated-ensemble-1'
            }

        # ---- Explainability Helpers ----
        def get_feature_importance(self) -> Optional[Dict[str, float]]:
            """Derive rudimentary feature importance if logistic regression present.

            For the fallback pipeline, only logistic coefficients are exposed; if
            XGBoost is available they are not combined (kept simple for speed).
            """
            try:
                lr = self.models.get('logistic')
                if lr is None or not hasattr(lr, 'coef_'):
                    return None
                import numpy as _np
                coef = lr.coef_[0]
                return {f'x{i}': float(v) for i, v in enumerate(coef)}
            except Exception:
                return None

        def predict_proba(self, X):  # type: ignore[override]
            import time
            start = time.time()
            proba = super().predict_proba(X)
            self._last_infer_ms = round((time.time() - start) * 1000, 3)
            self._total_inferences += 1
            return proba

        def predict_with_explanations(self, X) -> Tuple[np.ndarray, Optional[List[Dict[str, float]]]]:
            """Return predictions & per-sample simplistic explanation vectors.

            Explanations (if logistic model exists) are just the raw feature * coefficient
            contributions (no intercept) so downstream UI can surface relative influence.
            """
            import numpy as _np
            proba = self.predict_proba(X)
            explanations = None
            try:
                lr = self.models.get('logistic')
                if lr is not None and hasattr(lr, 'coef_'):
                    coef = lr.coef_[0]
                    # Assume X is ndarray-like
                    arr = _np.asarray(X)
                    # Clip to first len(coef) features
                    arr = arr[:, :len(coef)]
                    contrib = arr * coef  # shape (n_samples, n_features)
                    explanations = [
                        {f'x{i}': float(v) for i, v in enumerate(sample)} for sample in contrib
                    ]
            except Exception:
                explanations = None
            return proba, explanations

__all__ = [
    "CalibratedEnsemblePredictor",
    "EnhancedMLPipeline",
]