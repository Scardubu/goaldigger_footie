"""
Models module for GoalDiggers platform.
Provides machine learning model integration and management.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Import key model components
try:
    from .predictive.enhanced_ml_pipeline import EnhancedMLPipeline
    logger.info("Successfully imported EnhancedMLPipeline")
except ImportError as e:
    logger.warning(f"Could not import EnhancedMLPipeline: {e}")
    EnhancedMLPipeline = None

try:
    from .xgboost_predictor import XGBoostPredictor
    logger.info("Successfully imported XGBoostPredictor")
except ImportError as e:
    logger.warning(f"Could not import XGBoostPredictor: {e}")
    XGBoostPredictor = None

try:
    from .predictive.ensemble_model import EnsemblePredictor
    logger.info("Successfully imported EnsemblePredictor")
except ImportError as e:
    logger.warning(f"Could not import EnsemblePredictor: {e}")
    EnsemblePredictor = None

try:
    # Lazy-import FeatureGenerator to avoid circular imports at package import time
    FeatureGenerator  # type: ignore[name-defined]
except NameError:
    FeatureGenerator = None  # Will be resolved lazily via __getattr__ or factory

# Model factory functions for easy initialization
def create_xgboost_predictor(model_path: str = None):
    """Create XGBoost predictor with default model path if none provided."""
    if XGBoostPredictor is None:
        logger.error("XGBoostPredictor not available")
        raise ImportError("XGBoostPredictor not available")

    if model_path is None:
        # Use default trained model path
        default_path = Path(__file__).parent / "trained" / "xgboost_predictor.joblib"
        if default_path.exists() and default_path.stat().st_size > 10000:
            model_path = str(default_path)
        else:
            # Fallback to predictor_model.joblib
            fallback_path = Path(__file__).parent / "trained" / "predictor_model.joblib"
            if fallback_path.exists() and fallback_path.stat().st_size > 10000:
                model_path = str(fallback_path)
            else:
                logger.error("No valid trained XGBoost model found (missing or corrupt file)")
                raise FileNotFoundError("No valid trained XGBoost model found (missing or corrupt file)")

    try:
        return XGBoostPredictor(model_path)
    except Exception as e:
        logger.error(f"Failed to load XGBoostPredictor from {model_path}: {e}")
        raise

def create_ensemble_predictor():
    """Create ensemble predictor."""
    if EnsemblePredictor is None:
        logger.error("EnsemblePredictor not available")
        raise ImportError("EnsemblePredictor not available")
    try:
        return EnsemblePredictor()
    except Exception as e:
        logger.error(f"Failed to initialize EnsemblePredictor: {e}")
        raise

def create_feature_generator(db_storage=None):
    """Create feature generator with optional db_storage."""
    # Import lazily to prevent circular import issues
    try:
        from .feature_eng.feature_generator import \
            FeatureGenerator as _FeatureGenerator
    except Exception as e:
        logger.error(f"Failed to import FeatureGenerator lazily: {e}")
        raise

    if db_storage is None:
        # Import database manager as default storage
        try:
            from database.db_manager import DatabaseManager
            db_storage = DatabaseManager()
        except ImportError:
            logger.warning("DatabaseManager not available, using None for db_storage")

    try:
        return _FeatureGenerator(db_storage)
    except Exception as e:
        logger.error(f"Failed to initialize FeatureGenerator: {e}")
        raise

# Create a comprehensive ML integration module
class MLIntegration:
    """Comprehensive ML integration for the GoalDiggers platform."""

    def __init__(self):
        self.available_models = []
        self._check_available_models()

    def _check_available_models(self):
        """Check which models are available."""
        if XGBoostPredictor is not None:
            self.available_models.append('xgboost')
        if EnsemblePredictor is not None:
            self.available_models.append('ensemble')
        if EnhancedMLPipeline is not None:
            self.available_models.append('enhanced_pipeline')

    def get_available_models(self):
        """Get list of available models."""
        return self.available_models

    def create_predictor(self, model_type: str = 'ensemble', **kwargs):
        """Create a predictor of the specified type."""
        if model_type == 'enhanced_pipeline':
            if EnhancedMLPipeline is None:
                raise ImportError("EnhancedMLPipeline not available")
            return EnhancedMLPipeline()
        if model_type == 'xgboost':
            return create_xgboost_predictor(kwargs.get('model_path'))
        elif model_type == 'ensemble':
            return create_ensemble_predictor()
        elif model_type == 'enhanced_pipeline':
            # fallback handled above
            raise ImportError("EnhancedMLPipeline not available")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def create_feature_generator(self, db_storage=None):
        """Create feature generator."""
        return create_feature_generator(db_storage)

# Global instance
ml_integration = MLIntegration()

# Backward compatibility
SimpleMLIntegration = MLIntegration

# Export for compatibility
__all__ = ['EnhancedMLPipeline', 'XGBoostPredictor', 'EnsemblePredictor', 'FeatureGenerator',
           'MLIntegration', 'SimpleMLIntegration', 'ml_integration',
           'create_xgboost_predictor', 'create_ensemble_predictor', 'create_feature_generator']

# Provide lazy attribute access for FeatureGenerator to avoid circular imports on module import
def __getattr__(name):
    if name == 'FeatureGenerator':
        try:
            from .feature_eng.feature_generator import \
                FeatureGenerator as _FeatureGenerator
            globals()['FeatureGenerator'] = _FeatureGenerator
            return _FeatureGenerator
        except Exception as e:
            raise AttributeError(f"FeatureGenerator not available: {e}")
    raise AttributeError(name)