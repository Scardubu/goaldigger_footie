#!/usr/bin/env python3
"""
Model Singleton Manager

Implements singleton pattern for ML models to prevent repeated loading
and improve production performance.
"""

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Import safe symbols for Windows compatibility
try:
    from utils.safe_symbols import get_safe_logger
    logger = get_safe_logger(__name__)
except ImportError:
    # Fallback to regular logger if safe_symbols not available
    pass

class ModelSingleton:
    """Singleton manager for ML models to prevent repeated loading."""
    
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelSingleton, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            return cls()
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("Initializing Model Singleton Manager")
            self._initialized = True
    
    def get_xgboost_predictor(self, model_path: str = "models/predictor_model.joblib") -> Optional[Any]:
        """Get or create XGBoost predictor instance."""
        key = f"xgboost_{model_path}"
        
        if key not in self._models:
            with self._lock:
                if key not in self._models:  # Double-check locking
                    try:
                        logger.info(f"Loading XGBoost predictor for the first time: {model_path}")
                        from models.xgboost_predictor import XGBoostPredictor

                        # Load with production mode enabled
                        predictor = XGBoostPredictor(model_path, production_mode=True)
                        self._models[key] = predictor
                        logger.info(f"✅ XGBoost predictor loaded and cached: {model_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to load XGBoost predictor: {e}")
                        return None
        else:
            logger.debug(f"Using cached XGBoost predictor: {model_path}")
        
        return self._models.get(key)
    
    def get_enhanced_prediction_engine(self) -> Optional[Any]:
        """Get or create Enhanced Prediction Engine instance."""
        key = "enhanced_prediction_engine"
        
        if key not in self._models:
            with self._lock:
                if key not in self._models:  # Double-check locking
                    try:
                        logger.info("Loading Enhanced Prediction Engine for the first time")
                        from enhanced_prediction_engine import \
                            EnhancedPredictionEngine
                        
                        engine = EnhancedPredictionEngine()
                        self._models[key] = engine
                        logger.info("✅ Enhanced Prediction Engine loaded and cached")
                        
                    except Exception as e:
                        logger.error(f"Failed to load Enhanced Prediction Engine: {e}")
                        return None
        else:
            logger.debug("Using cached Enhanced Prediction Engine")
        
        return self._models.get(key)
    
    def get_feature_mapper(self) -> Optional[Any]:
        """Get or create Feature Mapper instance."""
        key = "feature_mapper"
        
        if key not in self._models:
            with self._lock:
                if key not in self._models:  # Double-check locking
                    try:
                        logger.info("Loading Feature Mapper for the first time")
                        from utils.feature_mapper import FeatureMapper
                        
                        mapper = FeatureMapper()
                        self._models[key] = mapper
                        logger.info("✅ Feature Mapper loaded and cached")
                        
                    except Exception as e:
                        logger.error(f"Failed to load Feature Mapper: {e}")
                        return None
        else:
            logger.debug("Using cached Feature Mapper")
        
        return self._models.get(key)
    
    def clear_cache(self):
        """Clear all cached models (for testing/debugging)."""
        with self._lock:
            logger.info("Clearing model cache")
            self._models.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        return {
            "cached_models": list(self._models.keys()),
            "cache_size": len(self._models),
            "initialized": self._initialized
        }

# Global singleton instance
model_manager = ModelSingleton()

def get_model_manager() -> ModelSingleton:
    """Get the global model manager instance."""
    return model_manager
