#!/usr/bin/env python3
"""
Model Compatibility Manager for GoalDiggers
Provides graceful handling of scikit-learn version differences between development and production
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class ModelCompatibilityManager:
    """
    Handles scikit-learn version compatibility issues in production environment.
    Provides safe loading of models with version mismatch handling.
    """
    
    def __init__(self):
        # Filter the specific unpickle warning
        warnings.filterwarnings(
            "ignore", 
            message="Trying to unpickle estimator .* from version .* when using version .*", 
            category=UserWarning
        )
        
        self.loaded_models: Dict[str, Any] = {}
        logger.info("ModelCompatibilityManager initialized - handling scikit-learn version differences")
    
    def load_model(self, model_path: Union[str, Path], key: Optional[str] = None) -> Any:
        """
        Load a model with compatibility handling
        
        Args:
            model_path: Path to the model file
            key: Optional cache key, if None uses the model_path as key
            
        Returns:
            The loaded model
        """
        model_path = Path(model_path) if isinstance(model_path, str) else model_path
        cache_key = key or str(model_path)
        
        if cache_key in self.loaded_models:
            logger.debug(f"Using cached model: {cache_key}")
            return self.loaded_models[cache_key]
        
        try:
            # Load with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                model = joblib.load(model_path)
            
            # Cache for future use
            self.loaded_models[cache_key] = model
            logger.info(f"Successfully loaded model: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            # Re-raise to allow proper handling by caller
            raise
    
    def validate_model_predictions(
        self, 
        model: Any, 
        sample_input: pd.DataFrame, 
        expected_output_shape: Optional[tuple] = None
    ) -> bool:
        """
        Validate that a model can make predictions on the given input
        
        Args:
            model: The model to validate
            sample_input: Sample input data to test prediction
            expected_output_shape: Optional expected shape of output
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Test prediction
            prediction = model.predict(sample_input)
            
            # Validate shape if provided
            if expected_output_shape and prediction.shape != expected_output_shape:
                logger.warning(
                    f"Model prediction shape mismatch: "
                    f"got {prediction.shape}, expected {expected_output_shape}"
                )
                return False
            
            logger.info("Model prediction validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model prediction validation failed: {e}")
            return False

# Create singleton instance
compatibility_manager = ModelCompatibilityManager()
