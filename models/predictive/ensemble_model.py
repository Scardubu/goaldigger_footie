"""
Ensemble model predictor for football match outcome predictions.
This module combines multiple prediction models to create a robust ensemble prediction.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from utils.config import Config

# Initialize logger first
logger = logging.getLogger(__name__)

# Import compatibility manager for model version differences
try:
    from utils.model_compatibility import compatibility_manager
    COMPATIBILITY_MANAGER_AVAILABLE = True
except ImportError:
    COMPATIBILITY_MANAGER_AVAILABLE = False
    logger.warning("Model compatibility manager not available, using direct loading")

class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models for football match outcome prediction.
    Combines a static (pre-trained) model with a dynamic (potentially updated) model.
    """
    
    def __init__(self):
        """
        Initialize the ensemble predictor by loading static and dynamic models.
        """
        self.static_model = None
        self.dynamic_model = None
        
        # Load models from the configured paths
        try:
            # Get paths from config
            project_root = Config.get('paths.project_root', os.getenv('PROJECT_ROOT', '.'))
            model_base_path = Config.get('paths.models.base', 'models/trained')
            static_model_filename = Config.get('paths.models.predictor_filename', 'predictor_model.joblib')
            dynamic_model_filename = Config.get('paths.models.dynamic_predictor_filename', 'dynamic_predictor_model.joblib')
            
            # Construct absolute paths
            static_model_path = os.path.join(project_root, model_base_path, static_model_filename)
            dynamic_model_path = os.path.join(project_root, model_base_path, dynamic_model_filename)
            
            # Load static model (required)
            if os.path.exists(static_model_path):
                logger.info(f"Loading static model from {static_model_path}")
                if COMPATIBILITY_MANAGER_AVAILABLE:
                    # Use compatibility manager for version differences
                    self.static_model = compatibility_manager.load_model(static_model_path, "static")
                else:
                    self.static_model = joblib.load(static_model_path)
            else:
                logger.warning(f"Static model not found at {static_model_path}")
                
            # Load dynamic model (optional)
            if os.path.exists(dynamic_model_path):
                logger.info(f"Loading dynamic model from {dynamic_model_path}")
                if COMPATIBILITY_MANAGER_AVAILABLE:
                    # Use compatibility manager for version differences
                    self.dynamic_model = compatibility_manager.load_model(dynamic_model_path, "dynamic")
                else:
                    self.dynamic_model = joblib.load(dynamic_model_path)
            else:
                logger.warning(f"Dynamic model not found at {dynamic_model_path}, using static model only")
                
        except Exception as e:
            logger.error(f"Error initializing EnsemblePredictor: {e}")
            raise
    
    def predict(self, features: pd.DataFrame, news_text: Optional[str] = None) -> List[float]:
        """
        Make predictions using the ensemble of models.
        
        Args:
            features: DataFrame containing features for a single match
            news_text: Optional text with news about the match
            
        Returns:
            List of probabilities [home_win_prob, draw_prob, away_win_prob]
        """
        if features is None or features.empty:
            logger.error("Empty features provided to EnsemblePredictor")
            return [1/3, 1/3, 1/3]  # Default to equal probabilities
            
        try:
            # Preprocess features if needed
            processed_features = self._preprocess_features(features)
            
            # Get predictions from static model
            static_probs = self._get_static_predictions(processed_features)
            
            # Get predictions from dynamic model if available
            if self.dynamic_model is not None:
                dynamic_probs = self._get_dynamic_predictions(processed_features, news_text)
                # Combine predictions (weighted average)
                combined_probs = self._combine_predictions(static_probs, dynamic_probs)
            else:
                # Use only static predictions
                combined_probs = static_probs
                
            # Ensure probabilities sum to 1
            normalized_probs = self._normalize_probabilities(combined_probs)
            
            return normalized_probs
            
        except Exception as e:
            logger.error(f"Error in EnsemblePredictor.predict: {e}")
            return [1/3, 1/3, 1/3]  # Default to equal probabilities
    
    def _preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before prediction.
        
        Args:
            features: DataFrame containing features
            
        Returns:
            Preprocessed features
        """
        # Clone to avoid modifying the original
        processed = features.copy()
        
        # Handle missing values
        for col in processed.columns:
            if processed[col].dtype in [np.float64, np.int64]:
                # Fill numeric columns with median
                processed[col] = processed[col].fillna(processed[col].median() if not processed[col].empty else 0)
            else:
                # Fill non-numeric columns with most frequent value or empty string
                processed[col] = processed[col].fillna(processed[col].mode()[0] if not processed[col].empty else '')
                
        return processed
    
    def _get_static_predictions(self, features: pd.DataFrame) -> List[float]:
        """
        Get predictions from the static model.
        
        Args:
            features: DataFrame containing preprocessed features
            
        Returns:
            List of probabilities [home_win_prob, draw_prob, away_win_prob]
        """
        if self.static_model is None:
            logger.warning("Static model not available, returning default probabilities")
            return [1/3, 1/3, 1/3]
            
        try:
            # Check if the model uses predict_proba (sklearn-like) or has custom prediction method
            if hasattr(self.static_model, 'predict_proba'):
                probs = self.static_model.predict_proba(features)[0]
                # Ensure we have 3 probability values (home win, draw, away win)
                if len(probs) == 3:
                    return probs.tolist()
                else:
                    logger.warning(f"Static model returned {len(probs)} probabilities, expected 3")
                    return [1/3, 1/3, 1/3]
            elif hasattr(self.static_model, 'predict'):
                # Custom prediction function
                prediction = self.static_model.predict(features)
                if isinstance(prediction, list) and len(prediction) == 3:
                    return prediction
                else:
                    logger.warning("Static model predict method returned unexpected format")
                    return [1/3, 1/3, 1/3]
            else:
                logger.warning("Static model has no predict_proba or predict method")
                return [1/3, 1/3, 1/3]
                
        except Exception as e:
            logger.error(f"Error getting predictions from static model: {e}")
            return [1/3, 1/3, 1/3]
    
    def _get_dynamic_predictions(self, features: pd.DataFrame, news_text: Optional[str]) -> List[float]:
        """
        Get predictions from the dynamic model, incorporating news text if available.
        
        Args:
            features: DataFrame containing preprocessed features
            news_text: Optional text with news about the match
            
        Returns:
            List of probabilities [home_win_prob, draw_prob, away_win_prob]
        """
        if self.dynamic_model is None:
            return [1/3, 1/3, 1/3]
            
        try:
            # Check if the dynamic model can use news text
            if hasattr(self.dynamic_model, 'predict_with_text') and news_text:
                # Model has a special method for text data
                return self.dynamic_model.predict_with_text(features, news_text)
            elif hasattr(self.dynamic_model, 'predict_proba'):
                # Standard sklearn-like interface
                probs = self.dynamic_model.predict_proba(features)[0]
                if len(probs) == 3:
                    return probs.tolist()
                else:
                    return [1/3, 1/3, 1/3]
            else:
                logger.warning("Dynamic model has no suitable prediction method")
                return [1/3, 1/3, 1/3]
                
        except Exception as e:
            logger.error(f"Error getting predictions from dynamic model: {e}")
            return [1/3, 1/3, 1/3]
    
    def _combine_predictions(self, static_probs: List[float], dynamic_probs: List[float], 
                           static_weight: float = 0.7) -> List[float]:
        """
        Combine predictions from static and dynamic models.
        
        Args:
            static_probs: Probabilities from static model
            dynamic_probs: Probabilities from dynamic model
            static_weight: Weight of static model (0-1)
            
        Returns:
            Combined probabilities
        """
        dynamic_weight = 1.0 - static_weight
        
        combined = [
            static_probs[0] * static_weight + dynamic_probs[0] * dynamic_weight,
            static_probs[1] * static_weight + dynamic_probs[1] * dynamic_weight,
            static_probs[2] * static_weight + dynamic_probs[2] * dynamic_weight
        ]
        
        return combined
    
    def _normalize_probabilities(self, probs: List[float]) -> List[float]:
        """
        Normalize probabilities to ensure they sum to 1.
        
        Args:
            probs: List of probabilities
            
        Returns:
            Normalized probabilities
        """
        total = sum(probs)
        if total > 0:
            return [p / total for p in probs]
        else:
            return [1/3, 1/3, 1/3]
