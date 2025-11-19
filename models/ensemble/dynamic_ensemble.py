#!/usr/bin/env python3
"""
Dynamic Ensemble Model for GoalDiggers Platform

This module contains the DynamicEnsemble class used for real-time predictions
with multiple ML models combined through weighted voting.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DynamicEnsemble:
    """
    Dynamic ensemble model for real-time predictions.
    
    This class combines multiple trained ML models using weighted voting
    to produce more robust and accurate predictions for football matches.
    """
    
    def __init__(self, models: Dict[str, Any], scaler: Any, feature_columns: List[str]):
        """
        Initialize the dynamic ensemble.
        
        Args:
            models: Dictionary of trained ML models
            scaler: Fitted scaler for feature normalization
            feature_columns: List of feature column names
        """
        self.models = models
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.weights = {
            'xgboost': 0.4,
            'random_forest': 0.3,
            'gradient_boosting': 0.2,
            'logistic_regression': 0.1
        }
        # Normalize weights to sum to 1
        total_w = sum(self.weights.values())
        if total_w > 0:
            for k in list(self.weights.keys()):
                self.weights[k] = self.weights[k] / total_w
        self.created_at = datetime.now()
        
        logger.info(f"DynamicEnsemble initialized with {len(models)} models")
    
    def predict_proba(self, X):
        """
        Predict class probabilities using weighted ensemble.
        
        Args:
            X: Input features (DataFrame or array)
            
        Returns:
            Array of prediction probabilities
        """
        try:
            # Handle DataFrame input
            if isinstance(X, pd.DataFrame):
                # If DataFrame has extra/missing columns, reindex safely
                X = X.reindex(columns=self.feature_columns, fill_value=0.0)
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            
            # Get weighted predictions from all models
            weighted_probs = None
            successful_models = 0
            
            for name, model in self.models.items():
                try:
                    probs = model.predict_proba(X_scaled)
                    weight = self.weights.get(name, 0.25)
                    
                    if weighted_probs is None:
                        weighted_probs = probs * weight
                    else:
                        weighted_probs += probs * weight
                    
                    successful_models += 1
                    
                except Exception as e:
                    logger.warning(f"Model {name} prediction failed: {e}")
                    continue
            
            if weighted_probs is None or successful_models == 0:
                # Fallback to uniform distribution
                logger.warning("All models failed, using uniform distribution")
                n_samples = X_scaled.shape[0] if len(X_scaled.shape) > 1 else 1
                return np.array([[1/3, 1/3, 1/3]] * n_samples)
            
            # Normalize probabilities to ensure they sum to 1
            row_sums = weighted_probs.sum(axis=1, keepdims=True)
            weighted_probs = np.divide(weighted_probs, row_sums, 
                                     out=np.full_like(weighted_probs, 1/3), 
                                     where=row_sums != 0)
            
            return weighted_probs
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            n_samples = 1
            if hasattr(X, 'shape') and len(X.shape) > 1:
                n_samples = X.shape[0]
            elif isinstance(X, pd.DataFrame):
                n_samples = len(X)
            return np.array([[1/3, 1/3, 1/3]] * n_samples)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Array of predicted class labels
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def get_feature_importance(self):
        """
        Get aggregated feature importance from all models.
        
        Returns:
            Dictionary of feature importance scores
        """
        try:
            feature_importance = {}
            total_weight = 0
            
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    weight = self.weights.get(name, 0.25)
                    importances = model.feature_importances_
                    
                    for i, importance in enumerate(importances):
                        if i < len(self.feature_columns):
                            feature_name = self.feature_columns[i]
                            if feature_name not in feature_importance:
                                feature_importance[feature_name] = 0
                            feature_importance[feature_name] += importance * weight
                    
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                for feature in feature_importance:
                    feature_importance[feature] /= total_weight
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def get_model_info(self):
        """
        Get information about the ensemble models.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_count': len(self.models),
            'model_names': list(self.models.keys()),
            'weights': self.weights,
            'feature_count': len(self.feature_columns),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update model weights for ensemble voting.
        
        Args:
            new_weights: Dictionary of new weights for each model
        """
        for model_name, weight in new_weights.items():
            if model_name in self.weights:
                self.weights[model_name] = weight
                logger.info(f"Updated weight for {model_name}: {weight}")
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for model_name in self.weights:
                self.weights[model_name] /= total_weight


def create_dynamic_ensemble(models: Dict[str, Any], scaler: Any, feature_columns: List[str]) -> DynamicEnsemble:
    """
    Factory function to create a DynamicEnsemble instance.
    
    Args:
        models: Dictionary of trained ML models
        scaler: Fitted scaler for feature normalization
        feature_columns: List of feature column names
        
    Returns:
        DynamicEnsemble instance
    """
    return DynamicEnsemble(models, scaler, feature_columns)
