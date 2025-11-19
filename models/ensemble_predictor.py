"""
Ensemble Predictor for Football Match Outcomes
Combines multiple models for improved prediction accuracy
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils.logging_config import get_logger

from .enhanced_ml_pipeline import EnhancedMLPipeline

logger = get_logger(__name__)

class EnsemblePredictor:
    """Ensemble predictor combining multiple models for better accuracy."""
    
    def __init__(self, model_config: Optional[Dict] = None):
        """Initialize the ensemble predictor."""
        self.model_config = model_config or self._default_config()
        self.base_pipeline = EnhancedMLPipeline(model_config)
        self.ensemble_model = None
        self.is_trained = False
        
        # Initialize ensemble
        self._initialize_ensemble()
        logger.info("Ensemble Predictor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default ensemble configuration."""
        return {
            'ensemble': {
                'voting_type': 'soft',  # 'hard' or 'soft'
                'weights': [1, 1, 1]  # weights for each base model
            },
            'models': {
                'random_forest': {
                    'n_estimators': 150,
                    'max_depth': 12,
                    'random_state': 42
                },
                'gradient_boosting': {
                    'n_estimators': 150,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'random_state': 42
                },
                'logistic_regression': {
                    'random_state': 42,
                    'max_iter': 2000
                }
            },
            'features': {
                'team_performance_window': 15,
                'form_indicators': True,
                'head_to_head': True,
                'home_advantage': True,
                'weather_factors': False,
                'player_ratings': False
            }
        }
    
    def _initialize_ensemble(self):
        """Initialize the ensemble voting classifier."""
        # Get base models from pipeline
        base_models = [
            ('rf', self.base_pipeline.models['random_forest']),
            ('gb', self.base_pipeline.models['gradient_boosting']),
            ('lr', self.base_pipeline.models['logistic_regression'])
        ]
        
        # Handle different config structures
        ensemble_config = self.model_config.get('ensemble', {})
        voting_type = ensemble_config.get('voting_type', 'soft')
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=base_models,
            voting=voting_type
        )
        
        logger.info("Ensemble model initialized with voting strategy")
    
    def train(self, match_data: pd.DataFrame, target_column: str = 'result', test_size: float = 0.2) -> Dict[str, Any]:
        """Train the ensemble model."""
        logger.info("Starting ensemble training...")
        
        # Prepare data using base pipeline
        X, y = self.base_pipeline.prepare_training_data(match_data, target_column)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Encode target
        y_train_encoded = self.base_pipeline.scalers['target'].fit_transform(y_train)
        y_test_encoded = self.base_pipeline.scalers['target'].transform(y_test)
        
        # Scale features for the ensemble
        X_train_scaled = self.base_pipeline.scalers['numerical'].fit_transform(X_train)
        X_test_scaled = self.base_pipeline.scalers['numerical'].transform(X_test)
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        
        # For ensemble, we need to handle different models differently
        # Use original features for tree-based models, scaled for logistic regression
        # This requires custom training
        
        # Train individual models first
        individual_results = {}
        
        # Random Forest
        rf_model = self.base_pipeline.models['random_forest']
        rf_model.fit(X_train, y_train_encoded)
        rf_pred = rf_model.predict(X_test)
        individual_results['random_forest'] = {
            'accuracy': accuracy_score(y_test_encoded, rf_pred)
        }
        
        # Gradient Boosting
        gb_model = self.base_pipeline.models['gradient_boosting']
        gb_model.fit(X_train, y_train_encoded)
        gb_pred = gb_model.predict(X_test)
        individual_results['gradient_boosting'] = {
            'accuracy': accuracy_score(y_test_encoded, gb_pred)
        }
        
        # Logistic Regression (needs scaled features)
        lr_model = self.base_pipeline.models['logistic_regression']
        lr_model.fit(X_train_scaled, y_train_encoded)
        lr_pred = lr_model.predict(X_test_scaled)
        individual_results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test_encoded, lr_pred)
        }
        
        # Create ensemble predictions using majority voting
        ensemble_predictions = self._ensemble_predict(
            rf_pred, gb_pred, lr_pred
        )
        
        # Calculate ensemble metrics
        ensemble_accuracy = accuracy_score(y_test_encoded, ensemble_predictions)
        
        self.is_trained = True
        
        results = {
            'individual_models': individual_results,
            'ensemble': {
                'accuracy': ensemble_accuracy,
                'improvement': ensemble_accuracy - max([r['accuracy'] for r in individual_results.values()]),
                'voting_type': self.model_config.get('ensemble', {}).get('voting_type', 'soft')
            },
            'best_individual': max(individual_results.keys(), key=lambda k: individual_results[k]['accuracy']),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': X_train.shape[1]
        }
        
        logger.info(f"Ensemble training completed - Accuracy: {ensemble_accuracy:.3f}")
        logger.info(f"Best individual model: {results['best_individual']} ({individual_results[results['best_individual']]['accuracy']:.3f})")
        
        return results
    
    def _ensemble_predict(self, rf_pred: np.ndarray, gb_pred: np.ndarray, lr_pred: np.ndarray) -> np.ndarray:
        """Combine predictions from individual models."""
        ensemble_config = self.model_config.get('ensemble', {})
        voting_type = ensemble_config.get('voting_type', 'hard')
        
        if voting_type == 'hard':
            # Majority voting
            predictions = np.array([rf_pred, gb_pred, lr_pred])
            from scipy import stats
            ensemble_pred, _ = stats.mode(predictions, axis=0)
            return ensemble_pred.flatten()
        else:
            # Weighted average (simplified soft voting)
            weights = ensemble_config.get('weights', [1, 1, 1])
            if len(weights) != 3:
                weights = [1, 1, 1]
            
            # Simple weighted voting (assuming predictions are class indices)
            weighted_pred = (
                weights[0] * rf_pred + 
                weights[1] * gb_pred + 
                weights[2] * lr_pred
            ) / sum(weights)
            
            return np.round(weighted_pred).astype(int)
    
    def predict(self, match_data: pd.DataFrame) -> Dict[str, Any]:
        """Make ensemble predictions."""
        if not self.is_trained:
            logger.warning("Ensemble not trained yet. Training with sample data...")
            sample_data = self.base_pipeline._generate_live_data(200)
            self.train(live_data)
        
        logger.info("Making ensemble predictions...")
        
        # Prepare features
        features_df = self.base_pipeline.engineer_features(match_data)
        X = features_df[self.base_pipeline.feature_columns]
        
        # Scale features
        X_scaled = self.base_pipeline.scalers['numerical'].transform(X)
        
        # Get individual model predictions
        rf_pred = self.base_pipeline.models['random_forest'].predict(X)
        gb_pred = self.base_pipeline.models['gradient_boosting'].predict(X)
        lr_pred = self.base_pipeline.models['logistic_regression'].predict(X_scaled)
        
        # Combine predictions
        ensemble_pred = self._ensemble_predict(rf_pred, gb_pred, lr_pred)
        
        # Decode predictions
        final_predictions = self.base_pipeline.scalers['target'].inverse_transform(ensemble_pred)
        
        # Get individual probabilities for confidence
        individual_probs = {
            'random_forest': self.base_pipeline.models['random_forest'].predict_proba(X),
            'gradient_boosting': self.base_pipeline.models['gradient_boosting'].predict_proba(X),
            'logistic_regression': self.base_pipeline.models['logistic_regression'].predict_proba(X_scaled)
        }
        
        # Calculate ensemble confidence (average of max probabilities)
        ensemble_confidence = np.mean([
            prob.max(axis=1) for prob in individual_probs.values()
        ], axis=0)
        
        results = {
            'predictions': final_predictions.tolist(),
            'confidence': ensemble_confidence.tolist(),
            'individual_predictions': {
                'random_forest': self.base_pipeline.scalers['target'].inverse_transform(rf_pred).tolist(),
                'gradient_boosting': self.base_pipeline.scalers['target'].inverse_transform(gb_pred).tolist(),
                'logistic_regression': self.base_pipeline.scalers['target'].inverse_transform(lr_pred).tolist()
            },
            'individual_probabilities': {
                model: probs.tolist() for model, probs in individual_probs.items()
            },
            'prediction_classes': self.base_pipeline.scalers['target'].classes_.tolist(),
            'model_type': 'ensemble',
            'voting_strategy': self.model_config.get('ensemble', {}).get('voting_type', 'soft')
        }
        
        logger.info(f"Ensemble predictions completed: {len(final_predictions)} matches")
        return results
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get detailed performance metrics for the ensemble."""
        if not self.is_trained:
            return {'status': 'not_trained', 'message': 'Model needs to be trained first'}
        
        return {
            'status': 'trained',
            'ensemble_type': 'voting_classifier',
            'voting_strategy': self.model_config.get('ensemble', {}).get('voting_type', 'soft'),
            'base_models': list(self.base_pipeline.models.keys()),
            'feature_count': len(self.base_pipeline.feature_columns),
            'is_trained': self.is_trained
        }
    
    def save_ensemble(self, filepath: str):
        """Save the trained ensemble model."""
        if not self.is_trained:
            logger.warning("No trained ensemble to save")
            return
        
        ensemble_data = {
            'base_pipeline': self.base_pipeline,
            'ensemble_model': self.ensemble_model,
            'model_config': self.model_config,
            'is_trained': self.is_trained
        }
        
        joblib.dump(ensemble_data, filepath)
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load a trained ensemble model."""
        if not os.path.exists(filepath):
            logger.warning(f"Ensemble file not found: {filepath}")
            return
        
        ensemble_data = joblib.load(filepath)
        
        self.base_pipeline = ensemble_data['base_pipeline']
        self.ensemble_model = ensemble_data['ensemble_model']
        self.model_config = ensemble_data['model_config']
        self.is_trained = ensemble_data['is_trained']
        
        logger.info(f"Ensemble model loaded from {filepath}")
    
    def cross_validate(self, match_data: pd.DataFrame, target_column: str = 'result', cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation on the ensemble model."""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        # Prepare data
        X, y = self.base_pipeline.prepare_training_data(match_data, target_column)
        y_encoded = self.base_pipeline.scalers['target'].fit_transform(y)
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validate individual models
        cv_results = {}
        
        # Note: For proper ensemble cross-validation, we'd need to handle
        # the different scaling requirements properly. This is a simplified version.
        
        for model_name, model in self.base_pipeline.models.items():
            if model_name == 'logistic_regression':
                X_scaled = self.base_pipeline.scalers['numerical'].fit_transform(X)
                scores = cross_val_score(model, X_scaled, y_encoded, cv=skf, scoring='accuracy')
            else:
                scores = cross_val_score(model, X, y_encoded, cv=skf, scoring='accuracy')
            
            cv_results[model_name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'individual_scores': scores.tolist()
            }
        
        logger.info("Cross-validation completed")
        return cv_results
        logger.info("Cross-validation completed")
        return cv_results
