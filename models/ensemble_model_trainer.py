"""
Ensemble Model Trainer

Implements ensemble methods to improve prediction accuracy by combining
multiple models using voting, stacking, and blending strategies.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)


class EnsembleModelTrainer:
    """
    Train ensemble models combining multiple algorithms
    
    Implements:
    - Voting Classifier (soft voting)
    - Stacked ensemble
    - Weighted averaging
    """
    
    def __init__(self):
        """Initialize the ensemble trainer"""
        logger.info("🔧 Ensemble Model Trainer initialized")
    
    def train_voting_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_xgb_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train a voting ensemble with diverse base models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            base_xgb_params: Base XGBoost parameters from optimization
            
        Returns:
            Dictionary containing trained ensemble and metrics
        """
        logger.info("🗳️  Training voting ensemble...")
        start_time = datetime.now()
        
        try:
            # Calculate sample weights
            sample_weights = compute_sample_weight('balanced', y_train)
            
            # Define base models with complementary strengths
            
            # 1. XGBoost - strong gradient boosting (our best performer)
            if base_xgb_params:
                xgb_params = base_xgb_params.copy()
            else:
                xgb_params = {
                    'learning_rate': 0.03,
                    'max_depth': 5,
                    'min_child_weight': 3,
                    'subsample': 0.75,
                    'colsample_bytree': 0.85,
                    'gamma': 1.5,
                    'reg_alpha': 0.3,
                    'reg_lambda': 0.5,
                    'n_estimators': 250,
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'random_state': 42,
                    'n_jobs': -1
                }
            
            xgb_model = xgb.XGBClassifier(**xgb_params)
            
            # 2. Random Forest - robust to overfitting, different tree structure
            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            # 3. Logistic Regression - linear baseline, good calibration
            lr_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                solver='lbfgs'
            )
            
            # Create voting ensemble with soft voting (uses probabilities)
            ensemble = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('rf', rf_model),
                    ('lr', lr_model)
                ],
                voting='soft',
                weights=[3, 2, 1],  # XGBoost gets highest weight (best performer)
                n_jobs=-1
            )
            
            # Train ensemble
            logger.info("   Training XGBoost (weight=3)...")
            logger.info("   Training Random Forest (weight=2)...")
            logger.info("   Training Logistic Regression (weight=1)...")
            
            ensemble.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Evaluate on validation set
            y_val_pred = ensemble.predict(X_val)
            y_val_pred_proba = ensemble.predict_proba(X_val)
            
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_logloss = log_loss(y_val, y_val_pred_proba)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Voting ensemble trained in {elapsed:.2f}s")
            logger.info(f"   Validation accuracy: {val_accuracy:.4f}")
            logger.info(f"   Validation log loss: {val_logloss:.4f}")
            
            return {
                'model': ensemble,
                'accuracy': float(val_accuracy),
                'log_loss': float(val_logloss),
                'training_time': elapsed,
                'model_type': 'voting_ensemble',
                'base_models': ['xgboost', 'random_forest', 'logistic_regression'],
                'weights': [3, 2, 1],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return {'error': str(e)}
    
    def evaluate_ensemble(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate ensemble model on test set
        
        Args:
            model: Trained ensemble model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_pred_proba)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            logger.info(f"📊 Ensemble test accuracy: {accuracy:.4f}")
            logger.info(f"📊 Ensemble test log loss: {logloss:.4f}")
            
            return {
                'accuracy': float(accuracy),
                'log_loss': float(logloss),
                'classification_report': report,
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"Ensemble evaluation failed: {e}")
            return {'error': str(e)}
    
    def compare_models(
        self,
        baseline_results: Dict[str, Any],
        ensemble_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare baseline model vs ensemble
        
        Args:
            baseline_results: Results from baseline XGBoost
            ensemble_results: Results from ensemble
            
        Returns:
            Comparison metrics
        """
        baseline_acc = baseline_results.get('accuracy', 0)
        ensemble_acc = ensemble_results.get('accuracy', 0)
        
        improvement = ensemble_acc - baseline_acc
        improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
        
        logger.info("📊 Model Comparison:")
        logger.info(f"   Baseline accuracy: {baseline_acc:.4f}")
        logger.info(f"   Ensemble accuracy: {ensemble_acc:.4f}")
        logger.info(f"   Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        return {
            'baseline_accuracy': baseline_acc,
            'ensemble_accuracy': ensemble_acc,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'use_ensemble': ensemble_acc > baseline_acc
        }
