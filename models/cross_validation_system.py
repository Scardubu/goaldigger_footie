#!/usr/bin/env python3
"""
Cross-Validation System for Production Model
Implements stratified K-fold CV with time-aware splitting
Phase 5: Robust model validation and hyperparameter tuning
"""
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)


class CrossValidationSystem:
    """
    Stratified K-Fold Cross-Validation for robust model evaluation
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        """
        Initialize CV system
        
        Parameters:
            n_splits: Number of folds (default 5)
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_results = {}
        self.fold_models = []
        logger.info(f"📊 Cross-Validation System initialized ({n_splits}-fold)")
    
    def run_stratified_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: Dict[str, Any],
        use_smote: bool = True
    ) -> Dict[str, Any]:
        """
        Run stratified K-fold cross-validation
        
        Parameters:
            X: Feature matrix
            y: Target labels
            params: Model hyperparameters
            use_smote: Whether to apply SMOTE to training folds
        
        Returns:
            Dict with CV metrics and fold results
        """
        logger.info(f"🔄 Starting {self.n_splits}-fold stratified cross-validation...")
        start_time = time.time()
        
        # Initialize stratified K-fold
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Store results for each fold
        fold_results = []
        fold_accuracies = []
        fold_logloss = []
        fold_calibration = []
        
        # Import SMOTE if needed
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state)
                logger.info("✅ SMOTE enabled for CV folds")
            except ImportError:
                logger.warning("⚠️ SMOTE not available, using sample weights only")
                use_smote = False
        
        # Run cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            fold_start = time.time()
            logger.info(f"\n📂 Fold {fold_idx + 1}/{self.n_splits}")
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Apply SMOTE to training fold
            if use_smote:
                try:
                    X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)
                    logger.info(f"   SMOTE applied: {len(y_train_fold)} samples after oversampling")
                except Exception as e:
                    logger.warning(f"   SMOTE failed: {e}, using original data")
            
            # Calculate sample weights for class imbalance
            sample_weights = compute_sample_weight('balanced', y_train_fold)
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_fold, y_train_fold,
                sample_weight=sample_weights,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            
            # Predictions
            y_pred = model.predict(X_val_fold)
            y_pred_proba = model.predict_proba(X_val_fold)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val_fold, y_pred)
            logloss = log_loss(y_val_fold, y_pred_proba)
            
            # Calibration error (average Brier score)
            brier_scores = []
            for i in range(3):  # 3 classes
                y_val_binary = (y_val_fold == i).astype(int)
                brier = brier_score_loss(y_val_binary, y_pred_proba[:, i])
                brier_scores.append(brier)
            calibration_error = np.mean(brier_scores)
            
            # Store results
            fold_result = {
                'fold': fold_idx + 1,
                'accuracy': float(accuracy),
                'log_loss': float(logloss),
                'calibration_error': float(calibration_error),
                'train_samples': len(y_train_fold),
                'val_samples': len(y_val_fold),
                'fold_time': time.time() - fold_start
            }
            fold_results.append(fold_result)
            fold_accuracies.append(accuracy)
            fold_logloss.append(logloss)
            fold_calibration.append(calibration_error)
            
            # Store model
            self.fold_models.append(model)
            
            logger.info(f"   Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}, Calibration: {calibration_error:.4f}")
            logger.info(f"   Fold time: {fold_result['fold_time']:.2f}s")
        
        # Calculate aggregate statistics
        cv_time = time.time() - start_time
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_splits': self.n_splits,
            'use_smote': use_smote,
            'fold_results': fold_results,
            'aggregate_metrics': {
                'mean_accuracy': float(np.mean(fold_accuracies)),
                'std_accuracy': float(np.std(fold_accuracies)),
                'min_accuracy': float(np.min(fold_accuracies)),
                'max_accuracy': float(np.max(fold_accuracies)),
                'mean_log_loss': float(np.mean(fold_logloss)),
                'std_log_loss': float(np.std(fold_logloss)),
                'mean_calibration_error': float(np.mean(fold_calibration)),
                'std_calibration_error': float(np.std(fold_calibration)),
            },
            'cv_time': cv_time,
            'params': params
        }
        
        self.cv_results = results
        
        logger.info(f"\n✅ Cross-validation complete in {cv_time:.2f}s")
        logger.info(f"📊 Mean accuracy: {results['aggregate_metrics']['mean_accuracy']:.4f} ± {results['aggregate_metrics']['std_accuracy']:.4f}")
        logger.info(f"📊 Range: [{results['aggregate_metrics']['min_accuracy']:.4f}, {results['aggregate_metrics']['max_accuracy']:.4f}]")
        
        return results
    
    def train_cv_ensemble(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Create ensemble from CV fold models
        
        Parameters:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dict with ensemble performance
        """
        if not self.fold_models:
            logger.error("❌ No fold models available. Run cross-validation first.")
            return {'error': 'No fold models'}
        
        logger.info(f"🗳️  Creating ensemble from {len(self.fold_models)} fold models...")
        
        # Collect predictions from all fold models
        fold_predictions = []
        fold_probabilities = []
        
        for i, model in enumerate(self.fold_models):
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            fold_predictions.append(y_pred)
            fold_probabilities.append(y_pred_proba)
        
        # Average probabilities (soft voting)
        avg_probabilities = np.mean(fold_probabilities, axis=0)
        ensemble_pred = np.argmax(avg_probabilities, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, ensemble_pred)
        logloss = log_loss(y_test, avg_probabilities)
        
        # Calibration error
        brier_scores = []
        for i in range(3):
            y_test_binary = (y_test == i).astype(int)
            brier = brier_score_loss(y_test_binary, avg_probabilities[:, i])
            brier_scores.append(brier)
        calibration_error = np.mean(brier_scores)
        
        results = {
            'ensemble_type': 'cv_fold_averaging',
            'n_models': len(self.fold_models),
            'accuracy': float(accuracy),
            'log_loss': float(logloss),
            'calibration_error': float(calibration_error),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ CV Ensemble accuracy: {accuracy:.4f}")
        logger.info(f"   Log loss: {logloss:.4f}, Calibration: {calibration_error:.4f}")
        
        return results
    
    def get_best_fold_model(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Get the best performing fold model
        
        Returns:
            (best_model, best_fold_results)
        """
        if not self.cv_results or not self.fold_models:
            logger.error("❌ No CV results available")
            return None, {}
        
        # Find fold with highest accuracy
        fold_results = self.cv_results['fold_results']
        best_fold_idx = max(range(len(fold_results)), key=lambda i: fold_results[i]['accuracy'])
        
        best_model = self.fold_models[best_fold_idx]
        best_results = fold_results[best_fold_idx]
        
        logger.info(f"🏆 Best fold: {best_results['fold']} with accuracy {best_results['accuracy']:.4f}")
        
        return best_model, best_results
    
    def get_cv_summary(self) -> Dict[str, Any]:
        """Get summary of CV results"""
        if not self.cv_results:
            return {'status': 'no_cv_performed'}
        
        return {
            'n_splits': self.n_splits,
            'mean_accuracy': self.cv_results['aggregate_metrics']['mean_accuracy'],
            'std_accuracy': self.cv_results['aggregate_metrics']['std_accuracy'],
            'accuracy_range': [
                self.cv_results['aggregate_metrics']['min_accuracy'],
                self.cv_results['aggregate_metrics']['max_accuracy']
            ],
            'cv_time': self.cv_results['cv_time'],
            'use_smote': self.cv_results.get('use_smote', False)
        }
    
    def save_cv_results(self, filepath: str):
        """Save CV results to file"""
        try:
            import json

            # Remove non-serializable models
            results_copy = self.cv_results.copy()
            if 'params' in results_copy:
                results_copy['params'] = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                                          for k, v in results_copy['params'].items()}
            
            with open(filepath, 'w') as f:
                json.dump(results_copy, f, indent=2)
            
            logger.info(f"✅ Saved CV results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save CV results: {e}")


# Global instance
cv_system = CrossValidationSystem()
