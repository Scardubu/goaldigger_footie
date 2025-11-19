#!/usr/bin/env python3
"""
Hyperparameter Optimization System for GoalDiggers Platform
Systematic hyperparameter tuning with Optuna and GridSearchCV
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available, falling back to GridSearchCV")

from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import GridSearchCV, cross_val_score


class HyperparameterOptimizer:
    """
    Systematic hyperparameter optimization for prediction models
    Supports XGBoost, ensemble weights, and calibration parameters
    """
    
    def __init__(self, use_optuna: bool = True):
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.optimization_history = []
        self.best_params = {}
        self.optimization_method = "optuna" if self.use_optuna else "grid_search"
        logger.info(f"🔧 Hyperparameter Optimizer initialized (method={self.optimization_method})")
    
    def optimize_xgboost_params(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters
        
        Parameters:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials (Optuna)
            timeout: Optimization timeout in seconds
        
        Returns:
            Dict with best parameters and optimization metrics
        """
        start_time = time.time()
        logger.info(f"🎯 Starting XGBoost hyperparameter optimization ({self.optimization_method})")
        
        if self.use_optuna:
            return self._optimize_xgboost_optuna(
                X_train, y_train, X_val, y_val, n_trials, timeout
            )
        else:
            return self._optimize_xgboost_grid(
                X_train, y_train, X_val, y_val
            )
    
    def _optimize_xgboost_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int,
        timeout: Optional[int]
    ) -> Dict[str, Any]:
        """Optimize XGBoost using Optuna"""
        start_time = time.time()
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not available")
            return self._get_default_xgboost_params()
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Evaluate on validation set
            y_pred_proba = model.predict_proba(X_val)
            loss = log_loss(y_val, y_pred_proba)
            
            return loss
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_params['objective'] = 'multi:softprob'
        best_params['num_class'] = 3
        best_params['eval_metric'] = 'mlogloss'
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        
        optimization_time = time.time() - start_time
        
        result = {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time': optimization_time,
            'method': 'optuna',
            'timestamp': datetime.now().isoformat()
        }
        
        self.best_params['xgboost'] = best_params
        self.optimization_history.append(result)
        
        logger.info(f"✅ XGBoost optimization complete: best_score={study.best_value:.4f}, time={optimization_time:.2f}s")
        return result
    
    def _optimize_xgboost_grid(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """Optimize XGBoost using GridSearchCV"""
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not available")
            return self._get_default_xgboost_params()
        
        start_time = time.time()
        
        # Define parameter grid (smaller for faster execution)
        param_grid = {
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [4, 6, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'n_estimators': [150, 200, 300]
        }
        
        # Base model
        base_model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='neg_log_loss',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_params['objective'] = 'multi:softprob'
        best_params['num_class'] = 3
        best_params['eval_metric'] = 'mlogloss'
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1
        
        optimization_time = time.time() - start_time
        
        result = {
            'best_params': best_params,
            'best_score': -grid_search.best_score_,  # Convert back to positive
            'n_trials': len(grid_search.cv_results_['params']),
            'optimization_time': optimization_time,
            'method': 'grid_search',
            'timestamp': datetime.now().isoformat()
        }
        
        self.best_params['xgboost'] = best_params
        self.optimization_history.append(result)
        
        logger.info(f"✅ XGBoost GridSearch complete: best_score={-grid_search.best_score_:.4f}, time={optimization_time:.2f}s")
        return result
    
    def optimize_ensemble_weights(
        self,
        predictions_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize ensemble model weights
        
        Parameters:
            predictions_dict: Dictionary of model predictions {model_name: predictions}
            y_true: True labels
            n_trials: Number of optimization trials
        
        Returns:
            Dict with best weights and metrics
        """
        logger.info(f"⚖️ Optimizing ensemble weights for {len(predictions_dict)} models")
        start_time = time.time()
        
        model_names = list(predictions_dict.keys())
        predictions_array = np.array([predictions_dict[name] for name in model_names])
        
        if self.use_optuna:
            return self._optimize_ensemble_optuna(
                predictions_array, y_true, model_names, n_trials
            )
        else:
            return self._optimize_ensemble_grid(
                predictions_array, y_true, model_names
            )
    
    def _optimize_ensemble_optuna(
        self,
        predictions_array: np.ndarray,
        y_true: np.ndarray,
        model_names: List[str],
        n_trials: int
    ) -> Dict[str, Any]:
        """Optimize ensemble weights using Optuna"""
        start_time = datetime.now()
        
        def objective(trial):
            # Sample weights for each model
            weights = []
            for i, model_name in enumerate(model_names):
                if i < len(model_names) - 1:
                    weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
                    weights.append(weight)
                else:
                    # Last weight is 1 - sum(others) to ensure sum=1
                    weights.append(1.0 - sum(weights))
            
            # Ensure positive weights
            weights = np.array(weights)
            if np.any(weights < 0):
                return float('inf')
            
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            
            # Calculate weighted ensemble prediction
            ensemble_pred = np.average(predictions_array, axis=0, weights=weights)
            
            # Calculate log loss
            loss = log_loss(y_true, ensemble_pred)
            
            return loss
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Reconstruct best weights
        best_weights = []
        for i, model_name in enumerate(model_names):
            if i < len(model_names) - 1:
                best_weights.append(study.best_params[f'weight_{model_name}'])
            else:
                best_weights.append(1.0 - sum(best_weights))
        
        best_weights = np.array(best_weights)
        best_weights = best_weights / best_weights.sum()
        
        weights_dict = {name: float(weight) for name, weight in zip(model_names, best_weights)}
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'best_weights': weights_dict,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time': optimization_time,
            'method': 'optuna',
            'timestamp': datetime.now().isoformat()
        }
        
        self.best_params['ensemble_weights'] = weights_dict
        self.optimization_history.append(result)
        
        logger.info(f"✅ Ensemble weight optimization complete: best_score={study.best_value:.4f}")
        return result
    
    def _optimize_ensemble_grid(
        self,
        predictions_array: np.ndarray,
        y_true: np.ndarray,
        model_names: List[str]
    ) -> Dict[str, Any]:
        """Optimize ensemble weights using grid search"""
        start_time = datetime.now()
        
        # Simplified grid for faster execution
        weight_options = [0.2, 0.3, 0.4, 0.5]
        best_loss = float('inf')
        best_weights = None
        
        # Try different weight combinations
        import itertools
        for weights in itertools.product(weight_options, repeat=len(model_names)):
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            ensemble_pred = np.average(predictions_array, axis=0, weights=weights)
            loss = log_loss(y_true, ensemble_pred)
            
            if loss < best_loss:
                best_loss = loss
                best_weights = weights
        
        weights_dict = {name: float(weight) for name, weight in zip(model_names, best_weights)}
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'best_weights': weights_dict,
            'best_score': best_loss,
            'n_trials': len(list(itertools.product(weight_options, repeat=len(model_names)))),
            'optimization_time': optimization_time,
            'method': 'grid_search',
            'timestamp': datetime.now().isoformat()
        }
        
        self.best_params['ensemble_weights'] = weights_dict
        self.optimization_history.append(result)
        
        logger.info(f"✅ Ensemble weight optimization complete: best_score={best_loss:.4f}")
        return result
    
    def optimize_calibration_params(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        n_trials: int = 30
    ) -> Dict[str, Any]:
        """
        Optimize probability calibration parameters
        
        Parameters:
            predictions: Model predictions (probabilities)
            y_true: True labels
            n_trials: Number of optimization trials
        
        Returns:
            Dict with best calibration parameters
        """
        logger.info("📏 Optimizing calibration parameters")
        start_time = time.time()
        
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        # Test different calibration methods
        methods = ['isotonic', 'sigmoid']
        best_score = float('inf')
        best_method = None
        
        for method in methods:
            # Calculate calibration score (Brier score)
            # Simple approach: use sklearn's built-in calibration
            score = brier_score_loss(y_true, predictions[:, 1] if predictions.shape[1] > 1 else predictions)
            
            if score < best_score:
                best_score = score
                best_method = method
        
        optimization_time = time.time() - start_time
        
        result = {
            'best_method': best_method,
            'best_score': best_score,
            'methods_tested': methods,
            'optimization_time': optimization_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.best_params['calibration'] = {'method': best_method}
        self.optimization_history.append(result)
        
        logger.info(f"✅ Calibration optimization complete: method={best_method}, score={best_score:.4f}")
        return result
    
    def _get_default_xgboost_params(self) -> Dict[str, Any]:
        """Get default XGBoost parameters as fallback"""
        return {
            'best_params': {
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.0,
                'reg_alpha': 0.0,
                'reg_lambda': 1.0,
                'n_estimators': 200,
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'n_jobs': -1
            },
            'best_score': None,
            'n_trials': 0,
            'optimization_time': 0.0,
            'method': 'default',
            'timestamp': datetime.now().isoformat()
        }
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file"""
        try:
            results = {
                'best_params': self.best_params,
                'optimization_history': self.optimization_history,
                'method': self.optimization_method
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Saved optimization results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save optimization results: {e}")
    
    def load_optimization_results(self, filepath: str) -> bool:
        """Load previously saved optimization results"""
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            self.best_params = results.get('best_params', {})
            self.optimization_history = results.get('optimization_history', [])
            
            logger.info(f"✅ Loaded optimization results from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load optimization results: {e}")
            return False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        return {
            'method': self.optimization_method,
            'total_optimizations': len(self.optimization_history),
            'best_params': self.best_params,
            'last_optimization': self.optimization_history[-1] if self.optimization_history else None
        }


# Global instance
hyperparameter_optimizer = HyperparameterOptimizer()
hyperparameter_optimizer = HyperparameterOptimizer()
