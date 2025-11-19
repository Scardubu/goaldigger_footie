"""
Enhanced ML Pipeline for Football Betting Predictions
Incorporates advanced techniques for improved predictive accuracy:
- Ensemble methods (XGBoost, LightGBM, CatBoost)
- Advanced feature engineering
- Real-time model updates
- Cross-validation and hyperparameter optimization
- Model interpretability with SHAP
"""
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Core ML imports with defensive handling
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    xgb = None

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None

# Sklearn imports (always available)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Optional advanced imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import optuna
    from optuna.integration import LightGBMPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    LightGBMPruningCallback = None

from database.db_manager import DatabaseManager
from utils.config import Config


# Lazy import to avoid circular dependency at import time
def _get_system_monitor():
    try:
        from utils.system_monitor import SystemMonitor
        return SystemMonitor()
    except Exception:
        class _DummyMonitor:
            def start_operation(self, *_, **__):
                return "dummy"
            def end_operation(self, *_, **__):
                return {}
        return _DummyMonitor()

logger = logging.getLogger(__name__)

class EnhancedMLPipeline:
    """
    Enhanced ML pipeline for football betting predictions with advanced techniques.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the enhanced ML pipeline.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.system_monitor = _get_system_monitor()

        # Model components
        self.models = {}
        self.ensemble_model = None
        self.feature_importance = {}
        self.scalers = {}
        self.label_encoders = {}
        self.selected_features = None  # populated if feature selection applied

        # Configuration
        self.config = Config.get('models.enhanced_pipeline', {})
        self.model_dir = Path(self.config.get('model_dir', 'models/trained'))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Performance tracking
        self.performance_history = []
        self.last_training_time = None

        # Monitoring / inference tracking (previously mis-indented at class scope)
        self._last_inference_ms: Optional[float] = None
        self._total_inferences: int = 0
        self._last_prediction_timestamp: Optional[str] = None
        self._shap_explainer = None  # Lazy created
        self._shap_feature_names: Optional[List[str]] = None

        # Dependency availability tracking
        self.available_models = self._check_available_models()

        logger.info(f"Enhanced ML Pipeline initialized with models: {list(self.available_models.keys())}")
        if not any(self.available_models.values()):
            logger.warning("⚠️ No advanced ML libraries available, falling back to sklearn only")

        # Attempt to register this pipeline instance for global monitoring access
        try:  # pragma: no cover - best effort
            from models.pipeline_registry import register_pipeline as _reg_pipeline
            _reg_pipeline(self)
        except Exception:
            pass

    def _check_available_models(self) -> Dict[str, bool]:
        """Check which ML libraries are available."""
        return {
            'xgboost': XGB_AVAILABLE,
            'lightgbm': LGB_AVAILABLE,
            'catboost': CATBOOST_AVAILABLE,
            'shap': SHAP_AVAILABLE,
            'optuna': OPTUNA_AVAILABLE,
            'sklearn': True  # Always available
        }

    async def train_ensemble_model(
        self, 
        features: pd.DataFrame, 
        labels: pd.Series,
        validation_split: float = 0.2,
        optimize_hyperparameters: bool = True,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Train an ensemble model with multiple algorithms for improved accuracy.
        
        Args:
            features: Input features
            labels: Target labels
            validation_split: Fraction of data to use for validation
            optimize_hyperparameters: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with training results and performance metrics
        """
        operation_id = self.system_monitor.start_operation("train_ensemble_model")
        
        try:
            logger.info("Starting ensemble model training...")
            start_time = time.time()
            
            # --- Align features to expected columns ---
            expected_features = [col for col in features.columns]
            features_aligned = features.copy()
            # Fill missing columns with NaN
            for col in expected_features:
                if col not in features_aligned.columns:
                    features_aligned[col] = np.nan
            # Ensure column order
            features_aligned = features_aligned[expected_features]

            # Data preprocessing
            features_processed, labels_processed = self._preprocess_data(features_aligned, labels)
            
            # Log missing features if any
            missing_cols = [col for col in expected_features if col not in features.columns]
            if missing_cols:
                logger.warning(f"Missing features in training data: {missing_cols}")
            
            # Split data
            split_idx = int(len(features_processed) * (1 - validation_split))
            X_train, X_val = features_processed[:split_idx], features_processed[split_idx:]
            y_train, y_val = labels_processed[:split_idx], labels_processed[split_idx:]
            
            # Train individual models
            individual_models = {}

            # XGBoost (if available)
            if XGB_AVAILABLE:
                if optimize_hyperparameters and OPTUNA_AVAILABLE:
                    xgb_params = self._optimize_xgboost_hyperparameters(X_train, y_train, n_trials//3)
                else:
                    xgb_params = self.config.get('xgboost', {})

                xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
                xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
                individual_models['xgboost'] = xgb_model
                logger.info("✅ XGBoost model trained")
            else:
                logger.warning("⚠️ XGBoost not available, skipping")

            # LightGBM (if available)
            if LGB_AVAILABLE:
                if optimize_hyperparameters and OPTUNA_AVAILABLE:
                    lgb_params = self._optimize_lightgbm_hyperparameters(X_train, y_train, n_trials//3)
                else:
                    lgb_params = self.config.get('lightgbm', {})

                lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42)
                lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
                individual_models['lightgbm'] = lgb_model
                logger.info("✅ LightGBM model trained")
            else:
                logger.warning("⚠️ LightGBM not available, skipping")

            # CatBoost (if available)
            if CATBOOST_AVAILABLE:
                if optimize_hyperparameters and OPTUNA_AVAILABLE:
                    cat_params = self._optimize_catboost_hyperparameters(X_train, y_train, n_trials//3)
                else:
                    cat_params = self.config.get('catboost', {})

                cat_model = cb.CatBoostClassifier(**cat_params, random_state=42, verbose=False)
                cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)
                individual_models['catboost'] = cat_model
                logger.info("✅ CatBoost model trained")
            else:
                logger.warning("⚠️ CatBoost not available, skipping")

            # Random Forest (always available)
            rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
            rf_model.fit(X_train, y_train)
            individual_models['random_forest'] = rf_model
            logger.info("✅ Random Forest model trained")

            # Lightweight model-based feature selection using Random Forest importances
            try:
                feature_names = list(features_aligned.columns)
                importances = rf_model.feature_importances_
                top_k = int(self.config.get('feature_selection.top_k', min(28, len(feature_names))))
                if top_k and top_k < len(feature_names):
                    top_idx = list(importances.argsort()[-top_k:][::-1])
                    selected = [feature_names[i] for i in top_idx]
                    self.selected_features = selected
                    logger.info(f"Selected top {len(selected)} features: {selected}")
                    # If selected features fewer than original, transform training/validation sets
                    X_train = pd.DataFrame(X_train, columns=feature_names)[selected].values
                    X_val = pd.DataFrame(X_val, columns=feature_names)[selected].values
            except Exception as fe:
                logger.debug(f"Feature selection skipped or failed: {fe}")

            # Create ensemble model with available models
            estimators = []
            for model_name, model in individual_models.items():
                estimators.append((model_name, model))

            if len(estimators) == 0:
                logger.error("❌ No models available for ensemble")
                raise RuntimeError("No ML models available for training")
            elif len(estimators) == 1:
                # Single model fallback
                logger.warning("⚠️ Only one model available, using single model instead of ensemble")
                self.ensemble_model = estimators[0][1]
            else:
                # Create weighted ensemble based on available models
                weights = [1.0 / len(estimators)] * len(estimators)  # Equal weights
                self.ensemble_model = VotingClassifier(
                    estimators=estimators,
                    voting='soft',  # Use probability predictions
                    weights=weights
                )
            
            # Train ensemble
            # If selected_features is set, X_train already adjusted above
            self.ensemble_model.fit(X_train, y_train)
            
            # Evaluate performance
            y_pred_proba = self.ensemble_model.predict_proba(X_val)
            y_pred = self.ensemble_model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            logloss = log_loss(y_val, y_pred_proba)
            
            # Calculate feature importance
            self._calculate_feature_importance(individual_models, features_aligned.columns)
            # Log feature importances for each model
            for model_name, importances in self.feature_importance.items():
                logger.info(f"Feature importances for {model_name}: {importances}")
            
            # Store models
            self.models = individual_models
            
            # Performance tracking
            training_time = time.time() - start_time
            performance_metrics = {
                'accuracy': accuracy,
                'log_loss': logloss,
                'training_time': training_time,
                'n_features': len(features_aligned.columns),
                'n_samples': len(features_aligned),
                'timestamp': datetime.now().isoformat()
            }
            
            self.performance_history.append(performance_metrics)
            self.last_training_time = datetime.now()
            
            # Save models
            await self._save_models()
            
            logger.info(f"Ensemble model training completed. Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}")
            
            return {
                'success': True,
                'metrics': performance_metrics,
                'feature_importance': self.feature_importance,
                'models_trained': list(individual_models.keys())
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            self.system_monitor.end_operation(operation_id)

    def _preprocess_data(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess features and labels for training."""
        # Handle missing values
        features = features.fillna(features.median())
        
        # Encode categorical variables
        for col in features.select_dtypes(include=['object']).columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features[col] = self.label_encoders[col].fit_transform(features[col])
            else:
                features[col] = self.label_encoders[col].transform(features[col])
        
        # Scale numerical features
        numerical_cols = features.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            if 'numerical' not in self.scalers:
                self.scalers['numerical'] = StandardScaler()
                features[numerical_cols] = self.scalers['numerical'].fit_transform(features[numerical_cols])
            else:
                features[numerical_cols] = self.scalers['numerical'].transform(features[numerical_cols])
        
        return features.values, labels.values

    def _optimize_xgboost_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE or not XGB_AVAILABLE:
            logger.warning("⚠️ Optuna or XGBoost not available, using default parameters")
            return self.config.get('xgboost', {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            })

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_log_loss')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

    def _optimize_lightgbm_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE or not LGB_AVAILABLE:
            logger.warning("⚠️ Optuna or LightGBM not available, using default parameters")
            return self.config.get('lightgbm', {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            })

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            model = lgb.LGBMClassifier(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_log_loss')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

    def _optimize_catboost_hyperparameters(self, X: np.ndarray, y: np.ndarray, n_trials: int) -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE or not CATBOOST_AVAILABLE:
            logger.warning("⚠️ Optuna or CatBoost not available, using default parameters")
            return self.config.get('catboost', {
                'iterations': 200,
                'depth': 6,
                'learning_rate': 0.1,
                'l2_leaf_reg': 3,
                'verbose': False
            })

        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_state': 42,
                'verbose': False
            }
            
            model = cb.CatBoostClassifier(**params)
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_log_loss')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params

    def _calculate_feature_importance(self, models: Dict[str, Any], feature_names: List[str]) -> None:
        """Calculate and store feature importance from all models."""
        self.feature_importance = {}
        
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                self.feature_importance[model_name] = dict(zip(feature_names, importance))
        
        # Calculate ensemble feature importance
        if self.feature_importance:
            ensemble_importance = {}
            for feature in feature_names:
                importance_sum = 0
                count = 0
                for model_importance in self.feature_importance.values():
                    if feature in model_importance:
                        importance_sum += model_importance[feature]
                        count += 1
                if count > 0:
                    ensemble_importance[feature] = importance_sum / count
            
            self.feature_importance['ensemble'] = ensemble_importance

    async def predict_match_outcome(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict match outcome using the ensemble model.
        
        Args:
            features: Match features
            
        Returns:
            Dictionary with prediction probabilities and confidence
        """
        if self.ensemble_model is None:
            return {'error': 'Model not trained'}
        
        try:
            import time as _time
            _t0 = _time.time()
            # Preprocess features
            features_processed, _ = self._preprocess_data(features, pd.Series([0]))  # Dummy labels
            
            # Get ensemble prediction
            ensemble_proba = self.ensemble_model.predict_proba(features_processed)
            
            # Get individual model predictions
            individual_predictions = {}
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    individual_predictions[name] = model.predict_proba(features_processed)[0]
            
            # Calculate prediction confidence
            confidence = self._calculate_prediction_confidence(individual_predictions)
            
            # Get feature importance for this prediction
            feature_importance = self._get_prediction_feature_importance(features_processed[0])
            
            duration_ms = (_time.time() - _t0) * 1000.0
            self._last_inference_ms = round(duration_ms, 3)
            self._total_inferences += 1
            self._last_prediction_timestamp = datetime.now().isoformat()

            return {
                'home_win': float(ensemble_proba[0][0]),
                'draw': float(ensemble_proba[0][1]),
                'away_win': float(ensemble_proba[0][2]),
                'confidence': confidence,
                'individual_predictions': individual_predictions,
                'feature_importance': feature_importance,
                'prediction_timestamp': self._last_prediction_timestamp,
                'latency_ms': self._last_inference_ms
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {'error': str(e)}

    def _calculate_prediction_confidence(self, individual_predictions: Dict[str, np.ndarray]) -> float:
        """Calculate confidence based on agreement between models."""
        if not individual_predictions:
            return 0.5
        
        # Calculate variance in predictions
        predictions_array = np.array(list(individual_predictions.values()))
        variance = np.var(predictions_array, axis=0).mean()
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = max(0.1, 1.0 - variance)
        return float(confidence)

    def _get_prediction_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for a specific prediction."""
        if 'ensemble' not in self.feature_importance:
            return {}
        
        # Use SHAP for local feature importance
        try:
            if 'xgboost' in self.models:
                explainer = shap.TreeExplainer(self.models['xgboost'])
                shap_values = explainer.shap_values(features.reshape(1, -1))
                
                if isinstance(shap_values, list):
                    # Multi-class case
                    shap_values = np.array(shap_values).sum(axis=0)
                
                feature_names = list(self.feature_importance['ensemble'].keys())
                importance_dict = dict(zip(feature_names, np.abs(shap_values[0])))
                
                # Normalize
                total_importance = sum(importance_dict.values())
                if total_importance > 0:
                    importance_dict = {k: v/total_importance for k, v in importance_dict.items()}
                
                return importance_dict
        except Exception as e:
            logger.warning(f"Could not calculate SHAP importance: {e}")
        
        # Fallback to global feature importance
        return self.feature_importance.get('ensemble', {})

    async def _save_models(self) -> None:
        """Save trained models and metadata."""
        try:
            # Save ensemble model
            ensemble_path = self.model_dir / 'ensemble_model.pkl'
            with open(ensemble_path, 'wb') as f:
                pickle.dump(self.ensemble_model, f)
            
            # Save individual models
            for name, model in self.models.items():
                model_path = self.model_dir / f'{name}_model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save metadata
            metadata = {
                'feature_importance': self.feature_importance,
                'performance_history': self.performance_history,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'scalers': self.scalers,
                'label_encoders': self.label_encoders
            }
            
            metadata_path = self.model_dir / 'metadata.pkl'
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info("Models and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    async def load_models(self) -> bool:
        """Load trained models and metadata."""
        try:
            # Load ensemble model
            ensemble_path = self.model_dir / 'ensemble_model.pkl'
            if ensemble_path.exists():
                with open(ensemble_path, 'rb') as f:
                    self.ensemble_model = pickle.load(f)
            
            # Load individual models
            model_names = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
            for name in model_names:
                model_path = self.model_dir / f'{name}_model.pkl'
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            # Load metadata
            metadata_path = self.model_dir / 'metadata.pkl'
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.feature_importance = metadata.get('feature_importance', {})
                    self.performance_history = metadata.get('performance_history', [])
                    self.scalers = metadata.get('scalers', {})
                    self.label_encoders = metadata.get('label_encoders', {})
                    
                    last_training_str = metadata.get('last_training_time')
                    if last_training_str:
                        self.last_training_time = datetime.fromisoformat(last_training_str)
            
            logger.info("Models and metadata loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and history."""
        if not self.performance_history:
            return {'error': 'No performance history available'}
        
        latest_metrics = self.performance_history[-1]
        
        return {
            'latest_metrics': latest_metrics,
            'performance_history': self.performance_history,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'feature_importance': self.feature_importance
        }

    def get_confidence(self) -> float:
        """Return a simple confidence estimate for the current pipeline based on latest accuracy."""
        if not self.performance_history:
            return 0.5
        return float(self.performance_history[-1].get('accuracy', 0.5))

    def get_version(self) -> str:
        """Return a version string for the pipeline based on last training time."""
        if self.last_training_time:
            return f"enhanced_pipeline:{self.last_training_time.isoformat()}"
        return "enhanced_pipeline:untrained"

    # ---------------- Monitoring & Explainability Extensions ---------------- #
    def get_monitoring_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight snapshot for dashboards/metrics."""
        return {
            'version': self.get_version(),
            'trained': self.ensemble_model is not None,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'last_inference_ms': self._last_inference_ms,
            'total_inferences': self._total_inferences,
            'last_prediction_timestamp': self._last_prediction_timestamp,
            'models_available': list(self.models.keys()),
            'selected_features': self.selected_features,
            'performance_latest': self.performance_history[-1] if self.performance_history else None,
        }

    def _ensure_shap_explainer(self):
        if self._shap_explainer is not None:
            return
        if not SHAP_AVAILABLE:
            return
        try:
            # Prefer tree-based model for speed/accuracy
            model = None
            for preferred in ('xgboost','lightgbm','catboost','random_forest'):
                if preferred in self.models:
                    model = self.models[preferred]
                    break
            if model is None:
                return
            self._shap_explainer = shap.TreeExplainer(model)
            # Attempt to infer feature names from global importance
            if 'ensemble' in self.feature_importance:
                self._shap_feature_names = list(self.feature_importance['ensemble'].keys())
        except Exception as e:
            logger.debug(f"SHAP explainer creation skipped: {e}")
            self._shap_explainer = None

    async def predict_with_explanations(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Predict with optional SHAP local explanations and monitoring data.

        Returns structured dict: probabilities, confidence, explanations, latency.
        """
        base = await self.predict_match_outcome(features)
        if 'error' in base:
            return base
        # Add SHAP explanations if available
        self._ensure_shap_explainer()
        explanations = None
        try:
            if self._shap_explainer is not None:
                # Reprocess features for consistent scaling
                f_processed, _ = self._preprocess_data(features, pd.Series([0]))
                shap_vals = self._shap_explainer.shap_values(f_processed)
                # Handle multi-class -> sum abs contributions across classes
                import numpy as _np
                if isinstance(shap_vals, list):
                    shap_arr = _np.sum(_np.abs(_np.array(shap_vals)), axis=0)[0]
                else:
                    shap_arr = _np.abs(shap_vals[0])
                names = self._shap_feature_names or [f'feature_{i}' for i in range(len(shap_arr))]
                explanations = {n: float(v) for n, v in zip(names, shap_arr)}
                total = sum(explanations.values())
                if total > 0:
                    explanations = {k: v/total for k,v in explanations.items()}
        except Exception as e:
            logger.debug(f"Local explanation generation failed: {e}")
            explanations = None

        base['explanations'] = explanations
        base['monitoring'] = self.get_monitoring_snapshot()
        return base

    async def update_model_performance(self, actual_outcomes: List[int], predicted_probabilities: List[np.ndarray]) -> Dict[str, Any]:
        """
        Update model performance with new data.
        
        Args:
            actual_outcomes: Actual match outcomes
            predicted_probabilities: Predicted probabilities
            
        Returns:
            Updated performance metrics
        """
        try:
            # Calculate new metrics
            accuracy = accuracy_score(actual_outcomes, np.argmax(predicted_probabilities, axis=1))
            logloss = log_loss(actual_outcomes, predicted_probabilities)
            
            # Update performance history
            new_metrics = {
                'accuracy': accuracy,
                'log_loss': logloss,
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(actual_outcomes)
            }
            
            self.performance_history.append(new_metrics)
            
            # Save updated metadata
            await self._save_models()
            
            return {
                'success': True,
                'new_metrics': new_metrics,
                'performance_trend': self._calculate_performance_trend()
            }
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_performance_trend(self) -> Dict[str, Any]:
        """Calculate performance trend over time."""
        if len(self.performance_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_metrics = self.performance_history[-5:]  # Last 5 updates
        
        accuracy_trend = np.mean([m['accuracy'] for m in recent_metrics])
        logloss_trend = np.mean([m['log_loss'] for m in recent_metrics])
        
        return {
            'trend': 'improving' if accuracy_trend > 0.5 and logloss_trend < 1.0 else 'stable',
            'avg_accuracy': accuracy_trend,
            'avg_logloss': logloss_trend
        } 