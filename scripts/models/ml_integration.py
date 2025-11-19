"""
Unified Machine Learning Integration Module for GoalDiggers Platform.
Provides a comprehensive interface for model management, training, prediction, and evaluation.
"""
import logging
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import ML libraries with error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

from database.db_manager import DatabaseManager
from utils.config import Config

logger = logging.getLogger(__name__)

class MLModelManager:
    """
    Comprehensive ML model manager for the GoalDiggers platform.
    Handles model training, evaluation, prediction, and deployment.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the ML model manager.
        
        Args:
            db_manager: Database manager instance for data access
        """
        self.db_manager = db_manager or DatabaseManager()
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_metadata = {}
        
        # Load configuration
        self.config = Config.get('models', {})
        self.model_dir = Path(self.config.get('enhanced_pipeline', {}).get('model_dir', 'models/trained'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize available models
        self.available_models = self._get_available_models()
        
        logger.info(f"ML Model Manager initialized with {len(self.available_models)} available model types")
    
    def _get_available_models(self) -> Dict[str, bool]:
        """Get available model types based on installed libraries."""
        return {
            'xgboost': XGBOOST_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'catboost': CATBOOST_AVAILABLE,
            'random_forest': True,  # Always available with sklearn
            'logistic_regression': True,
            'ensemble': True
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return [model for model, available in self.available_models.items() if available]
    
    def create_model(self, model_type: str, **kwargs) -> Any:
        """
        Create a model instance of the specified type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Model-specific parameters
            
        Returns:
            Model instance
        """
        if model_type not in self.available_models or not self.available_models[model_type]:
            raise ValueError(f"Model type '{model_type}' is not available")
        
        try:
            if model_type == 'xgboost':
                return self._create_xgboost_model(**kwargs)
            elif model_type == 'lightgbm':
                return self._create_lightgbm_model(**kwargs)
            elif model_type == 'catboost':
                return self._create_catboost_model(**kwargs)
            elif model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**kwargs)
            elif model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(**kwargs)
            elif model_type == 'ensemble':
                return self._create_ensemble_model(**kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Error creating {model_type} model: {e}")
            raise
    
    def _create_xgboost_model(self, **kwargs) -> Any:
        """Create XGBoost model with default parameters."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")
        
        default_params = self.config.get('xgboost', {})
        params = {**default_params, **kwargs}
        
        return xgb.XGBClassifier(**params)
    
    def _create_lightgbm_model(self, **kwargs) -> Any:
        """Create LightGBM model with default parameters."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not available")
        
        default_params = self.config.get('lightgbm', {})
        params = {**default_params, **kwargs}
        
        return lgb.LGBMClassifier(**params)
    
    def _create_catboost_model(self, **kwargs) -> Any:
        """Create CatBoost model with default parameters."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not available")
        
        default_params = self.config.get('catboost', {})
        params = {**default_params, **kwargs}
        
        return cb.CatBoostClassifier(**params)
    
    def _create_ensemble_model(self, **kwargs) -> Any:
        """Create ensemble model combining multiple base models."""
        from sklearn.ensemble import VotingClassifier

        # Get base models
        base_models = []
        weights = []
        
        for model_type in ['xgboost', 'lightgbm', 'random_forest']:
            if self.available_models.get(model_type, False):
                try:
                    model = self.create_model(model_type)
                    base_models.append((model_type, model))
                    weights.append(1.0)  # Equal weights
                except Exception as e:
                    logger.warning(f"Could not create {model_type} for ensemble: {e}")
        
        if len(base_models) < 2:
            raise ValueError("Need at least 2 base models for ensemble")
        
        return VotingClassifier(estimators=base_models, voting='soft', weights=weights)
    
    def train_model(self, model_type: str, X: pd.DataFrame, y: pd.Series, 
                   model_name: Optional[str] = None, **kwargs) -> str:
        """
        Train a model and save it.
        
        Args:
            model_type: Type of model to train
            X: Feature matrix
            y: Target variable
            model_name: Optional name for the model
            **kwargs: Additional training parameters
            
        Returns:
            Model identifier
        """
        try:
            # Create model
            model = self.create_model(model_type, **kwargs)
            
            # Preprocess data
            X_processed, scaler = self._preprocess_features(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            logger.info(f"Training {model_type} model...")
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            logger.info(f"Model training completed - Train score: {train_score:.4f}, Test score: {test_score:.4f}")
            
            # Generate model name if not provided
            if not model_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"{model_type}_{timestamp}"
            
            # Save model and components
            self._save_model(model_name, model, scaler, {
                'model_type': model_type,
                'train_score': train_score,
                'test_score': test_score,
                'training_date': datetime.now().isoformat(),
                'feature_count': X.shape[1],
                'sample_count': X.shape[0]
            })
            
            # Store in memory
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            self.model_metadata[model_name] = {
                'model_type': model_type,
                'train_score': train_score,
                'test_score': test_score,
                'training_date': datetime.now().isoformat()
            }
            
            return model_name
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            raise
    
    def _preprocess_features(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[StandardScaler]]:
        """Preprocess features for training."""
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X[numerical_cols])
            X[numerical_cols] = X_scaled
            return X.values, scaler
        
        return X.values, None
    
    def _save_model(self, model_name: str, model: Any, scaler: Optional[StandardScaler], 
                   metadata: Dict[str, Any]) -> None:
        """Save model and associated components."""
        model_path = self.model_dir / model_name
        model_path.mkdir(exist_ok=True)
        
        # Save model
        model_file = model_path / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler if exists
        if scaler:
            scaler_file = model_path / "scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save metadata
        metadata_file = model_path / "metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = self.model_dir / model_name
            
            # Load model
            model_file = model_path / "model.pkl"
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Load scaler if exists
            scaler = None
            scaler_file = model_path / "scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
            
            # Load metadata
            metadata = {}
            metadata_file = model_path / "metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Store in memory
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            self.model_metadata[model_name] = metadata
            
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def predict(self, model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions using a loaded model.
        
        Args:
            model_name: Name of the model to use
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            if not self.load_model(model_name):
                raise ValueError(f"Model {model_name} not found and could not be loaded")
        
        model = self.models[model_name]
        scaler = self.scalers.get(model_name)
        
        # Preprocess features
        X_processed = self._preprocess_for_prediction(X, scaler)
        
        # Make predictions
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
        
        return predictions, probabilities
    
    def _preprocess_for_prediction(self, X: pd.DataFrame, scaler: Optional[StandardScaler]) -> np.ndarray:
        """Preprocess features for prediction."""
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                X[col] = le.transform(X[col].astype(str))
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if scaler and len(numerical_cols) > 0:
            X_scaled = scaler.transform(X[numerical_cols])
            X[numerical_cols] = X_scaled
        
        return X.values
    
    def evaluate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate a model's performance.
        
        Args:
            model_name: Name of the model to evaluate
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            predictions, probabilities = self.predict(model_name, X)
            
            metrics = {
                'accuracy': accuracy_score(y, predictions),
                'precision': precision_score(y, predictions, average='weighted'),
                'recall': recall_score(y, predictions, average='weighted'),
                'f1_score': f1_score(y, predictions, average='weighted')
            }
            
            if probabilities is not None and len(np.unique(y)) == 2:
                metrics['roc_auc'] = roc_auc_score(y, probabilities[:, 1])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in self.model_metadata:
            if not self.load_model(model_name):
                return {}
        
        return self.model_metadata[model_name].copy()
    
    def list_models(self) -> List[str]:
        """List all available models."""
        models = []
        for model_dir in self.model_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "model.pkl").exists():
                models.append(model_dir.name)
        return models
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model and its files."""
        try:
            model_path = self.model_dir / model_name
            if model_path.exists():
                import shutil
                shutil.rmtree(model_path)
                
                # Remove from memory
                self.models.pop(model_name, None)
                self.scalers.pop(model_name, None)
                self.model_metadata.pop(model_name, None)
                
                logger.info(f"Model {model_name} deleted successfully")
                return True
            else:
                logger.warning(f"Model {model_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False

# Global instance for easy access
_model_manager = None

def get_model_manager() -> MLModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = MLModelManager()
    return _model_manager

def create_model(model_type: str, **kwargs) -> Any:
    """Create a model instance."""
    return get_model_manager().create_model(model_type, **kwargs)

def train_model(model_type: str, X: pd.DataFrame, y: pd.Series, 
                model_name: Optional[str] = None, **kwargs) -> str:
    """Train a model."""
    return get_model_manager().train_model(model_type, X, y, model_name, **kwargs)

def predict(model_name: str, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Make predictions."""
    return get_model_manager().predict(model_name, X)

def evaluate_model(model_name: str, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate a model."""
    return get_model_manager().evaluate_model(model_name, X, y)

def get_available_models() -> List[str]:
    """Get available model types."""
    return get_model_manager().get_available_models()

def list_models() -> List[str]:
    """List all available models."""
    return get_model_manager().list_models()
