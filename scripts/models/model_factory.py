"""
Model factory for football prediction models.
Manages multiple models and provides a unified interface for predictions.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from goaldiggers.ml_integration.prediction_model import PredictionModel

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory for creating and managing prediction models.
    Provides a centralized way to load and use various ML models.
    """
    
    def __init__(self, models_dir: str = None, use_gpu: bool = None):
        """
        Initialize the model factory.
        
        Args:
            models_dir: Directory containing model files
            use_gpu: Whether to use GPU acceleration if available
        """
        # Default models directory
        self.models_dir = models_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu
            
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Dictionary to store initialized models
        self.models: Dict[str, PredictionModel] = {}
        
        # Default model configurations
        self.model_configs = {
            'xgboost_main': {
                'file': 'xgboost_match_predictor.json',
                'type': 'xgboost',
                'scaler': 'xgboost_match_scaler.pkl',
                'feature_map': 'feature_descriptions.json',
                'description': 'XGBoost main match prediction model'
            },
            'lightgbm_main': {
                'file': 'lightgbm_match_predictor.txt',
                'type': 'lightgbm',
                'scaler': 'lightgbm_match_scaler.pkl',
                'feature_map': 'feature_descriptions.json',
                'description': 'LightGBM main match prediction model'
            },
            'xgboost_goals': {
                'file': 'xgboost_goals_predictor.json',
                'type': 'xgboost',
                'scaler': 'xgboost_goals_scaler.pkl',
                'feature_map': 'goals_feature_descriptions.json',
                'description': 'XGBoost goals prediction model'
            },
            'ensemble': {
                'file': 'ensemble_meta_predictor.pkl',
                'type': 'sklearn',
                'scaler': None,
                'feature_map': 'ensemble_feature_descriptions.json',
                'description': 'Ensemble meta-model combining multiple predictions'
            }
        }
        
        logger.info(f"ModelFactory initialized with models_dir={self.models_dir}, use_gpu={self.use_gpu}")
        
    def get_model(self, model_name: str) -> Optional[PredictionModel]:
        """
        Get a prediction model by name. Initializes the model if not already loaded.
        
        Args:
            model_name: Name of the model to get
            
        Returns:
            Initialized prediction model or None if not found
        """
        # Return cached model if available
        if model_name in self.models:
            return self.models[model_name]
            
        # Check if model configuration exists
        if model_name not in self.model_configs:
            logger.error(f"Model configuration not found: {model_name}")
            return None
            
        # Get model configuration
        config = self.model_configs[model_name]
        
        try:
            # Build file paths
            model_path = os.path.join(self.models_dir, config['file'])
            scaler_path = os.path.join(self.models_dir, config['scaler']) if config.get('scaler') else None
            feature_map_path = os.path.join(self.models_dir, config['feature_map']) if config.get('feature_map') else None
            
            # Initialize model
            model = PredictionModel(
                model_path=model_path,
                model_type=config['type'],
                scaler_path=scaler_path,
                feature_map_path=feature_map_path,
                use_gpu=self.use_gpu
            )
            
            # Cache model
            self.models[model_name] = model
            
            logger.info(f"Initialized model '{model_name}': {config['description']}")
            return model
            
        except Exception as e:
            logger.error(f"Error initializing model '{model_name}': {e}")
            return None
            
    def predict(self, model_name: str, features: Any) -> Dict[str, Any]:
        """
        Generate predictions using the specified model.
        
        Args:
            model_name: Name of the model to use
            features: Input features for prediction
            
        Returns:
            Dictionary with prediction results
        """
        # Get model
        model = self.get_model(model_name)
        
        if not model:
            logger.error(f"Failed to get model '{model_name}'")
            return self._create_default_prediction()
            
        try:
            # Generate prediction
            prediction = model.predict(features)
            
            # Add feature importance
            importance = model.explain(features)
            
            # Return prediction with metadata
            return {
                'prediction': prediction,
                'feature_importance': importance,
                'model_info': {
                    'name': model_name,
                    'type': model.model_type,
                    'description': self.model_configs[model_name]['description']
                }
            }
        except Exception as e:
            logger.error(f"Error generating prediction with model '{model_name}': {e}")
            return self._create_default_prediction()
            
    def _create_default_prediction(self) -> Dict[str, Any]:
        """Create a default prediction result for error cases."""
        return {
            'prediction': {
                'home_win': 0.33,
                'draw': 0.34,
                'away_win': 0.33
            },
            'feature_importance': {},
            'model_info': {
                'name': 'default',
                'type': 'fallback',
                'description': 'Fallback prediction (equal probabilities)'
            }
        }
        
    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Get list of available models with their descriptions.
        
        Returns:
            List of dictionaries with model information
        """
        return [
            {
                'name': name,
                'type': config['type'],
                'description': config['description'],
                'loaded': name in self.models
            }
            for name, config in self.model_configs.items()
        ]
        
    def close_all(self):
        """Close all initialized models and free resources."""
        self.models.clear()
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # Clear CUDA cache if GPU was used
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("Closed all models and freed resources")


# Create a singleton instance for easy access
model_factory = ModelFactory()

# Convenience functions for direct use without creating a factory instance

def get_model(model_name: str) -> Optional[PredictionModel]:
    """Get a prediction model by name."""
    return model_factory.get_model(model_name)

def predict(model_name: str, features: Any) -> Dict[str, Any]:
    """Generate predictions using the specified model."""
    return model_factory.predict(model_name, features)

def get_available_models() -> List[Dict[str, str]]:
    """Get list of available models with their descriptions."""
    return model_factory.get_available_models()

def close_all():
    """Close all initialized models and free resources."""
    model_factory.close_all()
