"""
Models module for GoalDiggers platform.
Provides machine learning model integration and management.
"""

# Import key model components
try:
    from .model_factory import ModelFactory
    from .value_bet_analyzer import ValueBetAnalyzer
except ImportError as e:
    print(f"Warning: Could not import some model components: {e}")

# Create a simple ML integration module to fix import issues
class SimpleMLIntegration:
    """Simple ML integration to fix import issues."""
    
    def __init__(self):
        self.available_models = ['xgboost', 'lightgbm', 'random_forest']
    
    def get_available_models(self):
        """Get list of available models."""
        return self.available_models
    
    def create_model(self, model_type: str, **kwargs):
        """Create a model instance."""
        if model_type == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(**kwargs)
            except ImportError:
                raise ImportError("XGBoost not available")
        elif model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(**kwargs)
            except ImportError:
                raise ImportError("LightGBM not available")
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Create a global instance
ml_integration = SimpleMLIntegration()

# Export for compatibility
__all__ = ['ModelFactory', 'ValueBetAnalyzer', 'SimpleMLIntegration', 'ml_integration'] 