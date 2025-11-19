"""
Predictive models module for GoalDiggers platform.
Provides enhanced ML pipeline and prediction capabilities.
"""

try:
    from .enhanced_ml_pipeline import EnhancedMLPipeline
    __all__ = ['EnhancedMLPipeline']
except ImportError as e:
    print(f"Warning: Could not import EnhancedMLPipeline: {e}")
    __all__ = [] 