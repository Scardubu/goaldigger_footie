#!/usr/bin/env python3
"""
Feature Importance Analysis System
SHAP and permutation importance for feature selection
"""
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available, using permutation importance only")

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance using SHAP and permutation importance
    Identify top predictive features and eliminate noise
    """
    
    def __init__(self):
        self.importance_results = {}
        self.feature_rankings = {}
        self.shap_values = None
        logger.info("📊 Feature Importance Analyzer initialized")
    
    def analyze_all_importances(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive feature importance analysis
        
        Parameters:
            model: Trained model
            X_train: Training features
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
        
        Returns:
            Dict with all importance metrics
        """
        logger.info("🔍 Starting comprehensive feature importance analysis")
        start_time = time.time()
        
        if feature_names is None:
            feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_features': len(feature_names),
            'feature_names': feature_names
        }
        
        # 1. Model-specific feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            results['model_importance'] = self._get_model_importance(model, feature_names)
            logger.info("✅ Extracted model-specific importance")
        
        # 2. Permutation importance
        results['permutation_importance'] = self._calculate_permutation_importance(
            model, X_test, y_test, feature_names
        )
        logger.info("✅ Calculated permutation importance")
        
        # 3. SHAP values (if available)
        if SHAP_AVAILABLE:
            results['shap_importance'] = self._calculate_shap_importance(
                model, X_train, X_test, feature_names
            )
            logger.info("✅ Calculated SHAP importance")
        else:
            logger.warning("⚠️ SHAP not available, skipping SHAP analysis")
        
        # 4. Aggregate importance scores
        results['aggregated_importance'] = self._aggregate_importance_scores(results)
        logger.info("✅ Aggregated importance scores")
        
        # 5. Feature selection recommendations
        results['feature_selection'] = self._generate_feature_selection_recommendations(
            results['aggregated_importance']
        )
        
        analysis_time = time.time() - start_time
        results['analysis_time'] = analysis_time
        
        self.importance_results = results
        logger.info(f"✅ Feature importance analysis complete in {analysis_time:.2f}s")
        
        return results
    
    def _get_model_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            importances = model.feature_importances_
            importance_dict = {
                name: float(importance)
                for name, importance in zip(feature_names, importances)
            }
            return importance_dict
        except Exception as e:
            logger.error(f"Failed to extract model importance: {e}")
            return {}
    
    def _calculate_permutation_importance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
        n_repeats: int = 10
    ) -> Dict[str, Any]:
        """Calculate permutation importance"""
        try:
            # Calculate permutation importance
            # Use n_jobs=1 to avoid serialization issues with complex models
            perm_importance = permutation_importance(
                model, X_test, y_test,
                n_repeats=n_repeats,
                random_state=42,
                n_jobs=1  # Changed from -1 to avoid pickling errors
            )
            
            # Get mean and std for each feature
            importance_dict = {}
            for i, name in enumerate(feature_names):
                importance_dict[name] = {
                    'mean': float(perm_importance.importances_mean[i]),
                    'std': float(perm_importance.importances_std[i])
                }
            
            return {
                'importances': importance_dict,
                'n_repeats': n_repeats
            }
        except Exception as e:
            logger.error(f"Failed to calculate permutation importance: {e}")
            return {'importances': {}, 'n_repeats': 0}
    
    def _calculate_shap_importance(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str],
        max_samples: int = 1000
    ) -> Dict[str, Any]:
        """Calculate SHAP importance"""
        try:
            # Limit samples for performance
            if len(X_train) > max_samples:
                X_train_sample = X_train.sample(n=max_samples, random_state=42)
            else:
                X_train_sample = X_train
            
            if len(X_test) > max_samples:
                X_test_sample = X_test.sample(n=max_samples, random_state=42)
            else:
                X_test_sample = X_test
            
            # Create SHAP explainer
            try:
                # Try TreeExplainer first (faster for tree-based models)
                explainer = shap.TreeExplainer(model)
            except:
                # Fall back to KernelExplainer
                explainer = shap.KernelExplainer(
                    model.predict_proba,
                    X_train_sample
                )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test_sample)
            
            # Store for later visualization
            self.shap_values = shap_values
            
            # Calculate mean absolute SHAP values per feature
            if isinstance(shap_values, list):
                # Multi-class: average across classes
                # Handle each class separately, then average
                class_importances = []
                for sv in shap_values:
                    if len(sv.shape) == 2:  # 2D array (samples x features)
                        class_importances.append(np.abs(sv).mean(axis=0))
                    else:  # 1D or other shape
                        class_importances.append(np.abs(sv))
                mean_abs_shap = np.mean(class_importances, axis=0)
            else:
                # Binary or single output
                if len(shap_values.shape) == 2:  # 2D array (samples x features)
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                else:  # 1D array
                    mean_abs_shap = np.abs(shap_values)
            
            # Ensure mean_abs_shap is 1D array matching feature count
            if len(mean_abs_shap.shape) > 1:
                mean_abs_shap = mean_abs_shap.flatten()
            
            # Verify length matches feature count
            if len(mean_abs_shap) != len(feature_names):
                logger.warning(f"SHAP values length ({len(mean_abs_shap)}) != features ({len(feature_names)})")
                # Pad or trim to match
                if len(mean_abs_shap) < len(feature_names):
                    mean_abs_shap = np.pad(mean_abs_shap, (0, len(feature_names) - len(mean_abs_shap)))
                else:
                    mean_abs_shap = mean_abs_shap[:len(feature_names)]
            
            importance_dict = {
                name: float(importance)
                for name, importance in zip(feature_names, mean_abs_shap)
            }
            
            return {
                'importances': importance_dict,
                'explainer_type': type(explainer).__name__,
                'n_samples': len(X_test_sample)
            }
        except Exception as e:
            logger.error(f"Failed to calculate SHAP importance: {e}")
            return {'importances': {}, 'explainer_type': None, 'n_samples': 0}
    
    def _aggregate_importance_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Aggregate importance scores from different methods"""
        aggregated = {}
        
        # Collect all importance scores
        importance_sources = []
        
        # Model-specific importance
        if 'model_importance' in results and results['model_importance']:
            importance_sources.append(self._normalize_dict(results['model_importance']))
        
        # Permutation importance
        if 'permutation_importance' in results and results['permutation_importance']['importances']:
            perm_dict = {
                k: v['mean']
                for k, v in results['permutation_importance']['importances'].items()
            }
            importance_sources.append(self._normalize_dict(perm_dict))
        
        # SHAP importance
        if 'shap_importance' in results and results['shap_importance'].get('importances'):
            importance_sources.append(self._normalize_dict(results['shap_importance']['importances']))
        
        # Aggregate by averaging
        if importance_sources:
            feature_names = importance_sources[0].keys()
            for feature_name in feature_names:
                scores = [source.get(feature_name, 0.0) for source in importance_sources]
                aggregated[feature_name] = float(np.mean(scores))
        
        return aggregated
    
    def _normalize_dict(self, importance_dict: Dict[str, float]) -> Dict[str, float]:
        """Normalize importance scores to sum to 1"""
        total = sum(importance_dict.values())
        if total == 0:
            return importance_dict
        return {k: v / total for k, v in importance_dict.items()}
    
    def _generate_feature_selection_recommendations(
        self,
        aggregated_importance: Dict[str, float],
        top_k: int = 20,
        min_importance_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """Generate feature selection recommendations"""
        
        # Sort features by importance
        sorted_features = sorted(
            aggregated_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top features
        top_features = [f for f, _ in sorted_features[:top_k]]
        
        # Select features above threshold
        important_features = [f for f, importance in sorted_features if importance >= min_importance_threshold]
        
        # Identify noise features (very low importance)
        noise_features = [f for f, importance in sorted_features if importance < min_importance_threshold / 2]
        
        return {
            'top_features': top_features,
            'top_k': top_k,
            'important_features': important_features,
            'n_important': len(important_features),
            'noise_features': noise_features,
            'n_noise': len(noise_features),
            'feature_reduction_potential': len(noise_features) / len(sorted_features),
            'recommended_features': important_features
        }
    
    def get_top_features(self, k: int = 20) -> List[str]:
        """Get top k most important features"""
        if not self.importance_results or 'feature_selection' not in self.importance_results:
            logger.warning("No importance results available")
            return []
        
        return self.importance_results['feature_selection']['top_features'][:k]
    
    def get_feature_rankings(self) -> Dict[str, int]:
        """Get feature rankings (1 = most important)"""
        if not self.importance_results or 'aggregated_importance' not in self.importance_results:
            logger.warning("No importance results available")
            return {}
        
        sorted_features = sorted(
            self.importance_results['aggregated_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        rankings = {feature: rank + 1 for rank, (feature, _) in enumerate(sorted_features)}
        return rankings
    
    def plot_feature_importance(self, top_k: int = 20, save_path: Optional[str] = None):
        """Plot feature importance (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available, skipping plot")
            return
        
        if not self.importance_results or 'aggregated_importance' not in self.importance_results:
            logger.warning("No importance results available")
            return
        
        # Get top features
        aggregated = self.importance_results['aggregated_importance']
        sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        features = [f for f, _ in sorted_features]
        importances = [i for _, i in sorted_features]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importances')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Saved feature importance plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_importance_results(self, filepath: str):
        """Save importance analysis results to file"""
        try:
            # Remove non-serializable SHAP values
            results_copy = self.importance_results.copy()
            if 'shap_values' in results_copy:
                del results_copy['shap_values']
            
            with open(filepath, 'w') as f:
                json.dump(results_copy, f, indent=2)
            
            logger.info(f"✅ Saved importance results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save importance results: {e}")
    
    def load_importance_results(self, filepath: str) -> bool:
        """Load previously saved importance results"""
        try:
            with open(filepath, 'r') as f:
                self.importance_results = json.load(f)
            
            logger.info(f"✅ Loaded importance results from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load importance results: {e}")
            return False
    
    def get_importance_summary(self) -> Dict[str, Any]:
        """Get summary of importance analysis"""
        if not self.importance_results:
            return {'status': 'no_analysis_performed'}
        
        return {
            'n_features': self.importance_results.get('n_features', 0),
            'top_10_features': self.get_top_features(10),
            'feature_reduction_potential': self.importance_results.get('feature_selection', {}).get('feature_reduction_potential', 0),
            'analysis_time': self.importance_results.get('analysis_time', 0),
            'methods_used': [
                key for key in ['model_importance', 'permutation_importance', 'shap_importance']
                if key in self.importance_results
            ]
        }


# Global instance
feature_importance_analyzer = FeatureImportanceAnalyzer()
feature_importance_analyzer = FeatureImportanceAnalyzer()
