#!/usr/bin/env python
"""
Enhanced Model Validation Script for Football Betting Platform
Provides comprehensive model performance analysis and overfitting detection.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from models.predictive.analytics_model import build_stacking_model, load_data
from utils.config import Config

logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive model validation and performance analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the validator with configuration."""
        self.config = Config(config_path)
        self.results = {}
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare data for validation."""
        logger.info("Loading data for validation...")
        
        # Load data using the same method as training
        df = load_data(
            raw_path="data/raw",
            cache_path="data/cache", 
            config=self.config.get_dict()
        )
        
        if df.empty:
            raise ValueError("No data available for validation")
            
        # Prepare features and target
        target_col = self.config.get('training.target_column', 'target')
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
            
        # Separate features and target
        X = df.drop(columns=[target_col, 'match_id'], errors='ignore')
        y = df[target_col]
        
        # Ensure all features are numeric
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                logger.warning(f"Converting non-numeric column {col} to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Handle missing values
        X = X.fillna(X.median())
        
        logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def validate_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive model validation."""
        logger.info("Starting comprehensive model validation...")
        
        results = {}
        
        # Build model
        model = build_stacking_model(self.config.get('models', {}))
        
        # 1. Cross-validation analysis
        logger.info("Performing cross-validation analysis...")
        cv_results = self._cross_validation_analysis(model, X, y)
        results['cross_validation'] = cv_results
        
        # 2. Learning curve analysis
        logger.info("Analyzing learning curves...")
        learning_results = self._learning_curve_analysis(model, X, y)
        results['learning_curves'] = learning_results
        
        # 3. Overfitting detection
        logger.info("Detecting overfitting...")
        overfitting_results = self._overfitting_detection(model, X, y)
        results['overfitting'] = overfitting_results
        
        # 4. Feature importance analysis
        logger.info("Analyzing feature importance...")
        feature_results = self._feature_importance_analysis(model, X, y)
        results['feature_importance'] = feature_results
        
        self.results = results
        return results
    
    def _cross_validation_analysis(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform detailed cross-validation analysis."""
        cv_folds = min(5, len(y) // 10)  # Ensure sufficient samples per fold
        if cv_folds < 2:
            cv_folds = 2
            
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Multiple scoring metrics
        scoring_metrics = ['accuracy', 'neg_log_loss', 'precision_macro', 'recall_macro', 'f1_macro']
        
        cv_results = {}
        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            
        return cv_results
    
    def _learning_curve_analysis(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze learning curves to detect overfitting."""
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=3, 
            scoring='accuracy', random_state=42
        )
        
        return {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist()
        }
    
    def _overfitting_detection(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Detect overfitting by comparing training and validation performance."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on both sets
        train_pred = model.predict(X_train)
        train_proba = model.predict_proba(X_train)
        val_pred = model.predict(X_val)
        val_proba = model.predict_proba(X_val)
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        train_logloss = log_loss(y_train, train_proba)
        val_logloss = log_loss(y_val, val_proba)
        
        # Overfitting indicators
        acc_gap = train_acc - val_acc
        logloss_gap = val_logloss - train_logloss
        
        overfitting_detected = acc_gap > 0.1 or logloss_gap > 0.2
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_logloss': train_logloss,
            'val_logloss': val_logloss,
            'accuracy_gap': acc_gap,
            'logloss_gap': logloss_gap,
            'overfitting_detected': overfitting_detected,
            'classification_report': classification_report(y_val, val_pred, output_dict=True)
        }
    
    def _feature_importance_analysis(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze feature importance."""
        model.fit(X, y)
        
        # Try to get feature importance
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                # For stacking models, try to get from final estimator
                if hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'coef_'):
                    importances = np.abs(model.final_estimator_.coef_[0])
                else:
                    importances = np.ones(len(X.columns))  # Default equal importance
                    
            feature_importance = dict(zip(X.columns, importances))
            
            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': feature_importance,
                'top_features': sorted_features[:10],
                'bottom_features': sorted_features[-5:]
            }
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {'error': str(e)}
    
    def generate_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.results:
            return "No validation results available. Run validate_model_performance first."
            
        report = []
        report.append("=" * 60)
        report.append("FOOTBALL BETTING MODEL VALIDATION REPORT")
        report.append("=" * 60)
        
        # Cross-validation results
        if 'cross_validation' in self.results:
            cv = self.results['cross_validation']
            report.append("\n📊 CROSS-VALIDATION RESULTS:")
            report.append("-" * 30)
            for metric, values in cv.items():
                metric_name = metric.replace('neg_', '').replace('_', ' ').title()
                mean_val = values['mean']
                std_val = values['std']
                if 'neg_' in metric:
                    mean_val = -mean_val  # Convert back from negative
                report.append(f"{metric_name:15}: {mean_val:.4f} (±{std_val:.4f})")
        
        # Overfitting analysis
        if 'overfitting' in self.results:
            ov = self.results['overfitting']
            report.append("\n🔍 OVERFITTING ANALYSIS:")
            report.append("-" * 25)
            report.append(f"Training Accuracy  : {ov['train_accuracy']:.4f}")
            report.append(f"Validation Accuracy: {ov['val_accuracy']:.4f}")
            report.append(f"Accuracy Gap       : {ov['accuracy_gap']:.4f}")
            report.append(f"Training Log Loss  : {ov['train_logloss']:.4f}")
            report.append(f"Validation Log Loss: {ov['val_logloss']:.4f}")
            report.append(f"Log Loss Gap       : {ov['logloss_gap']:.4f}")
            
            if ov['overfitting_detected']:
                report.append("\n⚠️  OVERFITTING DETECTED!")
                report.append("Recommendations:")
                report.append("- Reduce model complexity")
                report.append("- Add regularization")
                report.append("- Collect more training data")
                report.append("- Use feature selection")
            else:
                report.append("\n✅ No significant overfitting detected")
        
        # Feature importance
        if 'feature_importance' in self.results and 'top_features' in self.results['feature_importance']:
            fi = self.results['feature_importance']
            report.append("\n🎯 TOP FEATURES:")
            report.append("-" * 15)
            for feature, importance in fi['top_features'][:5]:
                report.append(f"{feature:20}: {importance:.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

def main():
    """Main validation function."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        validator = ModelValidator()
        X, y = validator.load_and_prepare_data()
        results = validator.validate_model_performance(X, y)
        
        # Generate and print report
        report = validator.generate_report()
        print(report)
        
        # Save results
        output_path = Path("validation_results.json")
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
