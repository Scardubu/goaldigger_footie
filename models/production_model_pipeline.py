#!/usr/bin/env python3
"""
Production Model Training and Evaluation Pipeline
Integrates feature engineering, hyperparameter optimization, and model training
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


class ProductionModelPipeline:
    """
    End-to-end model training pipeline for production
    Combines feature engineering, optimization, training, and evaluation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.baseline_metrics = {}
        self.improved_metrics = {}
        self.training_history = []
        logger.info("🚀 Production Model Pipeline initialized")
    
    def run_complete_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        optimize_hyperparams: bool = True,
        analyze_features: bool = True,
        n_trials: int = 30,
        use_cross_validation: bool = True,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Steps:
        1. Capture baseline metrics
        2. Analyze feature importance
        3. Optimize hyperparameters (with optional CV)
        4. Train model with optimized parameters
        5. Cross-validate for robust estimates (Phase 5)
        6. Evaluate and compare with baseline
        7. Generate reports
        
        Returns:
            Complete pipeline results and metrics
        """
        logger.info("=" * 80)
        logger.info("🏁 Starting Production Model Training Pipeline")
        logger.info("=" * 80)
        
        # Import accuracy_score for ensemble comparison
        from sklearn.metrics import accuracy_score
        
        pipeline_start = time.time()
        results = {
            'pipeline_start': datetime.now().isoformat(),
            'config': self.config,
            'use_cross_validation': use_cross_validation,
            'cv_folds': cv_folds
        }
        
        try:
            # Step 1: Capture baseline metrics
            logger.info("\n📊 Step 1/6: Capturing baseline metrics...")
            baseline = self._capture_baseline_metrics(X_train, y_train, X_test, y_test)
            results['baseline_metrics'] = baseline
            self.baseline_metrics = baseline
            logger.info(f"✅ Baseline accuracy: {baseline['accuracy']:.4f}")
            
            # Step 2: Feature importance analysis
            if analyze_features:
                logger.info("\n🔍 Step 2/6: Analyzing feature importance...")
                feature_analysis = self._analyze_features(
                    baseline['model'], X_train, X_test, y_test
                )
                results['feature_analysis'] = feature_analysis
                logger.info(f"✅ Top 10 features: {feature_analysis['summary']['top_10_features'][:5]}...")
            else:
                logger.info("\n⏭️  Step 2/6: Skipping feature analysis")
                results['feature_analysis'] = None
            
            # Step 3: Optimize hyperparameters
            if optimize_hyperparams:
                logger.info("\n⚙️  Step 3/6: Optimizing hyperparameters...")
                optimization = self._optimize_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials
                )
                results['hyperparameter_optimization'] = optimization
                if 'error' not in optimization:
                    logger.info(f"✅ Best score: {optimization.get('best_score', 'N/A')}")
                else:
                    logger.warning(f"⚠️ Optimization had errors: {optimization.get('error')}")
            else:
                logger.info("\n⏭️  Step 3/6: Skipping hyperparameter optimization")
                results['hyperparameter_optimization'] = None
            
            # Step 4: Train improved model
            logger.info("\n🎯 Step 4/6: Training improved model...")
            improved_model = self._train_improved_model(
                X_train, y_train, X_val, y_val,
                optimization if optimize_hyperparams else None
            )
            results['improved_model'] = improved_model
            logger.info(f"✅ Model trained successfully")
            
            # Step 5: Cross-Validation (Phase 5) - NEW
            if use_cross_validation:
                logger.info("\n🔄 Step 5/8: Running stratified cross-validation (Phase 5)...")
                cv_results = self._run_cross_validation(
                    X_train, y_train, X_test, y_test,
                    improved_model.get('params', {}),
                    cv_folds
                )
                results['cross_validation'] = cv_results
                logger.info(f"✅ CV Mean Accuracy: {cv_results['aggregate_metrics']['mean_accuracy']:.4f} ± {cv_results['aggregate_metrics']['std_accuracy']:.4f}")
            else:
                logger.info("\n⏭️  Step 5/8: Skipping cross-validation")
                results['cross_validation'] = None
            
            # Step 6: Train ensemble model (Phase 4)
            logger.info("\n🗳️  Step 6/8: Training ensemble model...")
            ensemble_model = self._train_ensemble_model(
                X_train, y_train, X_val, y_val, improved_model
            )
            results['ensemble_model'] = ensemble_model
            
            # Evaluate single model on validation set for comparison
            single_model_val_preds = improved_model['model'].predict(X_val)
            single_model_val_accuracy = accuracy_score(y_val, single_model_val_preds)
            logger.info(f"   Single XGBoost validation accuracy: {single_model_val_accuracy:.4f}")
            logger.info(f"   Ensemble validation accuracy: {ensemble_model.get('accuracy', 0):.4f}")
            
            # Determine best model (single vs ensemble vs CV ensemble)
            best_model_results = improved_model
            
            # Consider CV ensemble if available
            if use_cross_validation and 'cv_ensemble' in cv_results:
                cv_ensemble_accuracy = cv_results['cv_ensemble'].get('accuracy', 0)
                logger.info(f"   CV Ensemble test accuracy: {cv_ensemble_accuracy:.4f}")
                
                if cv_ensemble_accuracy > single_model_val_accuracy and cv_ensemble_accuracy > ensemble_model.get('accuracy', 0):
                    logger.info(f"✅ CV Ensemble performs best: {cv_ensemble_accuracy:.4f}")
                    best_model_results = {
                        'model_type': 'cv_ensemble',
                        'accuracy': cv_ensemble_accuracy,
                        'cv_system': cv_results.get('cv_system')
                    }
            
            # Compare voting ensemble vs single model
            if 'error' not in ensemble_model and ensemble_model.get('accuracy', 0) > single_model_val_accuracy:
                if best_model_results.get('model_type') != 'cv_ensemble' or ensemble_model['accuracy'] > best_model_results.get('accuracy', 0):
                    logger.info(f"✅ Voting ensemble outperforms: {ensemble_model['accuracy']:.4f}")
                    best_model_results = ensemble_model
                    best_model_results['model'] = ensemble_model['model']
            
            # Default to single model if nothing else is better
            if 'model' not in best_model_results:
                logger.info(f"✅ Single XGBoost performs best, using it")
                best_model_results = {'model': improved_model['model'], 'model_type': 'xgboost'}
            
            results['best_model_type'] = best_model_results.get('model_type', 'xgboost')
            
            # Step 7: Evaluate best model
            logger.info("\n📈 Step 7/8: Evaluating best model...")
            
            # Handle CV ensemble evaluation separately
            if best_model_results.get('model_type') == 'cv_ensemble':
                evaluation = {
                    'accuracy': best_model_results['accuracy'],
                    'log_loss': cv_results['cv_ensemble'].get('log_loss', 0),
                    'calibration_error': cv_results['cv_ensemble'].get('calibration_error', 0),
                    'model_type': 'cv_ensemble'
                }
            else:
                evaluation = self._evaluate_model(
                    best_model_results['model'], X_test, y_test, X_train, y_train
                )
            
            results['improved_metrics'] = evaluation
            self.improved_metrics = evaluation
            logger.info(f"✅ Best model accuracy: {evaluation['accuracy']:.4f}")
            
            # Step 8: Generate comparison report
            logger.info("\n📋 Step 8/8: Generating comparison report...")
            comparison = self._generate_comparison_report(baseline, evaluation)
            results['comparison'] = comparison
            logger.info(f"✅ Accuracy improvement: {comparison['improvements']['accuracy_improvement']:.4f}")
            
            # Calculate total pipeline time
            pipeline_time = time.time() - pipeline_start
            results['pipeline_time'] = pipeline_time
            results['pipeline_end'] = datetime.now().isoformat()
            
            # Store in history
            self.training_history.append(results)
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ Production Model Training Pipeline Complete!")
            logger.info(f"Total time: {pipeline_time:.2f}s")
            logger.info("=" * 80)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'pipeline_time': time.time() - pipeline_start
            }
    
    def _capture_baseline_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Capture baseline model performance"""
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score, classification_report, log_loss
            from sklearn.utils.class_weight import compute_sample_weight

            # Calculate sample weights for class imbalance
            # Training data: Home (51%), Draw (19%), Away (30%)
            sample_weights = compute_sample_weight('balanced', y_train)
            logger.info(f"📊 Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            logger.info(f"⚖️  Applied balanced sample weights to address 51% home win bias")

            # Train baseline model with default params from config
            baseline_params = {
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'n_estimators': 200,
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = xgb.XGBClassifier(**baseline_params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_pred_proba)
            
            # Calibration error (simplified Brier score)
            from sklearn.metrics import brier_score_loss
            brier_scores = []
            for i in range(3):  # 3 classes
                y_test_binary = (y_test == i).astype(int)
                brier = brier_score_loss(y_test_binary, y_pred_proba[:, i])
                brier_scores.append(brier)
            
            avg_brier = np.mean(brier_scores)
            
            return {
                'model': model,
                'params': baseline_params,
                'accuracy': float(accuracy),
                'log_loss': float(logloss),
                'calibration_error': float(avg_brier),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Baseline capture failed: {e}")
            return {'error': str(e)}
    
    def _analyze_features(
        self,
        model: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Analyze feature importance"""
        try:
            from models.feature_importance_analyzer import FeatureImportanceAnalyzer
            
            analyzer = FeatureImportanceAnalyzer()
            results = analyzer.analyze_all_importances(
                model, X_train, X_test, y_test
            )
            
            summary = analyzer.get_importance_summary()
            
            return {
                'full_results': results,
                'summary': summary
            }
        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            return {'error': str(e)}
    
    def _optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int
    ) -> Dict[str, Any]:
        """Optimize hyperparameters"""
        try:
            from models.hyperparameter_optimizer import HyperparameterOptimizer
            
            optimizer = HyperparameterOptimizer(use_optuna=True)
            results = optimizer.optimize_xgboost_params(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials
            )
            
            return results
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return {'error': str(e)}
    
    def _train_improved_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        optimization_results: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Train model with optimized parameters"""
        try:
            import xgboost as xgb
            from sklearn.utils.class_weight import compute_sample_weight

            # Apply SMOTE to address class imbalance (Phase 5 improvement)
            try:
                from imblearn.over_sampling import SMOTE

                # Check class distribution
                class_counts = y_train.value_counts()
                logger.info(f"📊 Original class distribution: {dict(class_counts)}")
                
                # Apply SMOTE to oversample minority classes (especially draws)
                # Target: bring draws (class 1) from 19% to ~30% of dataset
                smote = SMOTE(
                    sampling_strategy={
                        1: int(class_counts[0] * 0.6),  # Draws: 60% of home wins
                        2: int(class_counts[0] * 0.8)   # Away: 80% of home wins
                    },
                    random_state=42,
                    k_neighbors=5
                )
                
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                balanced_counts = pd.Series(y_train_balanced).value_counts()
                logger.info(f"✅ SMOTE balanced distribution: {dict(balanced_counts)}")
                logger.info(f"   Sample increase: {len(X_train_balanced) - len(X_train)} samples added")
                
                # Use balanced data
                X_train = X_train_balanced
                y_train = y_train_balanced
                
            except ImportError:
                logger.warning("⚠️ imbalanced-learn not available, skipping SMOTE")
            except Exception as e:
                logger.warning(f"⚠️ SMOTE failed: {e}, continuing with original data")

            # Calculate sample weights for remaining imbalance
            sample_weights = compute_sample_weight('balanced', y_train)

            # Get best parameters from optimization or use defaults
            if optimization_results and 'best_params' in optimization_results:
                params = optimization_results['best_params']
                logger.info(f"✅ Using optimized hyperparameters from Optuna")
            else:
                # Use balanced defaults - not over-regularized
                # Previous params were too conservative (max_depth=4, gamma=3.04)
                # causing underfitting (only 50.93% accuracy)
                params = {
                    'learning_rate': 0.03,       # Slightly higher for better learning
                    'max_depth': 5,              # Increase from 4 (allow more complexity)
                    'min_child_weight': 3,       # Balanced regularization
                    'subsample': 0.75,           # Balanced sampling
                    'colsample_bytree': 0.85,    # Balanced feature sampling
                    'gamma': 1.5,                # Reduce from 3.04 (less aggressive pruning)
                    'reg_alpha': 0.3,            # Reduce L1 (from 0.89)
                    'reg_lambda': 0.5,           # Increase L2 (from 0.01)
                    'n_estimators': 250,         # Increase from 150 (more learning rounds)
                    'scale_pos_weight': 1.5,     # Boost minority class (draws)
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'eval_metric': 'mlogloss',
                    'random_state': 42,
                    'n_jobs': -1
                }
                logger.info(f"⚠️ Using balanced default params (anti-underfitting configuration)")
            
            # Train model with early stopping when supported
            model = xgb.XGBClassifier(**params)
            fit_kwargs = {
                'X': X_train,
                'y': y_train,
                'sample_weight': sample_weights,
                'eval_set': [(X_val, y_val)],
                'verbose': False,
            }

            # Prefer callback-based early stopping (works across xgboost versions)
            try:
                from xgboost.callback import EarlyStopping as XgbEarlyStopping

                fit_kwargs['callbacks'] = [
                    XgbEarlyStopping(rounds=30, save_best=True, maximize=False)
                ]
            except Exception as cb_err:  # pragma: no cover - optional dependency
                logger.debug(f"Early stopping callback unavailable: {cb_err}")

            try:
                model.fit(**fit_kwargs)
            except TypeError as fit_err:
                # Some older wrappers do not support callbacks; retry without early stopping
                logger.warning(
                    f"Early stopping not supported in this XGBoost build ({fit_err}); training without it"
                )
                fit_kwargs.pop('callbacks', None)
                model.fit(**fit_kwargs)
            
            logger.info(f"✅ Model trained successfully")
            
            best_iteration = getattr(model, 'best_iteration', None)
            if best_iteration is None and hasattr(model, 'get_booster'):
                booster = model.get_booster()
                if booster is not None and hasattr(booster, 'best_iteration'):
                    best_iteration = booster.best_iteration

            return {
                'model': model,
                'params': params,
                'best_iteration': best_iteration,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {'error': str(e)}
    
    def _run_cross_validation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        params: Dict[str, Any],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """Run stratified K-fold cross-validation (Phase 5)"""
        try:
            from models.cross_validation_system import CrossValidationSystem

            # Initialize CV system
            cv_system = CrossValidationSystem(n_splits=cv_folds, random_state=42)
            
            # Run cross-validation on training data
            cv_results = cv_system.run_stratified_cv(
                X_train, y_train,
                params=params,
                use_smote=True
            )
            
            # Create ensemble from CV fold models
            cv_ensemble_results = cv_system.train_cv_ensemble(X_test, y_test)
            cv_results['cv_ensemble'] = cv_ensemble_results
            cv_results['cv_system'] = cv_system
            
            # Save CV results
            cv_system.save_cv_results('models/trained/cv_results.json')
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def _train_ensemble_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        base_model_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train ensemble model (Phase 4)"""
        try:
            from models.ensemble_model_trainer import EnsembleModelTrainer
            
            ensemble_trainer = EnsembleModelTrainer()
            
            # Extract base XGBoost params if available
            base_params = base_model_results.get('params', None)
            
            # Train voting ensemble
            ensemble_results = ensemble_trainer.train_voting_ensemble(
                X_train, y_train, X_val, y_val,
                base_xgb_params=base_params
            )
            
            return ensemble_results
            
        except Exception as e:
            logger.error(f"Ensemble training failed: {e}")
            return {'error': str(e)}
    
    def _evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            from sklearn.metrics import (
                accuracy_score,
                brier_score_loss,
                classification_report,
                log_loss,
            )

            # Test set predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_pred_proba)
            
            # Calibration error
            brier_scores = []
            for i in range(3):  # 3 classes
                y_test_binary = (y_test == i).astype(int)
                brier = brier_score_loss(y_test_binary, y_pred_proba[:, i])
                brier_scores.append(brier)
            
            avg_brier = np.mean(brier_scores)
            
            # Training set performance
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            return {
                'accuracy': float(accuracy),
                'log_loss': float(logloss),
                'calibration_error': float(avg_brier),
                'train_accuracy': float(train_accuracy),
                'overfitting_gap': float(train_accuracy - accuracy),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {'error': str(e)}
    
    def _generate_comparison_report(
        self,
        baseline: Dict[str, Any],
        improved: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate before/after comparison report"""
        
        comparison = {
            'baseline': {
                'accuracy': baseline.get('accuracy', 0),
                'log_loss': baseline.get('log_loss', 0),
                'calibration_error': baseline.get('calibration_error', 0)
            },
            'improved': {
                'accuracy': improved.get('accuracy', 0),
                'log_loss': improved.get('log_loss', 0),
                'calibration_error': improved.get('calibration_error', 0)
            },
            'improvements': {
                'accuracy_improvement': improved.get('accuracy', 0) - baseline.get('accuracy', 0),
                'log_loss_reduction': baseline.get('log_loss', 0) - improved.get('log_loss', 0),
                'calibration_improvement': baseline.get('calibration_error', 0) - improved.get('calibration_error', 0)
            },
            'relative_improvements': {
                'accuracy_pct': ((improved.get('accuracy', 0) - baseline.get('accuracy', 0)) / baseline.get('accuracy', 1)) * 100 if baseline.get('accuracy') else 0,
                'log_loss_pct': ((baseline.get('log_loss', 1) - improved.get('log_loss', 1)) / baseline.get('log_loss', 1)) * 100 if baseline.get('log_loss') else 0,
                'calibration_pct': ((baseline.get('calibration_error', 1) - improved.get('calibration_error', 1)) / baseline.get('calibration_error', 1)) * 100 if baseline.get('calibration_error') else 0
            }
        }
        
        return comparison
    
    def save_pipeline_results(self, filepath: str):
        """Save complete pipeline results"""
        try:
            results = {
                'baseline_metrics': self.baseline_metrics,
                'improved_metrics': self.improved_metrics,
                'training_history': self.training_history,
                'config': self.config
            }
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"✅ Saved pipeline results to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")
    
    def print_summary(self):
        """Print pipeline summary"""
        if not self.baseline_metrics or not self.improved_metrics:
            logger.warning("No metrics available for summary")
            return
        
        print("\n" + "=" * 80)
        print("MODEL TRAINING PIPELINE SUMMARY")
        print("=" * 80)
        
        print("\nBASELINE METRICS:")
        print(f"  Accuracy:          {self.baseline_metrics.get('accuracy', 0):.4f}")
        print(f"  Log Loss:          {self.baseline_metrics.get('log_loss', 0):.4f}")
        print(f"  Calibration Error: {self.baseline_metrics.get('calibration_error', 0):.4f}")
        
        # Cross-validation results (if available)
        if self.training_history and 'cross_validation' in self.training_history[-1]:
            cv_results = self.training_history[-1]['cross_validation']
            if cv_results and 'aggregate_metrics' in cv_results:
                print("\nCROSS-VALIDATION METRICS (Phase 5):")
                print(f"  Mean Accuracy:     {cv_results['aggregate_metrics']['mean_accuracy']:.4f} ± {cv_results['aggregate_metrics']['std_accuracy']:.4f}")
                print(f"  Accuracy Range:    [{cv_results['aggregate_metrics']['min_accuracy']:.4f}, {cv_results['aggregate_metrics']['max_accuracy']:.4f}]")
                print(f"  Mean Log Loss:     {cv_results['aggregate_metrics']['mean_log_loss']:.4f}")
                print(f"  CV Folds:          {cv_results['n_splits']}")
                if 'cv_ensemble' in cv_results:
                    print(f"\n  CV Ensemble:")
                    print(f"    Accuracy:        {cv_results['cv_ensemble'].get('accuracy', 0):.4f}")
                    print(f"    Log Loss:        {cv_results['cv_ensemble'].get('log_loss', 0):.4f}")
        
        print("\nIMPROVED METRICS (Best Model):")
        print(f"  Accuracy:          {self.improved_metrics.get('accuracy', 0):.4f}")
        print(f"  Log Loss:          {self.improved_metrics.get('log_loss', 0):.4f}")
        print(f"  Calibration Error: {self.improved_metrics.get('calibration_error', 0):.4f}")
        if 'model_type' in self.improved_metrics:
            print(f"  Model Type:        {self.improved_metrics.get('model_type', 'xgboost')}")
        
        acc_improvement = self.improved_metrics.get('accuracy', 0) - self.baseline_metrics.get('accuracy', 0)
        ll_improvement = self.baseline_metrics.get('log_loss', 0) - self.improved_metrics.get('log_loss', 0)
        cal_improvement = self.baseline_metrics.get('calibration_error', 0) - self.improved_metrics.get('calibration_error', 0)
        
        print("\nIMPROVEMENTS:")
        print(f"  Accuracy:          {acc_improvement:+.4f} ({acc_improvement/self.baseline_metrics.get('accuracy', 1)*100:+.2f}%)")
        print(f"  Log Loss:          {ll_improvement:+.4f} ({ll_improvement/self.baseline_metrics.get('log_loss', 1)*100:+.2f}%)")
        print(f"  Calibration Error: {cal_improvement:+.4f} ({cal_improvement/self.baseline_metrics.get('calibration_error', 1)*100:+.2f}%)")
        
        print("\n" + "=" * 80)


# Example usage function
def run_production_training_example():
    """Example of running the production training pipeline"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate example data
    X, y = make_classification(
        n_samples=5000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=3,
        random_state=42
    )
    
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(50)])
    y = pd.Series(y)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Run pipeline
    pipeline = ProductionModelPipeline()
    results = pipeline.run_complete_pipeline(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        optimize_hyperparams=True,
        analyze_features=True,
        n_trials=20
    )
    
    # Print summary
    pipeline.print_summary()
    
    # Save results
    pipeline.save_pipeline_results('pipeline_results.json')
    
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_production_training_example()
