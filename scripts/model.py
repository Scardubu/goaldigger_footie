#!/usr/bin/env python3
"""
Model training, evaluation, and prediction pipeline.

This module provides a comprehensive pipeline for:
- Training machine learning models with optional hyperparameter optimization
- Evaluating model performance on holdout sets
- Generating batch predictions for upcoming matches
- Managing model persistence and feature engineering workflows

Usage:
    python scripts/model.py train --optimize --optuna-trials 100
    python scripts/model.py evaluate --days-ahead 14
    python scripts/model.py batch_predict --model-filename custom_model.joblib
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Configure pandas settings early
pd.set_option('future.no_silent_downcasting', True)

# Project root path management
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import reproducibility settings first
from utils.reproducibility import ensure_deterministic_environment, set_global_seed

# Set global seed early for reproducibility
set_global_seed(42)

# Core imports
from utils.config import Config
from utils.logging_config import setup_logging


# Dynamic imports with robust error handling
def _safe_import_module(module_path: str, class_name: str, fallback_path: str = None):
    """Safely import a module with fallback to dynamic loading."""
    try:
        # Try standard import first
        if 'feature_generator' in module_path:
            from models.feature_eng.feature_generator import FeatureGenerator
            return FeatureGenerator
        elif 'analytics_model' in module_path:
            from models.predictive.analytics_model import MatchPredictor
            return MatchPredictor
    except (ImportError, ModuleNotFoundError) as e:
        logging.debug(f"Standard import failed for {class_name}: {e}. Attempting dynamic import.")
        
        # Dynamic import fallback
        import importlib.util
        
        if fallback_path:
            full_path = Path(PROJECT_ROOT) / fallback_path
        else:
            # Construct path from module_path
            full_path = Path(PROJECT_ROOT) / module_path.replace('.', '/')
            if not full_path.suffix:
                full_path = full_path.with_suffix('.py')
        
        if not full_path.exists():
            raise ImportError(f"Cannot find module at {full_path}")
        
        spec = importlib.util.spec_from_file_location(class_name.lower(), full_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {full_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[class_name.lower()] = module
        spec.loader.exec_module(module)
        
        return getattr(module, class_name)

# Import required classes
try:
    FeatureGenerator = _safe_import_module(
        'models.feature_eng.feature_generator', 
        'FeatureGenerator',
        'models/feature_eng/feature_generator.py'
    )
    MatchPredictor = _safe_import_module(
        'models.predictive.analytics_model', 
        'MatchPredictor',
        'models/predictive/analytics_model.py'
    )
    from database.db_manager import DatabaseManager
except Exception as e:
    logging.error(f"Critical import error: {e}")
    sys.exit(1)

# Configuration management
class ModelConfig:
    """Centralized configuration management for model operations."""
    
    def __init__(self):
        self._load_config()
    
    def _load_config(self):
        """Load configuration with error handling."""
        try:
            Config.load()
        except Exception as e:
            logging.warning(f"Could not load configuration: {e}. Using defaults.")
        
        # Core model settings
        self.model_type = Config.get("models.default_type", "xgboost")
        self.optuna_trials = Config.get("models.optuna_trials", 50)
        self.model_filename = Config.get("models.save_filename", "predictor_model.joblib")
        self.target_variable = Config.get("models.target_variable", "match_result")
        
        # Data paths
        self.training_dataset = Config.get(
            "paths.training_dataset", 
            "data/processed/training_features_v3.parquet"
        )
        
        # Feature engineering settings
        self.feature_fillna_method = Config.get("features.fillna_method", "median")
        self.feature_inf_replacement = Config.get("features.inf_replacement", 0)

# Global configuration instance
model_config = ModelConfig()

# Logging setup
setup_logging()
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading, cleaning, and preprocessing operations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load_training_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Load and prepare training data from parquet file.
        
        Returns:
            Tuple of (features DataFrame, labels Series) or (None, None) if error
        """
        self.logger.info("Loading pre-generated training data...")
        
        dataset_path = Path(self.config.training_dataset)
        
        if not dataset_path.exists():
            self.logger.error(f"Training dataset not found: {dataset_path}")
            self.logger.error("Run 'scripts/build_training_dataset.py' first")
            return None, None
        
        try:
            training_data = pd.read_parquet(dataset_path)
            self.logger.info(f"Loaded dataset: {training_data.shape}")
            
            if training_data.empty:
                self.logger.error("Dataset is empty")
                return None, None
            
            return self._prepare_features_and_labels(training_data)
            
        except Exception as e:
            self.logger.exception(f"Error loading training data: {e}")
            return None, None
    
    def _prepare_features_and_labels(self, data: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Separate features and labels, apply cleaning."""
        target_col = self.config.target_variable
        
        if target_col not in data.columns:
            self.logger.error(f"Target variable '{target_col}' not found")
            return None, None
        
        labels = data[target_col].copy()
        features = data.drop(columns=[target_col]).copy()
        
        # Clean features
        features = self._clean_features(features)
        
        if features.empty:
            self.logger.error("Features DataFrame empty after cleaning")
            return None, None
        
        # Align labels with cleaned features
        labels = labels.loc[features.index]
        
        self.logger.info(f"Prepared data - Features: {features.shape}, Labels: {labels.shape}")
        return features, labels
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive feature cleaning."""
        initial_rows = len(features)
        
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Infer object types
        features = features.infer_objects(copy=False)
        
        # Handle NaN values based on configuration
        if self.config.feature_fillna_method == "median":
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].median())
        elif self.config.feature_fillna_method == "mean":
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].mean())
        
        # Final fallback for persistent NaNs
        features = features.fillna(0)
        
        # Remove any remaining rows with NaN
        if features.isnull().any().any():
            features = features.dropna()
            dropped_rows = initial_rows - len(features)
            if dropped_rows > 0:
                self.logger.warning(f"Dropped {dropped_rows} rows with persistent NaNs")
        
        return features


class ModelTrainer:
    """Handles model training operations with optimization support."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def train_model(
        self, 
        optimize: bool = False, 
        optuna_trials: int = None,
        test_size: float = 0.2,
        use_shap: bool = False,
        save_model: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Complete model training pipeline.
        
        Args:
            optimize: Whether to run hyperparameter optimization
            optuna_trials: Number of Optuna trials (overrides config)
            test_size: Test set fraction
            use_shap: Whether to compute SHAP values
            save_model: Whether to save the trained model
            
        Returns:
            Training metrics dictionary or None if failed
        """
        self.logger.info(f"Starting training pipeline - Model: {self.config.model_type}")
        
        db = None
        try:
            # Initialize components
            db = DatabaseManager()
            feature_gen = FeatureGenerator(db_storage=db)
            predictor = MatchPredictor(model_type=self.config.model_type)
            
            # Load and prepare data
            data_processor = DataProcessor(self.config)
            features, labels = data_processor.load_training_data()
            
            if features is None or labels is None:
                self.logger.error("Failed to prepare training data")
                return None
            
            # Train model
            training_params = {
                'test_size': test_size,
                'optimize': optimize,
                'optuna_trials': optuna_trials or self.config.optuna_trials,
                'use_shap': use_shap
            }
            
            self.logger.info(f"Training with params: {training_params}")
            metrics = predictor.train(features, labels, **training_params)
            
            self.logger.info(f"Training completed. Metrics: {metrics}")
            
            # Save model if requested
            if save_model:
                try:
                    # Prefer modern JSON booster format if supported
                    saved = predictor.save_model(self.config.model_filename)
                    if saved:
                        self.logger.info(f"Model saved (modern format) to {self.config.model_filename}")
                except TypeError:
                    # Older predictor signature without preferred_format
                    if predictor.save_model(self.config.model_filename):  # type: ignore
                        self.logger.info(f"Model saved to {self.config.model_filename}")
            
            # Log feature importance and SHAP analysis
            if hasattr(predictor, 'study') and predictor.study:
                self._log_optimization_results(predictor.study)
            
            self._log_feature_importance(predictor, features)
            
            if use_shap:
                self._generate_shap_analysis(predictor, features)
            
            return metrics
            
        except Exception as e:
            self.logger.exception("Error in training pipeline")
            return None
        finally:
            if db:
                db.close()
    
    def _log_optimization_results(self, study):
        """Log Optuna optimization results."""
        try:
            import joblib
            study_path = Path("logs/optuna_study.pkl")
            study_path.parent.mkdir(exist_ok=True)
            joblib.dump(study, study_path)
            
            self.logger.info(f"Best trial - Value: {study.best_value:.4f}")
            self.logger.info(f"Best params: {study.best_params}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save optimization results: {e}")
    
    def _log_feature_importance(self, predictor, features):
        """Log and save feature importance analysis."""
        try:
            if hasattr(predictor.model, 'feature_importances_'):
                importances = predictor.model.feature_importances_
                feature_names = predictor.feature_names or features.columns.tolist()
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values(by='importance', ascending=False)
                
                # Log top features
                self.logger.info(f"Top 10 features:\n{importance_df.head(10)}")
                
                # Save to file
                output_path = Path("logs/feature_importance.csv")
                output_path.parent.mkdir(exist_ok=True)
                importance_df.to_csv(output_path, index=False)
                
        except Exception as e:
            self.logger.warning(f"Failed to analyze feature importance: {e}")
    
    def _generate_shap_analysis(self, predictor, features):
        """Generate and save SHAP analysis."""
        try:
            if predictor.explainer is None:
                self.logger.warning("No SHAP explainer available")
                return
            
            import matplotlib.pyplot as plt
            import shap

            # Sample data for SHAP if dataset is large
            sample_size = min(1000, len(features))
            if len(features) > sample_size:
                sample_features = features.sample(n=sample_size, random_state=42)
            else:
                sample_features = features
            
            shap_values = predictor.explainer.shap_values(sample_features)
            
            # Generate summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, sample_features, show=False)
            
            output_path = Path("logs/shap_summary.png")
            output_path.parent.mkdir(exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"SHAP analysis saved to {output_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to generate SHAP analysis: {e}")


class PredictionGenerator:
    """Handles prediction generation for upcoming matches."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def generate_predictions(
        self, 
        days_ahead: int = 7,
        model_filename: str = None,
        match_status: str = "TIMED"
    ) -> int:
        """
        Generate predictions for upcoming matches.
        
        Args:
            days_ahead: Number of days ahead to predict
            model_filename: Custom model filename (optional)
            match_status: Match status to filter for
            
        Returns:
            Number of predictions generated
        """
        model_file = model_filename or self.config.model_filename
        self.logger.info(f"Generating predictions - Days ahead: {days_ahead}, Model: {model_file}")
        
        db = None
        try:
            # Initialize components
            db = DatabaseManager()
            feature_gen = FeatureGenerator(db_storage=db)
            predictor = MatchPredictor(model_type=self.config.model_type)
            
            # Load model
            if not predictor.load_model():
                self.logger.error("Failed to load model")
                return 0
            
            # Get upcoming matches
            matches_df = self._get_upcoming_matches(db, days_ahead, match_status)
            if matches_df.empty:
                self.logger.info("No upcoming matches found")
                return 0
            
            # Generate features
            features_df = feature_gen.generate_features_for_dataset(matches_df)
            features_df = self._align_features(features_df)
            if features_df.empty:
                self.logger.error("Feature generation failed")
                return 0
            
            # Generate predictions
            return self._process_predictions(db, predictor, features_df)
            
        except Exception as e:
            self.logger.exception("Error in prediction generation")
            return 0
        finally:
            if db:
                db.close()
    
    def _get_upcoming_matches(self, db, days_ahead: int, status: str) -> pd.DataFrame:
        """Fetch upcoming matches from database."""
        today = datetime.now().date()
        date_from = datetime.combine(today, datetime.min.time())
        date_to = datetime.combine(today + timedelta(days=days_ahead), datetime.max.time())
        
        matches_df = db.get_matches_df(
            date_from=date_from,
            date_to=date_to,
            status=status
        )
        
        self.logger.info(f"Found {len(matches_df)} upcoming matches")
        return matches_df
    
    def _process_predictions(self, db, predictor, features_df: pd.DataFrame) -> int:
        """Process predictions for each match."""
        # Ensure match_id is available
        if "match_id" not in features_df.columns:
            if features_df.index.name == "match_id":
                features_df = features_df.reset_index()
            else:
                self.logger.error("match_id not found in features")
                return 0
        
        prediction_count = 0
        
        for _, row in features_df.iterrows():
            try:
                match_id = str(row["match_id"])
                
                # Prepare single match features
                single_match_features = self._prepare_single_match_features(
                    row, predictor.feature_names
                )
                
                if single_match_features is None:
                    continue
                
                # Generate prediction
                prediction_probs = predictor.predict_match_outcome(single_match_features)
                
                if prediction_probs:
                    # Save prediction
                    self._save_prediction(db, match_id, prediction_probs, row, predictor.model_type)
                    prediction_count += 1
                    self.logger.debug(f"Prediction saved for match {match_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to process match {row.get('match_id', 'unknown')}: {e}")
        
        self.logger.info(f"Generated {prediction_count} predictions")
        return prediction_count
    
    def _prepare_single_match_features(self, row: pd.Series, feature_names: list) -> Optional[pd.DataFrame]:
        """Prepare features for a single match prediction."""
        try:
            # Remove match_id from features
            match_features = row.drop("match_id")
            single_match_df = pd.DataFrame([match_features])
            
            # Align with trained model features
            if feature_names:
                try:
                    single_match_df = single_match_df[feature_names]
                except KeyError as e:
                    self.logger.error(f"Feature alignment failed - Missing: {e}")
                    return None
            
            # Clean features
            single_match_df = single_match_df.fillna(single_match_df.median())
            single_match_df = single_match_df.replace([np.inf, -np.inf], 0)
            
            return single_match_df
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
    
    def _save_prediction(self, db, match_id: str, prediction_probs: dict, row: pd.Series, model_type: str):
        """Save prediction to database."""
        try:
            model_version = f"{model_type}_loaded"
            features_dict = row.drop("match_id").to_dict()
            
            db.save_prediction(
                match_id=match_id,
                prediction=prediction_probs,
                model_version=model_version,
                features=features_dict
            )
        except Exception as e:
            self.logger.error(f"Failed to save prediction for {match_id}: {e}")


class ModelEvaluator:
    """Handles model evaluation on test datasets."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def evaluate_model(
        self, 
        model_filename: str = None,
        days_ahead: int = 7,
        match_status: str = "FINISHED"
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate model performance on historical matches.
        
        Args:
            model_filename: Model file to evaluate
            days_ahead: Days back to look for evaluation matches
            match_status: Status of matches to evaluate on
            
        Returns:
            Evaluation metrics dictionary
        """
        model_file = model_filename or self.config.model_filename
        self.logger.info(f"Evaluating model: {model_file}")
        
        db = None
        try:
            # Initialize components
            db = DatabaseManager()
            feature_gen = FeatureGenerator(db_storage=db)
            predictor = MatchPredictor(model_type=self.config.model_type)
            
            # Load model
            if not predictor.load_model():
                self.logger.error("Failed to load model for evaluation")
                return None
            
            # Get evaluation matches
            matches_df = self._get_evaluation_matches(db, days_ahead, match_status)
            if matches_df.empty:
                self.logger.info("No matches found for evaluation")
                return None
            
            # Generate features and evaluate
            return self._perform_evaluation(feature_gen, predictor, matches_df)
            
        except Exception as e:
            self.logger.exception("Error in model evaluation")
            return None
        finally:
            if db:
                db.close()
    
    def _get_evaluation_matches(self, db, days_ahead: int, status: str) -> pd.DataFrame:
        """Get matches for evaluation."""
        today = datetime.now().date()
        date_from = datetime.combine(today - timedelta(days=days_ahead), datetime.min.time())
        date_to = datetime.combine(today, datetime.max.time())
        
        return db.get_matches_df(
            date_from=date_from,
            date_to=date_to,
            status=status
        )
    
    def _perform_evaluation(self, feature_gen, predictor, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform the actual evaluation."""
        try:
            # Generate features
            X = feature_gen.generate_features_for_dataset(matches_df)
            y = matches_df[self.config.target_variable]
            
            # Make predictions
            y_pred = predictor.model.predict(X)
            y_pred_proba = predictor.model.predict_proba(X)
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix,
                log_loss,
            )
            
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'log_loss': log_loss(y, y_pred_proba),
                'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
                'classification_report': classification_report(y, y_pred, output_dict=True)
            }
            
            self.logger.info(f"Evaluation Results:")
            self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            self.logger.info(f"Log Loss: {metrics['log_loss']:.4f}")
            self.logger.info(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
            
            return metrics
            
        except Exception as e:
            self.logger.exception("Error performing evaluation")
            return None
    
    def _align_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Align features to canonical feature list, fill missing with 0.0, drop extra, log mismatches."""
        feature_list = Config.get("models.normalization.feature_list", [])
        if feature_list:
            missing = [feat for feat in feature_list if feat not in features_df.columns]
            extra = [feat for feat in features_df.columns if feat not in feature_list and feat != 'match_id']
            if missing:
                self.logger.warning(f"Missing features for prediction/eval: {missing}. Imputing with 0.0.")
                for feat in missing:
                    features_df[feat] = 0.0
            if extra:
                self.logger.warning(f"Extra features for prediction/eval: {extra}. Dropping them.")
                features_df = features_df.drop(columns=extra)
            # Ensure correct order
            features_df = features_df[[col for col in ['match_id'] + feature_list if col in features_df.columns]]
        else:
            self.logger.warning("No canonical feature_list found in config. Skipping strict alignment.")
        return features_df


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Model pipeline for training, evaluation, and prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --optimize --optuna-trials 100 --use-shap
  %(prog)s evaluate --days-ahead 14
  %(prog)s batch_predict --days-ahead 7
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Training subcommand
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--optimize", action="store_true", 
                            help="Run hyperparameter optimization")
    train_parser.add_argument("--optuna-trials", type=int, default=50,
                            help="Number of Optuna trials")
    train_parser.add_argument("--test-size", type=float, default=0.2,
                            help="Test set size fraction")
    train_parser.add_argument("--use-shap", action="store_true",
                            help="Generate SHAP analysis")
    train_parser.add_argument("--save-model", action="store_true",
                            help="Save the trained model")
    
    # Evaluation subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model performance")
    eval_parser.add_argument("--model-filename", 
                           help="Custom model filename")
    eval_parser.add_argument("--days-ahead", type=int, default=7,
                           help="Days back for evaluation matches")
    
    # Batch prediction subcommand
    batch_parser = subparsers.add_parser("batch_predict", help="Generate batch predictions")
    batch_parser.add_argument("--model-filename",
                            help="Custom model filename")
    batch_parser.add_argument("--days-ahead", type=int, default=7,
                            help="Days ahead for predictions")
    batch_parser.add_argument("--output-file",
                            help="Output file for predictions")
    
    return parser


def main():
    """Main entry point for the model pipeline."""
    # Ensure reproducible environment
    ensure_deterministic_environment()
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    # Initialize configuration
    config = ModelConfig()
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    try:
        if args.mode == "train":
            trainer = ModelTrainer(config)
            trainer.train_model(
                optimize=args.optimize,
                optuna_trials=args.optuna_trials,
                test_size=args.test_size,
                use_shap=args.use_shap,
                save_model=args.save_model,
            )
            logger.info("Training completed successfully")
        elif args.mode == "evaluate":
            evaluator = ModelEvaluator(config)
            metrics = evaluator.evaluate_model(
                model_filename=args.model_filename,
                days_ahead=args.days_ahead
            )
            
            if metrics:
                logger.info("Evaluation completed successfully")
            else:
                logger.error("Evaluation failed")
                sys.exit(1)
        
        elif args.mode == "batch_predict":
            predictor = PredictionGenerator(config)
            count = predictor.generate_predictions(
                days_ahead=args.days_ahead,
                model_filename=args.model_filename
            )
            
            if count > 0:
                logger.info(f"Generated {count} predictions successfully")
            else:
                logger.warning("No predictions generated")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error in main execution")
        sys.exit(1)

    logger.info("Model pipeline completed successfully")
if __name__ == "__main__":
    main()