#!/usr/bin/env python
import argparse
import logging
import os
import sys
from pathlib import Path

# Import environment setup first to silence warnings and configure environment
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.setup_env import setup_environment

# Configure environment
setup_environment()

import joblib
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    class MockMLFlow:
        @staticmethod
        def set_tracking_uri(*args, **kwargs): pass
        @staticmethod
        def set_experiment(*args, **kwargs): pass
        @staticmethod
        def start_run(*args, **kwargs): return MockMLFlow()
        @staticmethod
        def log_metric(*args, **kwargs): pass
        @staticmethod
        def log_param(*args, **kwargs): pass
        @staticmethod
        def sklearn_log_model(*args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    mlflow = MockMLFlow()
    mlflow.sklearn = MockMLFlow()
import pandas as pd

from scripts.ingest_data import harmonize_valid_matches


def get_logger():
    """Get configured logger."""
    return logging.getLogger(__name__)

def load_model(model_uri):
    """Load model from MLflow or local file."""
    logger = logging.getLogger(__name__)
    try:
        if model_uri.startswith("runs:/") or model_uri.startswith("models:/"):
            logger.info(f"Loading model from MLflow: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
        else:
            logger.info(f"Loading model from local file: {model_uri}")
            model = joblib.load(model_uri)
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)

def load_data(input_file):
    """Load and preprocess input data for prediction."""
    logger = logging.getLogger(__name__)
    try:
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)
            
        # Load the data
        df = pd.read_csv(input_file)
        logger.info(f"Loaded input data: {df.shape} from {input_file}")
        
        # Check if this is a valid_matches.csv format that needs harmonization
        if "feature1" in df.columns and "feature2" in df.columns:
            logger.info("Detected valid_matches.csv format, harmonizing columns...")
            df = harmonize_valid_matches(df)
            
        # Prepare features (similar to training pipeline)
        # Remove non-numeric columns that aren't needed for prediction
        columns_to_drop = []
        for col in df.columns:
            if col == "target":
                continue  # Keep target if it exists (for evaluation)
                
            if df[col].dtype == 'object':
                if col in ['match_id', 'league', 'home_team', 'away_team', 'team', 'home_team_id', 'away_team_id',
                       'match_date', 'manager_change_date']:
                    columns_to_drop.append(col)
                    logger.info(f"Dropping non-numeric column: {col}")
                else:
                    # Try to convert string columns to numeric if possible
                    try:
                        df[col] = pd.to_numeric(df[col])
                        logger.info(f"Converted column {col} to numeric")
                    except:
                        columns_to_drop.append(col)
                        logger.info(f"Dropping non-convertible column: {col}")
        
        features = df.drop(columns=columns_to_drop)
        if "target" in features.columns:
            y_true = features["target"]
            features = features.drop(columns=["target"])
        else:
            y_true = None
            
        logger.info(f"Prepared features: {features.shape}")
        return features, y_true
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Make predictions using a trained model."
    )
    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="URI to the model (can be MLflow URI like 'runs:/...' or path to saved model)"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to CSV file with input data"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save predictions (if not provided, prints to console)"
    )
    args = parser.parse_args()

    logger = get_logger()
    logger.info("Starting prediction process")
    
    # Load model
    model = load_model(args.model_uri)
    
    # Load and prepare data
    X, y_true = load_data(args.input_data)
    
    # Make predictions
    try:
        # Try to get probability predictions
        try:
            y_proba = model.predict_proba(X)
            logger.info("Generated probability predictions")
            
            # Extract win/draw/loss probabilities if appropriate shape
            if y_proba.shape[1] == 3:
                predictions = pd.DataFrame({
                    "home_win_prob": y_proba[:, 0],
                    "draw_prob": y_proba[:, 1],
                    "away_win_prob": y_proba[:, 2],
                    "prediction": model.predict(X)
                })
            else:
                predictions = pd.DataFrame({
                    "probability": y_proba[:, 1],
                    "prediction": model.predict(X)
                })
                
        except (AttributeError, IndexError):
            logger.info("Model doesn't support probability predictions, using class predictions")
            predictions = pd.DataFrame({
                "prediction": model.predict(X)
            })
    
        # Add original indices if available
        if X.index.name:
            predictions.index = X.index
            
        # Evaluate if ground truth is available
        if y_true is not None:
            from sklearn.metrics import accuracy_score, log_loss
            acc = accuracy_score(y_true, predictions["prediction"])
            logger.info(f"Prediction accuracy: {acc:.4f}")
            
            if "probability" in predictions.columns:
                try:
                    ll = log_loss(y_true, predictions["probability"])
                    logger.info(f"Prediction log loss: {ll:.4f}")
                except:
                    pass
            
        # Output results
        if args.output_file:
            predictions.to_csv(args.output_file)
            logger.info(f"Predictions saved to {args.output_file}")
        else:
            print("\nPrediction Results:")
            print(predictions)
            
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        sys.exit(1)
        
    logger.info("Prediction process completed successfully")

if __name__ == "__main__":
    main()
