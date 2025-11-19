# scripts/build_training_dataset.py
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml  # Keep yaml for model_params loading/dumping

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary components
try:
    from data.storage.database import DBManager
    from models.feature_eng.feature_generator import FeatureGenerator
    from utils.config import Config  # Import centralized config
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Load main config using the centralized utility
try:
    Config.load()
except Exception as e:
    logger.warning(f"Could not load configuration via utils.config: {e}. Using defaults.")
    # Define minimal defaults if config fails to load entirely
    DEFAULT_TRAINING_DAYS = 2 * 365
    DEFAULT_TARGET_PATH = "data/processed/training_features_v3.parquet"
    DEFAULT_PARAMS_PATH = "config/model_params.yaml"
    DEFAULT_TARGET_VAR = "match_result"
else:
    DEFAULT_TRAINING_DAYS = 2 * 365
    DEFAULT_TARGET_PATH = "data/processed/training_features_v3.parquet"
    DEFAULT_PARAMS_PATH = "config/model_params.yaml"
    DEFAULT_TARGET_VAR = "match_result"


# Get settings from Config, providing defaults
training_days_ago = Config.get("data_build.training_days_ago", DEFAULT_TRAINING_DAYS)
TRAINING_DATA_START_DATE = datetime.now() - timedelta(days=training_days_ago)
TARGET_DATASET_PATH = Config.get("paths.training_dataset", DEFAULT_TARGET_PATH)
MODEL_PARAMS_PATH = Config.get("paths.model_params", DEFAULT_PARAMS_PATH)
TARGET_VARIABLE_NAME = Config.get("models.target_variable", DEFAULT_TARGET_VAR)


def define_target_variable(df: pd.DataFrame, target_var_name: str) -> pd.DataFrame:
    """Adds the target variable based on match scores."""
    conditions = [
        (df["home_score"] > df["away_score"]),  # Home win
        (df["home_score"] == df["away_score"]), # Draw
        (df["home_score"] < df["away_score"])   # Away win
    ]
    choices = [2, 1, 0]
    df[target_var_name] = np.select(conditions, choices, default=-1) # Default -1 for errors/unexpected
    # Drop rows where target couldn't be determined (e.g., missing scores)
    df = df[df[target_var_name] != -1].copy()
    logger.info(f"Defined target variable '{target_var_name}'. Kept {len(df)} matches.")
    return df

def build_dataset():
    """
    Fetches historical matches, generates features using the enhanced generator,
    calculates normalization parameters, and saves the training dataset.
    """
    logger.info("Starting training dataset build process...")
    storage = None
    try:
        # Import the decorator now to avoid circular imports
        from utils.decorators import performance_tracker

        # 1. Initialize Storage and Feature Generator
        storage = DBManager()
        logger.info("Database connection established successfully")
        
        feature_gen = FeatureGenerator(storage)
        logger.info("Feature generator initialized successfully")

        # 2. Fetch Historical Matches
        logger.info(f"Fetching historical finished matches from {TRAINING_DATA_START_DATE}...")
        historical_matches_df = storage.get_matches_df(
            date_from=TRAINING_DATA_START_DATE,
            status="FINISHED"
        )

        if historical_matches_df.empty:
            logger.error("No historical finished matches found for the specified period. Cannot build dataset.")
            return

        logger.info(f"Fetched {len(historical_matches_df)} historical matches.")

        # 3. Generate Features
        # This now uses the enhanced generator which leverages scraped_data
        features_df = feature_gen.generate_features_for_dataset(historical_matches_df)

        # --- Robust feature alignment (enforce canonical feature list) ---
        feature_list = Config.get("models.normalization.feature_list", [])
        if feature_list:
            missing = [feat for feat in feature_list if feat not in features_df.columns]
            extra = [feat for feat in features_df.columns if feat not in feature_list and feat != 'match_id']
            if missing:
                logger.warning(f"Missing features in training set: {missing}. Imputing with 0.0.")
                for feat in missing:
                    features_df[feat] = 0.0
            if extra:
                logger.warning(f"Extra features in training set: {extra}. Dropping them.")
                features_df = features_df.drop(columns=extra)
            # Ensure correct order
            features_df = features_df[[col for col in ['match_id'] + feature_list if col in features_df.columns]]
        else:
            logger.warning("No canonical feature_list found in config. Skipping strict alignment.")

        if features_df.empty:
            logger.error("Feature generation resulted in an empty DataFrame. Cannot build dataset.")
            return

        # 4. Define Target Variable
        # Merge features with match results (scores) to define target
        # Ensure necessary score columns are present
        required_cols = ["id", "home_score", "away_score"]
        if not all(col in historical_matches_df.columns for col in required_cols):
             logger.error(f"Historical matches DataFrame missing required columns for target definition: {required_cols}")
             # Attempt to fetch scores separately if needed, or fail
             # For now, assume they are fetched by get_matches_df
             return

        # Merge based on match_id (ensure consistent type, e.g., string)
        features_df['match_id'] = features_df['match_id'].astype(str)
        historical_matches_df['id'] = historical_matches_df['id'].astype(str)
        final_df = pd.merge(
            features_df,
            historical_matches_df[['id', 'home_score', 'away_score']],
            left_on='match_id',
            right_on="id",
            how="inner" # Keep only matches where features could be generated
        )
        # Pass the target variable name from config
        final_df = define_target_variable(final_df, TARGET_VARIABLE_NAME)

        if final_df.empty:
             logger.error("DataFrame is empty after merging features and defining target variable.")
             return

        # Select only feature columns + target variable for the final dataset
        feature_columns = [col for col in final_df.columns if col not in ["match_id", "id", "home_score", "away_score", TARGET_VARIABLE_NAME]]
        dataset_df = final_df[feature_columns + [TARGET_VARIABLE_NAME]]

        # 5. Calculate Normalization Parameters (Means/Stds)
        logger.info("Calculating feature means and standard deviations...")
        # Ensure only numeric columns are used for calculation
        numeric_feature_cols = dataset_df[feature_columns].select_dtypes(include=np.number).columns.tolist()
        if not numeric_feature_cols:
             logger.error("No numeric feature columns found to calculate normalization parameters.")
             return

        feature_means = dataset_df[numeric_feature_cols].mean().to_dict()
        feature_stds = dataset_df[numeric_feature_cols].std().to_dict()
        # Handle potential zero standard deviation (replace with 1 to avoid division by zero)
        feature_stds = {k: (v if v > 1e-6 else 1.0) for k, v in feature_stds.items()}

        logger.info(f"Calculated means for {len(feature_means)} features.")
        logger.info(f"Calculated stds for {len(feature_stds)} features.")

        # 6. Save Training Dataset
        logger.info(f"Saving processed training dataset to: {TARGET_DATASET_PATH}")
        os.makedirs(os.path.dirname(TARGET_DATASET_PATH), exist_ok=True)
        dataset_df.to_parquet(TARGET_DATASET_PATH, index=False)
        logger.info("Training dataset saved successfully.")

        # 7. Update model_params.yaml (using path from config)
        model_params_path = MODEL_PARAMS_PATH # Use the config-loaded path
        logger.info(f"Updating normalization parameters in: {model_params_path}")
        try:
            # Ensure directory exists before reading/writing
            os.makedirs(os.path.dirname(model_params_path), exist_ok=True)

            model_params = {} # Start fresh or load existing
            if os.path.exists(model_params_path):
                with open(model_params_path, 'r') as f:
                    model_params = yaml.safe_load(f)
                    if model_params is None: model_params = {} # Handle empty file case

            if 'normalization' not in model_params: model_params['normalization'] = {}

            # Update with new values (convert numpy types if necessary)
            model_params['normalization']['feature_means'] = {k: float(v) for k, v in feature_means.items()}
            model_params['normalization']['feature_stds'] = {k: float(v) for k, v in feature_stds.items()}
            model_params['normalization']['feature_list'] = numeric_feature_cols # Store the list of features used

            with open(model_params_path, 'w') as f:
                yaml.dump(model_params, f, default_flow_style=None, sort_keys=False)
            logger.info(f"Successfully updated {model_params_path}.")

        except FileNotFoundError: # Should be less likely now with makedirs
            logger.error(f"Model params file path seems invalid: {model_params_path}. Cannot update.")
        except Exception as e:
            logger.exception(f"Error updating {MODEL_PARAMS_PATH}: {e}")

    except Exception as e:
        logger.exception(f"An error occurred during the dataset build process: {e}")
    finally:
        if storage:
            storage.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    build_dataset()
