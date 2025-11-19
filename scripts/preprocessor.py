import logging
import os

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv  # Added import
# Removed IterativeImputer as AIDataValidator will handle imputation
# from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder

from data.storage.database import DuckDBStorage
from models.feature_eng.feature_generator import FeatureGenerator
# Import the validator
from scripts.core.ai_validator import \
    AIDataValidator  # Assuming ai_validator is updated

load_dotenv() # Load environment variables for API keys etc.
logger = logging.getLogger(__name__)

# Path to the unified config file (though not directly used in this simplified version)
# CONFIG_PATH = os.path.join(
#     os.path.dirname(__file__), "..", "config", "config.yaml"
# )

class MatchPreprocessor:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False) # sparse=False often easier
        self.validator = AIDataValidator() # Instantiate the validator
        self.db_storage = DuckDBStorage()
        self.feature_gen = FeatureGenerator(self.db_storage)

    def preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing workflow with validation and feature engineering."""
        logger.info(f"Starting preprocessing for DataFrame with shape {raw_df.shape}")
        if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
             logger.error("Preprocessing input must be a non-empty pandas DataFrame.")
             return pd.DataFrame()

        try:
            df = raw_df.copy()

            # Removed call to _fetch_additional_data

            # --- AI Data Validation ---
            logger.info("Applying AI Data Validation...")
            # The validator now handles imputation if needed based on its checks
            validated_df = self.validator.validate(df)
            # Check if validation returned a valid DataFrame
            if validated_df is None or validated_df.empty:
                 logger.error("AI Data Validation failed or returned empty DataFrame. Aborting preprocessing.")
                 # Depending on requirements, might return df or raise error
                 return pd.DataFrame()
            df = validated_df # Use the validated (and potentially imputed) data
            logger.info(f"DataFrame shape after validation/imputation: {df.shape}")


            # --- Feature Engineering (refactored to use FeatureGenerator) ---
            logger.info("Generating features using FeatureGenerator...")
            features_df = self.feature_gen.generate_features_for_dataset(df)
            logger.info(f"DataFrame shape after feature generation: {features_df.shape}")


            # --- Encoding Categorical Features ---
            cat_cols = ["league", "venue"] # Define categorical columns to encode
            # Check which columns actually exist in the DataFrame
            cols_to_encode = [col for col in cat_cols if col in features_df.columns]
            if cols_to_encode:
                 logger.info(f"One-hot encoding columns: {cols_to_encode}")
                 encoded_data = self.encoder.fit_transform(features_df[cols_to_encode])
                 encoded_df = pd.DataFrame(encoded_data, index=features_df.index, columns=self.encoder.get_feature_names_out(cols_to_encode))

                 # Drop original categorical columns and concatenate encoded ones
                 features_df = features_df.drop(columns=cols_to_encode)
                 features_df = pd.concat([features_df, encoded_df], axis=1)
                 logger.info(f"DataFrame shape after encoding: {features_df.shape}")
            else:
                 logger.warning(f"None of the specified categorical columns {cat_cols} found for encoding.")

            # --- Final Checks ---
            # Check for any remaining NaNs after all steps
            final_nan_check = features_df.isnull().sum().sum()
            if final_nan_check > 0:
                 logger.warning(f"Preprocessing finished, but {final_nan_check} NaN values remain!")
            else:
                 logger.info("Preprocessing finished. No NaN values detected.")

            logger.info(f"Preprocessing complete. Final DataFrame shape: {features_df.shape}")
            return features_df

        except Exception as e:
            logger.critical(f"Preprocessing pipeline failed: {str(e)}", exc_info=True) # Log traceback
            raise


if __name__ == "__main__":
    raw_data = pd.read_json("data/raw/fixtures.json")
    preprocessor = MatchPreprocessor()
    clean_data = preprocessor.preprocess(raw_data)
    clean_data.to_csv("data/processed/training_data.csv", index=False)
    logger.info("Preprocessing completed successfully.")
