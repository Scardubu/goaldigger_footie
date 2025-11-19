import logging
import os
from functools import lru_cache
from typing import Optional  # Added Optional import

import joblib
import numpy as np
import pandas as pd  # Added for potential feature DataFrame handling
import torch  # Added import
# import yaml # No longer needed for direct loading here
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dashboard.error_log import log_error  # Import log_error
from utils.config import Config, ConfigError  # Import centralized config

# Note: XGBClassifier might not be needed directly if loading via joblib
# from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# DEFAULT_CONFIG_PATH is no longer needed as we use the central config

class EnsemblePredictor:
    def __init__(self): # Removed config_path argument
        # Ensure central config is loaded
        try:
            Config.load()
        except ConfigError as e:
            logger.error(f"CRITICAL: Failed to load configuration via utils.config: {e}")
            raise
        except Exception as e:
            logger.error(f"CRITICAL: Unexpected error loading configuration via utils.config: {e}", exc_info=True)
            raise

        # Get parameters directly from Config
        # Note: Using more specific paths from the updated config.yaml structure
        self.static_model_repo = Config.get("models.huggingface.static_model.repo_id", "your-org/football-xgboost") # Placeholder
        self.dynamic_model_repo = Config.get("models.huggingface.dynamic_model.repo_id", "your-org/football-bert") # Placeholder
        # Static model filename comes from paths.models.predictor_filename, loaded in _load_static_model
        # self.static_model_filename = Config.get("models.huggingface.static_model.filename", "predictor_model.joblib") # Redundant here
        weights_dict = Config.get("models.ensemble.weights", {"static": 0.7, "dynamic": 0.3})
        self.weights = [weights_dict.get("static", 0.7), weights_dict.get("dynamic", 0.3)] # Extract list from dict
        self.enable_gpu = Config.get("models.ensemble.enable_gpu", True)
        self.dynamic_batch_size = Config.get("models.ensemble.dynamic_batch_size", 1) # Load batch size

        if len(self.weights) != 2:
             logger.warning(f"Expected 2 ensemble weights, got {len(self.weights)}. Using default [0.7, 0.3].")
             self.weights = [0.7, 0.3]

        self.device = self._get_device()
        self.static_model = self._load_static_model()
        self.tokenizer, self.dynamic_model = self._load_dynamic_model()

    # _load_config method is removed as configuration is handled by utils.config.Config

    def _get_device(self):
        """Determines the appropriate device (GPU if available, else CPU)."""
        if self.enable_gpu and torch.cuda.is_available():
            logger.info("CUDA is available. Using GPU.")
            return torch.device("cuda")
        elif self.enable_gpu:
            logger.warning("CUDA not available, falling back to CPU despite enable_gpu=True.")
            return torch.device("cpu")
        else:
            logger.info("GPU usage disabled in config. Using CPU.")
            return torch.device("cpu")

    def _load_static_model(self):
        """Loads the static model (e.g., XGBoost) from the local filesystem using paths from central config."""
        # Get paths from central Config
        project_root = Config.get("paths.project_root") # Assumes PROJECT_ROOT is set correctly
        if not project_root:
             # Fallback if PROJECT_ROOT env var wasn't set during Config load
             project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
             logger.warning(f"PROJECT_ROOT not found in config, using calculated path: {project_root}")

        model_dir_relative = Config.get("paths.models.base", "models/") # Default if not in config
        model_filename = Config.get("paths.models.predictor_filename", "predictor_model.joblib") # Use consistent filename key

        model_dir_absolute = os.path.join(project_root, model_dir_relative)
        model_path = os.path.join(model_dir_absolute, model_filename)

        logger.info(f"Loading static model from local path: {model_path}")
        try:
            # Load the dictionary saved by MatchPredictor
            model_data = joblib.load(model_path)
            model = model_data.get("model") # Extract the actual model object

            if model is None:
                 raise ValueError("Loaded file does not contain a 'model' key in the dictionary.")

            # Basic check: does it have predict_proba?
            if not hasattr(model, 'predict_proba'):
                 logger.error(f"Loaded static model from {model_path} does not have a 'predict_proba' method.")
                 return None
            logger.info("Static model loaded successfully from local path.")
            return model
        except FileNotFoundError:
             logger.error(f"Static model file not found at {model_path}. Please ensure the model is trained and saved using 'scripts/model.py --mode train'.")
             return None
        except Exception as e:
            log_error("High-level operation failed", e)
            return None # Keep return statement

    @lru_cache(maxsize=1)  # Cache model loading
    def _load_dynamic_model(self):
        """Loads the dynamic model (transformer) and tokenizer from Hugging Face Hub."""
        logger.info(f"Loading dynamic model and tokenizer from repo '{self.dynamic_model_repo}' on {self.device}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.dynamic_model_repo)
            # Load model directly onto the target device using device_map
            # This requires the 'accelerate' library to be installed.
            model = AutoModelForSequenceClassification.from_pretrained(
                self.dynamic_model_repo,
                device_map=self.device
            )
            # No need for model.to(self.device) as device_map handles placement
            logger.info(f"Dynamic model and tokenizer loaded successfully onto {self.device}.")
            return tokenizer, model
        except ImportError:
            logger.warning("The 'accelerate' library is not installed. Falling back to loading model on CPU first.")
            try:
                # Fallback: Load on CPU first, then move. Might still hit meta device issue for large models.
                tokenizer = AutoTokenizer.from_pretrained(self.dynamic_model_repo)
                model = AutoModelForSequenceClassification.from_pretrained(self.dynamic_model_repo)
                model.to(self.device) # Attempt to move the model
                logger.info(f"Dynamic model and tokenizer loaded successfully onto {self.device} (CPU fallback).")
                return tokenizer, model
            except Exception as e_fallback:
                 logger.exception(
                    f"Failed to download or load dynamic model/tokenizer from HF Hub ({self.dynamic_model_repo}) even with CPU fallback: {e_fallback}"
                 )
                 return None, None
        except Exception as e:
            log_error("High-level operation failed", e)
            return None, None # Keep return statement

    def _normalize_features(self, features):
        """Normalize features using config-defined scaler, aligning to expected features."""
        # Get normalization parameters from central Config
        feature_means_dict = Config.get("models.normalization.feature_means", {})
        feature_stds_dict = Config.get("models.normalization.feature_stds", {})
        expected_features = Config.get("models.normalization.feature_list", []) # Get the list of features used

        if not expected_features:
             # Fallback if feature list is empty in config (less robust)
             logger.warning("Normalization feature list not found in config (models.normalization.feature_list). Attempting to infer from static model or using hardcoded fallback.")
             if hasattr(self.static_model, 'feature_names_in_'):
                 expected_features = list(self.static_model.feature_names_in_)
                 logger.info(f"Inferred expected features from static model: {expected_features}")
             else:
                 # Hardcoded fallback (least desirable)
                 expected_features = [
                     'home_avg_goals_scored_last_5', 'home_avg_goals_conceded_last_5', 'home_form_points_last_5',
                     'away_avg_goals_scored_last_5', 'away_avg_goals_conceded_last_5', 'away_form_points_last_5',
                     'h2h_team1_wins', 'h2h_draws', 'h2h_team2_wins', 'h2h_avg_goals',
                 ]
                 logger.warning(f"Using hardcoded feature list for normalization: {expected_features}")

        if not feature_means_dict or not feature_stds_dict:
             logger.error("Feature means or stds not found in config (models.normalization). Cannot normalize.")
             return features # Return original if params missing

        # Align input features to the expected list
        if isinstance(features, pd.DataFrame):
            # Ensure all expected columns exist, fill missing with 0 (or mean?)
            for col in expected_features:
                if col not in features.columns:
                    features[col] = 0 # Or handle differently?
            # Select and order columns
            features_aligned = features[expected_features]
        elif isinstance(features, (list, np.ndarray)):
            # Assuming the list/array is already in the correct order and length
            if len(features) != len(expected_features):
                 logger.error(f"Input feature array/list length ({len(features)}) does not match expected length ({len(expected_features)}). Cannot normalize reliably.")
                 # Create a zero-filled DataFrame as a fallback?
                 features_aligned = pd.DataFrame(np.zeros((1, len(expected_features))), columns=expected_features)
            else:
                 features_aligned = pd.DataFrame([features], columns=expected_features)
        else:
            logger.error(f"Unsupported feature type for normalization: {type(features)}")
            return features # Return original if type unknown

        # Create Series from dicts using the expected feature order
        means = pd.Series(feature_means_dict).reindex(expected_features).fillna(0)
        stds = pd.Series(feature_stds_dict).reindex(expected_features).fillna(1)
        stds[stds == 0] = 1 # Avoid division by zero

        logger.debug(f"Normalizing features using means: {means.to_dict()} and stds: {stds.to_dict()}")
        return (features_aligned - means) / stds

    def _get_perf_monitor(self):
        if not hasattr(self, '_perf_monitor'):
            from scripts.core.monitoring import PerformanceMonitor
            self._perf_monitor = PerformanceMonitor()
        return self._perf_monitor

    def predict(self, features, news_text: str) -> Optional[np.ndarray]:
        """
        Generates ensemble prediction with dynamic weighting and fallback logic.
        Logs confidence and source of each prediction.
        """
        perf_monitor = self._get_perf_monitor()
        import time
        start_time = time.time()
        try:
            # Try static model prediction
            static_pred = None
            static_conf = 0
            try:
                features_normalized = self._normalize_features(features)
                if isinstance(features_normalized, pd.DataFrame):
                    features_input = features_normalized
                elif isinstance(features_normalized, np.ndarray):
                    features_input = features_normalized.reshape(1, -1) if features_normalized.ndim == 1 else features_normalized
                else:
                    features_input = None
                if self.static_model and features_input is not None:
                    static_pred_proba = self.static_model.predict_proba(features_input)
                    static_pred = static_pred_proba[0]
                    static_conf = float(np.max(static_pred))
            except Exception as e:
                logger.warning(f"Static model prediction failed: {e}")

            # Try dynamic model prediction
            dynamic_pred = None
            dynamic_conf = 0
            try:
                if self.dynamic_model and self.tokenizer:
                    dynamic_pred_tensor = self._analyze_news(news_text)
                    if dynamic_pred_tensor is not None:
                        dynamic_pred = dynamic_pred_tensor.detach().cpu().numpy().flatten()
                        dynamic_conf = float(np.max(dynamic_pred))
            except Exception as e:
                logger.warning(f"Dynamic model prediction failed: {e}")

            # Fallback logic
            if static_pred is None and dynamic_pred is None:
                logger.error("Both static and dynamic models failed. Returning uniform probabilities.")
                perf_monitor.update('model', False, time.time() - start_time)
                return np.array([1/3, 1/3, 1/3])
            if static_pred is None:
                logger.warning("Static model failed. Using dynamic model only.")
                # Map 2-class dynamic output to 3-class (H/D/A) as best as possible
                # Here, treat dynamic_pred[1] as home win, dynamic_pred[0] as away win, draw as 1 - sum
                if dynamic_pred is not None and dynamic_pred.shape[0] == 2:
                    home = dynamic_pred[1]
                    away = dynamic_pred[0]
                    draw = max(0, 1 - (home + away))
                    pred = np.array([home, draw, away])
                    pred = pred / pred.sum() if pred.sum() > 0 else np.array([1/3, 1/3, 1/3])
                    logger.info(f"Prediction source: dynamic only. Confidence: {dynamic_conf:.3f}")
                    perf_monitor.update('model', True, time.time() - start_time)
                    return pred
            if dynamic_pred is None:
                logger.warning("Dynamic model failed. Using static model only.")
                logger.info(f"Prediction source: static only. Confidence: {static_conf:.3f}")
                perf_monitor.update('model', True, time.time() - start_time)
                return static_pred

            # Dynamic weighting based on confidence
            total_conf = static_conf + dynamic_conf
            static_weight = static_conf / total_conf if total_conf > 0 else self.weights[0]
            dynamic_weight = dynamic_conf / total_conf if total_conf > 0 else self.weights[1]
            # Map dynamic_pred to 3-class as above
            home = dynamic_pred[1]
            away = dynamic_pred[0]
            draw = max(0, 1 - (home + away))
            dynamic_pred_3 = np.array([home, draw, away])
            dynamic_pred_3 = dynamic_pred_3 / dynamic_pred_3.sum() if dynamic_pred_3.sum() > 0 else np.array([1/3, 1/3, 1/3])
            # Weighted average
            combined_pred = static_weight * static_pred + dynamic_weight * dynamic_pred_3
            combined_pred = combined_pred / combined_pred.sum() if combined_pred.sum() > 0 else np.array([1/3, 1/3, 1/3])
            logger.info(f"Prediction source: ensemble. Static conf: {static_conf:.3f}, Dynamic conf: {dynamic_conf:.3f}, Weights: {static_weight:.2f}/{dynamic_weight:.2f}")
            perf_monitor.update('model', True, time.time() - start_time)
            return combined_pred
        except Exception as e:
            perf_monitor.update('model', False, time.time() - start_time)
            logger.error(f"Ensemble prediction failed: {e}")
            return np.array([1/3, 1/3, 1/3])

    def _analyze_news(self, text: str) -> Optional[torch.Tensor]:
        """Analyzes news text using the dynamic model."""
        # Consistent indentation (8 spaces)
        if self.dynamic_model is None or self.tokenizer is None:
            # Consistent indentation (12 spaces)
            logger.error("Dynamic model or tokenizer not loaded.")
            return None
        # Consistent indentation (8 spaces)
        try:
            # Consistent indentation (12 spaces)
            # Use dynamic batch size from central Config (already loaded in __init__)
            # batch_size = self.dynamic_batch_size # Use the instance variable
            # Note: Simple implementation assumes single text input. Batching needs adjustment.
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                    padding=True, max_length=512).to(self.device) # Move inputs to device

            # Consistent indentation (12 spaces)
            with torch.no_grad(): # Disable gradient calculation for inference
                 # Consistent indentation (16 spaces)
                 outputs = self.dynamic_model(**inputs)

            # Consistent indentation (12 spaces) - Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Return probabilities for the first item in the batch (usually just one)
            return probabilities[0]
        # Consistent indentation (8 spaces)
        except Exception as e:
            # Consistent indentation (12 spaces)
            log_error("High-level operation failed", e)
            return None # Keep return statement

    # Placeholder for confidence scoring - requires defining how confidence is measured
    def get_confidence(self, prediction: np.ndarray) -> Optional[float]:
        """
        Calculates a confidence score for the prediction.
        Example: Use prediction entropy or max probability.
        """
        if prediction is None:
            return None
        # Example: Max probability as confidence
        confidence = np.max(prediction)
        logger.debug(f"Calculated prediction confidence: {confidence:.4f}")
        return float(confidence)


# Example Usage (Illustrative - requires actual models and config)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     # Ensure config/model_params.yaml exists and is configured
#     predictor = EnsemblePredictor()
#
#     # Example features (replace with actual data structure)
#     # Needs to match expected input for static model and normalization config
#     # Example assuming 5 features and DataFrame input:
#     sample_features_df = pd.DataFrame([[1.0, 2.5, 0.8, 5.1, -0.2]], columns=['f1', 'f2', 'f3', 'f4', 'f5'])
#     sample_news = "Team A looks strong with their new signing, but Team B has shown resilience."
#
#     if predictor.static_model and predictor.dynamic_model:
#         prediction = predictor.predict(sample_features_df, sample_news)
#         if prediction is not None:
#             print(f"Ensembled Prediction: {prediction}")
#             confidence = predictor.get_confidence(prediction)
#             print(f"Prediction Confidence: {confidence}")
#
#         # Example Batch Prediction (conceptual)
#         # features_batch = [sample_features_df, sample_features_df.copy()] # List of DataFrames
#         # text_batch = [sample_news, "Another match analysis text."]
#         # def predict_batch(features_list, texts):
#         #      # Note: Batching dynamic model inference requires tokenizer/model adjustments
#         #      return [predictor.predict(f, t) for f, t in zip(features_list, texts)]
#         # batch_results = predict_batch(features_batch, text_batch)
#         # print(f"Batch Predictions: {batch_results}")
#
#         # Validation Check Example (conceptual)
#         # if predictor.device.type == 'cuda':
#         #     assert next(predictor.dynamic_model.parameters()).is_cuda
#         #     print("GPU check passed.")
#         #
#         # raw_features = pd.DataFrame([[1.5, 3.0, 0.5, 6.0, 0.0]], columns=['f1', 'f2', 'f3', 'f4', 'f5'])
#         # normalized_features = predictor._normalize_features(raw_features)
#         # print(f"Normalized Features: \n{normalized_features}")
#         # # Add assertion based on expected means/stds from config
#         # # assert np.isclose(normalized_features.mean(axis=0)[0], 0, atol=0.1) # Check mean of first feature
#
#     else:
#         print("Models not loaded properly. Cannot run example.")
