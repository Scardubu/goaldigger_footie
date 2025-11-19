"""
Utility script to compare and print mismatches between the model's feature_columns and the config's feature_list.
"""
import sys

from models.xgboost_predictor import XGBoostPredictor
from utils.config import Config


def compare_features(model_path):
    # Load config feature list
    feature_list = Config.get("models.normalization.feature_list", [])
    print(f"Config feature_list ({len(feature_list)}): {feature_list}")

    # Load model and get feature_columns
    predictor = XGBoostPredictor(model_path)
    model_features = list(predictor.feature_columns)
    print(f"Model feature_columns ({len(model_features)}): {model_features}")

    # Compare
    missing_in_model = [f for f in feature_list if f not in model_features]
    extra_in_model = [f for f in model_features if f not in feature_list]
    order_mismatch = [i for i, (a, b) in enumerate(zip(feature_list, model_features)) if a != b]

    print("\n--- Feature Comparison ---")
    if not missing_in_model and not extra_in_model and not order_mismatch:
        print("Feature lists match exactly in name and order.")
    else:
        if missing_in_model:
            print(f"Features in config but missing in model: {missing_in_model}")
        if extra_in_model:
            print(f"Features in model but missing in config: {extra_in_model}")
        if order_mismatch:
            print(f"Order mismatch at indices: {order_mismatch}")
            print("Config order:", [feature_list[i] for i in order_mismatch])
            print("Model order:", [model_features[i] for i in order_mismatch])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_features.py <model_path>")
        sys.exit(1)
    compare_features(sys.argv[1])
