#!/usr/bin/env python
import os
import shutil

import joblib
import optuna
import pandas as pd
import yaml
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from models.predictive.analytics_model import build_pipeline
from scripts.ingest_data import preprocess


# helper to load & preprocess once
def load_data(raw_path, cache_path, config):
    cache_file = os.path.join(cache_path, "preprocessed.pkl")
    if os.path.exists(cache_file):
        df = joblib.load(cache_file)
    else:
        df_raw = pd.read_csv(os.path.join(raw_path, "matches.csv"))
        df = preprocess(df_raw, config)
        os.makedirs(cache_path, exist_ok=True)
        joblib.dump(df, cache_file)
    return df

def objective(trial):
    # Load base config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Suggest hyperparameters
    config["model"]["training"]["num_boost_round"] = trial.suggest_int("num_boost_round", 100, 1000)
    config["model"]["training"]["early_stopping_rounds"] = trial.suggest_int("early_stopping_rounds", 10, 100)
    config["class_balance"]["use_focal_loss"] = False  # use SMOTE during tuning
    config["hyperparameter_search"]["eta"] = trial.suggest_float("eta", 0.01, 0.3)

    # Save temp config for record
    temp_cfg_path = f"config/temp_optuna_{trial.number}.yaml"
    with open(temp_cfg_path, "w") as f:
        yaml.safe_dump(config, f)

    # Get data paths from config, using proper structure
    raw_path = config.get("paths", {}).get("data", {}).get("raw_path", "./data/raw")
    cache_path = config.get("paths", {}).get("data", {}).get("cache_path", "./data/cache")
    
    # Create directories if they don't exist
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(cache_path, exist_ok=True)
    
    # Load data
    df = load_data(
        raw_path=raw_path,
        cache_path=cache_path,
        config=config
    )

    # Split train/val
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build & fit pipeline
    model = build_pipeline(config)
    model.set_params(xgb__learning_rate=config["hyperparameter_search"]["eta"])
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)

    # Clean up temp config
    os.remove(temp_cfg_path)

    return loss

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    with open("config/config.yaml") as f:
        base_cfg = yaml.safe_load(f)
    n_trials = base_cfg["hyperparameter_search"]["n_trials"]
    timeout = base_cfg["hyperparameter_search"]["timeout"]

    study = optuna.create_study(direction="minimize", 
                                study_name="hp_tuning", 
                                sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    print("Best trial:")
    print(f"  Value (logloss): {study.best_value}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
