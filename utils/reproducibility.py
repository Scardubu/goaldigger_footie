"""
Reproducibility utilities for machine learning experiments.

This module provides comprehensive functionality for ensuring reproducible
machine learning experiments across different libraries and environments.
It handles random seed setting, environment configuration, and deterministic 
operations for popular ML libraries.

Usage:
    from utils.reproducibility import set_global_seed, ensure_deterministic_environment
    
    # Basic usage
    set_global_seed(42)
    
    # For maximum reproducibility
    ensure_deterministic_environment(seed=42)
"""

import logging
import os
import random
import warnings
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seed for reproducible training across all supported libraries.

    This includes:
    - Python random
    - NumPy
    - XGBoost (config only)
    - LightGBM
    - CatBoost
    - PyTorch
    - TensorFlow
    - Scikit-learn (via global config where applicable)

    Args:
        seed: Integer seed to be applied globally
    """
    logger.info(f"Setting global random seed to {seed}")
    
    # Core Python and NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # XGBoost
    try:
        import xgboost as xgb
        xgb.set_config(verbosity=0)
        logger.debug("XGBoost seed configuration applied")
    except ImportError:
        logger.debug("XGBoost not available - skipping seed configuration")

    # LightGBM
    try:
        import lightgbm as lgb
        lgb.basic._Config.set("seed", seed)
        lgb.basic._Config.set("feature_fraction_seed", seed)
        lgb.basic._Config.set("bagging_seed", seed)
        lgb.basic._Config.set("drop_seed", seed)
        lgb.basic._Config.set("data_random_seed", seed)
        logger.debug("LightGBM seed configuration applied")
    except Exception:
        logger.debug("LightGBM not available or failed - skipping seed configuration")

    # CatBoost
    try:
        import catboost
        os.environ["CATBOOST_RANDOM_SEED"] = str(seed)
        logger.debug("CatBoost seed environment variable set")
    except ImportError:
        logger.debug("CatBoost not available - skipping seed configuration")

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch seed configuration applied")
    except ImportError:
        logger.debug("PyTorch not available - skipping seed configuration")

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.debug("TensorFlow seed configuration applied")
    except ImportError:
        logger.debug("TensorFlow not available - skipping seed configuration")


def ensure_deterministic_environment(seed: int = 42) -> None:
    """
    Set up all environment variables and seed settings to ensure deterministic behavior.
    
    This is stronger than just `set_global_seed` and should be used at the very beginning 
    of any ML pipeline execution.
    
    Args:
        seed: Random seed to apply across all frameworks and system-level settings
    """
    logger.info(f"Ensuring deterministic environment with seed {seed}")

    # Core seeding
    set_global_seed(seed)

    # Extra environmental settings
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For deterministic cuBLAS
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Useful for debugging GPU errors

    # Warnings for potential nondeterminism
    warnings.filterwarnings("once", category=UserWarning)
    logger.info("Deterministic environment variables set")
