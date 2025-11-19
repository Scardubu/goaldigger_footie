#!/usr/bin/env python
"""
Environment setup script for GoalDiggers ML pipeline.
Handles common environment settings to ensure smooth operation.
"""
import logging
import os
import sys


def setup_environment():
    """Configure environment variables for consistent operation."""
    # Silence Git warnings in MLflow
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    
    # Silence setuptools/distutils warning
    os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"
    
    # Set MLflow tracking URI if not already set
    if "MLFLOW_TRACKING_URI" not in os.environ:
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    
    # Setup standard logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Add project root to Python path
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return logging.getLogger("setup_env")

if __name__ == "__main__":
    logger = setup_environment()
    logger.info("Environment successfully configured for GoalDiggers ML pipeline")
    logger.info(f"Project root: {os.path.dirname(os.path.dirname(__file__))}")
    logger.info(f"MLflow tracking URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
