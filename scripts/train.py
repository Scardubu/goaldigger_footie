#!/usr/bin/env python
import argparse
import logging
import os
import sys
from pathlib import Path

# Import environment setup to silence warnings and configure paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.setup_env import setup_environment

# Configure environment first
logger = setup_environment()

# Now import analytics modules
from models.predictive.analytics_model import train


def main():
    parser = argparse.ArgumentParser(
        description="Train the football insights model end-to-end."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to YAML config file."
    )
    args = parser.parse_args()

    # Logger already configured by setup_environment
    logger = logging.getLogger(__name__)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        exit(1)

    logger.info(f"Starting training with config: {config_path}")
    train(str(config_path))
    logger.info("Training script finished.")

if __name__ == "__main__":
    main()
