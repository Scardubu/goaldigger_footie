"""
Logging configuration for the GoalDiggers platform.
Sets up logging for both development and production environments.
"""

import logging
import logging.config
import os
from pathlib import Path


def setup_logging(environment="production"):
    """
    Set up logging for the GoalDiggers platform.
    
    Args:
        environment: Either "development" or "production"
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Define log file path
    log_file = os.path.join("logs", "goaldiggers.log")
    
    # Define logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO" if environment == "production" else "DEBUG",
                "formatter": "simple",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10
            }
        },
        "root": {
            "level": "INFO" if environment == "production" else "DEBUG",
            "handlers": ["console", "file"]
        },
        "loggers": {
            "dashboard": {
                "level": "INFO" if environment == "production" else "DEBUG"
            },
            "utils": {
                "level": "INFO" if environment == "production" else "DEBUG"
            },
            "ml": {
                "level": "INFO" if environment == "production" else "DEBUG"
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured for {environment} environment")


# Call this function at import time
if __name__ != "__main__":
    # Determine environment based on environment variable
    env = os.environ.get("GOALDIGGERS_ENV", "production")
    setup_logging(environment=env)
