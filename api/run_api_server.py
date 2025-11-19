#!/usr/bin/env python3
"""
Run script for the API server component only.
This can be used to start the API server independently for debugging or development.
"""

import logging
import os
import sys

import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def run_api_server():
    """Run the API server directly using Uvicorn."""
    logger.info("Starting API Server directly...")
    
    # Ensure we're in the correct directory context
    api_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(api_dir)
    os.chdir(project_root)
    
    # Set environment variables
    os.environ["GOALDIGGERS_PRODUCTION_MODE"] = "true"
    
    # Run the server
    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=5000,
            log_level="info",
            reload=False  # Set to True for development
        )
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_api_server()
