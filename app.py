#!/usr/bin/env python3
"""
GoalDiggers Main Application Entry Point

This is the main entry point that routes to the enhanced production application.
For direct access to the enhanced features, use enhanced_app.py.
"""

import logging
import sys
import warnings

from utils.logging_config import configure_logging

# Suppress warnings for production
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging using centralized configuration
configure_logging()
logger = logging.getLogger(__name__)

def main():
    """Main application entry point."""
    try:
        logger.info("[INFO] Starting GoalDiggers Platform...")
        
        # Import and run enhanced application
        from enhanced_app import main as enhanced_main
        enhanced_main()
        
    except ImportError as e:
        logger.error(f"Failed to import enhanced app: {e}")
        logger.info("Attempting fallback to basic dashboard...")
        
        # Fallback to basic functionality
        import streamlit as st

        # Note: set_page_config is called in enhanced_app, don't duplicate here
        
        st.title("⚽ GoalDiggers Platform")
        st.info("Running in basic mode. Enhanced features may be unavailable.")
        
        # Basic prediction interface
        st.subheader("Quick Prediction")
        home_team = st.selectbox("Home Team", ["Arsenal", "Chelsea", "Liverpool", "Manchester City"])
        away_team = st.selectbox("Away Team", ["Arsenal", "Chelsea", "Liverpool", "Manchester City"])
        
        if st.button("Generate Prediction"):
            st.success(f"Prediction generated for {home_team} vs {away_team}")
            
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
