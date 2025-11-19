"""
GoalDiggers Dashboard - Main Runner Script

This script provides a clean entry point to run the Streamlit application.
It handles initialization, asyncio configuration, and error reporting.
"""
import os
import sys
import asyncio
import logging
import traceback
import streamlit as st

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Apply nest_asyncio early
import nest_asyncio
nest_asyncio.apply()

# Configure asyncio for Windows compatibility
if sys.platform == "win32" and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Import the app module
from dashboard.app import async_main
from utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def global_exception_handler(loop, context):
    """Global exception handler for asyncio errors."""
    exception = context.get('exception')
    logger.critical(f"Unhandled asyncio exception: {exception}", exc_info=exception)
    print(f"CRITICAL ASYNCIO ERROR: {exception}", file=sys.stderr)

def main():
    """Entry point for the application."""
    try:
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(global_exception_handler)
        
        # Run the main application
        loop.run_until_complete(async_main())
    except Exception as e:
        # This top-level catch is for issues during asyncio.run() itself
        error_message = f"Fatal error during application startup: {e}"
        logger.critical(error_message, exc_info=True)
        
        # Try to display the error in Streamlit if possible
        try:
            st.error(error_message)
            with st.expander("Error Details"):
                st.code(traceback.format_exc(), language="python")
        except:
            # If Streamlit is not initialized yet, print to stderr
            print(error_message, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

if __name__ == "__main__":
    main()
