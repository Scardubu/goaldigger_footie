#!/usr/bin/env python
"""
Playwright browser installation script.
Installs the chromium browser for use with Playwright.
"""
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)

BROWSERS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pw_browsers")

def ensure_browsers_dir():
    """Ensure the browsers directory exists."""
    os.makedirs(BROWSERS_PATH, exist_ok=True)
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = BROWSERS_PATH
    logger.info(f"Set PLAYWRIGHT_BROWSERS_PATH to: {BROWSERS_PATH}")
    return BROWSERS_PATH

def install_playwright_browsers():
    """Install Playwright browsers."""
    try:
        logger.info("Installing Playwright browsers...")
        
        # Set the environment variable for the installation
        env = os.environ.copy()
        env["PLAYWRIGHT_BROWSERS_PATH"] = BROWSERS_PATH
        
        # Use subprocess to run the playwright install command
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Playwright browser installation output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Playwright browser installation stderr: {result.stderr}")
            
        # Verify the installation
        chrome_path = Path(BROWSERS_PATH) / "chromium-1112" / "chrome-win" / "chrome.exe"
        if chrome_path.exists():
            logger.info(f"Verified browser installation: {chrome_path}")
            return True
        else:
            logger.error(f"Browser executable not found at expected path: {chrome_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing Playwright browsers: {e}")
        logger.error(f"Command output: {e.stdout}")
        logger.error(f"Command error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error installing Playwright browsers: {e}")
        return False

def main():
    """Main function."""
    browsers_path = ensure_browsers_dir()
    logger.info(f"Browsers will be installed to: {browsers_path}")
    
    success = install_playwright_browsers()
    
    if success:
        logger.info("Playwright setup completed successfully.")
    else:
        logger.error("Playwright setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
