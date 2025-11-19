#!/usr/bin/env python3
"""
Dashboard Test Script

This script tests the dashboard deployment and accessibility.
It starts the dashboard and verifies it can be accessed.
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

async def test_dashboard_startup():
    """Test dashboard startup and accessibility."""
    try:
        logger.info("Testing dashboard startup...")
        
        # Dashboard file path
        dashboard_path = project_root / "dashboard" / "integrated_production_app.py"
        
        if not dashboard_path.exists():
            logger.error(f"Dashboard file not found: {dashboard_path}")
            return False
        
        # Start dashboard in background
        streamlit_cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", "8503",  # Use different port for testing
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ]
        
        logger.info(f"Starting dashboard on port 8503...")
        
        process = subprocess.Popen(
            streamlit_cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for startup
        logger.info("Waiting for Streamlit to initialize...")
        await asyncio.sleep(15)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("Dashboard process started successfully")
            
            # Try to access the dashboard
            try:
                import aiohttp
                
                url = "http://localhost:8503"
                logger.info(f"Attempting to access dashboard at {url}...")
                
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(url, timeout=30) as response:
                            if response.status == 200:
                                logger.info("✅ Dashboard is accessible!")
                                return True
                            else:
                                logger.warning(f"Dashboard returned status: {response.status}")
                                return False
                    except Exception as e:
                        logger.error(f"Failed to connect to dashboard: {e}")
                        return False
            except ImportError:
                logger.error("aiohttp not available - can't test HTTP access")
                return False
            finally:
                # Terminate the process
                logger.info("Terminating test dashboard process...")
                process.terminate()
                process.wait(timeout=5)
        else:
            stdout, stderr = process.communicate()
            logger.error(f"Dashboard failed to start: {stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Dashboard test failed: {e}")
        return False

async def main():
    """Run the dashboard test."""
    try:
        result = await test_dashboard_startup()
        
        if result:
            logger.info("✅ Dashboard test passed!")
            sys.exit(0)
        else:
            logger.error("❌ Dashboard test failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
