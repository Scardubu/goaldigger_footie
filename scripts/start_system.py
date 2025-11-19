#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from dashboard.app import DashboardApp
from scripts.core.startup_manager import StartupManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def start_scraper_server():
    """Start the Firecrawl scraper server."""
    try:
        # Get scraper server configuration
        scraper_config = Config.get('scraping.server', {})
        host = scraper_config.get('host', 'localhost')
        port = scraper_config.get('port', 3000)
        
        # Start the server using subprocess
        import subprocess
        server_path = os.path.join('MCP', 'fircrawl-scraper-server')
        if not os.path.exists(server_path):
            logger.error(f"Scraper server directory not found: {server_path}")
            return False
            
        # Start the server
        process = subprocess.Popen(
            ['npm', 'start'],
            cwd=server_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)  # Give server time to start
        
        # Check if server is running
        import requests
        try:
            response = requests.get(f'http://{host}:{port}/health')
            if response.status_code == 200:
                logger.info("Scraper server started successfully")
                return True
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to scraper server")
            return False
            
    except Exception as e:
        logger.error(f"Error starting scraper server: {e}")
        return False

async def start_dashboard():
    """Start the dashboard application."""
    try:
        dashboard = DashboardApp()
        if await dashboard.initialize():
            logger.info("Dashboard initialized successfully")
            dashboard.render()
            return True
        else:
            logger.error("Failed to initialize dashboard")
            return False
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False

async def main():
    """Main startup function."""
    start_time = datetime.now()
    logger.info("Starting system...")
    
    try:
        # 1. Initialize startup manager
        startup_manager = StartupManager()
        if not await startup_manager.initialize():
            logger.error("System initialization failed")
            return False
            
        # 2. Start scraper server
        if not await start_scraper_server():
            logger.error("Failed to start scraper server")
            return False
            
        # 3. Start dashboard
        if not await start_dashboard():
            logger.error("Failed to start dashboard")
            return False
            
        # 4. Log startup time
        end_time = datetime.now()
        startup_duration = (end_time - start_time).total_seconds()
        logger.info(f"System started successfully in {startup_duration:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        return False
    finally:
        # Cleanup
        if 'startup_manager' in locals():
            await startup_manager.cleanup()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Run the startup process
    from utils.asyncio_compat import ensure_loop
    loop = ensure_loop()
    try:
        success = loop.run_until_complete(main())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Startup interrupted by user")
        sys.exit(1)
    finally:
        try:
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass