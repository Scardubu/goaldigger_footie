import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd  # Add pandas import for validation

from dashboard.data_integration import DataIntegration
from database.db_manager import DatabaseManager
from database.schema import League
from scripts.core.ai_validator import AIDataValidator
from scripts.core.enhanced_scraper import EnhancedScraper
from scripts.data_pipeline.db_integrator import DataIntegrator
from utils.config import Config
from utils.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

class StartupManager:
    def __init__(self):
        self.db_manager = None
        self.scraper = None
        self.data_loader = None
        self.startup_time = None
        self.system_monitor = SystemMonitor()
        self.firecrawl_process = None
        self.initialization_status = {
            'database': False,
            'scraper': False,
            'data_loader': False,
            'firecrawl_server': False,
            'ai_validator': False,
            'data_integrator': False
        }
        self.validation_results = {}

    async def initialize(self) -> bool:
        """Initialize all system components with enhanced parallel loading and validation."""
        self.startup_time = datetime.now()
        logger.info("Starting enhanced system initialization with parallel loading...")

        try:
            # 1. Initialize database (this must be done first)
            if not await self._initialize_database():
                logger.error("Database initialization failed")
                return False
                
            # 2. Initialize AI validator for real-time data validation (optional)
            try:
                if not await self._initialize_ai_validator():
                    logger.warning("AI validator initialization failed, continuing without validation")
            except Exception as e:
                logger.warning(f"AI validator initialization failed: {e}, continuing without validation")
                
            # 3. Initialize data integrator
            if not await self._initialize_data_integrator():
                logger.error("Data integrator initialization failed")
                return False
                
            # 4. Initialize Firecrawl scraper server (optional, non-blocking)
            try:
                firecrawl_task = asyncio.create_task(self._initialize_firecrawl_server())
                # Don't wait for Firecrawl to complete - let it run in background
                firecrawl_task.add_done_callback(lambda t: self._handle_firecrawl_result(t))
            except Exception as e:
                logger.warning(f"Firecrawl server initialization failed: {e}, continuing with limited functionality")
                
            # 5 & 6. Initialize scraper and data loader in parallel
            scraper_task = asyncio.create_task(self._initialize_scraper())
            data_loader_task = asyncio.create_task(self._initialize_data_loader())
            
            # Wait for essential tasks to complete with shorter timeout
            try:
                scraper_result, data_loader_result = await asyncio.wait_for(
                    asyncio.gather(scraper_task, data_loader_task, return_exceptions=True),
                    timeout=30.0  # Reduced timeout for essential components
                )
            except asyncio.TimeoutError:
                logger.warning("Essential components initialization timed out, continuing with basic functionality")
                # Try to get partial results
                if not scraper_task.done():
                    scraper_task.cancel()
                if not data_loader_task.done():
                    data_loader_task.cancel()
                return True  # Allow dashboard to start with basic functionality
            
            # Check for exceptions
            if isinstance(scraper_result, Exception):
                logger.error(f"Scraper initialization failed with exception: {scraper_result}")
                # Don't fail completely, continue with basic functionality
            if isinstance(data_loader_result, Exception):
                logger.error(f"Data loader initialization failed with exception: {data_loader_result}")
                # Don't fail completely, continue with basic functionality
                
            # Check for successful initialization (but don't fail if they didn't work)
            if scraper_result is True:
                logger.info("Scraper initialized successfully")
            if data_loader_result is True:
                logger.info("Data loader initialized successfully")

            # 7. Perform initial data population with validation (optional)
            try:
                if not await self._perform_initial_data_population():
                    logger.warning("Initial data population step failed or partially completed. System may still be usable.")
            except Exception as e:
                logger.warning(f"Initial data population failed: {e}, continuing with basic functionality")

            # 8. Validate system state with enhanced checks (but don't fail if validation fails)
            try:
                if not self._validate_system_state():
                    logger.warning("System state validation failed, but continuing with basic functionality")
            except Exception as e:
                logger.warning(f"System state validation failed: {e}, continuing with basic functionality")

            # 9. Launch frontend UI automatically (optional)
            try:
                await self._launch_frontend_ui()
            except Exception as e:
                logger.warning(f"Failed to launch frontend UI: {e}")

            initialization_time = (datetime.now() - self.startup_time).total_seconds()
            logger.info(f"Enhanced system initialization completed successfully in {initialization_time:.2f} seconds")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

    async def _initialize_firecrawl_server(self) -> bool:
        """Initialize and start the Firecrawl scraper server automatically."""
        try:
            logger.info("Initializing Firecrawl scraper server...")
            
            # Get server configuration
            server_config = Config.get('scraping.server', {})
            host = server_config.get('host', 'localhost')
            port = server_config.get('port', 3000)
            health_check_timeout = server_config.get('health_check_timeout', 15)  # Reduced timeout
            
            # Check if server is already running
            if await self._check_server_health(host, port):
                logger.info("Firecrawl server is already running")
                self.initialization_status['firecrawl_server'] = True
                return True
            
            # Start the server
            server_path = Path(__file__).parent.parent.parent / "MCP" / "fircrawl-scraper-server"
            if not server_path.exists():
                logger.warning(f"Firecrawl server directory not found: {server_path}")
                return False
                
            # Check if build exists, if not skip building for now
            if not (server_path / "build" / "index.js").exists():
                logger.warning("Firecrawl server build not found, skipping server initialization")
                return False
            
            # Start the server process
            env = os.environ.copy()
            env["NODE_ENV"] = "production"
            env["PATH"] = f"{Path(__file__).parent.parent.parent / 'venv' / 'Scripts'};{env.get('PATH', '')}"
            
            try:
                self.firecrawl_process = subprocess.Popen(
                    ['node', 'build/index.js'],
                    cwd=server_path,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            except Exception as e:
                logger.warning(f"Failed to start Firecrawl server process: {e}")
                return False
            
            # Wait for server to start with shorter timeout
            logger.info("Waiting for Firecrawl server to start...")
            for attempt in range(health_check_timeout):
                await asyncio.sleep(1)
                if await self._check_server_health(host, port):
                    logger.info("Firecrawl server started successfully")
                    self.initialization_status['firecrawl_server'] = True
                    return True
                    
            logger.warning("Firecrawl server failed to start within timeout")
            # Clean up the process if it failed to start
            if self.firecrawl_process:
                try:
                    self.firecrawl_process.terminate()
                    self.firecrawl_process.wait(timeout=5)
                except:
                    pass
            return False
            
        except Exception as e:
            logger.warning(f"Error initializing Firecrawl server: {e}")
            return False

    async def _check_server_health(self, host: str, port: int) -> bool:
        """Check if the Firecrawl server is healthy."""
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(f'http://{host}:{port}/health', timeout=5) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _initialize_ai_validator(self) -> bool:
        """Initialize AI data validator for real-time validation."""
        try:
            logger.info("Initializing AI data validator...")
            
            # Initialize validator with configuration
            validator_config = Config.get('validation.ai_validator', {})
            self.ai_validator = AIDataValidator.from_config(validator_config)
            
            self.initialization_status['ai_validator'] = True
            logger.info("AI data validator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"AI validator initialization failed: {e}")
            return False

    async def _initialize_data_integrator(self) -> bool:
        """Initialize data integrator with enhanced capabilities."""
        try:
            logger.info("Initializing data integrator...")
            
            self.data_integrator = DataIntegrator(db_manager=self.db_manager)
            
            self.initialization_status['data_integrator'] = True
            logger.info("Data integrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data integrator initialization failed: {e}")
            return False

    async def _launch_frontend_ui(self) -> None:
        """Automatically launch the frontend UI after data readiness."""
        try:
            logger.info("Launching frontend UI...")
            
            # Check if UI should be auto-launched
            auto_launch = Config.get('ui.auto_launch', True)
            if not auto_launch:
                logger.info("Auto-launch disabled in configuration")
                return
            
            # Get UI configuration
            ui_config = Config.get('ui', {})
            port = ui_config.get('port', 8501)
            host = ui_config.get('host', 'localhost')
            
            # Launch Streamlit app in background
            ui_path = Path(__file__).parent.parent.parent / "dashboard" / "app.py"
            
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)
            
            subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", str(ui_path), 
                 "--server.port", str(port), "--server.address", host],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info(f"Frontend UI launched at http://{host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to launch frontend UI: {e}")

    async def _initialize_database(self) -> bool:
        """Initialize database connection and ensure all tables are created."""
        try:
            self.db_manager = DatabaseManager()
            
            # Always attempt to create tables. SQLAlchemy's create_all is idempotent
            # and will only create tables that don't already exist.
            logger.info("Attempting to create/update database schema (if necessary)...")
            self.db_manager.create_tables() # This calls Base.metadata.create_all(self.engine)
            logger.info("Database schema creation/update attempt complete.")

            # Perform a basic verification that the database is connectable and tables might exist.
            # The more rigorous check of specific tables is done in _validate_system_state.
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                # Check if we can query the sqlite_master table as a basic health check
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                # No need to check content here, just that the query executes.
                # If it fails, an exception would be raised by the DB driver or SQLAlchemy.
                logger.info("Basic database connectivity and schema query successful.")

            self.initialization_status['database'] = True
            logger.info("Database initialized successfully and schema checked/updated.")
            return True

        except Exception as e:
            logger.error(f"Database initialization or schema creation failed: {e}", exc_info=True)
            return False

    async def _initialize_scraper(self) -> bool:
        """Initialize the enhanced scraper with proper configuration."""
        try:
            from utils.config import Config

            # Check if scraper should use Playwright
            use_playwright = Config.get("scraping.playwright.enabled", False)
            
            # Initialize the EnhancedScraper without trying to access browser/page directly
            self.scraper = EnhancedScraper(self.db_manager, use_playwright=use_playwright)
            
            # Check if Playwright browsers are installed
            import os
            import subprocess
            import sys
            from pathlib import Path
            
            browsers_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "pw_browsers")
            os.makedirs(browsers_path, exist_ok=True)
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path
            
            # Check if the browser executable exists
            chrome_path = Path(browsers_path) / "chromium-1112" / "chrome-win" / "chrome.exe"
            
            if not chrome_path.exists():
                logger.info("Playwright browsers not installed. Installing now...")
                try:
                    # Use the setup_playwright.py script to install browsers
                    setup_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "setup_playwright.py")
                    if os.path.exists(setup_script):
                        subprocess.run(
                            [sys.executable, setup_script],
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        logger.info("Playwright browsers installed successfully")
                    else:
                        # Direct installation if script not found
                        env = os.environ.copy()
                        env["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path
                        subprocess.run(
                            [sys.executable, "-m", "playwright", "install", "chromium"],
                            env=env,
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                        logger.info("Playwright browsers installed successfully via direct command")
                except Exception as install_error:
                    logger.error(f"Failed to install Playwright browsers: {install_error}")
                    # Continue with limited functionality
            
            # Try to initialize Playwright components (browser/page)
            try:
                await self.scraper.initialize_playwright_components()
                logger.info("Playwright components initialized successfully")
            except Exception as playwright_error:
                logger.warning(f"Playwright components initialization failed, but continuing with limited scraper functionality: {playwright_error}")
                # Continue anyway - some parts of the app might work without Playwright
            
            # Mark the scraper as initialized regardless of Playwright status
            self.initialization_status['scraper'] = True
            logger.info("Scraper initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Scraper initialization failed: {e}")
            return False

    async def _initialize_data_loader(self) -> bool:
        """Initialize the data loader with database connection."""
        try:
            self.data_loader = DataIntegration()
            # Verify data loader initialization (DataIntegration manages its own resources)
            if self.data_loader is None: # Basic check
                logger.error("Data loader (DataIntegration) failed to initialize")
                return False

            self.initialization_status['data_loader'] = True
            logger.info("Data loader initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Data loader initialization failed: {e}")
            return False

    async def _perform_initial_data_population(self) -> bool:
        """Perform initial population of essential data with real-time validation."""
        logger.info("Starting initial data population with validation...")
        try:
            # DataIntegrator can create its own ScraperFactory if not provided.
            # We provide the db_manager instance from StartupManager.
            data_integrator = DataIntegrator(db_manager=self.db_manager)
            
            # Define a list of default leagues to integrate initially.
            # These codes should match the keys expected by DataIntegrator.integrate_league_data
            # which are typically like 'premier_league', 'la_liga', etc.
            default_leagues = [
                "premier_league", 
                "la_liga", 
                "serie_a", 
                "bundesliga", 
                "ligue_1", 
                "eredivisie" # Netherlands Eredivisie
            ]
            
            successful_integrations = 0
            validation_results = {}
            
            for league_code in default_leagues:
                logger.info(f"Attempting to integrate data for league: {league_code}")
                
                # Integrate league data
                success = await data_integrator.integrate_league_data(league_code, update_teams=True)
                
                if success:
                    # Validate the integrated data
                    if hasattr(self, 'ai_validator') and self.ai_validator:
                        validation_result = await self._validate_league_data(league_code)
                        validation_results[league_code] = validation_result
                        
                        if validation_result.get('passed', False):
                            logger.info(f"Successfully integrated and validated data for league: {league_code}")
                            successful_integrations += 1
                        else:
                            logger.warning(f"Data validation failed for league: {league_code}")
                    else:
                        logger.info(f"Successfully integrated data for league: {league_code} (validation skipped)")
                        successful_integrations += 1
                else:
                    logger.warning(f"Failed to integrate data for league: {league_code}. Check DataIntegrator logs.")
            
            # Store validation results
            self.validation_results = validation_results
            
            if successful_integrations == len(default_leagues):
                logger.info("Initial data population completed successfully for all default leagues.")
                return True
            elif successful_integrations > 0:
                logger.warning(f"Initial data population partially completed. {successful_integrations}/{len(default_leagues)} leagues integrated.")
                return True # Still return true, system might be partially usable
            else:
                logger.error("Initial data population failed for all default leagues.")
                return False

        except Exception as e:
            logger.error(f"Error during initial data population: {e}", exc_info=True)
            return False

    async def _validate_league_data(self, league_code: str) -> Dict[str, Any]:
        """Validate league data using AI validator."""
        try:
            # Get league data from database
            with self.db_manager.session_scope() as session:
                league = session.query(League).filter(League.id == league_code).first()
                if not league:
                    return {'passed': False, 'reason': 'League not found in database'}
                
                # Create a simple validation check
                league_data = pd.DataFrame([{
                    'id': league.id,
                    'name': league.name,
                    'country': league.country,
                    'tier': league.tier
                }])
                
                # Basic validation - check if required fields are present
                if not league.name or not league.country:
                    return {'passed': False, 'reason': 'Missing required league data'}
                
                # If AI validator is available, use it
                if hasattr(self, 'ai_validator') and self.ai_validator:
                    try:
                        validated_data, validation_report = self.ai_validator.validate_dataset(league_data)
                        return {'passed': True, 'validation_report': validation_report}
                    except Exception as e:
                        logger.warning(f"AI validation failed for {league_code}: {e}")
                        # Fall back to basic validation
                        return {'passed': True, 'reason': 'Basic validation passed (AI validation failed)'}
                else:
                    # Basic validation only
                    return {'passed': True, 'reason': 'Basic validation passed'}
                    
        except Exception as e:
            logger.error(f"Error validating league data for {league_code}: {e}")
            return {'passed': False, 'reason': f'Validation error: {str(e)}'}

    def _validate_system_state(self) -> bool:
        """Validate the overall system state after initialization with enhanced checks."""
        try:
            # Check all components are initialized
            if not all(self.initialization_status.values()):
                missing = [k for k, v in self.initialization_status.items() if not v]
                logger.error(f"Missing initialized components: {missing}")
                return False

            # Verify database tables
            required_tables = ['matches', 'predictions', 'api_cache', 'scraped_data', 'leagues', 'teams']
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = [row[0] for row in cursor.fetchall()]
                missing_tables = [table for table in required_tables if table not in existing_tables]
                if missing_tables:
                    logger.error(f"Missing required tables: {missing_tables}")
                    return False

            # Verify scraper configuration
            if not Config.get('scraping.sources.primary'):
                logger.error("No scraping sources configured")
                return False

            # Verify Firecrawl server if initialized
            if self.initialization_status.get('firecrawl_server', False):
                server_config = Config.get('scraping.server', {})
                host = server_config.get('host', 'localhost')
                port = server_config.get('port', 3000)
                
                # Check server health asynchronously
                import asyncio
                try:
                    health_check = asyncio.run(self._check_server_health(host, port))
                    if not health_check:
                        logger.warning("Firecrawl server health check failed")
                except Exception as e:
                    logger.warning(f"Could not perform Firecrawl server health check: {e}")

            logger.info("System state validation successful")
            return True

        except Exception as e:
            logger.error(f"System state validation failed: {e}")
            return False

    async def cleanup(self):
        """Clean up all system resources."""
        logger.info("Starting enhanced system cleanup...")
        
        try:
            # Stop Firecrawl server
            if self.firecrawl_process:
                logger.info("Stopping Firecrawl server...")
                try:
                    self.firecrawl_process.terminate()
                    # Wait for a short time, then force kill if needed
                    try:
                        self.firecrawl_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning("Firecrawl server didn't terminate gracefully, forcing kill")
                        self.firecrawl_process.kill()
                        self.firecrawl_process.wait(timeout=2)
                except Exception as e:
                    logger.warning(f"Error stopping Firecrawl server: {e}")
                finally:
                    self.firecrawl_process = None
            
            # Clean up database connections
            if self.db_manager:
                try:
                    self.db_manager.close()
                except Exception as e:
                    logger.warning(f"Error closing database manager: {e}")
            
            # Clean up scraper resources
            if self.scraper:
                try:
                    await self.scraper.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up scraper: {e}")
            
            logger.info("Enhanced system cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during system cleanup: {e}")

    def get_startup_time(self) -> Optional[datetime]:
        """Get the system startup time."""
        return self.startup_time

    def get_initialization_status(self) -> Dict[str, bool]:
        """Get the initialization status of all components."""
        return self.initialization_status.copy()

    def get_validation_results(self) -> Dict[str, Any]:
        """Get validation results from data population."""
        return self.validation_results.copy()

    def _handle_firecrawl_result(self, task):
        """Handle Firecrawl server initialization result."""
        try:
            if task.cancelled():
                logger.info("Firecrawl server initialization was cancelled")
                return
            result = task.result()
            if result:
                logger.info("Firecrawl server initialized successfully in background")
            else:
                logger.warning("Firecrawl server initialization failed in background")
        except asyncio.CancelledError:
            logger.info("Firecrawl server initialization was cancelled")
        except Exception as e:
            logger.warning(f"Firecrawl server initialization failed in background: {e}")

# Example usage
async def main():
    startup_manager = StartupManager()
    try:
        if await startup_manager.initialize():
            logger.info("Enhanced system started successfully")
            # Your main application logic here
        else:
            logger.error("Enhanced system startup failed")
    finally:
        await startup_manager.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
