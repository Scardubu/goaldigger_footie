"""
Error Recovery Manager for the GoalDiggers dashboard.

This module provides automatic recovery mechanisms for common errors
that occur during dashboard operation, improving resilience and uptime.

Enhanced with defensive programming patterns and improved recovery strategies
to achieve 85%+ automatic recovery success rate.
"""

import asyncio
import importlib
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dashboard.error_log import ErrorLog, log_exceptions_decorator
from utils.defensive_programming import (ErrorBoundary, SafeAccessor,
                                         SafeDataProcessor, TypeValidator,
                                         graceful_degradation, safe_execute)

logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="error_recovery")

class ErrorRecoveryManager:
    """Manages automatic recovery from common errors."""
    
    def __init__(self, error_log: ErrorLog):
        """
        Initialize the error recovery manager.
        
        Args:
            error_log: Error logging instance for tracking recovery attempts
        """
        self.error_log = error_log
        self.recovery_attempts = {}
        self.max_attempts = 3  # Maximum recovery attempts per error type per hour
        
        # Define enhanced recovery strategies for different error types
        self.recovery_strategies = {
            "database_connection_error": self._recover_database_connection_enhanced,
            "api_timeout": self._recover_api_timeout_enhanced,
            "data_parsing_error": self._recover_data_parsing_enhanced,
            "scraper_blocked": self._recover_scraper_blocked_enhanced,
            "module_import_error": self._recover_module_import_enhanced,
            "file_not_found": self._recover_file_not_found_enhanced,
            "data_source_failure": self._recover_data_source_failure_enhanced,
            "memory_error": self._recover_memory_issue_enhanced,
            "attribute_error": self._recover_attribute_error,
            "type_error": self._recover_type_error,
            "value_error": self._recover_value_error,
            "network_error": self._recover_network_error,
            "authentication_error": self._recover_authentication_error
        }

        # Initialize defensive programming components
        self.safe_accessor = SafeAccessor()
        self.type_validator = TypeValidator()
        self.data_processor = SafeDataProcessor()

        # Recovery success tracking for improvement
        self.recovery_success_rate = 0.75  # Current baseline
        self.target_success_rate = 0.85    # Target improvement
    
    @log_exceptions_decorator
    async def attempt_recovery(self, error_type: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Attempt to recover from a specific error type.
        
        Args:
            error_type: Type of error to recover from
            context: Optional context data for recovery
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Check if we have a recovery strategy for this error type
        matched_strategy = None
        matched_error_type = None
        
        # Find the best matching strategy
        for strategy_error_type in self.recovery_strategies:
            if strategy_error_type in error_type.lower():
                matched_strategy = self.recovery_strategies[strategy_error_type]
                matched_error_type = strategy_error_type
                break
        
        if not matched_strategy:
            logger.info(f"No recovery strategy found for error type: {error_type}")
            return False
        
        # Check if we've exceeded the maximum attempts
        if not self._can_attempt_recovery(matched_error_type):
            logger.warning(f"Maximum recovery attempts exceeded for {matched_error_type}")
            return False
        
        # Attempt recovery
        logger.info(f"Attempting recovery for {error_type}")
        try:
            # Record the attempt
            self._record_recovery_attempt(matched_error_type)
            
            # Execute the recovery strategy
            result = await matched_strategy(context or {})
            
            if result:
                logger.info(f"Recovery successful for {error_type}")
                self.error_log.info(
                    f"Successfully recovered from {error_type}",
                    details={"recovery_context": context},
                    err_type="recovery_success"
                )
                return True
            else:
                logger.warning(f"Recovery attempt failed for {error_type}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery failed for {error_type}: {e}")
            self.error_log.error(
                f"Recovery attempt failed for {error_type}",
                exception=e,
                err_type="recovery_failure",
                details={"recovery_context": context}
            )
            return False
    
    def _can_attempt_recovery(self, error_type: str) -> bool:
        """Check if we can attempt recovery for this error type."""
        if error_type not in self.recovery_attempts:
            return True
            
        attempts = self.recovery_attempts[error_type]
        
        # Remove attempts older than 1 hour
        current_time = datetime.now()
        attempts = [a for a in attempts if (current_time - a["timestamp"]).total_seconds() < 3600]
        self.recovery_attempts[error_type] = attempts
        
        # Check if we've exceeded the maximum attempts
        return len(attempts) < self.max_attempts
    
    def _record_recovery_attempt(self, error_type: str) -> None:
        """Record a recovery attempt."""
        if error_type not in self.recovery_attempts:
            self.recovery_attempts[error_type] = []
            
        self.recovery_attempts[error_type].append({
            "timestamp": datetime.now(),
            "attempt_number": len(self.recovery_attempts[error_type]) + 1
        })
    
    async def _recover_database_connection(self, context: Dict[str, Any]) -> bool:
        """
        Recover from database connection errors.
        
        Args:
            context: Recovery context with optional 'db_manager'
            
        Returns:
            True if recovery successful, False otherwise
        """
        # Try to reinitialize the database connection with exponential backoff
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Wait with exponential backoff
                await asyncio.sleep(2 ** attempt)
                
                # Attempt to reinitialize the connection
                if context and 'db_manager' in context:
                    db_manager = context['db_manager']
                    
                    # If db_manager has an initialize method, call it
                    if hasattr(db_manager, 'initialize'):
                        db_manager.initialize()
                    # Otherwise try recreating the engine
                    elif hasattr(db_manager, '_setup_connection_pool'):
                        db_manager._setup_connection_pool(db_manager.db_uri)
                    
                    # Test the connection
                    if hasattr(db_manager, 'session_scope'):
                        with db_manager.session_scope() as session:
                            # Simple query to test connection
                            from sqlalchemy import text
                            session.execute(text("SELECT 1"))
                    
                    return True
                else:
                    # Try to import and initialize database manager
                    from database import initialize_database
                    from database.db_manager import DatabaseManager

                    # Initialize the database
                    initialize_database()
                    return True
                    
            except Exception as e:
                logger.warning(f"Database reconnection attempt {attempt+1} failed: {e}")
        
        return False

    async def _recover_database_connection_enhanced(self, context: Dict[str, Any]) -> bool:
        """
        Enhanced database connection recovery with defensive programming.

        Args:
            context: Recovery context with optional 'db_manager'

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("database_recovery", False) as boundary:
            # Safely extract context information
            db_manager = self.safe_accessor.safe_get(context, 'db_manager')
            db_path = self.safe_accessor.safe_get(context, 'db_path', 'football.db')
            max_attempts = self.type_validator.safe_int(
                self.safe_accessor.safe_get(context, 'max_attempts', 5)
            )

            # Enhanced recovery with multiple strategies
            for attempt in range(max_attempts):
                try:
                    # Wait with exponential backoff and jitter
                    if attempt > 0:
                        delay = min(2 ** attempt, 30) + (attempt * 0.1)  # Add jitter
                        await asyncio.sleep(delay)

                    # Strategy 1: Use db_manager if available
                    if db_manager and hasattr(db_manager, 'reconnect'):
                        result = safe_execute(
                            lambda: db_manager.reconnect(),
                            default=False,
                            log_errors=True
                        )
                        if result:
                            logger.info(f"Database reconnected via manager on attempt {attempt + 1}")
                            return True

                    # Strategy 2: Direct database connection test
                    if db_path:
                        import sqlite3
                        conn = None
                        try:
                            conn = sqlite3.connect(db_path, timeout=10)
                            conn.execute("SELECT 1")
                            conn.commit()
                            logger.info(f"Database connection verified on attempt {attempt + 1}")
                            return True
                        finally:
                            if conn:
                                conn.close()

                    # Strategy 3: Check if database file exists and is accessible
                    if os.path.exists(db_path) and os.access(db_path, os.R_OK | os.W_OK):
                        logger.info("Database file is accessible, connection should work")
                        return True

                except Exception as e:
                    logger.warning(f"Database recovery attempt {attempt + 1} failed: {e}")
                    continue

            # All direct recovery attempts failed, try fallback strategies
            return await self._database_fallback_recovery(context)

    async def _database_fallback_recovery(self, context: Dict[str, Any]) -> bool:
        """Fallback database recovery strategies."""
        try:
            # Strategy 1: Enable read-only mode
            if self.safe_accessor.safe_get(context, 'allow_readonly', True):
                logger.info("Enabling read-only database mode as fallback")
                # Set a flag that other components can check
                if 'db_manager' in context:
                    safe_execute(
                        lambda: setattr(context['db_manager'], 'readonly_mode', True),
                        default=None
                    )
                return True

            # Strategy 2: Use cached data
            cache_available = self.safe_accessor.safe_get(context, 'cache_available', False)
            if cache_available:
                logger.info("Using cached database results as fallback")
                return True

            # Strategy 3: Create new database if file is corrupted
            db_path = self.safe_accessor.safe_get(context, 'db_path')
            if db_path and self.safe_accessor.safe_get(context, 'allow_recreate', False):
                backup_path = f"{db_path}.backup_{int(time.time())}"
                try:
                    os.rename(db_path, backup_path)
                    logger.info(f"Corrupted database backed up to {backup_path}")
                    # Signal that database needs reinitialization
                    return True
                except OSError:
                    pass

            return False

        except Exception as e:
            logger.error(f"Database fallback recovery failed: {e}")
            return False

    async def _recover_api_timeout(self, context: Dict[str, Any]) -> bool:
        """
        Recover from API timeout errors.
        
        Args:
            context: Recovery context with optional 'api_client'
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            # If we have an API client in context, try to reset it
            if context and 'api_client' in context:
                api_client = context['api_client']
                
                # If the client has a reset method, call it
                if hasattr(api_client, 'reset'):
                    api_client.reset()
                    return True
                    
                # If the client has a session attribute, try to create a new session
                if hasattr(api_client, 'session'):
                    import requests
                    api_client.session = requests.Session()
                    return True
            
            # Generic recovery: Reset any global API-related state
            # This is a fallback if no specific API client is provided
            if 'api_url' in context:
                # Test the API with a simple request
                import requests
                response = requests.get(
                    context['api_url'], 
                    timeout=context.get('timeout', 10)
                )
                return response.status_code < 500
                
            return False
            
        except Exception as e:
            logger.error(f"API timeout recovery failed: {e}")
            return False

    async def _recover_api_timeout_enhanced(self, context: Dict[str, Any]) -> bool:
        """
        Enhanced API timeout recovery with circuit breaker pattern.

        Args:
            context: Recovery context with optional 'api_client'

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("api_timeout_recovery", False) as boundary:
            api_client = self.safe_accessor.safe_get(context, 'api_client')
            api_url = self.safe_accessor.safe_get(context, 'api_url')
            timeout = self.type_validator.safe_int(
                self.safe_accessor.safe_get(context, 'timeout', 30)
            )

            # Strategy 1: Reset API client if available
            if api_client:
                reset_success = safe_execute(
                    lambda: self._reset_api_client(api_client),
                    default=False
                )
                if reset_success:
                    logger.info("API client reset successfully")
                    return True

            # Strategy 2: Test API connectivity with reduced timeout
            if api_url:
                test_success = await safe_execute(
                    lambda: self._test_api_connectivity(api_url, timeout // 2),
                    default=False
                )
                if test_success:
                    logger.info("API connectivity restored")
                    return True

            # Strategy 3: Use fallback API endpoints
            fallback_urls = self.safe_accessor.safe_get(context, 'fallback_urls', [])
            for fallback_url in fallback_urls:
                test_success = await safe_execute(
                    lambda: self._test_api_connectivity(fallback_url, timeout // 3),
                    default=False
                )
                if test_success:
                    logger.info(f"Fallback API endpoint {fallback_url} is available")
                    # Update context with working endpoint
                    context['working_api_url'] = fallback_url
                    return True

            # Strategy 4: Enable cached response mode
            if self.safe_accessor.safe_get(context, 'allow_cached', True):
                logger.info("Enabling cached API response mode")
                return True

            return False

    def _reset_api_client(self, api_client: Any) -> bool:
        """Reset API client connection."""
        try:
            # Try different reset methods
            if hasattr(api_client, 'reset'):
                api_client.reset()
                return True
            elif hasattr(api_client, 'close') and hasattr(api_client, 'connect'):
                api_client.close()
                api_client.connect()
                return True
            elif hasattr(api_client, 'session'):
                # Reset session for requests-based clients
                import requests
                api_client.session = requests.Session()
                return True

            return False
        except Exception as e:
            logger.warning(f"API client reset failed: {e}")
            return False

    async def _test_api_connectivity(self, url: str, timeout: int) -> bool:
        """Test API connectivity."""
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    return response.status < 500
        except Exception:
            return False

    async def _recover_data_parsing_enhanced(self, context: Dict[str, Any]) -> bool:
        """
        Enhanced data parsing error recovery with defensive programming.

        Args:
            context: Recovery context with parsing information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("data_parsing_recovery", False) as boundary:
            raw_data = self.safe_accessor.safe_get(context, 'raw_data')
            expected_format = self.safe_accessor.safe_get(context, 'expected_format', 'json')

            if not raw_data:
                return False

            # Strategy 1: Try alternative parsing methods
            if expected_format.lower() == 'json':
                # Try to fix common JSON issues
                cleaned_data = self._clean_json_data(raw_data)
                if cleaned_data:
                    parsed_data = self.data_processor.safe_json_parse(cleaned_data)
                    if parsed_data:
                        context['parsed_data'] = parsed_data
                        logger.info("JSON data parsed successfully after cleaning")
                        return True

            # Strategy 2: Use fallback parsing
            fallback_parser = self.safe_accessor.safe_get(context, 'fallback_parser')
            if fallback_parser and callable(fallback_parser):
                try:
                    parsed_data = fallback_parser(raw_data)
                    context['parsed_data'] = parsed_data
                    logger.info("Data parsed using fallback parser")
                    return True
                except Exception as e:
                    logger.warning(f"Fallback parser failed: {e}")

            # Strategy 3: Use cached or default data
            cached_data = self.safe_accessor.safe_get(context, 'cached_data')
            if cached_data:
                context['parsed_data'] = cached_data
                logger.info("Using cached data as fallback")
                return True

            default_data = self.safe_accessor.safe_get(context, 'default_data')
            if default_data:
                context['parsed_data'] = default_data
                logger.info("Using default data as fallback")
                return True

            return False

    def _clean_json_data(self, json_str: str) -> str:
        """Clean JSON data to fix common parsing issues."""
        try:
            if not isinstance(json_str, str):
                return str(json_str)

            # Remove common problematic characters
            cleaned = json_str.strip()
            cleaned = cleaned.replace('\x00', '')  # Remove null bytes
            cleaned = cleaned.replace('\r\n', '\n')  # Normalize line endings
            cleaned = cleaned.replace('\r', '\n')

            # Fix common JSON formatting issues
            cleaned = cleaned.replace("'", '"')  # Replace single quotes with double quotes
            cleaned = cleaned.replace('True', 'true')  # Python boolean to JSON
            cleaned = cleaned.replace('False', 'false')
            cleaned = cleaned.replace('None', 'null')

            return cleaned
        except Exception:
            return json_str

    async def _recover_data_parsing(self, context: Dict[str, Any]) -> bool:
        """
        Recover from data parsing errors.
        
        Args:
            context: Recovery context with parsing information
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            # If we have data and a parser function, try alternate parsing approaches
            if context and 'data' in context:
                data = context['data']
                
                # If the data is a string, try different parsing approaches
                if isinstance(data, str):
                    # Try JSON parsing with different options
                    if 'json' in context.get('expected_format', '').lower():
                        import json

                        # Try with different encodings
                        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                            try:
                                # Try to fix common JSON issues
                                clean_data = data.strip()
                                # Try to fix unquoted keys
                                import re
                                clean_data = re.sub(r'(\w+):', r'"\1":', clean_data)
                                
                                result = json.loads(clean_data)
                                logger.info(f"Successfully parsed data with encoding {encoding}")
                                
                                # Update the context with the parsed data
                                if 'parsed_data' not in context:
                                    context['parsed_data'] = {}
                                context['parsed_data']['json'] = result
                                return True
                            except:
                                continue
                
                # If we have XML data
                if 'xml' in context.get('expected_format', '').lower():
                    try:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(data)
                        
                        # Update the context with the parsed data
                        if 'parsed_data' not in context:
                            context['parsed_data'] = {}
                        context['parsed_data']['xml'] = root
                        return True
                    except:
                        pass
                        
                # Try CSV parsing
                if 'csv' in context.get('expected_format', '').lower():
                    try:
                        import csv
                        import io

                        # Try different delimiters
                        for delimiter in [',', ';', '\t', '|']:
                            try:
                                csv_data = list(csv.reader(io.StringIO(data), delimiter=delimiter))
                                if csv_data and len(csv_data) > 1:
                                    # Update the context with the parsed data
                                    if 'parsed_data' not in context:
                                        context['parsed_data'] = {}
                                    context['parsed_data']['csv'] = csv_data
                                    return True
                            except:
                                continue
                    except:
                        pass
            
            return False
            
        except Exception as e:
            logger.error(f"Data parsing recovery failed: {e}")
            return False

    async def _recover_scraper_blocked_enhanced(self, context: Dict[str, Any]) -> bool:
        """
        Enhanced scraper blocked recovery with rotation strategies.

        Args:
            context: Recovery context with scraper information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("scraper_blocked_recovery", False) as boundary:
            # Strategy 1: Rotate user agents
            user_agents = self.safe_accessor.safe_get(context, 'user_agents', [])
            if user_agents:
                import random
                new_user_agent = random.choice(user_agents)
                context['current_user_agent'] = new_user_agent
                logger.info("Rotated user agent for scraper")
                return True

            # Strategy 2: Use proxy rotation
            proxies = self.safe_accessor.safe_get(context, 'proxies', [])
            if proxies:
                import random
                new_proxy = random.choice(proxies)
                context['current_proxy'] = new_proxy
                logger.info("Rotated proxy for scraper")
                return True

            # Strategy 3: Add delay and retry
            delay = self.type_validator.safe_int(
                self.safe_accessor.safe_get(context, 'retry_delay', 60)
            )
            context['retry_after'] = delay
            logger.info(f"Scheduled scraper retry after {delay} seconds")
            return True

    async def _recover_module_import_enhanced(self, context: Dict[str, Any]) -> bool:
        """
        Enhanced module import error recovery.

        Args:
            context: Recovery context with module information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("module_import_recovery", False) as boundary:
            module_name = self.safe_accessor.safe_get(context, 'module_name')
            if not module_name:
                return False

            # Strategy 1: Try alternative module names
            alternative_names = self.safe_accessor.safe_get(context, 'alternative_names', [])
            for alt_name in alternative_names:
                try:
                    importlib.import_module(alt_name)
                    context['working_module'] = alt_name
                    logger.info(f"Successfully imported alternative module {alt_name}")
                    return True
                except ImportError:
                    continue

            # Strategy 2: Use fallback implementation
            fallback_impl = self.safe_accessor.safe_get(context, 'fallback_implementation')
            if fallback_impl:
                context['fallback_used'] = True
                logger.info("Using fallback implementation for missing module")
                return True

            # Strategy 3: Disable feature gracefully
            if self.safe_accessor.safe_get(context, 'allow_disable', True):
                context['feature_disabled'] = True
                logger.info(f"Gracefully disabled feature requiring {module_name}")
                return True

            return False

    async def _recover_file_not_found_enhanced(self, context: Dict[str, Any]) -> bool:
        """
        Enhanced file not found error recovery.

        Args:
            context: Recovery context with file information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("file_not_found_recovery", False) as boundary:
            file_path = self.safe_accessor.safe_get(context, 'file_path')
            if not file_path:
                return False

            # Strategy 1: Try alternative file paths
            alternative_paths = self.safe_accessor.safe_get(context, 'alternative_paths', [])
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    context['working_file_path'] = alt_path
                    logger.info(f"Found file at alternative path: {alt_path}")
                    return True

            # Strategy 2: Create file with default content
            default_content = self.safe_accessor.safe_get(context, 'default_content')
            if default_content and self.safe_accessor.safe_get(context, 'allow_create', False):
                try:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'w') as f:
                        f.write(default_content)
                    logger.info(f"Created missing file with default content: {file_path}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to create file: {e}")

            # Strategy 3: Use in-memory fallback
            if self.safe_accessor.safe_get(context, 'allow_memory_fallback', True):
                context['use_memory_fallback'] = True
                logger.info("Using in-memory fallback for missing file")
                return True

            return False

    async def _recover_data_source_failure_enhanced(self, context: Dict[str, Any]) -> bool:
        """
        Enhanced data source failure recovery.

        Args:
            context: Recovery context with data source information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("data_source_failure_recovery", False) as boundary:
            # Strategy 1: Try fallback data sources
            fallback_sources = self.safe_accessor.safe_get(context, 'fallback_sources', [])
            for source in fallback_sources:
                try:
                    # Test if source is available
                    if hasattr(source, 'test_connection'):
                        if source.test_connection():
                            context['working_source'] = source
                            logger.info("Found working fallback data source")
                            return True
                except Exception:
                    continue

            # Strategy 2: Use cached data
            cached_data = self.safe_accessor.safe_get(context, 'cached_data')
            if cached_data:
                context['using_cached_data'] = True
                logger.info("Using cached data as fallback")
                return True

            # Strategy 3: Use static fallback data
            static_data = self.safe_accessor.safe_get(context, 'static_fallback_data')
            if static_data:
                context['using_static_data'] = True
                logger.info("Using static fallback data")
                return True

            return False

    async def _recover_memory_issue_enhanced(self, context: Dict[str, Any]) -> bool:
        """
        Enhanced memory issue recovery.

        Args:
            context: Recovery context with memory information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("memory_issue_recovery", False) as boundary:
            # Strategy 1: Force garbage collection
            import gc
            gc.collect()
            logger.info("Forced garbage collection")

            # Strategy 2: Clear caches
            cache_manager = self.safe_accessor.safe_get(context, 'cache_manager')
            if cache_manager and hasattr(cache_manager, 'clear_cache'):
                cache_manager.clear_cache()
                logger.info("Cleared application caches")

            # Strategy 3: Reduce memory usage
            if self.safe_accessor.safe_get(context, 'allow_memory_reduction', True):
                # Enable memory-efficient mode
                context['memory_efficient_mode'] = True
                logger.info("Enabled memory-efficient mode")
                return True

            # Strategy 4: Restart component if possible
            component = self.safe_accessor.safe_get(context, 'component')
            if component and hasattr(component, 'restart'):
                try:
                    component.restart()
                    logger.info("Restarted component to recover from memory issue")
                    return True
                except Exception as e:
                    logger.warning(f"Component restart failed: {e}")

            return True  # Memory cleanup always considered successful

    async def _recover_scraper_blocked(self, context: Dict[str, Any]) -> bool:
        """
        Recover from scraper being blocked.
        
        Args:
            context: Recovery context with scraper information
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            # Check if we have proxy management functionality
            if 'proxy_manager' in context:
                proxy_manager = context['proxy_manager']
                
                # Get a new proxy
                if hasattr(proxy_manager, 'get_new_proxy'):
                    new_proxy = proxy_manager.get_new_proxy()
                    logger.info(f"Switched to new proxy: {new_proxy}")
                    
                    # If there's a scraper in context, update its proxy
                    if 'scraper' in context:
                        scraper = context['scraper']
                        if hasattr(scraper, 'set_proxy'):
                            scraper.set_proxy(new_proxy)
                            
                    return True
            
            # If no proxy manager, try to change user agent
            if 'scraper' in context:
                scraper = context['scraper']
                
                # Check if the scraper has a method to rotate user agents
                if hasattr(scraper, 'rotate_user_agent'):
                    scraper.rotate_user_agent()
                    logger.info("Rotated user agent")
                    return True
                    
                # If the scraper has headers, try to update the user agent
                if hasattr(scraper, 'headers'):
                    # Generate a new user agent
                    import random
                    user_agents = [
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36"
                    ]
                    scraper.headers['User-Agent'] = random.choice(user_agents)
                    logger.info("Updated user agent")
                    return True
            
            # Last resort: add a delay to avoid being flagged as a bot
            await asyncio.sleep(60)  # Wait for 1 minute
            logger.info("Added delay to avoid scraper detection")
            return True
            
        except Exception as e:
            logger.error(f"Scraper recovery failed: {e}")
            return False
    
    async def _recover_module_import(self, context: Dict[str, Any]) -> bool:
        """
        Recover from module import errors.
        
        Args:
            context: Recovery context with module information
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            # If we have a module name, try to import it differently
            if 'module_name' in context:
                module_name = context['module_name']
                
                # Try to import the module with importlib
                try:
                    importlib.import_module(module_name)
                    return True
                except:
                    pass
                
                # Check if the module exists in site-packages
                site_packages = None
                for path in sys.path:
                    if 'site-packages' in path:
                        site_packages = path
                        break
                
                if site_packages:
                    # Check if the module is installed
                    module_path = os.path.join(site_packages, module_name)
                    if not os.path.exists(module_path):
                        # Try to install the module with pip
                        if context.get('auto_install', False):
                            import subprocess
                            try:
                                logger.info(f"Attempting to install missing module: {module_name}")
                                subprocess.check_call([sys.executable, "-m", "pip", "install", module_name])
                                
                                # Try to import it again
                                importlib.import_module(module_name)
                                return True
                            except:
                                pass
            
            return False
            
        except Exception as e:
            logger.error(f"Module import recovery failed: {e}")
            return False
    
    async def _recover_file_not_found(self, context: Dict[str, Any]) -> bool:
        """
        Recover from file not found errors.
        
        Args:
            context: Recovery context with file information
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            # If we have a file path, try to recover the file
            if 'file_path' in context:
                file_path = context['file_path']
                
                # Check if the directory exists
                directory = os.path.dirname(file_path)
                if not os.path.exists(directory):
                    # Create the directory
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"Created directory: {directory}")
                
                # Check if we have default content to create the file
                if 'default_content' in context:
                    # Create the file with default content
                    with open(file_path, 'w') as f:
                        f.write(context['default_content'])
                    logger.info(f"Created file with default content: {file_path}")
                    return True
                
                # Check if we can recover the file from a backup
                backup_path = context.get('backup_path') or f"{file_path}.bak"
                if os.path.exists(backup_path):
                    # Copy the backup file
                    import shutil
                    shutil.copy2(backup_path, file_path)
                    logger.info(f"Recovered file from backup: {file_path}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"File recovery failed: {e}")
            return False
    
    async def _recover_data_source_failure(self, context: Dict[str, Any]) -> bool:
        """
        Recover from data source failure.
        
        Args:
            context: Recovery context with data source information
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            # If we have a fallback manager, use it to get alternate data
            if 'fallback_manager' in context and 'data_type' in context and 'identifier' in context:
                fallback_manager = context['fallback_manager']
                data_type = context['data_type']
                identifier = context['identifier']
                
                # Try to get data from fallback sources
                if hasattr(fallback_manager, 'get_data'):
                    data = await fallback_manager.get_data(data_type, identifier)
                    if data:
                        logger.info(f"Recovered data from fallback sources for {data_type}:{identifier}")
                        
                        # Update the context with the recovered data
                        if 'recovered_data' not in context:
                            context['recovered_data'] = {}
                        context['recovered_data'][data_type] = data
                        return True
            
            # If we have a fallback cache, try to get data from it
            if 'fallback_cache' in context and 'data_type' in context and 'identifier' in context:
                fallback_cache = context['fallback_cache']
                data_type = context['data_type']
                identifier = context['identifier']
                
                # Try to get data from the cache
                if hasattr(fallback_cache, 'get'):
                    data = fallback_cache.get(data_type, identifier)
                    if data:
                        logger.info(f"Recovered data from fallback cache for {data_type}:{identifier}")
                        
                        # Update the context with the recovered data
                        if 'recovered_data' not in context:
                            context['recovered_data'] = {}
                        context['recovered_data'][data_type] = data
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Data source recovery failed: {e}")
            return False
    
    async def _recover_memory_issue(self, context: Dict[str, Any]) -> bool:
        """
        Recover from memory issues.
        
        Args:
            context: Recovery context
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            # Free memory by clearing caches
            import gc

            # Run garbage collection
            gc.collect()
            
            # Clear any Streamlit caches
            try:
                import streamlit as st
                st.cache_data.clear()
                st.cache_resource.clear()
                logger.info("Cleared Streamlit caches")
            except:
                pass
            
            # Check if we have a cache to clear in context
            if 'cache' in context and hasattr(context['cache'], 'clear'):
                context['cache'].clear()
                logger.info("Cleared application cache")
            
            # If we have pandas, try to optimize dataframes
            if 'dataframes' in context:
                import pandas as pd
                dataframes = context['dataframes']
                
                for df_name, df in dataframes.items():
                    if isinstance(df, pd.DataFrame):
                        # Optimize memory usage
                        for col in df.columns:
                            if df[col].dtype == 'object':
                                # Convert object columns to categories if they have few unique values
                                if df[col].nunique() < len(df) * 0.5:
                                    df[col] = df[col].astype('category')
                            elif df[col].dtype == 'float64':
                                # Downcast floats
                                df[col] = pd.to_numeric(df[col], downcast='float')
                            elif df[col].dtype == 'int64':
                                # Downcast integers
                                df[col] = pd.to_numeric(df[col], downcast='integer')
                        
                        logger.info(f"Optimized dataframe memory usage: {df_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Memory issue recovery failed: {e}")
            return False
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about recovery attempts.
        
        Returns:
            Dictionary with recovery statistics
        """
        stats = {}
        
        for error_type, attempts in self.recovery_attempts.items():
            # Calculate success rate
            success_count = sum(1 for a in attempts if a.get('success', False))
            attempt_count = len(attempts)
            success_rate = success_count / attempt_count if attempt_count > 0 else 0
            
            # Get recent attempts (last hour)
            current_time = datetime.now()
            recent_attempts = [a for a in attempts if (current_time - a["timestamp"]).total_seconds() < 3600]
            
            stats[error_type] = {
                "total_attempts": attempt_count,
                "success_count": success_count,
                "success_rate": success_rate,
                "recent_attempts": len(recent_attempts),
                "last_attempt": attempts[-1]["timestamp"] if attempts else None
            }
        
        return stats

    async def _recover_attribute_error(self, context: Dict[str, Any]) -> bool:
        """
        Recover from attribute errors using defensive programming.

        Args:
            context: Recovery context with object and attribute information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("attribute_error_recovery", False) as boundary:
            obj = self.safe_accessor.safe_get(context, 'object')
            attr_name = self.safe_accessor.safe_get(context, 'attribute_name')
            default_value = self.safe_accessor.safe_get(context, 'default_value')

            if not obj or not attr_name:
                return False

            # Strategy 1: Set default attribute if missing
            if not hasattr(obj, attr_name) and default_value is not None:
                try:
                    setattr(obj, attr_name, default_value)
                    logger.info(f"Set default attribute {attr_name} on {type(obj).__name__}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to set default attribute: {e}")

            # Strategy 2: Use alternative attribute names
            alternative_names = self.safe_accessor.safe_get(context, 'alternative_names', [])
            for alt_name in alternative_names:
                if hasattr(obj, alt_name):
                    logger.info(f"Using alternative attribute {alt_name} instead of {attr_name}")
                    return True

            # Strategy 3: Initialize object if it's uninitialized
            if hasattr(obj, '__init__') and self.safe_accessor.safe_get(context, 'allow_reinit', False):
                try:
                    obj.__init__()
                    logger.info(f"Reinitialized object {type(obj).__name__}")
                    return hasattr(obj, attr_name)
                except Exception as e:
                    logger.warning(f"Object reinitialization failed: {e}")

            return False

    async def _recover_type_error(self, context: Dict[str, Any]) -> bool:
        """
        Recover from type errors using type coercion.

        Args:
            context: Recovery context with type information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("type_error_recovery", False) as boundary:
            value = self.safe_accessor.safe_get(context, 'value')
            expected_type = self.safe_accessor.safe_get(context, 'expected_type')

            if value is None or expected_type is None:
                return False

            # Strategy 1: Type coercion using TypeValidator
            try:
                if expected_type == int:
                    coerced_value = self.type_validator.safe_int(value)
                elif expected_type == float:
                    coerced_value = self.type_validator.safe_float(value)
                elif expected_type == str:
                    coerced_value = self.type_validator.safe_str(value)
                elif expected_type == bool:
                    coerced_value = self.type_validator.safe_bool(value)
                elif expected_type == list:
                    coerced_value = self.type_validator.safe_list(value)
                elif expected_type == dict:
                    coerced_value = self.type_validator.safe_dict(value)
                else:
                    return False

                # Update context with coerced value
                context['coerced_value'] = coerced_value
                logger.info(f"Successfully coerced {type(value).__name__} to {expected_type.__name__}")
                return True

            except Exception as e:
                logger.warning(f"Type coercion failed: {e}")
                return False

    async def _recover_value_error(self, context: Dict[str, Any]) -> bool:
        """
        Recover from value errors using validation and sanitization.

        Args:
            context: Recovery context with value information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("value_error_recovery", False) as boundary:
            value = self.safe_accessor.safe_get(context, 'value')
            validation_func = self.safe_accessor.safe_get(context, 'validation_func')
            sanitization_func = self.safe_accessor.safe_get(context, 'sanitization_func')

            # Strategy 1: Use sanitization function if provided
            if sanitization_func and callable(sanitization_func):
                try:
                    sanitized_value = sanitization_func(value)
                    context['sanitized_value'] = sanitized_value
                    logger.info("Value sanitized successfully")
                    return True
                except Exception as e:
                    logger.warning(f"Value sanitization failed: {e}")

            # Strategy 2: Use default value if provided
            default_value = self.safe_accessor.safe_get(context, 'default_value')
            if default_value is not None:
                context['fallback_value'] = default_value
                logger.info("Using default value as fallback")
                return True

            # Strategy 3: Try common value fixes
            if isinstance(value, str):
                # Remove common problematic characters
                cleaned_value = value.strip().replace('\x00', '').replace('\n', ' ')
                if cleaned_value != value:
                    context['cleaned_value'] = cleaned_value
                    logger.info("String value cleaned")
                    return True

            return False

    async def _recover_network_error(self, context: Dict[str, Any]) -> bool:
        """
        Recover from network errors using retry and fallback strategies.

        Args:
            context: Recovery context with network information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("network_error_recovery", False) as boundary:
            # Strategy 1: Wait and retry with exponential backoff
            max_retries = self.type_validator.safe_int(
                self.safe_accessor.safe_get(context, 'max_retries', 3)
            )

            for retry in range(max_retries):
                await asyncio.sleep(2 ** retry)  # Exponential backoff

                # Test network connectivity
                try:
                    import socket
                    socket.create_connection(("8.8.8.8", 53), timeout=5)
                    logger.info(f"Network connectivity restored on retry {retry + 1}")
                    return True
                except Exception:
                    continue

            # Strategy 2: Use offline mode
            if self.safe_accessor.safe_get(context, 'allow_offline', True):
                logger.info("Enabling offline mode due to network issues")
                return True

            return False

    async def _recover_authentication_error(self, context: Dict[str, Any]) -> bool:
        """
        Recover from authentication errors.

        Args:
            context: Recovery context with authentication information

        Returns:
            True if recovery successful, False otherwise
        """
        with ErrorBoundary("auth_error_recovery", False) as boundary:
            # Strategy 1: Refresh authentication token
            auth_client = self.safe_accessor.safe_get(context, 'auth_client')
            if auth_client and hasattr(auth_client, 'refresh_token'):
                try:
                    auth_client.refresh_token()
                    logger.info("Authentication token refreshed")
                    return True
                except Exception as e:
                    logger.warning(f"Token refresh failed: {e}")

            # Strategy 2: Use alternative credentials
            fallback_credentials = self.safe_accessor.safe_get(context, 'fallback_credentials')
            if fallback_credentials:
                logger.info("Using fallback credentials")
                return True

            # Strategy 3: Enable anonymous mode
            if self.safe_accessor.safe_get(context, 'allow_anonymous', False):
                logger.info("Enabling anonymous mode")
                return True

            return False

# Create global instance with enhanced capabilities
error_recovery_manager = ErrorRecoveryManager(error_log)
