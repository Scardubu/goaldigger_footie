#!/usr/bin/env python3
"""
Data Source Checker for GoalDiggers

This module provides functionality to check the availability and status
of all data sources used by the GoalDiggers platform.

Usage:
    Run directly: python utils/data_source_checker.py
    Import: from utils.data_source_checker import DataSourceChecker
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException

# Configure logging
logger = logging.getLogger(__name__)

class DataSourceChecker:
    """Checks the availability and status of data sources."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize data source checker.
        
        Args:
            config_path: Path to data source config file (optional)
        """
        self.config_path = config_path or os.path.join('config', 'data_source_config.json')
        self.config = self._load_config()
        self.timeout = 10  # Default timeout for requests in seconds
        
    def _load_config(self) -> Dict[str, Any]:
        """Load data source configuration.
        
        Returns:
            Dictionary with data source configuration
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded data source configuration from {self.config_path}")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading data source configuration: {e}")
            return {"data_sources": {}}
    
    def check_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """Check all configured data sources.
        
        Returns:
            Dictionary with status for each data source
        """
        results = {}
        
        data_sources = self.config.get("data_sources", {})
        
        for source_name, source_config in data_sources.items():
            # Skip disabled sources
            if not source_config.get("enabled", True):
                results[source_name] = {
                    "accessible": False,
                    "status": "disabled",
                    "message": "Data source is disabled in configuration",
                    "last_check": datetime.now().isoformat()
                }
                continue
            
            # Check based on source type
            source_type = source_config.get("type", "")
            
            if source_type == "rest_api":
                results[source_name] = self._check_api_source(source_name, source_config)
            elif source_type == "file":
                results[source_name] = self._check_file_source(source_name, source_config)
            else:
                results[source_name] = {
                    "accessible": False,
                    "status": "unknown_type",
                    "message": f"Unknown data source type: {source_type}",
                    "last_check": datetime.now().isoformat()
                }
        
        return results
    
    def _check_api_source(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an API data source is accessible.
        
        Args:
            source_name: Name of the data source
            config: Configuration for the data source
            
        Returns:
            Dictionary with status information
        """
        try:
            base_url = config.get("base_url", "")
            if not base_url:
                return {
                    "accessible": False,
                    "status": "misconfigured",
                    "message": "No base URL configured",
                    "last_check": datetime.now().isoformat()
                }
            
            # Get a test endpoint - prefer a status endpoint if available
            endpoints = config.get("endpoints", {})
            test_endpoint = endpoints.get("status", endpoints.get("health", ""))
            
            # If no status endpoint, use the first available endpoint
            if not test_endpoint and endpoints:
                test_endpoint = next(iter(endpoints.values()))
            
            # Format the URL - replace any placeholder variables with 1
            if test_endpoint:
                import re

                # Replace placeholders like {id} with 1
                test_endpoint = re.sub(r'\{[^{}]*\}', '1', test_endpoint)
                url = f"{base_url}{test_endpoint}"
            else:
                # Fallback to base URL
                url = base_url
            
            # Add authentication if configured
            headers = {}
            api_key = config.get("api_key")
            
            if api_key:
                # Try to find the right header name from config
                header_name = "X-API-Key"  # Default
                
                # Check if there's a custom header specified
                custom_headers = config.get("headers", {})
                if custom_headers:
                    # Use the first header that might contain a key
                    for name, value in custom_headers.items():
                        if "key" in name.lower() or "token" in name.lower():
                            header_name = name
                            break
                
                headers[header_name] = api_key
            
            # Add any additional headers
            headers.update(config.get("headers", {}))
            
            # Add query parameters if configured
            params = config.get("params", {})
            
            # Make the request
            timeout = config.get("timeout", self.timeout)
            start_time = time.time()
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Check the response
            if response.status_code < 400:
                return {
                    "accessible": True,
                    "status": "ok",
                    "message": f"API responded with status code {response.status_code}",
                    "response_time_ms": round(response_time, 2),
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {
                    "accessible": False,
                    "status": "error",
                    "message": f"API responded with error status code {response.status_code}",
                    "response_time_ms": round(response_time, 2),
                    "last_check": datetime.now().isoformat()
                }
                
        except RequestException as e:
            return {
                "accessible": False,
                "status": "error",
                "message": f"Error connecting to API: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
        except Exception as e:
            return {
                "accessible": False,
                "status": "error",
                "message": f"Unexpected error checking API: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _check_file_source(self, source_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a file data source is accessible.
        
        Args:
            source_name: Name of the data source
            config: Configuration for the data source
            
        Returns:
            Dictionary with status information
        """
        try:
            path = config.get("path", "")
            if not path:
                return {
                    "accessible": False,
                    "status": "misconfigured",
                    "message": "No path configured",
                    "last_check": datetime.now().isoformat()
                }
            
            # Check if directory exists
            if not os.path.isdir(path):
                return {
                    "accessible": False,
                    "status": "not_found",
                    "message": f"Directory not found: {path}",
                    "last_check": datetime.now().isoformat()
                }
            
            # Check for required files
            files = config.get("files", {})
            missing_files = []
            found_files = []
            
            for file_name, file_path in files.items():
                full_path = os.path.join(path, file_path)
                if not os.path.isfile(full_path):
                    missing_files.append(file_path)
                else:
                    found_files.append(file_path)
            
            if missing_files:
                return {
                    "accessible": False if not found_files else True,
                    "status": "partial" if found_files else "not_found",
                    "message": f"Missing {len(missing_files)} required files: {', '.join(missing_files)}",
                    "missing_files": missing_files,
                    "found_files": found_files,
                    "last_check": datetime.now().isoformat()
                }
            else:
                return {
                    "accessible": True,
                    "status": "ok",
                    "message": f"All {len(found_files)} required files found",
                    "found_files": found_files,
                    "last_check": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "accessible": False,
                "status": "error",
                "message": f"Unexpected error checking file source: {str(e)}",
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def check_source(self, source_name: str) -> Dict[str, Any]:
        """Check a specific data source by name.
        
        Args:
            source_name: Name of the data source to check
            
        Returns:
            Dictionary with status information
        """
        data_sources = self.config.get("data_sources", {})
        
        if source_name not in data_sources:
            return {
                "accessible": False,
                "status": "not_found",
                "message": f"Data source '{source_name}' not found in configuration",
                "last_check": datetime.now().isoformat()
            }
        
        source_config = data_sources[source_name]
        
        # Skip disabled sources
        if not source_config.get("enabled", True):
            return {
                "accessible": False,
                "status": "disabled",
                "message": "Data source is disabled in configuration",
                "last_check": datetime.now().isoformat()
            }
        
        # Check based on source type
        source_type = source_config.get("type", "")
        
        if source_type == "rest_api":
            return self._check_api_source(source_name, source_config)
        elif source_type == "file":
            return self._check_file_source(source_name, source_config)
        else:
            return {
                "accessible": False,
                "status": "unknown_type",
                "message": f"Unknown data source type: {source_type}",
                "last_check": datetime.now().isoformat()
            }
    
    def get_failover_source(self, source_name: str) -> Optional[str]:
        """Get a failover data source for a given source.
        
        Args:
            source_name: Name of the data source that failed
            
        Returns:
            Name of the failover source if available, None otherwise
        """
        # Check if failover is enabled
        failover_config = self.config.get("failover", {})
        if not failover_config.get("enabled", True):
            return None
        
        # Check if the source exists
        data_sources = self.config.get("data_sources", {})
        if source_name not in data_sources:
            return None
        
        # Get the priority order
        integration_config = self.config.get("data_integration", {})
        priority_order = integration_config.get("priority_order", [])
        
        if not priority_order or source_name not in priority_order:
            return None
        
        # Find the next source in priority order
        source_index = priority_order.index(source_name)
        
        # Try each subsequent source in order
        for i in range(source_index + 1, len(priority_order)):
            next_source = priority_order[i]
            
            # Check if it's enabled
            if next_source in data_sources and data_sources[next_source].get("enabled", True):
                # Check if it's accessible
                status = self.check_source(next_source)
                if status.get("accessible", False):
                    return next_source
        
        # No suitable failover found
        return None

# Singleton instance
_data_source_checker_instance = None

def get_data_source_checker() -> DataSourceChecker:
    """Get the global data source checker instance.
    
    Returns:
        DataSourceChecker instance
    """
    global _data_source_checker_instance
    if _data_source_checker_instance is None:
        _data_source_checker_instance = DataSourceChecker()
    return _data_source_checker_instance


# If running as a script
if __name__ == "__main__":
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from utils.logging_config import configure_logging  # type: ignore
            configure_logging()
        except Exception:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    
    # Check command line arguments
    import sys

    # Get checker
    checker = get_data_source_checker()
    
    # Check if a specific source was specified
    if len(sys.argv) > 1:
        source_name = sys.argv[1]
        logger.info(f"Checking data source: {source_name}")
        result = checker.check_source(source_name)
        
        # Print result
        status = "ACCESSIBLE" if result.get("accessible") else "NOT ACCESSIBLE"
        logger.info(f"{source_name}: {status} - {result.get('message')}")
        
        # Check for failover if not accessible
        if not result.get("accessible"):
            failover = checker.get_failover_source(source_name)
            if failover:
                logger.info(f"Failover source available: {failover}")
            else:
                logger.info("No failover source available")
    else:
        # Check all sources
        logger.info("Checking all data sources...")
        results = checker.check_all_sources()
        
        # Print results
        for source_name, result in results.items():
            status = "ACCESSIBLE" if result.get("accessible") else "NOT ACCESSIBLE"
            logger.info(f"{source_name}: {status} - {result.get('message')}")
    
    # Exit with success status
    sys.exit(0)
