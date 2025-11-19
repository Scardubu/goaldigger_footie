"""
System monitoring utility for tracking health of various components.
Provides real-time metrics and status information for the dashboard.
"""
import logging
import os
import platform
import socket
import statistics
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import psutil
import requests

from utils.config import Config

logger = logging.getLogger(__name__)

# Lazy import function to avoid circular dependency
def log_error(error, context=None):
    """Lazily import log_error to avoid circular imports."""
    try:
        from dashboard.error_log import log_error as dashboard_log_error
        return dashboard_log_error(error, context)
    except ImportError:
        logger.error(f"Error: {error}, Context: {context}")

class SystemMonitor:
    """
    System monitoring utility for tracking health of various components.
    Provides real-time metrics and status information for the dashboard.
    """
    
    def __init__(self):
        """Initialize the system monitor."""
        # Load configuration
        self.config = self._load_config()
        
        # Initialize metrics
        self.metrics = {
            "last_update": datetime.now(),
            "system": self._get_system_info(),
            "resources": self._get_resource_usage(),
            "components": self._get_component_status(),
            "api_keys": self._get_api_key_status()
        }
        
        # Operations tracking
        self.operations = {}
        self.operation_history = {}
        self.max_history_size = 100
        
    def _load_config(self) -> Dict[str, Any]:
        """Load system monitoring configuration from Config."""
        return Config.get("system", {})
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        try:
            return {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": platform.python_version(),
                "hostname": socket.gethostname(),
                "cpu_count": psutil.cpu_count(logical=True),
                "memory_total": psutil.virtual_memory().total,
                "disk_total": psutil.disk_usage('/').total
            }
        except Exception as e:
            log_error("Error getting system info", e)
            return {}
            
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage metrics."""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available": psutil.virtual_memory().available,
                "disk_percent": psutil.disk_usage('/').percent,
                "disk_free": psutil.disk_usage('/').free,
                "network_io": psutil.net_io_counters()._asdict() if hasattr(psutil, 'net_io_counters') else {}
            }
        except Exception as e:
            log_error("Error getting resource usage", e)
            return {}
            
    def _get_component_status(self) -> Dict[str, Any]:
        """Get status of various system components."""
        components = {}
        
        # Check database status
        db_path = Config.get("database.path", "football.db")
        components["database"] = {
            "status": "ok" if os.path.exists(db_path) else "error",
            "path": db_path,
            "exists": os.path.exists(db_path),
            "size_mb": os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0
        }
        
        # Check MCP server status
        mcp_server_url = Config.get("mcp_server_url", "http://localhost:3000")
        mcp_status = "unknown"
        try:
            response = requests.get(f"{mcp_server_url}/health", timeout=2)
            mcp_status = "ok" if response.status_code == 200 else "error"
        except:
            mcp_status = "error"
            
        components["mcp"] = {
            "status": mcp_status,
            "url": mcp_server_url
        }
        
        # Check proxy status
        proxy_config = Config.get("proxies", {})
        proxy_enabled = proxy_config.get("enabled", False)
        components["proxies"] = {
            "status": "ok" if proxy_enabled else "disabled",
            "enabled": proxy_enabled,
            "provider": proxy_config.get("provider", "None"),
            "count": proxy_config.get("count", 0)
        }
        
        return components
        
    def _get_api_key_status(self) -> Dict[str, bool]:
        """Check if required API keys are set in environment variables."""
        required_keys = [
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "FOOTBALL_DATA_API_KEY",
            "API_FOOTBALL_KEY"
        ]
        
        return {key: bool(os.getenv(key)) for key in required_keys}
        
    def update(self) -> Dict[str, Any]:
        """Update all metrics and return the current status."""
        self.metrics = {
            "last_update": datetime.now(),
            "system": self._get_system_info(),
            "resources": self._get_resource_usage(),
            "components": self._get_component_status(),
            "api_keys": self._get_api_key_status()
        }
        
        return self.metrics
        
    def get_status(self) -> Dict[str, Any]:
        """Get the current system status."""
        # Update if it's been more than 30 seconds since last update
        if (datetime.now() - self.metrics["last_update"]).total_seconds() > 30:
            self.update()
            
        return self.metrics
        
    def get_health_summary(self) -> Dict[str, str]:
        """Get a summary of system health for the dashboard."""
        status = self.get_status()
        
        # Check API keys
        api_keys_status = "ok" if all(status["api_keys"].values()) else "error"
        
        # Check database
        database_status = status["components"]["database"]["status"]
        
        # Check MCP server
        mcp_status = status["components"]["mcp"]["status"]
        
        # Check proxies
        proxies_status = status["components"]["proxies"]["status"]
        
        # Check resource usage
        resources = status["resources"]
        resource_status = "ok"
        if resources.get("cpu_percent", 0) > 90 or resources.get("memory_percent", 0) > 90 or resources.get("disk_percent", 0) > 90:
            resource_status = "warning"
            
        return {
            "api_keys": {
                "status": api_keys_status,
                "details": status["api_keys"]
            },
            "database": {
                "status": database_status,
                "details": status["components"]["database"]
            },
            "mcp": {
                "status": mcp_status,
                "details": status["components"]["mcp"]
            },
            "proxies": {
                "status": proxies_status,
                "details": status["components"]["proxies"]
            },
            "resources": {
                "status": resource_status,
                "cpu": resources.get("cpu_percent", 0),
                "memory": resources.get("memory_percent", 0),
                "disk": resources.get("disk_percent", 0)
            }
        }
        
    def test_api_keys(self) -> Dict[str, bool]:
        """Test API keys to ensure they are valid and working."""
        results = {}
        
        # Test GEMINI_API_KEY
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                # Simple request to test Gemini API key
                url = "https://generativelanguage.googleapis.com/v1beta/models"
                headers = {"x-goog-api-key": gemini_key}
                response = requests.get(url, headers=headers, timeout=5)
                results["GEMINI_API_KEY"] = response.status_code == 200
            except:
                results["GEMINI_API_KEY"] = False
        else:
            results["GEMINI_API_KEY"] = False
            
        # Test OPENROUTER_API_KEY
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                # Simple request to test OpenRouter API key
                url = "https://openrouter.ai/api/v1/models"
                headers = {"Authorization": f"Bearer {openrouter_key}"}
                response = requests.get(url, headers=headers, timeout=5)
                results["OPENROUTER_API_KEY"] = response.status_code == 200
            except:
                results["OPENROUTER_API_KEY"] = False
        else:
            results["OPENROUTER_API_KEY"] = False
            
        # Test FOOTBALL_DATA_API_KEY
        football_data_key = os.getenv("FOOTBALL_DATA_API_KEY")
        if football_data_key:
            try:
                # Simple request to test Football-Data API key
                url = "https://api.football-data.org/v4/competitions"
                headers = {"X-Auth-Token": football_data_key}
                response = requests.get(url, headers=headers, timeout=5)
                results["FOOTBALL_DATA_API_KEY"] = response.status_code == 200
            except:
                results["FOOTBALL_DATA_API_KEY"] = False
        else:
            results["FOOTBALL_DATA_API_KEY"] = False
            
        # Test API_FOOTBALL_KEY
        api_football_key = os.getenv("API_FOOTBALL_KEY")
        if api_football_key:
            try:
                # Simple request to test API-Football key
                url = "https://api-football-v1.p.rapidapi.com/v3/status"
                headers = {
                    "x-rapidapi-key": api_football_key,
                    "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
                }
                response = requests.get(url, headers=headers, timeout=5)
                results["API_FOOTBALL_KEY"] = response.status_code == 200
            except:
                results["API_FOOTBALL_KEY"] = False
        else:
            results["API_FOOTBALL_KEY"] = False
            
        return results
        
    def restart_mcp_server(self) -> bool:
        """Attempt to restart the MCP server."""
        try:
            mcp_server_url = Config.get("mcp_server_url", "http://localhost:3000")
            response = requests.post(f"{mcp_server_url}/restart", timeout=5)
            return response.status_code == 200
        except Exception as e:
            log_error("Error restarting MCP server", e)
            return False
            
    def reset_proxies(self) -> bool:
        """Attempt to reset proxies."""
        try:
            # This would need to be implemented based on your proxy management system
            # For now, just return True as a placeholder
            logger.info("Proxy reset requested - implementation needed")
            return True
        except Exception as e:
            log_error("Error resetting proxies", e)
            return False
            
    def start_operation(self, operation_name: str) -> str:
        """Start tracking a new operation.
        
        Args:
            operation_name: Name of the operation to track
            
        Returns:
            Operation ID that can be used to end the operation
        """
        operation_id = str(uuid.uuid4())
        self.operations[operation_id] = {
            "name": operation_name,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "status": "running"
        }
        
        # Initialize operation history if not already present
        if operation_name not in self.operation_history:
            self.operation_history[operation_name] = {
                "count": 0,
                "success_count": 0,
                "failure_count": 0,
                "durations": [],
                "avg_duration": None,
                "min_duration": None,
                "max_duration": None,
                "p90_duration": None,
                "last_status": None,
                "last_completed": None
            }
            
        logger.debug(f"Started operation: {operation_name} (ID: {operation_id})")
        return operation_id
    
    def end_operation(self, operation_id: str, status: str = "success") -> Dict[str, Any]:
        """End a tracked operation and record metrics.
        
        Args:
            operation_id: ID of the operation to end
            status: Operation status ("success" or "failure")
            
        Returns:
            Dictionary with operation metrics
        """
        if operation_id not in self.operations:
            logger.warning(f"Operation ID not found: {operation_id}")
            return {}
            
        operation = self.operations[operation_id]
        operation_name = operation["name"]
        operation["end_time"] = time.time()
        operation["duration"] = operation["end_time"] - operation["start_time"]
        operation["status"] = status
        
        # Update operation history
        history = self.operation_history[operation_name]
        history["count"] += 1
        history["last_completed"] = datetime.now()
        history["last_status"] = status
        
        if status == "success":
            history["success_count"] += 1
        else:
            history["failure_count"] += 1
            
        # Update duration statistics
        history["durations"].append(operation["duration"])
        # Keep only the last N durations
        if len(history["durations"]) > self.max_history_size:
            history["durations"] = history["durations"][-self.max_history_size:]
            
        # Calculate statistics
        if history["durations"]:
            history["avg_duration"] = statistics.mean(history["durations"])
            history["min_duration"] = min(history["durations"])
            history["max_duration"] = max(history["durations"])
            if len(history["durations"]) >= 10:  # Only calculate p90 if we have enough data
                sorted_durations = sorted(history["durations"])
                p90_index = int(len(sorted_durations) * 0.9)
                history["p90_duration"] = sorted_durations[p90_index]
        
        logger.debug(f"Ended operation: {operation_name} (ID: {operation_id}) - "
                    f"Duration: {operation['duration']:.3f}s, Status: {status}")
        
        # Remove operation from active tracking
        self.operations.pop(operation_id)
        
        return operation
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get statistics for tracked operations.
        
        Args:
            operation_name: Optional name of specific operation to get stats for
            
        Returns:
            Dictionary of operation statistics or list of all operation statistics
        """
        if operation_name:
            if operation_name in self.operation_history:
                return self.operation_history[operation_name]
            return {}
        
        # Return stats for all operations
        result = []
        for name, stats in self.operation_history.items():
            stats_copy = stats.copy()
            stats_copy["name"] = name
            result.append(stats_copy)
            
        return result


    def get_system_health(self) -> Dict[str, str]:
        """Alias for get_health_summary for backwards compatibility."""
        return self.get_health_summary()


"""
Lazy singleton accessor to avoid import-time side effects and circular imports.
"""
_system_monitor_singleton: Optional[SystemMonitor] = None

def _get_system_monitor() -> SystemMonitor:
    global _system_monitor_singleton
    if _system_monitor_singleton is None:
        _system_monitor_singleton = SystemMonitor()
    return _system_monitor_singleton

# Convenience functions for direct use without creating a monitor instance

def get_system_status() -> Dict[str, Any]:
    """Get the current system status."""
    return _get_system_monitor().get_status()
    
def get_health_summary() -> Dict[str, str]:
    """Get a summary of system health for the dashboard."""
    return _get_system_monitor().get_health_summary()
    
def get_system_health() -> Dict[str, str]:
    """Alias for get_health_summary."""
    return _get_system_monitor().get_health_summary()
    
def test_api_keys() -> Dict[str, bool]:
    """Test API keys to ensure they are valid and working."""
    return _get_system_monitor().test_api_keys()
    
def restart_mcp_server() -> bool:
    """Attempt to restart the MCP server."""
    return _get_system_monitor().restart_mcp_server()
    
def reset_proxies() -> bool:
    """Attempt to reset proxies."""
    return _get_system_monitor().reset_proxies()
    
def start_operation(operation_name: str) -> str:
    """Start tracking a new operation."""
    return _get_system_monitor().start_operation(operation_name)
    
def end_operation(operation_id: str, status: str = "success") -> Dict[str, Any]:
    """End a tracked operation."""
    return _get_system_monitor().end_operation(operation_id, status)
    
def get_operation_stats(operation_name: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Get statistics for tracked operations."""
    return _get_system_monitor().get_operation_stats(operation_name)

# Export the SystemMonitor class for direct imports
__all__ = ['SystemMonitor', 'get_system_status', 'get_health_summary', 'get_system_health', 
           'test_api_keys', 'restart_mcp_server', 'reset_proxies', 
           'start_operation', 'end_operation', 'get_operation_stats']

# Ensure the SystemMonitor class is available for import
def get_system_monitor_class():
    """Get the SystemMonitor class for direct import."""
    return SystemMonitor
