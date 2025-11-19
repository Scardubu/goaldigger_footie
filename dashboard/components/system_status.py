"""
System status component for the dashboard.
Monitors health of API keys, database, MCP server, and proxies.
"""
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.components.ui_elements import (create_metric_card,
                                              create_themed_card)
from dashboard.data_integration import get_scraper_status
from dashboard.error_log import ErrorLog, log_error
from utils.config import Config
from utils.system_monitor import get_operation_stats
from utils.system_monitor import get_system_status as get_monitor_status

logger = logging.getLogger(__name__)

def render_system_status(
    error_log: ErrorLog,
    restart_mcp_fn: Optional[Callable[[], bool]] = None,
    reset_proxies_fn: Optional[Callable[[], bool]] = None,
    test_api_keys_fn: Optional[Callable[[], Dict[str, bool]]] = None,
) -> None:
    """
    Render system status dashboard with monitoring and admin controls.

    Args:
        error_log: Error log instance to display recent errors.
        restart_mcp_fn: Optional function to restart MCP server.
        reset_proxies_fn: Optional function to reset proxies.
        test_api_keys_fn: Optional function to test API keys.
    """
    try:
        st.header("System Health & Status")
        st.markdown("An overview of system resources, component health, and operational metrics.")

        # System resource usage
        st.subheader("System Resources")
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        col1, col2, col3 = st.columns(3)
        with col1:
            create_metric_card("CPU Usage", f"{cpu_percent}%", "Normal" if cpu_percent < 75 else "High")
        with col2:
            create_metric_card("Memory Usage", f"{memory.percent}%", f"{memory.used / (1024**3):.2f} / {memory.total / (1024**3):.2f} GB")
        with col3:
            create_metric_card("Disk Usage", f"{disk.percent}%", f"{disk.used / (1024**3):.2f} / {disk.total / (1024**3):.2f} GB")

        # Component Health
        st.subheader("Component Health")
        health_tabs = st.tabs(["APIs & Services", "Data Scrapers", "Performance", "Cache"])

        with health_tabs[0]: # APIs & Services
            api_key_status = test_api_keys_fn() if test_api_keys_fn else {}
            api_status_content = ""
            for key, is_active in api_key_status.items():
                status_icon = "✅" if is_active else "❌"
                api_status_content += f"- **{key.replace('_', ' ').title()}**: {status_icon} {'Active' if is_active else 'Inactive'}<br>"
            create_themed_card("API Key Status", api_status_content or "No API keys configured to test.")

            mcp_server_url = Config.get("mcp_server_url", "http://localhost:3000")
            create_themed_card("MCP Server", f"**URL**: {mcp_server_url}", bg_color="#f0f2f6")
            if restart_mcp_fn and st.button("Restart MCP Server"):
                with st.spinner("Restarting MCP server..."):
                    if restart_mcp_fn():
                        st.success("MCP server restarted successfully.")
                    else:
                        st.error("Failed to restart MCP server.")

        with health_tabs[1]: # Data Scrapers
            scraper_status = get_scraper_status()
            if scraper_status:
                for name, status in scraper_status.items():
                    success_rate = status.get("stats", {}).get("success_rate", 0) * 100
                    content = f"""
                    **Status**: {'✅ Online' if status.get("available") else '❌ Offline'}<br>
                    **Success Rate**: {success_rate:.1f}%<br>
                    **Avg. Response**: {status.get("stats", {}).get("avg_response_time", 0):.2f}s
                    """
                    create_themed_card(name.replace('_', ' ').title(), content)
            else:
                st.info("No scraper status data available.")

        with health_tabs[2]: # Performance
            operation_stats = get_operation_stats()
            if operation_stats:
                perf_df = pd.DataFrame(operation_stats).set_index("name")
                st.dataframe(perf_df, use_container_width=True)
            else:
                st.info("No performance data available.")

        with health_tabs[3]: # Cache
            try:
                from dashboard.data_integration import (clear_cache,
                                                        data_integration)
                cache_stats = data_integration.get_cache_stats()
                if cache_stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        create_metric_card("Cached Items", str(cache_stats.get("cached_items", 0)))
                    with col2:
                        create_metric_card("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1f}%")
                    with col3:
                        create_metric_card("Memory Usage", f"{cache_stats.get('memory_usage_mb', 0):.2f} MB")
                    
                    if st.button("Clear Cache"):
                        clear_cache()
                        st.success("Cache cleared successfully!")
                        st.rerun()
                else:
                    st.info("No cache statistics available.")
            except ImportError:
                st.warning("Could not load cache management components.")
            except Exception as e:
                log_error("Error rendering cache status", e)
                st.error("An error occurred while displaying cache statistics.")

        # Recent Errors
        st.subheader("Recent System Errors")
        try:
            error_log.display(max_errors=5, expanded=False)
        except Exception as e:
            log_error("Failed to display error log", e)
            st.warning("Could not display the interactive error log. Please check the logs.")

    except Exception as e:
        log_error("Error rendering system status page", e)
        st.error("An unexpected error occurred while rendering the system status page.")


def get_system_status() -> Dict[str, Any]:
    """
    Get current system status for all components including detailed performance metrics.
    
    Returns:
        Dictionary containing comprehensive status information for all components
    """
    try:
        status = {
            "api_keys": {
                "status": "ok",
                "details": {}
            },
            "database": {
                "status": "ok",
                "details": {}
            },
            "mcp": {
                "status": "ok",
                "details": {}
            },
            "proxies": {
                "status": "ok",
                "details": {}
            },
            "resources": {
                "cpu": psutil.cpu_percent(),
                "memory": psutil.virtual_memory().percent,
                "disk": psutil.disk_usage('/').percent
            }
        }
        
        # Check API keys
        required_keys = ["GEMINI_API_KEY", "OPENROUTER_API_KEY", "FOOTBALL_DATA_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            status["api_keys"]["status"] = "error"
            status["api_keys"]["details"]["missing_keys"] = missing_keys
            
        # Check database
        db_path = Config.get("database.path", "football.db")
        if not os.path.exists(db_path):
            status["database"]["status"] = "error"
            status["database"]["details"]["error"] = "Database file not found"
            
        # Check MCP server
        # This is a basic check - in production, you would want to ping the server
        mcp_server_url = Config.get("mcp_server_url", "")
        if not mcp_server_url:
            status["mcp"]["status"] = "unknown"
            status["mcp"]["details"]["error"] = "MCP server URL not configured"
            
        # Check proxies
        proxy_config = Config.get("proxies", {})
        if not proxy_config.get("enabled", False):
            status["proxies"]["status"] = "disabled"
            
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "api_keys": {"status": "unknown"},
            "database": {"status": "unknown"},
            "mcp": {"status": "unknown"},
            "proxies": {"status": "unknown"},
            "resources": {
                "cpu": 0,
                "memory": 0,
                "disk": 0
            }
        }
