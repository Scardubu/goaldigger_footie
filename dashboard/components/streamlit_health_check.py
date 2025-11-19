#!/usr/bin/env python3
"""
Streamlit Health Check Integration Component
===========================================

This component provides health check functionality for the GoalDiggers app
and integrates with the health check server.
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
import streamlit as st

logger = logging.getLogger(__name__)

class StreamlitHealthCheck:
    """Health check component for Streamlit integration."""
    
    def __init__(self):
        """Initialize the health check component."""
        self.health_check_port = int(os.environ.get('HEALTH_CHECK_PORT', 8502))
        self.health_check_url = f"http://localhost:{self.health_check_port}/health"
        self.detailed_health_url = f"http://localhost:{self.health_check_port}/health/detailed"
        self.database_health_url = f"http://localhost:{self.health_check_port}/health/database"
        self.last_health_status = None
        self.last_check_time = None
        
        # Set up background thread for health monitoring
        self.stopping = False
        self.health_thread = threading.Thread(target=self._monitor_health, daemon=True)
        self.health_thread.start()
    
    def _monitor_health(self):
        """Background thread to monitor health status."""
        while not self.stopping:
            try:
                self.check_health()
            except Exception as e:
                logger.error(f"Error monitoring health: {e}")
            
            # Wait before next check
            time.sleep(60)  # Check every minute
    
    def check_health(self) -> Dict:
        """Check the current health status."""
        try:
            response = requests.get(self.health_check_url, timeout=5)
            if response.status_code == 200:
                self.last_health_status = response.json()
                self.last_check_time = datetime.now()
                return self.last_health_status
            else:
                logger.warning(f"Health check returned status code: {response.status_code}")
                return {"status": "error", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_detailed_health(self) -> Dict:
        """Get detailed health information."""
        try:
            response = requests.get(self.detailed_health_url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            logger.error(f"Error getting detailed health: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_database_health(self) -> Dict:
        """Get database health information."""
        try:
            response = requests.get(self.database_health_url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "error": f"Status code: {response.status_code}"}
        except Exception as e:
            logger.error(f"Error getting database health: {e}")
            return {"status": "error", "error": str(e)}
    
    def render_health_indicator(self):
        """Render a small health status indicator in the Streamlit UI."""
        if self.last_health_status is None:
            status = self.check_health().get("status", "unknown")
        else:
            status = self.last_health_status.get("status", "unknown")
        
        if status == "healthy":
            st.sidebar.success("● System: Healthy")
        elif status == "degraded":
            st.sidebar.warning("● System: Degraded")
        else:
            st.sidebar.error("● System: Unhealthy")
    
    def render_health_dashboard(self):
        """Render a complete health dashboard."""
        st.title("🏥 System Health Dashboard")
        
        # Get current health status
        health = self.check_health()
        detailed = self.get_detailed_health()
        db_health = self.get_database_health()
        
        # Basic health metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status = health.get("status", "unknown")
            if status == "healthy":
                st.success("SYSTEM STATUS: HEALTHY")
            elif status == "degraded":
                st.warning("SYSTEM STATUS: DEGRADED")
            else:
                st.error("SYSTEM STATUS: UNHEALTHY")
        
        with col2:
            uptime_seconds = health.get("uptime_seconds", 0)
            hours, remainder = divmod(uptime_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            st.info(f"UPTIME: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        with col3:
            st.info(f"VERSION: {health.get('version', 'unknown')}")
        
        # System metrics
        if "system" in detailed:
            st.subheader("System Resources")
            sys_metrics = detailed["system"]
            
            sys_col1, sys_col2, sys_col3 = st.columns(3)
            with sys_col1:
                st.metric("CPU Usage", f"{sys_metrics.get('cpu_percent', 0):.1f}%")
            with sys_col2:
                st.metric("Memory Usage", f"{sys_metrics.get('memory_percent', 0):.1f}%")
            with sys_col3:
                st.metric("Disk Space Free", f"{sys_metrics.get('disk_free_gb', 0):.1f} GB")
        
        # Component status
        if "components" in detailed:
            st.subheader("Component Status")
            components = detailed["components"]
            
            for name, info in components.items():
                status = info.get("status", "unknown")
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if status == "healthy":
                        st.success(f"{name.upper()}")
                    elif status == "degraded":
                        st.warning(f"{name.upper()}")
                    elif status == "error":
                        st.error(f"{name.upper()}")
                    else:
                        st.info(f"{name.upper()}")
                
                with col2:
                    last_check = info.get("last_check", "Never")
                    if isinstance(last_check, str) and "T" in last_check:
                        # Parse ISO format
                        try:
                            dt = datetime.fromisoformat(last_check.replace('Z', '+00:00'))
                            last_check = dt.strftime("%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    
                    st.write(f"Status: {status}, Last Check: {last_check}")
        
        # Database status
        st.subheader("Database Health")
        
        db_col1, db_col2 = st.columns(2)
        with db_col1:
            db_status = db_health.get("status", "unknown")
            if db_status == "healthy":
                st.success("Database Status: HEALTHY")
            elif db_status == "degraded":
                st.warning("Database Status: DEGRADED")
            else:
                st.error("Database Status: UNHEALTHY")
        
        with db_col2:
            if "tables_count" in db_health:
                st.info(f"Tables: {db_health['tables_count']}")
            if "missing_tables" in db_health and db_health["missing_tables"]:
                st.warning(f"Missing tables: {', '.join(db_health['missing_tables'])}")
        
        # Last check time
        if self.last_check_time:
            st.caption(f"Last refreshed: {self.last_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Refresh button
        if st.button("Refresh Health Data"):
            st.rerun()


# Singleton instance
_streamlit_health_check = None

def get_streamlit_health_check():
    """Get the singleton instance of the health check component."""
    global _streamlit_health_check
    if _streamlit_health_check is None:
        _streamlit_health_check = StreamlitHealthCheck()
    return _streamlit_health_check

if __name__ == "__main__":
    # For testing
    health_check = StreamlitHealthCheck()
    print(json.dumps(health_check.check_health(), indent=2))
