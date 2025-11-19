#!/usr/bin/env python3
"""
System Health Check for GoalDiggers Platform

This utility performs comprehensive checks to verify the health and integrity
of all GoalDiggers platform components including:
- Database connections and table structure
- API endpoints and servers
- Real-time services
- Dashboard components
- Data integrity
- Model status and availability
"""

import importlib
import inspect
import logging
import os
import socket
import sqlite3
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Union

import requests
from termcolor import colored

# Central logging configuration (skip in pytest) with optional fallback
if not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        from utils.logging_config import configure_logging  # type: ignore
        configure_logging()
    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
logger = logging.getLogger(__name__)

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import needed modules
try:
    import streamlit as st
    import uvicorn

    from database.db_manager import DatabaseManager
    from utils.config import Config
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    print(colored(f"❌ ERROR: Failed to import required modules: {e}", "red"))
    sys.exit(1)


class SystemHealthCheck:
    """Performs comprehensive health checks on all system components."""

    def __init__(self):
        self.results = {
            "database": {"status": "pending", "details": []},
            "api": {"status": "pending", "details": []},
            "sse": {"status": "pending", "details": []},
            "dashboard": {"status": "pending", "details": []},
            "models": {"status": "pending", "details": []},
            "static_assets": {"status": "pending", "details": []},
            "component_compatibility": {"status": "pending", "details": []},
        }
        
        self.config = Config()
        self.db_path = os.path.join(PROJECT_ROOT, "data", "football.db")
        
        # API endpoints
        self.api_url = "http://localhost:5000"
        self.sse_url = "http://localhost:8079"
        self.dashboard_url = "http://localhost:8501"
        
        # Initialize DB manager
        try:
            self.db_manager = DatabaseManager(f"sqlite:///{self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            self.db_manager = None

    def run_all_checks(self) -> Dict:
        """Run all system health checks."""
        start_time = time.time()
        logger.info("Starting GoalDiggers System Health Check")
        print(colored("\n🔍 GOALDIGGERS SYSTEM HEALTH CHECK", "cyan", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        
        # Run checks
        self.check_database()
        self.check_api_server()
        self.check_sse_server()
        self.check_dashboard()
        self.check_models()
        self.check_static_assets()
        self.check_component_compatibility()
        
        # Summarize results
        passed = sum(1 for component in self.results.values() if component["status"] == "pass")
        failed = sum(1 for component in self.results.values() if component["status"] == "fail")
        warning = sum(1 for component in self.results.values() if component["status"] == "warning")
        
        total_time = time.time() - start_time
        
        print(colored("\n📊 SYSTEM HEALTH SUMMARY", "cyan", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        print(f"✅ Passed: {passed}")
        print(f"⚠️ Warning: {warning}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️ Total check time: {total_time:.2f}s")
        print(colored("=" * 60, "cyan"))
        
        # Overall system status
        overall_status = "operational"
        if failed > 0:
            overall_status = "critical"
        elif warning > 0:
            overall_status = "degraded"
            
        print(f"\n🚦 Overall System Status: {self._colorize_status(overall_status)}")
        
        if overall_status != "operational":
            print("\n🔧 Recommended Actions:")
            if failed > 0:
                self._print_failed_components()
            if warning > 0:
                self._print_warning_components()
        else:
            print(colored("\n✨ All systems operational and optimized!", "green"))
            
        return self.results

    def check_database(self) -> None:
        """Check database connectivity and integrity."""
        print(colored("\n📁 Database Check", "yellow", attrs=["bold"]))
        
        # Check if database file exists
        if not os.path.exists(self.db_path):
            self._record_result("database", "fail", "Database file not found")
            print(colored("❌ Database file not found", "red"))
            return
        
        print(colored("✓ Database file exists", "green"))
        
        # Check if DB connection works
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            self._record_result("database", "pass", "Database connection successful")
            print(colored("✓ Database connection successful", "green"))
            
            # Check for required tables
            required_tables = ["matches", "teams", "leagues", "predictions", "odds"]
            existing_tables = []
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for row in cursor.fetchall():
                existing_tables.append(row[0])
            
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            if missing_tables:
                self._record_result("database", "warning", f"Missing tables: {', '.join(missing_tables)}")
                print(colored(f"⚠️ Missing tables: {', '.join(missing_tables)}", "yellow"))
            else:
                self._record_result("database", "pass", "All required tables exist")
                print(colored("✓ All required tables exist", "green"))
            
            # Check table structures
            for table in existing_tables:
                if table in required_tables:
                    try:
                        cursor.execute(f"PRAGMA table_info({table})")
                        columns = [row[1] for row in cursor.fetchall()]
                        self._record_result("database", "pass", f"Table '{table}' has {len(columns)} columns")
                        print(colored(f"✓ Table '{table}' has {len(columns)} columns", "green"))
                    except sqlite3.Error as e:
                        self._record_result("database", "warning", f"Error checking structure of table '{table}': {str(e)}")
                        print(colored(f"⚠️ Error checking structure of table '{table}': {str(e)}", "yellow"))
            
            # Check for data in critical tables
            for table in ["matches", "teams"]:
                if table in existing_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        if count > 0:
                            self._record_result("database", "pass", f"Table '{table}' has {count} rows")
                            print(colored(f"✓ Table '{table}' has {count} rows", "green"))
                        else:
                            self._record_result("database", "warning", f"Table '{table}' has no data")
                            print(colored(f"⚠️ Table '{table}' has no data", "yellow"))
                    except sqlite3.Error as e:
                        self._record_result("database", "warning", f"Error counting rows in table '{table}': {str(e)}")
                        print(colored(f"⚠️ Error counting rows in table '{table}': {str(e)}", "yellow"))
                        
            # Check for data anomalies
            try:
                # Check for future matches with scores
                cursor.execute("SELECT COUNT(*) FROM matches WHERE match_date > datetime('now') AND (home_score IS NOT NULL OR away_score IS NOT NULL)")
                future_matches_with_scores = cursor.fetchone()[0]
                
                if future_matches_with_scores > 0:
                    self._record_result("database", "warning", f"Found {future_matches_with_scores} future matches with scores")
                    print(colored(f"⚠️ Found {future_matches_with_scores} future matches with scores", "yellow"))
                else:
                    self._record_result("database", "pass", "No future matches have scores")
                    print(colored("✓ No future matches have scores", "green"))
                
                # Check for duplicate matches
                cursor.execute("""
                    SELECT m1.id, COUNT(*) as count 
                    FROM matches m1 
                    JOIN matches m2 ON m1.home_team_id = m2.home_team_id 
                                    AND m1.away_team_id = m2.away_team_id 
                                    AND m1.match_date = m2.match_date 
                                    AND m1.id != m2.id 
                    GROUP BY m1.home_team_id, m1.away_team_id, m1.match_date 
                    HAVING count > 0
                """)
                duplicate_matches = cursor.fetchall()
                
                if duplicate_matches:
                    self._record_result("database", "warning", f"Found {len(duplicate_matches)} sets of duplicate matches")
                    print(colored(f"⚠️ Found {len(duplicate_matches)} sets of duplicate matches", "yellow"))
                else:
                    self._record_result("database", "pass", "No duplicate matches found")
                    print(colored("✓ No duplicate matches found", "green"))
            except sqlite3.Error as e:
                self._record_result("database", "warning", f"Error checking for data anomalies: {str(e)}")
                print(colored(f"⚠️ Error checking for data anomalies: {str(e)}", "yellow"))
                
        except Exception as e:
            self._record_result("database", "fail", f"Database connection failed: {str(e)}")
            print(colored(f"❌ Database connection failed: {str(e)}", "red"))

    def check_api_server(self) -> None:
        """Check if the API server is running and responsive."""
        print(colored("\n🌐 API Server Check", "yellow", attrs=["bold"]))
        
        # Check if the server is running using socket connection
        api_host = self.api_url.split("://")[1].split(":")[0]
        api_port = int(self.api_url.split(":")[-1])
        
        if self._is_service_running(api_host, api_port):
            self._record_result("api", "pass", "API server is running")
            print(colored("✓ API server is running", "green"))
            
            # Check if the API endpoints are responsive
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    health_data = response.json()
                    self._record_result("api", "pass", f"API health endpoint responded with status: {health_data.get('status', 'unknown')}")
                    print(colored(f"✓ API health endpoint responded with status: {health_data.get('status', 'unknown')}", "green"))
                else:
                    self._record_result("api", "warning", f"API health endpoint returned status code {response.status_code}")
                    print(colored(f"⚠️ API health endpoint returned status code {response.status_code}", "yellow"))
            except requests.exceptions.RequestException as e:
                self._record_result("api", "warning", f"Failed to connect to API health endpoint: {str(e)}")
                print(colored(f"⚠️ Failed to connect to API health endpoint: {str(e)}", "yellow"))
                
            # Check predict endpoint
            try:
                test_data = {"home_team": "Liverpool", "away_team": "Chelsea", "league": "Premier League"}
                response = requests.post(f"{self.api_url}/predict", json=test_data, timeout=5)
                if response.status_code == 200:
                    self._record_result("api", "pass", "API predict endpoint is working")
                    print(colored("✓ API predict endpoint is working", "green"))
                else:
                    self._record_result("api", "warning", f"API predict endpoint returned status code {response.status_code}")
                    print(colored(f"⚠️ API predict endpoint returned status code {response.status_code}", "yellow"))
            except requests.exceptions.RequestException as e:
                self._record_result("api", "warning", f"Failed to connect to API predict endpoint: {str(e)}")
                print(colored(f"⚠️ Failed to connect to API predict endpoint: {str(e)}", "yellow"))
        else:
            self._record_result("api", "fail", "API server is not running")
            print(colored("❌ API server is not running", "red"))

    def check_sse_server(self) -> None:
        """Check if the SSE server is running and responsive."""
        print(colored("\n📡 SSE Server Check", "yellow", attrs=["bold"]))
        
        # Check if the server is running using socket connection
        sse_host = self.sse_url.split("://")[1].split(":")[0]
        sse_port = int(self.sse_url.split(":")[-1])
        
        if self._is_service_running(sse_host, sse_port):
            self._record_result("sse", "pass", "SSE server is running")
            print(colored("✓ SSE server is running", "green"))
            
            # Check if SSE endpoint is accessible
            try:
                response = requests.get(f"{self.sse_url}/health", timeout=5)
                if response.status_code == 200:
                    self._record_result("sse", "pass", "SSE health endpoint is working")
                    print(colored("✓ SSE health endpoint is working", "green"))
                else:
                    self._record_result("sse", "warning", f"SSE health endpoint returned status code {response.status_code}")
                    print(colored(f"⚠️ SSE health endpoint returned status code {response.status_code}", "yellow"))
            except requests.exceptions.RequestException as e:
                self._record_result("sse", "warning", f"Failed to connect to SSE health endpoint: {str(e)}")
                print(colored(f"⚠️ Failed to connect to SSE health endpoint: {str(e)}", "yellow"))
        else:
            self._record_result("sse", "fail", "SSE server is not running")
            print(colored("❌ SSE server is not running", "red"))

    def check_dashboard(self) -> None:
        """Check if the Streamlit dashboard is running and accessible."""
        print(colored("\n📊 Dashboard Check", "yellow", attrs=["bold"]))
        
        # Check if the dashboard is running using socket connection
        dashboard_host = self.dashboard_url.split("://")[1].split(":")[0]
        dashboard_port = int(self.dashboard_url.split(":")[-1])
        
        if self._is_service_running(dashboard_host, dashboard_port):
            self._record_result("dashboard", "pass", "Dashboard is running")
            print(colored("✓ Dashboard is running", "green"))
            
            # Check for required dashboard components
            required_components = [
                "dashboard.components.ui_elements",
                "dashboard.components.betting_insights",
                "dashboard.components.match_insights",
                "dashboard.components.sidebar",
                "dashboard.components.enhanced_renderer"
            ]
            
            missing_components = []
            for component in required_components:
                try:
                    module = importlib.import_module(component)
                    self._record_result("dashboard", "pass", f"Component {component} is available")
                    print(colored(f"✓ Component {component} is available", "green"))
                except ImportError as e:
                    missing_components.append(component)
                    self._record_result("dashboard", "warning", f"Component {component} is missing: {str(e)}")
                    print(colored(f"⚠️ Component {component} is missing: {str(e)}", "yellow"))
            
            # Check for missing UIElements class in ui_elements module
            try:
                ui_elements_module = importlib.import_module("dashboard.components.ui_elements")
                if not hasattr(ui_elements_module, "UIElements"):
                    self._record_result("dashboard", "warning", "Missing UIElements class in ui_elements module")
                    print(colored("⚠️ Missing UIElements class in ui_elements module", "yellow"))
            except ImportError:
                pass  # Already recorded above
            
        else:
            self._record_result("dashboard", "fail", "Dashboard is not running")
            print(colored("❌ Dashboard is not running", "red"))

    def check_models(self) -> None:
        """Check if all required ML models are available and loaded correctly."""
        print(colored("\n🧠 Models Check", "yellow", attrs=["bold"]))
        
        # Check for model files
        model_dir = os.path.join(PROJECT_ROOT, "models", "trained")
        if not os.path.exists(model_dir):
            self._record_result("models", "fail", "Models directory not found")
            print(colored("❌ Models directory not found", "red"))
            return
        
        required_models = [
            "predictor_model.joblib",
            "predictor_model_xgb.json"
        ]
        
        missing_models = []
        for model_file in required_models:
            model_path = os.path.join(model_dir, model_file)
            if os.path.exists(model_path):
                self._record_result("models", "pass", f"Model file {model_file} exists")
                print(colored(f"✓ Model file {model_file} exists", "green"))
                
                # Check file size to ensure it's not empty/corrupt
                file_size = os.path.getsize(model_path)
                if file_size < 1000:  # Less than 1KB is suspicious
                    self._record_result("models", "warning", f"Model file {model_file} may be corrupt (size: {file_size} bytes)")
                    print(colored(f"⚠️ Model file {model_file} may be corrupt (size: {file_size} bytes)", "yellow"))
            else:
                missing_models.append(model_file)
                self._record_result("models", "warning", f"Model file {model_file} is missing")
                print(colored(f"⚠️ Model file {model_file} is missing", "yellow"))
        
        if missing_models:
            self._record_result("models", "warning", f"Some model files are missing: {', '.join(missing_models)}")
        else:
            # Try loading the models
            try:
                from utils.model_singleton import ModelSingleton
                model_manager = ModelSingleton.get_instance()
                self._record_result("models", "pass", "ModelSingleton can be initialized")
                print(colored("✓ ModelSingleton can be initialized", "green"))
            except Exception as e:
                self._record_result("models", "warning", f"Failed to initialize ModelSingleton: {str(e)}")
                print(colored(f"⚠️ Failed to initialize ModelSingleton: {str(e)}", "yellow"))
                
    def check_static_assets(self) -> None:
        """Check if all required static assets are available."""
        print(colored("\n🖼️ Static Assets Check", "yellow", attrs=["bold"]))
        
        # Check for CSS files
        css_path = os.path.join(PROJECT_ROOT, "dashboard", "static", "enhanced_dashboard_layout.css")
        if os.path.exists(css_path):
            self._record_result("static_assets", "pass", "Enhanced dashboard CSS exists")
            print(colored("✓ Enhanced dashboard CSS exists", "green"))
        else:
            self._record_result("static_assets", "warning", "Enhanced dashboard CSS is missing")
            print(colored("⚠️ Enhanced dashboard CSS is missing", "yellow"))
            
        # Check for banner image
        banner_path = os.path.join(PROJECT_ROOT, "dashboard", "static", "images", "GoalDiggers_banner.png")
        if os.path.exists(banner_path):
            self._record_result("static_assets", "pass", "Banner image exists")
            print(colored("✓ Banner image exists", "green"))
        else:
            self._record_result("static_assets", "warning", "Banner image is missing")
            print(colored("⚠️ Banner image is missing", "yellow"))
            
        # Check for fallback logo image
        logo_path = os.path.join(PROJECT_ROOT, "GoalDiggers_logo.png")
        if os.path.exists(logo_path):
            self._record_result("static_assets", "pass", "Fallback logo exists")
            print(colored("✓ Fallback logo exists", "green"))
        else:
            self._record_result("static_assets", "warning", "Fallback logo is missing")
            print(colored("⚠️ Fallback logo is missing", "yellow"))

    def check_component_compatibility(self) -> None:
        """Check compatibility between components."""
        print(colored("\n🔄 Component Compatibility Check", "yellow", attrs=["bold"]))
        
        # Check if render_banner function exists in ui_elements
        try:
            from dashboard.components.ui_elements import render_banner
            self._record_result("component_compatibility", "pass", "render_banner function exists in ui_elements")
            print(colored("✓ render_banner function exists in ui_elements", "green"))
        except ImportError:
            self._record_result("component_compatibility", "warning", "Cannot import ui_elements module")
            print(colored("⚠️ Cannot import ui_elements module", "yellow"))
        except AttributeError:
            self._record_result("component_compatibility", "fail", "render_banner function is missing from ui_elements")
            print(colored("❌ render_banner function is missing from ui_elements", "red"))
        
        # Check if UIElements class exists
        try:
            from dashboard.components.ui_elements import UIElements
            self._record_result("component_compatibility", "pass", "UIElements class exists")
            print(colored("✓ UIElements class exists", "green"))
        except ImportError:
            pass  # Already captured above
        except AttributeError:
            self._record_result("component_compatibility", "warning", "UIElements class is missing")
            print(colored("⚠️ UIElements class is missing from ui_elements module", "yellow"))
            print(colored("  This will cause errors in modern_interactive_dashboard.py", "yellow"))
            
            # Offer to fix the issue
            print(colored("\n🔧 Suggested Fix: Create UIElements class in ui_elements.py", "cyan"))

    def _is_service_running(self, host: str, port: int) -> bool:
        """Check if a service is running on the specified host and port."""
        try:
            if host == "localhost" or host == "127.0.0.1":
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            else:
                # For non-localhost, try HTTP request
                requests.get(f"http://{host}:{port}", timeout=2)
                return True
        except:
            return False

    def _record_result(self, component: str, status: str, message: str) -> None:
        """Record a check result."""
        self.results[component]["details"].append({"status": status, "message": message})
        
        # Update component status based on worst result
        if status == "fail":
            self.results[component]["status"] = "fail"
        elif status == "warning" and self.results[component]["status"] != "fail":
            self.results[component]["status"] = "warning"
        elif self.results[component]["status"] == "pending":
            self.results[component]["status"] = "pass"

    def _colorize_status(self, status: str) -> str:
        """Return a colorized string for the given status."""
        if status == "pass" or status == "operational":
            return colored(status.upper(), "green", attrs=["bold"])
        elif status == "warning" or status == "degraded":
            return colored(status.upper(), "yellow", attrs=["bold"])
        else:
            return colored(status.upper(), "red", attrs=["bold"])

    def _print_failed_components(self) -> None:
        """Print details of failed components."""
        for component, results in self.results.items():
            if results["status"] == "fail":
                print(colored(f"\n❌ {component.upper()} FAILURES:", "red"))
                for detail in results["details"]:
                    if detail["status"] == "fail":
                        print(colored(f"  - {detail['message']}", "red"))

    def _print_warning_components(self) -> None:
        """Print details of components with warnings."""
        for component, results in self.results.items():
            if results["status"] == "warning":
                print(colored(f"\n⚠️ {component.upper()} WARNINGS:", "yellow"))
                for detail in results["details"]:
                    if detail["status"] == "warning":
                        print(colored(f"  - {detail['message']}", "yellow"))


def create_ui_elements_class():
    """Create the UIElements class in ui_elements.py."""
    try:
        ui_elements_path = os.path.join(PROJECT_ROOT, "dashboard", "components", "ui_elements.py")
        
        # Read current file content
        with open(ui_elements_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if UIElements class already exists
        if "class UIElements" in content:
            print(colored("UIElements class already exists in ui_elements.py", "green"))
            return
        
        # Add UIElements class
        new_content = content + """

class UIElements:
    \"\"\"
    UI Elements class that provides a unified interface to access UI components.
    This class wraps the standalone UI element functions for backward compatibility.
    \"\"\"
    
    @staticmethod
    def render_banner(width="100%", container=None):
        \"\"\"
        Renders the GoalDiggers banner at the top of the page.
        Wrapper for the standalone render_banner function.
        \"\"\"
        return render_banner(width=width, container=container)
    
    @staticmethod
    def create_metric_card(title, value, description=None, delta=None, delta_color="normal", tooltip=None, icon=None, trend=None, help_text=None):
        \"\"\"
        Create a metric card for displaying KPIs and important metrics.
        Wrapper for the standalone create_metric_card function.
        \"\"\"
        return create_metric_card(
            title=title, value=value, description=description, delta=delta,
            delta_color=delta_color, tooltip=tooltip, icon=icon,
            trend=trend, help_text=help_text
        )
    
    @staticmethod
    def create_info_card(title, content, icon=None, color=None, border=True):
        \"\"\"
        Create an information card with a title and content.
        Wrapper for the standalone create_info_card function.
        \"\"\"
        return create_info_card(title=title, content=content, icon=icon, color=color, border=border)
    
    @staticmethod
    def styled_card(content, padding="20px", margin="10px 0", border_radius="12px", card_type="default", hover_effect=True, animate=True):
        \"\"\"
        Create a styled card component with custom content and multiple design options.
        Wrapper for the standalone styled_card function.
        \"\"\"
        return styled_card(
            content=content, padding=padding, margin=margin, 
            border_radius=border_radius, card_type=card_type, 
            hover_effect=hover_effect, animate=animate
        )
    
    @staticmethod
    def badge(text, badge_type="default", tooltip=None):
        \"\"\"
        Creates a simple styled badge using Streamlit markdown.
        Wrapper for the standalone badge function.
        \"\"\"
        return badge(text=text, badge_type=badge_type, tooltip=tooltip)
    
    @staticmethod
    def header(text, level=1, color=None, align="left", margin="20px 0 10px 0", icon=None,
               subtitle=None, divider=True, accent_bar=True, animation='slide'):
        \"\"\"
        Create a modern styled header component.
        Wrapper for the standalone header function.
        \"\"\"
        return header(
            text=text, level=level, color=color, align=align, margin=margin,
            icon=icon, subtitle=subtitle, divider=divider, accent_bar=accent_bar,
            animation=animation
        )
    
    @staticmethod
    def progress_indicator(value, max_value=100, color=None, style="bar", label=None, size="medium", show_percentage=True):
        \"\"\"
        Create an enhanced progress indicator with multiple design options.
        Wrapper for the standalone progress_indicator function.
        \"\"\"
        return progress_indicator(
            value=value, max_value=max_value, color=color, style=style,
            label=label, size=size, show_percentage=show_percentage
        )
    
    @staticmethod
    def info_tooltip(content, icon="ℹ️", placement="top", style="icon", max_width="300px", color=None):
        \"\"\"
        Create an enhanced tooltip component with multiple design options.
        Wrapper for the standalone info_tooltip function.
        \"\"\"
        return info_tooltip(
            content=content, icon=icon, placement=placement, 
            style=style, max_width=max_width, color=color
        )
"""
        
        # Write updated content
        with open(ui_elements_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print(colored("✅ Successfully created UIElements class in ui_elements.py", "green"))
    except Exception as e:
        print(colored(f"❌ Failed to create UIElements class: {e}", "red"))


if __name__ == "__main__":
    # Check if we should fix the UIElements issue
    if len(sys.argv) > 1 and sys.argv[1] == "--fix":
        print(colored("🔧 Fixing UIElements class issue...", "cyan"))
        create_ui_elements_class()
        sys.exit(0)
        
    # Run health check
    health_checker = SystemHealthCheck()
    results = health_checker.run_all_checks()
    
    # Determine if any action is needed
    if any(results[component]["status"] == "fail" for component in results):
        print(colored("\n🔧 AUTOMATIC FIXES AVAILABLE:", "cyan"))
        
        # Check if UIElements class is missing
        ui_elements_missing = False
        for detail in results["component_compatibility"]["details"]:
            if "UIElements class is missing" in detail["message"] and detail["status"] in ["warning", "fail"]:
                ui_elements_missing = True
        
        if ui_elements_missing:
            print("1. Create UIElements class in ui_elements.py")
            print(colored("\nTo apply fixes, run:", "cyan"))
            print(colored(f"python {os.path.basename(__file__)} --fix", "yellow"))
