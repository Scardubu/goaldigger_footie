import importlib
import os
import sqlite3
import time
import traceback
from pathlib import Path
# import requests  # For MCP check (MCP server check disabled)
import yaml
import streamlit as st
import logging
import sys
import pandas as pd
import asyncio
from typing import Dict, Any, List, Optional, Tuple

from dashboard.error_log import ErrorLog
from dashboard.error_recovery import error_recovery_manager
from dashboard.optimizations.integration_manager import integration_manager
from dashboard.optimizations.data_pipeline_monitor import DataPipelineMonitor

# --- Configuration ---
# Define paths relative to the expected CWD (project root: c:/Users/scart/goaldiggers)
PROJECT_ROOT = Path("c:/Users/scart/goaldiggers") # Use absolute path for reliability
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "football.db"
MODEL_PATH = PROJECT_ROOT / "models" / "predictor_model.joblib" # Example path, adjust if needed
CONFIG_DIR = PROJECT_ROOT / "config"
REQUIRED_CONFIG_FILES = ["config.yaml", "paths.yaml", "api_endpoints.yaml"] # Add others as needed
REQUIRED_MODULES = [
    'yaml', 'dotenv', 'pandas', 'numpy', 'requests', 'streamlit',
    'sqlalchemy', 'psutil', 'plotly', 'scipy', 'sklearn', # Add other critical libs
    'nest_asyncio'
]
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:3000") # Get from env or default

# --- Check Functions ---

def check_python_modules():
    """Checks if required Python modules can be imported."""
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        return False, f"Missing Python modules: {', '.join(missing)}. Please run 'pip install -r requirements.txt'."
    return True, "All required Python modules are installed."

def check_database_connection():
    """Checks if the database file exists and a basic connection can be established."""
    if not DB_PATH.exists():
        return False, f"Database file not found at {DB_PATH}."
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        # Optional: Perform a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
        cursor.fetchone()
        conn.close()
        return True, f"Database connection successful ({DB_PATH})."
    except sqlite3.Error as e:
        return False, f"Database connection/query failed: {e} ({DB_PATH})."
    except Exception as e:
        return False, f"Unexpected error connecting to database: {e} ({DB_PATH})."

def check_model_file():
    """Checks if the primary predictor model file exists."""
    if not MODEL_PATH.exists():
        return False, f"Predictor model file not found at {MODEL_PATH}."
    return True, f"Predictor model file found ({MODEL_PATH})."

def check_config_files():
    """Checks if required configuration files exist."""
    missing_configs = []
    for fname in REQUIRED_CONFIG_FILES:
        fpath = CONFIG_DIR / fname
        if not fpath.exists():
            missing_configs.append(fname)
        else:
            # Optional: Try parsing YAML to catch syntax errors early
            try:
                with open(fpath, 'r') as yf:
                    yaml.safe_load(yf)
            except yaml.YAMLError as e:
                 return False, f"Config file '{fname}' has YAML syntax error: {e}"
            except Exception as e:
                 return False, f"Error reading config file '{fname}': {e}"

    if missing_configs:
        return False, f"Missing configuration files in {CONFIG_DIR}: {', '.join(missing_configs)}."
    return True, "All required configuration files found and readable."

# def check_mcp_server():
#     """Checks the health of the MCP server.""" (MCP server check disabled)
#     health_url = f"{MCP_SERVER_URL.rstrip('/')}/health"
#     try:
#         response = requests.get(health_url, timeout=5) # 5-second timeout
#         if response.status_code == 200:
#             # Optional: Check response content if health endpoint provides details
#             # data = response.json()
#             # if data.get("status") == "OK":
#             #    return True, f"MCP server is healthy ({health_url})."
#             # else:
#             #    return False, f"MCP server health check returned status: {data.get('status', 'Unknown')}"
#             return True, f"MCP server responded successfully ({health_url}, Status: {response.status_code})."
#         else:
#             return False, f"MCP server health check failed ({health_url}, Status: {response.status_code})."
#     except requests.exceptions.RequestException as e:
#         return False, f"Could not connect to MCP server at {health_url}: {e}"
#     except Exception as e:
#         return False, f"Unexpected error checking MCP server: {e}"

# --- Main Diagnostic Function ---

def run_startup_diagnostics():
    """Runs all diagnostic checks and returns a summary."""
    results = {}
    start_time = time.time()

    checks_to_run = {
        "Python Modules": check_python_modules,
        "Config Files": check_config_files,
        "Database Connection": check_database_connection,
        "Predictor Model File": check_model_file,
        # "MCP Server": check_mcp_server,  # MCP server check disabled
        # Add more checks here if needed
    }

    overall_status = True
    for name, check_func in checks_to_run.items():
        try:
            status, message = check_func()
            results[name] = {"status": "✅ OK" if status else "❌ ERROR", "message": message}
            if not status:
                overall_status = False
        except Exception as e:
            # Catch errors within a check function itself
            error_trace = traceback.format_exc()
            results[name] = {"status": "❌ ERROR", "message": f"Check failed unexpectedly: {e}\n{error_trace}"}
            overall_status = False

    end_time = time.time()
    results["Overall Status"] = "✅ All Critical Checks Passed" if overall_status else "❌ One or more checks failed"
    results["Duration (seconds)"] = round(end_time - start_time, 2)

    return results

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    print("Running Startup Diagnostics...")
    diagnostic_results = run_startup_diagnostics()
    import json
    print(json.dumps(diagnostic_results, indent=2))

    # Example of how to check the overall status
    if "❌" in diagnostic_results.get("Overall Status", "❌"):
        print("\nStartup diagnostics failed. Please review the errors above.")
    else:
        print("\nStartup diagnostics passed.")
