#!/usr/bin/env python3
"""
API Server Startup Script

This script provides a convenient way to start and manage the GoalDiggers API server.
It includes error handling, automatic port selection, and status reporting.
"""

import atexit
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("api_startup")

def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def check_port_availability(port):
    """Check if the specified port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def find_available_port(start_port=5000, max_attempts=10):
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if check_port_availability(port):
            return port
    raise RuntimeError(f"Could not find available port after {max_attempts} attempts")

def start_api_server(port=None, host="0.0.0.0", log_file=None):
    """
    Start the API server as a subprocess.
    
    Args:
        port: Port number (default: auto-select)
        host: Host to bind to (default: 0.0.0.0)
        log_file: File to redirect server logs to
    
    Returns:
        Tuple of (process, port)
    """
    # Add project root to path
    project_root = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(project_root))
    
    # Determine port
    if port is None:
        port = find_available_port()
    
    # Set up logs directory
    logs_dir = project_root / "logs"
    ensure_directory_exists(logs_dir)
    
    # Set up log file
    if log_file is None:
        log_file = logs_dir / f"api_server_{port}.log"
    
    logger.info(f"Starting API server on {host}:{port}, logs will be written to {log_file}")
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["API_PORT"] = str(port)
    env["API_HOST"] = host
    
    # Start the server process
    server_script = Path(__file__).parent / "server.py"
    
    if sys.platform == "win32":
        # On Windows, use python executable directly and CREATE_NEW_PROCESS_GROUP flag
        process = subprocess.Popen(
            [sys.executable, str(server_script)],
            env=env,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        # On Unix-like systems
        process = subprocess.Popen(
            [sys.executable, str(server_script)],
            env=env,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    # Register cleanup function
    def cleanup():
        if process.poll() is None:
            logger.info("Stopping API server...")
            if sys.platform == "win32":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            process.wait(timeout=5)
    
    atexit.register(cleanup)
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Check if the process is still running
    if process.poll() is not None:
        logger.error(f"Server process exited with code {process.returncode}")
        raise RuntimeError(f"Server failed to start. Check the logs at {log_file}")
    
    return process, port

def create_startup_batch_file(port=5000):
    """Create a batch file for easy API server startup on Windows."""
    project_root = Path(__file__).parent.parent.absolute()
    batch_file = project_root / "start_api_server.bat"
    
    with open(batch_file, "w") as f:
        f.write(f"@echo off\n")
        f.write("echo Starting GoalDiggers API Server...\n")
        f.write(f'set API_PORT={port}\n')
        f.write(f'"{sys.executable}" "{Path(__file__)}" --port {port}\n')
        f.write("pause\n")
    
    logger.info(f"Created startup batch file at {batch_file}")
    return batch_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the GoalDiggers API server")
    parser.add_argument("--port", type=int, default=None, help="Port to use (default: auto-select)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--create-batch", action="store_true", help="Create a batch file for easy startup")
    
    args = parser.parse_args()
    
    if args.create_batch:
        create_startup_batch_file(args.port or 5000)
        return
    
    try:
        process, port = start_api_server(args.port, args.host, args.log_file)
        logger.info(f"API server running on http://{args.host}:{port}")
        logger.info("Press Ctrl+C to stop the server")
        
        # Create a simple file to indicate the server is running
        project_root = Path(__file__).parent.parent.absolute()
        with open(project_root / "api_server_status.txt", "w") as f:
            f.write(f"running\n{port}\n{args.host}\n")
        
        # Keep the main process running
        while True:
            if process.poll() is not None:
                logger.error(f"Server process exited unexpectedly with code {process.returncode}")
                break
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        # Remove the status file
        try:
            os.remove(project_root / "api_server_status.txt")
        except (FileNotFoundError, NameError):
            pass

if __name__ == "__main__":
    main()
