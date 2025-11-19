"""
PostgreSQL Configuration Validator

This module provides utilities for validating PostgreSQL configuration,
testing connections, and providing detailed diagnostics for connection issues.
"""

import logging
import os
import socket
import time
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# Define required and optional PostgreSQL environment variables
REQUIRED_PG_VARS = [
    # At least one of these connection methods must be provided
    ["DATABASE_URL", "POSTGRES_HOST"]
]

OPTIONAL_PG_VARS = [
    "POSTGRES_PORT",
    "POSTGRES_DB", "POSTGRES_DATABASE",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_SSLMODE",
    "DATABASE_SSLMODE",
    "POSTGRES_SSLROOTCERT",
    "POSTGRES_SSLCERT",
    "POSTGRES_SSLKEY",
    "POSTGRES_TARGET_SESSION_ATTRS",
    "POSTGRES_APPLICATION_NAME",
    "POSTGRES_OPTIONS",
    "POSTGRES_CONNECT_TIMEOUT"
]


def validate_postgres_config() -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that the necessary PostgreSQL configuration is available.
    
    Returns:
        Tuple containing:
            - Boolean indicating if config is valid
            - Dictionary with validation details
    """
    # Check if any of the required variable groups are fully present
    config_provided = False
    missing_vars = []
    present_vars = []
    
    # Check if DATABASE_URL is provided (simplest case)
    if os.getenv("DATABASE_URL") and "postgresql" in os.getenv("DATABASE_URL", ""):
        config_provided = True
        present_vars.append("DATABASE_URL")
    
    # Check if discrete variables are provided
    if os.getenv("POSTGRES_HOST") or os.getenv("PGHOST"):
        host_present = True
        present_vars.append("POSTGRES_HOST/PGHOST")
        
        # These are required if using discrete variables
        for var in ["POSTGRES_DB", "POSTGRES_DATABASE", "PGDATABASE"]:
            if os.getenv(var):
                present_vars.append(var)
                break
        else:
            missing_vars.append("POSTGRES_DB/POSTGRES_DATABASE/PGDATABASE")
            
        for var in ["POSTGRES_USER", "PGUSER"]:
            if os.getenv(var):
                present_vars.append(var)
                break
        else:
            missing_vars.append("POSTGRES_USER/PGUSER")
            
        if not missing_vars:
            config_provided = True
    
    # Check which optional variables are set
    optional_present = []
    for var in OPTIONAL_PG_VARS:
        if os.getenv(var):
            optional_present.append(var)
    
    return config_provided, {
        "config_valid": config_provided,
        "missing_required": missing_vars,
        "present_required": present_vars,
        "present_optional": optional_present,
    }


def test_postgres_connection(uri: Optional[str] = None, 
                            connect_args: Optional[Dict[str, Any]] = None, 
                            timeout_seconds: int = 10) -> Tuple[bool, Dict[str, Any]]:
    """
    Test PostgreSQL connection with the configured parameters.
    
    Args:
        uri: Optional PostgreSQL connection URI
        connect_args: Optional connection arguments
        timeout_seconds: Connection timeout in seconds
        
    Returns:
        Tuple containing:
            - Boolean indicating if connection was successful
            - Dictionary with connection test details
    """
    result = {
        "success": False,
        "error": None,
        "latency_ms": None,
        "server_version": None,
        "connection_info": None
    }
    
    # Use provided URI or build from environment variables
    if not uri:
        from database.db_manager import _build_postgres_uri_from_env
        uri, env_connect_args = _build_postgres_uri_from_env()
        if not uri:
            result["error"] = "No PostgreSQL URI available from environment variables"
            return False, result
        if connect_args is None:
            connect_args = env_connect_args
    
    # Measure connection latency
    start_time = time.time()
    
    try:
        # Create engine with timeout
        if connect_args is None:
            connect_args = {}
        connect_args.setdefault("connect_timeout", timeout_seconds)
        
        engine = create_engine(uri, connect_args=connect_args)
        
        # Test connection
        with engine.connect() as connection:
            # Get PostgreSQL version
            version_result = connection.execute(text("SELECT version()")).fetchone()
            server_version = version_result[0] if version_result else "Unknown"
            
            # Get connection info
            conn_info = connection.execute(text("""
                SELECT current_database(), current_user, inet_server_addr(), 
                       inet_server_port(), pg_backend_pid()
            """)).fetchone()
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            result.update({
                "success": True,
                "latency_ms": latency_ms,
                "server_version": server_version,
                "connection_info": {
                    "database": conn_info[0],
                    "user": conn_info[1],
                    "server_addr": str(conn_info[2]),
                    "server_port": conn_info[3],
                    "backend_pid": conn_info[4]
                }
            })
            
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        result.update({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "latency_ms": latency_ms
        })
        
        # Add network diagnostics for connection failures
        if isinstance(e, (psycopg2.OperationalError, SQLAlchemyError)):
            result["diagnostics"] = perform_network_diagnostics(uri, connect_args)
    
    return result["success"], result


def perform_network_diagnostics(uri: str, connect_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Perform network diagnostics for PostgreSQL connection failures.
    
    Args:
        uri: PostgreSQL connection URI
        connect_args: Optional connection arguments
        
    Returns:
        Dictionary with network diagnostic results
    """
    from sqlalchemy.engine.url import make_url
    
    diagnostics = {
        "hostname": None,
        "port": None,
        "dns_resolution": None,
        "port_open": None,
        "network_reachable": None,
        "connect_args": connect_args
    }
    
    try:
        # Parse URI to get hostname and port
        parsed = make_url(uri)
        hostname = parsed.host
        port = parsed.port or 5432
        
        diagnostics["hostname"] = hostname
        diagnostics["port"] = port
        
        # DNS resolution check
        try:
            ip_address = socket.gethostbyname(hostname)
            diagnostics["dns_resolution"] = {
                "success": True,
                "ip_address": ip_address
            }
            
            # Port check
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((ip_address, port))
                sock.close()
                
                if result == 0:
                    diagnostics["port_open"] = True
                    diagnostics["network_reachable"] = True
                else:
                    diagnostics["port_open"] = False
                    diagnostics["network_reachable"] = False
                    diagnostics["port_error"] = f"Port {port} is closed (error code: {result})"
                    
            except Exception as e:
                diagnostics["port_open"] = False
                diagnostics["port_error"] = str(e)
                
        except socket.gaierror as e:
            diagnostics["dns_resolution"] = {
                "success": False,
                "error": str(e)
            }
            diagnostics["network_reachable"] = False
            
    except Exception as e:
        diagnostics["parse_error"] = str(e)
    
    return diagnostics


def verify_postgres_permissions(uri: Optional[str] = None, 
                               connect_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Verify that the PostgreSQL user has necessary permissions.
    
    Args:
        uri: Optional PostgreSQL connection URI
        connect_args: Optional connection arguments
        
    Returns:
        Dictionary with permission verification results
    """
    results = {
        "can_connect": False,
        "can_create_table": False,
        "can_insert": False,
        "can_select": False,
        "can_update": False,
        "can_delete": False,
        "error": None
    }
    
    # Use provided URI or build from environment variables
    if not uri:
        from database.db_manager import _build_postgres_uri_from_env
        uri, env_connect_args = _build_postgres_uri_from_env()
        if not uri:
            results["error"] = "No PostgreSQL URI available from environment variables"
            return results
        if connect_args is None:
            connect_args = env_connect_args
    
    try:
        # Create engine
        engine = create_engine(uri, connect_args=connect_args or {})
        
        # Test connection
        with engine.connect() as connection:
            results["can_connect"] = True
            
            # Test CREATE TABLE
            try:
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS __permission_test (
                        id SERIAL PRIMARY KEY,
                        test_data VARCHAR(255)
                    )
                """))
                results["can_create_table"] = True
                
                # Test INSERT
                try:
                    connection.execute(text("""
                        INSERT INTO __permission_test (test_data) VALUES ('test')
                    """))
                    connection.commit()
                    results["can_insert"] = True
                    
                    # Test SELECT
                    try:
                        connection.execute(text("SELECT * FROM __permission_test")).fetchall()
                        results["can_select"] = True
                        
                        # Test UPDATE
                        try:
                            connection.execute(text("""
                                UPDATE __permission_test SET test_data = 'updated'
                            """))
                            connection.commit()
                            results["can_update"] = True
                            
                            # Test DELETE
                            try:
                                connection.execute(text("DELETE FROM __permission_test"))
                                connection.commit()
                                results["can_delete"] = True
                                
                            except Exception as e:
                                results["delete_error"] = str(e)
                                
                        except Exception as e:
                            results["update_error"] = str(e)
                            
                    except Exception as e:
                        results["select_error"] = str(e)
                        
                except Exception as e:
                    results["insert_error"] = str(e)
                    
                # Clean up test table
                try:
                    connection.execute(text("DROP TABLE IF EXISTS __permission_test"))
                    connection.commit()
                except Exception:
                    pass
                    
            except Exception as e:
                results["create_table_error"] = str(e)
                
    except Exception as e:
        results["connection_error"] = str(e)
    
    return results


def get_postgres_config_status() -> Dict[str, Any]:
    """
    Get comprehensive PostgreSQL configuration status.
    
    Returns:
        Dictionary with PostgreSQL configuration status
    """
    # Check if config is valid
    is_valid, validation_details = validate_postgres_config()
    
    result = {
        "config_valid": is_valid,
        "validation_details": validation_details,
        "connection_test": None,
        "permissions": None
    }
    
    # Test connection if config is valid
    if is_valid:
        from database.db_manager import (
            _build_postgres_uri_from_env,
            _connect_args_for_postgres_from_env,
        )
        
        uri, connect_args = _build_postgres_uri_from_env()
        if not uri:
            uri = os.getenv("DATABASE_URL")
            connect_args = _connect_args_for_postgres_from_env()
            
        success, conn_result = test_postgres_connection(uri, connect_args)
        result["connection_test"] = conn_result
        
        # Verify permissions if connection was successful
        if success:
            result["permissions"] = verify_postgres_permissions(uri, connect_args)
    
    return result


def print_postgres_diagnostic_report(status: Optional[Dict[str, Any]] = None) -> None:
    """
    Print a diagnostic report for PostgreSQL configuration.
    
    Args:
        status: Optional pre-generated status, if None will be generated
    """
    if status is None:
        status = get_postgres_config_status()
    
    print("=" * 80)
    print("PostgreSQL Configuration Diagnostic Report")
    print("=" * 80)
    
    # Configuration validation
    print("\nConfiguration Validation:")
    print(f"  Valid Configuration: {status['config_valid']}")
    
    if 'validation_details' in status:
        details = status['validation_details']
        print("\n  Required Variables:")
        if details.get('present_required'):
            for var in details['present_required']:
                print(f"    ✓ {var}")
        
        if details.get('missing_required'):
            for var in details['missing_required']:
                print(f"    ✗ {var}")
        
        print("\n  Optional Variables:")
        if details.get('present_optional'):
            for var in details['present_optional']:
                print(f"    ✓ {var}")
    
    # Connection test
    if 'connection_test' in status and status['connection_test']:
        test = status['connection_test']
        print("\nConnection Test:")
        print(f"  Success: {test['success']}")
        if test['latency_ms']:
            print(f"  Latency: {test['latency_ms']:.2f}ms")
        
        if test['success']:
            print(f"  Server Version: {test['server_version']}")
            if test['connection_info']:
                info = test['connection_info']
                print("\n  Connection Information:")
                print(f"    Database: {info['database']}")
                print(f"    User: {info['user']}")
                print(f"    Server: {info['server_addr']}:{info['server_port']}")
                print(f"    Backend PID: {info['backend_pid']}")
        else:
            print(f"  Error: {test['error']}")
            if 'diagnostics' in test:
                diag = test['diagnostics']
                print("\n  Network Diagnostics:")
                print(f"    Hostname: {diag['hostname']}")
                print(f"    Port: {diag['port']}")
                
                if diag['dns_resolution']:
                    if diag['dns_resolution'].get('success'):
                        print(f"    DNS Resolution: Success ({diag['dns_resolution']['ip_address']})")
                    else:
                        print(f"    DNS Resolution: Failed ({diag['dns_resolution'].get('error')})")
                
                if diag['port_open'] is not None:
                    print(f"    Port Open: {diag['port_open']}")
                    if not diag['port_open'] and 'port_error' in diag:
                        print(f"      Error: {diag['port_error']}")
    
    # Permissions
    if 'permissions' in status and status['permissions']:
        perms = status['permissions']
        print("\nPermissions:")
        print(f"  Can Connect: {perms['can_connect']}")
        if perms['can_connect']:
            print(f"  Can Create Table: {perms['can_create_table']}")
            print(f"  Can Insert: {perms['can_insert']}")
            print(f"  Can Select: {perms['can_select']}")
            print(f"  Can Update: {perms['can_update']}")
            print(f"  Can Delete: {perms['can_delete']}")
    
    print("\nRecommendations:")
    if not status['config_valid']:
        print("  • Set required PostgreSQL environment variables")
        print("  • Create .env file based on .env.example")
    elif 'connection_test' in status and not status['connection_test'].get('success'):
        print("  • Verify PostgreSQL server is running")
        print("  • Check network connectivity to database server")
        print("  • Verify credentials and connection parameters")
    elif 'permissions' in status and not all([
        status['permissions'].get('can_connect'),
        status['permissions'].get('can_create_table'),
        status['permissions'].get('can_insert'),
        status['permissions'].get('can_select'),
        status['permissions'].get('can_update'),
        status['permissions'].get('can_delete')
    ]):
        print("  • Grant necessary permissions to database user")
    else:
        print("  • PostgreSQL configuration is valid and connection is working")
        print("  • Ensure appropriate connection pooling for production workloads")
        print("  • Consider implementing database migration management")
    
    print("=" * 80)


if __name__ == "__main__":
    # Run diagnostics when script is executed directly
    status = get_postgres_config_status()
    print_postgres_diagnostic_report(status)