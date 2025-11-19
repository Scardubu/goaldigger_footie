#!/usr/bin/env python3
"""
Memory Optimization Initializer for GoalDiggers Platform

This script configures and initializes memory optimization systems 
for the GoalDiggers football betting platform to ensure efficient memory usage.

It configures both high-level memory management (memory_optimization_manager) 
and low-level optimization (memory_optimizer) to ensure memory usage stays 
within target limits.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

# Setup logging
if not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        from utils.logging_config import configure_logging  # type: ignore
        configure_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_memory_optimization(
    target_memory_mb: float = 400.0,
    warning_threshold_mb: float = 350.0,
    critical_threshold_mb: float = 380.0,
    enable_monitoring: bool = True,
    enable_automatic_cleanup: bool = True
) -> Dict[str, Any]:
    """
    Initialize memory optimization systems for the application.
    
    Args:
        target_memory_mb: Target memory usage in MB
        warning_threshold_mb: Warning threshold in MB
        critical_threshold_mb: Critical threshold in MB
        enable_monitoring: Whether to enable memory monitoring
        enable_automatic_cleanup: Whether to enable automatic cleanup
        
    Returns:
        Dictionary with initialization status information
    """
    result = {
        "success": False,
        "manager_initialized": False,
        "optimizer_initialized": False,
        "error": None
    }
    
    logger.info(f"Initializing memory optimization (target: {target_memory_mb}MB)")
    
    # Set environment variables
    os.environ["ENABLE_MEMORY_OPTIMIZATION"] = "true"
    os.environ["MEMORY_TARGET_MB"] = str(target_memory_mb)
    os.environ["MEMORY_WARNING_THRESHOLD_MB"] = str(warning_threshold_mb)
    os.environ["MEMORY_CRITICAL_THRESHOLD_MB"] = str(critical_threshold_mb)
    
    # Try to initialize memory optimization manager (preferred)
    try:
        logger.info("Initializing memory optimization manager...")
        from dashboard.optimizations.memory_optimization_manager import (
            MemoryConfig,
            MemoryOptimizationManager,
            get_memory_manager,
        )

        # Create configuration
        config = MemoryConfig(
            target_memory_mb=target_memory_mb,
            warning_threshold_mb=warning_threshold_mb,
            critical_threshold_mb=critical_threshold_mb,
            cleanup_interval_s=30.0,
            enable_monitoring=enable_monitoring,
            enable_automatic_cleanup=enable_automatic_cleanup
        )
        
        # Get manager instance
        memory_manager = get_memory_manager(config)
        
        # Register critical components (app, database, models)
        memory_manager.register_component(
            name="app_core",
            component="app_core",
            priority=memory_manager.MemoryPriority.CRITICAL
        )
        
        memory_manager.register_component(
            name="database",
            component="database",
            priority=memory_manager.MemoryPriority.CRITICAL
        )
        
        memory_manager.register_component(
            name="prediction_models",
            component="prediction_models",
            priority=memory_manager.MemoryPriority.HIGH
        )
        
        # Start monitoring if enabled
        if enable_monitoring:
            memory_manager.start_monitoring()
        
        result["manager_initialized"] = True
        logger.info("Memory optimization manager initialized successfully")
        
        # If manager initialization succeeded, mark as successful
        result["success"] = True
        
    except ImportError:
        logger.warning("Memory optimization manager not available, falling back to memory optimizer")
    except Exception as e:
        logger.error(f"Error initializing memory optimization manager: {e}")
        result["error"] = str(e)
    
    # If manager failed, try to initialize memory optimizer as fallback
    if not result["manager_initialized"]:
        try:
            logger.info("Initializing memory optimizer (fallback)...")
            from utils.memory_optimizer import memory_optimizer

            # Configure optimizer
            memory_optimizer.memory_threshold_bytes = int(target_memory_mb * 1024 * 1024)
            
            # Register cleanup callbacks
            def cleanup_session_state():
                import streamlit as st
                if hasattr(st, "session_state"):
                    # Store essential data that shouldn't be cleared
                    essential_keys = ["user_id", "logged_in", "theme"]
                    essential_data = {}
                    for key in essential_keys:
                        if key in st.session_state:
                            essential_data[key] = st.session_state[key]
                    
                    # Clear large data objects
                    for key in list(st.session_state.keys()):
                        if key not in essential_keys:
                            del st.session_state[key]
                    
                    # Restore essential data
                    for key, value in essential_data.items():
                        st.session_state[key] = value
            
            # Register cleanup callbacks
            memory_optimizer.register_cleanup_callback(cleanup_session_state)
            
            # Start monitoring
            if enable_monitoring:
                memory_optimizer.start_monitoring()
            
            result["optimizer_initialized"] = True
            logger.info("Memory optimizer initialized successfully")
            
            # If optimizer initialization succeeded, mark as successful
            result["success"] = True
            
        except ImportError:
            logger.warning("Memory optimizer not available")
        except Exception as e:
            logger.error(f"Error initializing memory optimizer: {e}")
            result["error"] = str(e) if result["error"] is None else f"{result['error']}; {e}"
    
    # Final result
    if result["success"]:
        logger.info("Memory optimization initialized successfully")
    else:
        logger.warning("Memory optimization initialization failed")
    
    return result

def get_memory_status() -> Dict[str, Any]:
    """
    Get current memory status from the optimization system.
    
    Returns:
        Dictionary with memory status information
    """
    result = {
        "available": False,
        "current_memory_mb": None,
        "target_memory_mb": None,
        "status": "unknown"
    }
    
    # Try memory manager first
    try:
        from dashboard.optimizations.memory_optimization_manager import (
            get_memory_manager,
        )
        memory_manager = get_memory_manager()
        status = memory_manager.check_memory_status()
        
        result["available"] = True
        result["current_memory_mb"] = status["current_memory_mb"]
        result["target_memory_mb"] = status["target_memory_mb"]
        result["status"] = status["status"]
        result["memory_utilization"] = status["memory_utilization"]
        result["component_count"] = status["component_count"]
        
        return result
        
    except ImportError:
        logger.debug("Memory optimization manager not available, trying memory optimizer")
    except Exception as e:
        logger.warning(f"Error getting memory status from manager: {e}")
    
    # Try memory optimizer as fallback
    try:
        from utils.memory_optimizer import memory_optimizer
        memory_usage = memory_optimizer.get_memory_usage()
        
        result["available"] = True
        result["current_memory_mb"] = memory_usage["rss_mb"]
        result["target_memory_mb"] = memory_optimizer.memory_threshold_bytes / 1024 / 1024
        result["status"] = "CRITICAL" if memory_usage["rss_mb"] > result["target_memory_mb"] else "GOOD"
        result["memory_utilization"] = (memory_usage["rss_mb"] / result["target_memory_mb"]) * 100
        
        return result
        
    except ImportError:
        logger.warning("Neither memory optimization manager nor memory optimizer available")
    except Exception as e:
        logger.warning(f"Error getting memory status from optimizer: {e}")
    
    return result

def force_memory_cleanup() -> Dict[str, Any]:
    """
    Force an immediate memory cleanup.
    
    Returns:
        Dictionary with cleanup results
    """
    result = {
        "success": False,
        "memory_freed_mb": 0,
        "error": None
    }
    
    # Try memory manager first
    try:
        from dashboard.optimizations.memory_optimization_manager import (
            get_memory_manager,
        )
        memory_manager = get_memory_manager()
        
        before_status = memory_manager.check_memory_status()
        cleanup_result = memory_manager.cleanup_components(force=True)
        after_status = memory_manager.check_memory_status()
        
        result["success"] = True
        result["memory_freed_mb"] = before_status["current_memory_mb"] - after_status["current_memory_mb"]
        result["components_cleaned"] = cleanup_result["components_cleaned"]
        
        return result
        
    except ImportError:
        logger.debug("Memory optimization manager not available, trying memory optimizer")
    except Exception as e:
        logger.warning(f"Error forcing cleanup with manager: {e}")
        result["error"] = str(e)
    
    # Try memory optimizer as fallback
    try:
        from utils.memory_optimizer import memory_optimizer
        
        before_usage = memory_optimizer.get_memory_usage()["rss_mb"]
        cleanup_stats = memory_optimizer.force_cleanup()
        after_usage = memory_optimizer.get_memory_usage()["rss_mb"]
        
        result["success"] = True
        result["memory_freed_mb"] = before_usage - after_usage
        result["objects_collected"] = cleanup_stats["objects_collected"]
        
        return result
        
    except ImportError:
        logger.warning("Neither memory optimization manager nor memory optimizer available")
    except Exception as e:
        logger.warning(f"Error forcing cleanup with optimizer: {e}")
        result["error"] = str(e) if result["error"] is None else f"{result['error']}; {e}"
    
    return result

if __name__ == "__main__":
    # Initialize memory optimization when script is run directly
    result = initialize_memory_optimization()
    
    if result["success"]:
        print("Memory optimization initialized successfully")
        sys.exit(0)
    else:
        print(f"Memory optimization initialization failed: {result['error']}")
        sys.exit(1)
