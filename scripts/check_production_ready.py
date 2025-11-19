#!/usr/bin/env python3
"""
Quick production readiness checker.

Performs essential checks to verify the system is ready for production use.
This is a lightweight version of the full validation script.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import settings
from utils.logging_config import configure_logging

# Configure logging
configure_logging()
import logging

logger = logging.getLogger(__name__)


def check_critical_files():
    """Check critical files exist."""
    critical_files = [
        "config/settings.py",
        "utils/logging_config.py", 
        "utils/error_handling.py",
        "health_check.py",
        "PRODUCTION_READINESS_CHECKLIST.md"
    ]
    
    missing = [f for f in critical_files if not Path(f).exists()]
    if missing:
        logger.error(f"Missing critical files: {missing}")
        return False
    return True


def check_environment():
    """Check environment configuration."""
    try:
        # Test settings loading
        assert settings.DB_PATH
        assert settings.ENV
        logger.info(f"Environment: {settings.ENV}")
        logger.info(f"Database: {settings.DB_PATH}")
        logger.info(f"Calibration enabled: {settings.ENABLE_CALIBRATION}")
        return True
    except Exception as e:
        logger.error(f"Environment check failed: {e}")
        return False


def check_health():
    """Check system health."""
    try:
        from health_check import get_system_health
        health = get_system_health()
        
        if health.get("status") == "healthy":
            logger.info("✅ System health check passed")
            return True
        else:
            logger.warning(f"⚠️ System health issues: {health}")
            return False
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False


def main():
    """Run quick readiness check."""
    logger.info("🚀 Running production readiness check...")
    
    checks = [
        ("Critical Files", check_critical_files),
        ("Environment", check_environment), 
        ("System Health", check_health)
    ]
    
    all_passed = True
    
    for name, check_func in checks:
        logger.info(f"Checking {name}...")
        if not check_func():
            all_passed = False
        else:
            logger.info(f"✅ {name} OK")
    
    if all_passed:
        logger.info("🎉 System is ready for production!")
        return 0
    else:
        logger.error("❌ Production readiness checks failed")
        logger.info("Run 'python scripts/validate_deployment.py' for detailed analysis")
        return 1


if __name__ == "__main__":
    sys.exit(main())