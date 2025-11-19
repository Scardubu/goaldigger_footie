#!/usr/bin/env python3
"""
Deployment validation and environment setup script.

Validates environment configuration, dependencies, and system readiness
before deployment to production. Performs comprehensive pre-flight checks.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import settings
from utils.error_handling import safe_execute
from utils.logging_config import configure_logging

# Configure logging
configure_logging()
import logging

logger = logging.getLogger(__name__)


class DeploymentValidator:
    """Validates deployment readiness."""

    def __init__(self):
        self.validation_results: List[Dict] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_environment(self) -> bool:
        """Validate environment configuration."""
        logger.info("Validating environment configuration...")
        
        # Check required environment variables (they have defaults in settings)
        required_env_vars = [
            "APP_ENV",
            "DATA_DB_PATH", 
            "LOG_LEVEL"
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                # Check if settings provides a default
                if var == "APP_ENV" and settings.ENV:
                    continue
                elif var == "DATA_DB_PATH" and settings.DB_PATH:
                    continue
                elif var == "LOG_LEVEL" and settings.LOG_LEVEL:
                    continue
                missing_vars.append(var)
        
        if missing_vars:
            self.errors.append(f"Missing environment variables: {missing_vars}")
            return False
        
        # Validate settings loading
        try:
            assert settings.DB_PATH
            assert settings.METRICS_DIR
            assert settings.FRESHNESS_RUN_DIR
        except Exception as e:
            self.errors.append(f"Settings validation failed: {e}")
            return False
        
        logger.info("Environment configuration valid")
        return True

    def validate_dependencies(self) -> bool:
        """Validate all required dependencies are installed."""
        logger.info("Validating dependencies...")
        
        required_packages = [
            "streamlit",
            "pandas", 
            "numpy",
            "scikit-learn",
            "psutil"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.errors.append(f"Missing required packages: {missing_packages}")
            return False
        
        logger.info("All dependencies validated")
        return True

    def validate_file_structure(self) -> bool:
        """Validate required file structure exists."""
        logger.info("Validating file structure...")
        
        required_dirs = [
            "data",
            "logs", 
            "metrics",
            "calibration",
            "tests"
        ]
        
        required_files = [
            "config/settings.py",
            "utils/logging_config.py",
            "utils/error_handling.py",
            "health_check.py",
            "PRODUCTION_READINESS_CHECKLIST.md"
        ]
        
        missing_items = []
        
        # Check directories
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created missing directory: {dir_path}")
                except Exception as e:
                    missing_items.append(f"Directory {dir_path}: {e}")
        
        # Check files
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_items.append(f"File {file_path}")
        
        if missing_items:
            self.errors.extend(missing_items)
            return False
        
        logger.info("File structure validated")
        return True

    def validate_database(self) -> bool:
        """Validate database connectivity and structure."""
        logger.info("Validating database...")
        
        def check_db():
            import sqlite3
            db_path = settings.DB_PATH
            
            # Check file exists
            if not Path(db_path).exists():
                raise FileNotFoundError(f"Database file not found: {db_path}")
            
            # Test connectivity
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                required_tables = ["matches", "teams", "predictions"]
                missing_tables = [t for t in required_tables if t not in tables]
                
                if missing_tables:
                    raise ValueError(f"Missing database tables: {missing_tables}")
            
            return True
        
        result = safe_execute(
            check_db,
            fallback_value=False,
            log_errors=True
        )
        
        if not result:
            self.errors.append("Database validation failed")
        else:
            logger.info("Database validated successfully")
        
        return result

    def validate_logging(self) -> bool:
        """Validate logging configuration."""
        logger.info("Validating logging configuration...")
        
        try:
            # Test log directory creation
            log_path = Path("logs/deployment_test.log")
            log_path.parent.mkdir(exist_ok=True)
            
            # Test log writing
            test_logger = logging.getLogger("deployment_test")
            handler = logging.FileHandler(log_path)
            test_logger.addHandler(handler)
            test_logger.info("Deployment validation test")
            handler.close()
            
            # Cleanup test file
            log_path.unlink(missing_ok=True)
            
            logger.info("Logging configuration validated")
            return True
            
        except Exception as e:
            self.errors.append(f"Logging validation failed: {e}")
            return False

    def validate_metrics(self) -> bool:
        """Validate metrics system."""
        logger.info("Validating metrics system...")
        
        try:
            from metrics.recorder import MetricsRecorder

            # Test metrics recording
            metrics_dir = Path(settings.METRICS_DIR)
            metrics_dir.mkdir(exist_ok=True)
            
            test_file = metrics_dir / "deployment_test.jsonl"
            recorder = MetricsRecorder(str(test_file))
            recorder.counter("deployment_test", 1)
            recorder.flush()
            
            # Verify file was written
            if not test_file.exists():
                raise ValueError("Metrics file was not created")
            
            # Cleanup
            test_file.unlink(missing_ok=True)
            
            logger.info("Metrics system validated")
            return True
            
        except Exception as e:
            self.errors.append(f"Metrics validation failed: {e}")
            return False

    def validate_health_check(self) -> bool:
        """Validate health check endpoint."""
        logger.info("Validating health check...")
        
        try:
            from health_check import get_system_health
            health = get_system_health()
            
            if not isinstance(health, dict):
                raise ValueError("Health check did not return dict")
            
            required_fields = ["status", "timestamp", "system", "database"]
            missing_fields = [f for f in required_fields if f not in health]
            
            if missing_fields:
                raise ValueError(f"Health check missing fields: {missing_fields}")
            
            logger.info("Health check validated")
            return True
            
        except Exception as e:
            self.errors.append(f"Health check validation failed: {e}")
            return False

    def run_tests(self) -> bool:
        """Run the test suite."""
        logger.info("Running test suite...")
        
        try:
            # Run pytest if available
            result = subprocess.run([
                sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.warnings.append(f"Some tests failed: {result.stdout[-500:]}")
                return False
            
            logger.info("Test suite completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            self.warnings.append("Test suite timed out")
            return False
        except FileNotFoundError:
            self.warnings.append("pytest not available, skipping tests")
            return True
        except Exception as e:
            self.warnings.append(f"Test execution failed: {e}")
            return False

    def generate_report(self) -> Dict:
        """Generate deployment validation report."""
        from datetime import timezone
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_results": self.validation_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "status": "ready" if not self.errors else "not_ready",
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "settings": {
                    "env": settings.ENV,
                    "db_path": settings.DB_PATH,
                    "calibration_enabled": settings.ENABLE_CALIBRATION
                }
            }
        }

    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("Starting deployment validation...")
        
        validations = [
            ("Environment", self.validate_environment),
            ("Dependencies", self.validate_dependencies), 
            ("File Structure", self.validate_file_structure),
            ("Database", self.validate_database),
            ("Logging", self.validate_logging),
            ("Metrics", self.validate_metrics),
            ("Health Check", self.validate_health_check),
            ("Tests", self.run_tests)
        ]
        
        all_passed = True
        
        for name, validation_func in validations:
            try:
                passed = validation_func()
                from datetime import timezone
                self.validation_results.append({
                    "name": name,
                    "passed": passed,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                logger.error(f"Validation {name} failed with exception: {e}")
                self.errors.append(f"{name}: {e}")
                from datetime import timezone
                self.validation_results.append({
                    "name": name,
                    "passed": False,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                all_passed = False
        
        # Generate report
        report = self.generate_report()
        report_path = Path("deployment_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
        
        if all_passed:
            logger.info("🎉 All validations passed! System is ready for deployment.")
        else:
            logger.error("❌ Some validations failed. Check errors before deployment.")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning("⚠️ Warnings found:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        return all_passed


def main():
    """Main entry point for deployment validation."""
    validator = DeploymentValidator()
    
    try:
        success = validator.validate_all()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()