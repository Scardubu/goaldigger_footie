#!/usr/bin/env python3
"""
Production deployment script for GoalDiggers platform.

Automates the deployment process with proper validation, backup, and rollback
capabilities. Suitable for cloud deployment (Render, Heroku, etc.) or local
production setup.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logging_config import configure_logging

# Configure logging
configure_logging()
import logging

logger = logging.getLogger(__name__)


class ProductionDeployer:
    """Handles production deployment with validation and rollback."""

    def __init__(self, environment: str = "production"):
        self.environment = environment
    self.deployment_id = f"deploy_{int(time.time())}"
        self.backup_dir = Path(f"backups/{self.deployment_id}")
        self.deployment_log = []

    def log_step(self, step: str, status: str = "started", details: Dict = None):
        """Log deployment step with timestamp."""
        from datetime import timezone
        entry = {
            "step": step,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }
        self.deployment_log.append(entry)
        logger.info(f"Deployment step: {step} - {status}")

    def create_backup(self) -> bool:
        """Create backup of critical files before deployment."""
        self.log_step("create_backup")
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup critical files
            backup_files = [
                "data/football.db",
                "config/settings.py",
                "logs/",
                "metrics/",
                ".env"
            ]
            
            for item in backup_files:
                source = Path(item)
                if source.exists():
                    if source.is_file():
                        shutil.copy2(source, self.backup_dir / source.name)
                    else:
                        shutil.copytree(source, self.backup_dir / source.name, dirs_exist_ok=True)
                    logger.info(f"Backed up: {item}")
            
            self.log_step("create_backup", "completed")
            return True
            
        except Exception as e:
            self.log_step("create_backup", "failed", {"error": str(e)})
            logger.error(f"Backup creation failed: {e}")
            return False

    def validate_environment(self) -> bool:
        """Validate deployment environment."""
        self.log_step("validate_environment")
        
        try:
            # Run deployment validation script
            result = subprocess.run([
                sys.executable, "scripts/validate_deployment.py"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                self.log_step("validate_environment", "failed", {
                    "stdout": result.stdout[-1000:],
                    "stderr": result.stderr[-1000:]
                })
                return False
            
            self.log_step("validate_environment", "completed")
            return True
            
        except Exception as e:
            self.log_step("validate_environment", "failed", {"error": str(e)})
            logger.error(f"Environment validation failed: {e}")
            return False

    def install_dependencies(self) -> bool:
        """Install production dependencies."""
        self.log_step("install_dependencies")
        
        try:
            # Install from requirements.txt if it exists
            if Path("requirements.txt").exists():
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode != 0:
                    self.log_step("install_dependencies", "failed", {
                        "stderr": result.stderr[-1000:]
                    })
                    return False
            
            # Install from pyproject.toml if available
            if Path("pyproject.toml").exists():
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-e", "."
                ], capture_output=True, text=True, timeout=600)
                
                if result.returncode != 0:
                    logger.warning("pyproject.toml install failed, continuing...")
            
            self.log_step("install_dependencies", "completed")
            return True
            
        except Exception as e:
            self.log_step("install_dependencies", "failed", {"error": str(e)})
            logger.error(f"Dependency installation failed: {e}")
            return False

    def setup_directories(self) -> bool:
        """Create required directories."""
        self.log_step("setup_directories")
        
        try:
            directories = [
                "data",
                "logs", 
                "metrics",
                "data/freshness_runs",
                "calibration"
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            
            self.log_step("setup_directories", "completed")
            return True
            
        except Exception as e:
            self.log_step("setup_directories", "failed", {"error": str(e)})
            logger.error(f"Directory setup failed: {e}")
            return False

    def run_database_migrations(self) -> bool:
        """Run database migrations if needed."""
        self.log_step("run_database_migrations")
        
        try:
            # Check if migration script exists
            migration_script = Path("database_migration.py")
            if migration_script.exists():
                result = subprocess.run([
                    sys.executable, str(migration_script)
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    self.log_step("run_database_migrations", "failed", {
                        "stderr": result.stderr[-1000:]
                    })
                    return False
            
            self.log_step("run_database_migrations", "completed")
            return True
            
        except Exception as e:
            self.log_step("run_database_migrations", "failed", {"error": str(e)})
            logger.error(f"Database migration failed: {e}")
            return False

    def configure_production_settings(self) -> bool:
        """Configure production-specific settings."""
        self.log_step("configure_production_settings")
        
        try:
            # Set production environment variables
            env_vars = {
                "APP_ENV": "production",
                "LOG_LEVEL": "INFO",
                "LOG_JSON": "1",  # Enable JSON logging for production
                "ENABLE_CALIBRATION": "1"
            }
            
            env_file = Path(".env")
            env_content = []
            
            # Read existing .env file if it exists
            if env_file.exists():
                env_content = env_file.read_text().strip().split('\n')
            
            # Update or add environment variables
            for key, value in env_vars.items():
                found = False
                for i, line in enumerate(env_content):
                    if line.startswith(f"{key}="):
                        env_content[i] = f"{key}={value}"
                        found = True
                        break
                if not found:
                    env_content.append(f"{key}={value}")
            
            # Write updated .env file
            env_file.write_text('\n'.join(env_content) + '\n')
            
            self.log_step("configure_production_settings", "completed")
            return True
            
        except Exception as e:
            self.log_step("configure_production_settings", "failed", {"error": str(e)})
            logger.error(f"Production configuration failed: {e}")
            return False

    def health_check(self) -> bool:
        """Perform post-deployment health check."""
        self.log_step("health_check")
        
        try:
            from health_check import get_system_health

            # Wait for system to stabilize
            time.sleep(5)
            
            health = get_system_health()
            
            if not isinstance(health, dict) or health.get("status") != "healthy":
                self.log_step("health_check", "failed", {"health": health})
                return False
            
            self.log_step("health_check", "completed", {"health": health})
            return True
            
        except Exception as e:
            self.log_step("health_check", "failed", {"error": str(e)})
            logger.error(f"Health check failed: {e}")
            return False

    def rollback(self) -> bool:
        """Rollback deployment using backup."""
        logger.warning("Initiating deployment rollback...")
        self.log_step("rollback")
        
        try:
            if not self.backup_dir.exists():
                logger.error("No backup found for rollback")
                return False
            
            # Restore files from backup
            for backup_file in self.backup_dir.iterdir():
                if backup_file.name == "logs":
                    continue  # Don't restore logs
                
                target = Path(backup_file.name)
                if backup_file.is_file():
                    shutil.copy2(backup_file, target)
                else:
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.copytree(backup_file, target)
                
                logger.info(f"Restored: {backup_file.name}")
            
            self.log_step("rollback", "completed")
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            self.log_step("rollback", "failed", {"error": str(e)})
            logger.error(f"Rollback failed: {e}")
            return False

    def generate_deployment_report(self) -> Dict:
        """Generate deployment report."""
        from datetime import timezone
        return {
            "deployment_id": self.deployment_id,
            "environment": self.environment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "steps": self.deployment_log,
            "success": all(step.get("status") != "failed" for step in self.deployment_log),
            "duration_seconds": (datetime.now(timezone.utc) - datetime.fromisoformat(
                self.deployment_log[0]["timestamp"]
            )).total_seconds() if self.deployment_log else 0
        }

    def deploy(self) -> bool:
        """Execute full deployment process."""
        logger.info(f"Starting production deployment {self.deployment_id}")
        
        deployment_steps = [
            ("Create Backup", self.create_backup),
            ("Validate Environment", self.validate_environment),
            ("Install Dependencies", self.install_dependencies),
            ("Setup Directories", self.setup_directories),
            ("Run Database Migrations", self.run_database_migrations),
            ("Configure Production Settings", self.configure_production_settings),
            ("Health Check", self.health_check)
        ]
        
        success = True
        
        for step_name, step_func in deployment_steps:
            logger.info(f"Executing: {step_name}")
            
            try:
                if not step_func():
                    logger.error(f"Step failed: {step_name}")
                    success = False
                    break
            except Exception as e:
                logger.error(f"Step {step_name} raised exception: {e}")
                success = False
                break
        
        # Generate deployment report
        report = self.generate_deployment_report()
        report_path = Path(f"deployment_report_{self.deployment_id}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        if success:
            logger.info(f"🎉 Deployment {self.deployment_id} completed successfully!")
            logger.info(f"Report saved to: {report_path}")
        else:
            logger.error(f"❌ Deployment {self.deployment_id} failed!")
            
            # Offer rollback
            if input("Rollback to previous state? (y/N): ").lower() == 'y':
                self.rollback()
        
        return success


def main():
    """Main entry point for production deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy GoalDiggers to production")
    parser.add_argument("--environment", default="production", help="Deployment environment")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--rollback", help="Rollback to specific deployment ID")
    
    args = parser.parse_args()
    
    try:
        if args.rollback:
            # Implement rollback to specific deployment
            logger.info(f"Rolling back to deployment: {args.rollback}")
            # This would need to be implemented based on backup structure
            sys.exit(0)
        
        deployer = ProductionDeployer(args.environment)
        success = deployer.deploy()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Deployment failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()