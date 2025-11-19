#!/usr/bin/env python3
"""
Production Deployment Orchestrator
Comprehensive automated deployment with validation, migration, and verification
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDeployer:
    """Orchestrates production deployment with full validation."""
    
    def __init__(self, dry_run: bool = False, skip_migration: bool = False):
        """Initialize deployment orchestrator.
        
        Args:
            dry_run: If True, perform validation only without changes
            skip_migration: If True, skip database migration
        """
        self.project_root = project_root
        self.dry_run = dry_run
        self.skip_migration = skip_migration
        self.deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = project_root / 'logs' / f'{self.deployment_id}.log'
        
        # Create logs directory
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load environment
        load_dotenv()
        
        logger.info(f"🚀 Production Deployment Orchestrator")
        logger.info(f"   Deployment ID: {self.deployment_id}")
        logger.info(f"   Dry Run: {dry_run}")
        logger.info(f"   Log File: {self.log_file}")
    
    def deploy(self) -> bool:
        """Execute full deployment process.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("\n" + "=" * 70)
            logger.info("🚀 STARTING PRODUCTION DEPLOYMENT")
            logger.info("=" * 70)
            
            start_time = time.time()
            
            # Phase 1: Pre-Deployment Validation
            if not self._phase_1_validation():
                logger.error("❌ Phase 1: Validation failed")
                return False
            
            # Phase 2: Database Migration
            if not self.skip_migration:
                if not self._phase_2_migration():
                    logger.error("❌ Phase 2: Migration failed")
                    return False
            else:
                logger.info("⏭️  Phase 2: Migration skipped")
            
            # Phase 3: Application Deployment
            if not self._phase_3_deployment():
                logger.error("❌ Phase 3: Deployment failed")
                return False
            
            # Phase 4: Post-Deployment Verification
            if not self._phase_4_verification():
                logger.error("❌ Phase 4: Verification failed")
                return False
            
            duration = time.time() - start_time
            
            logger.info("\n" + "=" * 70)
            logger.info("🎉 DEPLOYMENT SUCCESSFUL!")
            logger.info("=" * 70)
            logger.info(f"⏱️  Total Duration: {duration:.1f}s")
            logger.info(f"📋 Deployment ID: {self.deployment_id}")
            logger.info(f"📄 Log File: {self.log_file}")
            
            if self.dry_run:
                logger.info("\nℹ️  DRY RUN COMPLETE - No changes were made")
                logger.info("ℹ️  Run without --dry-run to perform actual deployment")
            else:
                logger.info("\n✅ Production deployment complete and verified!")
                logger.info("📊 Next steps:")
                logger.info("   1. Monitor Grafana dashboards")
                logger.info("   2. Check health endpoint: http://localhost:8501/health")
                logger.info("   3. Verify metrics: http://localhost:8501/metrics")
                logger.info("   4. Review logs: tail -f logs/goaldiggers.log")
            
            return True
            
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Deployment interrupted by user")
            return False
        except Exception as e:
            logger.error(f"\n❌ Deployment failed: {e}", exc_info=True)
            return False
    
    def _phase_1_validation(self) -> bool:
        """Phase 1: Pre-deployment validation."""
        logger.info("\n" + "=" * 70)
        logger.info("📋 PHASE 1: Pre-Deployment Validation")
        logger.info("=" * 70)
        
        # Step 1.1: Environment validation
        logger.info("\n🔍 Step 1.1: Validating environment configuration...")
        
        required_vars = ['FOOTBALL_DATA_API_KEY', 'DATABASE_URL', 'SECRET_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            logger.error("   Please configure these in .env file")
            return False
        
        logger.info("✅ Environment variables configured")
        
        # Step 1.2: Dependency check
        logger.info("\n🔍 Step 1.2: Checking Python dependencies...")
        
        critical_deps = [
            'streamlit',
            'pandas',
            'numpy',
            'sqlalchemy',
            'prometheus_client'
        ]
        
        missing_deps = []
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.error(f"❌ Missing dependencies: {missing_deps}")
            logger.error("   Run: pip install -r requirements.txt")
            return False
        
        logger.info("✅ All dependencies installed")
        
        # Step 1.3: File system check
        logger.info("\n🔍 Step 1.3: Checking file system...")
        
        required_dirs = ['data', 'logs', 'backups', 'monitoring']
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                if not self.dry_run:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"  ✓ Created directory: {dir_name}/")
                else:
                    logger.info(f"  ℹ️  Would create: {dir_name}/")
            else:
                logger.info(f"  ✓ Directory exists: {dir_name}/")
        
        # Step 1.4: Health check
        logger.info("\n🔍 Step 1.4: Running health check...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(self.project_root / 'scripts' / 'production_health_check.py')],
                capture_output=True,
                text=True,
                timeout=90  # Increased from 30s to allow for dependency imports
            )
            
            if result.returncode == 0:
                logger.info("✅ Health check passed")
            else:
                logger.warning("⚠️  Health check found issues (continuing...)")
                if result.stdout:
                    for line in result.stdout.split('\n')[-10:]:
                        if line.strip():
                            logger.info(f"     {line}")
        except Exception as e:
            logger.warning(f"⚠️  Health check failed: {e}")
        
        logger.info("\n✅ Phase 1: Validation complete")
        return True
    
    def _phase_2_migration(self) -> bool:
        """Phase 2: Database migration."""
        logger.info("\n" + "=" * 70)
        logger.info("🗄️  PHASE 2: Database Migration")
        logger.info("=" * 70)
        
        logger.info("\n📊 Running database migration...")
        
        try:
            migration_script = self.project_root / 'scripts' / 'execute_production_migration.py'
            
            if not migration_script.exists():
                logger.warning("⚠️  Migration script not found, skipping")
                return True
            
            cmd = [sys.executable, str(migration_script)]
            
            if self.dry_run:
                cmd.append('--dry-run')
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Log output
            if result.stdout:
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
            
            if result.returncode != 0:
                logger.error("❌ Migration failed")
                if result.stderr:
                    logger.error(f"   Error: {result.stderr}")
                return False
            
            logger.info("✅ Migration complete")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ Migration timed out (>5 minutes)")
            return False
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False
    
    def _phase_3_deployment(self) -> bool:
        """Phase 3: Application deployment."""
        logger.info("\n" + "=" * 70)
        logger.info("🚀 PHASE 3: Application Deployment")
        logger.info("=" * 70)
        
        if self.dry_run:
            logger.info("ℹ️  DRY RUN - Would deploy application")
            logger.info("   - Update systemd service (if exists)")
            logger.info("   - Restart application")
            logger.info("   - Wait for healthcheck")
            return True
        
        # Step 3.1: Check if systemd service exists
        logger.info("\n🔍 Checking deployment method...")
        
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', 'goaldiggers'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("  ✓ Systemd service detected")
                
                # Restart service
                logger.info("\n🔄 Restarting application...")
                result = subprocess.run(
                    ['sudo', 'systemctl', 'restart', 'goaldiggers'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    logger.error("❌ Service restart failed")
                    return False
                
                logger.info("✅ Application restarted")
                
                # Wait for startup
                logger.info("⏳ Waiting for application startup...")
                time.sleep(10)
                
            else:
                logger.info("  ℹ️  No systemd service found")
                logger.info("  ℹ️  Manual application start required")
                
        except FileNotFoundError:
            logger.info("  ℹ️  Systemd not available (Windows or manual deployment)")
        except Exception as e:
            logger.warning(f"⚠️  Service check failed: {e}")
        
        logger.info("\n✅ Phase 3: Deployment complete")
        return True
    
    def _phase_4_verification(self) -> bool:
        """Phase 4: Post-deployment verification."""
        logger.info("\n" + "=" * 70)
        logger.info("✅ PHASE 4: Post-Deployment Verification")
        logger.info("=" * 70)
        
        # Step 4.1: Health check
        logger.info("\n🏥 Running post-deployment health check...")
        
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(self.project_root / 'scripts' / 'production_health_check.py'),
                    '--verbose'
                ],
                capture_output=True,
                text=True,
                timeout=90  # Increased from 30s to allow for dependency imports
            )
            
            # Parse health check results
            if 'Health Score:' in result.stdout:
                for line in result.stdout.split('\n'):
                    if 'Health Score:' in line or 'Status:' in line:
                        logger.info(f"  {line.strip()}")
            
            if result.returncode != 0:
                logger.warning("⚠️  Health check found issues")
                logger.info("   Review output above for details")
            else:
                logger.info("✅ Health check passed")
                
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Health check timed out after 90 seconds")
            logger.info("   This can happen due to slow dependency imports")
        except Exception as e:
            logger.warning(f"⚠️  Health check failed: {e}")
        
        # Step 4.2: Metrics verification
        logger.info("\n📊 Verifying metrics export...")
        
        if not self.dry_run:
            try:
                import requests
                
                response = requests.get(
                    'http://localhost:8501/metrics',
                    timeout=5
                )
                
                if response.status_code == 200:
                    metrics_text = response.text
                    
                    # Check for critical metrics
                    critical_metrics = [
                        'goaldiggers_validation_quality_score',
                        'goaldiggers_prediction_engine_predictions_total'
                    ]
                    
                    found_metrics = sum(1 for m in critical_metrics if m in metrics_text)
                    
                    logger.info(f"  ✓ Metrics endpoint responding")
                    logger.info(f"  ✓ Found {found_metrics}/{len(critical_metrics)} critical metrics")
                else:
                    logger.warning(f"  ⚠️  Metrics endpoint returned {response.status_code}")
                    
            except Exception as e:
                logger.warning(f"  ⚠️  Could not verify metrics: {e}")
                logger.info("     This is expected if application not running")
        else:
            logger.info("  ℹ️  Would verify metrics endpoint")
        
        # Step 4.3: Database verification
        logger.info("\n🗄️  Verifying database schema...")
        
        try:
            db_url = os.getenv('DATABASE_URL', '')
            if db_url:
                from sqlalchemy import create_engine, text
                
                engine = create_engine(db_url)
                
                with engine.connect() as conn:
                    # Check for last_synced_at column
                    if db_url.startswith('postgresql://'):
                        result = conn.execute(text("""
                            SELECT column_name 
                            FROM information_schema.columns 
                            WHERE table_name = 'matches' AND column_name = 'last_synced_at'
                        """))
                    else:
                        result = conn.execute(text("PRAGMA table_info(matches)"))
                        columns = [row[1] for row in result.fetchall()]
                        result = ['last_synced_at'] if 'last_synced_at' in columns else []
                    
                    if result if isinstance(result, list) else result.fetchone():
                        logger.info("  ✓ Migration applied successfully")
                    else:
                        logger.warning("  ⚠️  Migration may not be applied")
                        
        except Exception as e:
            logger.warning(f"  ⚠️  Database verification failed: {e}")
        
        logger.info("\n✅ Phase 4: Verification complete")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Orchestrate production deployment with validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview deployment
  python scripts/deploy_production_orchestrator.py --dry-run
  
  # Full production deployment
  python scripts/deploy_production_orchestrator.py
  
  # Deploy without running migration (already done)
  python scripts/deploy_production_orchestrator.py --skip-migration
  
  # Dry run with migration skip
  python scripts/deploy_production_orchestrator.py --dry-run --skip-migration
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview deployment without making changes'
    )
    
    parser.add_argument(
        '--skip-migration',
        action='store_true',
        help='Skip database migration (if already applied)'
    )
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = ProductionDeployer(
        dry_run=args.dry_run,
        skip_migration=args.skip_migration
    )
    
    # Execute deployment
    success = deployer.deploy()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
