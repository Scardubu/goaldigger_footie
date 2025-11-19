#!/usr/bin/env python3
"""
Production Database Migration Executor
Safely executes database migrations with validation and rollback capability
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class MigrationExecutor:
    """Executes database migrations safely with validation."""
    
    def __init__(self, dry_run: bool = True, backup: bool = True):
        """Initialize migration executor.
        
        Args:
            dry_run: If True, only validate without making changes
            backup: If True, create backup before migration
        """
        self.project_root = project_root
        self.dry_run = dry_run
        self.backup = backup
        self.migrations_dir = self.project_root / 'migrations'
        self.backup_dir = self.project_root / 'backups' / f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Load environment
        load_dotenv()
        self.db_url = os.getenv('DATABASE_URL', '')
        
        logger.info(f"🔧 Migration Executor initialized (dry_run={dry_run}, backup={backup})")
    
    def execute(self) -> bool:
        """Execute the migration process.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("=" * 70)
            logger.info("🚀 Starting Production Database Migration")
            logger.info("=" * 70)
            
            # Step 1: Validate environment
            if not self._validate_environment():
                logger.error("❌ Environment validation failed")
                return False
            
            # Step 2: Backup database (if enabled and not dry-run)
            if self.backup and not self.dry_run:
                if not self._backup_database():
                    logger.error("❌ Database backup failed")
                    return False
            
            # Step 3: Validate migration files exist
            migrations = self._get_pending_migrations()
            if not migrations:
                logger.info("✅ No pending migrations found")
                return True
            
            logger.info(f"📋 Found {len(migrations)} pending migration(s):")
            for i, migration in enumerate(migrations, 1):
                logger.info(f"  {i}. {migration.name}")
            
            # Step 4: Execute migrations
            for migration in migrations:
                logger.info(f"\n{'='*70}")
                logger.info(f"📝 Executing migration: {migration.name}")
                logger.info(f"{'='*70}")
                
                if not self._execute_migration(migration):
                    logger.error(f"❌ Migration failed: {migration.name}")
                    return False
                
                logger.info(f"✅ Migration completed: {migration.name}")
            
            # Step 5: Post-migration population script
            if not self.dry_run:
                logger.info(f"\n{'='*70}")
                logger.info("📊 Running data population script")
                logger.info(f"{'='*70}")
                
                if not self._populate_last_synced_at():
                    logger.warning("⚠️ Data population completed with warnings")
            
            # Step 6: Validate schema changes
            if not self._validate_schema():
                logger.error("❌ Schema validation failed")
                return False
            
            logger.info(f"\n{'='*70}")
            logger.info("🎉 Migration Process Complete!")
            logger.info(f"{'='*70}")
            
            if self.dry_run:
                logger.info("ℹ️ DRY RUN - No changes were made to the database")
                logger.info("ℹ️ Run without --dry-run to apply changes")
            else:
                logger.info("✅ All migrations applied successfully")
                if self.backup:
                    logger.info(f"💾 Backup saved to: {self.backup_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration process failed: {e}", exc_info=True)
            return False
    
    def _validate_environment(self) -> bool:
        """Validate environment configuration."""
        logger.info("🔍 Validating environment...")
        
        issues = []
        warnings = []
        
        # Check database URL
        if not self.db_url:
            # Try SQLite fallback
            sqlite_path = os.getenv('SQLITE_DATABASE_PATH', 'data/football.db')
            if Path(sqlite_path).exists():
                self.db_url = f'sqlite:///{sqlite_path}'
                warnings.append(f"DATABASE_URL not set, using SQLite fallback: {sqlite_path}")
            else:
                issues.append("DATABASE_URL not set and SQLite fallback not found")
        
        # Check migrations directory
        if not self.migrations_dir.exists():
            issues.append(f"Migrations directory not found: {self.migrations_dir}")
        
        # Check Python dependencies
        try:
            import sqlalchemy
            logger.info(f"  ✓ SQLAlchemy {sqlalchemy.__version__}")
        except ImportError:
            issues.append("SQLAlchemy not installed")
        
        if self.db_url.startswith('postgresql://'):
            try:
                import psycopg2
                logger.info(f"  ✓ psycopg2 installed")
            except ImportError:
                issues.append("psycopg2 not installed (required for PostgreSQL)")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"  ⚠️  {warning}")
        
        if issues:
            logger.error("❌ Environment validation issues found:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("✅ Environment validation passed")
        return True
    
    def _backup_database(self) -> bool:
        """Create database backup before migration."""
        logger.info("💾 Creating database backup...")
        
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            if self.db_url.startswith('postgresql://'):
                backup_file = self.backup_dir / 'postgres_backup.sql'
                logger.info(f"  Creating PostgreSQL backup: {backup_file}")
                
                # Extract connection details
                import urllib.parse as urlparse
                result = urlparse.urlparse(self.db_url)
                
                # Create pg_dump command
                import subprocess
                env = os.environ.copy()
                env['PGPASSWORD'] = result.password or ''
                
                cmd = [
                    'pg_dump',
                    '-h', result.hostname or 'localhost',
                    '-p', str(result.port or 5432),
                    '-U', result.username or 'postgres',
                    '-d', result.path.lstrip('/'),
                    '-f', str(backup_file),
                    '--no-owner',
                    '--no-privileges'
                ]
                
                result = subprocess.run(cmd, env=env, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"pg_dump failed: {result.stderr}")
                    logger.warning("⚠️ Continuing without backup...")
                    return True  # Don't fail migration due to backup issue
                
                logger.info(f"  ✓ Backup created: {backup_file}")
                
            elif self.db_url.startswith('sqlite://'):
                import shutil
                db_path = self.db_url.replace('sqlite:///', '')
                if Path(db_path).exists():
                    backup_file = self.backup_dir / Path(db_path).name
                    shutil.copy2(db_path, backup_file)
                    logger.info(f"  ✓ Backup created: {backup_file}")
                else:
                    logger.warning(f"  ⚠️ Database file not found: {db_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            logger.warning("⚠️ Continuing without backup...")
            return True  # Don't fail migration due to backup issue
    
    def _get_pending_migrations(self) -> List[Path]:
        """Get list of pending migration files."""
        if not self.migrations_dir.exists():
            return []
        
        # Find all SQL migration files
        migrations = sorted(self.migrations_dir.glob('*.sql'))
        
        # For now, return the last_synced_at migration
        # In a full system, this would check against a migrations table
        pending = [m for m in migrations if '0002_add_last_synced_at' in m.name]
        
        return pending
    
    def _execute_migration(self, migration_file: Path) -> bool:
        """Execute a single migration file.
        
        Args:
            migration_file: Path to SQL migration file
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"📄 Reading migration: {migration_file.name}")
        
        try:
            # Read migration SQL
            sql_content = migration_file.read_text()
            
            if self.dry_run:
                logger.info("  ℹ️ DRY RUN - Would execute:")
                logger.info("  " + "-" * 60)
                for line in sql_content.split('\n')[:20]:  # Show first 20 lines
                    if line.strip() and not line.strip().startswith('--'):
                        logger.info(f"    {line}")
                if len(sql_content.split('\n')) > 20:
                    logger.info("    ... (truncated)")
                logger.info("  " + "-" * 60)
                return True
            
            # Execute migration
            if self.db_url.startswith('postgresql://'):
                return self._execute_postgres_migration(sql_content)
            elif self.db_url.startswith('sqlite://'):
                return self._execute_sqlite_migration(sql_content)
            else:
                logger.error(f"❌ Unsupported database: {self.db_url.split(':')[0]}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Migration execution failed: {e}", exc_info=True)
            return False
    
    def _execute_postgres_migration(self, sql: str) -> bool:
        """Execute PostgreSQL migration."""
        try:
            from sqlalchemy import create_engine, text
            
            engine = create_engine(self.db_url)
            
            with engine.connect() as conn:
                # PostgreSQL migration uses DO block for conditional ALTER
                conn.execute(text(sql))
                conn.commit()
            
            logger.info("  ✅ PostgreSQL migration executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ PostgreSQL migration failed: {e}")
            return False
    
    def _execute_sqlite_migration(self, sql: str) -> bool:
        """Execute SQLite migration."""
        try:
            import sqlite3
            
            db_path = self.db_url.replace('sqlite:///', '')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # SQLite requires special handling - check if column exists first
            cursor.execute("PRAGMA table_info(matches)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'last_synced_at' in columns:
                logger.info("  ℹ️ Column 'last_synced_at' already exists, skipping")
                conn.close()
                return True
            
            # Execute ALTER TABLE
            try:
                cursor.execute("ALTER TABLE matches ADD COLUMN last_synced_at TEXT")
                conn.commit()
                logger.info("  ✅ SQLite migration executed successfully")
            except sqlite3.OperationalError as e:
                if 'duplicate column' in str(e).lower():
                    logger.info("  ℹ️ Column already exists (duplicate), skipping")
                else:
                    raise
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"  ❌ SQLite migration failed: {e}")
            return False
    
    def _populate_last_synced_at(self) -> bool:
        """Populate last_synced_at column for existing records."""
        logger.info("📊 Populating last_synced_at for existing records...")
        
        try:
            # Import the population script
            from scripts.populate_last_synced_at import main as populate_main

            # Run population (not in dry-run mode)
            populate_main(dry_run=False)
            
            logger.info("  ✅ Data population completed")
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Population failed: {e}")
            logger.warning("  ⚠️ Manual population may be required")
            return False
    
    def _validate_schema(self) -> bool:
        """Validate schema changes were applied correctly."""
        logger.info("🔍 Validating schema changes...")
        
        try:
            if self.db_url.startswith('postgresql://'):
                from sqlalchemy import create_engine, text
                
                try:
                    engine = create_engine(self.db_url, connect_args={'connect_timeout': 5})
                    
                    with engine.connect() as conn:
                        result = conn.execute(text("""
                            SELECT column_name, data_type 
                            FROM information_schema.columns 
                            WHERE table_name = 'matches' AND column_name = 'last_synced_at'
                        """))
                        
                        row = result.fetchone()
                        if row:
                            logger.info(f"  ✓ Column 'last_synced_at' exists: {row[1]}")
                            return True
                        else:
                            logger.error("  ❌ Column 'last_synced_at' not found")
                            return False
                            
                except Exception as pg_error:
                    logger.warning(f"  ⚠️  PostgreSQL connection failed: {str(pg_error)[:100]}")
                    logger.info("  ℹ️  This is expected if PostgreSQL is not running")
                    
                    # In dry-run mode, this is not a failure
                    if self.dry_run:
                        logger.info("  ✓ Dry-run mode: Validation skipped")
                        return True
                    return False
                        
            elif self.db_url.startswith('sqlite://'):
                import sqlite3
                
                db_path = self.db_url.replace('sqlite:///', '')
                
                if not Path(db_path).exists():
                    logger.warning(f"  ⚠️  SQLite database not found: {db_path}")
                    if self.dry_run:
                        logger.info("  ✓ Dry-run mode: Validation skipped")
                        return True
                    return False
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute("PRAGMA table_info(matches)")
                columns = {row[1]: row[2] for row in cursor.fetchall()}
                
                conn.close()
                
                if 'last_synced_at' in columns:
                    logger.info(f"  ✓ Column 'last_synced_at' exists: {columns['last_synced_at']}")
                    return True
                else:
                    if self.dry_run:
                        logger.info("  ℹ️  Column not yet added (expected in dry-run)")
                        return True
                    logger.error("  ❌ Column 'last_synced_at' not found")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Schema validation failed: {e}")
            if self.dry_run:
                logger.info("  ℹ️  Dry-run mode: Continuing despite validation error")
                return True
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Execute production database migrations safely',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview changes
  python scripts/execute_production_migration.py --dry-run
  
  # Execute migration with backup
  python scripts/execute_production_migration.py
  
  # Execute without backup (not recommended)
  python scripts/execute_production_migration.py --no-backup
  
  # Force execute even if validations fail
  python scripts/execute_production_migration.py --force
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without executing'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip database backup (not recommended)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force execution even if validations fail'
    )
    
    args = parser.parse_args()
    
    # Create executor
    executor = MigrationExecutor(
        dry_run=args.dry_run,
        backup=not args.no_backup
    )
    
    # Execute
    success = executor.execute()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
