#!/usr/bin/env python3
"""
Production Health Check System
Comprehensive health validation for production deployment
"""
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class HealthCheckSystem:
    """Comprehensive production health check system."""
    
    def __init__(self, verbose: bool = False):
        """Initialize health check system.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.project_root = project_root
        self.checks_passed = 0
        self.checks_failed = 0
        self.checks_warnings = 0
        self.results = []
        
        # Load environment
        load_dotenv()
        
        logger.info("🏥 Production Health Check System initialized")
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks.
        
        Returns:
            Dictionary with health check results
        """
        logger.info("=" * 70)
        logger.info("🏥 Starting Production Health Checks")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Run all check categories
        self._check_environment()
        self._check_database()
        self._check_redis()
        self._check_api_connectivity()
        self._check_file_system()
        self._check_dependencies()
        self._check_metrics_export()
        self._check_data_quality()
        self._check_performance()
        
        duration = time.time() - start_time
        
        # Calculate health score
        total_checks = self.checks_passed + self.checks_failed + self.checks_warnings
        health_score = (self.checks_passed / total_checks * 100) if total_checks > 0 else 0
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': round(duration, 2),
            'health_score': round(health_score, 1),
            'status': self._get_overall_status(health_score),
            'summary': {
                'passed': self.checks_passed,
                'failed': self.checks_failed,
                'warnings': self.checks_warnings,
                'total': total_checks
            },
            'checks': self.results
        }
        
        # Log summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 Health Check Summary")
        logger.info("=" * 70)
        logger.info(f"✅ Passed:   {self.checks_passed}")
        logger.info(f"⚠️  Warnings: {self.checks_warnings}")
        logger.info(f"❌ Failed:   {self.checks_failed}")
        logger.info(f"📈 Health Score: {health_score:.1f}%")
        logger.info(f"🏆 Status: {report['status']}")
        logger.info(f"⏱️  Duration: {duration:.2f}s")
        logger.info("=" * 70)
        
        return report
    
    def _get_overall_status(self, health_score: float) -> str:
        """Get overall status based on health score."""
        if health_score >= 95:
            return "EXCELLENT"
        elif health_score >= 85:
            return "GOOD"
        elif health_score >= 70:
            return "FAIR"
        elif health_score >= 50:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _add_result(self, category: str, check: str, status: str, 
                   message: str, details: Optional[Dict] = None):
        """Add check result."""
        result = {
            'category': category,
            'check': check,
            'status': status,
            'message': message,
            'details': details or {}
        }
        
        self.results.append(result)
        
        # Update counters
        if status == 'pass':
            self.checks_passed += 1
            icon = "✅"
        elif status == 'warning':
            self.checks_warnings += 1
            icon = "⚠️"
        else:
            self.checks_failed += 1
            icon = "❌"
        
        # Log result
        if self.verbose or status != 'pass':
            logger.info(f"{icon} [{category}] {check}: {message}")
    
    def _check_environment(self):
        """Check environment configuration."""
        logger.info("\n📋 Checking Environment Configuration...")
        
        # Required environment variables
        required_vars = [
            'FOOTBALL_DATA_API_KEY',
            'DATABASE_URL',
            'SECRET_KEY'
        ]
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                # Don't log sensitive values
                display_value = value[:10] + '***' if len(value) > 10 else '***'
                self._add_result(
                    'Environment',
                    f'{var}',
                    'pass',
                    f'Set ({display_value})'
                )
            else:
                self._add_result(
                    'Environment',
                    f'{var}',
                    'fail',
                    'Not set'
                )
        
        # Optional but recommended variables
        optional_vars = [
            'REDIS_URL',
            'GEMINI_API_KEY',
            'OPENROUTER_API_KEY'
        ]
        
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                self._add_result(
                    'Environment',
                    f'{var} (optional)',
                    'pass',
                    'Configured'
                )
            else:
                self._add_result(
                    'Environment',
                    f'{var} (optional)',
                    'warning',
                    'Not configured (optional feature disabled)'
                )
        
        # Check data quality settings
        quality_vars = {
            'REAL_DATA_MIN_QUALITY_SCORE': '0.5',
            'REAL_DATA_LOOKBACK_DAYS': '3',
            'PROGRESSIVE_CACHE_MAX_AGE': '3600'
        }
        
        for var, default in quality_vars.items():
            value = os.getenv(var, default)
            self._add_result(
                'Environment',
                var,
                'pass',
                f'Set to {value}',
                {'value': value, 'default': default}
            )
    
    def _check_database(self):
        """Check database connectivity and schema."""
        logger.info("\n🗄️  Checking Database...")
        
        try:
            db_url = os.getenv('DATABASE_URL', '')
            
            if not db_url:
                # Try SQLite fallback
                sqlite_path = os.getenv('SQLITE_DATABASE_PATH', 'data/football.db')
                if Path(sqlite_path).exists():
                    db_url = f'sqlite:///{sqlite_path}'
                    self._add_result(
                        'Database',
                        'Configuration',
                        'warning',
                        f'Using SQLite fallback: {sqlite_path}'
                    )
                else:
                    self._add_result(
                        'Database',
                        'Connection',
                        'fail',
                        'DATABASE_URL not configured'
                    )
                    return
            
            # Try to connect
            from sqlalchemy import create_engine, text
            
            try:
                # Add timeout for PostgreSQL connections
                connect_args = {}
                if db_url.startswith('postgresql://'):
                    connect_args = {'connect_timeout': 5}
                
                engine = create_engine(
                    db_url, 
                    pool_pre_ping=True,
                    connect_args=connect_args
                )
                
                with engine.connect() as conn:
                    # Test connection
                    result = conn.execute(text("SELECT 1"))
                    result.fetchone()
                    
                    self._add_result(
                        'Database',
                        'Connection',
                        'pass',
                        f'Connected ({db_url.split(":")[0]})'
                    )
                    
                    # Check if matches table exists
                    if db_url.startswith('postgresql://'):
                        result = conn.execute(text("""
                            SELECT table_name 
                            FROM information_schema.tables 
                            WHERE table_name = 'matches'
                        """))
                    else:
                        result = conn.execute(text("""
                            SELECT name 
                            FROM sqlite_master 
                            WHERE type='table' AND name='matches'
                        """))
                    
                    if result.fetchone():
                        self._add_result(
                            'Database',
                            'Schema',
                            'pass',
                            'Matches table exists'
                        )
                        
                        # Check for last_synced_at column
                        if db_url.startswith('postgresql://'):
                            result = conn.execute(text("""
                                SELECT column_name 
                                FROM information_schema.columns 
                                WHERE table_name = 'matches' AND column_name = 'last_synced_at'
                            """))
                            has_column = result.fetchone() is not None
                        else:
                            result = conn.execute(text("PRAGMA table_info(matches)"))
                            columns = [row[1] for row in result.fetchall()]
                            has_column = 'last_synced_at' in columns
                        
                        if has_column:
                            self._add_result(
                                'Database',
                                'Migration',
                                'pass',
                                'last_synced_at column exists'
                            )
                        else:
                            self._add_result(
                                'Database',
                                'Migration',
                                'warning',
                                'last_synced_at column missing (migration needed)'
                            )
                    else:
                        self._add_result(
                            'Database',
                            'Schema',
                            'fail',
                            'Matches table not found'
                        )
                        
            except Exception as db_error:
                error_msg = str(db_error)[:100]
                
                # Check if this is a connection refused error
                if 'Connection refused' in error_msg or 'connect' in error_msg.lower():
                    self._add_result(
                        'Database',
                        'Connection',
                        'warning',
                        f'PostgreSQL not running - trying SQLite fallback'
                    )
                    
                    # Try SQLite fallback
                    sqlite_path = os.getenv('SQLITE_DATABASE_PATH', 'data/football.db')
                    if Path(sqlite_path).exists():
                        self._check_sqlite_database(sqlite_path)
                    else:
                        self._add_result(
                            'Database',
                            'Fallback',
                            'fail',
                            'SQLite fallback also not available'
                        )
                else:
                    self._add_result(
                        'Database',
                        'Connection',
                        'fail',
                        f'Connection failed: {error_msg}'
                    )
                    
        except ImportError:
            self._add_result(
                'Database',
                'Dependencies',
                'fail',
                'SQLAlchemy not installed'
            )
        except Exception as e:
            self._add_result(
                'Database',
                'Connection',
                'fail',
                f'Unexpected error: {str(e)[:100]}'
            )
    
    def _check_sqlite_database(self, sqlite_path: str):
        """Check SQLite database as fallback."""
        try:
            import sqlite3
            
            conn = sqlite3.connect(sqlite_path)
            cursor = conn.cursor()
            
            # Test connection
            cursor.execute("SELECT 1")
            
            self._add_result(
                'Database',
                'Connection (SQLite)',
                'pass',
                f'Connected to {sqlite_path}'
            )
            
            # Check tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='matches'")
            if cursor.fetchone():
                self._add_result(
                    'Database',
                    'Schema (SQLite)',
                    'pass',
                    'Matches table exists'
                )
                
                # Check for last_synced_at
                cursor.execute("PRAGMA table_info(matches)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'last_synced_at' in columns:
                    self._add_result(
                        'Database',
                        'Migration (SQLite)',
                        'pass',
                        'last_synced_at column exists'
                    )
                else:
                    self._add_result(
                        'Database',
                        'Migration (SQLite)',
                        'warning',
                        'last_synced_at column missing'
                    )
            
            conn.close()
            
        except Exception as e:
            self._add_result(
                'Database',
                'SQLite Fallback',
                'fail',
                f'SQLite check failed: {str(e)[:100]}'
            )
    
    def _check_redis(self):
        """Check Redis connectivity."""
        logger.info("\n🔴 Checking Redis...")
        
        redis_url = os.getenv('REDIS_URL', '')
        
        if not redis_url:
            self._add_result(
                'Redis',
                'Configuration',
                'warning',
                'Not configured (caching disabled)'
            )
            return
        
        try:
            import redis
            
            r = redis.from_url(redis_url, socket_connect_timeout=5)
            r.ping()
            
            self._add_result(
                'Redis',
                'Connection',
                'pass',
                'Connected and responding'
            )
            
            # Check memory usage
            info = r.info('memory')
            used_memory_mb = info.get('used_memory', 0) / (1024 * 1024)
            
            self._add_result(
                'Redis',
                'Memory',
                'pass',
                f'Using {used_memory_mb:.1f} MB',
                {'used_memory_mb': round(used_memory_mb, 1)}
            )
            
        except ImportError:
            self._add_result(
                'Redis',
                'Dependencies',
                'warning',
                'redis-py not installed (caching disabled)'
            )
        except Exception as e:
            self._add_result(
                'Redis',
                'Connection',
                'warning',
                f'Connection failed: {str(e)[:100]}'
            )
    
    def _check_api_connectivity(self):
        """Check external API connectivity."""
        logger.info("\n🌐 Checking API Connectivity...")
        
        api_key = os.getenv('FOOTBALL_DATA_API_KEY', '')
        
        if not api_key:
            self._add_result(
                'API',
                'Football-Data.org',
                'fail',
                'API key not configured'
            )
            return
        
        try:
            import requests

            # Set a short timeout
            response = requests.get(
                'https://api.football-data.org/v4/competitions',
                headers={'X-Auth-Token': api_key},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                count = len(data.get('competitions', []))
                self._add_result(
                    'API',
                    'Football-Data.org',
                    'pass',
                    f'Connected ({count} competitions available)'
                )
            elif response.status_code == 403:
                self._add_result(
                    'API',
                    'Football-Data.org',
                    'fail',
                    'Invalid API key'
                )
            else:
                self._add_result(
                    'API',
                    'Football-Data.org',
                    'warning',
                    f'HTTP {response.status_code}'
                )
                
        except ImportError:
            self._add_result(
                'API',
                'Dependencies',
                'fail',
                'requests library not installed'
            )
        except requests.exceptions.Timeout:
            self._add_result(
                'API',
                'Football-Data.org',
                'warning',
                'Connection timeout (network may be slow)'
            )
        except Exception as e:
            self._add_result(
                'API',
                'Football-Data.org',
                'warning',
                f'Connection failed: {str(e)[:100]}'
            )
    
    def _check_file_system(self):
        """Check file system and directories."""
        logger.info("\n📁 Checking File System...")
        
        required_dirs = [
            'data',
            'logs',
            'backups',
            'monitoring'
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                # Check write permission
                test_file = dir_path / '.health_check_test'
                try:
                    test_file.write_text('test')
                    test_file.unlink()
                    self._add_result(
                        'FileSystem',
                        f'{dir_name}/',
                        'pass',
                        'Exists and writable'
                    )
                except:
                    self._add_result(
                        'FileSystem',
                        f'{dir_name}/',
                        'warning',
                        'Exists but not writable'
                    )
            else:
                self._add_result(
                    'FileSystem',
                    f'{dir_name}/',
                    'warning',
                    'Does not exist (will be created)'
                )
    
    def _check_dependencies(self):
        """Check Python dependencies."""
        logger.info("\n📦 Checking Dependencies...")
        
        critical_deps = {
            'streamlit': 'Streamlit',
            'pandas': 'Pandas',
            'numpy': 'NumPy',
            'sqlalchemy': 'SQLAlchemy',
            'prometheus_client': 'Prometheus Client'
        }
        
        for module, name in critical_deps.items():
            try:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                self._add_result(
                    'Dependencies',
                    name,
                    'pass',
                    f'v{version}'
                )
            except ImportError:
                self._add_result(
                    'Dependencies',
                    name,
                    'fail',
                    'Not installed'
                )
    
    def _check_metrics_export(self):
        """Check Prometheus metrics export."""
        logger.info("\n📊 Checking Metrics Export...")
        
        try:
            from utils.metrics_exporter import (
                prediction_engine_predictions_total,
                validation_quality_score,
            )
            
            self._add_result(
                'Metrics',
                'Metrics Exporter',
                'pass',
                'Metrics module loaded'
            )
            
            # Check if metrics are registered
            from prometheus_client import REGISTRY
            
            metric_count = len(list(REGISTRY.collect()))
            
            self._add_result(
                'Metrics',
                'Prometheus Registry',
                'pass',
                f'{metric_count} metrics registered'
            )
            
        except ImportError as e:
            self._add_result(
                'Metrics',
                'Metrics Exporter',
                'warning',
                f'Import failed: {str(e)[:100]}'
            )
    
    def _check_data_quality(self):
        """Check data quality validator."""
        logger.info("\n✨ Checking Data Quality System...")
        
        try:
            from utils.real_data_validator import RealDataValidator
            
            validator = RealDataValidator()
            
            self._add_result(
                'DataQuality',
                'Validator',
                'pass',
                'Validator initialized'
            )
            
            # Check configuration
            min_quality = float(os.getenv('REAL_DATA_MIN_QUALITY_SCORE', '0.5'))
            
            self._add_result(
                'DataQuality',
                'Min Quality Threshold',
                'pass',
                f'{min_quality:.1%}',
                {'threshold': min_quality}
            )
            
        except ImportError as e:
            self._add_result(
                'DataQuality',
                'Validator',
                'warning',
                f'Import failed: {str(e)[:100]}'
            )
    
    def _check_performance(self):
        """Check system performance metrics."""
        logger.info("\n⚡ Checking Performance...")
        
        try:
            import psutil

            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            status = 'pass' if cpu_percent < 80 else 'warning'
            self._add_result(
                'Performance',
                'CPU Usage',
                status,
                f'{cpu_percent:.1f}%',
                {'cpu_percent': cpu_percent}
            )
            
            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            status = 'pass' if memory_percent < 80 else 'warning'
            self._add_result(
                'Performance',
                'Memory Usage',
                status,
                f'{memory_percent:.1f}% ({memory.used // (1024**3):.1f} GB / {memory.total // (1024**3):.1f} GB)',
                {'memory_percent': memory_percent}
            )
            
            # Disk
            disk = psutil.disk_usage('.')
            disk_percent = disk.percent
            status = 'pass' if disk_percent < 90 else 'warning'
            self._add_result(
                'Performance',
                'Disk Usage',
                status,
                f'{disk_percent:.1f}%',
                {'disk_percent': disk_percent}
            )
            
        except ImportError:
            self._add_result(
                'Performance',
                'Monitoring',
                'warning',
                'psutil not installed (performance monitoring disabled)'
            )


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run production health checks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--fail-on-warning',
        action='store_true',
        help='Exit with error code if warnings found'
    )
    
    args = parser.parse_args()
    
    # Run health checks
    system = HealthCheckSystem(verbose=args.verbose)
    report = system.run_all_checks()
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        logger.info(f"\n💾 Report saved to: {output_path}")
    
    # Determine exit code
    if report['summary']['failed'] > 0:
        exit_code = 1
    elif args.fail_on_warning and report['summary']['warnings'] > 0:
        exit_code = 1
    else:
        exit_code = 0
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
