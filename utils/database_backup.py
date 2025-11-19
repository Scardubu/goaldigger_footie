#!/usr/bin/env python3
"""
Database Backup and Restoration Utility for GoalDiggers Platform
===============================================================

This script provides automated database backup and restoration functionality
for the GoalDiggers football analytics platform.

Features:
- Scheduled daily backups
- Cloud upload options (optional)
- Automatic corruption detection
- Point-in-time restoration capability

Usage:
    python utils/database_backup.py --backup
    python utils/database_backup.py --restore <backup_file>
    python utils/database_backup.py --auto-detect-corruption
    python utils/database_backup.py --cloud-upload
"""

import argparse
import datetime
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from utils.logging_config import configure_logging

# Configure logging using centralized configuration
configure_logging()
logger = logging.getLogger(__name__)

class DatabaseBackupManager:
    """Handle database backup and restoration operations."""
    
    def __init__(self):
        """Initialize the backup manager."""
        self.project_root = Path(__file__).parent.parent.absolute()
        self.data_dir = self.project_root / "data"
        self.backup_dir = self.data_dir / "backups"
        self.db_file = self.data_dir / "football.db"
        
        # Create backup directory if it doesn't exist
        self.backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration
        self.max_backups = 14  # Keep 2 weeks of daily backups
        self.cloud_enabled = False
    
    def create_backup(self, backup_name: Optional[str] = None) -> Path:
        """Create a backup of the database.
        
        Args:
            backup_name: Optional backup name. If not provided, uses timestamp.
            
        Returns:
            Path to the backup file
        """
        if not self.db_file.exists():
            raise FileNotFoundError(f"Database file not found: {self.db_file}")
        
        # Generate backup filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if backup_name:
            backup_filename = f"{backup_name}_{timestamp}.db"
        else:
            backup_filename = f"football_backup_{timestamp}.db"
        
        backup_path = self.backup_dir / backup_filename
        
        # Create backup
        logger.info(f"Creating backup: {backup_path}")
        
        try:
            # Check if database is locked
            conn = sqlite3.connect(str(self.db_file))
            conn.execute("PRAGMA quick_check")
            conn.close()
            
            # Copy the database file
            shutil.copy2(self.db_file, backup_path)
            
            # Verify backup integrity
            self._verify_backup(backup_path)
            
            logger.info(f"Backup created successfully: {backup_path}")
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return backup_path
            
        except sqlite3.Error as e:
            logger.error(f"Database error during backup: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise
    
    def restore_from_backup(self, backup_path: Union[str, Path]) -> bool:
        """Restore database from a backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful
        """
        if isinstance(backup_path, str):
            backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_path}")
        
        # First verify backup integrity
        self._verify_backup(backup_path)
        
        # Create additional backup of current state before restoring
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pre_restore_backup = self.backup_dir / f"pre_restore_{timestamp}.db"
        
        if self.db_file.exists():
            logger.info(f"Creating pre-restore backup: {pre_restore_backup}")
            shutil.copy2(self.db_file, pre_restore_backup)
        
        # Restore from backup
        logger.info(f"Restoring from backup: {backup_path}")
        
        try:
            # Copy backup file to database location
            shutil.copy2(backup_path, self.db_file)
            
            logger.info(f"Database restored successfully from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            
            # Try to restore from pre-restore backup if available
            if pre_restore_backup.exists():
                logger.info("Attempting to revert to pre-restore state")
                try:
                    shutil.copy2(pre_restore_backup, self.db_file)
                    logger.info("Successfully reverted to pre-restore state")
                except Exception as revert_error:
                    logger.error(f"Failed to revert to pre-restore state: {revert_error}")
            
            raise
    
    def detect_corruption(self) -> Dict:
        """Check for database corruption.
        
        Returns:
            Dict with corruption check results
        """
        logger.info("Checking database for corruption...")
        
        results = {
            "is_corrupted": False,
            "errors": [],
            "checked": datetime.datetime.now().isoformat()
        }
        
        if not self.db_file.exists():
            results["is_corrupted"] = True
            results["errors"].append("Database file not found")
            return results
        
        try:
            # Connect to database
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            
            if integrity_result != "ok":
                results["is_corrupted"] = True
                results["errors"].append(f"Integrity check failed: {integrity_result}")
            
            # Check if all required tables exist
            required_tables = ['teams', 'matches', 'predictions', 'team_stats', 'leagues']
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = [table for table in required_tables if table not in existing_tables]
            if missing_tables:
                results["is_corrupted"] = True
                results["errors"].append(f"Missing tables: {', '.join(missing_tables)}")
            
            # Close connection
            conn.close()
            
        except sqlite3.Error as e:
            results["is_corrupted"] = True
            results["errors"].append(f"SQLite error: {str(e)}")
        except Exception as e:
            results["is_corrupted"] = True
            results["errors"].append(f"Error checking corruption: {str(e)}")
        
        # Log results
        if results["is_corrupted"]:
            logger.warning(f"Database corruption detected: {', '.join(results['errors'])}")
        else:
            logger.info("Database integrity check passed")
        
        return results
    
    def auto_restore_if_corrupted(self) -> Dict:
        """Automatically detect corruption and restore from latest backup if needed.
        
        Returns:
            Dict with operation results
        """
        results = {
            "action_taken": "none",
            "success": True,
            "message": ""
        }
        
        # Check for corruption
        corruption_check = self.detect_corruption()
        
        if corruption_check["is_corrupted"]:
            logger.warning("Database corruption detected, attempting auto-restore...")
            
            # Find most recent backup
            backups = self._get_available_backups()
            
            if not backups:
                results["action_taken"] = "none"
                results["success"] = False
                results["message"] = "Corruption detected but no backups available"
                logger.error("No backups available for auto-restore")
                return results
            
            # Get most recent backup
            latest_backup = backups[0]["path"]
            
            try:
                # Restore from backup
                self.restore_from_backup(latest_backup)
                
                results["action_taken"] = "restore"
                results["success"] = True
                results["message"] = f"Successfully restored from backup: {latest_backup.name}"
                results["backup_used"] = str(latest_backup)
                
                logger.info(f"Auto-restore successful using backup: {latest_backup}")
                
            except Exception as e:
                results["action_taken"] = "restore_attempted"
                results["success"] = False
                results["message"] = f"Auto-restore failed: {str(e)}"
                logger.error(f"Auto-restore failed: {e}")
        else:
            results["message"] = "No corruption detected, no action needed"
        
        return results
    
    def cloud_upload(self) -> Dict:
        """Upload latest backup to cloud storage.
        
        This is a placeholder for cloud integration - implement with your preferred cloud
        storage provider (S3, Google Cloud Storage, Azure Blob Storage, etc.)
        
        Returns:
            Dict with upload results
        """
        if not self.cloud_enabled:
            logger.warning("Cloud upload not configured")
            return {
                "success": False, 
                "message": "Cloud upload not configured. Update code to implement with your preferred provider."
            }
        
        # Find most recent backup
        backups = self._get_available_backups()
        
        if not backups:
            return {"success": False, "message": "No backups available for upload"}
        
        latest_backup = backups[0]["path"]
        
        # PLACEHOLDER FOR CLOUD UPLOAD IMPLEMENTATION
        # This would typically use boto3 for AWS S3, google-cloud-storage, etc.
        logger.info(f"Would upload {latest_backup} to cloud storage")
        
        # Simulate cloud upload for now
        time.sleep(2)
        
        return {
            "success": True,
            "message": "Backup uploaded to cloud storage (simulated)",
            "backup": str(latest_backup),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    def _verify_backup(self, backup_path: Path) -> bool:
        """Verify backup file integrity.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if backup is valid
        """
        logger.info(f"Verifying backup integrity: {backup_path}")
        
        try:
            # Try to open and check database integrity
            conn = sqlite3.connect(str(backup_path))
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            
            # Check if all required tables exist
            required_tables = ['teams', 'matches', 'predictions', 'team_stats', 'leagues']
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = [table for table in required_tables if table not in existing_tables]
            
            conn.close()
            
            if integrity_result != "ok":
                logger.error(f"Backup integrity check failed: {integrity_result}")
                return False
            
            if missing_tables:
                logger.error(f"Backup is missing tables: {', '.join(missing_tables)}")
                return False
            
            logger.info("Backup verified successfully")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error verifying backup: {e}")
            return False
        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return False
    
    def _cleanup_old_backups(self):
        """Remove old backups exceeding the maximum backup count."""
        backups = self._get_available_backups()
        
        # Skip cleanup if we're under the limit
        if len(backups) <= self.max_backups:
            return
        
        # Remove oldest backups
        for backup in backups[self.max_backups:]:
            try:
                logger.info(f"Removing old backup: {backup['path']}")
                backup['path'].unlink()
            except Exception as e:
                logger.error(f"Error removing backup {backup['path']}: {e}")
    
    def _get_available_backups(self) -> List[Dict]:
        """Get list of available backups sorted by date (newest first).
        
        Returns:
            List of backup info dicts
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*.db"):
            try:
                mtime = backup_file.stat().st_mtime
                size_mb = backup_file.stat().st_size / (1024 * 1024)
                
                backups.append({
                    "path": backup_file,
                    "name": backup_file.name,
                    "timestamp": mtime,
                    "date": datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "size_mb": round(size_mb, 2)
                })
            except Exception as e:
                logger.error(f"Error getting backup info for {backup_file}: {e}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return backups


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='GoalDiggers Database Backup Utility')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--backup', action='store_true', help='Create a backup')
    group.add_argument('--restore', type=str, help='Restore from a specified backup file')
    group.add_argument('--list', action='store_true', help='List available backups')
    group.add_argument('--auto-detect-corruption', action='store_true', help='Detect and auto-fix corruption')
    group.add_argument('--cloud-upload', action='store_true', help='Upload latest backup to cloud storage')
    
    parser.add_argument('--name', type=str, help='Optional backup name (used with --backup)')
    
    args = parser.parse_args()
    
    backup_manager = DatabaseBackupManager()
    
    try:
        if args.backup:
            backup_path = backup_manager.create_backup(args.name)
            print(f"Backup created: {backup_path}")
            return 0
        
        elif args.restore:
            restore_path = Path(args.restore)
            if not restore_path.is_absolute():
                restore_path = backup_manager.backup_dir / restore_path
            
            backup_manager.restore_from_backup(restore_path)
            print(f"Database restored from: {restore_path}")
            return 0
        
        elif args.list:
            backups = backup_manager._get_available_backups()
            
            if not backups:
                print("No backups available")
                return 0
            
            print(f"Available backups ({len(backups)}):")
            print("-" * 80)
            print(f"{'Name':<40} {'Date':<20} {'Size':>10}")
            print("-" * 80)
            
            for backup in backups:
                print(f"{backup['name']:<40} {backup['date']:<20} {backup['size_mb']:>8.1f} MB")
            
            return 0
        
        elif args.auto_detect_corruption:
            result = backup_manager.auto_restore_if_corrupted()
            
            if result["action_taken"] == "restore":
                print(f"Corruption detected and fixed: {result['message']}")
                return 0
            elif result["action_taken"] == "restore_attempted" and not result["success"]:
                print(f"Corruption detected but fix failed: {result['message']}")
                return 1
            else:
                print(result["message"])
                return 0
        
        elif args.cloud_upload:
            result = backup_manager.cloud_upload()
            
            if result["success"]:
                print(f"Cloud upload successful: {result['message']}")
                return 0
            else:
                print(f"Cloud upload failed: {result['message']}")
                return 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
