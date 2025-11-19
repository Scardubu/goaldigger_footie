#!/usr/bin/env python3
"""
Database Restoration Utility for GoalDiggers Platform
=====================================================

This script provides database restoration functionality from backups
for the GoalDiggers football analytics platform.

Usage:
    python utils/database_restore.py --list
    python utils/database_restore.py --restore <backup_name>
    python utils/database_restore.py --auto-detect-corruption
"""

import argparse
import sys

# Import the backup manager for restoration functionality
from database_backup import DatabaseBackupManager


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='GoalDiggers Database Restoration Utility')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true', help='List available backups')
    group.add_argument('--restore', type=str, help='Restore from a specified backup file')
    group.add_argument('--auto-detect-corruption', action='store_true', help='Detect and auto-fix corruption')
    
    args = parser.parse_args()
    
    backup_manager = DatabaseBackupManager()
    
    try:
        if args.list:
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
        
        elif args.restore:
            backup_name = args.restore
            
            # Find matching backup
            backups = backup_manager._get_available_backups()
            matching_backups = [b for b in backups if backup_name in b['name']]
            
            if not matching_backups:
                print(f"No backup found matching: {backup_name}")
                return 1
            
            if len(matching_backups) > 1:
                print(f"Multiple backups found matching: {backup_name}")
                print("Please be more specific. Available matches:")
                for backup in matching_backups:
                    print(f"  {backup['name']}")
                return 1
            
            restore_path = matching_backups[0]['path']
            backup_manager.restore_from_backup(restore_path)
            print(f"Database restored from: {restore_path}")
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
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
