#!/usr/bin/env python3
"""
Comprehensive Codebase Cleanup Analysis

Systematically scans and analyzes the entire codebase to detect and flag files or folders 
that are unused, redundant, or orphaned. Safely removes only non-essential elements while 
maintaining integrity of all dependencies, entry points, and runtime functionality.
"""

import ast
import importlib.util
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CodebaseCleanupAnalyzer:
    """Comprehensive codebase cleanup analyzer."""
    
    def __init__(self):
        self.project_root = project_root
        self.all_files = set()
        self.python_files = set()
        self.imported_modules = set()
        self.entry_points = set()
        self.critical_files = set()
        self.unused_files = set()
        self.redundant_files = set()
        self.orphaned_files = set()
        self.safe_to_remove = set()
        self.analysis_results = {}
        
        # Define critical patterns that must be preserved
        self.critical_patterns = [
            'main.py',
            'app.py',
            'enhanced_prediction_engine.py',
            'requirements.txt',
            '.env',
            'README.md',
            'database_migration.py',
            'models/',
            'utils/',
            'dashboard/optimized_production_app.py',
            'config/',
            'data/football.db'
        ]
        
        # Define patterns for files that are likely safe to remove
        self.cleanup_candidates = [
            r'.*\.log$',
            r'.*\.bak$',
            r'.*\.backup$',
            r'.*\.old$',
            r'.*\.tmp$',
            r'temp_.*\.py$',
            r'test_.*\.py$',
            r'.*_test\.py$',
            r'integration_backup_.*',
            r'.*\.pyc$',
            r'__pycache__',
            r'\.git',
            r'venv',
            r'\.venv'
        ]
    
    def run_comprehensive_analysis(self):
        """Run complete codebase cleanup analysis."""
        logger.info("🔍 Starting Comprehensive Codebase Cleanup Analysis")
        logger.info("=" * 70)
        
        try:
            # Phase 1: Discover all files
            self.discover_all_files()
            
            # Phase 2: Identify entry points and critical files
            self.identify_critical_files()
            
            # Phase 3: Analyze import dependencies
            self.analyze_import_dependencies()
            
            # Phase 4: Identify unused files
            self.identify_unused_files()
            
            # Phase 5: Identify redundant files
            self.identify_redundant_files()
            
            # Phase 6: Identify orphaned files
            self.identify_orphaned_files()
            
            # Phase 7: Determine safe removal candidates
            self.determine_safe_removal_candidates()
            
            # Phase 8: Generate cleanup report
            self.generate_cleanup_report()
            
            # Phase 9: Execute safe cleanup (if approved)
            self.execute_safe_cleanup()
            
            return True
            
        except Exception as e:
            logger.error(f"Cleanup analysis failed: {e}")
            return False
    
    def discover_all_files(self):
        """Discover all files in the codebase."""
        logger.info("📁 Discovering all files in codebase...")
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '__pycache__']]
            
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(self.project_root)
                
                self.all_files.add(str(relative_path))
                
                if file.endswith('.py'):
                    self.python_files.add(str(relative_path))
        
        logger.info(f"   Total files discovered: {len(self.all_files)}")
        logger.info(f"   Python files: {len(self.python_files)}")
    
    def identify_critical_files(self):
        """Identify critical files that must be preserved."""
        logger.info("🔒 Identifying critical files...")
        
        # Entry points
        entry_points = [
            'main.py',
            'app.py',
            'dashboard/optimized_production_app.py',
            'deploy_production.py',
            'launch_goaldiggers_crossleague.py'
        ]
        
        for entry_point in entry_points:
            if entry_point in self.all_files:
                self.entry_points.add(entry_point)
                self.critical_files.add(entry_point)
        
        # Critical system files
        critical_system_files = [
            'enhanced_prediction_engine.py',
            'database_migration.py',
            'requirements.txt',
            'README.md',
            'GETTING_STARTED.md',
            'QUICK_REFERENCE.md',
            'API_REFERENCE.md',
            'DEPLOYMENT_GUIDE.md'
        ]
        
        for critical_file in critical_system_files:
            if critical_file in self.all_files:
                self.critical_files.add(critical_file)
        
        # Critical directories
        critical_dirs = ['models', 'utils', 'config', 'database', 'data']
        for file_path in self.all_files:
            for critical_dir in critical_dirs:
                if file_path.startswith(critical_dir + '/'):
                    self.critical_files.add(file_path)
        
        logger.info(f"   Critical files identified: {len(self.critical_files)}")
        logger.info(f"   Entry points: {len(self.entry_points)}")

    def analyze_import_dependencies(self):
        """Analyze import dependencies in Python files."""
        logger.info("🔗 Analyzing import dependencies...")

        for python_file in self.python_files:
            try:
                file_path = self.project_root / python_file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parse imports using AST
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                self.imported_modules.add(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                self.imported_modules.add(node.module)
                except SyntaxError:
                    # Skip files with syntax errors
                    continue

                # Also check for string-based imports
                import_patterns = [
                    r'from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import',
                    r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
                    r'importlib\.import_module\([\'"]([^\'\"]+)[\'"]'
                ]

                for pattern in import_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        self.imported_modules.add(match)

            except Exception as e:
                logger.warning(f"Failed to analyze imports in {python_file}: {e}")

        logger.info(f"   Imported modules found: {len(self.imported_modules)}")

    def identify_unused_files(self):
        """Identify files that appear to be unused."""
        logger.info("🗑️ Identifying unused files...")

        # Check for Python files that are never imported
        for python_file in self.python_files:
            if python_file in self.critical_files:
                continue

            # Convert file path to module name
            module_name = python_file.replace('/', '.').replace('\\', '.').replace('.py', '')

            # Check if this module is imported anywhere
            is_imported = False
            for imported_module in self.imported_modules:
                if (module_name in imported_module or
                    imported_module in module_name or
                    module_name.split('.')[-1] in imported_module):
                    is_imported = True
                    break

            if not is_imported:
                self.unused_files.add(python_file)

        # Check for non-Python files that might be unused
        for file_path in self.all_files:
            if file_path.endswith('.py') or file_path in self.critical_files:
                continue

            # Check for common unused file patterns
            for pattern in self.cleanup_candidates:
                if re.match(pattern, file_path):
                    self.unused_files.add(file_path)
                    break

        logger.info(f"   Unused files identified: {len(self.unused_files)}")

    def identify_redundant_files(self):
        """Identify redundant or duplicate files."""
        logger.info("📋 Identifying redundant files...")

        # Look for backup files and duplicates
        file_groups = {}

        for file_path in self.all_files:
            if file_path in self.critical_files:
                continue

            # Group files by base name
            base_name = Path(file_path).stem
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)

        # Identify redundant files in groups
        for base_name, files in file_groups.items():
            if len(files) > 1:
                # Sort by preference (keep the main file, mark others as redundant)
                files.sort(key=lambda x: (
                    '.bak' in x,
                    '.backup' in x,
                    '.old' in x,
                    '_backup' in x,
                    'backup_' in x,
                    len(x)  # Prefer shorter names
                ))

                # Mark all but the first as redundant
                for redundant_file in files[1:]:
                    if redundant_file not in self.critical_files:
                        self.redundant_files.add(redundant_file)

        logger.info(f"   Redundant files identified: {len(self.redundant_files)}")

    def identify_orphaned_files(self):
        """Identify orphaned files (files in directories that are no longer used)."""
        logger.info("🏚️ Identifying orphaned files...")

        # Check for files in backup directories
        backup_patterns = [
            r'backup/',
            r'archive/',
            r'integration_backup_',
            r'old/',
            r'unused/'
        ]

        for file_path in self.all_files:
            if file_path in self.critical_files:
                continue

            for pattern in backup_patterns:
                if pattern in file_path:
                    self.orphaned_files.add(file_path)
                    break

        logger.info(f"   Orphaned files identified: {len(self.orphaned_files)}")

    def determine_safe_removal_candidates(self):
        """Determine which files are safe to remove."""
        logger.info("✅ Determining safe removal candidates...")

        # Combine all categories of files that might be safe to remove
        potential_removals = (
            self.unused_files |
            self.redundant_files |
            self.orphaned_files
        )

        # Filter out any critical files (double-check)
        for file_path in potential_removals:
            if file_path not in self.critical_files:
                # Additional safety checks
                if self._is_safe_to_remove(file_path):
                    self.safe_to_remove.add(file_path)

        logger.info(f"   Safe removal candidates: {len(self.safe_to_remove)}")

    def _is_safe_to_remove(self, file_path: str) -> bool:
        """Check if a file is safe to remove with additional safety checks."""

        # Never remove these critical patterns
        never_remove = [
            'main.py',
            'app.py',
            'enhanced_prediction_engine.py',
            'requirements.txt',
            'README.md',
            'database_migration.py'
        ]

        for critical in never_remove:
            if critical in file_path:
                return False

        # Safe patterns to remove
        safe_patterns = [
            r'.*\.log$',
            r'.*\.bak$',
            r'.*\.backup$',
            r'.*\.old$',
            r'.*\.tmp$',
            r'temp_.*',
            r'integration_backup_.*',
            r'.*_backup.*',
            r'backup_.*',
            r'test_.*\.py$',
            r'.*_test\.py$'
        ]

        for pattern in safe_patterns:
            if re.match(pattern, file_path):
                return True

        # Check if it's in a backup directory
        backup_dirs = ['backup/', 'archive/', 'integration_backup_']
        for backup_dir in backup_dirs:
            if backup_dir in file_path:
                return True

        return False

    def generate_cleanup_report(self):
        """Generate comprehensive cleanup report."""
        logger.info("📋 Generating cleanup report...")

        self.analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': len(self.all_files),
                'python_files': len(self.python_files),
                'critical_files': len(self.critical_files),
                'unused_files': len(self.unused_files),
                'redundant_files': len(self.redundant_files),
                'orphaned_files': len(self.orphaned_files),
                'safe_to_remove': len(self.safe_to_remove)
            },
            'categories': {
                'critical_files': sorted(list(self.critical_files)),
                'unused_files': sorted(list(self.unused_files)),
                'redundant_files': sorted(list(self.redundant_files)),
                'orphaned_files': sorted(list(self.orphaned_files)),
                'safe_to_remove': sorted(list(self.safe_to_remove))
            },
            'entry_points': sorted(list(self.entry_points)),
            'imported_modules': sorted(list(self.imported_modules))
        }

        # Save detailed report
        report_file = self.project_root / 'codebase_cleanup_analysis.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2)

        # Generate summary report
        summary_report = f"""
# 🧹 Codebase Cleanup Analysis Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Summary Statistics

- **Total Files**: {len(self.all_files)}
- **Python Files**: {len(self.python_files)}
- **Critical Files**: {len(self.critical_files)}
- **Entry Points**: {len(self.entry_points)}

## 🗑️ Cleanup Candidates

- **Unused Files**: {len(self.unused_files)}
- **Redundant Files**: {len(self.redundant_files)}
- **Orphaned Files**: {len(self.orphaned_files)}
- **Safe to Remove**: {len(self.safe_to_remove)}

## 🔒 Critical Files (Protected)

{chr(10).join(f"- {f}" for f in sorted(self.critical_files)[:20])}
{"..." if len(self.critical_files) > 20 else ""}

## ✅ Safe Removal Candidates

{chr(10).join(f"- {f}" for f in sorted(self.safe_to_remove)[:30])}
{"..." if len(self.safe_to_remove) > 30 else ""}

## 📈 Potential Space Savings

Removing {len(self.safe_to_remove)} files could free up disk space and reduce codebase complexity.

**Detailed analysis saved to:** `codebase_cleanup_analysis.json`
"""

        summary_file = self.project_root / 'CODEBASE_CLEANUP_REPORT.md'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)

        logger.info(f"📊 Cleanup Analysis Summary:")
        logger.info(f"   Total files: {len(self.all_files)}")
        logger.info(f"   Critical files (protected): {len(self.critical_files)}")
        logger.info(f"   Safe removal candidates: {len(self.safe_to_remove)}")
        logger.info(f"   Detailed report: {report_file}")
        logger.info(f"   Summary report: {summary_file}")

    def execute_safe_cleanup(self):
        """Execute safe cleanup of identified files."""
        logger.info("🧹 Executing safe cleanup...")

        if not self.safe_to_remove:
            logger.info("   No files identified for safe removal.")
            return

        # Create backup directory for removed files
        backup_dir = self.project_root / 'cleanup_backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir.mkdir(parents=True, exist_ok=True)

        removed_files = []
        failed_removals = []

        for file_path in self.safe_to_remove:
            try:
                full_path = self.project_root / file_path

                if full_path.exists():
                    # Create backup
                    backup_path = backup_dir / file_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)

                    if full_path.is_file():
                        # Backup file
                        import shutil
                        shutil.copy2(full_path, backup_path)

                        # Remove original
                        full_path.unlink()
                        removed_files.append(file_path)
                        logger.info(f"   Removed: {file_path}")

                    elif full_path.is_dir():
                        # Backup directory
                        import shutil
                        shutil.copytree(full_path, backup_path, dirs_exist_ok=True)

                        # Remove original
                        shutil.rmtree(full_path)
                        removed_files.append(file_path)
                        logger.info(f"   Removed directory: {file_path}")

            except Exception as e:
                failed_removals.append((file_path, str(e)))
                logger.warning(f"   Failed to remove {file_path}: {e}")

        # Generate cleanup summary
        cleanup_summary = {
            'timestamp': datetime.now().isoformat(),
            'backup_location': str(backup_dir),
            'removed_files': removed_files,
            'failed_removals': failed_removals,
            'total_removed': len(removed_files),
            'total_failed': len(failed_removals)
        }

        # Save cleanup log
        cleanup_log_file = self.project_root / 'cleanup_execution_log.json'
        with open(cleanup_log_file, 'w', encoding='utf-8') as f:
            json.dump(cleanup_summary, f, indent=2)

        logger.info(f"🎉 Cleanup completed:")
        logger.info(f"   Files removed: {len(removed_files)}")
        logger.info(f"   Failed removals: {len(failed_removals)}")
        logger.info(f"   Backup location: {backup_dir}")
        logger.info(f"   Cleanup log: {cleanup_log_file}")

        # Clean up empty directories
        self._cleanup_empty_directories()

    def _cleanup_empty_directories(self):
        """Remove empty directories after file cleanup."""
        logger.info("📁 Cleaning up empty directories...")

        removed_dirs = []

        # Walk through directories bottom-up
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name

                # Skip critical directories
                if any(critical in str(dir_path) for critical in ['models', 'utils', 'config', 'database', 'data']):
                    continue

                try:
                    # Check if directory is empty
                    if dir_path.exists() and not any(dir_path.iterdir()):
                        relative_path = dir_path.relative_to(self.project_root)
                        dir_path.rmdir()
                        removed_dirs.append(str(relative_path))
                        logger.info(f"   Removed empty directory: {relative_path}")

                except Exception as e:
                    logger.warning(f"   Failed to remove empty directory {dir_path}: {e}")

        if removed_dirs:
            logger.info(f"   Empty directories removed: {len(removed_dirs)}")
        else:
            logger.info("   No empty directories found.")

def main():
    """Main cleanup analysis function."""
    analyzer = CodebaseCleanupAnalyzer()
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        logger.info("🎉 Codebase cleanup analysis completed successfully!")
        return 0
    else:
        logger.error("❌ Codebase cleanup analysis failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
