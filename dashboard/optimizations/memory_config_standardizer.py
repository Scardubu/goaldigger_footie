#!/usr/bin/env python3
"""
Memory Configuration Standardizer
Phase 3A: Technical Debt Resolution - Configuration Standardization

This script standardizes memory configuration to 400MB target across all
components, addressing the memory configuration inconsistencies identified
in the technical debt analysis.

Files Updated:
- config/enhanced_config.yaml (2GB → 400MB)
- Various validation scripts (outdated memory targets)
- Dashboard performance metrics (inconsistent thresholds)
- Component initialization files (mixed memory limits)

Key Features:
- Automated configuration file updates
- Validation script synchronization
- Performance metric standardization
- Backup and rollback capabilities
"""

import os
import sys
import yaml
import json
import logging
import shutil
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryConfigStandardizer:
    """
    Standardizes memory configuration across all GoalDiggers platform components.
    Ensures consistent 400MB memory target implementation.
    """
    
    def __init__(self, target_memory_mb: int = 400):
        """Initialize memory configuration standardizer."""
        self.target_memory_mb = target_memory_mb
        self.backup_dir = Path("config_backups") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.updated_files = []
        self.errors = []
        
        # Create backup directory
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎯 Memory config standardizer initialized (Target: {target_memory_mb}MB)")
    
    def standardize_all_configs(self) -> Dict[str, Any]:
        """Standardize memory configuration across all components."""
        logger.info("🚀 Starting comprehensive memory configuration standardization")
        
        results = {
            'updated_files': [],
            'errors': [],
            'summary': {}
        }
        
        # 1. Update YAML configuration files
        yaml_results = self._update_yaml_configs()
        results['updated_files'].extend(yaml_results)
        
        # 2. Update Python configuration files
        python_results = self._update_python_configs()
        results['updated_files'].extend(python_results)
        
        # 3. Update validation scripts
        validation_results = self._update_validation_scripts()
        results['updated_files'].extend(validation_results)
        
        # 4. Update dashboard performance metrics
        dashboard_results = self._update_dashboard_configs()
        results['updated_files'].extend(dashboard_results)
        
        # 5. Update test files
        test_results = self._update_test_configs()
        results['updated_files'].extend(test_results)
        
        # Generate summary
        results['summary'] = {
            'total_files_updated': len(results['updated_files']),
            'yaml_files': len(yaml_results),
            'python_files': len(python_results),
            'validation_scripts': len(validation_results),
            'dashboard_configs': len(dashboard_results),
            'test_files': len(test_results),
            'target_memory_mb': self.target_memory_mb,
            'backup_location': str(self.backup_dir)
        }
        
        results['errors'] = self.errors
        
        logger.info(f"✅ Configuration standardization completed: {len(results['updated_files'])} files updated")
        return results
    
    def _update_yaml_configs(self) -> List[str]:
        """Update YAML configuration files."""
        logger.info("📝 Updating YAML configuration files")
        
        updated_files = []
        yaml_files = [
            'config/enhanced_config.yaml',
            'config/dashboard_config.yaml',
            'config/ml_config.yaml',
            'config/performance_config.yaml'
        ]
        
        for file_path in yaml_files:
            if os.path.exists(file_path):
                try:
                    # Backup original file
                    self._backup_file(file_path)
                    
                    # Load and update YAML
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f) or {}
                    
                    # Update memory-related configurations
                    updated = self._update_yaml_memory_config(config)
                    
                    if updated:
                        # Write updated configuration
                        with open(file_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False, indent=2)
                        
                        updated_files.append(file_path)
                        logger.info(f"✅ Updated YAML config: {file_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to update YAML config {file_path}: {e}"
                    logger.error(f"❌ {error_msg}")
                    self.errors.append(error_msg)
        
        return updated_files
    
    def _update_yaml_memory_config(self, config: Dict[str, Any]) -> bool:
        """Update memory configuration in YAML config."""
        updated = False
        
        # Common memory configuration keys to update
        memory_keys = [
            'memory_limit_mb',
            'memory_target_mb',
            'max_memory_mb',
            'memory_threshold_mb',
            'dashboard_memory_mb',
            'component_memory_mb'
        ]
        
        def update_nested_dict(d: Dict[str, Any]) -> bool:
            local_updated = False
            for key, value in d.items():
                if isinstance(value, dict):
                    local_updated |= update_nested_dict(value)
                elif key in memory_keys:
                    old_value = value
                    if isinstance(value, (int, float)) and value != self.target_memory_mb:
                        d[key] = self.target_memory_mb
                        local_updated = True
                        logger.info(f"  Updated {key}: {old_value} → {self.target_memory_mb}")
                elif key == 'memory_limit' and isinstance(value, str):
                    # Handle string values like "2GB", "1024MB"
                    if 'GB' in value or ('MB' in value and int(value.replace('MB', '')) != self.target_memory_mb):
                        d[key] = f"{self.target_memory_mb}MB"
                        local_updated = True
                        logger.info(f"  Updated {key}: {value} → {self.target_memory_mb}MB")
            return local_updated
        
        updated = update_nested_dict(config)
        return updated
    
    def _update_python_configs(self) -> List[str]:
        """Update Python configuration files."""
        logger.info("🐍 Updating Python configuration files")
        
        updated_files = []
        python_files = [
            'config/config.py',
            'dashboard/config.py',
            'ml/config.py',
            'utils/config.py'
        ]
        
        for file_path in python_files:
            if os.path.exists(file_path):
                try:
                    # Backup original file
                    self._backup_file(file_path)
                    
                    # Read file content
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Update memory configurations
                    updated_content, was_updated = self._update_python_memory_config(content)
                    
                    if was_updated:
                        # Write updated content
                        with open(file_path, 'w') as f:
                            f.write(updated_content)
                        
                        updated_files.append(file_path)
                        logger.info(f"✅ Updated Python config: {file_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to update Python config {file_path}: {e}"
                    logger.error(f"❌ {error_msg}")
                    self.errors.append(error_msg)
        
        return updated_files
    
    def _update_python_memory_config(self, content: str) -> tuple[str, bool]:
        """Update memory configuration in Python file content."""
        updated_content = content
        was_updated = False
        
        # Patterns to match memory configuration
        patterns = [
            (r'MEMORY_LIMIT_MB\s*=\s*\d+', f'MEMORY_LIMIT_MB = {self.target_memory_mb}'),
            (r'MEMORY_TARGET_MB\s*=\s*\d+', f'MEMORY_TARGET_MB = {self.target_memory_mb}'),
            (r'MAX_MEMORY_MB\s*=\s*\d+', f'MAX_MEMORY_MB = {self.target_memory_mb}'),
            (r'memory_limit_mb\s*=\s*\d+', f'memory_limit_mb = {self.target_memory_mb}'),
            (r'memory_target_mb\s*=\s*\d+', f'memory_target_mb = {self.target_memory_mb}'),
            (r'"memory_limit_mb":\s*\d+', f'"memory_limit_mb": {self.target_memory_mb}'),
            (r"'memory_limit_mb':\s*\d+", f"'memory_limit_mb': {self.target_memory_mb}"),
        ]
        
        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, updated_content)
            if new_content != updated_content:
                updated_content = new_content
                was_updated = True
                logger.info(f"  Applied pattern: {pattern}")
        
        return updated_content, was_updated
    
    def _update_validation_scripts(self) -> List[str]:
        """Update validation scripts with correct memory targets."""
        logger.info("🔍 Updating validation scripts")
        
        updated_files = []
        validation_files = [
            'test_dashboard_consolidation.py',
            'cross_league_performance_validation.py',
            'cross_league_performance_assessment.py',
            'validation/memory_validation.py',
            'validation/performance_validation.py'
        ]
        
        for file_path in validation_files:
            if os.path.exists(file_path):
                try:
                    # Backup original file
                    self._backup_file(file_path)
                    
                    # Read and update file
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    updated_content, was_updated = self._update_validation_memory_targets(content)
                    
                    if was_updated:
                        with open(file_path, 'w') as f:
                            f.write(updated_content)
                        
                        updated_files.append(file_path)
                        logger.info(f"✅ Updated validation script: {file_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to update validation script {file_path}: {e}"
                    logger.error(f"❌ {error_msg}")
                    self.errors.append(error_msg)
        
        return updated_files
    
    def _update_validation_memory_targets(self, content: str) -> tuple[str, bool]:
        """Update memory targets in validation scripts."""
        updated_content = content
        was_updated = False
        
        # Patterns for validation scripts
        patterns = [
            (r'memory_usage_mb.*?>\s*\d+', f'memory_usage_mb > {self.target_memory_mb}'),
            (r'memory.*?target.*?\d+', f'memory target {self.target_memory_mb}MB'),
            (r'<\s*\d+MB.*?memory', f'<{self.target_memory_mb}MB memory'),
            (r'memory.*?limit.*?\d+', f'memory limit {self.target_memory_mb}'),
            (r'target_mb.*?=.*?\d+', f'target_mb = {self.target_memory_mb}'),
            (r'memory_target.*?=.*?\d+', f'memory_target = {self.target_memory_mb}'),
        ]
        
        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, updated_content, flags=re.IGNORECASE)
            if new_content != updated_content:
                updated_content = new_content
                was_updated = True
                logger.info(f"  Applied validation pattern: {pattern}")
        
        return updated_content, was_updated
    
    def _update_dashboard_configs(self) -> List[str]:
        """Update dashboard configuration files."""
        logger.info("📊 Updating dashboard configurations")
        
        updated_files = []
        dashboard_files = [
            'dashboard/unified_goaldiggers_dashboard.py',
            'dashboard/dashboard_router.py',
            'dashboard/optimizations/render_optimization.py'
        ]
        
        for file_path in dashboard_files:
            if os.path.exists(file_path):
                try:
                    # Backup original file
                    self._backup_file(file_path)
                    
                    # Read and update file
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    updated_content, was_updated = self._update_dashboard_memory_config(content)
                    
                    if was_updated:
                        with open(file_path, 'w') as f:
                            f.write(updated_content)
                        
                        updated_files.append(file_path)
                        logger.info(f"✅ Updated dashboard config: {file_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to update dashboard config {file_path}: {e}"
                    logger.error(f"❌ {error_msg}")
                    self.errors.append(error_msg)
        
        return updated_files
    
    def _update_dashboard_memory_config(self, content: str) -> tuple[str, bool]:
        """Update memory configuration in dashboard files."""
        updated_content = content
        was_updated = False
        
        # Dashboard-specific patterns
        patterns = [
            (r'memory_target_mb:\s*int\s*=\s*\d+', f'memory_target_mb: int = {self.target_memory_mb}'),
            (r'memory_limit_mb.*?=.*?\d+', f'memory_limit_mb = {self.target_memory_mb}'),
            (r'MEMORY_LIMIT.*?=.*?\d+', f'MEMORY_LIMIT = {self.target_memory_mb}'),
            (r'memory.*?threshold.*?\d+', f'memory threshold {self.target_memory_mb}'),
        ]
        
        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, updated_content)
            if new_content != updated_content:
                updated_content = new_content
                was_updated = True
                logger.info(f"  Applied dashboard pattern: {pattern}")
        
        return updated_content, was_updated
    
    def _update_test_configs(self) -> List[str]:
        """Update test configuration files."""
        logger.info("🧪 Updating test configurations")
        
        updated_files = []
        test_files = [
            'tests/test_memory_optimization.py',
            'tests/test_performance.py',
            'tests/test_dashboard_performance.py'
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                try:
                    # Backup original file
                    self._backup_file(file_path)
                    
                    # Read and update file
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    updated_content, was_updated = self._update_test_memory_config(content)
                    
                    if was_updated:
                        with open(file_path, 'w') as f:
                            f.write(updated_content)
                        
                        updated_files.append(file_path)
                        logger.info(f"✅ Updated test config: {file_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to update test config {file_path}: {e}"
                    logger.error(f"❌ {error_msg}")
                    self.errors.append(error_msg)
        
        return updated_files
    
    def _update_test_memory_config(self, content: str) -> tuple[str, bool]:
        """Update memory configuration in test files."""
        updated_content = content
        was_updated = False
        
        # Test-specific patterns
        patterns = [
            (r'assert.*?memory.*?<\s*\d+', f'assert memory < {self.target_memory_mb}'),
            (r'memory.*?limit.*?\d+', f'memory limit {self.target_memory_mb}'),
            (r'expected.*?memory.*?\d+', f'expected memory {self.target_memory_mb}'),
            (r'memory.*?target.*?\d+', f'memory target {self.target_memory_mb}'),
        ]
        
        for pattern, replacement in patterns:
            new_content = re.sub(pattern, replacement, updated_content, flags=re.IGNORECASE)
            if new_content != updated_content:
                updated_content = new_content
                was_updated = True
                logger.info(f"  Applied test pattern: {pattern}")
        
        return updated_content, was_updated
    
    def _backup_file(self, file_path: str):
        """Create backup of file before modification."""
        try:
            backup_path = self.backup_dir / Path(file_path).name
            shutil.copy2(file_path, backup_path)
            logger.debug(f"📁 Backed up {file_path} to {backup_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to backup {file_path}: {e}")
    
    def rollback_changes(self) -> Dict[str, Any]:
        """Rollback all changes using backups."""
        logger.info("🔄 Rolling back configuration changes")
        
        rollback_results = {
            'restored_files': [],
            'errors': []
        }
        
        try:
            for backup_file in self.backup_dir.glob('*'):
                original_path = backup_file.name
                
                # Find original file location
                for updated_file in self.updated_files:
                    if Path(updated_file).name == original_path:
                        try:
                            shutil.copy2(backup_file, updated_file)
                            rollback_results['restored_files'].append(updated_file)
                            logger.info(f"✅ Restored {updated_file}")
                        except Exception as e:
                            error_msg = f"Failed to restore {updated_file}: {e}"
                            rollback_results['errors'].append(error_msg)
                            logger.error(f"❌ {error_msg}")
                        break
            
            logger.info(f"🔄 Rollback completed: {len(rollback_results['restored_files'])} files restored")
            
        except Exception as e:
            error_msg = f"Rollback failed: {e}"
            rollback_results['errors'].append(error_msg)
            logger.error(f"❌ {error_msg}")
        
        return rollback_results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate standardization report."""
        report = f"""
# Memory Configuration Standardization Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Target Memory: {self.target_memory_mb}MB
- Files Updated: {results['summary']['total_files_updated']}
- Backup Location: {results['summary']['backup_location']}

## File Categories Updated
- YAML Configs: {results['summary']['yaml_files']}
- Python Configs: {results['summary']['python_files']}
- Validation Scripts: {results['summary']['validation_scripts']}
- Dashboard Configs: {results['summary']['dashboard_configs']}
- Test Files: {results['summary']['test_files']}

## Updated Files
"""
        
        for file_path in results['updated_files']:
            report += f"- {file_path}\n"
        
        if results['errors']:
            report += "\n## Errors\n"
            for error in results['errors']:
                report += f"- {error}\n"
        
        report += f"\n## Next Steps\n"
        report += f"- Validate updated configurations\n"
        report += f"- Run performance tests to verify 400MB target\n"
        report += f"- Update documentation if needed\n"
        report += f"- Backup location: {self.backup_dir}\n"
        
        return report

def main():
    """Main function to run memory configuration standardization."""
    standardizer = MemoryConfigStandardizer(target_memory_mb=400)
    
    try:
        # Run standardization
        results = standardizer.standardize_all_configs()
        
        # Generate and save report
        report = standardizer.generate_report(results)
        report_path = "memory_config_standardization_report.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        logger.info(f"📄 Report saved to: {report_path}")
        
        # Return appropriate exit code
        if results['errors']:
            logger.warning("⚠️ Standardization completed with errors")
            return 1
        else:
            logger.info("🎉 Memory configuration standardization completed successfully!")
            return 0
            
    except Exception as e:
        logger.error(f"❌ Standardization failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
