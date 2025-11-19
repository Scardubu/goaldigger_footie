#!/usr/bin/env python3
"""
Dependency Checker for GoalDiggers Platform
Analyzes requirements.txt and identifies potential dependency conflicts.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from importlib.metadata import distributions
except ImportError:
    print("ERROR: Python >=3.8 required for dependency analysis. Please upgrade your Python version.")
    sys.exit(1)

# Use centralized logging only if not under pytest (avoid handler duplication)
if not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        from utils.logging_config import configure_logging  # type: ignore
        configure_logging()
    except Exception:
        # Fallback to basicConfig only if centralized config import fails
        logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DependencyChecker:
    """Analyzes Python package dependencies for conflicts and compatibility issues."""

    def __init__(self, requirements_file: str = "requirements.txt"):
        self.requirements_file = Path(requirements_file)
        self.installed_packages = {}
        self.requirements = {}
        self.conflicts = []
        self.warnings = []

    def load_requirements(self) -> Dict[str, str]:
        """Load requirements from requirements.txt file."""
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return {}

        requirements = {}
        with open(self.requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '==' in line:
                    package, version = line.split('==', 1)
                    requirements[package.lower()] = version

        self.requirements = requirements
        logger.info(f"Loaded {len(requirements)} requirements")
        return requirements

    def get_installed_packages(self) -> Dict[str, str]:
        """Get currently installed packages and their versions."""
        try:
            installed = {}
            for dist in distributions():
                if dist.metadata.get('Name'):
                    installed[dist.metadata['Name'].lower()] = dist.version
            self.installed_packages = installed
            logger.info(f"Found {len(installed)} installed packages")
            return installed
        except Exception as e:
            logger.error(f"Error getting installed packages: {e}")
            return {}

    def check_version_conflicts(self) -> List[Dict[str, str]]:
        """Check for version conflicts between requirements and installed packages."""
        conflicts = []

        for package, required_version in self.requirements.items():
            if package in self.installed_packages:
                installed_version = self.installed_packages[package]
                if installed_version != required_version:
                    conflicts.append({
                        'package': package,
                        'required': required_version,
                        'installed': installed_version,
                        'type': 'version_mismatch'
                    })

        self.conflicts.extend(conflicts)
        return conflicts

    def check_known_conflicts(self) -> List[Dict[str, str]]:
        """Check for known package conflicts."""
        known_conflicts = [
            # XGBoost and LightGBM version conflicts
            {
                'packages': ['xgboost', 'lightgbm'],
                'issue': 'May have conflicting C++ dependencies',
                'solution': 'Install with --no-binary flag if issues occur'
            },
            # NumPy version conflicts with ML libraries
            {
                'packages': ['numpy', 'pandas', 'scikit-learn'],
                'issue': 'Version compatibility between data science packages',
                'solution': 'Ensure compatible versions (numpy>=1.21, pandas>=1.3)'
            },
            # PyTorch and CUDA conflicts
            {
                'packages': ['torch', 'torchvision'],
                'issue': 'CUDA version compatibility',
                'solution': 'Install PyTorch with matching CUDA version'
            },
            # Async library conflicts
            {
                'packages': ['asyncio', 'aiohttp', 'nest_asyncio'],
                'issue': 'Event loop policy conflicts on Windows',
                'solution': 'Use nest_asyncio.apply() early in application'
            }
        ]

        conflicts = []
        for conflict in known_conflicts:
            packages = conflict['packages']
            found_packages = [p for p in packages if p in self.requirements]
            if len(found_packages) > 1:
                conflicts.append({
                    'packages': found_packages,
                    'issue': conflict['issue'],
                    'solution': conflict['solution'],
                    'type': 'known_conflict'
                })

        self.conflicts.extend(conflicts)
        return conflicts

    def check_missing_dependencies(self) -> List[str]:
        """Check for missing dependencies."""
        missing = []
        for package in self.requirements:
            if package not in self.installed_packages:
                missing.append(package)

        if missing:
            self.warnings.append(f"Missing packages: {', '.join(missing)}")

        return missing

    def check_python_version_compatibility(self) -> List[str]:
        """Check Python version compatibility with packages."""
        python_version = sys.version_info
        warnings = []

        # Check for packages that may have Python version issues
        version_sensitive_packages = {
            'xgboost': (3, 8),
            'lightgbm': (3, 8),
            'catboost': (3, 8),
            'torch': (3, 8),
            'transformers': (3, 8),
            'streamlit': (3, 8)
        }

        for package, min_version in version_sensitive_packages.items():
            if package in self.requirements:
                if python_version < min_version:
                    warnings.append(
                        f"{package} requires Python {min_version[0]}.{min_version[1]}+, "
                        f"current: {python_version[0]}.{python_version[1]}"
                    )

        self.warnings.extend(warnings)
        return warnings

    def check_platform_compatibility(self) -> List[str]:
        """Check platform-specific compatibility issues."""
        platform = sys.platform
        warnings = []

        if platform == "win32":
            # Windows-specific issues
            windows_issues = [
                "Some packages may require Visual C++ build tools on Windows",
                "Playwright browsers need to be installed separately",
                "CUDA setup may require specific NVIDIA drivers"
            ]
            warnings.extend(windows_issues)
        elif platform == "darwin":
            # macOS-specific issues
            macos_issues = [
                "M1/M2 Mac users may need ARM64 versions of some packages",
                "Some packages may require Rosetta 2 on Apple Silicon"
            ]
            warnings.extend(macos_issues)

        self.warnings.extend(warnings)
        return warnings

    def generate_installation_script(self) -> str:
        """Generate an installation script with conflict resolution."""
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated installation script for GoalDiggers platform",
            "",
            "# Create virtual environment",
            "python -m venv venv",
            "",
            "# Activate virtual environment",
            "if [[ \"$OSTYPE\" == \"msys\" || \"$OSTYPE\" == \"win32\" ]]; then",
            "    source venv/Scripts/activate",
            "else",
            "    source venv/bin/activate",
            "fi",
            "",
            "# Upgrade pip",
            "pip install --upgrade pip",
            "",
            "# Install core dependencies first",
            "pip install setuptools wheel",
            "",
            "# Install data science packages",
            "pip install numpy pandas scikit-learn",
            "",
            "# Install ML libraries (with fallback options)",
            "pip install xgboost || pip install xgboost --no-binary xgboost",
            "pip install lightgbm || pip install lightgbm --no-binary lightgbm",
            "pip install catboost",
            "",
            "# Install remaining packages",
            "pip install -r requirements.txt",
            "",
            "# Install Playwright browsers",
            "playwright install",
            "",
            "echo 'Installation completed successfully!'"
        ]

        return "\n".join(script_lines)

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete dependency analysis."""
        logger.info("Starting dependency analysis...")

        # Load requirements and installed packages
        self.load_requirements()
        self.get_installed_packages()

        # Run all checks
        version_conflicts = self.check_version_conflicts()
        known_conflicts = self.check_known_conflicts()
        missing_packages = self.check_missing_dependencies()
        python_warnings = self.check_python_version_compatibility()
        platform_warnings = self.check_platform_compatibility()

        # Generate report
        report = {
            'summary': {
                'total_requirements': len(self.requirements),
                'total_installed': len(self.installed_packages),
                'version_conflicts': len(version_conflicts),
                'known_conflicts': len(known_conflicts),
                'missing_packages': len(missing_packages),
                'warnings': len(self.warnings)
            },
            'conflicts': self.conflicts,
            'warnings': self.warnings,
            'missing_packages': missing_packages,
            'recommendations': self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        if self.conflicts:
            recommendations.append("Resolve version conflicts before proceeding")
            recommendations.append("Consider using a fresh virtual environment")

        if self.warnings:
            recommendations.append("Review warnings for potential issues")

        if any('xgboost' in str(c) for c in self.conflicts):
            recommendations.append("Try installing XGBoost with --no-binary flag")

        if any('lightgbm' in str(c) for c in self.conflicts):
            recommendations.append("Try installing LightGBM with --no-binary flag")

        recommendations.extend([
            "Install Playwright browsers: playwright install",
            "For GPU support, install CUDA toolkit and cuDNN",
            "Consider using conda for complex ML package dependencies"
        ])

        return recommendations

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print formatted analysis report."""
        print("=" * 60)
        print("GOALDIGGERS PLATFORM - DEPENDENCY ANALYSIS REPORT")
        print("=" * 60)

        # Summary
        summary = report['summary']
        print(f"\n📊 SUMMARY:")
        print(f"   Requirements: {summary['total_requirements']}")
        print(f"   Installed: {summary['total_installed']}")
        print(f"   Version Conflicts: {summary['version_conflicts']}")
        print(f"   Known Conflicts: {summary['known_conflicts']}")
        print(f"   Missing Packages: {summary['missing_packages']}")
        print(f"   Warnings: {summary['warnings']}")

        # Conflicts
        if report['conflicts']:
            print(f"\n❌ CONFLICTS FOUND:")
            for conflict in report['conflicts']:
                if conflict['type'] == 'version_mismatch':
                    print(f"   {conflict['package']}: required {conflict['required']}, installed {conflict['installed']}")
                else:
                    print(f"   {', '.join(conflict['packages'])}: {conflict['issue']}")
                    print(f"     Solution: {conflict['solution']}")

        # Warnings
        if report['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in report['warnings']:
                print(f"   {warning}")

        # Missing packages
        if report['missing_packages']:
            print(f"\n📦 MISSING PACKAGES:")
            for package in report['missing_packages']:
                print(f"   {package}")

        # Recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   • {rec}")

        print("\n" + "=" * 60)

def main():
    """Main function to run dependency checker."""
    checker = DependencyChecker()
    report = checker.run_full_analysis()
    checker.print_report(report)

    # Generate installation script
    script = checker.generate_installation_script()
    script_path = Path("install_footie.sh")
    with open(script_path, 'w') as f:
        f.write(script)
    print(f"\n📝 Installation script generated: {script_path}")

    # Exit with error code if conflicts found
    if report['summary']['version_conflicts'] > 0 or report['summary']['known_conflicts'] > 0:
        sys.exit(1)
    else:
        print("\n✅ No critical conflicts found. Installation should proceed normally.")

if __name__ == "__main__":
    main()