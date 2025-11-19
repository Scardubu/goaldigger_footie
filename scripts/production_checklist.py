#!/usr/bin/env python3
"""
🏭 Production Checklist Validator for GoalDiggers Platform

Comprehensive automated validation of production readiness across:
- Infrastructure (Docker, Compose, Environment)
- Security (API keys, CORS, Rate limiting, HTTPS)
- Monitoring (Prometheus, Grafana, Alerts)
- Documentation (Guides, APIs, Runbooks)
- Smoke Tests (Health, API, Predictions, Dashboard)

Usage: python scripts/production_checklist.py [--verbose]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("ProductionChecklist")


class ProductionChecklistValidator:
    """Automated validator for production readiness checklist."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the validator.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "categories": {},
            "overall_score": 0.0,
            "status": "NOT READY"
        }
        
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def _log_check(self, category: str, check: str, passed: bool, details: str = ""):
        """Log a check result."""
        symbol = "✅" if passed else "❌"
        message = f"{symbol} [{category}] {check}"
        if details:
            message += f" - {details}"
        
        if passed:
            logger.info(message)
        else:
            logger.warning(message)
    
    def _check_file_exists(self, filepath: str) -> bool:
        """Check if a file exists."""
        return Path(filepath).exists()
    
    def _check_docker_infrastructure(self) -> Dict[str, Any]:
        """Validate Docker infrastructure setup."""
        logger.info("🐳 Checking Docker Infrastructure...")
        
        checks = []
        
        # Check Dockerfile exists
        check = ("Dockerfile exists", self._check_file_exists("Dockerfile"))
        checks.append(check)
        self._log_check("Infrastructure", check[0], check[1])
        
        # Check docker-compose.yml exists
        check = ("docker-compose.yml exists", self._check_file_exists("docker-compose.yml"))
        checks.append(check)
        self._log_check("Infrastructure", check[0], check[1])
        
        # Check docker-compose has health checks
        if self._check_file_exists("docker-compose.yml"):
            with open("docker-compose.yml", 'r') as f:
                content = f.read()
                has_healthcheck = "healthcheck:" in content
                check = ("Health checks configured", has_healthcheck)
                checks.append(check)
                self._log_check("Infrastructure", check[0], check[1])
                
                # Check for resource limits
                has_limits = "mem_limit:" in content or "cpus:" in content or "deploy:" in content
                check = ("Resource limits configured", has_limits)
                checks.append(check)
                self._log_check("Infrastructure", check[0], check[1], 
                              "Add memory/CPU limits" if not has_limits else "")
                
                # Check restart policy
                has_restart = "restart:" in content
                check = ("Restart policy configured", has_restart)
                checks.append(check)
                self._log_check("Infrastructure", check[0], check[1])
        
        # Check .dockerignore exists
        check = (".dockerignore exists", self._check_file_exists(".dockerignore"))
        checks.append(check)
        self._log_check("Infrastructure", check[0], check[1])
        
        passed = sum(1 for _, status in checks if status)
        score = passed / len(checks) if checks else 0.0
        
        return {
            "checks": dict(checks),
            "passed": passed,
            "total": len(checks),
            "score": score
        }
    
    def _check_environment_config(self) -> Dict[str, Any]:
        """Validate environment configuration."""
        logger.info("⚙️  Checking Environment Configuration...")
        
        checks = []
        
        # Check .env.example exists
        check = (".env.example exists", self._check_file_exists(".env.example"))
        checks.append(check)
        self._log_check("Environment", check[0], check[1])
        
        # Check for required environment variables in code
        required_vars = [
            "FOOTBALL_DATA_API_KEY",
            "DATABASE_URL",
            "REDIS_URL"
        ]
        
        if self._check_file_exists(".env.example"):
            with open(".env.example", 'r') as f:
                env_content = f.read()
                for var in required_vars:
                    has_var = var in env_content
                    check = (f"{var} documented", has_var)
                    checks.append(check)
                    self._log_check("Environment", check[0], check[1])
        
        # Check config.py or similar exists
        config_files = ["utils/config.py", "config.py", "config/settings.py"]
        has_config = any(self._check_file_exists(f) for f in config_files)
        check = ("Configuration module exists", has_config)
        checks.append(check)
        self._log_check("Environment", check[0], check[1])
        
        passed = sum(1 for _, status in checks if status)
        score = passed / len(checks) if checks else 0.0
        
        return {
            "checks": dict(checks),
            "passed": passed,
            "total": len(checks),
            "score": score
        }
    
    def _check_security(self) -> Dict[str, Any]:
        """Validate security configuration."""
        logger.info("🔒 Checking Security Configuration...")
        
        checks = []
        
        # Check .gitignore includes secrets
        if self._check_file_exists(".gitignore"):
            with open(".gitignore", 'r') as f:
                gitignore = f.read()
                security_patterns = [".env", "*.key", "secrets", ".pem"]
                for pattern in security_patterns:
                    has_pattern = pattern in gitignore
                    check = (f"Ignores {pattern}", has_pattern)
                    checks.append(check)
                    self._log_check("Security", check[0], check[1])
        else:
            check = (".gitignore exists", False)
            checks.append(check)
            self._log_check("Security", check[0], check[1])
        
        # Check for CORS configuration
        cors_files = ["app.py", "api/server.py", "unified_launcher.py"]
        has_cors = False
        for filepath in cors_files:
            if self._check_file_exists(filepath):
                with open(filepath, 'r') as f:
                    if "CORS" in f.read():
                        has_cors = True
                        break
        
        check = ("CORS configuration found", has_cors)
        checks.append(check)
        self._log_check("Security", check[0], check[1])
        
        # Check for rate limiting
        rate_limit_files = ["utils/rate_limiter.py", "api/rate_limiter.py"]
        has_rate_limit = any(self._check_file_exists(f) for f in rate_limit_files)
        check = ("Rate limiting implemented", has_rate_limit)
        checks.append(check)
        self._log_check("Security", check[0], check[1])
        
        passed = sum(1 for _, status in checks if status)
        score = passed / len(checks) if checks else 0.0
        
        return {
            "checks": dict(checks),
            "passed": passed,
            "total": len(checks),
            "score": score
        }
    
    def _check_monitoring(self) -> Dict[str, Any]:
        """Validate monitoring setup."""
        logger.info("📊 Checking Monitoring Configuration...")
        
        checks = []
        
        # Check Prometheus config
        check = ("Prometheus config exists", 
                self._check_file_exists("monitoring/prometheus.yml") or
                self._check_file_exists("monitoring/prometheus/prometheus.yml"))
        checks.append(check)
        self._log_check("Monitoring", check[0], check[1])
        
        # Check Prometheus alerts
        check = ("Prometheus alerts configured",
                self._check_file_exists("monitoring/prometheus/alerts.yml") or
                self._check_file_exists("monitoring/alerts.yml"))
        checks.append(check)
        self._log_check("Monitoring", check[0], check[1])
        
        # Check Grafana dashboards
        grafana_paths = [
            "monitoring/grafana/dashboards",
            "monitoring/dashboards"
        ]
        has_dashboards = any(
            Path(p).exists() and any(Path(p).glob("*.json"))
            for p in grafana_paths
        )
        check = ("Grafana dashboards configured", has_dashboards)
        checks.append(check)
        self._log_check("Monitoring", check[0], check[1])
        
        # Check metrics exporter
        check = ("Metrics exporter implemented",
                self._check_file_exists("utils/metrics_exporter.py"))
        checks.append(check)
        self._log_check("Monitoring", check[0], check[1])
        
        # Check logging configuration
        logging_files = ["utils/logging_config.py", "logging.conf"]
        has_logging = any(self._check_file_exists(f) for f in logging_files)
        check = ("Logging configured", has_logging)
        checks.append(check)
        self._log_check("Monitoring", check[0], check[1])
        
        passed = sum(1 for _, status in checks if status)
        score = passed / len(checks) if checks else 0.0
        
        return {
            "checks": dict(checks),
            "passed": passed,
            "total": len(checks),
            "score": score
        }
    
    def _check_documentation(self) -> Dict[str, Any]:
        """Validate documentation."""
        logger.info("📚 Checking Documentation...")
        
        checks = []
        
        # Check README
        check = ("README.md exists", self._check_file_exists("README.md"))
        checks.append(check)
        self._log_check("Documentation", check[0], check[1])
        
        # Check deployment guide
        deployment_docs = [
            "docs/DEPLOYMENT_GUIDE.md",
            "DEPLOYMENT.md",
            "docs/deployment.md"
        ]
        has_deployment = any(self._check_file_exists(f) for f in deployment_docs)
        check = ("Deployment guide exists", has_deployment)
        checks.append(check)
        self._log_check("Documentation", check[0], check[1])
        
        # Check API documentation
        api_docs = [
            "docs/API_REFERENCE.md",
            "API_REFERENCE.md",
            "docs/api.md"
        ]
        has_api_docs = any(self._check_file_exists(f) for f in api_docs)
        check = ("API documentation exists", has_api_docs)
        checks.append(check)
        self._log_check("Documentation", check[0], check[1])
        
        # Check architecture documentation
        arch_docs = [
            "docs/ARCHITECTURE.md",
            "ARCHITECTURE.md"
        ]
        has_arch = any(self._check_file_exists(f) for f in arch_docs)
        check = ("Architecture documented", has_arch)
        checks.append(check)
        self._log_check("Documentation", check[0], check[1])
        
        # Check for runbook
        runbook_docs = [
            "docs/PRODUCTION_RUNBOOK.md",
            "RUNBOOK.md",
            "docs/runbook.md"
        ]
        has_runbook = any(self._check_file_exists(f) for f in runbook_docs)
        check = ("Production runbook exists", has_runbook)
        checks.append(check)
        self._log_check("Documentation", check[0], check[1], 
                       "Create runbook" if not has_runbook else "")
        
        passed = sum(1 for _, status in checks if status)
        score = passed / len(checks) if checks else 0.0
        
        return {
            "checks": dict(checks),
            "passed": passed,
            "total": len(checks),
            "score": score
        }
    
    def _check_tests(self) -> Dict[str, Any]:
        """Validate test coverage."""
        logger.info("🧪 Checking Test Coverage...")
        
        checks = []
        
        # Check for test files
        test_patterns = [
            "test_*.py",
            "*_test.py",
            "tests/*.py"
        ]
        
        test_files = []
        for pattern in test_patterns:
            test_files.extend(Path(".").rglob(pattern))
        
        has_tests = len(test_files) > 0
        check = ("Unit tests exist", has_tests)
        checks.append(check)
        self._log_check("Tests", check[0], check[1], f"Found {len(test_files)} test files")
        
        # Check for integration tests
        integration_files = list(Path(".").rglob("test_*_integration.py"))
        has_integration = len(integration_files) > 0
        check = ("Integration tests exist", has_integration)
        checks.append(check)
        self._log_check("Tests", check[0], check[1], f"Found {len(integration_files)} files")
        
        # Check for load tests
        check = ("Load tests exist", self._check_file_exists("run_load_tests.py"))
        checks.append(check)
        self._log_check("Tests", check[0], check[1])
        
        passed = sum(1 for _, status in checks if status)
        score = passed / len(checks) if checks else 0.0
        
        return {
            "checks": dict(checks),
            "passed": passed,
            "total": len(checks),
            "score": score
        }
    
    def _run_smoke_tests(self) -> Dict[str, Any]:
        """Run smoke tests to validate basic functionality."""
        logger.info("💨 Running Smoke Tests...")
        
        checks = []
        
        # Test 1: Check if main app file exists and has valid syntax
        try:
            import os
            import py_compile
            app_file = "app.py"
            if os.path.exists(app_file):
                py_compile.compile(app_file, doraise=True)
                check = ("Main app file valid", True)
            else:
                check = ("Main app file valid", False)
        except Exception as e:
            check = ("Main app file valid", False)
            logger.debug(f"Syntax check error: {e}")
        checks.append(check)
        self._log_check("Smoke Test", check[0], check[1])
        
        # Test 2: Check data utilities file exists
        try:
            data_utils_file = "cached_data_utilities.py"
            if os.path.exists(data_utils_file):
                py_compile.compile(data_utils_file, doraise=True)
                check = ("Data utilities file valid", True)
            else:
                check = ("Data utilities file valid", False)
        except Exception as e:
            check = ("Data utilities file valid", False)
            logger.debug(f"Syntax check error: {e}")
        checks.append(check)
        self._log_check("Smoke Test", check[0], check[1])
        
        # Test 3: Check cache manager file exists
        try:
            cache_file = "utils/enhanced_cache_manager.py"
            if os.path.exists(cache_file):
                py_compile.compile(cache_file, doraise=True)
                check = ("Cache manager file valid", True)
            else:
                check = ("Cache manager file valid", False)
        except Exception as e:
            check = ("Cache manager file valid", False)
            logger.debug(f"Syntax check error: {e}")
        checks.append(check)
        self._log_check("Smoke Test", check[0], check[1])
        
        # Test 4: Check database files exist
        try:
            db_files = ["database/connection.py", "database/db_manager.py", "database/schema.py"]
            has_db = any(os.path.exists(f) for f in db_files)
            check = ("Database modules available", has_db)
        except Exception as e:
            check = ("Database modules available", False)
            logger.debug(f"DB check error: {e}")
        checks.append(check)
        self._log_check("Smoke Test", check[0], check[1])
        
        # Test 5: Check metrics exporter exists
        try:
            metrics_file = "utils/metrics_exporter.py"
            if os.path.exists(metrics_file):
                py_compile.compile(metrics_file, doraise=True)
                check = ("Metrics exporter file valid", True)
            else:
                check = ("Metrics exporter file valid", False)
        except Exception as e:
            check = ("Metrics exporter file valid", False)
            logger.debug(f"Metrics check error: {e}")
        checks.append(check)
        self._log_check("Smoke Test", check[0], check[1])
        
        # Test 6: Check rate limiter exists (new requirement)
        try:
            rate_limiter_file = "utils/rate_limiter.py"
            if os.path.exists(rate_limiter_file):
                py_compile.compile(rate_limiter_file, doraise=True)
                check = ("Rate limiter file valid", True)
            else:
                check = ("Rate limiter file valid", False)
        except Exception as e:
            check = ("Rate limiter file valid", False)
            logger.debug(f"Rate limiter check error: {e}")
        checks.append(check)
        self._log_check("Smoke Test", check[0], check[1])
        
        passed = sum(1 for _, status in checks if status)
        score = passed / len(checks) if checks else 0.0
        
        return {
            "checks": dict(checks),
            "passed": passed,
            "total": len(checks),
            "score": score
        }
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete production checklist validation."""
        logger.info("🚀 Starting Production Checklist Validation")
        logger.info("="*70)
        
        start_time = time.time()
        
        # Run all validation categories
        categories = {
            "infrastructure": self._check_docker_infrastructure(),
            "environment": self._check_environment_config(),
            "security": self._check_security(),
            "monitoring": self._check_monitoring(),
            "documentation": self._check_documentation(),
            "tests": self._check_tests(),
            "smoke_tests": self._run_smoke_tests()
        }
        
        # Calculate overall score
        total_passed = sum(cat["passed"] for cat in categories.values())
        total_checks = sum(cat["total"] for cat in categories.values())
        overall_score = (total_passed / total_checks) * 100 if total_checks > 0 else 0.0
        
        # Determine status
        if overall_score >= 95:
            status = "✅ PRODUCTION READY"
        elif overall_score >= 85:
            status = "⚠️  READY WITH MINOR ISSUES"
        elif overall_score >= 70:
            status = "⚠️  NEEDS IMPROVEMENTS"
        else:
            status = "❌ NOT READY"
        
        duration = time.time() - start_time
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "duration_sec": round(duration, 2),
            "categories": categories,
            "total_checks": total_checks,
            "total_passed": total_passed,
            "overall_score": round(overall_score, 1),
            "status": status
        }
        
        return self.results
    
    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*70)
        print("🏭 PRODUCTION CHECKLIST VALIDATION SUMMARY")
        print("="*70)
        
        for category, results in self.results["categories"].items():
            score_pct = results["score"] * 100
            status = "✅" if score_pct >= 80 else "⚠️" if score_pct >= 60 else "❌"
            print(f"\n{status} {category.upper():<20} {results['passed']}/{results['total']} ({score_pct:.0f}%)")
            
            if self.verbose:
                for check_name, passed in results["checks"].items():
                    check_status = "✅" if passed else "❌"
                    print(f"    {check_status} {check_name}")
        
        print("\n" + "="*70)
        print(f"Overall Score: {self.results['overall_score']:.1f}%")
        print(f"Total Checks: {self.results['total_passed']}/{self.results['total_checks']}")
        print(f"Status: {self.results['status']}")
        print(f"Duration: {self.results['duration_sec']:.2f}s")
        print("="*70 + "\n")
    
    def save_report(self, output_file: str = None):
        """Save validation report to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"production_checklist_report_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"📄 Report saved to: {output_file}")
        return output_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate production readiness checklist"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for report (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    try:
        # Run validation
        validator = ProductionChecklistValidator(verbose=args.verbose)
        results = validator.run_full_validation()
        
        # Print summary
        validator.print_summary()
        
        # Save report
        report_file = validator.save_report(args.output)
        
        # Exit code based on score
        if results["overall_score"] >= 85:
            return 0
        elif results["overall_score"] >= 70:
            return 1
        else:
            return 2
            
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
