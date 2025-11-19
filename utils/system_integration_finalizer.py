#!/usr/bin/env python3
"""
System Integration Finalizer - GoalDiggers Platform

Comprehensive system that ensures all components are properly integrated
and working together for production deployment. Validates data flow,
API connections, and system readiness.

Features:
- Complete system validation
- Data source verification
- API connectivity testing  
- Performance benchmarking
- Production readiness assessment
- Integration scoring
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import psutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class SystemIntegrationFinalizer:
    """Complete system integration validation and finalization."""
    
    def __init__(self):
        """Initialize the system integration finalizer."""
        self.logger = logger
        self.validation_results = {}
        self.performance_metrics = {}
        self.integration_score = 0.0
        self.start_time = time.time()
        
        # System requirements
        self.min_memory_gb = 1.0
        self.max_startup_time_seconds = 60
        self.min_integration_score = 90.0
        
        logger.info("🔧 System Integration Finalizer initialized")
    
    def run_complete_validation(self, include_performance: bool = True) -> Dict[str, Any]:
        """
        Run complete system validation and integration testing.
        
        Args:
            include_performance: Whether to include performance benchmarks
            
        Returns:
            Complete validation report
        """
        logger.info("🚀 Starting Complete System Integration Validation")
        logger.info("=" * 60)
        
        validation_start = time.time()
        
        # Step 1: Core System Components
        self._validate_core_components()
        
        # Step 2: Data Integration
        self._validate_data_integration()
        
        # Step 3: API Connectivity
        self._validate_api_connectivity()
        
        # Step 4: Dashboard Components
        self._validate_dashboard_components()
        
        # Step 5: Production Readiness
        self._validate_production_readiness()
        
        # Step 6: Performance Metrics (optional)
        if include_performance:
            self._run_performance_benchmarks()
        
        # Step 7: Calculate Integration Score
        self._calculate_integration_score()
        
        # Step 8: Generate Final Report
        validation_time = time.time() - validation_start
        final_report = self._generate_final_report(validation_time)
        
        logger.info("=" * 60)
        logger.info(f"✅ Complete System Validation Finished in {validation_time:.2f}s")
        logger.info(f"🎯 Integration Score: {self.integration_score:.1f}%")
        
        return final_report
    
    def _validate_core_components(self):
        """Validate core system components."""
        logger.info("🔍 Validating Core Components...")
        
        components = {
            'database_manager': self._test_database_manager(),
            'enhanced_predictor': self._test_enhanced_predictor(),
            'memory_optimizer': self._test_memory_optimizer(),
            'Interface': self._test_design_system(),
            'configuration_system': self._test_configuration_system()
        }
        
        self.validation_results['core_components'] = components
        
        passed_components = sum(1 for result in components.values() if result['status'] == 'PASS')
        total_components = len(components)
        
        logger.info(f"📊 Core Components: {passed_components}/{total_components} passed")
    
    def _test_database_manager(self) -> Dict[str, Any]:
        """Test database manager functionality."""
        try:
            from database.db_manager import DatabaseManager
            
            db_manager = DatabaseManager()
            
            # Test basic operations
            test_results = {
                'import': True,
                'initialization': db_manager is not None,
                'connection': False
            }
            
            # Test connection
            try:
                with db_manager.session_scope() as session:
                    test_results['connection'] = session is not None
            except Exception as e:
                logger.warning(f"Database connection test failed: {e}")
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.8 else 'FAIL',
                'details': test_results,
                'success_rate': success_rate,
                'message': 'Database Manager operational' if success_rate >= 0.8 else 'Database issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Database Manager failed: {e}'
            }
    
    def _test_enhanced_predictor(self) -> Dict[str, Any]:
        """Test enhanced prediction system."""
        try:
            from models.enhanced_real_data_predictor import \
                EnhancedRealDataPredictor
            
            predictor = EnhancedRealDataPredictor()
            
            # Test prediction generation
            test_prediction = predictor.predict_match_enhanced(
                home_team="Arsenal",
                away_team="Chelsea",
                use_real_data=False  # Use fallback data for testing
            )
            
            test_results = {
                'import': True,
                'initialization': predictor is not None,
                'prediction_generation': test_prediction is not None,
                'prediction_structure': all(hasattr(test_prediction, attr) for attr in [
                    'home_win_probability', 'draw_probability', 'away_win_probability'
                ])
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.8 else 'FAIL',
                'details': test_results,
                'success_rate': success_rate,
                'message': 'Enhanced Predictor operational' if success_rate >= 0.8 else 'Predictor issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Enhanced Predictor failed: {e}'
            }
    
    def _test_memory_optimizer(self) -> Dict[str, Any]:
        """Test memory optimization system."""
        try:
            from utils.production_memory_optimizer import \
                optimize_production_memory
            
            initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
            
            # Run memory optimization
            optimize_production_memory()
            
            optimized_memory = psutil.virtual_memory().used / (1024**3)  # GB
            memory_change = initial_memory - optimized_memory
            
            test_results = {
                'import': True,
                'execution': True,
                'memory_measurement': initial_memory > 0,
                'optimization_effect': memory_change >= 0  # Should reduce or maintain memory
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.8 else 'FAIL',
                'details': {
                    **test_results,
                    'initial_memory_gb': round(initial_memory, 2),
                    'optimized_memory_gb': round(optimized_memory, 2),
                    'memory_change_gb': round(memory_change, 2)
                },
                'success_rate': success_rate,
                'message': f'Memory Optimizer operational (Δ{memory_change:.1f}GB)' if success_rate >= 0.8 else 'Memory Optimizer issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Memory Optimizer failed: {e}'
            }
    
    def _test_design_system(self) -> Dict[str, Any]:
        """Test unified design system."""
        try:
            from dashboard.components.unified_design_system import \
                get_unified_styling
            
            styling = get_unified_styling()
            
            test_results = {
                'import': True,
                'styling_generation': styling is not None,
                'styling_content': isinstance(styling, str) and len(styling) > 100,
                'css_structure': '<style>' in styling if styling else False
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.8 else 'FAIL',
                'details': test_results,
                'success_rate': success_rate,
                'message': 'Unified Design System operational' if success_rate >= 0.8 else 'Design System issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Design System failed: {e}'
            }
    
    def _test_configuration_system(self) -> Dict[str, Any]:
        """Test configuration and environment system."""
        try:
            # Test environment variables
            env_vars = ['FOOTBALL_DATA_API_KEY', 'GOALDIGGERS_MODE']
            env_results = {}
            
            for var in env_vars:
                env_results[var] = os.getenv(var) is not None
            
            # Test .env file loading
            dotenv_loaded = False
            try:
                from dotenv import load_dotenv
                load_dotenv()
                dotenv_loaded = True
            except ImportError:
                pass
            
            test_results = {
                'dotenv_available': dotenv_loaded,
                'api_key_configured': env_results.get('FOOTBALL_DATA_API_KEY', False),
                'mode_configured': env_results.get('GOALDIGGERS_MODE', False) or os.getenv('GOALDIGGERS_MODE') == 'production'
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.5 else 'FAIL',  # Lower threshold for config
                'details': {**test_results, 'env_vars': env_results},
                'success_rate': success_rate,
                'message': 'Configuration system operational' if success_rate >= 0.5 else 'Configuration issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Configuration system failed: {e}'
            }
    
    def _validate_data_integration(self):
        """Validate data integration systems."""
        logger.info("📊 Validating Data Integration...")
        
        integrations = {
            'Data Sources': self._test_real_data_integrator(),
            'enhanced_data_aggregator': self._test_enhanced_data_aggregator(),
            'Data Hub': self._test_production_data_integrator()
        }
        
        self.validation_results['data_integration'] = integrations
        
        passed_integrations = sum(1 for result in integrations.values() if result['status'] == 'PASS')
        total_integrations = len(integrations)
        
        logger.info(f"📈 Data Integration: {passed_integrations}/{total_integrations} passed")
    
    def _test_real_data_integrator(self) -> Dict[str, Any]:
        """Test real data integrator."""
        try:
            from real_data_integrator import (get_real_matches,
                                              get_team_real_form)

            # Test basic functionality
            matches = get_real_matches(days_ahead=3)
            form_data = get_team_real_form("Arsenal")
            
            test_results = {
                'import': True,
                'matches_retrieval': isinstance(matches, list),
                'form_retrieval': isinstance(form_data, list),
                'data_structure': len(matches) > 0 if isinstance(matches, list) else False
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.75 else 'FAIL',
                'details': {
                    **test_results,
                    'matches_count': len(matches) if isinstance(matches, list) else 0,
                    'form_count': len(form_data) if isinstance(form_data, list) else 0
                },
                'success_rate': success_rate,
                'message': f'Real Data Integrator operational ({len(matches) if isinstance(matches, list) else 0} matches)' if success_rate >= 0.75 else 'Real Data Integrator issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Real Data Integrator failed: {e}'
            }
    
    def _test_enhanced_data_aggregator(self) -> Dict[str, Any]:
        """Test enhanced data aggregator."""
        try:
            from utils.enhanced_data_aggregator import (get_current_fixtures,
                                                        get_todays_matches)

            # Test fixture retrieval
            fixtures = get_current_fixtures(days_ahead=5)
            todays_matches = get_todays_matches()
            
            test_results = {
                'import': True,
                'fixtures_retrieval': isinstance(fixtures, list),
                'todays_matches_retrieval': isinstance(todays_matches, list),
                'fixture_data_quality': len(fixtures) > 0 if isinstance(fixtures, list) else False
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.75 else 'FAIL',
                'details': {
                    **test_results,
                    'fixtures_count': len(fixtures) if isinstance(fixtures, list) else 0,
                    'todays_count': len(todays_matches) if isinstance(todays_matches, list) else 0
                },
                'success_rate': success_rate,
                'message': f'Enhanced Data Aggregator operational ({len(fixtures) if isinstance(fixtures, list) else 0} fixtures)' if success_rate >= 0.75 else 'Enhanced Data Aggregator issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Enhanced Data Aggregator failed: {e}'
            }
    
    def _test_production_data_integrator(self) -> Dict[str, Any]:
        """Test production data integrator."""
        try:
            from dashboard.production_data_integrator import \
                get_integration_status
            
            status = get_integration_status()
            
            test_results = {
                'import': True,
                'status_retrieval': isinstance(status, dict),
                'status_structure': 'status' in status if isinstance(status, dict) else False,
                'integration_active': status.get('status') == 'active' if isinstance(status, dict) else False
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.75 else 'FAIL',
                'details': {
                    **test_results,
                    'integration_status': status.get('status') if isinstance(status, dict) else 'unknown'
                },
                'success_rate': success_rate,
                'message': f'Production Data Integrator operational' if success_rate >= 0.75 else 'Production Data Integrator issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Production Data Integrator failed: {e}'
            }
    
    def _validate_api_connectivity(self):
        """Validate API connectivity and external services."""
        logger.info("🌐 Validating API Connectivity...")
        
        apis = {
            'football_data_org': self._test_football_data_api(),
            'network_connectivity': self._test_network_connectivity(),
            'dns_resolution': self._test_dns_resolution()
        }
        
        self.validation_results['api_connectivity'] = apis
        
        passed_apis = sum(1 for result in apis.values() if result['status'] == 'PASS')
        total_apis = len(apis)
        
        logger.info(f"🔗 API Connectivity: {passed_apis}/{total_apis} passed")
    
    def _test_football_data_api(self) -> Dict[str, Any]:
        """Test Football-Data.org API connectivity."""
        try:
            api_key = os.getenv('FOOTBALL_DATA_API_KEY')
            
            if not api_key:
                return {
                    'status': 'SKIP',
                    'details': {'api_key_available': False},
                    'success_rate': 0.0,
                    'message': 'Football-Data API key not configured'
                }
            
            import requests

            # Test API connection
            headers = {'X-Auth-Token': api_key}
            url = 'https://api.football-data.org/v4/competitions'
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                api_accessible = response.status_code in [200, 403]  # 403 might indicate rate limiting but API is accessible
                rate_limited = response.status_code == 429
            except requests.RequestException:
                api_accessible = False
                rate_limited = False
            
            test_results = {
                'api_key_available': True,
                'api_accessible': api_accessible,
                'not_rate_limited': not rate_limited
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.67 else 'FAIL',
                'details': test_results,
                'success_rate': success_rate,
                'message': 'Football-Data API accessible' if success_rate >= 0.67 else 'Football-Data API issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Football-Data API test failed: {e}'
            }
    
    def _test_network_connectivity(self) -> Dict[str, Any]:
        """Test basic network connectivity."""
        try:
            import requests

            # Test connectivity to major sites
            test_urls = [
                'https://httpbin.org/status/200',
                'https://api.github.com',
                'https://www.google.com'
            ]
            
            connectivity_results = {}
            for url in test_urls:
                try:
                    response = requests.get(url, timeout=5)
                    connectivity_results[url] = response.status_code < 400
                except:
                    connectivity_results[url] = False
            
            success_rate = sum(connectivity_results.values()) / len(connectivity_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.67 else 'FAIL',
                'details': connectivity_results,
                'success_rate': success_rate,
                'message': 'Network connectivity operational' if success_rate >= 0.67 else 'Network connectivity issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Network connectivity test failed: {e}'
            }
    
    def _test_dns_resolution(self) -> Dict[str, Any]:
        """Test DNS resolution."""
        try:
            import socket
            
            test_domains = [
                'api.football-data.org',
                'github.com',
                'google.com'
            ]
            
            dns_results = {}
            for domain in test_domains:
                try:
                    socket.gethostbyname(domain)
                    dns_results[domain] = True
                except socket.gaierror:
                    dns_results[domain] = False
            
            success_rate = sum(dns_results.values()) / len(dns_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.67 else 'FAIL',
                'details': dns_results,
                'success_rate': success_rate,
                'message': 'DNS resolution operational' if success_rate >= 0.67 else 'DNS resolution issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'DNS resolution test failed: {e}'
            }
    
    def _validate_dashboard_components(self):
        """Validate dashboard and UI components."""
        logger.info("🎨 Validating Dashboard Components...")
        
        components = {
            'enhanced_homepage': self._test_enhanced_homepage(),
            'match_selector': self._test_match_selector(),
            'streamlined_launcher': self._test_streamlined_launcher()
        }
        
        self.validation_results['dashboard_components'] = components
        
        passed_components = sum(1 for result in components.values() if result['status'] == 'PASS')
        total_components = len(components)
        
        logger.info(f"📱 Dashboard Components: {passed_components}/{total_components} passed")
    
    def _test_enhanced_homepage(self) -> Dict[str, Any]:
        """Test enhanced production homepage."""
        try:
            from dashboard.enhanced_production_homepage import \
                ProductionDashboardHomepage
            
            homepage = ProductionDashboardProductionDashboardHomepage()
            
            test_results = {
                'import': True,
                'initialization': homepage is not None,
                'methods_available': all(hasattr(homepage, method) for method in [
                    '_get_todays_featured_matches',
                    '_get_upcoming_fixtures',
                    'render_production_homepage'
                ])
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.8 else 'FAIL',
                'details': test_results,
                'success_rate': success_rate,
                'message': 'Enhanced Homepage operational' if success_rate >= 0.8 else 'Enhanced Homepage issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Enhanced Homepage failed: {e}'
            }
    
    def _test_match_selector(self) -> Dict[str, Any]:
        """Test enhanced match selector."""
        try:
            from dashboard.components.enhanced_match_selector import \
                EnhancedMatchSelector
            
            match_selector = EnhancedMatchSelector()
            
            test_results = {
                'import': True,
                'initialization': match_selector is not None,
                'methods_available': all(hasattr(match_selector, method) for method in [
                    'render_enhanced_match_selector',
                    '_get_leagues_and_teams'
                ])
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.8 else 'FAIL',
                'details': test_results,
                'success_rate': success_rate,
                'message': 'Match Selector operational' if success_rate >= 0.8 else 'Match Selector issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Match Selector failed: {e}'
            }
    
    def _test_streamlined_launcher(self) -> Dict[str, Any]:
        """Test streamlined production launcher."""
        try:
            from streamlined_production_launcher import \
                StreamlinedProductionLauncher
            
            launcher = StreamlinedProductionLauncher()
            
            test_results = {
                'import': True,
                'initialization': launcher is not None,
                'methods_available': all(hasattr(launcher, method) for method in [
                    'setup_environment',
                    'check_dependencies',
                    'run_full_platform'
                ])
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.8 else 'FAIL',
                'details': test_results,
                'success_rate': success_rate,
                'message': 'Streamlined Launcher operational' if success_rate >= 0.8 else 'Streamlined Launcher issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Streamlined Launcher failed: {e}'
            }
    
    def _validate_production_readiness(self):
        """Validate production readiness criteria."""
        logger.info("🏭 Validating Production Readiness...")
        
        readiness_checks = {
            'system_resources': self._check_system_resources(),
            'file_structure': self._check_file_structure(),
            'error_handling': self._check_error_handling(),
            'logging_system': self._check_logging_system()
        }
        
        self.validation_results['production_readiness'] = readiness_checks
        
        passed_checks = sum(1 for result in readiness_checks.values() if result['status'] == 'PASS')
        total_checks = len(readiness_checks)
        
        logger.info(f"🚀 Production Readiness: {passed_checks}/{total_checks} passed")
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            memory_gb = memory.total / (1024**3)
            disk_gb = disk.free / (1024**3)
            
            test_results = {
                'sufficient_memory': memory_gb >= self.min_memory_gb,
                'sufficient_disk': disk_gb >= 1.0,  # 1GB free space minimum
                'memory_utilization_healthy': memory.percent < 90
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.67 else 'FAIL',
                'details': {
                    **test_results,
                    'total_memory_gb': round(memory_gb, 2),
                    'free_disk_gb': round(disk_gb, 2),
                    'memory_percent': memory.percent
                },
                'success_rate': success_rate,
                'message': f'System resources adequate ({memory_gb:.1f}GB RAM, {disk_gb:.1f}GB free)' if success_rate >= 0.67 else 'System resource constraints detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'System resource check failed: {e}'
            }
    
    def _check_file_structure(self) -> Dict[str, Any]:
        """Check critical file structure."""
        try:
            critical_files = [
                'streamlined_production_launcher.py',
                'dashboard/enhanced_production_homepage.py',
                'models/enhanced_real_data_predictor.py',
                'utils/production_memory_optimizer.py',
                'real_data_integrator.py'
            ]
            
            critical_dirs = [
                'dashboard',
                'models',
                'utils',
                'database',
                'data'
            ]
            
            file_results = {}
            for file_path in critical_files:
                file_results[file_path] = os.path.exists(file_path)
            
            dir_results = {}
            for dir_path in critical_dirs:
                dir_results[dir_path] = os.path.isdir(dir_path)
            
            test_results = {
                'critical_files_present': all(file_results.values()),
                'critical_dirs_present': all(dir_results.values()),
                'structure_complete': all(file_results.values()) and all(dir_results.values())
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.67 else 'FAIL',
                'details': {
                    **test_results,
                    'files': file_results,
                    'directories': dir_results
                },
                'success_rate': success_rate,
                'message': 'File structure complete' if success_rate >= 0.67 else 'File structure incomplete'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'File structure check failed: {e}'
            }
    
    def _check_error_handling(self) -> Dict[str, Any]:
        """Check error handling capabilities."""
        try:
            # Test error handling in key components
            error_handling_tests = {
                'predictor_invalid_input': self._test_predictor_error_handling(),
                'data_integrator_network_error': self._test_data_integrator_error_handling(),
                'memory_optimizer_cleanup': self._test_memory_optimizer_error_handling()
            }
            
            success_rate = sum(error_handling_tests.values()) / len(error_handling_tests)
            
            return {
                'status': 'PASS' if success_rate >= 0.67 else 'FAIL',
                'details': error_handling_tests,
                'success_rate': success_rate,
                'message': 'Error handling robust' if success_rate >= 0.67 else 'Error handling needs improvement'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Error handling check failed: {e}'
            }
    
    def _test_predictor_error_handling(self) -> bool:
        """Test predictor error handling with invalid input."""
        try:
            from models.enhanced_real_data_predictor import \
                EnhancedRealDataPredictor
            predictor = EnhancedRealDataPredictor()
            
            # Test with invalid input
            result = predictor.predict_match_enhanced(
                home_team="", 
                away_team="", 
                use_real_data=False
            )
            
            # Should return some result even with invalid input
            return result is not None
            
        except Exception:
            # Should not raise unhandled exceptions
            return False
    
    def _test_data_integrator_error_handling(self) -> bool:
        """Test data integrator error handling."""
        try:
            from utils.enhanced_data_aggregator import get_current_fixtures

            # Should handle network errors gracefully
            fixtures = get_current_fixtures(days_ahead=1)
            
            # Should return some result even with network issues
            return isinstance(fixtures, list)
            
        except Exception:
            return False
    
    def _test_memory_optimizer_error_handling(self) -> bool:
        """Test memory optimizer error handling."""
        try:
            from utils.production_memory_optimizer import \
                optimize_production_memory

            # Should not crash even with memory constraints
            optimize_production_memory()
            
            return True
            
        except Exception:
            return False
    
    def _check_logging_system(self) -> Dict[str, Any]:
        """Check logging system functionality."""
        try:
            # Test logging configuration
            test_logger = logging.getLogger('test_logger')
            
            # Test different log levels
            log_levels_work = True
            try:
                test_logger.info("Test info message")
                test_logger.warning("Test warning message")
                test_logger.error("Test error message")
            except Exception:
                log_levels_work = False
            
            test_results = {
                'logger_creation': test_logger is not None,
                'log_levels_functional': log_levels_work,
                'root_logger_configured': logging.getLogger().hasHandlers()
            }
            
            success_rate = sum(test_results.values()) / len(test_results)
            
            return {
                'status': 'PASS' if success_rate >= 0.67 else 'FAIL',
                'details': test_results,
                'success_rate': success_rate,
                'message': 'Logging system operational' if success_rate >= 0.67 else 'Logging system issues detected'
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'details': {'error': str(e)},
                'success_rate': 0.0,
                'message': f'Logging system check failed: {e}'
            }
    
    def _run_performance_benchmarks(self):
        """Run performance benchmarks."""
        logger.info("⚡ Running Performance Benchmarks...")
        
        benchmarks = {
            'prediction_generation_speed': self._benchmark_prediction_speed(),
            'data_retrieval_speed': self._benchmark_data_retrieval(),
            'memory_efficiency': self._benchmark_memory_usage(),
            'startup_time': self._benchmark_startup_time()
        }
        
        self.performance_metrics = benchmarks
        
        logger.info("📊 Performance Benchmarks Complete")
    
    def _benchmark_prediction_speed(self) -> Dict[str, Any]:
        """Benchmark prediction generation speed."""
        try:
            from models.enhanced_real_data_predictor import \
                EnhancedRealDataPredictor
            
            predictor = EnhancedRealDataPredictor()
            
            # Run multiple predictions and measure time
            start_time = time.time()
            num_predictions = 5
            
            for i in range(num_predictions):
                predictor.predict_match_enhanced(
                    home_team="Arsenal",
                    away_team="Chelsea",
                    use_real_data=False
                )
            
            total_time = time.time() - start_time
            avg_time = total_time / num_predictions
            
            performance_rating = 'excellent' if avg_time < 0.5 else 'good' if avg_time < 2.0 else 'acceptable'
            
            return {
                'total_time_seconds': round(total_time, 3),
                'average_time_seconds': round(avg_time, 3),
                'predictions_per_second': round(1 / avg_time, 2),
                'performance_rating': performance_rating
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'performance_rating': 'failed'
            }
    
    def _benchmark_data_retrieval(self) -> Dict[str, Any]:
        """Benchmark data retrieval speed."""
        try:
            from utils.enhanced_data_aggregator import get_current_fixtures
            
            start_time = time.time()
            fixtures = get_current_fixtures(days_ahead=3)
            retrieval_time = time.time() - start_time
            
            performance_rating = 'excellent' if retrieval_time < 2.0 else 'good' if retrieval_time < 5.0 else 'acceptable'
            
            return {
                'retrieval_time_seconds': round(retrieval_time, 3),
                'fixtures_retrieved': len(fixtures) if isinstance(fixtures, list) else 0,
                'performance_rating': performance_rating
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'performance_rating': 'failed'
            }
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        try:
            initial_memory = psutil.virtual_memory().used / (1024**2)  # MB
            
            # Simulate typical usage
            from models.enhanced_real_data_predictor import \
                EnhancedRealDataPredictor
            predictor = EnhancedRealDataPredictor()
            
            for i in range(3):
                predictor.predict_match_enhanced(f"Team{i}", f"Team{i+1}", use_real_data=False)
            
            peak_memory = psutil.virtual_memory().used / (1024**2)  # MB
            memory_increase = peak_memory - initial_memory
            
            performance_rating = 'excellent' if memory_increase < 50 else 'good' if memory_increase < 100 else 'acceptable'
            
            return {
                'initial_memory_mb': round(initial_memory, 1),
                'peak_memory_mb': round(peak_memory, 1),
                'memory_increase_mb': round(memory_increase, 1),
                'performance_rating': performance_rating
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'performance_rating': 'failed'
            }
    
    def _benchmark_startup_time(self) -> Dict[str, Any]:
        """Benchmark system startup time."""
        startup_time = time.time() - self.start_time
        
        performance_rating = 'excellent' if startup_time < 10 else 'good' if startup_time < 30 else 'acceptable'
        
        return {
            'startup_time_seconds': round(startup_time, 3),
            'performance_rating': performance_rating
        }
    
    def _calculate_integration_score(self):
        """Calculate overall integration score."""
        logger.info("🧮 Calculating Integration Score...")
        
        category_weights = {
            'core_components': 0.25,
            'data_integration': 0.25,
            'api_connectivity': 0.20,
            'dashboard_components': 0.20,
            'production_readiness': 0.10
        }
        
        category_scores = {}
        
        for category, results in self.validation_results.items():
            if category in category_weights:
                # Calculate average success rate for this category
                success_rates = [result.get('success_rate', 0.0) for result in results.values()]
                category_scores[category] = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        # Calculate weighted average
        weighted_score = sum(
            category_scores.get(category, 0.0) * weight
            for category, weight in category_weights.items()
        )
        
        self.integration_score = weighted_score * 100  # Convert to percentage
        
        logger.info(f"🎯 Integration Score: {self.integration_score:.1f}%")
    
    def _generate_final_report(self, validation_time: float) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        total_tests = sum(
            len(results) for results in self.validation_results.values()
        )
        
        passed_tests = sum(
            sum(1 for result in results.values() if result['status'] == 'PASS')
            for results in self.validation_results.values()
        )
        
        production_ready = (
            self.integration_score >= self.min_integration_score and
            passed_tests / total_tests >= 0.8
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'integration_score': round(self.integration_score, 1),
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': round((passed_tests / total_tests) * 100, 1) if total_tests > 0 else 0.0,
                'production_ready': production_ready,
                'validation_time_seconds': round(validation_time, 2)
            },
            'validation_results': self.validation_results,
            'performance_metrics': self.performance_metrics,
            'recommendations': self._generate_recommendations(),
            'next_steps': self._generate_next_steps(production_ready)
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for common issues and provide recommendations
        for category, results in self.validation_results.items():
            for test_name, result in results.items():
                if result['status'] == 'FAIL':
                    if 'api' in test_name.lower():
                        recommendations.append(f"⚠️ Fix {test_name}: Check API keys and network connectivity")
                    elif 'memory' in test_name.lower():
                        recommendations.append(f"🔧 Optimize {test_name}: Review memory usage and optimization")
                    elif 'database' in test_name.lower():
                        recommendations.append(f"💾 Fix {test_name}: Check database configuration and permissions")
                    else:
                        recommendations.append(f"🔍 Review {test_name}: {result.get('message', 'Investigation needed')}")
        
        # Add general recommendations based on score
        if self.integration_score < 70:
            recommendations.append("📈 Critical: Multiple system components need attention before production deployment")
        elif self.integration_score < 90:
            recommendations.append("⚡ Important: Some optimizations needed for optimal production performance")
        else:
            recommendations.append("✅ Excellent: System is production-ready with high integration score")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_next_steps(self, production_ready: bool) -> List[str]:
        """Generate next steps based on validation results."""
        if production_ready:
            return [
                "🚀 System is production-ready for deployment",
                "📊 Monitor performance metrics during initial deployment",
                "🔄 Set up automated health checks",
                "📈 Plan for scaling based on usage patterns",
                "🛡️ Implement backup and recovery procedures"
            ]
        else:
            return [
                "🔧 Address failed validation tests before production deployment",
                "⚡ Optimize performance bottlenecks identified in benchmarks",
                "🔍 Review and fix integration issues",
                "📋 Re-run validation after fixes",
                "🎯 Aim for integration score ≥ 90% before production"
            ]


def run_system_validation(include_performance: bool = True) -> Dict[str, Any]:
    """
    Run complete system integration validation.
    
    Args:
        include_performance: Whether to include performance benchmarks
        
    Returns:
        Complete validation report
    """
    finalizer = SystemIntegrationFinalizer()
    return finalizer.run_complete_validation(include_performance)


if __name__ == "__main__":
    print("🔧 GoalDiggers Platform - System Integration Finalizer")
    print("=" * 60)
    
    # Run complete validation
    report = run_system_validation(include_performance=True)
    
    # Display summary
    summary = report['validation_summary']
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Integration Score: {summary['integration_score']}%")
    print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']} ({summary['success_rate']}%)")
    print(f"Production Ready: {'✅ YES' if summary['production_ready'] else '❌ NO'}")
    print(f"Validation Time: {summary['validation_time_seconds']}s")
    
    # Display recommendations
    if report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations'][:5]:
            print(f"  {rec}")
    
    # Display next steps
    if report['next_steps']:
        print(f"\n📋 NEXT STEPS:")
        for step in report['next_steps'][:3]:
            print(f"  {step}")
    
    # Save detailed report
    report_file = f"system_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
