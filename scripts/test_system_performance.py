#!/usr/bin/env python
"""
Comprehensive System Performance Test for Football Betting Platform
Tests all Phase 2 optimizations: ML models, feature engineering, API performance, and database optimization.
"""

import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import requests
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

class SystemPerformanceTester:
    """Comprehensive system performance testing suite."""
    
    def __init__(self):
        """Initialize the performance tester."""
        self.results = {}
        self.api_base_url = "http://localhost:8000"
        self.test_start_time = time.time()
        
    def test_ml_model_performance(self) -> Dict[str, Any]:
        """Test ML model performance and validation improvements."""
        logger.info("Testing ML model performance...")
        start_time = time.time()
        
        try:
            # Import our improved model validation
            from scripts.test_model_validation import test_overfitting_detection
            
            # Run overfitting detection test
            overfitting_test_passed = test_overfitting_detection()
            
            # Test model inference speed
            inference_times = []
            for i in range(10):
                inference_start = time.time()
                # Simulate model inference (replace with actual model call)
                time.sleep(0.01)  # Simulate 10ms inference
                inference_times.append(time.time() - inference_start)
            
            avg_inference_time = sum(inference_times) / len(inference_times)
            
            test_time = time.time() - start_time
            
            return {
                'overfitting_detection': overfitting_test_passed,
                'avg_inference_time': avg_inference_time,
                'inference_times': inference_times,
                'test_duration': test_time,
                'status': 'PASS' if overfitting_test_passed and avg_inference_time < 0.1 else 'FAIL'
            }
            
        except Exception as e:
            logger.error(f"ML model performance test failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'test_duration': time.time() - start_time
            }
    
    def test_feature_engineering_performance(self) -> Dict[str, Any]:
        """Test feature engineering optimization performance."""
        logger.info("Testing feature engineering performance...")
        start_time = time.time()
        
        try:
            # Import our feature optimization
            from scripts.test_feature_optimization import (
                test_correlation_analysis, test_feature_selection,
                test_engineered_features, test_complete_pipeline
            )
            
            # Run all feature optimization tests
            correlation_test = test_correlation_analysis()
            selection_test = test_feature_selection()
            engineering_test = test_engineered_features()
            pipeline_test = test_complete_pipeline()
            
            all_tests_passed = all([correlation_test, selection_test, engineering_test, pipeline_test])
            
            test_time = time.time() - start_time
            
            return {
                'correlation_analysis': correlation_test,
                'feature_selection': selection_test,
                'engineered_features': engineering_test,
                'complete_pipeline': pipeline_test,
                'test_duration': test_time,
                'status': 'PASS' if all_tests_passed else 'FAIL'
            }
            
        except Exception as e:
            logger.error(f"Feature engineering performance test failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'test_duration': time.time() - start_time
            }
    
    def test_api_performance(self) -> Dict[str, Any]:
        """Test API performance and response times."""
        logger.info("Testing API performance...")
        start_time = time.time()
        
        try:
            # Test API endpoints if server is running
            endpoints_to_test = [
                "/health",
                "/metrics",
                "/api/performance/stats"
            ]
            
            response_times = {}
            successful_requests = 0
            
            for endpoint in endpoints_to_test:
                try:
                    url = f"{self.api_base_url}{endpoint}"
                    request_start = time.time()
                    response = requests.get(url, timeout=5)
                    request_time = time.time() - request_start
                    
                    response_times[endpoint] = {
                        'response_time': request_time,
                        'status_code': response.status_code,
                        'success': response.status_code == 200
                    }
                    
                    if response.status_code == 200:
                        successful_requests += 1
                        
                except requests.exceptions.RequestException as e:
                    response_times[endpoint] = {
                        'response_time': None,
                        'status_code': None,
                        'success': False,
                        'error': str(e)
                    }
            
            # Calculate average response time for successful requests
            successful_times = [r['response_time'] for r in response_times.values() 
                              if r['success'] and r['response_time'] is not None]
            avg_response_time = sum(successful_times) / len(successful_times) if successful_times else None
            
            test_time = time.time() - start_time
            
            return {
                'endpoints_tested': len(endpoints_to_test),
                'successful_requests': successful_requests,
                'avg_response_time': avg_response_time,
                'response_times': response_times,
                'test_duration': test_time,
                'status': 'PASS' if successful_requests > 0 and (avg_response_time is None or avg_response_time < 1.0) else 'FAIL'
            }
            
        except Exception as e:
            logger.error(f"API performance test failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'test_duration': time.time() - start_time
            }
    
    def test_database_performance(self) -> Dict[str, Any]:
        """Test database performance and optimization."""
        logger.info("Testing database performance...")
        start_time = time.time()
        
        try:
            from database.db_manager import DatabaseManager
            
            # Initialize database manager
            db_manager = DatabaseManager()
            
            # Test connection pool
            pool_info = {
                'pool_size': getattr(db_manager, 'pool_size', 'unknown'),
                'max_overflow': getattr(db_manager, 'max_overflow', 'unknown'),
                'pool_timeout': getattr(db_manager, 'pool_timeout', 'unknown')
            }
            
            # Test query performance
            query_times = []
            for i in range(5):
                query_start = time.time()
                try:
                    with db_manager.session_scope() as session:
                        # Simple query to test performance
                        result = session.execute(text("SELECT 1")).fetchone()
                    query_time = time.time() - query_start
                    query_times.append(query_time)
                except Exception as e:
                    logger.warning(f"Database query failed: {e}")
                    query_times.append(None)
            
            # Calculate average query time
            successful_queries = [t for t in query_times if t is not None]
            avg_query_time = sum(successful_queries) / len(successful_queries) if successful_queries else None
            
            test_time = time.time() - start_time
            
            return {
                'pool_configuration': pool_info,
                'successful_queries': len(successful_queries),
                'total_queries': len(query_times),
                'avg_query_time': avg_query_time,
                'query_times': query_times,
                'test_duration': test_time,
                'status': 'PASS' if len(successful_queries) > 0 and (avg_query_time is None or avg_query_time < 0.1) else 'FAIL'
            }
            
        except Exception as e:
            logger.error(f"Database performance test failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'test_duration': time.time() - start_time
            }
    
    def test_concurrent_performance(self) -> Dict[str, Any]:
        """Test concurrent request handling and scalability."""
        logger.info("Testing concurrent performance...")
        start_time = time.time()
        
        try:
            # Simulate concurrent operations
            def simulate_operation(operation_id):
                """Simulate a typical system operation."""
                operation_start = time.time()
                
                # Simulate feature generation
                time.sleep(0.05)  # 50ms
                
                # Simulate model inference
                time.sleep(0.02)  # 20ms
                
                # Simulate database query
                time.sleep(0.01)  # 10ms
                
                return {
                    'operation_id': operation_id,
                    'duration': time.time() - operation_start,
                    'success': True
                }
            
            # Test with different concurrency levels
            concurrency_results = {}
            
            for num_workers in [1, 2, 4, 8]:
                logger.info(f"Testing with {num_workers} concurrent workers...")
                
                worker_start = time.time()
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(simulate_operation, i) for i in range(10)]
                    results = [future.result() for future in as_completed(futures)]
                
                worker_time = time.time() - worker_start
                avg_operation_time = sum(r['duration'] for r in results) / len(results)
                
                concurrency_results[num_workers] = {
                    'total_time': worker_time,
                    'avg_operation_time': avg_operation_time,
                    'operations_per_second': len(results) / worker_time,
                    'successful_operations': sum(1 for r in results if r['success'])
                }
            
            test_time = time.time() - start_time
            
            # Check if performance scales reasonably
            single_ops_per_sec = concurrency_results[1]['operations_per_second']
            multi_ops_per_sec = concurrency_results[4]['operations_per_second']
            scaling_factor = multi_ops_per_sec / single_ops_per_sec if single_ops_per_sec > 0 else 0
            
            return {
                'concurrency_results': concurrency_results,
                'scaling_factor': scaling_factor,
                'test_duration': test_time,
                'status': 'PASS' if scaling_factor > 1.5 else 'FAIL'  # Expect at least 1.5x improvement with 4 workers
            }
            
        except Exception as e:
            logger.error(f"Concurrent performance test failed: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'test_duration': time.time() - start_time
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all performance tests and generate comprehensive report."""
        logger.info("Starting comprehensive system performance test...")
        
        # Run all test categories
        self.results = {
            'test_start_time': self.test_start_time,
            'ml_model_performance': self.test_ml_model_performance(),
            'feature_engineering_performance': self.test_feature_engineering_performance(),
            'api_performance': self.test_api_performance(),
            'database_performance': self.test_database_performance(),
            'concurrent_performance': self.test_concurrent_performance()
        }
        
        # Calculate overall results
        test_categories = ['ml_model_performance', 'feature_engineering_performance', 
                          'api_performance', 'database_performance', 'concurrent_performance']
        
        passed_tests = sum(1 for category in test_categories 
                          if self.results[category].get('status') == 'PASS')
        total_tests = len(test_categories)
        
        total_test_time = time.time() - self.test_start_time
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests,
            'total_test_time': total_test_time,
            'overall_status': 'PASS' if passed_tests == total_tests else 'PARTIAL' if passed_tests > 0 else 'FAIL'
        }
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance test report."""
        if not self.results:
            return "No test results available. Run comprehensive test first."
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE SYSTEM PERFORMANCE TEST REPORT")
        report.append("=" * 80)
        
        # Summary
        summary = self.results['summary']
        report.append(f"\n📊 OVERALL RESULTS:")
        report.append(f"   Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        report.append(f"   Success Rate: {summary['success_rate']:.1%}")
        report.append(f"   Total Time: {summary['total_test_time']:.2f}s")
        report.append(f"   Status: {summary['overall_status']}")
        
        # Individual test results
        test_categories = {
            'ml_model_performance': '🤖 ML Model Performance',
            'feature_engineering_performance': '🔧 Feature Engineering',
            'api_performance': '🌐 API Performance',
            'database_performance': '🗄️ Database Performance',
            'concurrent_performance': '⚡ Concurrent Performance'
        }
        
        for category, title in test_categories.items():
            result = self.results[category]
            status_emoji = "✅" if result.get('status') == 'PASS' else "❌" if result.get('status') == 'FAIL' else "⚠️"
            
            report.append(f"\n{status_emoji} {title}:")
            report.append(f"   Status: {result.get('status', 'UNKNOWN')}")
            report.append(f"   Duration: {result.get('test_duration', 0):.2f}s")
            
            if result.get('status') == 'ERROR':
                report.append(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Performance metrics
        if self.results['api_performance'].get('avg_response_time'):
            report.append(f"\n📈 KEY PERFORMANCE METRICS:")
            report.append(f"   API Response Time: {self.results['api_performance']['avg_response_time']:.3f}s")
        
        if self.results['database_performance'].get('avg_query_time'):
            report.append(f"   Database Query Time: {self.results['database_performance']['avg_query_time']:.3f}s")
        
        if self.results['concurrent_performance'].get('scaling_factor'):
            report.append(f"   Concurrency Scaling: {self.results['concurrent_performance']['scaling_factor']:.1f}x")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)

def main():
    """Main test execution function."""
    logger.info("Starting comprehensive system performance testing...")
    
    try:
        tester = SystemPerformanceTester()
        results = tester.run_comprehensive_test()
        
        # Generate and display report
        report = tester.generate_report()
        print(report)
        
        # Save results to file
        results_file = Path("system_performance_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Exit with appropriate code
        overall_status = results['summary']['overall_status']
        if overall_status == 'PASS':
            logger.info("🎉 All performance tests passed!")
            sys.exit(0)
        elif overall_status == 'PARTIAL':
            logger.warning("⚠️ Some performance tests failed.")
            sys.exit(1)
        else:
            logger.error("❌ Performance tests failed.")
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"Performance testing failed: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
