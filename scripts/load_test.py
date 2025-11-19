#!/usr/bin/env python3
"""
Production Load Testing Suite
Validates system performance under production-like traffic
"""
import argparse
import asyncio
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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


class LoadTester:
    """Performs load testing on production system."""
    
    def __init__(self, duration: int = 60, workers: int = 4, rps: int = 10):
        """Initialize load tester.
        
        Args:
            duration: Test duration in seconds
            workers: Number of concurrent workers
            rps: Target requests per second
        """
        self.duration = duration
        self.workers = workers
        self.rps = rps
        self.results = []
        self.metrics = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'errors': [],
            'response_times': [],
            'validation_checks': 0,
            'quality_scores': []
        }
        
        load_dotenv()
        
        logger.info(f"🧪 Load Tester initialized")
        logger.info(f"   Duration: {duration}s")
        logger.info(f"   Workers: {workers}")
        logger.info(f"   Target RPS: {rps}")
    
    def run(self) -> Dict[str, Any]:
        """Run load test.
        
        Returns:
            Dictionary with test results
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 Starting Load Test")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Run load test
        self._execute_load_test()
        
        duration = time.time() - start_time
        
        # Calculate statistics
        report = self._generate_report(duration)
        
        # Log summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 Load Test Results")
        logger.info("=" * 70)
        logger.info(f"⏱️  Duration: {duration:.1f}s")
        logger.info(f"📈 Total Requests: {report['summary']['total_requests']}")
        logger.info(f"✅ Successful: {report['summary']['successful']} ({report['summary']['success_rate']:.1f}%)")
        logger.info(f"❌ Failed: {report['summary']['failed']}")
        logger.info(f"⚡ Throughput: {report['summary']['actual_rps']:.1f} req/s")
        logger.info(f"🕐 Avg Response Time: {report['summary']['avg_response_time']:.3f}s")
        logger.info(f"🕐 P95 Response Time: {report['summary']['p95_response_time']:.3f}s")
        logger.info(f"🕐 P99 Response Time: {report['summary']['p99_response_time']:.3f}s")
        
        if report['validation']['quality_scores']:
            logger.info(f"✨ Avg Quality Score: {report['validation']['avg_quality_score']:.1%}")
            logger.info(f"✨ Min Quality Score: {report['validation']['min_quality_score']:.1%}")
        
        logger.info("=" * 70)
        
        return report
    
    def _execute_load_test(self):
        """Execute the load test."""
        logger.info("\n🔥 Executing load test...")
        
        end_time = time.time() + self.duration
        requests_per_worker = self.rps // self.workers
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            
            for worker_id in range(self.workers):
                future = executor.submit(
                    self._worker_loop,
                    worker_id,
                    end_time,
                    requests_per_worker
                )
                futures.append(future)
            
            # Wait for all workers to complete
            for future in futures:
                future.result()
        
        logger.info(f"✅ Load test complete: {self.metrics['total_requests']} requests")
    
    def _worker_loop(self, worker_id: int, end_time: float, target_rps: int):
        """Worker loop to generate load.
        
        Args:
            worker_id: Worker identifier
            end_time: When to stop
            target_rps: Target requests per second for this worker
        """
        request_interval = 1.0 / target_rps if target_rps > 0 else 1.0
        
        while time.time() < end_time:
            try:
                # Execute test request
                self._execute_test_request(worker_id)
                
                # Sleep to maintain target RPS
                time.sleep(request_interval)
                
            except Exception as e:
                self.metrics['errors'].append(str(e))
                logger.debug(f"Worker {worker_id} error: {e}")
    
    def _execute_test_request(self, worker_id: int):
        """Execute a single test request.
        
        Args:
            worker_id: Worker identifier
        """
        start_time = time.time()
        
        try:
            # Simulate prediction generation
            from batched_prediction_engine import BatchedPredictionEngine
            
            engine = BatchedPredictionEngine()
            
            # Create test match data
            test_match = {
                'home_team_id': f'TEAM_{random.randint(1, 20)}',
                'away_team_id': f'TEAM_{random.randint(1, 20)}',
                'match_date': datetime.now().isoformat(),
                'league_id': f'LEAGUE_{random.randint(1, 5)}'
            }
            
            # Generate prediction (this will trigger validation)
            prediction = engine._generate_single_prediction(test_match)
            
            response_time = time.time() - start_time
            
            # Record metrics
            self.metrics['total_requests'] += 1
            self.metrics['response_times'].append(response_time)
            
            if prediction:
                self.metrics['successful'] += 1
                
                # Extract quality score if available
                if 'validation_report' in prediction:
                    validation = prediction['validation_report']
                    if 'quality_score' in validation:
                        self.metrics['quality_scores'].append(validation['quality_score'])
                        self.metrics['validation_checks'] += 1
            else:
                self.metrics['failed'] += 1
                
        except Exception as e:
            response_time = time.time() - start_time
            self.metrics['total_requests'] += 1
            self.metrics['failed'] += 1
            self.metrics['response_times'].append(response_time)
            self.metrics['errors'].append(str(e)[:100])
    
    def _generate_report(self, duration: float) -> Dict[str, Any]:
        """Generate load test report.
        
        Args:
            duration: Actual test duration
            
        Returns:
            Comprehensive test report
        """
        response_times = sorted(self.metrics['response_times'])
        
        # Calculate percentiles
        def percentile(data, p):
            if not data:
                return 0
            k = (len(data) - 1) * (p / 100)
            f = int(k)
            c = f + 1 if f < len(data) - 1 else f
            return data[f] + (data[c] - data[f]) * (k - f)
        
        total_requests = self.metrics['total_requests']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'duration': self.duration,
                'workers': self.workers,
                'target_rps': self.rps
            },
            'summary': {
                'total_requests': total_requests,
                'successful': self.metrics['successful'],
                'failed': self.metrics['failed'],
                'success_rate': (self.metrics['successful'] / total_requests * 100) if total_requests > 0 else 0,
                'actual_rps': total_requests / duration if duration > 0 else 0,
                'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
                'min_response_time': min(response_times) if response_times else 0,
                'max_response_time': max(response_times) if response_times else 0,
                'p50_response_time': percentile(response_times, 50),
                'p95_response_time': percentile(response_times, 95),
                'p99_response_time': percentile(response_times, 99)
            },
            'validation': {
                'validation_checks': self.metrics['validation_checks'],
                'avg_quality_score': sum(self.metrics['quality_scores']) / len(self.metrics['quality_scores']) if self.metrics['quality_scores'] else 0,
                'min_quality_score': min(self.metrics['quality_scores']) if self.metrics['quality_scores'] else 0,
                'max_quality_score': max(self.metrics['quality_scores']) if self.metrics['quality_scores'] else 0,
                'quality_scores': self.metrics['quality_scores'][:100]  # Sample
            },
            'errors': {
                'count': len(self.metrics['errors']),
                'samples': self.metrics['errors'][:10]  # First 10 errors
            }
        }
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run production load testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick 30-second test
  python scripts/load_test.py --duration 30 --rps 5
  
  # Standard 2-minute test
  python scripts/load_test.py --duration 120 --rps 10
  
  # Stress test with high load
  python scripts/load_test.py --duration 300 --rps 50 --workers 8
  
  # Light test with metrics output
  python scripts/load_test.py --duration 60 --rps 5 --output results/load_test.json
        """
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Test duration in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--rps', '-r',
        type=int,
        default=10,
        help='Target requests per second (default: 10)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of concurrent workers (default: 4)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration < 1:
        logger.error("Duration must be at least 1 second")
        sys.exit(1)
    
    if args.rps < 1:
        logger.error("RPS must be at least 1")
        sys.exit(1)
    
    if args.workers < 1:
        logger.error("Workers must be at least 1")
        sys.exit(1)
    
    # Create tester
    tester = LoadTester(
        duration=args.duration,
        workers=args.workers,
        rps=args.rps
    )
    
    # Run test
    try:
        report = tester.run()
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2))
            logger.info(f"\n💾 Results saved to: {output_path}")
        
        # Determine success
        success_rate = report['summary']['success_rate']
        
        if success_rate >= 95:
            logger.info("\n🎉 Load test PASSED (success rate >= 95%)")
            sys.exit(0)
        elif success_rate >= 80:
            logger.warning("\n⚠️  Load test MARGINAL (success rate 80-95%)")
            sys.exit(0)
        else:
            logger.error("\n❌ Load test FAILED (success rate < 80%)")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Load test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Load test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
