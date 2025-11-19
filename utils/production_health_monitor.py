#!/usr/bin/env python3
"""
Production Health Monitor - GoalDiggers Platform

Comprehensive health monitoring system that tracks:
1. Real data integration availability and freshness
2. API connectivity and circuit breaker status
3. Prediction quality and confidence metrics
4. Database and cache health
5. Memory usage and performance

Usage:
    python -m utils.production_health_monitor --check-all
    python -m utils.production_health_monitor --report
"""
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Container for a single health metric."""
    name: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    value: Any
    threshold: Optional[Any] = None
    message: str = ""
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class ProductionHealthMonitor:
    """Comprehensive health monitoring for production systems."""

    def __init__(self):
        self.metrics: List[HealthMetric] = []
        self.report_dir = Path("logs/health_reports")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def check_real_data_integration(self) -> HealthMetric:
        """Check if real data integrator is available and functioning."""
        try:
            from real_data_integrator import get_real_matches, real_data_integrator

            # Test fetch
            start = time.time()
            matches = get_real_matches(days_ahead=7)
            latency = time.time() - start

            # Check for real (non-fallback) matches
            real_matches = [
                m for m in matches 
                if not str(m.get('api_id', '')).startswith('fallback_')
            ]

            if len(real_matches) >= 5:
                status = 'healthy'
                message = f"✅ {len(real_matches)} real matches available (latency: {latency:.2f}s)"
            elif len(matches) > 0:
                status = 'warning'
                message = f"⚠️ Only {len(real_matches)} real matches, {len(matches) - len(real_matches)} fallback"
            else:
                status = 'critical'
                message = "❌ No matches available from real data integrator"

            return HealthMetric(
                name="real_data_integration",
                status=status,
                value={
                    'total_matches': len(matches),
                    'real_matches': len(real_matches),
                    'fallback_matches': len(matches) - len(real_matches),
                    'latency_seconds': round(latency, 3)
                },
                threshold={'min_real_matches': 5},
                message=message
            )

        except Exception as e:
            logger.error(f"Real data integration check failed: {e}")
            return HealthMetric(
                name="real_data_integration",
                status='critical',
                value=None,
                message=f"❌ Real data integrator not available: {str(e)}"
            )

    def check_api_connectivity(self) -> HealthMetric:
        """Check API connectivity and circuit breaker status."""
        try:
            from async_data_integrator import AsyncDataIntegrator

            integrator = AsyncDataIntegrator()
            cb_state = integrator._circuit_breaker

            if cb_state['state'] == 'open':
                status = 'critical'
                message = f"❌ Circuit breaker OPEN ({cb_state['failures']} failures)"
            elif cb_state['state'] == 'half_open':
                status = 'warning'
                message = f"⚠️ Circuit breaker HALF-OPEN (testing recovery)"
            else:
                status = 'healthy'
                message = f"✅ Circuit breaker CLOSED (operational)"

            return HealthMetric(
                name="api_connectivity",
                status=status,
                value={
                    'circuit_breaker_state': cb_state['state'],
                    'failures': cb_state['failures'],
                    'success_count': cb_state['success_count']
                },
                threshold={'max_failures': cb_state['failure_threshold']},
                message=message
            )

        except Exception as e:
            logger.error(f"API connectivity check failed: {e}")
            return HealthMetric(
                name="api_connectivity",
                status='unknown',
                value=None,
                message=f"⚠️ Could not check API status: {str(e)}"
            )

    def check_prediction_quality(self) -> HealthMetric:
        """Check prediction quality and confidence metrics."""
        try:
            from models.enhanced_real_data_predictor import EnhancedRealDataPredictor

            predictor = EnhancedRealDataPredictor()

            # Check if we have performance metrics
            if predictor.performance_metrics['total_predictions'] > 0:
                accuracy = predictor.performance_metrics.get('accuracy', 0.0)
                total_preds = predictor.performance_metrics['total_predictions']

                if accuracy >= 0.50:
                    status = 'healthy'
                    message = f"✅ Accuracy: {accuracy:.1%} ({total_preds} predictions)"
                elif accuracy >= 0.45:
                    status = 'warning'
                    message = f"⚠️ Accuracy: {accuracy:.1%} (below target 50%)"
                else:
                    status = 'critical'
                    message = f"❌ Accuracy: {accuracy:.1%} (significantly below target)"

                return HealthMetric(
                    name="prediction_quality",
                    status=status,
                    value={
                        'accuracy': round(accuracy, 4),
                        'total_predictions': total_preds,
                        'successful_predictions': predictor.performance_metrics.get('successful_predictions', 0)
                    },
                    threshold={'min_accuracy': 0.50},
                    message=message
                )
            else:
                return HealthMetric(
                    name="prediction_quality",
                    status='unknown',
                    value={'total_predictions': 0},
                    message="ℹ️ No predictions tracked yet"
                )

        except Exception as e:
            logger.error(f"Prediction quality check failed: {e}")
            return HealthMetric(
                name="prediction_quality",
                status='unknown',
                value=None,
                message=f"⚠️ Could not check prediction quality: {str(e)}"
            )

    def check_database_health(self) -> HealthMetric:
        """Check database connectivity and health."""
        try:
            from database.db_manager import DatabaseManager

            db_manager = DatabaseManager()
            conn_info = db_manager.connection_info()

            if conn_info.get('using_fallback'):
                status = 'warning'
                message = f"⚠️ Using SQLite fallback (Primary DB unreachable)"
            else:
                status = 'healthy'
                message = f"✅ Primary database operational"

            return HealthMetric(
                name="database_health",
                status=status,
                value={
                    'using_fallback': conn_info.get('using_fallback', False),
                    'active_uri': conn_info.get('masked_active_uri', 'Unknown'),
                    'reason': conn_info.get('fallback_reason')
                },
                message=message
            )

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return HealthMetric(
                name="database_health",
                status='critical',
                value=None,
                message=f"❌ Database connection failed: {str(e)}"
            )

    def check_memory_usage(self) -> HealthMetric:
        """Check memory usage and optimization status."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb < 400:
                status = 'healthy'
                message = f"✅ Memory: {memory_mb:.1f}MB (optimal)"
            elif memory_mb < 500:
                status = 'warning'
                message = f"⚠️ Memory: {memory_mb:.1f}MB (approaching threshold)"
            else:
                status = 'critical'
                message = f"❌ Memory: {memory_mb:.1f}MB (exceeds target)"

            return HealthMetric(
                name="memory_usage",
                status=status,
                value={'memory_mb': round(memory_mb, 2)},
                threshold={'target_mb': 400, 'max_mb': 500},
                message=message
            )

        except ImportError:
            return HealthMetric(
                name="memory_usage",
                status='unknown',
                value=None,
                message="ℹ️ psutil not available for memory monitoring"
            )
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return HealthMetric(
                name="memory_usage",
                status='unknown',
                value=None,
                message=f"⚠️ Could not check memory: {str(e)}"
            )

    def check_data_freshness(self) -> HealthMetric:
        """Check data freshness and update timestamps."""
        try:
            from real_data_integrator import real_data_integrator

            snapshot = real_data_integrator.get_data_pipeline_snapshot()
            
            # Check last fetch time
            last_fetch = snapshot.get('last_fetch_completed_at')
            if last_fetch:
                age_seconds = time.time() - last_fetch
                age_hours = age_seconds / 3600

                if age_hours < 1:
                    status = 'healthy'
                    message = f"✅ Data refreshed {age_hours:.1f} hours ago"
                elif age_hours < 6:
                    status = 'warning'
                    message = f"⚠️ Data age: {age_hours:.1f} hours (consider refresh)"
                else:
                    status = 'critical'
                    message = f"❌ Data stale: {age_hours:.1f} hours old"

                return HealthMetric(
                    name="data_freshness",
                    status=status,
                    value={
                        'age_hours': round(age_hours, 2),
                        'last_fetch': datetime.fromtimestamp(last_fetch).isoformat(),
                        'real_matches': snapshot.get('last_real_match_count', 0),
                        'fallback_matches': snapshot.get('last_fallback_match_count', 0)
                    },
                    threshold={'max_age_hours': 6},
                    message=message
                )
            else:
                return HealthMetric(
                    name="data_freshness",
                    status='unknown',
                    value=None,
                    message="ℹ️ No data fetch timestamp available"
                )

        except Exception as e:
            logger.error(f"Data freshness check failed: {e}")
            return HealthMetric(
                name="data_freshness",
                status='unknown',
                value=None,
                message=f"⚠️ Could not check data freshness: {str(e)}"
            )

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return results."""
        logger.info("🔍 Running comprehensive health checks...")

        self.metrics = [
            self.check_real_data_integration(),
            self.check_api_connectivity(),
            self.check_prediction_quality(),
            self.check_database_health(),
            self.check_memory_usage(),
            self.check_data_freshness(),
        ]

        # Compute overall health
        statuses = [m.status for m in self.metrics]
        if 'critical' in statuses:
            overall = 'critical'
        elif 'warning' in statuses:
            overall = 'warning'
        elif 'unknown' in statuses:
            overall = 'degraded'
        else:
            overall = 'healthy'

        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_status': overall,
            'metrics': [asdict(m) for m in self.metrics],
            'summary': {
                'total_checks': len(self.metrics),
                'healthy': statuses.count('healthy'),
                'warning': statuses.count('warning'),
                'critical': statuses.count('critical'),
                'unknown': statuses.count('unknown')
            }
        }

        return results

    def save_report(self, results: Dict[str, Any]) -> Path:
        """Save health report to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.report_dir / f"health_report_{timestamp}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"📊 Health report saved to: {report_path}")
        return report_path

    def print_report(self, results: Dict[str, Any]):
        """Print formatted health report to console."""
        print("\n" + "="*70)
        print("🏥 PRODUCTION HEALTH REPORT")
        print("="*70)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Overall Status: {results['overall_status'].upper()}")
        print(f"\nSummary: {results['summary']['healthy']} healthy, "
              f"{results['summary']['warning']} warnings, "
              f"{results['summary']['critical']} critical")
        print("\n" + "-"*70)

        for metric in results['metrics']:
            status_icon = {
                'healthy': '✅',
                'warning': '⚠️',
                'critical': '❌',
                'unknown': 'ℹ️'
            }.get(metric['status'], '❓')

            print(f"\n{status_icon} {metric['name'].replace('_', ' ').title()}")
            print(f"   Status: {metric['status'].upper()}")
            print(f"   {metric['message']}")
            if metric['value']:
                print(f"   Details: {metric['value']}")

        print("\n" + "="*70 + "\n")


def main():
    """CLI entry point for health monitoring."""
    import argparse

    parser = argparse.ArgumentParser(description="GoalDiggers Production Health Monitor")
    parser.add_argument('--check-all', action='store_true', help='Run all health checks')
    parser.add_argument('--report', action='store_true', help='Generate and save health report')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=300, help='Watch interval in seconds (default: 300)')

    args = parser.parse_args()

    monitor = ProductionHealthMonitor()

    if args.watch:
        print(f"🔄 Starting continuous monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop...")
        try:
            while True:
                results = monitor.run_all_checks()
                monitor.print_report(results)
                
                if results['overall_status'] in ['critical', 'warning']:
                    monitor.save_report(results)
                
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            sys.exit(0)
    else:
        results = monitor.run_all_checks()
        monitor.print_report(results)

        if args.report or results['overall_status'] != 'healthy':
            monitor.save_report(results)

        # Exit with error code if critical issues detected
        if results['overall_status'] == 'critical':
            sys.exit(1)


if __name__ == "__main__":
    main()
