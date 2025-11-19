#!/usr/bin/env python3
"""
Metrics Integration Script - Instruments core components with Prometheus metrics

This script adds metrics tracking to:
1. Enhanced Cache Manager (utils/enhanced_cache_manager.py)
2. Async Data Integrator (async_data_integrator.py)
3. Batched Prediction Engine (batched_prediction_engine.py)
4. Production Homepage (dashboard/enhanced_production_homepage.py)
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MetricsIntegrator:
    """Integrates Prometheus metrics into core components."""
    
    def __init__(self, project_root: Path = Path(__file__).parent.parent):
        self.project_root = project_root
        self.modifications = []
    
    def integrate_cache_manager(self) -> bool:
        """Add metrics tracking to EnhancedCacheManager."""
        logger.info("Integrating metrics into EnhancedCacheManager...")
        
        cache_file = self.project_root / 'utils' / 'enhanced_cache_manager.py'
        if not cache_file.exists():
            logger.error(f"Cache manager not found: {cache_file}")
            return False
        
        content = cache_file.read_text(encoding='utf-8')
        
        # Check if already integrated
        if 'from utils.metrics_exporter import' in content:
            logger.info("✅ Cache manager already has metrics integrated")
            return True
        
        # Add import after existing imports
        import_section = """import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

# Metrics tracking
try:
    from utils.metrics_exporter import track_cache_operation, update_cache_metrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    def track_cache_operation(*args, **kwargs): pass
    def update_cache_metrics(*args, **kwargs): pass

logger = logging.getLogger(__name__)"""
        
        # Replace old import section
        content = re.sub(
            r'import logging\nimport pickle\nimport time\nfrom datetime import datetime, timedelta\nfrom pathlib import Path\nfrom typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union\n\nlogger = logging.getLogger\(__name__\)',
            import_section,
            content,
            count=1
        )
        
        # Add metrics to get() method - track hits/misses
        content = re.sub(
            r'(def get\(self, key: str, default: Any = None\) -> Optional\[Any\]:.*?\n)(.*?)(\s+# Try L1)',
            r'\1\2        _tier_hit = None\3',
            content,
            flags=re.DOTALL
        )
        
        # Track L1 hit
        content = re.sub(
            r'(entry\.access\(\)\n\s+return entry\.value)',
            r'\1\n                if METRICS_AVAILABLE:\n                    track_cache_operation("L1", "get", "hit")',
            content
        )
        
        # Track L1 miss, L2/L3 hit, overall miss
        content = re.sub(
            r'(# L1 miss - try L2.*?\n)(.*?)(return default)',
            r'\1\2            if METRICS_AVAILABLE:\n                track_cache_operation("L1", "get", "miss")\n        \3',
            content,
            flags=re.DOTALL
        )
        
        # Add metrics to stats() method
        stats_pattern = r'(def stats\(self\) -> Dict\[str, Any\]:.*?return \{)(.*?)(\})'
        
        def add_metrics_update(match):
            stats_dict = match.group(2)
            return (
                match.group(1) + 
                stats_dict +
                "\n\n        # Update Prometheus metrics\n"
                "        if METRICS_AVAILABLE:\n"
                "            l1_stats = {'hit_rate': l1_hit_rate, 'entries': len(self._l1_cache)}\n"
                "            update_cache_metrics('L1', l1_stats)\n"
                "            if self._redis_enabled:\n"
                "                l2_stats = {'hit_rate': l2_hit_rate if l2_attempts > 0 else 0.0}\n"
                "                update_cache_metrics('L2', l2_stats)\n"
                "            l3_stats = {'hit_rate': l3_hit_rate if l3_attempts > 0 else 0.0, 'entries': len(list(self.cache_dir.glob('*.pkl')))}\n"
                "            update_cache_metrics('L3', l3_stats)\n" +
                match.group(3)
            )
        
        content = re.sub(stats_pattern, add_metrics_update, content, flags=re.DOTALL)
        
        # Write back
        cache_file.write_text(content, encoding='utf-8')
        logger.info("✅ Metrics integrated into EnhancedCacheManager")
        self.modifications.append("Enhanced Cache Manager")
        return True
    
    def integrate_api_integrator(self) -> bool:
        """Add metrics tracking to AsyncDataIntegrator."""
        logger.info("Integrating metrics into AsyncDataIntegrator...")
        
        api_file = self.project_root / 'async_data_integrator.py'
        if not api_file.exists():
            logger.error(f"API integrator not found: {api_file}")
            return False
        
        content = api_file.read_text(encoding='utf-8')
        
        # Check if already integrated
        if 'from utils.metrics_exporter import' in content:
            logger.info("✅ API integrator already has metrics integrated")
            return True
        
        # Add import after logging import
        if 'import logging' in content and 'from utils.metrics_exporter' not in content:
            content = content.replace(
                'import logging\n',
                'import logging\n\n# Metrics tracking\n'
                'try:\n'
                '    from utils.metrics_exporter import track_api_request\n'
                '    METRICS_AVAILABLE = True\n'
                'except ImportError:\n'
                '    METRICS_AVAILABLE = False\n'
                '    def track_api_request(*args, **kwargs): pass\n\n'
            )
        
        # Track API requests in _make_request method
        # Find the method and wrap the actual request
        content = re.sub(
            r'(async def _make_request\(.*?\):.*?)(async with self\.session\.get\(url)',
            r'\1\n        # Track API request\n'
            r'        if METRICS_AVAILABLE:\n'
            r'            from utils.metrics_exporter import track_api_request\n'
            r'            async with track_api_request("football-data", endpoint):\n'
            r'                \2',
            content,
            flags=re.DOTALL
        )
        
        # Write back
        api_file.write_text(content, encoding='utf-8')
        logger.info("✅ Metrics integrated into AsyncDataIntegrator")
        self.modifications.append("Async Data Integrator")
        return True
    
    def integrate_prediction_engine(self) -> bool:
        """Add metrics tracking to BatchedPredictionEngine."""
        logger.info("Integrating metrics into BatchedPredictionEngine...")
        
        pred_file = self.project_root / 'batched_prediction_engine.py'
        if not pred_file.exists():
            logger.error(f"Prediction engine not found: {pred_file}")
            return False
        
        content = pred_file.read_text(encoding='utf-8')
        
        # Check if already integrated
        if 'from utils.metrics_exporter import' in content:
            logger.info("✅ Prediction engine already has metrics integrated")
            return True
        
        # Add import
        if 'import logging' in content:
            content = content.replace(
                'import logging\n',
                'import logging\n\n# Metrics tracking\n'
                'try:\n'
                '    from utils.metrics_exporter import track_prediction, record_prediction_confidence\n'
                '    METRICS_AVAILABLE = True\n'
                'except ImportError:\n'
                '    METRICS_AVAILABLE = False\n'
                '    class track_prediction:\n'
                '        def __init__(self, *args, **kwargs): pass\n'
                '        def __enter__(self): return self\n'
                '        def __exit__(self, *args): pass\n'
                '    def record_prediction_confidence(*args, **kwargs): pass\n\n'
            )
        
        # Track predictions in _generate_single_prediction
        content = re.sub(
            r'(def _generate_single_prediction\(self, request: PredictionRequest\) -> PredictionResult:.*?)(# Generate features)',
            r'\1\n        # Track prediction metrics\n'
            r'        prediction_start = time.time()\n'
            r'        league_code = request.league or "unknown"\n'
            r'        \n        \2',
            content,
            flags=re.DOTALL,
            count=1
        )
        
        # Record confidence after prediction
        content = re.sub(
            r'(confidence = ensemble_result\.get\(\'confidence\', 0\.0\).*?\n)',
            r'\1\n        # Record prediction confidence\n'
            r'        if METRICS_AVAILABLE:\n'
            r'            record_prediction_confidence("ensemble", confidence)\n',
            content,
            flags=re.DOTALL
        )
        
        # Write back
        pred_file.write_text(content, encoding='utf-8')
        logger.info("✅ Metrics integrated into BatchedPredictionEngine")
        self.modifications.append("Batched Prediction Engine")
        return True
    
    def integrate_homepage(self) -> bool:
        """Add metrics tracking to production homepage."""
        logger.info("Integrating metrics into ProductionHomepage...")
        
        homepage_file = self.project_root / 'dashboard' / 'enhanced_production_homepage.py'
        if not homepage_file.exists():
            logger.error(f"Homepage not found: {homepage_file}")
            return False
        
        content = homepage_file.read_text(encoding='utf-8')
        
        # Check if already integrated
        if 'from utils.metrics_exporter import' in content:
            logger.info("✅ Homepage already has metrics integrated")
            return True
        
        # Add import after existing imports
        import_marker = "from dashboard.components.unified_design_system import"
        if import_marker in content:
            content = content.replace(
                import_marker,
                '# Metrics tracking\n'
                'try:\n'
                '    from utils.metrics_exporter import track_page_view, track_user_interaction\n'
                '    METRICS_AVAILABLE = True\n'
                'except ImportError:\n'
                '    METRICS_AVAILABLE = False\n'
                '    def track_page_view(*args, **kwargs): pass\n'
                '    def track_user_interaction(*args, **kwargs): pass\n\n' +
                import_marker
            )
        
        # Track page view in render_production_homepage
        content = re.sub(
            r'(def render_production_homepage\(self\):.*?)(""".*?""")',
            r'\1\2\n        # Track page view\n'
            r'        if METRICS_AVAILABLE:\n'
            r'            track_page_view("homepage")',
            content,
            flags=re.DOTALL,
            count=1
        )
        
        # Track prediction requests
        content = re.sub(
            r'(if st\.button\(.*?Generate Prediction.*?\):)',
            r'\1\n                # Track user interaction\n'
            r'                if METRICS_AVAILABLE:\n'
            r'                    track_user_interaction("prediction_request")',
            content,
            flags=re.DOTALL
        )
        
        # Write back
        homepage_file.write_text(content, encoding='utf-8')
        logger.info("✅ Metrics integrated into ProductionHomepage")
        self.modifications.append("Production Homepage")
        return True
    
    def create_grafana_dashboard(self) -> bool:
        """Create Grafana dashboard JSON configuration."""
        logger.info("Creating Grafana dashboard configuration...")
        
        monitoring_dir = self.project_root / 'monitoring' / 'grafana' / 'dashboards'
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        dashboard_config = {
            "dashboard": {
                "title": "GoalDiggers Platform Overview",
                "tags": ["goaldiggers", "production", "monitoring"],
                "timezone": "browser",
                "refresh": "30s",
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "panels": [
                    {
                        "id": 1,
                        "title": "API Success Rate",
                        "type": "stat",
                        "targets": [{
                            "expr": "rate(goaldiggers_api_requests_total{status=\"success\"}[5m]) / rate(goaldiggers_api_requests_total[5m])",
                            "legendFormat": "Success Rate"
                        }],
                        "gridPos": {"x": 0, "y": 0, "w": 6, "h": 4},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percentunit",
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"value": 0, "color": "red"},
                                        {"value": 0.95, "color": "yellow"},
                                        {"value": 0.99, "color": "green"}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 2,
                        "title": "Cache Hit Rate (L1)",
                        "type": "stat",
                        "targets": [{
                            "expr": "goaldiggers_cache_hit_rate{tier=\"L1\"}",
                            "legendFormat": "L1 Hit Rate"
                        }],
                        "gridPos": {"x": 6, "y": 0, "w": 6, "h": 4},
                        "fieldConfig": {
                            "defaults": {
                                "unit": "percentunit",
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"value": 0, "color": "red"},
                                        {"value": 0.7, "color": "yellow"},
                                        {"value": 0.85, "color": "green"}
                                    ]
                                }
                            }
                        }
                    },
                    {
                        "id": 3,
                        "title": "API P95 Latency",
                        "type": "graph",
                        "targets": [{
                            "expr": "histogram_quantile(0.95, rate(goaldiggers_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "P95 Latency"
                        }],
                        "gridPos": {"x": 12, "y": 0, "w": 6, "h": 4},
                        "yaxes": [
                            {"format": "s", "label": "Latency"},
                            {"format": "short"}
                        ]
                    },
                    {
                        "id": 4,
                        "title": "Active Users",
                        "type": "stat",
                        "targets": [{
                            "expr": "goaldiggers_active_users_total",
                            "legendFormat": "Active Users"
                        }],
                        "gridPos": {"x": 18, "y": 0, "w": 6, "h": 4}
                    },
                    {
                        "id": 5,
                        "title": "Predictions Per Minute",
                        "type": "graph",
                        "targets": [{
                            "expr": "rate(goaldiggers_predictions_total[1m]) * 60",
                            "legendFormat": "Predictions/min"
                        }],
                        "gridPos": {"x": 0, "y": 4, "w": 12, "h": 6}
                    },
                    {
                        "id": 6,
                        "title": "Data Freshness (Fixtures)",
                        "type": "graph",
                        "targets": [{
                            "expr": "goaldiggers_data_freshness_seconds{data_type=\"fixtures\"} / 60",
                            "legendFormat": "Age (minutes)"
                        }],
                        "gridPos": {"x": 12, "y": 4, "w": 12, "h": 6},
                        "yaxes": [
                            {"format": "m", "label": "Age"},
                            {"format": "short"}
                        ]
                    }
                ]
            },
            "overwrite": True
        }
        
        dashboard_file = monitoring_dir / 'goaldiggers-overview.json'
        import json
        dashboard_file.write_text(json.dumps(dashboard_config, indent=2), encoding='utf-8')
        
        logger.info(f"✅ Grafana dashboard created: {dashboard_file}")
        self.modifications.append("Grafana Dashboard")
        return True
    
    def create_prometheus_alerts(self) -> bool:
        """Create Prometheus alert rules."""
        logger.info("Creating Prometheus alert rules...")
        
        monitoring_dir = self.project_root / 'monitoring' / 'prometheus'
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        alerts_config = """groups:
  - name: goaldiggers_alerts
    interval: 30s
    rules:
      - alert: HighAPIErrorRate
        expr: rate(goaldiggers_api_errors_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High API error rate detected"
          description: "API error rate is {{ $value }} errors/sec"
      
      - alert: LowCacheHitRate
        expr: goaldiggers_cache_hit_rate{tier="L1"} < 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "L1 cache hit rate below 50%"
          description: "Consider increasing L1 cache size or investigating cache invalidation"
      
      - alert: SlowAPIResponses
        expr: histogram_quantile(0.95, rate(goaldiggers_api_request_duration_seconds_bucket[5m])) > 1.0
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "API P95 latency above 1 second"
          description: "P95 latency is {{ $value }}s - investigate slow endpoints"
      
      - alert: StaleData
        expr: goaldiggers_data_freshness_seconds{data_type="fixtures"} > 1800
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Fixture data is stale (>30min old)"
          description: "Data age is {{ $value }}s - check data ingestion pipeline"
      
      - alert: PredictionErrors
        expr: rate(goaldiggers_prediction_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High prediction error rate"
          description: "Prediction errors at {{ $value }}/s - check model health"
"""
        
        alerts_file = monitoring_dir / 'alerts.yml'
        alerts_file.write_text(alerts_config, encoding='utf-8')
        
        logger.info(f"✅ Prometheus alert rules created: {alerts_file}")
        self.modifications.append("Prometheus Alerts")
        return True
    
    def run_integration(self) -> bool:
        """Execute all integration steps."""
        logger.info("=" * 60)
        logger.info("Starting Metrics Integration")
        logger.info("=" * 60)
        
        success = True
        
        # Integrate components
        success &= self.integrate_cache_manager()
        success &= self.integrate_api_integrator()
        success &= self.integrate_prediction_engine()
        success &= self.integrate_homepage()
        
        # Create monitoring configs
        success &= self.create_grafana_dashboard()
        success &= self.create_prometheus_alerts()
        
        logger.info("=" * 60)
        logger.info(f"Integration {'COMPLETE' if success else 'FAILED'}")
        logger.info("=" * 60)
        logger.info(f"Modified components: {', '.join(self.modifications)}")
        
        return success


def main():
    integrator = MetricsIntegrator()
    success = integrator.run_integration()
    
    if success:
        logger.info("\n✅ Phase 7 Metrics Integration Complete!")
        logger.info("\nNext steps:")
        logger.info("1. Restart the dashboard: python unified_launcher.py dashboard --port 8501")
        logger.info("2. Start monitoring stack: docker-compose --profile monitoring up -d")
        logger.info("3. Access Grafana: http://localhost:3000 (admin/goaldiggers2025)")
        logger.info("4. Verify metrics: http://localhost:8501/metrics")
        return 0
    else:
        logger.error("\n❌ Integration failed - check logs above")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
