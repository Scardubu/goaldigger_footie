#!/usr/bin/env python3
"""
System Metrics Collector for Production Monitoring
Tracks performance, cache, API usage, and system health metrics
"""

import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """Collects and stores system metrics for monitoring dashboard"""
    
    def __init__(self, storage_path: str = "data/metrics"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics storage (recent data)
        self.latency_buffer = deque(maxlen=1000)
        self.cache_stats_buffer = deque(maxlen=1000)
        self.api_usage_buffer = deque(maxlen=1000)
        self.resource_usage_buffer = deque(maxlen=1000)
        self.quality_metrics_buffer = deque(maxlen=1000)
        self.alerts_buffer = deque(maxlen=100)
        
        # Current metrics
        self.current_metrics = {
            'prediction_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls_today': 0,
            'api_quota_used': 0,
            'api_quota_total': 10000,
            'memory_target_mb': 350,
            'start_time': time.time()
        }
        
        # Load persisted metrics
        self._load_metrics()
    
    def record_prediction_latency(self, latency: float, metadata: Optional[Dict] = None):
        """Record prediction latency"""
        self.latency_buffer.append({
            'timestamp': datetime.now(),
            'latency': latency,
            'metadata': metadata or {}
        })
        self.current_metrics['prediction_count'] += 1
        self._persist_metric('latency', latency)
    
    def record_cache_operation(self, operation: str, layer: str, hit: bool, metadata: Optional[Dict] = None):
        """Record cache operation"""
        if hit:
            self.current_metrics['cache_hits'] += 1
        else:
            self.current_metrics['cache_misses'] += 1
        
        self.cache_stats_buffer.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'layer': layer,
            'hit': hit,
            'metadata': metadata or {}
        })
        self._persist_metric('cache', {'operation': operation, 'layer': layer, 'hit': hit})
    
    def record_api_call(self, endpoint: str, response_time: float, success: bool, metadata: Optional[Dict] = None):
        """Record API call"""
        self.current_metrics['api_calls_today'] += 1
        
        self.api_usage_buffer.append({
            'timestamp': datetime.now(),
            'endpoint': endpoint,
            'response_time': response_time,
            'success': success,
            'metadata': metadata or {}
        })
        self._persist_metric('api', {'endpoint': endpoint, 'response_time': response_time, 'success': success})
    
    def record_resource_usage(self):
        """Record current resource usage"""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=0.1)
            
            self.resource_usage_buffer.append({
                'timestamp': datetime.now(),
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'active_sessions': self._get_active_sessions(),
                'requests_per_minute': self._calculate_rpm()
            })
        except Exception as e:
            logger.error(f"Failed to record resource usage: {e}")
    
    def record_quality_metric(self, quality_score: float, features: Dict[str, bool], confidence: float):
        """Record prediction quality metrics"""
        self.quality_metrics_buffer.append({
            'timestamp': datetime.now(),
            'quality_score': quality_score,
            'features': features,
            'confidence': confidence
        })
    
    def add_alert(self, severity: str, component: str, title: str, message: str, details: Optional[Dict] = None):
        """Add system alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,  # critical, warning, info
            'component': component,
            'title': title,
            'message': message,
            'details': details or {}
        }
        self.alerts_buffer.append(alert)
        
        # Persist critical alerts
        if severity == 'critical':
            self._persist_alert(alert)
        
        logger.log(
            logging.CRITICAL if severity == 'critical' else logging.WARNING,
            f"[ALERT] {component}: {title} - {message}"
        )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        total_cache_ops = self.current_metrics['cache_hits'] + self.current_metrics['cache_misses']
        cache_hit_rate = (self.current_metrics['cache_hits'] / total_cache_ops) if total_cache_ops > 0 else 0.0
        
        # Calculate recent latency
        recent_latencies = [m['latency'] for m in list(self.latency_buffer)[-100:]]
        avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0.0
        
        # Memory usage
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
        except:
            memory_mb = 0
        
        # API quota
        api_remaining = self.current_metrics['api_quota_total'] - self.current_metrics['api_quota_used']
        
        return {
            'cache_hit_rate': cache_hit_rate,
            'cache_hit_rate_delta': 0.0,  # TODO: Calculate delta from historical
            'prediction_latency_avg': avg_latency,
            'prediction_latency_delta': 0.0,  # TODO: Calculate delta
            'api_quota_remaining': max(0, api_remaining),
            'api_quota_total': self.current_metrics['api_quota_total'],
            'memory_usage_mb': memory_mb,
            'memory_target_mb': self.current_metrics['memory_target_mb'],
            'prediction_count': self.current_metrics['prediction_count'],
            'uptime_hours': (time.time() - self.current_metrics['start_time']) / 3600
        }
    
    def get_historical_metrics(self, time_range: str = "24h") -> Dict[str, List[Dict]]:
        """Get historical metrics for specified time range"""
        # Parse time range
        hours = {
            "1h": 1,
            "6h": 6,
            "24h": 24,
            "7d": 168,
            "30d": 720
        }.get(time_range, 24)
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filter buffers by time range
        def filter_by_time(buffer):
            return [m for m in buffer if m['timestamp'] > cutoff]
        
        # Process latency data
        latency_data = filter_by_time(self.latency_buffer)
        latency_history = self._compute_latency_percentiles(latency_data)
        
        # Process cache data
        cache_data = filter_by_time(self.cache_stats_buffer)
        cache_stats = self._compute_cache_stats(cache_data)
        
        # Process API data
        api_data = filter_by_time(self.api_usage_buffer)
        api_usage = self._compute_api_stats(api_data)
        
        # Process resource data
        resource_data = filter_by_time(self.resource_usage_buffer)
        
        return {
            'latency_history': latency_history,
            'cache_stats': cache_stats,
            'api_usage': api_usage,
            'resource_usage': resource_data
        }
    
    def get_health_status(self) -> Dict[str, Dict]:
        """Get current system health status"""
        current = self.get_current_metrics()
        
        health = {}
        
        # Cache health
        cache_hit_rate = current['cache_hit_rate']
        if cache_hit_rate > 0.7:
            health['cache'] = {'status': 'healthy', 'message': f'Hit rate: {cache_hit_rate:.1%}'}
        elif cache_hit_rate > 0.5:
            health['cache'] = {'status': 'warning', 'message': f'Low hit rate: {cache_hit_rate:.1%}'}
        else:
            health['cache'] = {'status': 'critical', 'message': f'Critical hit rate: {cache_hit_rate:.1%}'}
        
        # Memory health
        memory_pct = (current['memory_usage_mb'] / current['memory_target_mb']) * 100
        if memory_pct < 100:
            health['memory'] = {'status': 'healthy', 'message': f'{memory_pct:.0f}% of target'}
        elif memory_pct < 120:
            health['memory'] = {'status': 'warning', 'message': f'{memory_pct:.0f}% of target'}
        else:
            health['memory'] = {'status': 'critical', 'message': f'{memory_pct:.0f}% of target'}
        
        # API quota health
        api_remaining_pct = (current['api_quota_remaining'] / current['api_quota_total']) * 100
        if api_remaining_pct > 20:
            health['api_quota'] = {'status': 'healthy', 'message': f'{api_remaining_pct:.0f}% remaining'}
        elif api_remaining_pct > 10:
            health['api_quota'] = {'status': 'warning', 'message': f'{api_remaining_pct:.0f}% remaining'}
        else:
            health['api_quota'] = {'status': 'critical', 'message': f'{api_remaining_pct:.0f}% remaining'}
        
        # Latency health
        avg_latency = current['prediction_latency_avg']
        if avg_latency < 0.5:
            health['latency'] = {'status': 'healthy', 'message': f'Avg: {avg_latency:.2f}s'}
        elif avg_latency < 1.0:
            health['latency'] = {'status': 'warning', 'message': f'Avg: {avg_latency:.2f}s'}
        else:
            health['latency'] = {'status': 'critical', 'message': f'Avg: {avg_latency:.2f}s'}
        
        # Calculate overall score
        status_scores = {'healthy': 100, 'warning': 60, 'critical': 20}
        scores = [status_scores[h['status']] for h in health.values()]
        overall_score = sum(scores) / len(scores) if scores else 0
        
        health['overall_score'] = overall_score
        
        return health
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get prediction quality metrics"""
        if not self.quality_metrics_buffer:
            return {
                'avg_quality_score': 0.0,
                'feature_completeness': 0.0,
                'avg_confidence': 0.0,
                'quality_distribution': {},
                'feature_availability': {}
            }
        
        recent_quality = list(self.quality_metrics_buffer)[-100:]
        
        # Average quality score
        avg_quality = sum(m['quality_score'] for m in recent_quality) / len(recent_quality)
        
        # Average confidence
        avg_confidence = sum(m['confidence'] for m in recent_quality) / len(recent_quality)
        
        # Quality distribution
        excellent = sum(1 for m in recent_quality if m['quality_score'] > 0.85)
        good = sum(1 for m in recent_quality if 0.70 <= m['quality_score'] <= 0.85)
        fair = sum(1 for m in recent_quality if m['quality_score'] < 0.70)
        
        # Feature availability
        feature_counts = defaultdict(int)
        for m in recent_quality:
            for feature, available in m['features'].items():
                if available:
                    feature_counts[feature] += 1
        
        feature_availability = {
            feature: (count / len(recent_quality)) * 100
            for feature, count in feature_counts.items()
        }
        
        # Feature completeness
        total_features = sum(1 for m in recent_quality for available in m['features'].values() if available)
        possible_features = len(recent_quality) * len(recent_quality[0]['features']) if recent_quality else 1
        feature_completeness = total_features / possible_features if possible_features > 0 else 0.0
        
        return {
            'avg_quality_score': avg_quality,
            'feature_completeness': feature_completeness,
            'avg_confidence': avg_confidence,
            'quality_distribution': {
                'excellent': excellent,
                'good': good,
                'fair': fair
            },
            'feature_availability': feature_availability,
            'historical_accuracy': None  # TODO: Implement historical accuracy tracking
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return list(self.alerts_buffer)[-limit:]
    
    def _compute_latency_percentiles(self, data: List[Dict]) -> List[Dict]:
        """Compute latency percentiles over time"""
        if not data:
            return []
        
        # Group by 5-minute intervals
        import statistics
        from collections import defaultdict
        
        intervals = defaultdict(list)
        for m in data:
            interval_key = m['timestamp'].replace(minute=m['timestamp'].minute // 5 * 5, second=0, microsecond=0)
            intervals[interval_key].append(m['latency'])
        
        result = []
        for timestamp, latencies in sorted(intervals.items()):
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            result.append({
                'timestamp': timestamp,
                'p50': sorted_latencies[int(n * 0.5)] if n > 0 else 0,
                'p95': sorted_latencies[int(n * 0.95)] if n > 0 else 0,
                'p99': sorted_latencies[int(n * 0.99)] if n > 0 else 0,
                'max': max(sorted_latencies) if n > 0 else 0,
                'avg': statistics.mean(sorted_latencies) if n > 0 else 0
            })
        
        return result
    
    def _compute_cache_stats(self, data: List[Dict]) -> List[Dict]:
        """Compute cache statistics over time"""
        if not data:
            return []
        
        from collections import defaultdict
        
        intervals = defaultdict(lambda: {'hits': 0, 'misses': 0, 'l1': 0, 'l2': 0, 'l3': 0})
        for m in data:
            interval_key = m['timestamp'].replace(minute=m['timestamp'].minute // 5 * 5, second=0, microsecond=0)
            if m['hit']:
                intervals[interval_key]['hits'] += 1
            else:
                intervals[interval_key]['misses'] += 1
            
            layer = m.get('layer', 'l1')
            if layer in ['l1', 'l2', 'l3']:
                intervals[interval_key][layer] += 1
        
        result = []
        for timestamp, stats in sorted(intervals.items()):
            total = stats['hits'] + stats['misses']
            hit_rate = (stats['hits'] / total) if total > 0 else 0
            
            result.append({
                'timestamp': timestamp,
                'hits': stats['hits'],
                'misses': stats['misses'],
                'hit_rate': hit_rate,
                'l1_hit_rate': hit_rate * 0.7,  # Approximate distribution
                'l2_hit_rate': hit_rate * 0.2,
                'l3_hit_rate': hit_rate * 0.1,
                'size_mb': 0  # TODO: Track actual cache size
            })
        
        return result
    
    def _compute_api_stats(self, data: List[Dict]) -> List[Dict]:
        """Compute API usage statistics over time"""
        if not data:
            return []
        
        import statistics
        from collections import defaultdict
        
        intervals = defaultdict(lambda: {
            'calls': 0,
            'response_times': [],
            'errors': 0,
            'rate_limits': 0
        })
        
        for m in data:
            interval_key = m['timestamp'].replace(minute=m['timestamp'].minute // 5 * 5, second=0, microsecond=0)
            intervals[interval_key]['calls'] += 1
            intervals[interval_key]['response_times'].append(m['response_time'])
            if not m['success']:
                intervals[interval_key]['errors'] += 1
        
        result = []
        cumulative_calls = 0
        for timestamp, stats in sorted(intervals.items()):
            cumulative_calls += stats['calls']
            avg_response_time = statistics.mean(stats['response_times']) if stats['response_times'] else 0
            error_rate = (stats['errors'] / stats['calls']) if stats['calls'] > 0 else 0
            
            result.append({
                'timestamp': timestamp,
                'calls_per_minute': stats['calls'],
                'calls_today': cumulative_calls,
                'quota_remaining': max(0, self.current_metrics['api_quota_total'] - cumulative_calls),
                'avg_response_time': avg_response_time,
                'error_rate': error_rate,
                'rate_limit_hits': stats['rate_limits']
            })
        
        return result
    
    def _get_active_sessions(self) -> int:
        """Get count of active user sessions"""
        # TODO: Implement actual session tracking
        return 0
    
    def _calculate_rpm(self) -> float:
        """Calculate requests per minute"""
        recent = list(self.latency_buffer)[-60:]  # Last minute
        return len(recent)
    
    def _persist_metric(self, metric_type: str, data: Any):
        """Persist metric to disk"""
        try:
            filename = self.storage_path / f"{metric_type}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(filename, 'a') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to persist metric {metric_type}: {e}")
    
    def _persist_alert(self, alert: Dict):
        """Persist critical alert"""
        try:
            filename = self.storage_path / "alerts.jsonl"
            with open(filename, 'a') as f:
                json.dump(alert, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")
    
    def _load_metrics(self):
        """Load persisted metrics from disk"""
        # TODO: Implement loading of historical metrics
        pass

# Global instance
_metrics_collector = None

def get_metrics_collector() -> SystemMetricsCollector:
    """Get or create global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = SystemMetricsCollector()
    return _metrics_collector
