#!/usr/bin/env python3
"""
Performance Analytics System

Advanced performance monitoring and analytics system for tracking system performance,
user behavior, prediction accuracy, and optimization opportunities.

Key Features:
- Real-time performance monitoring
- User behavior analytics
- Prediction accuracy tracking
- System optimization recommendations
- Performance trend analysis
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceAnalytics:
    """
    Advanced performance analytics system for comprehensive monitoring and optimization.
    
    Tracks system performance, user behavior, and provides actionable insights
    for continuous improvement and optimization.
    """
    
    def __init__(self):
        """Initialize performance analytics system."""
        self.logger = logger
        self._metrics_store = {}
        self._user_sessions = {}
        self._prediction_history = []
        self._performance_trends = {}
        
        # Performance thresholds
        self._performance_thresholds = {
            'load_time_warning': 3.0,  # 3 seconds
            'load_time_critical': 5.0,  # 5 seconds
            'memory_warning': 200,  # 200MB
            'memory_critical': 500,  # 500MB
            'prediction_accuracy_min': 0.60,  # 60% minimum accuracy
            'user_engagement_min': 0.30  # 30% minimum engagement
        }
        
        # Initialize metrics
        self._initialize_metrics()
        
        self.logger.info("📈 Performance Analytics initialized")
    
    def _initialize_metrics(self):
        """Initialize performance metrics tracking."""
        self._metrics_store = {
            'system_performance': {
                'load_times': [],
                'memory_usage': [],
                'cpu_usage': [],
                'response_times': [],
                'error_rates': []
            },
            'user_behavior': {
                'session_durations': [],
                'page_views': [],
                'interactions': [],
                'conversion_events': [],
                'bounce_rates': []
            },
            'prediction_performance': {
                'accuracy_scores': [],
                'confidence_levels': [],
                'prediction_times': [],
                'model_performance': {}
            },
            'feature_usage': {
                'cross_league_usage': 0,
                'value_betting_usage': 0,
                'personalization_usage': 0,
                'real_time_usage': 0
            }
        }
    
    def track_performance_metrics(self, metric_type: str, metric_data: Dict[str, Any]):
        """Track performance metrics."""
        try:
            timestamp = datetime.now()
            
            if metric_type == 'system_performance':
                self._track_system_performance(metric_data, timestamp)
            elif metric_type == 'user_behavior':
                self._track_user_behavior(metric_data, timestamp)
            elif metric_type == 'prediction_performance':
                self._track_prediction_performance(metric_data, timestamp)
            elif metric_type == 'feature_usage':
                self._track_feature_usage(metric_data, timestamp)
            
            self.logger.debug(f"📊 Tracked {metric_type} metrics")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to track {metric_type} metrics: {e}")
    
    def _track_system_performance(self, data: Dict[str, Any], timestamp: datetime):
        """Track system performance metrics."""
        metrics = self._metrics_store['system_performance']
        
        if 'load_time' in data:
            metrics['load_times'].append({
                'value': data['load_time'],
                'timestamp': timestamp,
                'component': data.get('component', 'unknown')
            })
        
        if 'memory_usage' in data:
            metrics['memory_usage'].append({
                'value': data['memory_usage'],
                'timestamp': timestamp
            })
        
        if 'response_time' in data:
            metrics['response_times'].append({
                'value': data['response_time'],
                'timestamp': timestamp,
                'endpoint': data.get('endpoint', 'unknown')
            })
        
        # Keep only last 1000 entries for performance
        for key in metrics:
            if len(metrics[key]) > 1000:
                metrics[key] = metrics[key][-1000:]
    
    def _track_user_behavior(self, data: Dict[str, Any], timestamp: datetime):
        """Track user behavior metrics."""
        metrics = self._metrics_store['user_behavior']
        
        if 'session_duration' in data:
            metrics['session_durations'].append({
                'value': data['session_duration'],
                'timestamp': timestamp,
                'user_id': data.get('user_id', 'anonymous')
            })
        
        if 'page_view' in data:
            metrics['page_views'].append({
                'page': data['page_view'],
                'timestamp': timestamp,
                'user_id': data.get('user_id', 'anonymous')
            })
        
        if 'interaction' in data:
            metrics['interactions'].append({
                'type': data['interaction'],
                'timestamp': timestamp,
                'user_id': data.get('user_id', 'anonymous')
            })
    
    def _track_prediction_performance(self, data: Dict[str, Any], timestamp: datetime):
        """Track prediction performance metrics."""
        metrics = self._metrics_store['prediction_performance']
        
        if 'accuracy' in data:
            metrics['accuracy_scores'].append({
                'value': data['accuracy'],
                'timestamp': timestamp,
                'model': data.get('model', 'unknown')
            })
        
        if 'confidence' in data:
            metrics['confidence_levels'].append({
                'value': data['confidence'],
                'timestamp': timestamp
            })
        
        if 'prediction_time' in data:
            metrics['prediction_times'].append({
                'value': data['prediction_time'],
                'timestamp': timestamp
            })
    
    def _track_feature_usage(self, data: Dict[str, Any], timestamp: datetime):
        """Track feature usage metrics."""
        metrics = self._metrics_store['feature_usage']
        
        for feature in ['cross_league_usage', 'value_betting_usage', 'personalization_usage', 'real_time_usage']:
            if feature in data:
                metrics[feature] += data[feature]
    
    def generate_analytics_report(self, time_period: str = '24h') -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        try:
            report_start = time.time()
            
            # Calculate time range
            now = datetime.now()
            if time_period == '1h':
                start_time = now - timedelta(hours=1)
            elif time_period == '24h':
                start_time = now - timedelta(hours=24)
            elif time_period == '7d':
                start_time = now - timedelta(days=7)
            else:
                start_time = now - timedelta(hours=24)
            
            # Generate report sections
            system_report = self._generate_system_performance_report(start_time, now)
            user_report = self._generate_user_behavior_report(start_time, now)
            prediction_report = self._generate_prediction_performance_report(start_time, now)
            feature_report = self._generate_feature_usage_report()
            
            # Generate recommendations
            recommendations = self._generate_optimization_recommendations(
                system_report, user_report, prediction_report
            )
            
            report_time = time.time() - report_start
            
            return {
                'report_metadata': {
                    'generated_at': now.isoformat(),
                    'time_period': time_period,
                    'start_time': start_time.isoformat(),
                    'end_time': now.isoformat(),
                    'generation_time': report_time
                },
                'system_performance': system_report,
                'user_behavior': user_report,
                'prediction_performance': prediction_report,
                'feature_usage': feature_report,
                'optimization_recommendations': recommendations,
                'overall_health_score': self._calculate_overall_health_score(
                    system_report, user_report, prediction_report
                )
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate analytics report: {e}")
            return {'error': str(e)}
    
    def _generate_system_performance_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate system performance report."""
        metrics = self._metrics_store['system_performance']
        
        # Filter metrics by time range
        load_times = [m for m in metrics['load_times'] if start_time <= m['timestamp'] <= end_time]
        memory_usage = [m for m in metrics['memory_usage'] if start_time <= m['timestamp'] <= end_time]
        response_times = [m for m in metrics['response_times'] if start_time <= m['timestamp'] <= end_time]
        
        return {
            'load_times': {
                'average': sum(m['value'] for m in load_times) / max(len(load_times), 1),
                'max': max((m['value'] for m in load_times), default=0),
                'min': min((m['value'] for m in load_times), default=0),
                'count': len(load_times)
            },
            'memory_usage': {
                'average': sum(m['value'] for m in memory_usage) / max(len(memory_usage), 1),
                'max': max((m['value'] for m in memory_usage), default=0),
                'current': memory_usage[-1]['value'] if memory_usage else 0
            },
            'response_times': {
                'average': sum(m['value'] for m in response_times) / max(len(response_times), 1),
                'max': max((m['value'] for m in response_times), default=0),
                'count': len(response_times)
            }
        }
    
    def _generate_user_behavior_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate user behavior report."""
        metrics = self._metrics_store['user_behavior']
        
        # Filter metrics by time range
        sessions = [m for m in metrics['session_durations'] if start_time <= m['timestamp'] <= end_time]
        page_views = [m for m in metrics['page_views'] if start_time <= m['timestamp'] <= end_time]
        interactions = [m for m in metrics['interactions'] if start_time <= m['timestamp'] <= end_time]
        
        return {
            'sessions': {
                'total': len(sessions),
                'average_duration': sum(m['value'] for m in sessions) / max(len(sessions), 1),
                'unique_users': len(set(m['user_id'] for m in sessions))
            },
            'page_views': {
                'total': len(page_views),
                'unique_pages': len(set(m['page'] for m in page_views))
            },
            'interactions': {
                'total': len(interactions),
                'types': list(set(m['type'] for m in interactions))
            }
        }
    
    def _generate_prediction_performance_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate prediction performance report."""
        metrics = self._metrics_store['prediction_performance']
        
        # Filter metrics by time range
        accuracy_scores = [m for m in metrics['accuracy_scores'] if start_time <= m['timestamp'] <= end_time]
        confidence_levels = [m for m in metrics['confidence_levels'] if start_time <= m['timestamp'] <= end_time]
        prediction_times = [m for m in metrics['prediction_times'] if start_time <= m['timestamp'] <= end_time]
        
        return {
            'accuracy': {
                'average': sum(m['value'] for m in accuracy_scores) / max(len(accuracy_scores), 1),
                'max': max((m['value'] for m in accuracy_scores), default=0),
                'min': min((m['value'] for m in accuracy_scores), default=0),
                'count': len(accuracy_scores)
            },
            'confidence': {
                'average': sum(m['value'] for m in confidence_levels) / max(len(confidence_levels), 1),
                'count': len(confidence_levels)
            },
            'prediction_times': {
                'average': sum(m['value'] for m in prediction_times) / max(len(prediction_times), 1),
                'count': len(prediction_times)
            }
        }
    
    def _generate_feature_usage_report(self) -> Dict[str, Any]:
        """Generate feature usage report."""
        return self._metrics_store['feature_usage'].copy()
    
    def _generate_optimization_recommendations(self, system_report: Dict, user_report: Dict, prediction_report: Dict) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # System performance recommendations
        avg_load_time = system_report['load_times']['average']
        if avg_load_time > self._performance_thresholds['load_time_critical']:
            recommendations.append("🚨 Critical: Load times exceed 5s - immediate optimization required")
        elif avg_load_time > self._performance_thresholds['load_time_warning']:
            recommendations.append("⚠️ Warning: Load times exceed 3s - consider optimization")
        
        # Memory usage recommendations
        max_memory = system_report['memory_usage']['max']
        if max_memory > self._performance_thresholds['memory_critical']:
            recommendations.append("🚨 Critical: Memory usage exceeds 500MB - memory leak investigation needed")
        elif max_memory > self._performance_thresholds['memory_warning']:
            recommendations.append("⚠️ Warning: Memory usage exceeds 200MB - monitor for leaks")
        
        # User behavior recommendations
        avg_session = user_report['sessions']['average_duration']
        if avg_session < 60:  # Less than 1 minute
            recommendations.append("📊 User engagement low - consider UX improvements")
        
        # Prediction performance recommendations
        avg_accuracy = prediction_report['accuracy']['average']
        if avg_accuracy < self._performance_thresholds['prediction_accuracy_min']:
            recommendations.append("🎯 Prediction accuracy below 60% - model retraining recommended")
        
        if not recommendations:
            recommendations.append("✅ All systems performing within acceptable parameters")
        
        return recommendations
    
    def _calculate_overall_health_score(self, system_report: Dict, user_report: Dict, prediction_report: Dict) -> float:
        """Calculate overall system health score (0-100)."""
        scores = []
        
        # System performance score (0-100)
        load_time_score = max(0, 100 - (system_report['load_times']['average'] * 20))
        scores.append(load_time_score)
        
        # User engagement score (0-100)
        session_score = min(100, user_report['sessions']['average_duration'] / 3)  # 3 minutes = 100%
        scores.append(session_score)
        
        # Prediction accuracy score (0-100)
        accuracy_score = prediction_report['accuracy']['average'] * 100
        scores.append(accuracy_score)
        
        return sum(scores) / len(scores)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get quick performance summary."""
        return {
            'total_metrics_tracked': sum(len(category) for category in self._metrics_store.values() if isinstance(category, list)),
            'active_sessions': len(self._user_sessions),
            'prediction_history_size': len(self._prediction_history),
            'last_updated': datetime.now().isoformat()
        }

def main():
    """Test the performance analytics system."""
    analytics = PerformanceAnalytics()
    
    # Test tracking
    analytics.track_performance_metrics('system_performance', {
        'load_time': 2.5,
        'memory_usage': 150,
        'component': 'dashboard'
    })
    
    # Generate report
    report = analytics.generate_analytics_report('24h')
    
    print("📈 Performance Analytics Test:")
    print(f"Overall Health Score: {report['overall_health_score']:.1f}%")
    print(f"System Load Time: {report['system_performance']['load_times']['average']:.2f}s")
    print(f"Recommendations: {len(report['optimization_recommendations'])}")

if __name__ == "__main__":
    main()
