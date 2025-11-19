#!/usr/bin/env python3
"""
Prometheus Metrics Exporter for GoalDiggers Platform

Provides comprehensive instrumentation for production monitoring:
- API request metrics (count, latency, errors)
- Cache performance (hit rates per tier, evictions)
- Prediction metrics (latency, confidence, volume)
- User activity (active sessions, page views)
- Data pipeline health (freshness, ingestion rate)

Usage:
    from utils.metrics_exporter import (
        track_api_request,
        track_cache_operation,
        track_prediction,
        metrics_registry
    )
    
    # Instrument API calls
    with track_api_request("football-data", "/fixtures"):
        response = await api_call()
    
    # Export metrics endpoint
    from prometheus_client import generate_latest
    metrics_data = generate_latest(metrics_registry)
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Callable, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

logger = logging.getLogger(__name__)

# ============================================================================
# METRICS REGISTRY
# ============================================================================

# Create custom registry to avoid conflicts with default registry
metrics_registry = CollectorRegistry()

# ============================================================================
# API METRICS
# ============================================================================

api_requests_total = Counter(
    'goaldiggers_api_requests_total',
    'Total number of API requests by provider and endpoint',
    ['provider', 'endpoint', 'status'],
    registry=metrics_registry
)

api_request_duration_seconds = Histogram(
    'goaldiggers_api_request_duration_seconds',
    'API request latency in seconds',
    ['provider', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=metrics_registry
)

api_errors_total = Counter(
    'goaldiggers_api_errors_total',
    'Total number of API errors by provider and error type',
    ['provider', 'error_type'],
    registry=metrics_registry
)

api_rate_limit_hits_total = Counter(
    'goaldiggers_api_rate_limit_hits_total',
    'Total number of rate limit hits by provider',
    ['provider'],
    registry=metrics_registry
)

# Circuit breaker metrics
api_circuit_breaker_state = Gauge(
    'goaldiggers_api_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half_open, 2=open)',
    ['provider'],
    registry=metrics_registry
)

api_circuit_breaker_failures = Counter(
    'goaldiggers_api_circuit_breaker_failures_total',
    'Total circuit breaker failures by provider',
    ['provider'],
    registry=metrics_registry
)

api_circuit_breaker_opens = Counter(
    'goaldiggers_api_circuit_breaker_opens_total',
    'Total circuit breaker openings by provider',
    ['provider'],
    registry=metrics_registry
)

api_fallback_usage = Counter(
    'goaldiggers_api_fallback_usage_total',
    'Total fallback data source usage by provider and reason',
    ['provider', 'reason'],
    registry=metrics_registry
)

# ============================================================================
# CACHE METRICS
# ============================================================================

cache_operations_total = Counter(
    'goaldiggers_cache_operations_total',
    'Total cache operations by tier and operation type',
    ['tier', 'operation', 'result'],
    registry=metrics_registry
)

cache_hit_rate = Gauge(
    'goaldiggers_cache_hit_rate',
    'Cache hit rate by tier (0.0 to 1.0)',
    ['tier'],
    registry=metrics_registry
)

cache_entries = Gauge(
    'goaldiggers_cache_entries',
    'Number of entries in cache by tier',
    ['tier'],
    registry=metrics_registry
)

cache_evictions_total = Counter(
    'goaldiggers_cache_evictions_total',
    'Total cache evictions by tier and reason',
    ['tier', 'reason'],
    registry=metrics_registry
)

cache_promotions_total = Counter(
    'goaldiggers_cache_promotions_total',
    'Total cache promotions between tiers',
    ['from_tier', 'to_tier'],
    registry=metrics_registry
)

cache_stale_served_total = Counter(
    'goaldiggers_cache_stale_served_total',
    'Total stale cache entries served (stale-while-revalidate)',
    ['tier'],
    registry=metrics_registry
)

# ============================================================================
# DATA QUALITY METRICS
# ============================================================================

data_quality_score = Gauge(
    'goaldiggers_data_quality_score',
    'Data quality score for predictions (0.0-1.0)',
    ['data_type'],  # form, h2h, standings, xg, overall
    registry=metrics_registry
)

data_validation_checks = Counter(
    'goaldiggers_data_validation_checks_total',
    'Total data validation checks by result',
    ['check_type', 'result'],  # result: pass, fail, warning
    registry=metrics_registry
)

data_freshness_seconds = Gauge(
    'goaldiggers_data_freshness_seconds',
    'Data age in seconds by data type',
    ['data_type'],
    registry=metrics_registry
)

xg_estimation_usage = Counter(
    'goaldiggers_xg_estimation_usage_total',
    'Total xG estimations (fallback) vs real xG data',
    ['source'],  # real, estimated
    registry=metrics_registry
)

# ============================================================================
# PREDICTION METRICS
# ============================================================================

predictions_total = Counter(
    'goaldiggers_predictions_total',
    'Total predictions generated by model type',
    ['model_type', 'league'],
    registry=metrics_registry
)

prediction_latency_seconds = Histogram(
    'goaldiggers_prediction_latency_seconds',
    'Prediction generation latency in seconds',
    ['model_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=metrics_registry
)

prediction_confidence = Histogram(
    'goaldiggers_prediction_confidence',
    'Prediction confidence score distribution',
    ['model_type'],
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    registry=metrics_registry
)

prediction_errors_total = Counter(
    'goaldiggers_prediction_errors_total',
    'Total prediction errors by error type',
    ['error_type'],
    registry=metrics_registry
)

# ============================================================================
# DATA VALIDATION METRICS
# ============================================================================

validation_quality_score = Gauge(
    'goaldiggers_validation_quality_score',
    'Current data quality score from validator (0.0 to 1.0)',
    ['component'],
    registry=metrics_registry
)

validation_checks_total = Counter(
    'goaldiggers_validation_checks_total',
    'Total validation checks performed',
    ['component', 'result'],
    registry=metrics_registry
)

validation_passed_total = Counter(
    'goaldiggers_validation_passed_total',
    'Total validations that passed quality checks',
    ['component'],
    registry=metrics_registry
)

validation_failed_total = Counter(
    'goaldiggers_validation_failed_total',
    'Total validations that failed quality checks',
    ['component', 'reason'],
    registry=metrics_registry
)

validation_blocked_total = Counter(
    'goaldiggers_validation_blocked_total',
    'Total predictions blocked due to low quality',
    ['component', 'blocking_issue'],
    registry=metrics_registry
)

validation_feature_coverage = Gauge(
    'goaldiggers_validation_feature_coverage',
    'Feature coverage percentage by feature type (0.0 to 1.0)',
    ['feature_type'],
    registry=metrics_registry
)

# ============================================================================
# USER ACTIVITY METRICS
# ============================================================================

active_users = Gauge(
    'goaldiggers_active_users_total',
    'Number of active users/sessions',
    registry=metrics_registry
)

page_views_total = Counter(
    'goaldiggers_page_views_total',
    'Total page views by page type',
    ['page'],
    registry=metrics_registry
)

user_interactions_total = Counter(
    'goaldiggers_user_interactions_total',
    'Total user interactions by action type',
    ['action'],
    registry=metrics_registry
)

# ============================================================================
# DATA PIPELINE METRICS
# ============================================================================

# Note: data_freshness_seconds already defined in DATA QUALITY METRICS section

data_ingestion_total = Counter(
    'goaldiggers_data_ingestion_total',
    'Total data ingestion operations by source',
    ['source', 'status'],
    registry=metrics_registry
)

data_ingestion_duration_seconds = Histogram(
    'goaldiggers_data_ingestion_duration_seconds',
    'Data ingestion duration in seconds',
    ['source'],
    buckets=[1, 5, 10, 30, 60, 120, 300],
    registry=metrics_registry
)

database_query_duration_seconds = Histogram(
    'goaldiggers_database_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    registry=metrics_registry
)

# ============================================================================
# SYSTEM METRICS
# ============================================================================

system_info = Info(
    'goaldiggers_system',
    'GoalDiggers system information',
    registry=metrics_registry
)

system_errors_total = Counter(
    'goaldiggers_system_errors_total',
    'Total system errors by component and severity',
    ['component', 'severity'],
    registry=metrics_registry
)

# ============================================================================
# INSTRUMENTATION HELPERS
# ============================================================================

@contextmanager
def track_api_request(provider: str, endpoint: str):
    """
    Context manager to track API request metrics.
    
    Usage:
        with track_api_request("football-data", "/fixtures"):
            response = await make_api_call()
    """
    start_time = time.time()
    status = "success"
    
    try:
        yield
    except Exception as e:
        status = "error"
        error_type = type(e).__name__
        api_errors_total.labels(provider=provider, error_type=error_type).inc()
        
        if "rate limit" in str(e).lower():
            api_rate_limit_hits_total.labels(provider=provider).inc()
        
        raise
    finally:
        duration = time.time() - start_time
        api_requests_total.labels(
            provider=provider,
            endpoint=endpoint,
            status=status
        ).inc()
        api_request_duration_seconds.labels(
            provider=provider,
            endpoint=endpoint
        ).observe(duration)


def track_cache_operation(tier: str, operation: str, result: str = "success"):
    """
    Track a cache operation.
    
    Args:
        tier: Cache tier (L1, L2, L3)
        operation: Operation type (get, set, delete, clear)
        result: Operation result (hit, miss, success, error)
    """
    cache_operations_total.labels(
        tier=tier,
        operation=operation,
        result=result
    ).inc()


def update_cache_metrics(tier: str, stats: dict):
    """
    Update cache metrics from cache statistics.
    
    Args:
        tier: Cache tier (L1, L2, L3)
        stats: Dictionary with cache statistics
    """
    if 'hit_rate' in stats:
        cache_hit_rate.labels(tier=tier).set(stats['hit_rate'])
    
    if 'entries' in stats:
        cache_entries.labels(tier=tier).set(stats['entries'])


def track_cache_eviction(tier: str, reason: str = "LRU"):
    """Track a cache eviction."""
    cache_evictions_total.labels(tier=tier, reason=reason).inc()


def track_cache_promotion(from_tier: str, to_tier: str):
    """Track a cache promotion between tiers."""
    cache_promotions_total.labels(from_tier=from_tier, to_tier=to_tier).inc()


def track_stale_served(tier: str):
    """Track serving of stale cache entry."""
    cache_stale_served_total.labels(tier=tier).inc()


def track_circuit_breaker_state(provider: str, state: str):
    """
    Track circuit breaker state change.
    
    Args:
        provider: API provider (e.g., 'football-data', 'understat')
        state: Circuit breaker state ('closed', 'half_open', 'open')
    """
    state_map = {'closed': 0, 'half_open': 1, 'open': 2}
    api_circuit_breaker_state.labels(provider=provider).set(state_map.get(state, 0))
    
    if state == 'open':
        api_circuit_breaker_opens.labels(provider=provider).inc()


def track_circuit_breaker_failure(provider: str):
    """Track a circuit breaker failure."""
    api_circuit_breaker_failures.labels(provider=provider).inc()


def track_fallback_usage(provider: str, reason: str = "circuit_open"):
    """
    Track usage of fallback data source.
    
    Args:
        provider: Primary provider that failed
        reason: Reason for fallback (circuit_open, timeout, error)
    """
    api_fallback_usage.labels(provider=provider, reason=reason).inc()


def track_data_quality(data_type: str, quality_score: float):
    """
    Track data quality score.
    
    Args:
        data_type: Type of data (form, h2h, standings, xg, overall)
        quality_score: Quality score from 0.0 to 1.0
    """
    data_quality_score.labels(data_type=data_type).set(quality_score)


def track_data_validation(check_type: str, result: str):
    """
    Track data validation check result.
    
    Args:
        check_type: Type of validation (freshness, completeness, consistency)
        result: Check result (pass, fail, warning)
    """
    data_validation_checks.labels(check_type=check_type, result=result).inc()


def track_data_freshness(data_type: str, age_seconds: float):
    """
    Track data freshness in seconds.
    
    Args:
        data_type: Type of data (fixtures, form, standings, h2h)
        age_seconds: Age of data in seconds
    """
    data_freshness_seconds.labels(data_type=data_type).set(age_seconds)


def track_xg_usage(source: str):
    """
    Track xG data source usage.
    
    Args:
        source: Data source ('real' for Understat, 'estimated' for fallback)
    """
    xg_estimation_usage.labels(source=source).inc()


@contextmanager
def track_prediction(model_type: str, league: str = "unknown"):
    """
    Context manager to track prediction metrics.
    
    Usage:
        with track_prediction("ensemble", "PL"):
            prediction = model.predict(features)
    """
    start_time = time.time()
    
    try:
        yield
        predictions_total.labels(model_type=model_type, league=league).inc()
    except Exception as e:
        error_type = type(e).__name__
        prediction_errors_total.labels(error_type=error_type).inc()
        raise
    finally:
        duration = time.time() - start_time
        prediction_latency_seconds.labels(model_type=model_type).observe(duration)


def record_prediction_confidence(model_type: str, confidence: float):
    """Record prediction confidence score."""
    prediction_confidence.labels(model_type=model_type).observe(confidence)


def track_page_view(page: str):
    """Track a page view."""
    page_views_total.labels(page=page).inc()


def track_user_interaction(action: str):
    """Track a user interaction."""
    user_interactions_total.labels(action=action).inc()


def update_active_users(count: int):
    """Update active users count."""
    active_users.set(count)


def update_data_freshness(data_type: str, age_seconds: float):
    """Update data freshness metric."""
    data_freshness_seconds.labels(data_type=data_type).set(age_seconds)


@contextmanager
def track_data_ingestion(source: str):
    """
    Context manager to track data ingestion.
    
    Usage:
        with track_data_ingestion("football-data"):
            await ingest_fixtures()
    """
    start_time = time.time()
    status = "success"
    
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = time.time() - start_time
        data_ingestion_total.labels(source=source, status=status).inc()
        data_ingestion_duration_seconds.labels(source=source).observe(duration)


@contextmanager
def track_database_query(query_type: str):
    """
    Context manager to track database query performance.
    
    Usage:
        with track_database_query("select_matches"):
            matches = session.query(Match).all()
    """
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        database_query_duration_seconds.labels(query_type=query_type).observe(duration)


def track_system_error(component: str, severity: str = "error"):
    """Track a system error."""
    system_errors_total.labels(component=component, severity=severity).inc()


def set_system_info(version: str, environment: str, python_version: str):
    """Set system information."""
    system_info.info({
        'version': version,
        'environment': environment,
        'python_version': python_version
    })


# ============================================================================
# DATA VALIDATION INSTRUMENTATION
# ============================================================================

def record_validation_quality_score(component: str, quality_score: float):
    """
    Record current data quality score from validator.
    
    Args:
        component: Component name (e.g., 'prediction_engine', 'data_pipeline')
        quality_score: Quality score between 0.0 and 1.0
    """
    validation_quality_score.labels(component=component).set(quality_score)


def record_validation_check(component: str, passed: bool):
    """
    Record a validation check result.
    
    Args:
        component: Component performing validation
        passed: Whether validation passed
    """
    result = 'passed' if passed else 'failed'
    validation_checks_total.labels(component=component, result=result).inc()
    
    if passed:
        validation_passed_total.labels(component=component).inc()


def record_validation_failure(component: str, reason: str):
    """
    Record a validation failure with reason.
    
    Args:
        component: Component where validation failed
        reason: Failure reason (e.g., 'stale_data', 'missing_features', 'low_quality')
    """
    validation_failed_total.labels(component=component, reason=reason).inc()


def record_validation_blocked(component: str, blocking_issue: str):
    """
    Record a prediction blocked due to validation failure.
    
    Args:
        component: Component that blocked prediction
        blocking_issue: Issue that caused blocking (e.g., 'all_sources_unavailable', 'fixture_data_too_old')
    """
    validation_blocked_total.labels(component=component, blocking_issue=blocking_issue).inc()


def update_feature_coverage(feature_type: str, coverage: float):
    """
    Update feature coverage percentage.
    
    Args:
        feature_type: Type of feature (e.g., 'form', 'h2h', 'standings', 'xg')
        coverage: Coverage percentage between 0.0 and 1.0
    """
    validation_feature_coverage.labels(feature_type=feature_type).set(coverage)


@contextmanager
def track_validation(component: str):
    """
    Context manager to track validation operations.
    
    Usage:
        with track_validation("prediction_engine"):
            is_valid, report = validator.validate(features)
            if not is_valid:
                record_validation_failure("prediction_engine", "low_quality")
    """
    try:
        yield
    except Exception as e:
        error_type = type(e).__name__
        validation_failed_total.labels(component=component, reason=error_type).inc()
        raise


# ============================================================================
# DECORATOR FOR FUNCTION INSTRUMENTATION
# ============================================================================

def instrument_function(metric_name: str, labels: Optional[dict] = None):
    """
    Decorator to instrument a function with metrics.
    
    Usage:
        @instrument_function("my_function", {"component": "data_pipeline"})
        def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                logger.debug(f"{metric_name} executed in {duration:.3f}s")
        
        return wrapper
    return decorator


# ============================================================================
# METRICS EXPORT
# ============================================================================

def get_metrics_text() -> bytes:
    """
    Get metrics in Prometheus text format.
    
    Returns:
        Metrics data in Prometheus exposition format
    """
    return generate_latest(metrics_registry)


def get_metrics_dict() -> dict:
    """
    Get current metrics as dictionary for debugging.
    
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    try:
        for collector in metrics_registry._collector_to_names.keys():
            for metric in collector.collect():
                metrics[metric.name] = {
                    'type': metric.type,
                    'documentation': metric.documentation,
                    'samples': [
                        {
                            'labels': dict(sample.labels),
                            'value': sample.value
                        }
                        for sample in metric.samples
                    ]
                }
    except Exception as e:
        logger.error(f"Error collecting metrics: {e}")
    
    return metrics


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_metrics(version: str = "1.0.0", environment: str = "production"):
    """
    Initialize metrics system with system information.
    
    Args:
        version: Application version
        environment: Deployment environment (production, staging, development)
    """
    import sys
    
    set_system_info(
        version=version,
        environment=environment,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    
    logger.info(f"✅ Metrics exporter initialized (version={version}, env={environment})")


__all__ = [
    'metrics_registry',
    'track_api_request',
    'track_cache_operation',
    'update_cache_metrics',
    'track_cache_eviction',
    'track_cache_promotion',
    'track_stale_served',
    'track_prediction',
    'record_prediction_confidence',
    'track_page_view',
    'track_user_interaction',
    'update_active_users',
    'update_data_freshness',
    'track_data_ingestion',
    'track_database_query',
    'track_system_error',
    'set_system_info',
    'instrument_function',
    'get_metrics_text',
    'get_metrics_dict',
    'initialize_metrics',
]


if __name__ == "__main__":
    # Test metrics exporter
    print("✅ Metrics Exporter initialized")
    
    initialize_metrics(version="1.0.0", environment="development")
    
    # Simulate some metrics
    with track_api_request("football-data", "/fixtures"):
        time.sleep(0.1)
    
    track_cache_operation("L1", "get", "hit")
    update_cache_metrics("L1", {"hit_rate": 0.85, "entries": 100})
    
    with track_prediction("ensemble", "PL"):
        time.sleep(0.05)
    
    record_prediction_confidence("ensemble", 0.78)
    track_page_view("homepage")
    update_active_users(42)
    
    # Export metrics
    metrics_text = get_metrics_text().decode('utf-8')
    print("\n📊 Sample metrics:")
    for line in metrics_text.split('\n')[:20]:
        if line and not line.startswith('#'):
            print(f"  {line}")
    
    print(f"\n✅ Metrics exported successfully ({len(metrics_text)} bytes)")
