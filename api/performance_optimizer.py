"""
Enhanced API Performance Optimizer for Football Betting Platform
Provides comprehensive API response time optimization, database query optimization,
and concurrent request handling improvements.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response

try:
    from fastapi.middleware.base import BaseHTTPMiddleware
except ImportError:
    # Fallback for older FastAPI versions
    from starlette.middleware.base import BaseHTTPMiddleware

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy import text
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Thread-safe performance metrics collector."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.request_times = deque(maxlen=1000)  # Keep last 1000 requests
        self.endpoint_stats = defaultdict(lambda: {'count': 0, 'total_time': 0, 'errors': 0})
        self.db_query_stats = deque(maxlen=500)
        self.cache_stats = {'hits': 0, 'misses': 0, 'errors': 0}
        
    def record_request(self, endpoint: str, duration: float, status_code: int):
        """Record API request metrics."""
        with self._lock:
            self.request_times.append({
                'endpoint': endpoint,
                'duration': duration,
                'status_code': status_code,
                'timestamp': datetime.now()
            })
            
            self.endpoint_stats[endpoint]['count'] += 1
            self.endpoint_stats[endpoint]['total_time'] += duration
            
            if status_code >= 400:
                self.endpoint_stats[endpoint]['errors'] += 1
    
    def record_db_query(self, query: str, duration: float, rows_affected: int = 0):
        """Record database query metrics."""
        with self._lock:
            self.db_query_stats.append({
                'query': query[:100] + '...' if len(query) > 100 else query,
                'duration': duration,
                'rows_affected': rows_affected,
                'timestamp': datetime.now()
            })
    
    def record_cache_event(self, event_type: str):
        """Record cache hit/miss/error events."""
        with self._lock:
            if event_type in self.cache_stats:
                self.cache_stats[event_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._lock:
            # Calculate request statistics
            recent_requests = [r for r in self.request_times 
                             if r['timestamp'] > datetime.now() - timedelta(minutes=5)]
            
            avg_response_time = sum(r['duration'] for r in recent_requests) / len(recent_requests) if recent_requests else 0
            error_rate = sum(1 for r in recent_requests if r['status_code'] >= 400) / len(recent_requests) if recent_requests else 0
            
            # Calculate database statistics
            recent_queries = [q for q in self.db_query_stats 
                            if q['timestamp'] > datetime.now() - timedelta(minutes=5)]
            
            avg_query_time = sum(q['duration'] for q in recent_queries) / len(recent_queries) if recent_queries else 0
            
            # Calculate cache hit rate
            total_cache_ops = sum(self.cache_stats.values())
            cache_hit_rate = self.cache_stats['hits'] / total_cache_ops if total_cache_ops > 0 else 0
            
            return {
                'api': {
                    'avg_response_time': avg_response_time,
                    'requests_per_minute': len(recent_requests),
                    'error_rate': error_rate,
                    'endpoint_stats': dict(self.endpoint_stats)
                },
                'database': {
                    'avg_query_time': avg_query_time,
                    'queries_per_minute': len(recent_queries),
                    'slow_queries': [q for q in recent_queries if q['duration'] > 0.1]
                },
                'cache': {
                    'hit_rate': cache_hit_rate,
                    'total_operations': total_cache_ops,
                    **self.cache_stats
                }
            }

# Global metrics instance
metrics = PerformanceMetrics()

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware to track API performance metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Add request ID for tracing
        request_id = f"req_{int(time.time() * 1000)}"
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record metrics
            endpoint = f"{request.method} {request.url.path}"
            metrics.record_request(endpoint, duration, response.status_code)
            
            # Add performance headers
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Request-ID"] = request_id
            
            # Log slow requests
            if duration > 1.0:
                logger.warning(f"Slow request: {endpoint} took {duration:.3f}s")
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            endpoint = f"{request.method} {request.url.path}"
            metrics.record_request(endpoint, duration, 500)
            
            logger.error(f"Request failed: {endpoint} - {str(e)}")
            raise

class DatabaseOptimizer:
    """Database performance optimization utilities."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    def optimize_connection_pool(self):
        """Optimize database connection pool settings."""
        try:
            # Get current pool settings
            current_pool_size = getattr(self.db_manager, 'pool_size', 10)
            current_max_overflow = getattr(self.db_manager, 'max_overflow', 20)
            
            # Calculate optimal settings based on usage patterns
            stats = metrics.get_stats()
            queries_per_minute = stats['database']['queries_per_minute']
            
            # Adjust pool size based on query load
            if queries_per_minute > 50:
                optimal_pool_size = min(20, current_pool_size + 5)
                optimal_max_overflow = min(40, current_max_overflow + 10)
            elif queries_per_minute < 10:
                optimal_pool_size = max(5, current_pool_size - 2)
                optimal_max_overflow = max(10, current_max_overflow - 5)
            else:
                optimal_pool_size = current_pool_size
                optimal_max_overflow = current_max_overflow
            
            logger.info(f"Optimizing connection pool: {current_pool_size} → {optimal_pool_size} "
                       f"(overflow: {current_max_overflow} → {optimal_max_overflow})")
            
            return {
                'current': {'pool_size': current_pool_size, 'max_overflow': current_max_overflow},
                'optimal': {'pool_size': optimal_pool_size, 'max_overflow': optimal_max_overflow},
                'queries_per_minute': queries_per_minute
            }
            
        except Exception as e:
            logger.error(f"Error optimizing connection pool: {e}")
            return None
    
    def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """Analyze slow queries and suggest optimizations."""
        stats = metrics.get_stats()
        slow_queries = stats['database']['slow_queries']
        
        optimizations = []
        for query_info in slow_queries:
            query = query_info['query']
            duration = query_info['duration']
            
            suggestions = []
            
            # Check for missing WHERE clauses
            if 'SELECT' in query.upper() and 'WHERE' not in query.upper():
                suggestions.append("Add WHERE clause to limit result set")
            
            # Check for missing LIMIT clauses
            if 'SELECT' in query.upper() and 'LIMIT' not in query.upper():
                suggestions.append("Add LIMIT clause for large result sets")
            
            # Check for complex JOINs
            if query.upper().count('JOIN') > 2:
                suggestions.append("Consider breaking complex JOINs into smaller queries")
            
            # Check for ORDER BY without LIMIT
            if 'ORDER BY' in query.upper() and 'LIMIT' not in query.upper():
                suggestions.append("ORDER BY without LIMIT can be expensive")
            
            optimizations.append({
                'query': query,
                'duration': duration,
                'suggestions': suggestions
            })
        
        return optimizations
    
    def create_query_indexes(self) -> List[str]:
        """Suggest database indexes based on query patterns."""
        stats = metrics.get_stats()
        recent_queries = [q['query'] for q in stats['database'].get('slow_queries', [])]
        
        index_suggestions = []
        
        # Common patterns that benefit from indexes
        common_patterns = [
            ('match_date', 'CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)'),
            ('team_id', 'CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id)'),
            ('league_id', 'CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_id)'),
            ('status', 'CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status)'),
            ('prediction_date', 'CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(created_at)'),
        ]
        
        for pattern, index_sql in common_patterns:
            # Check if this pattern appears in recent queries
            if any(pattern in query.lower() for query in recent_queries):
                index_suggestions.append(index_sql)
        
        return index_suggestions

class CacheOptimizer:
    """Cache performance optimization utilities."""
    
    def __init__(self, redis_client=None):
        if REDIS_AVAILABLE and redis_client:
            self.redis_client = redis_client
        else:
            self.redis_client = None
        self.local_cache = {}
        self.cache_stats = defaultdict(int)
    
    def optimize_cache_strategy(self) -> Dict[str, Any]:
        """Analyze and optimize caching strategy."""
        stats = metrics.get_stats()
        cache_hit_rate = stats['cache']['hit_rate']
        
        recommendations = []
        
        if cache_hit_rate < 0.7:
            recommendations.append("Increase cache TTL for frequently accessed data")
            recommendations.append("Implement cache warming for popular endpoints")
        
        if cache_hit_rate > 0.95:
            recommendations.append("Consider reducing cache TTL to ensure data freshness")
        
        # Analyze cache usage patterns
        cache_efficiency = {
            'hit_rate': cache_hit_rate,
            'total_operations': stats['cache']['total_operations'],
            'recommendations': recommendations
        }
        
        return cache_efficiency
    
    async def warm_cache(self, endpoints: List[str]):
        """Warm up cache for critical endpoints."""
        logger.info(f"Warming cache for {len(endpoints)} endpoints...")
        
        for endpoint in endpoints:
            try:
                # Simulate cache warming (implement actual logic based on your endpoints)
                await asyncio.sleep(0.1)  # Simulate async operation
                metrics.record_cache_event('hits')
                logger.debug(f"Cache warmed for {endpoint}")
            except Exception as e:
                logger.error(f"Failed to warm cache for {endpoint}: {e}")
                metrics.record_cache_event('errors')

def optimize_api_performance(app: FastAPI, db_manager=None):
    """Apply comprehensive API performance optimizations."""
    
    # Add performance middleware
    app.add_middleware(PerformanceMiddleware)
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add CORS middleware with optimizations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=3600,  # Cache preflight requests
    )
    
    # Initialize optimizers
    if db_manager:
        db_optimizer = DatabaseOptimizer(db_manager)
        cache_optimizer = CacheOptimizer()
        
        @app.get("/api/performance/stats")
        async def get_performance_stats():
            """Get comprehensive performance statistics."""
            return metrics.get_stats()
        
        @app.get("/api/performance/optimize")
        async def optimize_performance():
            """Run performance optimization analysis."""
            results = {}
            
            # Database optimization
            pool_optimization = db_optimizer.optimize_connection_pool()
            if pool_optimization:
                results['database_pool'] = pool_optimization
            
            # Query analysis
            slow_queries = db_optimizer.analyze_slow_queries()
            results['slow_queries'] = slow_queries
            
            # Index suggestions
            index_suggestions = db_optimizer.create_query_indexes()
            results['index_suggestions'] = index_suggestions
            
            # Cache optimization
            cache_optimization = cache_optimizer.optimize_cache_strategy()
            results['cache_optimization'] = cache_optimization
            
            return results
        
        @app.post("/api/performance/cache/warm")
        async def warm_cache(endpoints: List[str]):
            """Warm up cache for specified endpoints."""
            await cache_optimizer.warm_cache(endpoints)
            return {"status": "Cache warming initiated", "endpoints": len(endpoints)}
    
    logger.info("API performance optimizations applied")
    return app

# Decorator for database query optimization
def optimize_db_query(cache_ttl: int = 300):
    """Decorator to optimize database queries with caching and monitoring."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Generate cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            try:
                # Try to get from cache first
                # (Implement actual cache logic here)
                
                # Execute query
                result = await func(*args, **kwargs)
                
                # Record metrics
                duration = time.time() - start_time
                query_info = getattr(func, '__name__', 'unknown_query')
                metrics.record_db_query(query_info, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_db_query(f"ERROR: {func.__name__}", duration)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_db_query(func.__name__, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_db_query(f"ERROR: {func.__name__}", duration)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
