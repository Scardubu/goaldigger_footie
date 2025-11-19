#!/usr/bin/env python3
"""
Enhanced Cache Manager - Multi-tier caching with L1/L2/L3 strategy
Implements memory → Redis → disk fallback with TTL, stale-while-revalidate, and promotion logic
"""

import asyncio
import json
import logging
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

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheEntry:
    """Wrapper for cached values with metadata."""
    
    def __init__(self, value: Any, ttl: int, created_at: Optional[float] = None):
        """
        Initialize cache entry.
        
        Args:
            value: The cached value
            ttl: Time-to-live in seconds
            created_at: Creation timestamp (defaults to now)
        """
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or time.time()
        self.hits = 0
        self.last_accessed = self.created_at
    
    def is_fresh(self) -> bool:
        """Check if entry is within TTL."""
        return (time.time() - self.created_at) < self.ttl
    
    def is_stale_acceptable(self, stale_ttl: int) -> bool:
        """Check if entry is stale but still usable."""
        age = time.time() - self.created_at
        return age < (self.ttl + stale_ttl)
    
    def access(self):
        """Record access for LRU tracking."""
        self.hits += 1
        self.last_accessed = time.time()
    
    def age(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at


class EnhancedCacheManager:
    """
    Multi-tier cache manager with L1 (memory) → L2 (Redis) → L3 (disk) strategy.
    
    Features:
    - TTL-based expiration with configurable staleness tolerance
    - Stale-while-revalidate for non-blocking updates
    - Automatic promotion from slower tiers to faster tiers
    - LRU eviction within each tier
    - get_or_fetch API for cache-aside pattern
    - Metrics tracking (hit rate, tier distribution)
    """
    
    def __init__(
        self,
        l1_max_size: int = 100,
        l2_enabled: bool = False,
        l2_redis_url: Optional[str] = None,
        l3_cache_dir: Optional[Path] = None,
        default_ttl: int = 3600,
        stale_ttl: int = 7200,
        enable_metrics: bool = True
    ):
        """
        Initialize enhanced cache manager.
        
        Args:
            l1_max_size: Maximum entries in L1 memory cache
            l2_enabled: Whether to use L2 Redis cache
            l2_redis_url: Redis connection URL
            l3_cache_dir: Directory for L3 disk cache
            default_ttl: Default TTL in seconds (fresh data)
            stale_ttl: Additional time to serve stale data
            enable_metrics: Track cache metrics
        """
        # L1: Memory cache (fastest)
        self._l1_cache: Dict[str, CacheEntry] = {}
        self._l1_max_size = l1_max_size
        
        # L2: Redis cache (fast, shared)
        self._l2_enabled = l2_enabled
        self._l2_client = None
        if l2_enabled and l2_redis_url:
            try:
                import redis
                self._l2_client = redis.from_url(l2_redis_url, decode_responses=False)
                logger.info(f"✅ L2 Redis cache connected: {l2_redis_url}")
            except Exception as e:
                logger.warning(f"⚠️ L2 Redis cache unavailable: {e}")
                self._l2_enabled = False
        
        # L3: Disk cache (slow, persistent)
        self._l3_cache_dir = l3_cache_dir or Path("data/cache")
        self._l3_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self._default_ttl = default_ttl
        self._stale_ttl = stale_ttl
        self._enable_metrics = enable_metrics
        
        # Metrics
        self._metrics = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'misses': 0,
            'promotions': 0,
            'evictions': 0,
            'stale_served': 0,
            'background_refreshes': 0
        }
        
        # External metrics collector (optional)
        try:
            from utils.metrics_collector import get_metrics_collector
            self.metrics_collector = get_metrics_collector()
            logger.info("📊 Metrics collector initialized for EnhancedCacheManager")
        except Exception as e:
            logger.debug(f"Metrics collector unavailable: {e}")
            self.metrics_collector = None
        
        logger.info(
            f"🗄️ EnhancedCacheManager initialized: "
            f"L1(memory:{l1_max_size}) | L2(redis:{l2_enabled}) | L3(disk:{self._l3_cache_dir})"
        )
    
    @property
    def redis_available(self) -> bool:
        """Public property to check if Redis (L2 cache) is available."""
        return self._l2_enabled
    
    def _make_cache_key(self, key: str) -> str:
        """Normalize cache key."""
        return key.replace('/', '_').replace(':', '_').replace(' ', '_')
    
    def _evict_lru_from_l1(self):
        """Evict least recently used entry from L1."""
        if not self._l1_cache:
            return
        
        lru_key = min(
            self._l1_cache.keys(),
            key=lambda k: self._l1_cache[k].last_accessed
        )
        
        del self._l1_cache[lru_key]
        self._metrics['evictions'] += 1
        logger.debug(f"Evicted LRU from L1: {lru_key}")
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """
        Get value from cache, checking L1 → L2 → L3.
        
        Args:
            key: Cache key
            default: Default value if not found
        
        Returns:
            Cached value or default
        """
        cache_key = self._make_cache_key(key)
        
        # Check L1 (memory)
        if cache_key in self._l1_cache:
            entry = self._l1_cache[cache_key]
            if entry.is_fresh() or entry.is_stale_acceptable(self._stale_ttl):
                entry.access()
                self._metrics['l1_hits'] += 1
                if not entry.is_fresh():
                    self._metrics['stale_served'] += 1
                
                # Record metrics
                if self.metrics_collector:
                    try:
                        self.metrics_collector.record_cache_operation(
                            operation='get',
                            layer='l1',
                            hit=True,
                            metadata={'stale': not entry.is_fresh(), 'age': entry.age()}
                        )
                    except Exception:
                        pass
                
                return entry.value
            else:
                # Expired, remove
                del self._l1_cache[cache_key]
        
        # Check L2 (Redis)
        if self._l2_enabled and self._l2_client:
            try:
                data = self._l2_client.get(cache_key)
                if data:
                    entry = pickle.loads(data)
                    if isinstance(entry, CacheEntry):
                        if entry.is_fresh() or entry.is_stale_acceptable(self._stale_ttl):
                            # Promote to L1
                            self._promote_to_l1(cache_key, entry)
                            self._metrics['l2_hits'] += 1
                            self._metrics['promotions'] += 1
                            if not entry.is_fresh():
                                self._metrics['stale_served'] += 1
                            
                            # Record metrics
                            if self.metrics_collector:
                                try:
                                    self.metrics_collector.record_cache_operation(
                                        operation='get',
                                        layer='l2',
                                        hit=True,
                                        metadata={'stale': not entry.is_fresh(), 'promoted': True}
                                    )
                                except Exception:
                                    pass
                            
                            return entry.value
            except Exception as e:
                logger.debug(f"L2 cache read error for {cache_key}: {e}")
        
        # Check L3 (disk)
        l3_path = self._l3_cache_dir / f"{cache_key}.cache"
        if l3_path.exists():
            try:
                with open(l3_path, 'rb') as f:
                    entry = pickle.load(f)
                    if isinstance(entry, CacheEntry):
                        if entry.is_fresh() or entry.is_stale_acceptable(self._stale_ttl):
                            # Promote to L2 and L1
                            self._promote_to_l2(cache_key, entry)
                            self._promote_to_l1(cache_key, entry)
                            self._metrics['l3_hits'] += 1
                            self._metrics['promotions'] += 2
                            if not entry.is_fresh():
                                self._metrics['stale_served'] += 1
                            return entry.value
                        else:
                            # Expired, delete
                            l3_path.unlink()
            except Exception as e:
                logger.debug(f"L3 cache read error for {cache_key}: {e}")
        
        # Cache miss
        self._metrics['misses'] += 1
        
        # Record cache miss
        if self.metrics_collector:
            try:
                self.metrics_collector.record_cache_operation(
                    operation='get',
                    layer='all',
                    hit=False,
                    metadata={'key': cache_key}
                )
            except Exception:
                pass
        
        return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        skip_l1: bool = False,
        skip_l2: bool = False,
        skip_l3: bool = False
    ):
        """
        Set value in cache across all tiers.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live (defaults to default_ttl)
            skip_l1: Skip L1 cache
            skip_l2: Skip L2 cache
            skip_l3: Skip L3 cache
        """
        cache_key = self._make_cache_key(key)
        ttl = ttl or self._default_ttl
        entry = CacheEntry(value, ttl)
        
        # Set in L1
        if not skip_l1:
            if len(self._l1_cache) >= self._l1_max_size:
                self._evict_lru_from_l1()
            self._l1_cache[cache_key] = entry
            
            # Record L1 set operation
            if self.metrics_collector:
                try:
                    self.metrics_collector.record_cache_operation(
                        operation='set',
                        layer='l1',
                        hit=True,
                        metadata={'ttl': ttl}
                    )
                except Exception:
                    pass
        
        # Set in L2
        if not skip_l2 and self._l2_enabled and self._l2_client:
            try:
                data = pickle.dumps(entry)
                self._l2_client.setex(cache_key, ttl + self._stale_ttl, data)
                
                # Record L2 set operation
                if self.metrics_collector:
                    try:
                        self.metrics_collector.record_cache_operation(
                            operation='set',
                            layer='l2',
                            hit=True,
                            metadata={'ttl': ttl}
                        )
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"L2 cache write error for {cache_key}: {e}")
        
        # Set in L3
        if not skip_l3:
            try:
                l3_path = self._l3_cache_dir / f"{cache_key}.cache"
                with open(l3_path, 'wb') as f:
                    pickle.dump(entry, f)
            except Exception as e:
                logger.debug(f"L3 cache write error for {cache_key}: {e}")
    
    def _promote_to_l1(self, cache_key: str, entry: CacheEntry):
        """Promote entry from L2/L3 to L1."""
        if len(self._l1_cache) >= self._l1_max_size:
            self._evict_lru_from_l1()
        self._l1_cache[cache_key] = entry
    
    def _promote_to_l2(self, cache_key: str, entry: CacheEntry):
        """Promote entry from L3 to L2."""
        if self._l2_enabled and self._l2_client:
            try:
                data = pickle.dumps(entry)
                self._l2_client.setex(
                    cache_key,
                    entry.ttl + self._stale_ttl,
                    data
                )
            except Exception as e:
                logger.debug(f"L2 promotion error for {cache_key}: {e}")
    
    def get_or_fetch(
        self,
        key: str,
        fetch_func: Callable[[], T],
        ttl: Optional[int] = None,
        stale_while_revalidate: bool = True
    ) -> T:
        """
        Get from cache or fetch if missing (cache-aside pattern).
        
        Args:
            key: Cache key
            fetch_func: Function to call if cache miss
            ttl: Time-to-live for cached value
            stale_while_revalidate: Serve stale and refresh in background
        
        Returns:
            Cached or freshly fetched value
        """
        cache_key = self._make_cache_key(key)
        
        # Try cache first
        cached = self.get(cache_key)
        
        if cached is not None:
            # Check if stale and needs refresh
            if cache_key in self._l1_cache:
                entry = self._l1_cache[cache_key]
                if not entry.is_fresh() and stale_while_revalidate:
                    # Serve stale but refresh in background
                    logger.debug(f"Serving stale data for {key}, refreshing in background")
                    asyncio.create_task(self._background_refresh(key, fetch_func, ttl))
                    self._metrics['background_refreshes'] += 1
            
            return cached
        
        # Cache miss - fetch fresh data
        try:
            fresh_value = fetch_func()
            self.set(key, fresh_value, ttl)
            return fresh_value
        except Exception as e:
            logger.error(f"Fetch function failed for {key}: {e}")
            raise
    
    async def _background_refresh(
        self,
        key: str,
        fetch_func: Callable[[], T],
        ttl: Optional[int] = None
    ):
        """Background task to refresh stale cache entry."""
        try:
            fresh_value = fetch_func()
            self.set(key, fresh_value, ttl)
            logger.debug(f"Background refresh completed for {key}")
        except Exception as e:
            logger.warning(f"Background refresh failed for {key}: {e}")
    
    def delete(self, key: str):
        """Delete key from all cache tiers."""
        cache_key = self._make_cache_key(key)
        
        # Delete from L1
        self._l1_cache.pop(cache_key, None)
        
        # Delete from L2
        if self._l2_enabled and self._l2_client:
            try:
                self._l2_client.delete(cache_key)
            except Exception as e:
                logger.debug(f"L2 delete error for {cache_key}: {e}")
        
        # Delete from L3
        l3_path = self._l3_cache_dir / f"{cache_key}.cache"
        if l3_path.exists():
            try:
                l3_path.unlink()
            except Exception as e:
                logger.debug(f"L3 delete error for {cache_key}: {e}")
    
    def clear(self, pattern: Optional[str] = None):
        """
        Clear cache entries matching pattern.
        
        Args:
            pattern: Optional pattern to match keys (substring match)
        """
        if pattern:
            # Clear matching keys
            keys_to_delete = [
                k for k in self._l1_cache.keys()
                if pattern in k
            ]
            for key in keys_to_delete:
                self.delete(key)
        else:
            # Clear all
            self._l1_cache.clear()
            
            if self._l2_enabled and self._l2_client:
                try:
                    self._l2_client.flushdb()
                except Exception as e:
                    logger.debug(f"L2 flush error: {e}")
            
            for cache_file in self._l3_cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.debug(f"L3 file delete error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.
        
        Returns:
            Dictionary with hit rates, tier distribution, etc.
        """
        total_requests = (
            self._metrics['l1_hits'] +
            self._metrics['l2_hits'] +
            self._metrics['l3_hits'] +
            self._metrics['misses']
        )
        
        if total_requests == 0:
            return {**self._metrics, 'hit_rate': 0.0, 'l1_size': 0}
        
        total_hits = (
            self._metrics['l1_hits'] +
            self._metrics['l2_hits'] +
            self._metrics['l3_hits']
        )
        
        return {
            **self._metrics,
            'hit_rate': total_hits / total_requests,
            'l1_hit_rate': self._metrics['l1_hits'] / total_requests,
            'l2_hit_rate': self._metrics['l2_hits'] / total_requests,
            'l3_hit_rate': self._metrics['l3_hits'] / total_requests,
            'miss_rate': self._metrics['misses'] / total_requests,
            'l1_size': len(self._l1_cache),
            'l1_utilization': len(self._l1_cache) / self._l1_max_size,
            'total_requests': total_requests
        }
    
    def print_metrics(self):
        """Print formatted cache metrics."""
        metrics = self.get_metrics()
        
        logger.info("📊 Cache Performance Metrics:")
        logger.info(f"  Overall Hit Rate: {metrics['hit_rate']:.1%}")
        logger.info(f"  L1 Memory Hits: {metrics['l1_hits']} ({metrics['l1_hit_rate']:.1%})")
        logger.info(f"  L2 Redis Hits: {metrics['l2_hits']} ({metrics['l2_hit_rate']:.1%})")
        logger.info(f"  L3 Disk Hits: {metrics['l3_hits']} ({metrics['l3_hit_rate']:.1%})")
        logger.info(f"  Cache Misses: {metrics['misses']} ({metrics['miss_rate']:.1%})")
        logger.info(f"  L1 Utilization: {metrics['l1_size']}/{self._l1_max_size} ({metrics['l1_utilization']:.1%})")
        logger.info(f"  Promotions: {metrics['promotions']}")
        logger.info(f"  Stale Served: {metrics['stale_served']}")
        logger.info(f"  Background Refreshes: {metrics['background_refreshes']}")


# Global cache manager instance
_global_cache_manager: Optional[EnhancedCacheManager] = None


def get_cache_manager(
    l1_max_size: int = 100,
    l2_enabled: bool = False,
    l2_redis_url: Optional[str] = None,
    l3_cache_dir: Optional[Path] = None,
    **kwargs
) -> EnhancedCacheManager:
    """
    Get or create global cache manager instance.
    
    Args:
        l1_max_size: Max L1 entries
        l2_enabled: Enable L2 Redis
        l2_redis_url: Redis URL
        l3_cache_dir: L3 disk directory
        **kwargs: Additional args for EnhancedCacheManager
    
    Returns:
        Global EnhancedCacheManager instance
    """
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = EnhancedCacheManager(
            l1_max_size=l1_max_size,
            l2_enabled=l2_enabled,
            l2_redis_url=l2_redis_url,
            l3_cache_dir=l3_cache_dir,
            **kwargs
        )
    
    return _global_cache_manager


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize cache manager
    cache = EnhancedCacheManager(
        l1_max_size=10,
        l2_enabled=False,  # Disable Redis for example
        default_ttl=60,
        stale_ttl=120
    )
    
    # Test cache operations
    print("\n=== Testing Cache Operations ===")
    
    # Set and get
    cache.set("test_key", {"data": "example"}, ttl=5)
    result = cache.get("test_key")
    print(f"✅ Set and retrieved: {result}")
    
    # get_or_fetch pattern
    def fetch_data():
        print("  📡 Fetching fresh data...")
        return {"fetched": True, "timestamp": time.time()}
    
    data1 = cache.get_or_fetch("fetch_test", fetch_data, ttl=10)
    print(f"✅ First fetch: {data1}")
    
    data2 = cache.get_or_fetch("fetch_test", fetch_data, ttl=10)
    print(f"✅ Cached fetch: {data2}")
    
    # Metrics
    print("\n=== Cache Metrics ===")
    cache.print_metrics()
