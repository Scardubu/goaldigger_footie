#!/usr/bin/env python3
"""
Rate Limiter for GoalDiggers Platform

Provides rate limiting functionality for API endpoints and user actions.
Uses Redis for distributed rate limiting when available, falls back to in-memory.

Usage:
    from utils.rate_limiter import rate_limit, check_rate_limit
    
    @rate_limit(max_requests=100, window_seconds=60)
    def my_api_endpoint():
        return "data"
    
    # Or manual check:
    if not check_rate_limit(key="user:123", max_requests=100, window_seconds=60):
        raise RateLimitExceeded("Too many requests")
"""

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import Redis, fallback to in-memory
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Using in-memory rate limiting.")


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class InMemoryRateLimiter:
    """In-memory rate limiter using sliding window algorithm."""
    
    def __init__(self):
        """Initialize in-memory rate limiter."""
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self.last_cleanup = time.time()
    
    def check_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is within rate limit.
        
        Args:
            key: Unique identifier (e.g., user ID, IP address)
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if within limit, False if exceeded
        """
        with self.lock:
            now = time.time()
            
            # Cleanup old entries periodically
            if now - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_entries(now)
                self.last_cleanup = now
            
            # Get request timestamps for this key
            timestamps = self.requests[key]
            
            # Remove timestamps outside the window
            cutoff = now - window_seconds
            timestamps[:] = [ts for ts in timestamps if ts > cutoff]
            
            # Check if under limit
            if len(timestamps) < max_requests:
                timestamps.append(now)
                return True
            
            return False
    
    def _cleanup_old_entries(self, now: float):
        """Remove old entries to prevent memory growth."""
        # Remove keys that haven't been accessed in 1 hour
        cutoff = now - 3600
        keys_to_remove = [
            key for key, timestamps in self.requests.items()
            if not timestamps or max(timestamps) < cutoff
        ]
        for key in keys_to_remove:
            del self.requests[key]
    
    def get_remaining(self, key: str, max_requests: int, window_seconds: int) -> int:
        """Get remaining requests for a key.
        
        Args:
            key: Unique identifier
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Number of remaining requests
        """
        with self.lock:
            now = time.time()
            timestamps = self.requests.get(key, [])
            
            # Remove timestamps outside the window
            cutoff = now - window_seconds
            valid_timestamps = [ts for ts in timestamps if ts > cutoff]
            
            return max(0, max_requests - len(valid_timestamps))
    
    def reset(self, key: str):
        """Reset rate limit for a key.
        
        Args:
            key: Unique identifier to reset
        """
        with self.lock:
            if key in self.requests:
                del self.requests[key]


class RedisRateLimiter:
    """Redis-based rate limiter using sliding window algorithm."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize Redis rate limiter.
        
        Args:
            redis_url: Redis connection URL
        """
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis rate limiter initialized")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def check_rate_limit(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is within rate limit using Redis.
        
        Args:
            key: Unique identifier (e.g., user ID, IP address)
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            True if within limit, False if exceeded
        """
        try:
            now = time.time()
            redis_key = f"ratelimit:{key}"
            
            # Use Redis sorted set with timestamps as scores
            pipe = self.redis_client.pipeline()
            
            # Remove old entries outside the window
            cutoff = now - window_seconds
            pipe.zremrangebyscore(redis_key, 0, cutoff)
            
            # Count entries in window
            pipe.zcard(redis_key)
            
            # Add current timestamp
            pipe.zadd(redis_key, {str(now): now})
            
            # Set expiry on the key
            pipe.expire(redis_key, window_seconds + 60)
            
            results = pipe.execute()
            count = results[1]  # Result from zcard
            
            return count < max_requests
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fail open - allow request if Redis is down
            return True
    
    def get_remaining(self, key: str, max_requests: int, window_seconds: int) -> int:
        """Get remaining requests for a key.
        
        Args:
            key: Unique identifier
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
            
        Returns:
            Number of remaining requests
        """
        try:
            now = time.time()
            redis_key = f"ratelimit:{key}"
            
            # Remove old entries and count
            pipe = self.redis_client.pipeline()
            cutoff = now - window_seconds
            pipe.zremrangebyscore(redis_key, 0, cutoff)
            pipe.zcard(redis_key)
            results = pipe.execute()
            
            count = results[1]
            return max(0, max_requests - count)
            
        except Exception as e:
            logger.error(f"Failed to get remaining requests: {e}")
            return max_requests
    
    def reset(self, key: str):
        """Reset rate limit for a key.
        
        Args:
            key: Unique identifier to reset
        """
        try:
            redis_key = f"ratelimit:{key}"
            self.redis_client.delete(redis_key)
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")


# Global rate limiter instance
_rate_limiter: Optional[Any] = None


def get_rate_limiter():
    """Get or create rate limiter instance.
    
    Returns:
        Rate limiter instance (Redis or in-memory)
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        # Try Redis first if available
        if REDIS_AVAILABLE:
            try:
                import os
                redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
                redis_enabled = os.environ.get("REDIS_ENABLED", "false").lower() == "true"
                
                if redis_enabled:
                    _rate_limiter = RedisRateLimiter(redis_url)
                    logger.info("Using Redis rate limiter")
                else:
                    _rate_limiter = InMemoryRateLimiter()
                    logger.info("Using in-memory rate limiter (Redis disabled)")
            except Exception as e:
                logger.warning(f"Redis rate limiter failed, using in-memory: {e}")
                _rate_limiter = InMemoryRateLimiter()
        else:
            _rate_limiter = InMemoryRateLimiter()
            logger.info("Using in-memory rate limiter")
    
    return _rate_limiter


def check_rate_limit(key: str, max_requests: int = 100, window_seconds: int = 60) -> bool:
    """Check if request is within rate limit.
    
    Args:
        key: Unique identifier (e.g., user:123, ip:192.168.1.1)
        max_requests: Maximum number of requests allowed (default: 100)
        window_seconds: Time window in seconds (default: 60)
        
    Returns:
        True if within limit, False if exceeded
    """
    limiter = get_rate_limiter()
    return limiter.check_rate_limit(key, max_requests, window_seconds)


def get_remaining_requests(key: str, max_requests: int = 100, window_seconds: int = 60) -> int:
    """Get remaining requests for a key.
    
    Args:
        key: Unique identifier
        max_requests: Maximum number of requests allowed
        window_seconds: Time window in seconds
        
    Returns:
        Number of remaining requests
    """
    limiter = get_rate_limiter()
    return limiter.get_remaining(key, max_requests, window_seconds)


def reset_rate_limit(key: str):
    """Reset rate limit for a key.
    
    Args:
        key: Unique identifier to reset
    """
    limiter = get_rate_limiter()
    limiter.reset(key)


def rate_limit(max_requests: int = 100, window_seconds: int = 60, key_func=None):
    """Decorator for rate limiting functions.
    
    Args:
        max_requests: Maximum number of requests allowed
        window_seconds: Time window in seconds
        key_func: Optional function to generate rate limit key from function args
                 If None, uses function name as key
    
    Usage:
        @rate_limit(max_requests=10, window_seconds=60)
        def my_function(user_id):
            return "data"
        
        # Custom key function
        @rate_limit(max_requests=10, window_seconds=60, 
                   key_func=lambda user_id: f"user:{user_id}")
        def my_function(user_id):
            return "data"
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"func:{func.__name__}"
            
            # Check rate limit
            if not check_rate_limit(key, max_requests, window_seconds):
                remaining = get_remaining_requests(key, max_requests, window_seconds)
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {key}. "
                    f"Max {max_requests} requests per {window_seconds}s. "
                    f"Remaining: {remaining}"
                )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Predefined rate limiters for common use cases
def rate_limit_api(max_requests: int = 100, window_seconds: int = 60):
    """Rate limiter for API endpoints.
    
    Args:
        max_requests: Maximum requests per window (default: 100/min)
        window_seconds: Time window in seconds (default: 60s)
    """
    return rate_limit(max_requests, window_seconds)


def rate_limit_user(max_requests: int = 50, window_seconds: int = 60):
    """Rate limiter for user actions.
    
    Args:
        max_requests: Maximum requests per window (default: 50/min)
        window_seconds: Time window in seconds (default: 60s)
    """
    return rate_limit(max_requests, window_seconds)


def rate_limit_ip(max_requests: int = 200, window_seconds: int = 60):
    """Rate limiter for IP addresses.
    
    Args:
        max_requests: Maximum requests per window (default: 200/min)
        window_seconds: Time window in seconds (default: 60s)
    """
    return rate_limit(max_requests, window_seconds)


if __name__ == "__main__":
    # Test rate limiter
    print("Testing rate limiter...")
    
    # Test with 5 requests per 10 seconds
    key = "test:user:123"
    max_req = 5
    window = 10
    
    print(f"\nTesting with {max_req} requests per {window}s window")
    
    for i in range(7):
        allowed = check_rate_limit(key, max_req, window)
        remaining = get_remaining_requests(key, max_req, window)
        
        if allowed:
            print(f"Request {i+1}: ✅ Allowed (Remaining: {remaining})")
        else:
            print(f"Request {i+1}: ❌ Rate limit exceeded (Remaining: {remaining})")
        
        time.sleep(0.5)
    
    print("\nResetting rate limit...")
    reset_rate_limit(key)
    
    allowed = check_rate_limit(key, max_req, window)
    remaining = get_remaining_requests(key, max_req, window)
    print(f"After reset: {'✅ Allowed' if allowed else '❌ Blocked'} (Remaining: {remaining})")
    
    print("\nRate limiter test complete!")
