"""
Enhanced Proxy Manager with ML-based patterns, CAPTCHA solving, and advanced anti-detection.
"""
import asyncio
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ProxyMetrics:
    """Metrics for proxy performance tracking."""
    success_count: int = 0
    failure_count: int = 0
    avg_response_time: float = 0.0
    last_used: Optional[datetime] = None
    consecutive_failures: int = 0
    reliability_score: float = 1.0
    geographic_location: Optional[str] = None
    provider: Optional[str] = None

class MLPatternGenerator:
    """ML-based request pattern generator for anti-detection."""
    
    def __init__(self):
        self.request_history = []
        self.pattern_cache = {}
        
    def generate_delay_pattern(self, base_delay: float = 1.0) -> float:
        """Generate human-like delay patterns using ML algorithms."""
        # Simulate human behavior with gamma distribution
        shape = 2.0  # Shape parameter for gamma distribution
        scale = base_delay / shape
        
        # Generate delay with some randomness
        delay = np.random.gamma(shape, scale)
        
        # Add occasional longer pauses (human-like behavior)
        if random.random() < 0.1:  # 10% chance of longer pause
            delay += np.random.exponential(base_delay * 2)
        
        # Ensure minimum delay
        return max(0.5, min(delay, base_delay * 5))
    
    def generate_user_agent_rotation(self) -> str:
        """Generate realistic user agent rotation."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0"
        ]
        return random.choice(user_agents)
    
    def generate_headers(self) -> Dict[str, str]:
        """Generate realistic HTTP headers."""
        return {
            'User-Agent': self.generate_user_agent_rotation(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': random.choice(['en-US,en;q=0.5', 'en-GB,en;q=0.9', 'en-US,en;q=0.9']),
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }

class CAPTCHASolver:
    """CAPTCHA solving integration for anti-bot measures."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.solver_url = "https://2captcha.com/in.php"
        self.result_url = "https://2captcha.com/res.php"
        
    async def solve_captcha(self, captcha_image_url: str, captcha_type: str = "image") -> Optional[str]:
        """Solve CAPTCHA using 2captcha service."""
        if not self.api_key:
            logger.warning("CAPTCHA API key not configured")
            return None
            
        try:
            # Submit CAPTCHA for solving
            submit_data = {
                'key': self.api_key,
                'method': 'base64',
                'body': captcha_image_url,
                'json': 1
            }
            
            # Explicit timeout to avoid hanging CAPTCHA requests
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                # Submit CAPTCHA
                async with session.post(self.solver_url, data=submit_data) as response:
                    result = await response.json()
                    
                if result.get('status') != 1:
                    logger.error(f"CAPTCHA submission failed: {result}")
                    return None
                
                captcha_id = result.get('request')
                
                # Poll for result
                for _ in range(30):  # Wait up to 5 minutes
                    await asyncio.sleep(10)
                    
                    result_data = {
                        'key': self.api_key,
                        'action': 'get',
                        'id': captcha_id,
                        'json': 1
                    }
                    
                    async with session.get(self.result_url, params=result_data) as response:
                        result = await response.json()
                        
                    if result.get('status') == 1:
                        return result.get('request')
                    elif result.get('error_text') == 'CAPCHA_NOT_READY':
                        continue
                    else:
                        logger.error(f"CAPTCHA solving failed: {result}")
                        return None
                        
        except Exception as e:
            logger.error(f"CAPTCHA solving error: {e}")
            return None

class EnhancedProxyManager:
    """Enhanced proxy manager with ML patterns, CAPTCHA solving, and advanced metrics."""
    
    def __init__(self, 
                 proxy_sources: Optional[List[str]] = None,
                 enable_ml_patterns: bool = True,
                 enable_captcha_solving: bool = False,
                 captcha_api_key: Optional[str] = None):
        """
        Initialize enhanced proxy manager.
        
        Args:
            proxy_sources: List of proxy sources/providers
            enable_ml_patterns: Enable ML-based request patterns
            enable_captcha_solving: Enable CAPTCHA solving capabilities
            captcha_api_key: API key for CAPTCHA solving service
        """
        self.proxy_pool = deque()
        self.proxy_metrics: Dict[str, ProxyMetrics] = {}
        self.failed_proxies: Dict[str, datetime] = {}
        self.request_history = []
        
        # Enhanced features
        self.ml_pattern_generator = MLPatternGenerator() if enable_ml_patterns else None
        self.captcha_solver = CAPTCHASolver(captcha_api_key) if enable_captcha_solving else None
        
        # Configuration
        self.max_failures = 3
        self.failure_timeout = timedelta(minutes=10)
        self.reliability_threshold = 0.3
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        
        # Initialize proxy sources
        if proxy_sources:
            asyncio.create_task(self._load_proxy_sources(proxy_sources))
    
    async def _load_proxy_sources(self, sources: List[str]):
        """Load proxies from multiple sources."""
        for source in sources:
            try:
                proxies = await self._fetch_proxies_from_source(source)
                for proxy in proxies:
                    self._add_proxy(proxy)
                logger.info(f"Loaded {len(proxies)} proxies from {source}")
            except Exception as e:
                logger.error(f"Failed to load proxies from {source}: {e}")
    
    async def _fetch_proxies_from_source(self, source: str) -> List[str]:
        """Fetch proxies from a specific source."""
        # Implement proxy fetching from various sources
        # This is a placeholder - implement actual proxy fetching logic
        return []
    
    def _add_proxy(self, proxy: str):
        """Add proxy to the pool with initial metrics."""
        if proxy not in self.proxy_metrics:
            self.proxy_pool.append(proxy)
            self.proxy_metrics[proxy] = ProxyMetrics()
    
    async def get_optimal_proxy(self) -> Optional[str]:
        """Get the most optimal proxy based on performance metrics."""
        if not self.proxy_pool:
            return None
        
        # Filter out failed proxies
        available_proxies = [
            proxy for proxy in self.proxy_pool
            if proxy not in self.failed_proxies or 
            datetime.now() - self.failed_proxies[proxy] > self.failure_timeout
        ]
        
        if not available_proxies:
            return None
        
        # Score proxies based on reliability and performance
        scored_proxies = []
        for proxy in available_proxies:
            metrics = self.proxy_metrics[proxy]
            
            # Calculate composite score
            reliability_score = metrics.reliability_score
            speed_score = 1.0 / (metrics.avg_response_time + 0.1)  # Avoid division by zero
            recency_score = 1.0 if not metrics.last_used else \
                           1.0 / ((datetime.now() - metrics.last_used).total_seconds() / 3600 + 1)
            
            composite_score = (reliability_score * 0.5 + 
                             speed_score * 0.3 + 
                             recency_score * 0.2)
            
            scored_proxies.append((proxy, composite_score))
        
        # Sort by score and add some randomness to avoid patterns
        scored_proxies.sort(key=lambda x: x[1], reverse=True)
        
        # Select from top 30% with weighted randomness
        top_count = max(1, len(scored_proxies) // 3)
        top_proxies = scored_proxies[:top_count]
        
        # Weighted random selection
        weights = [score for _, score in top_proxies]
        selected_proxy = random.choices(top_proxies, weights=weights)[0][0]
        
        return selected_proxy
    
    async def record_proxy_performance(self, proxy: str, success: bool, response_time: float):
        """Record proxy performance metrics."""
        if proxy not in self.proxy_metrics:
            self.proxy_metrics[proxy] = ProxyMetrics()
        
        metrics = self.proxy_metrics[proxy]
        metrics.last_used = datetime.now()
        
        if success:
            metrics.success_count += 1
            metrics.consecutive_failures = 0
            
            # Update average response time
            total_time = metrics.avg_response_time * (metrics.success_count - 1) + response_time
            metrics.avg_response_time = total_time / metrics.success_count
            
            self.successful_requests += 1
        else:
            metrics.failure_count += 1
            metrics.consecutive_failures += 1
            
            # Mark as failed if too many consecutive failures
            if metrics.consecutive_failures >= self.max_failures:
                self.failed_proxies[proxy] = datetime.now()
        
        # Update reliability score
        total_attempts = metrics.success_count + metrics.failure_count
        metrics.reliability_score = metrics.success_count / total_attempts if total_attempts > 0 else 1.0
        
        self.total_requests += 1
    
    def get_ml_delay(self) -> float:
        """Get ML-generated delay for human-like patterns."""
        if self.ml_pattern_generator:
            return self.ml_pattern_generator.generate_delay_pattern()
        return random.uniform(1.0, 3.0)
    
    def get_ml_headers(self) -> Dict[str, str]:
        """Get ML-generated headers for anti-detection."""
        if self.ml_pattern_generator:
            return self.ml_pattern_generator.generate_headers()
        return {'User-Agent': 'GoalDiggers/1.0'}
    
    async def solve_captcha_if_needed(self, captcha_image_url: str) -> Optional[str]:
        """Solve CAPTCHA if encountered."""
        if self.captcha_solver:
            return await self.captcha_solver.solve_captcha(captcha_image_url)
        return None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        active_proxies = len([p for p in self.proxy_pool if p not in self.failed_proxies])
        
        avg_reliability = np.mean([m.reliability_score for m in self.proxy_metrics.values()]) if self.proxy_metrics else 0
        
        return {
            'total_proxies': len(self.proxy_pool),
            'active_proxies': active_proxies,
            'failed_proxies': len(self.failed_proxies),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'success_rate': success_rate,
            'average_reliability': avg_reliability
        }
