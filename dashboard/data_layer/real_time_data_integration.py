#!/usr/bin/env python3
"""
Real-Time Data Integration Layer for Dashboard
Handles real-time data updates for the GoalDiggers dashboard with performance optimization.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RealTimeUpdate:
    """Data class for real-time updates."""
    update_type: str  # 'match_event', 'odds_change', 'prediction_update'
    match_id: str
    data: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=high, 2=medium, 3=low


class RealTimeDataIntegration:
    """
    Real-time data integration layer for dashboard updates.
    
    Features:
    - Real-time data streaming to dashboard components
    - Intelligent update batching for performance
    - Priority-based update processing
    - Automatic fallback to cached data
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize real-time data integration."""
        self.config = config or self._get_default_config()
        
        # Data sources
        self.live_data_processor = None
        self.odds_aggregator = None
        self.cache_manager = None
        
        # Update management
        self.update_queue = asyncio.Queue()
        self.subscribers = {}  # component_id -> callback function
        self.active_matches = set()
        
        # Performance tracking
        self.update_stats = {
            'updates_processed': 0,
            'avg_processing_time': 0,
            'cache_hit_rate': 0,
            'data_freshness': 0
        }
        
        # Processing state
        self.is_processing = False
        self.processing_task = None
        
        logger.info("🔄 Real-Time Data Integration initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'processing': {
                'batch_size': 50,  # Process updates in batches
                'batch_timeout': 1.0,  # Maximum batch wait time (seconds)
                'max_queue_size': 1000,  # Maximum queued updates
                'priority_processing': True  # Process high priority updates first
            },
            'performance': {
                'target_update_latency': 500,  # Target <500ms update latency
                'max_data_staleness': 30,  # Maximum 30 seconds data staleness
                'cache_refresh_interval': 15,  # Refresh cache every 15 seconds
                'performance_monitoring_interval': 60  # Monitor every minute
            },
            'fallback': {
                'enable_cache_fallback': True,
                'cache_fallback_timeout': 5,  # Fallback to cache after 5s
                'max_fallback_age': 300  # Maximum 5 minutes fallback data age
            }
        }
    
    async def initialize_data_sources(self):
        """Initialize connections to data sources."""
        try:
            # Initialize live data processor
            from data.streams.live_data_processor import get_live_data_processor
            self.live_data_processor = get_live_data_processor()
            
            # Initialize odds aggregator
            from data.market.odds_aggregator import get_odds_aggregator
            self.odds_aggregator = get_odds_aggregator()
            
            # Initialize cache manager
            from data.caching.intelligent_cache_manager import get_cache_manager
            self.cache_manager = get_cache_manager()
            
            logger.info("✅ Data sources initialized for real-time integration")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize data sources: {e}")
            raise
    
    async def start_real_time_processing(self):
        """Start real-time data processing."""
        if self.is_processing:
            logger.warning("Real-time processing already active")
            return
        
        try:
            # Initialize data sources
            await self.initialize_data_sources()
            
            # Start processing task
            self.is_processing = True
            self.processing_task = asyncio.create_task(self._processing_loop())
            
            # Start performance monitoring
            asyncio.create_task(self._performance_monitoring())
            
            # Subscribe to data source updates
            await self._subscribe_to_data_sources()
            
            logger.info("🚀 Real-time data processing started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start real-time processing: {e}")
            self.is_processing = False
            raise
    
    async def stop_real_time_processing(self):
        """Stop real-time data processing."""
        self.is_processing = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("⏹️ Real-time data processing stopped")
    
    async def _subscribe_to_data_sources(self):
        """Subscribe to updates from data sources."""
        try:
            # Subscribe to live data processor
            if self.live_data_processor and hasattr(self.live_data_processor, 'subscribe'):
                await self.live_data_processor.subscribe(self._handle_live_data_update)
            
            # Subscribe to odds aggregator
            if self.odds_aggregator and hasattr(self.odds_aggregator, 'subscribe'):
                await self.odds_aggregator.subscribe(self._handle_odds_update)
            
            logger.info("✅ Subscribed to data source updates")
            
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to data sources: {e}")
    
    async def _handle_live_data_update(self, update_data: Dict[str, Any]):
        """Handle live data updates."""
        try:
            update = RealTimeUpdate(
                update_type='match_event',
                match_id=update_data.get('match_id', ''),
                data=update_data,
                timestamp=datetime.now(),
                priority=1  # High priority for live events
            )
            
            await self._queue_update(update)
            
        except Exception as e:
            logger.error(f"Error handling live data update: {e}")
    
    async def _handle_odds_update(self, update_data: Dict[str, Any]):
        """Handle odds updates."""
        try:
            update = RealTimeUpdate(
                update_type='odds_change',
                match_id=update_data.get('match_id', ''),
                data=update_data,
                timestamp=datetime.now(),
                priority=2  # Medium priority for odds changes
            )
            
            await self._queue_update(update)
            
        except Exception as e:
            logger.error(f"Error handling odds update: {e}")
    
    async def _queue_update(self, update: RealTimeUpdate):
        """Queue update for processing."""
        try:
            if self.update_queue.qsize() >= self.config['processing']['max_queue_size']:
                # Remove oldest update to make room
                try:
                    await self.update_queue.get_nowait()
                    logger.warning("Dropped oldest update due to queue overflow")
                except asyncio.QueueEmpty:
                    pass
            
            await self.update_queue.put(update)
            
        except Exception as e:
            logger.error(f"Error queuing update: {e}")
    
    async def _processing_loop(self):
        """Main processing loop for real-time updates."""
        while self.is_processing:
            try:
                # Collect batch of updates
                updates = await self._collect_update_batch()
                
                if updates:
                    # Process batch
                    await self._process_update_batch(updates)
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)  # Brief pause on error
    
    async def _collect_update_batch(self) -> List[RealTimeUpdate]:
        """Collect a batch of updates for processing."""
        updates = []
        batch_size = self.config['processing']['batch_size']
        batch_timeout = self.config['processing']['batch_timeout']
        
        try:
            # Wait for first update
            update = await asyncio.wait_for(
                self.update_queue.get(),
                timeout=batch_timeout
            )
            updates.append(update)
            
            # Collect additional updates up to batch size
            while len(updates) < batch_size:
                try:
                    update = await asyncio.wait_for(
                        self.update_queue.get(),
                        timeout=0.1  # Short timeout for additional updates
                    )
                    updates.append(update)
                except asyncio.TimeoutError:
                    break  # No more updates available
            
        except asyncio.TimeoutError:
            pass  # No updates available
        
        return updates
    
    async def _process_update_batch(self, updates: List[RealTimeUpdate]):
        """Process a batch of updates."""
        start_time = time.time()
        
        try:
            # Sort by priority if enabled
            if self.config['processing']['priority_processing']:
                updates.sort(key=lambda x: x.priority)
            
            # Group updates by type for efficient processing
            grouped_updates = {}
            for update in updates:
                if update.update_type not in grouped_updates:
                    grouped_updates[update.update_type] = []
                grouped_updates[update.update_type].append(update)
            
            # Process each group
            for update_type, type_updates in grouped_updates.items():
                await self._process_updates_by_type(update_type, type_updates)
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_performance_stats(len(updates), processing_time)
            
            logger.debug(f"Processed {len(updates)} updates in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error processing update batch: {e}")
    
    async def _process_updates_by_type(self, update_type: str, updates: List[RealTimeUpdate]):
        """Process updates of a specific type."""
        try:
            if update_type == 'match_event':
                await self._process_match_events(updates)
            elif update_type == 'odds_change':
                await self._process_odds_changes(updates)
            elif update_type == 'prediction_update':
                await self._process_prediction_updates(updates)
            
        except Exception as e:
            logger.error(f"Error processing {update_type} updates: {e}")
    
    async def _process_match_events(self, updates: List[RealTimeUpdate]):
        """Process match event updates."""
        for update in updates:
            try:
                # Cache the update
                if self.cache_manager:
                    cache_key = f"match_event:{update.match_id}:{update.timestamp.isoformat()}"
                    await self.cache_manager.set(cache_key, update.data, ttl=300)
                
                # Notify subscribers
                await self._notify_subscribers('match_event', update.data)
                
            except Exception as e:
                logger.error(f"Error processing match event: {e}")
    
    async def _process_odds_changes(self, updates: List[RealTimeUpdate]):
        """Process odds change updates."""
        for update in updates:
            try:
                # Cache the update
                if self.cache_manager:
                    cache_key = f"odds:{update.match_id}"
                    await self.cache_manager.set(cache_key, update.data, ttl=60)
                
                # Notify subscribers
                await self._notify_subscribers('odds_change', update.data)
                
            except Exception as e:
                logger.error(f"Error processing odds change: {e}")
    
    async def _process_prediction_updates(self, updates: List[RealTimeUpdate]):
        """Process prediction updates."""
        for update in updates:
            try:
                # Cache the update
                if self.cache_manager:
                    cache_key = f"prediction:{update.match_id}"
                    await self.cache_manager.set(cache_key, update.data, ttl=300)
                
                # Notify subscribers
                await self._notify_subscribers('prediction_update', update.data)
                
            except Exception as e:
                logger.error(f"Error processing prediction update: {e}")
    
    async def _notify_subscribers(self, update_type: str, data: Dict[str, Any]):
        """Notify subscribers of updates."""
        try:
            for component_id, callback in self.subscribers.items():
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update_type, data)
                    else:
                        callback(update_type, data)
                except Exception as e:
                    logger.error(f"Error notifying subscriber {component_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error notifying subscribers: {e}")
    
    def subscribe(self, component_id: str, callback: Callable):
        """Subscribe to real-time updates."""
        self.subscribers[component_id] = callback
        logger.info(f"Component {component_id} subscribed to real-time updates")
    
    def unsubscribe(self, component_id: str):
        """Unsubscribe from real-time updates."""
        if component_id in self.subscribers:
            del self.subscribers[component_id]
            logger.info(f"Component {component_id} unsubscribed from real-time updates")
    
    async def get_cached_data(self, data_type: str, match_id: str) -> Optional[Dict[str, Any]]:
        """Get cached data with fallback."""
        try:
            if not self.cache_manager:
                return None
            
            cache_key = f"{data_type}:{match_id}"
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_data
            else:
                logger.debug(f"Cache miss for {cache_key}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None
    
    def _update_performance_stats(self, updates_count: int, processing_time_ms: float):
        """Update performance statistics."""
        self.update_stats['updates_processed'] += updates_count
        
        # Update average processing time
        if self.update_stats['avg_processing_time'] == 0:
            self.update_stats['avg_processing_time'] = processing_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.update_stats['avg_processing_time'] = (
                alpha * processing_time_ms + 
                (1 - alpha) * self.update_stats['avg_processing_time']
            )
    
    async def _performance_monitoring(self):
        """Monitor performance and trigger alerts if needed."""
        while self.is_processing:
            try:
                await asyncio.sleep(self.config['performance']['performance_monitoring_interval'])
                
                # Check performance metrics
                avg_latency = self.update_stats['avg_processing_time']
                target_latency = self.config['performance']['target_update_latency']
                
                if avg_latency > target_latency:
                    logger.warning(f"⚠️ Update latency high: {avg_latency:.1f}ms > {target_latency}ms")
                
                # Get cache statistics
                if self.cache_manager:
                    cache_stats = self.cache_manager.get_cache_stats()
                    self.update_stats['cache_hit_rate'] = cache_stats['hit_rate']
                
                logger.info(f"📊 Performance: {self.update_stats['updates_processed']} updates, "
                           f"{avg_latency:.1f}ms avg latency, "
                           f"{self.update_stats['cache_hit_rate']:.1%} cache hit rate")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            'updates_processed': self.update_stats['updates_processed'],
            'avg_processing_time_ms': self.update_stats['avg_processing_time'],
            'cache_hit_rate': self.update_stats['cache_hit_rate'],
            'active_subscribers': len(self.subscribers),
            'queue_size': self.update_queue.qsize(),
            'is_processing': self.is_processing
        }


# Singleton instance
_real_time_integration_instance = None

def get_real_time_integration(config: Optional[Dict[str, Any]] = None) -> RealTimeDataIntegration:
    """Get singleton real-time integration instance."""
    global _real_time_integration_instance
    if _real_time_integration_instance is None:
        _real_time_integration_instance = RealTimeDataIntegration(config)
    return _real_time_integration_instance
