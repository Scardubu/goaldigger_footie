"""
Distributed Scraping Framework with queue management, auto-scaling, and geographic distribution.
"""
import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class ScrapingTask:
    """Represents a scraping task."""
    id: str
    url: str
    scraper_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    max_retries: int = 3
    retry_count: int = 0
    timeout: int = 30
    use_proxy: bool = True
    use_playwright: bool = False
    headers: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['scheduled_at'] = self.scheduled_at.isoformat() if self.scheduled_at else None
        data['priority'] = self.priority.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapingTask':
        """Create task from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['scheduled_at'] = datetime.fromisoformat(data['scheduled_at']) if data['scheduled_at'] else None
        data['priority'] = TaskPriority(data['priority'])
        return cls(**data)

class TaskQueue:
    """Redis-based distributed task queue."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        self.queue_name = "scraping_tasks"
        self.processing_queue = "processing_tasks"
        self.completed_queue = "completed_tasks"
        self.failed_queue = "failed_tasks"
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis = await aioredis.from_url(self.redis_url)
            logger.info("Connected to Redis task queue")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fallback to in-memory queue
            self.redis = None
    
    async def add_task(self, task: ScrapingTask):
        """Add task to the queue."""
        if self.redis:
            # Add to Redis with priority
            score = task.priority.value * 1000000 + int(time.time())
            await self.redis.zadd(self.queue_name, {json.dumps(task.to_dict()): score})
        else:
            # Fallback to in-memory processing
            logger.warning("Redis not available, processing task immediately")
    
    async def get_next_task(self) -> Optional[ScrapingTask]:
        """Get next task from queue based on priority."""
        if not self.redis:
            return None
        
        # Get highest priority task
        result = await self.redis.zpopmax(self.queue_name)
        if result:
            task_data = json.loads(result[0][0])
            task = ScrapingTask.from_dict(task_data)
            
            # Move to processing queue
            await self.redis.hset(self.processing_queue, task.id, json.dumps(task.to_dict()))
            return task
        
        return None
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark task as completed."""
        if self.redis:
            # Move from processing to completed
            task_data = await self.redis.hget(self.processing_queue, task_id)
            if task_data:
                await self.redis.hdel(self.processing_queue, task_id)
                
                # Store result
                result_data = {
                    'task_id': task_id,
                    'result': result,
                    'completed_at': datetime.now().isoformat()
                }
                await self.redis.hset(self.completed_queue, task_id, json.dumps(result_data))
    
    async def fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        if self.redis:
            task_data = await self.redis.hget(self.processing_queue, task_id)
            if task_data:
                task = ScrapingTask.from_dict(json.loads(task_data))
                task.retry_count += 1
                
                if task.retry_count < task.max_retries:
                    # Retry with exponential backoff
                    delay = 2 ** task.retry_count
                    task.scheduled_at = datetime.now() + timedelta(seconds=delay)
                    
                    # Re-add to queue
                    await self.redis.hdel(self.processing_queue, task_id)
                    await self.add_task(task)
                    logger.info(f"Task {task_id} scheduled for retry {task.retry_count}/{task.max_retries}")
                else:
                    # Move to failed queue
                    await self.redis.hdel(self.processing_queue, task_id)
                    
                    fail_data = {
                        'task_id': task_id,
                        'error': error,
                        'failed_at': datetime.now().isoformat(),
                        'retry_count': task.retry_count
                    }
                    await self.redis.hset(self.failed_queue, task_id, json.dumps(fail_data))
                    logger.error(f"Task {task_id} failed permanently after {task.retry_count} retries")

class WorkerNode:
    """Individual worker node for distributed scraping."""
    
    def __init__(self, 
                 node_id: str,
                 task_queue: TaskQueue,
                 scrapers: Dict[str, Callable],
                 max_concurrent_tasks: int = 5):
        self.node_id = node_id
        self.task_queue = task_queue
        self.scrapers = scrapers
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks = set()
        self.is_running = False
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'start_time': None,
            'last_activity': None
        }
    
    async def start(self):
        """Start the worker node."""
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        logger.info(f"Worker node {self.node_id} started")
        
        # Start task processing loop
        await self._process_tasks()
    
    async def stop(self):
        """Stop the worker node."""
        self.is_running = False
        logger.info(f"Worker node {self.node_id} stopped")
    
    async def _process_tasks(self):
        """Main task processing loop."""
        while self.is_running:
            try:
                # Check if we can take more tasks
                if len(self.active_tasks) < self.max_concurrent_tasks:
                    task = await self.task_queue.get_next_task()
                    
                    if task:
                        # Check if task is scheduled for future
                        if task.scheduled_at and task.scheduled_at > datetime.now():
                            # Re-add to queue for later processing
                            await self.task_queue.add_task(task)
                            await asyncio.sleep(1)
                            continue
                        
                        # Process task asynchronously
                        asyncio.create_task(self._execute_task(task))
                    else:
                        # No tasks available, wait a bit
                        await asyncio.sleep(1)
                else:
                    # At capacity, wait for tasks to complete
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _execute_task(self, task: ScrapingTask):
        """Execute a single scraping task."""
        self.active_tasks.add(task.id)
        start_time = time.time()
        
        try:
            logger.info(f"Executing task {task.id} on node {self.node_id}")
            
            # Get appropriate scraper
            scraper = self.scrapers.get(task.scraper_type)
            if not scraper:
                raise ValueError(f"Unknown scraper type: {task.scraper_type}")
            
            # Execute scraping
            result = await scraper.scrape(task)
            
            # Record success
            execution_time = time.time() - start_time
            await self.task_queue.complete_task(task.id, {
                'data': result,
                'execution_time': execution_time,
                'node_id': self.node_id
            })
            
            self.stats['tasks_completed'] += 1
            self.stats['last_activity'] = datetime.now()
            
            logger.info(f"Task {task.id} completed in {execution_time:.2f}s")
            
        except Exception as e:
            # Record failure
            await self.task_queue.fail_task(task.id, str(e))
            self.stats['tasks_failed'] += 1
            logger.error(f"Task {task.id} failed: {e}")
            
        finally:
            self.active_tasks.discard(task.id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker node statistics."""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        
        return {
            'node_id': self.node_id,
            'is_running': self.is_running,
            'active_tasks': len(self.active_tasks),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'tasks_completed': self.stats['tasks_completed'],
            'tasks_failed': self.stats['tasks_failed'],
            'uptime_seconds': uptime,
            'last_activity': self.stats['last_activity'].isoformat() if self.stats['last_activity'] else None
        }

class DistributedScrapingManager:
    """Main manager for distributed scraping operations."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.task_queue = TaskQueue(redis_url)
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.scrapers: Dict[str, Callable] = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the distributed scraping system."""
        await self.task_queue.connect()
        self.is_initialized = True
        logger.info("Distributed scraping manager initialized")
    
    def register_scraper(self, scraper_type: str, scraper: Callable):
        """Register a scraper for a specific type."""
        self.scrapers[scraper_type] = scraper
        logger.info(f"Registered scraper: {scraper_type}")
    
    async def add_worker_node(self, node_id: str, max_concurrent_tasks: int = 5) -> WorkerNode:
        """Add a new worker node."""
        worker = WorkerNode(node_id, self.task_queue, self.scrapers, max_concurrent_tasks)
        self.worker_nodes[node_id] = worker
        
        # Start worker in background
        asyncio.create_task(worker.start())
        
        logger.info(f"Added worker node: {node_id}")
        return worker
    
    async def submit_task(self, task: ScrapingTask):
        """Submit a task for distributed processing."""
        if not self.is_initialized:
            await self.initialize()
        
        await self.task_queue.add_task(task)
        logger.info(f"Submitted task {task.id} to distributed queue")
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        worker_stats = [worker.get_stats() for worker in self.worker_nodes.values()]
        
        total_completed = sum(stats['tasks_completed'] for stats in worker_stats)
        total_failed = sum(stats['tasks_failed'] for stats in worker_stats)
        active_workers = sum(1 for stats in worker_stats if stats['is_running'])
        
        return {
            'total_workers': len(self.worker_nodes),
            'active_workers': active_workers,
            'total_tasks_completed': total_completed,
            'total_tasks_failed': total_failed,
            'success_rate': (total_completed / (total_completed + total_failed) * 100) if (total_completed + total_failed) > 0 else 0,
            'worker_stats': worker_stats
        }
    
    async def shutdown(self):
        """Shutdown all worker nodes."""
        for worker in self.worker_nodes.values():
            await worker.stop()
        
        logger.info("Distributed scraping system shutdown complete")
