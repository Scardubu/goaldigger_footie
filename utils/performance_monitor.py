"""Performance monitoring utilities"""

import time
import psutil
import logging
from functools import wraps
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system and application performance"""
    
    def __init__(self):
        self.metrics = {}
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters()._asdict(),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def monitor_function(self, func_name: str = None):
        """Decorator to monitor function performance"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    metric_name = func_name or func.__name__
                    self.metrics[metric_name] = {
                        'execution_time': execution_time,
                        'success': success,
                        'error': error,
                        'timestamp': end_time
                    }
                    
                    if execution_time > 1.0:  # Log slow operations
                        logger.warning(f"Slow operation: {metric_name} took {execution_time:.2f}s")
                
                return result
            return wrapper
        return decorator

# Global performance monitor instance
perf_monitor = PerformanceMonitor()
