"""
Background scheduler for automated data ingestion and refresh.

Provides a lightweight scheduler that runs ingestion jobs at configurable
intervals without external dependencies like Celery or APScheduler.
"""
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class BackgroundScheduler:
    """Simple background scheduler for ingestion tasks."""
    
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
    def start_daily_ingestion(self, 
                            db_uri: Optional[str] = None,
                            interval_hours: int = 6,
                            days_back: int = 2,
                            days_ahead: int = 7):
        """Start background ingestion that runs every `interval_hours`."""
        if self._running:
            logger.warning("Scheduler already running")
            return
            
        def _ingestion_loop():
            logger.info(f"Background ingestion started (interval: {interval_hours}h)")
            while not self._stop_event.is_set():
                try:
                    from ingestion.etl_pipeline import ingest_from_sources
                    from ingestion.historical_backfill import backfill_if_empty
                    
                    # Ensure we have baseline data
                    backfill_if_empty(db_uri, min_threshold=500)
                    
                    # Ingest fresh data
                    upserted = ingest_from_sources(db_uri, days_back, days_ahead)
                    logger.info(f"Background ingestion completed: {upserted} matches processed")
                    
                except Exception as e:
                    logger.error(f"Background ingestion error: {e}")
                
                # Wait for next interval
                if self._stop_event.wait(interval_hours * 3600):
                    break
                    
            logger.info("Background ingestion stopped")
            
        self._running = True
        self._thread = threading.Thread(target=_ingestion_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        """Stop the background scheduler."""
        if not self._running:
            return
            
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        logger.info("Background scheduler stopped")
        
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running and self._thread and self._thread.is_alive()


# Global scheduler instance
_scheduler: Optional[BackgroundScheduler] = None


def start_background_ingestion(db_uri: Optional[str] = None, interval_hours: int = 6):
    """Start global background ingestion scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = BackgroundScheduler()
    _scheduler.start_daily_ingestion(db_uri, interval_hours)
    return _scheduler


def stop_background_ingestion():
    """Stop global background ingestion scheduler."""
    global _scheduler
    if _scheduler:
        _scheduler.stop()
        _scheduler = None


def is_background_ingestion_running() -> bool:
    """Check if background ingestion is running."""
    return _scheduler is not None and _scheduler.is_running()
