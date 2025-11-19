#!/usr/bin/env python3
"""
Automated Data Refresh Scheduler - GoalDiggers Platform

Schedules periodic data refresh to keep predictions fresh and accurate.
This script should be run as a background service or scheduled task.

Usage:
    # Run once
    python scripts/schedule_data_refresh.py --run-once
    
    # Run as daemon (continuous scheduling)
    python scripts/schedule_data_refresh.py --daemon
    
    # Configure via environment
    export DATA_REFRESH_INTERVAL=3600  # seconds (default: 1 hour)
    python scripts/schedule_data_refresh.py --daemon
"""
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_refresh_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataRefreshScheduler:
    """Automated scheduler for data refresh operations."""

    def __init__(self, interval_seconds: int = 3600):
        """
        Initialize the scheduler.
        
        Args:
            interval_seconds: Time between refresh cycles (default: 1 hour)
        """
        self.interval = interval_seconds
        self.last_refresh = None
        self.refresh_count = 0
        self.error_count = 0

    def should_refresh(self) -> bool:
        """Check if it's time to refresh data."""
        if self.last_refresh is None:
            return True
        
        elapsed = (datetime.now() - self.last_refresh).total_seconds()
        return elapsed >= self.interval

    def refresh_data(self) -> bool:
        """
        Execute data refresh operation.
        
        Returns:
            True if refresh succeeded, False otherwise
        """
        logger.info("🔄 Starting data refresh cycle...")
        
        try:
            # 1. Refresh real data integrator cache
            self._refresh_real_data()
            
            # 2. Update async data integrator cache
            self._refresh_async_data()
            
            # 3. Clear stale prediction cache
            self._clear_stale_cache()
            
            # 4. Verify data freshness
            self._verify_freshness()
            
            self.last_refresh = datetime.now()
            self.refresh_count += 1
            
            logger.info(f"✅ Data refresh completed successfully (cycle #{self.refresh_count})")
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Data refresh failed: {e}", exc_info=True)
            return False

    def _refresh_real_data(self):
        """Refresh real data integrator."""
        try:
            from real_data_integrator import get_real_matches, real_data_integrator
            
            logger.info("  📡 Fetching fresh match data...")
            matches = get_real_matches(days_ahead=14)
            
            real_matches = [
                m for m in matches 
                if not str(m.get('api_id', '')).startswith('fallback_')
            ]
            
            logger.info(f"  ✅ Retrieved {len(real_matches)} real matches, {len(matches) - len(real_matches)} fallback")
            
            # Log data pipeline snapshot
            snapshot = real_data_integrator.get_data_pipeline_snapshot()
            logger.info(f"  📊 Pipeline snapshot: {snapshot}")
            
        except Exception as e:
            logger.warning(f"  ⚠️ Real data refresh warning: {e}")

    def _refresh_async_data(self):
        """Refresh async data integrator cache."""
        try:
            import asyncio
            from async_data_integrator import get_async_comprehensive_data
            
            logger.info("  🔄 Refreshing async data cache...")
            
            # Run async fetch
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                data = loop.run_until_complete(get_async_comprehensive_data())
                logger.info(f"  ✅ Async data refreshed: {len(data.get('fixtures', []))} fixtures")
            finally:
                loop.close()
                
        except Exception as e:
            logger.warning(f"  ⚠️ Async data refresh warning: {e}")

    def _clear_stale_cache(self):
        """Clear stale prediction cache entries."""
        try:
            from utils.enhanced_cache_manager import EnhancedCacheManager
            
            logger.info("  🧹 Clearing stale cache...")
            cache_manager = EnhancedCacheManager()
            
            # Clear entries older than 6 hours
            cleared = cache_manager.clear_stale_entries(max_age_hours=6)
            logger.info(f"  ✅ Cleared {cleared} stale cache entries")
            
        except Exception as e:
            logger.warning(f"  ⚠️ Cache cleanup warning: {e}")

    def _verify_freshness(self):
        """Verify data freshness after refresh."""
        try:
            from real_data_integrator import real_data_integrator
            
            snapshot = real_data_integrator.get_data_pipeline_snapshot()
            
            if snapshot.get('last_fetch_completed_at'):
                age_hours = (time.time() - snapshot['last_fetch_completed_at']) / 3600
                
                if age_hours < 1:
                    logger.info(f"  ✅ Data freshness verified: {age_hours:.2f} hours old")
                else:
                    logger.warning(f"  ⚠️ Data may be stale: {age_hours:.2f} hours old")
            else:
                logger.warning("  ⚠️ Could not verify data freshness")
                
        except Exception as e:
            logger.warning(f"  ⚠️ Freshness verification warning: {e}")

    def run_once(self):
        """Execute a single refresh cycle."""
        logger.info("🚀 Running single data refresh cycle...")
        success = self.refresh_data()
        
        if success:
            logger.info("✅ Single refresh completed")
            sys.exit(0)
        else:
            logger.error("❌ Single refresh failed")
            sys.exit(1)

    def run_daemon(self):
        """Run as continuous daemon with scheduled refreshes."""
        logger.info(f"🚀 Starting data refresh daemon (interval: {self.interval}s)")
        logger.info("Press Ctrl+C to stop...")
        
        try:
            while True:
                if self.should_refresh():
                    self.refresh_data()
                    
                    # Log status
                    logger.info(
                        f"📊 Status: {self.refresh_count} refreshes, "
                        f"{self.error_count} errors, "
                        f"next refresh in {self.interval}s"
                    )
                
                # Sleep in smaller intervals for responsive shutdown
                sleep_interval = min(60, self.interval)
                time.sleep(sleep_interval)
                
        except KeyboardInterrupt:
            logger.info("\n👋 Daemon stopped gracefully")
            logger.info(f"📊 Final stats: {self.refresh_count} refreshes, {self.error_count} errors")
            sys.exit(0)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GoalDiggers Data Refresh Scheduler")
    parser.add_argument('--run-once', action='store_true', help='Run single refresh cycle')
    parser.add_argument('--daemon', action='store_true', help='Run as continuous daemon')
    parser.add_argument('--interval', type=int, 
                       default=int(os.getenv('DATA_REFRESH_INTERVAL', '3600')),
                       help='Refresh interval in seconds (default: 3600)')
    
    args = parser.parse_args()
    
    scheduler = DataRefreshScheduler(interval_seconds=args.interval)
    
    if args.run_once:
        scheduler.run_once()
    elif args.daemon:
        scheduler.run_daemon()
    else:
        # Default: run once
        logger.info("No mode specified, running single refresh (use --daemon for continuous)")
        scheduler.run_once()


if __name__ == "__main__":
    main()
