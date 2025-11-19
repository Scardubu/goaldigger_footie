# utils/scheduling/scheduler.py
import logging

import pytz  # Good practice for timezone handling
from apscheduler.executors.pool import ProcessPoolExecutor, ThreadPoolExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from dashboard.error_log import log_error  # Import log_error

# Configure logging basic setup if not done elsewhere
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AppScheduler:
    def __init__(self, timezone_str: str = "UTC"):
        """
        Initializes the APScheduler with MemoryJobStore.

        Args:
            timezone_str (str): Timezone for the scheduler (e.g., 'UTC', 'Europe/London').
        """
        # Configure job stores and executors
        # MemoryJobStore is simpler, jobs are lost on shutdown.
        # If persistence is needed, consider alternatives like RedisJobStore or a DB store,
        # or simply re-adding jobs from config on startup.
        jobstores = {"default": MemoryJobStore()}
        # Consider ProcessPoolExecutor if jobs are CPU-bound, ThreadPoolExecutor for I/O bound
        executors = {
            "default": ThreadPoolExecutor(10),  # Adjust pool size as needed
            "processpool": ProcessPoolExecutor(3),  # Example for CPU-bound tasks
        }
        job_defaults = {
            "coalesce": False,  # Run missed jobs if scheduler was down (use True carefully)
            "max_instances": 3,  # Max concurrent instances of the same job
        }

        try:
            self.timezone = pytz.timezone(timezone_str)
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone '{timezone_str}', defaulting to UTC.")
            self.timezone = pytz.utc

        # Create scheduler
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=self.timezone,
        )
        logger.info(f"Scheduler initialized with timezone {self.timezone}")

    def start(self):
        """Start the scheduler in the background."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler started.")
        else:
            logger.info("Scheduler is already running.")

    def add_job(self, job_func, job_id: str, trigger="interval", **trigger_args):
        """
        Adds a job to the scheduler.

        Args:
            job_func: The function or callable to execute.
            job_id (str): A unique identifier for the job.
            trigger (str): The trigger type ('interval', 'cron', 'date').
            **trigger_args: Arguments specific to the trigger
                              (e.g., hours=6 for interval, day_of_week='sun', hour=0 for cron).
        """
        try:
            self.scheduler.add_job(
                job_func,
                trigger=trigger,
                id=job_id,
                replace_existing=True,  # Overwrite if job_id exists
                **trigger_args,
            )
            logger.info(
                f"Scheduled job '{job_id}' with trigger '{trigger}' and args {trigger_args}"
            )
        except Exception as e:
            log_error(f"Failed to schedule job '{job_id}'", e) # Use log_error

    # Example specific methods (can be replaced by calling add_job directly)
    def add_interval_job(self, job_func, job_id: str, **interval_args):
        """Adds an interval-based job."""
        self.add_job(job_func, job_id, trigger="interval", **interval_args)

    def add_cron_job(self, job_func, job_id: str, **cron_args):
        """Adds a cron-based job."""
        # Example: add_cron_job(my_func, 'weekly_report', day_of_week='mon', hour=8)
        self.add_job(job_func, job_id, trigger="cron", **cron_args)

    def refresh_proxy_pool(self):
        """Refresh proxy pool periodically"""
        try:
            # Example logic for refreshing proxy pool
            logger.info("Refreshing proxy pool...")
            # Implement proxy pool update logic here
        except Exception as e:
            log_error("Failed to refresh proxy pool", e) # Use log_error

    def add_scraping_job(self, job_func, job_id: str, **interval_args):
        """Add a scraping job with enhanced error handling"""
        def wrapped_job():
            try:
                job_func()
            except Exception as e:
                log_error(f"Scraping job '{job_id}' failed", e) # Use log_error

        self.add_interval_job(wrapped_job, job_id, **interval_args)

    def list_jobs(self):
        """Prints currently scheduled jobs."""
        self.scheduler.print_jobs()

    def shutdown(self, wait=True):
        """Shutdown the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            logger.info("Scheduler shutdown.")
        else:
            logger.info("Scheduler was not running.")

    def __del__(self):
        """Ensure scheduler is shutdown when object is deleted."""
        self.shutdown(wait=False)


# Example Usage (Illustrative - would typically be integrated into the main app/script)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#
#     def my_interval_task(arg1, arg2):
#         print(f"Running interval task at {datetime.now()} with args: {arg1}, {arg2}")
#
#     def my_cron_task():
#         print(f"Running cron task at {datetime.now()}")
#
#     scheduler = AppScheduler(timezone_str='Europe/London')
#
#     # Add jobs
#     scheduler.add_interval_job(my_interval_task, 'task1', seconds=10, args=['hello', 'world'])
#     scheduler.add_cron_job(my_cron_task, 'task2', second='*/15') # Every 15 seconds
#
#     scheduler.start()
#     scheduler.list_jobs()
#
#     try:
#         # Keep the script running otherwise the background scheduler exits
#         while True:
#             time.sleep(2)
#     except (KeyboardInterrupt, SystemExit):
#         scheduler.shutdown()
