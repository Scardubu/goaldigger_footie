import time
from functools import wraps

from dashboard.error_log import log_error  # Import log_error


def retry_with_backoff(retries=3, delay=5):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    log_error("High-level operation failed within retry loop", e) # Log the error
                    if attempt == retries - 1:
                        raise # Re-raise on the last attempt
                    time.sleep(delay)
        return wrapped
    return decorator
