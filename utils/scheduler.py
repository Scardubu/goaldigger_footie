import threading


class AppScheduler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppScheduler, cls).__new__(cls)
        return cls._instance

    def start(self):
        print("Scheduler started (stub).")