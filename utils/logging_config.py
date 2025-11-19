# utils/logging_config.py
import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def setup_logging(
    log_level=logging.INFO,
    log_to_file=True,
    log_to_console=True,
    log_file=None,
    log_format=None,
    rotation_days=7,
    retention_days=30,
    config_dict=None
):
    """
    Configures logging for the application, optionally using a config dict.
    Args:
        log_level: Logging level (int or str)
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        log_file: Path to log file
        log_format: Log format string
        rotation_days: Days before rotating log
        retention_days: Days to keep old logs
        config_dict: Optional config dict from config.yaml
    """
    import logging.handlers

    # If config_dict is provided, override args
    if config_dict:
        log_cfg = config_dict.get("logging", {})
        log_level = log_cfg.get("level", log_level)
        log_format = log_cfg.get("format", log_format)
        log_file = log_cfg.get("file", log_file)
        # Use app log path from config if present
        if not log_file:
            log_paths = config_dict.get("paths", {}).get("logs", {})
            log_file = log_paths.get("app", log_file)
        # Parse rotation/retention
        rotation = log_cfg.get("rotation", "7 days")
        retention = log_cfg.get("retention", "30 days")
        try:
            rotation_days = int(str(rotation).split()[0])
        except Exception:
            rotation_days = 7
        try:
            retention_days = int(str(retention).split()[0])
        except Exception:
            retention_days = 30

    log_formatter = logging.Formatter(
        log_format or "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level if isinstance(log_level, int) else getattr(logging, str(log_level).upper(), logging.INFO))
    # When running under pytest, do not clear or replace root handlers so pytest's capture handler
    # (LogCaptureHandler) remains attached. Tests and pytest will control handler attachment.
    under_pytest = os.getenv("PYTEST_CURRENT_TEST") is not None
    if not under_pytest:
        root_logger.handlers.clear()
    if log_to_console:
        # Direct console logs to stderr to keep stdout clean for script results
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)
        # Log the initial message to stderr as well
        root_logger.info("Logging to console enabled (stderr).")
    if log_to_file and log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            # Use TimedRotatingFileHandler for day-based rotation
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file, when="D", interval=rotation_days, backupCount=retention_days
            )
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)
            logging.info(f"Logging to file enabled: {log_file}")
        except Exception as e:
            log_error(f"Failed to configure file logging to {log_file}", e) # Use log_error


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (defaults to calling module name)
        
    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        from datetime import timezone
        base = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base)

def configure_logging(level: str = None, json_mode: bool = None, clear_existing: bool = False) -> None:
    """Idempotent lightweight logger configuration.

    Args:
        level: Log level name (default INFO or existing root level)
        json_mode: If True emit JSON lines; auto-detect via LOG_JSON env if None
        clear_existing: If True, clears existing handlers first
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    if json_mode is None:
        json_mode = os.getenv("LOG_JSON", "0").lower() not in ("0", "false")
    root = logging.getLogger()
    if root.handlers and not clear_existing:
        return
    if clear_existing:
        root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    if json_mode:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    root.addHandler(handler)
    logging.getLogger(__name__).debug("Logging configured", extra={"json_mode": json_mode, "level": level})


# Example of how to use it at the start of your main script/app:
# if __name__ == "__main__":
#     from utils.logging_config import setup_logging
#     setup_logging(log_level=logging.DEBUG) # Set desired level
#
#     logger = logging.getLogger(__name__)
#     logger.debug("This is a debug message.")
#     logger.info("This is an info message.")
#     logger.warning("This is a warning message.")
#     logger.error("This is an error message.")
