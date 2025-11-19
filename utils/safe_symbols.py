#!/usr/bin/env python3
"""
Utility to standardize symbol usage in GoalDiggers platform.

This module provides safe alternatives to Unicode emojis that might cause
display issues in certain terminals, especially on Windows.
"""

import logging


class SafeSymbols:
    """Provides ASCII alternatives to common Unicode symbols/emojis."""
    
    # Status indicators
    CHECK = "[+]"  # Instead of ✅
    CROSS = "[x]"  # Instead of ❌
    WARNING = "[!]"  # Instead of ⚠️
    INFO = "[i]"  # Instead of ℹ️
    
    # Progress indicators
    LOADING = "..."  # Instead of ⏳
    RUNNING = "[>]"  # Instead of 🚀
    WAITING = "[.]"  # Instead of ⏱️
    COMPLETE = "[=]"  # Instead of 🏁
    
    # Feature indicators
    DASHBOARD = "[D]"  # Instead of 📊
    DATA = "[#]"  # Instead of 📊
    ANALYSIS = "[A]"  # Instead of 📈
    CONFIG = "[C]"  # Instead of ⚙️
    BETTING = "[B]"  # Instead of 🎲
    
    # Header decorator
    @staticmethod
    def header(text):
        """Create a header with ASCII box drawing."""
        width = max(len(text) + 4, 50)
        top = "+" + "-" * (width - 2) + "+"
        middle = "| " + text + " " * (width - len(text) - 4) + " |"
        bottom = "+" + "-" * (width - 2) + "+"
        return f"{top}\n{middle}\n{bottom}"

def patch_logger(logger):
    """
    Patch a logger object to use safe symbols.
    
    Args:
        logger: The logger object to patch
    
    Returns:
        The patched logger
    """
    original_info = logger.info
    original_warning = logger.warning
    original_error = logger.error
    original_debug = logger.debug
    
    def safe_info(msg, *args, **kwargs):
        # Replace common emojis with safe alternatives
        msg = (msg.replace("✅", SafeSymbols.CHECK)
                  .replace("🔍", SafeSymbols.INFO)
                  .replace("🚀", SafeSymbols.RUNNING)
                  .replace("🎯", SafeSymbols.COMPLETE)
                  .replace("📋", "[LIST]")
                  .replace("🔧", SafeSymbols.CONFIG)
                  .replace("⚙️", SafeSymbols.CONFIG)
                  .replace("🎉", "[PARTY]")
                  .replace("⏱️", SafeSymbols.WAITING))
        return original_info(msg, *args, **kwargs)
    
    def safe_warning(msg, *args, **kwargs):
        msg = msg.replace("⚠️", SafeSymbols.WARNING)
        return original_warning(msg, *args, **kwargs)
    
    def safe_error(msg, *args, **kwargs):
        msg = msg.replace("❌", SafeSymbols.CROSS)
        return original_error(msg, *args, **kwargs)
    
    def safe_debug(msg, *args, **kwargs):
        msg = msg.replace("🔍", SafeSymbols.INFO)
        return original_debug(msg, *args, **kwargs)
    
    # Patch the logger methods
    logger.info = safe_info
    logger.warning = safe_warning
    logger.error = safe_error
    logger.debug = safe_debug
    
    return logger

def get_safe_logger(name):
    """
    Get a logger that safely handles emojis by replacing them with ASCII alternatives.
    
    Args:
        name: Logger name
    
    Returns:
        A patched logger that handles emojis safely
    """
    logger = logging.getLogger(name)
    return patch_logger(logger)

# Example usage
if __name__ == "__main__":
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from utils.logging_config import configure_logging  # type: ignore
            configure_logging()
        except Exception:
            logging.basicConfig(level=logging.INFO)
    
    # Get a safe logger
    logger = get_safe_logger(__name__)
    
    # Test with various symbols that would normally cause issues
    logger.info("✅ This checkmark will be replaced with [+]")
    logger.warning("⚠️ This warning symbol will be replaced with [!]")
    logger.error("❌ This error symbol will be replaced with [x]")
    
    # Test header
    print(SafeSymbols.header("GoalDiggers Platform"))
