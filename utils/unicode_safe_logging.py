#!/usr/bin/env python3
"""
Unicode-Safe Logging Configuration for GoalDiggers Platform

Provides comprehensive Unicode-safe logging configuration to prevent
UnicodeEncodeError issues on Windows systems with cp1252 encoding.
"""

import logging
import sys
import os
from typing import Optional


class UnicodeCompatibleStreamHandler(logging.StreamHandler):
    """Stream handler that safely handles Unicode characters on Windows."""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Replace any characters that can't be encoded in the console
            msg = self.format(record)
            # Replace common emoji characters with ASCII alternatives
            emoji_replacements = {
                '✅': '[+]',
                '❌': '[x]',
                '⚠️': '[!]',
                '🔄': '[>]',
                '📊': '[#]',
                '🎯': '[*]',
                '⚡': '[!]',
                '🔍': '[?]',
                '🚀': '[^]',
                '📡': '[~]',
                '👤': '[U]',
                '🎉': '[P]',
                '🔧': '[T]',
                '⚙️': '[C]',
                '🎲': '[D]',
                '📈': '[/]',
                '📋': '[L]',
                '⏱️': '[.]',
                '🏁': '[=]',
                '🔗': '[&]'
            }
            
            for emoji, replacement in emoji_replacements.items():
                msg = msg.replace(emoji, replacement)
            
            # If still failing, encode with replacement
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                msg = msg.encode('ascii', 'replace').decode('ascii')
                stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_unicode_safe_logging(level: int = logging.INFO, 
                              log_file: Optional[str] = None) -> None:
    """
    Setup Unicode-safe logging configuration for the entire application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler with Unicode safety
    console_handler = UnicodeCompatibleStreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Add console handler
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # Set root logger level
    root_logger.setLevel(level)


def get_unicode_safe_logger(name: str) -> logging.Logger:
    """
    Get a logger that safely handles Unicode characters.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance with Unicode-safe configuration
    """
    logger = logging.getLogger(name)
    
    # If no handlers are configured, setup basic Unicode-safe logging
    if not logger.handlers and not logging.getLogger().handlers:
        setup_unicode_safe_logging()
    
    return logger


def patch_existing_loggers():
    """
    Patch existing loggers in the application to use Unicode-safe handlers.
    """
    # Get all existing loggers
    loggers = [logging.getLogger(name) for name in logging.Logger.manager.loggerDict]
    loggers.append(logging.getLogger())  # Add root logger
    
    for logger in loggers:
        # Replace stream handlers with Unicode-safe versions
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, UnicodeCompatibleStreamHandler):
                # Create new Unicode-safe handler
                new_handler = UnicodeCompatibleStreamHandler(handler.stream)
                new_handler.setFormatter(handler.formatter)
                new_handler.setLevel(handler.level)
                
                # Replace the handler
                logger.removeHandler(handler)
                logger.addHandler(new_handler)


# Auto-setup when module is imported
if __name__ != "__main__":
    # Only setup if not already configured
    if not logging.getLogger().handlers:
        setup_unicode_safe_logging()


if __name__ == "__main__":
    # Test the Unicode-safe logging
    setup_unicode_safe_logging()
    
    logger = get_unicode_safe_logger(__name__)
    
    # Test with various Unicode characters
    logger.info("✅ Testing checkmark")
    logger.warning("⚠️ Testing warning symbol")
    logger.error("❌ Testing error symbol")
    logger.info("🔄 Testing loading symbol")
    logger.info("📊 Testing dashboard symbol")
    logger.info("🎯 Testing target symbol")
    
    print("Unicode-safe logging test completed successfully!")
