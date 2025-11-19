"""
Mobile optimization components for GoalDiggers Football Betting Platform.

This package provides mobile-specific optimizations including:
- Mobile device detection
- Touch-optimized interfaces
- Responsive layout components
- Performance optimizations for mobile devices
"""

# Version info
__version__ = "1.0.0"
__author__ = "GoalDiggers Development Team"

# Import main components
from .mobile_detection import detect_mobile
from .responsive_layout import ResponsiveLayout

__all__ = [
    "detect_mobile",
    "ResponsiveLayout",
]
