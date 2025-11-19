"""
Dashboard UI components package.
"""

# Import key components for easier access
try:
    from . import achievement_system, gamification_integration
except ImportError as e:
    # Components may not be available in all contexts
    pass
