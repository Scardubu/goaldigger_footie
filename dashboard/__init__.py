# This file marks the dashboard directory as a Python package.

# Phase 6 Cleanup: Export only active production dashboard
from dashboard.enhanced_production_homepage import ProductionDashboardHomepage
from dashboard.unified_goaldiggers_dashboard import UnifiedGoalDiggersDashboard

__all__ = [
    'ProductionDashboardHomepage',
    'UnifiedGoalDiggersDashboard'
]

# Legacy dashboard exports removed - Phase 6 cleanup completed
