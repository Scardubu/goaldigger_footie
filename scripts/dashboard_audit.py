#!/usr/bin/env python3
"""
Dashboard Audit Script

Test all dashboard implementations to determine which ones work
and which should be the primary production dashboard.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_dashboard_import(module_path, class_name):
    """Test importing a dashboard module and class."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        dashboard_class = getattr(module, class_name)
        print(f"✅ {module_path} - IMPORT SUCCESS")
        
        # Try to instantiate
        try:
            dashboard = dashboard_class()
            print(f"✅ {class_name} - INSTANTIATION SUCCESS")
            return True, dashboard
        except Exception as e:
            print(f"⚠️  {class_name} - INSTANTIATION FAILED: {e}")
            return False, None
            
    except ImportError as e:
        print(f"❌ {module_path} - IMPORT FAILED: {e}")
        return False, None
    except Exception as e:
        print(f"❌ {module_path} - ERROR: {e}")
        return False, None

def main():
    """Test all dashboard implementations."""
    print("🔍 Dashboard Implementation Audit")
    print("=" * 50)
    
    dashboards_to_test = [
        ("dashboard.optimized_production_app", "OptimizedDashboard"),
        ("dashboard.enhanced_production_app", "EnhancedProductionDashboard"),
        ("dashboard.integrated_production_app", "IntegratedProductionDashboard"),
        ("dashboard.production_app", "ProductionBettingApp"),
        ("app", "main"),  # Special case for app.py
    ]
    
    working_dashboards = []
    
    for module_path, class_name in dashboards_to_test:
        print(f"\nTesting {module_path}...")
        
        if class_name == "main":
            # Special handling for app.py
            try:
                import app
                print(f"✅ {module_path} - IMPORT SUCCESS")
                working_dashboards.append((module_path, class_name, "main function"))
            except Exception as e:
                print(f"❌ {module_path} - IMPORT FAILED: {e}")
        else:
            success, dashboard = test_dashboard_import(module_path, class_name)
            if success:
                working_dashboards.append((module_path, class_name, dashboard))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DASHBOARD AUDIT SUMMARY")
    print("=" * 50)
    
    if working_dashboards:
        print(f"✅ Working Dashboards: {len(working_dashboards)}")
        for module_path, class_name, dashboard in working_dashboards:
            print(f"  - {module_path} ({class_name})")
    else:
        print("❌ No working dashboards found!")
    
    # Recommendation
    print("\n🎯 RECOMMENDATION:")
    if working_dashboards:
        primary = working_dashboards[0]  # First working one
        print(f"Primary Dashboard: {primary[0]} ({primary[1]})")
        
        if len(working_dashboards) > 1:
            print("\n🧹 CLEANUP NEEDED:")
            print("Multiple working dashboards found. Consider:")
            for i, (module_path, class_name, _) in enumerate(working_dashboards[1:], 1):
                print(f"  {i}. Archive or remove: {module_path}")
    
    return working_dashboards

if __name__ == "__main__":
    working_dashboards = main()
    
    # Exit with appropriate code
    sys.exit(0 if working_dashboards else 1)
