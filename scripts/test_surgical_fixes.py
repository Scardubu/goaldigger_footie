"""
Test Surgical Fixes - Verify that the specific failing tests are now resolved.
"""

import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_graceful_degradation_fix():
    """Test that the graceful degradation issue is fixed."""
    try:
        from dashboard.components.phase3_integration_adapter import \
            Phase3IntegrationAdapter
        
        adapter = Phase3IntegrationAdapter()
        
        # Test integration status
        status = adapter.get_integration_status()
        assert isinstance(status, dict)
        
        # Test fallback mechanisms
        mock_prediction = adapter.generate_mock_prediction("Test Home", "Test Away")
        assert isinstance(mock_prediction, dict)
        assert "probabilities" in mock_prediction
        
        # Test that the missing attribute is now present
        assert adapter.integration_adapter is not None
        assert adapter.integration_adapter == adapter  # Should be self-reference
        
        logger.info("✅ Graceful degradation fix validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Graceful degradation fix test failed: {e}")
        return False

def test_component_isolation_fix():
    """Test that the component isolation issue is fixed."""
    try:
        from dashboard.components.phase3_navigation_system import \
            Phase3NavigationSystem
        from dashboard.components.phase3_prediction_display import \
            Phase3PredictionDisplay
        from dashboard.components.phase3_team_selector import \
            Phase3TeamSelector

        # Test that components can be created independently with unique instances
        nav1 = Phase3NavigationSystem(instance_id="test1")
        nav2 = Phase3NavigationSystem(instance_id="test2")
        
        team1 = Phase3TeamSelector()
        team2 = Phase3TeamSelector()
        
        pred1 = Phase3PredictionDisplay()
        pred2 = Phase3PredictionDisplay()
        
        # Test that navigation instances don't interfere with each other
        logger.info(f"Before: nav1 step = {nav1.get_current_step()}, nav2 step = {nav2.get_current_step()}")
        nav1.go_to_step(1)
        logger.info(f"After nav1.go_to_step(1): nav1 step = {nav1.get_current_step()}, nav2 step = {nav2.get_current_step()}")
        nav2.go_to_step(3)
        logger.info(f"After nav2.go_to_step(3): nav1 step = {nav1.get_current_step()}, nav2 step = {nav2.get_current_step()}")

        # Debug: Check what values we actually get
        step1 = nav1.get_current_step()
        step2 = nav2.get_current_step()
        logger.info(f"Debug: nav1 (id={nav1.instance_id}) step = {step1}, nav2 (id={nav2.instance_id}) step = {step2}")
        logger.info(f"Debug: nav1 session_key = {nav1.session_key}, nav2 session_key = {nav2.session_key}")
        logger.info(f"Debug: Fallback states keys = {list(nav1._fallback_states.keys())}")

        # Verify isolation
        assert nav1.get_current_step() == 1, f"Expected nav1 step to be 1, got {nav1.get_current_step()}"
        assert nav2.get_current_step() == 3, f"Expected nav2 step to be 3, got {nav2.get_current_step()}"
        
        # Test that they have different instance IDs
        assert nav1.instance_id != nav2.instance_id
        assert nav1.session_key != nav2.session_key
        
        logger.info("✅ Component isolation fix validated")
        return True
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Component isolation fix test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_integration_adapter_attributes():
    """Test that all expected attributes are present in integration adapter."""
    try:
        from dashboard.components.phase3_integration_adapter import \
            Phase3IntegrationAdapter
        
        adapter = Phase3IntegrationAdapter()
        
        # Test all expected attributes
        expected_attributes = [
            'prediction_engine',
            'data_loader', 
            'config',
            'database',
            'scrapers',
            'initialized',
            'integration_adapter',
            'integration_status'
        ]
        
        for attr in expected_attributes:
            assert hasattr(adapter, attr), f"Missing attribute: {attr}"
        
        # Test that integration_adapter is self-reference
        assert adapter.integration_adapter is adapter
        
        logger.info("✅ Integration adapter attributes validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration adapter attributes test failed: {e}")
        return False

def test_navigation_backward_compatibility():
    """Test that navigation system maintains backward compatibility."""
    try:
        from dashboard.components.phase3_navigation_system import \
            Phase3NavigationSystem

        # Test default instance (backward compatibility)
        nav_default = Phase3NavigationSystem()
        
        # Test that it still works with default session state
        nav_default.go_to_step(2)
        assert nav_default.get_current_step() == 2
        
        # Test instance-specific navigation
        nav_specific = Phase3NavigationSystem(instance_id="specific")
        nav_specific.go_to_step(4)
        assert nav_specific.get_current_step() == 4
        
        # Test that they don't interfere
        assert nav_default.get_current_step() == 2
        assert nav_specific.get_current_step() == 4
        
        logger.info("✅ Navigation backward compatibility validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Navigation backward compatibility test failed: {e}")
        return False

def test_integration_status_completeness():
    """Test that integration status is complete."""
    try:
        from dashboard.components.phase3_integration_adapter import \
            Phase3IntegrationAdapter
        
        adapter = Phase3IntegrationAdapter()
        integration_status = adapter.initialize_integrations()
        
        # Should have all 5 components
        expected_components = [
            'prediction_engine',
            'data_loader',
            'config', 
            'database',
            'scrapers'
        ]
        
        for component in expected_components:
            assert component in integration_status, f"Missing component: {component}"
        
        # Should have high integration success
        available_components = sum(integration_status.values())
        assert available_components >= 4, f"Expected at least 4 components, got {available_components}"
        
        logger.info(f"✅ Integration status completeness validated ({available_components}/5 components)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration status completeness test failed: {e}")
        return False

def main():
    """Run all surgical fix tests."""
    logger.info("🔧 Testing Surgical Fixes...")
    logger.info("=" * 60)
    
    tests = [
        ("Graceful Degradation Fix", test_graceful_degradation_fix),
        ("Component Isolation Fix", test_component_isolation_fix),
        ("Integration Adapter Attributes", test_integration_adapter_attributes),
        ("Navigation Backward Compatibility", test_navigation_backward_compatibility),
        ("Integration Status Completeness", test_integration_status_completeness)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        if test_func():
            passed_tests += 1
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"📊 SURGICAL FIX TEST RESULTS")
    logger.info(f"Passed: {passed_tests}/{total_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL SURGICAL FIXES SUCCESSFUL!")
        return True
    else:
        logger.error(f"⚠️ {total_tests - passed_tests} surgical fixes need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
