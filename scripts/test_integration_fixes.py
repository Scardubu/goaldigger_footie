"""
Test Integration Fixes - Verify that the specific integration issues have been resolved.
"""

import logging
import sys
import os

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

def test_navigation_system_get_current_step():
    """Test that get_current_step method exists and works."""
    try:
        from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
        
        nav_system = Phase3NavigationSystem()
        
        # Test get_current_step method
        current_step = nav_system.get_current_step()
        assert isinstance(current_step, int)
        assert 1 <= current_step <= 4
        
        logger.info("✅ Navigation system get_current_step method works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Navigation system get_current_step test failed: {e}")
        return False

def test_integration_adapter_generate_mock_prediction():
    """Test that generate_mock_prediction method exists and works."""
    try:
        from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
        
        adapter = Phase3IntegrationAdapter()
        
        # Test generate_mock_prediction method
        prediction = adapter.generate_mock_prediction("Manchester City", "Arsenal")
        assert isinstance(prediction, dict)
        assert "probabilities" in prediction
        assert "confidence" in prediction
        assert "source" in prediction
        
        logger.info("✅ Integration adapter generate_mock_prediction method works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration adapter generate_mock_prediction test failed: {e}")
        return False

def test_integration_adapter_get_fallback_team_data():
    """Test that get_fallback_team_data method exists and works."""
    try:
        from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
        
        adapter = Phase3IntegrationAdapter()
        
        # Test get_fallback_team_data method
        team_data = adapter.get_fallback_team_data("Premier League")
        assert isinstance(team_data, list)
        assert len(team_data) > 0
        
        logger.info("✅ Integration adapter get_fallback_team_data method works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration adapter get_fallback_team_data test failed: {e}")
        return False

def test_html_validation_security():
    """Test that HTML validation correctly detects malicious content."""
    try:
        from dashboard.components.phase3_html_renderer import html_renderer
        
        # Test malicious inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(1)'></iframe>"
        ]
        
        for malicious_input in malicious_inputs:
            is_safe = html_renderer.validate_html_content(malicious_input)
            assert is_safe == False, f"Should detect {malicious_input} as unsafe"
            
            # Test escaping
            escaped = html_renderer.escape_user_content(malicious_input)
            assert "<script>" not in escaped, f"Should escape script tags in {malicious_input}"
        
        logger.info("✅ HTML validation security works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ HTML validation security test failed: {e}")
        return False

def test_data_processor_integration():
    """Test that data processor integration works."""
    try:
        from data.processors.data_processor import DataProcessor
        
        processor = DataProcessor()
        assert processor.initialized == True
        
        # Test team data processing
        mock_team = {"name": "Test Team", "league": "Test League"}
        processed = processor.process_team_data(mock_team)
        assert isinstance(processed, dict)
        assert "name" in processed
        
        logger.info("✅ Data processor integration works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data processor integration test failed: {e}")
        return False

def test_enhanced_scraper_integration():
    """Test that enhanced scraper integration works."""
    try:
        from data.scrapers.enhanced_scraper import EnhancedScraper
        
        scraper = EnhancedScraper()
        assert scraper.initialized == True
        
        # Test scraping stats
        stats = scraper.get_scraping_stats()
        assert isinstance(stats, dict)
        assert "initialized" in stats
        
        logger.info("✅ Enhanced scraper integration works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced scraper integration test failed: {e}")
        return False

def test_integration_adapter_full_initialization():
    """Test that integration adapter can fully initialize with new components."""
    try:
        from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
        
        adapter = Phase3IntegrationAdapter()
        integration_status = adapter.initialize_integrations()
        
        # Check that more components are now available
        available_components = sum(integration_status.values())
        logger.info(f"Integration status: {integration_status}")
        logger.info(f"Available components: {available_components}/5")
        
        # Should have at least 4/5 components now (was 3/5 before)
        assert available_components >= 4, f"Expected at least 4 components, got {available_components}"
        
        logger.info("✅ Integration adapter full initialization works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration adapter full initialization test failed: {e}")
        return False

def main():
    """Run all integration fix tests."""
    logger.info("🧪 Testing Integration Fixes...")
    logger.info("=" * 60)
    
    tests = [
        ("Navigation System get_current_step", test_navigation_system_get_current_step),
        ("Integration Adapter generate_mock_prediction", test_integration_adapter_generate_mock_prediction),
        ("Integration Adapter get_fallback_team_data", test_integration_adapter_get_fallback_team_data),
        ("HTML Validation Security", test_html_validation_security),
        ("Data Processor Integration", test_data_processor_integration),
        ("Enhanced Scraper Integration", test_enhanced_scraper_integration),
        ("Integration Adapter Full Initialization", test_integration_adapter_full_initialization)
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
    logger.info(f"📊 INTEGRATION FIX TEST RESULTS")
    logger.info(f"Passed: {passed_tests}/{total_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL INTEGRATION FIXES SUCCESSFUL!")
        return True
    else:
        logger.error(f"⚠️ {total_tests - passed_tests} integration issues remain")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
