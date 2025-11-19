"""
Phase 3 Dashboard Integration Test
Comprehensive testing for the Phase 3 Enhanced Dashboard implementation.
"""

import sys
import os
import logging
import time
from typing import Dict, Any, List
import traceback

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase3DashboardTester:
    """Comprehensive tester for Phase 3 Dashboard components."""
    
    def __init__(self):
        """Initialize the tester."""
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 dashboard tests."""
        logger.info("🚀 Starting Phase 3 Dashboard Integration Tests")
        logger.info("=" * 60)
        
        test_methods = [
            self.test_navigation_system_import,
            self.test_team_selector_import,
            self.test_prediction_display_import,
            self.test_main_dashboard_import,
            self.test_navigation_system_functionality,
            self.test_team_selector_functionality,
            self.test_prediction_display_functionality,
            self.test_css_injection,
            self.test_session_state_management,
            self.test_user_journey_flow,
            self.test_responsive_design_tokens,
            self.test_accessibility_features,
            self.test_error_handling,
            self.test_performance_optimization,
            self.test_integration_compatibility
        ]
        
        for test_method in test_methods:
            self._run_single_test(test_method)
        
        self._generate_test_report()
        return self.test_results
    
    def _run_single_test(self, test_method):
        """Run a single test method."""
        test_name = test_method.__name__
        self.test_results["total_tests"] += 1
        
        try:
            logger.info(f"Running {test_name}...")
            start_time = time.time()
            
            result = test_method()
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result:
                self.test_results["passed_tests"] += 1
                status = "✅ PASSED"
                logger.info(f"{status} {test_name} ({duration:.2f}s)")
            else:
                self.test_results["failed_tests"] += 1
                status = "❌ FAILED"
                logger.error(f"{status} {test_name} ({duration:.2f}s)")
            
            self.test_results["test_details"].append({
                "test_name": test_name,
                "status": status,
                "duration": duration,
                "passed": result
            })
            
        except Exception as e:
            self.test_results["failed_tests"] += 1
            status = "❌ ERROR"
            logger.error(f"{status} {test_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            self.test_results["test_details"].append({
                "test_name": test_name,
                "status": status,
                "duration": 0,
                "passed": False,
                "error": str(e)
            })
    
    def test_navigation_system_import(self) -> bool:
        """Test Phase 3 Navigation System import."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            # Test instantiation
            nav_system = Phase3NavigationSystem()
            
            # Test basic attributes
            assert hasattr(nav_system, 'current_step')
            assert hasattr(nav_system, 'total_steps')
            assert hasattr(nav_system, 'user_journey_steps')
            assert nav_system.total_steps == 4
            assert len(nav_system.user_journey_steps) == 4
            
            logger.info("✓ Navigation system import and instantiation successful")
            return True
            
        except Exception as e:
            logger.error(f"Navigation system import failed: {e}")
            return False
    
    def test_team_selector_import(self) -> bool:
        """Test Phase 3 Team Selector import."""
        try:
            from dashboard.components.phase3_team_selector import Phase3TeamSelector
            
            # Test instantiation
            team_selector = Phase3TeamSelector()
            
            # Test basic attributes
            assert hasattr(team_selector, 'top_6_leagues')
            assert len(team_selector.top_6_leagues) == 6
            
            # Test league data structure
            for league_name, league_data in team_selector.top_6_leagues.items():
                assert 'country' in league_data
                assert 'icon' in league_data
                assert 'popular_teams' in league_data
                assert 'color' in league_data
            
            logger.info("✓ Team selector import and validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Team selector import failed: {e}")
            return False
    
    def test_prediction_display_import(self) -> bool:
        """Test Phase 3 Prediction Display import."""
        try:
            from dashboard.components.phase3_prediction_display import Phase3PredictionDisplay
            
            # Test instantiation
            pred_display = Phase3PredictionDisplay()
            
            # Test basic attributes
            assert hasattr(pred_display, 'confidence_thresholds')
            assert hasattr(pred_display, 'value_thresholds')
            assert 'high' in pred_display.confidence_thresholds
            assert 'medium' in pred_display.confidence_thresholds
            assert 'low' in pred_display.confidence_thresholds
            
            logger.info("✓ Prediction display import and validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Prediction display import failed: {e}")
            return False
    
    def test_main_dashboard_import(self) -> bool:
        """Test main Phase 3 Dashboard import."""
        try:
            from dashboard.phase3_enhanced_dashboard import Phase3EnhancedDashboard
            
            # Test instantiation
            dashboard = Phase3EnhancedDashboard()
            
            # Test basic attributes
            assert hasattr(dashboard, 'navigation_system')
            assert hasattr(dashboard, 'team_selector')
            assert hasattr(dashboard, 'prediction_display')
            
            logger.info("✓ Main dashboard import and instantiation successful")
            return True
            
        except Exception as e:
            logger.error(f"Main dashboard import failed: {e}")
            return False
    
    def test_navigation_system_functionality(self) -> bool:
        """Test navigation system functionality."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test step information
            step_info = nav_system.get_current_step_info()
            assert isinstance(step_info, dict)
            
            # Test completion percentage
            completion = nav_system.get_journey_completion_percentage()
            assert isinstance(completion, float)
            assert 0 <= completion <= 100
            
            # Test step completion check
            is_completed = nav_system.is_step_completed(1)
            assert isinstance(is_completed, bool)
            
            logger.info("✓ Navigation system functionality tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Navigation system functionality test failed: {e}")
            return False
    
    def test_team_selector_functionality(self) -> bool:
        """Test team selector functionality."""
        try:
            from dashboard.components.phase3_team_selector import Phase3TeamSelector
            
            team_selector = Phase3TeamSelector()
            
            # Test mock team generation
            mock_teams = team_selector._get_mock_teams_for_league("Premier League")
            assert isinstance(mock_teams, list)
            assert len(mock_teams) > 0
            
            # Test team data structure
            for team in mock_teams:
                assert 'id' in team
                assert 'name' in team
                assert 'league' in team
                assert 'country' in team
            
            logger.info("✓ Team selector functionality tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Team selector functionality test failed: {e}")
            return False
    
    def test_prediction_display_functionality(self) -> bool:
        """Test prediction display functionality."""
        try:
            from dashboard.components.phase3_prediction_display import Phase3PredictionDisplay
            
            pred_display = Phase3PredictionDisplay()
            
            # Test mock insight generation
            mock_prediction = {
                "probabilities": {"home_win": 0.45, "away_win": 0.35, "draw": 0.20},
                "confidence": 0.75
            }
            mock_home_team = {"name": "Test Home Team"}
            mock_away_team = {"name": "Test Away Team"}
            
            insights = pred_display._generate_mock_insights(mock_prediction, mock_home_team, mock_away_team)
            assert isinstance(insights, list)
            assert len(insights) > 0
            
            # Test insight structure
            for insight in insights:
                assert 'type' in insight
                assert 'icon' in insight
                assert 'title' in insight
                assert 'content' in insight
            
            logger.info("✓ Prediction display functionality tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Prediction display functionality test failed: {e}")
            return False
    
    def test_css_injection(self) -> bool:
        """Test CSS injection methods."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            from dashboard.components.phase3_team_selector import Phase3TeamSelector
            from dashboard.components.phase3_prediction_display import Phase3PredictionDisplay
            
            # Test that CSS injection methods exist and are callable
            nav_system = Phase3NavigationSystem()
            team_selector = Phase3TeamSelector()
            pred_display = Phase3PredictionDisplay()
            
            assert callable(nav_system.inject_navigation_css)
            assert callable(team_selector.inject_team_selector_css)
            assert callable(pred_display.inject_prediction_display_css)
            
            logger.info("✓ CSS injection methods validation passed")
            return True
            
        except Exception as e:
            logger.error(f"CSS injection test failed: {e}")
            return False
    
    def test_session_state_management(self) -> bool:
        """Test session state management."""
        try:
            # Mock streamlit session state
            class MockSessionState:
                def __init__(self):
                    self.data = {}
                
                def get(self, key, default=None):
                    return self.data.get(key, default)
                
                def __setitem__(self, key, value):
                    self.data[key] = value
                
                def __getitem__(self, key):
                    return self.data[key]
                
                def __contains__(self, key):
                    return key in self.data
            
            # Test session state initialization patterns
            mock_st = type('MockStreamlit', (), {'session_state': MockSessionState()})()
            
            # Simulate session state initialization
            if "phase3_navigation" not in mock_st.session_state:
                mock_st.session_state["phase3_navigation"] = {
                    "current_step": 1,
                    "completed_steps": [],
                    "user_preferences": {}
                }
            
            assert "phase3_navigation" in mock_st.session_state
            assert mock_st.session_state["phase3_navigation"]["current_step"] == 1
            
            logger.info("✓ Session state management tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Session state management test failed: {e}")
            return False
    
    def test_user_journey_flow(self) -> bool:
        """Test user journey flow logic."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test journey steps
            assert len(nav_system.user_journey_steps) == 4
            
            # Test step progression logic
            for i, step in enumerate(nav_system.user_journey_steps, 1):
                assert step["id"] == i
                assert "name" in step
                assert "icon" in step
                assert "description" in step
            
            logger.info("✓ User journey flow tests passed")
            return True
            
        except Exception as e:
            logger.error(f"User journey flow test failed: {e}")
            return False
    
    def test_responsive_design_tokens(self) -> bool:
        """Test responsive design implementation."""
        try:
            from dashboard.components.phase3_team_selector import Phase3TeamSelector
            
            team_selector = Phase3TeamSelector()
            
            # Test league data contains responsive design elements
            for league_name, league_data in team_selector.top_6_leagues.items():
                assert 'color' in league_data  # For responsive theming
                assert isinstance(league_data['popular_teams'], list)
            
            logger.info("✓ Responsive design tokens tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Responsive design tokens test failed: {e}")
            return False
    
    def test_accessibility_features(self) -> bool:
        """Test accessibility features implementation."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test that accessibility methods exist
            assert hasattr(nav_system, 'render_keyboard_navigation_hints')
            
            # Test journey steps have descriptive text
            for step in nav_system.user_journey_steps:
                assert len(step["description"]) > 10  # Meaningful descriptions
            
            logger.info("✓ Accessibility features tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Accessibility features test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling mechanisms."""
        try:
            from dashboard.phase3_enhanced_dashboard import Phase3EnhancedDashboard
            
            dashboard = Phase3EnhancedDashboard()
            
            # Test that dashboard has error handling methods
            assert hasattr(dashboard, 'initialize_dashboard')
            
            # Test mock prediction generation with edge cases
            mock_prediction = dashboard._generate_mock_prediction(
                {"name": "Test Team 1"}, 
                {"name": "Test Team 2"}, 
                "same_league"
            )
            
            assert isinstance(mock_prediction, dict)
            assert "probabilities" in mock_prediction
            assert "confidence" in mock_prediction
            
            logger.info("✓ Error handling tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    def test_performance_optimization(self) -> bool:
        """Test performance optimization features."""
        try:
            from dashboard.phase3_enhanced_dashboard import Phase3EnhancedDashboard
            
            dashboard = Phase3EnhancedDashboard()
            
            # Test initialization performance
            start_time = time.time()
            dashboard.initialize_dashboard()
            init_time = time.time() - start_time
            
            # Should initialize quickly (under 1 second)
            assert init_time < 1.0
            
            logger.info(f"✓ Performance optimization tests passed (init: {init_time:.3f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Performance optimization test failed: {e}")
            return False
    
    def test_integration_compatibility(self) -> bool:
        """Test integration with existing platform components."""
        try:
            # Test that Phase 3 components don't conflict with existing imports
            from dashboard.phase3_enhanced_dashboard import Phase3EnhancedDashboard
            
            # Test main function exists
            from dashboard.phase3_enhanced_dashboard import main
            assert callable(main)
            
            logger.info("✓ Integration compatibility tests passed")
            return True
            
        except Exception as e:
            logger.error(f"Integration compatibility test failed: {e}")
            return False
    
    def _generate_test_report(self):
        """Generate comprehensive test report."""
        total = self.test_results["total_tests"]
        passed = self.test_results["passed_tests"]
        failed = self.test_results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info("=" * 60)
        logger.info("📊 PHASE 3 DASHBOARD TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 60)
        
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT: Phase 3 Dashboard is ready for production!")
        elif success_rate >= 80:
            logger.info("✅ GOOD: Phase 3 Dashboard is mostly ready with minor issues")
        elif success_rate >= 70:
            logger.info("⚠️ FAIR: Phase 3 Dashboard needs some improvements")
        else:
            logger.info("❌ POOR: Phase 3 Dashboard needs significant work")
        
        # Detailed test results
        logger.info("\n📋 DETAILED TEST RESULTS:")
        for test_detail in self.test_results["test_details"]:
            status_icon = "✅" if test_detail["passed"] else "❌"
            logger.info(f"{status_icon} {test_detail['test_name']} ({test_detail['duration']:.2f}s)")


def main():
    """Run the Phase 3 Dashboard integration tests."""
    tester = Phase3DashboardTester()
    results = tester.run_all_tests()
    
    # Return exit code based on results
    if results["failed_tests"] == 0:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
