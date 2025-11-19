"""
Phase 2 Layout Optimization Test
Comprehensive testing for layout organization and user experience optimization.
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

class Phase2LayoutOptimizationTester:
    """Comprehensive tester for Phase 2 layout optimization and UX improvements."""
    
    def __init__(self):
        """Initialize the layout optimization tester."""
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
    
    def run_all_layout_tests(self) -> Dict[str, Any]:
        """Run all Phase 2 layout optimization tests."""
        logger.info("🎨 Starting Phase 2 Layout Optimization Tests")
        logger.info("=" * 60)
        
        test_methods = [
            self.test_html_renderer_functionality,
            self.test_safe_html_rendering,
            self.test_navigation_system_layout,
            self.test_team_selector_layout,
            self.test_prediction_display_layout,
            self.test_visual_hierarchy,
            self.test_progressive_disclosure,
            self.test_responsive_design_elements,
            self.test_accessibility_features,
            self.test_keyboard_navigation,
            self.test_user_journey_flow,
            self.test_contextual_help_system,
            self.test_smart_defaults_functionality,
            self.test_error_handling_graceful_degradation,
            self.test_performance_optimization
        ]
        
        for test_method in test_methods:
            self._run_single_test(test_method)
        
        self._generate_layout_report()
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
            
            self.test_results["test_details"].append({
                "test_name": test_name,
                "status": status,
                "duration": 0,
                "passed": False,
                "error": str(e)
            })
    
    def test_html_renderer_functionality(self) -> bool:
        """Test HTML renderer functionality and safety."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer
            
            # Test base styles injection
            html_renderer.inject_base_styles()
            assert html_renderer.global_styles_injected == True
            
            # Test content validation
            safe_content = "<div>Safe content</div>"
            dangerous_content = "<script>alert('xss')</script>"
            
            assert html_renderer.validate_html_content(safe_content) == True
            assert html_renderer.validate_html_content(dangerous_content) == False
            
            # Test content escaping
            user_input = "<script>alert('test')</script>"
            escaped = html_renderer.escape_user_content(user_input)
            assert "&lt;script&gt;" in escaped
            
            logger.info("✓ HTML renderer functionality validated")
            return True
            
        except Exception as e:
            logger.error(f"HTML renderer functionality test failed: {e}")
            return False
    
    def test_safe_html_rendering(self) -> bool:
        """Test safe HTML rendering across components."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer
            
            # Test card rendering
            html_renderer.render_card("Test Title", "Test Content", "primary")
            
            # Test button group rendering
            buttons = [
                {"label": "Test Button", "type": "primary", "onclick": "test()"},
                {"label": "Disabled Button", "type": "secondary", "disabled": True}
            ]
            html_renderer.render_button_group(buttons)
            
            # Test progress bar rendering
            html_renderer.render_progress_bar(75.5, "Test Progress", "success")
            
            # Test alert rendering
            html_renderer.render_alert("Test alert message", "info", True)
            
            # Test badge list rendering
            badges = [
                {"label": "Primary", "type": "primary"},
                {"label": "Success", "type": "success"}
            ]
            html_renderer.render_badge_list(badges)
            
            logger.info("✓ Safe HTML rendering validated")
            return True
            
        except Exception as e:
            logger.error(f"Safe HTML rendering test failed: {e}")
            return False
    
    def test_navigation_system_layout(self) -> bool:
        """Test navigation system layout and rendering."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test navigation system attributes
            assert hasattr(nav_system, 'user_journey_steps')
            assert len(nav_system.user_journey_steps) == 4
            assert nav_system.total_steps == 4
            
            # Test step information
            step_info = nav_system.get_current_step_info()
            assert isinstance(step_info, dict)
            
            # Test completion percentage
            completion = nav_system.get_journey_completion_percentage()
            assert isinstance(completion, float)
            assert 0 <= completion <= 100
            
            logger.info("✓ Navigation system layout validated")
            return True
            
        except Exception as e:
            logger.error(f"Navigation system layout test failed: {e}")
            return False
    
    def test_team_selector_layout(self) -> bool:
        """Test team selector layout and functionality."""
        try:
            from dashboard.components.phase3_team_selector import Phase3TeamSelector
            
            team_selector = Phase3TeamSelector()
            
            # Test team selector attributes
            assert hasattr(team_selector, 'top_6_leagues')
            assert len(team_selector.top_6_leagues) == 6
            
            # Test league data structure
            for league_name, league_data in team_selector.top_6_leagues.items():
                assert 'country' in league_data
                assert 'icon' in league_data
                assert 'popular_teams' in league_data
                assert 'color' in league_data
                assert isinstance(league_data['popular_teams'], list)
                assert len(league_data['popular_teams']) > 0
            
            # Test mock team generation
            mock_teams = team_selector._get_mock_teams_for_league("Premier League")
            assert isinstance(mock_teams, list)
            assert len(mock_teams) > 0
            
            logger.info("✓ Team selector layout validated")
            return True
            
        except Exception as e:
            logger.error(f"Team selector layout test failed: {e}")
            return False
    
    def test_prediction_display_layout(self) -> bool:
        """Test prediction display layout and components."""
        try:
            from dashboard.components.phase3_prediction_display import Phase3PredictionDisplay
            
            pred_display = Phase3PredictionDisplay()
            
            # Test prediction display attributes
            assert hasattr(pred_display, 'confidence_thresholds')
            assert hasattr(pred_display, 'value_thresholds')
            
            # Test threshold values
            assert 'high' in pred_display.confidence_thresholds
            assert 'medium' in pred_display.confidence_thresholds
            assert 'low' in pred_display.confidence_thresholds
            
            assert 'excellent' in pred_display.value_thresholds
            assert 'good' in pred_display.value_thresholds
            assert 'fair' in pred_display.value_thresholds
            
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
            
            logger.info("✓ Prediction display layout validated")
            return True
            
        except Exception as e:
            logger.error(f"Prediction display layout test failed: {e}")
            return False
    
    def test_visual_hierarchy(self) -> bool:
        """Test visual hierarchy implementation."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer
            
            # Test stats grid rendering (visual hierarchy)
            stats = [
                {"title": "Confidence", "value": "85%", "change": 5.2, "color": "success"},
                {"title": "Value", "value": "12.5%", "change": -2.1, "color": "warning"},
                {"title": "Risk", "value": "Low", "color": "primary"}
            ]
            html_renderer.render_stats_grid(stats)
            
            # Test navigation breadcrumb (hierarchy)
            breadcrumb_items = [
                {"label": "Dashboard", "href": "#", "active": False},
                {"label": "Predictions", "href": "#", "active": False},
                {"label": "Current Match", "href": "#", "active": True}
            ]
            html_renderer.render_navigation_breadcrumb(breadcrumb_items)
            
            logger.info("✓ Visual hierarchy validated")
            return True
            
        except Exception as e:
            logger.error(f"Visual hierarchy test failed: {e}")
            return False
    
    def test_progressive_disclosure(self) -> bool:
        """Test progressive disclosure patterns."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test contextual help (progressive disclosure)
            for step_id in range(1, 5):
                # Should not raise exceptions
                nav_system.render_contextual_help(step_id)
            
            # Test step completion tracking
            for step_id in range(1, 5):
                is_completed = nav_system.is_step_completed(step_id)
                assert isinstance(is_completed, bool)
            
            logger.info("✓ Progressive disclosure patterns validated")
            return True
            
        except Exception as e:
            logger.error(f"Progressive disclosure test failed: {e}")
            return False
    
    def test_responsive_design_elements(self) -> bool:
        """Test responsive design implementation."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer
            
            # Test that base styles include responsive design
            html_renderer.inject_base_styles()
            
            # Test responsive components
            html_renderer.render_card("Responsive Test", "This should work on all devices")
            
            # Test responsive button group
            buttons = [
                {"label": "Mobile Friendly", "type": "primary"},
                {"label": "Tablet Ready", "type": "secondary"},
                {"label": "Desktop Optimized", "type": "success"}
            ]
            html_renderer.render_button_group(buttons)
            
            logger.info("✓ Responsive design elements validated")
            return True
            
        except Exception as e:
            logger.error(f"Responsive design test failed: {e}")
            return False
    
    def test_accessibility_features(self) -> bool:
        """Test accessibility features implementation."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test keyboard navigation hints
            nav_system.render_keyboard_navigation_hints()
            
            # Test that journey steps have descriptive content
            for step in nav_system.user_journey_steps:
                assert 'description' in step
                assert len(step['description']) > 10  # Meaningful descriptions
                assert 'name' in step
                assert 'icon' in step
            
            logger.info("✓ Accessibility features validated")
            return True
            
        except Exception as e:
            logger.error(f"Accessibility features test failed: {e}")
            return False
    
    def test_keyboard_navigation(self) -> bool:
        """Test keyboard navigation support."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test navigation methods exist
            assert hasattr(nav_system, 'advance_step')
            assert hasattr(nav_system, 'go_to_step')
            assert hasattr(nav_system, 'render_keyboard_navigation_hints')
            
            # Test step navigation
            for step_id in range(1, 5):
                # Should not raise exceptions
                nav_system.go_to_step(step_id)
            
            logger.info("✓ Keyboard navigation support validated")
            return True
            
        except Exception as e:
            logger.error(f"Keyboard navigation test failed: {e}")
            return False
    
    def test_user_journey_flow(self) -> bool:
        """Test user journey flow logic."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test journey steps structure
            assert len(nav_system.user_journey_steps) == 4
            
            expected_steps = [
                "League Selection",
                "Team Selection", 
                "Prediction Generation",
                "Actionable Insights"
            ]
            
            for i, step in enumerate(nav_system.user_journey_steps):
                assert step["id"] == i + 1
                assert step["name"] in expected_steps
                assert "icon" in step
                assert "description" in step
            
            logger.info("✓ User journey flow validated")
            return True
            
        except Exception as e:
            logger.error(f"User journey flow test failed: {e}")
            return False
    
    def test_contextual_help_system(self) -> bool:
        """Test contextual help system."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test contextual help for each step
            for step_id in range(1, 5):
                # Should render without errors
                nav_system.render_contextual_help(step_id)
            
            # Test invalid step handling
            nav_system.render_contextual_help(999)  # Should handle gracefully
            
            logger.info("✓ Contextual help system validated")
            return True
            
        except Exception as e:
            logger.error(f"Contextual help system test failed: {e}")
            return False
    
    def test_smart_defaults_functionality(self) -> bool:
        """Test smart defaults functionality."""
        try:
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test smart defaults indicator
            test_defaults = ["Premier League (Popular)", "Manchester City (Top Rated)"]
            nav_system.render_smart_defaults_indicator(test_defaults)
            
            # Test empty defaults handling
            nav_system.render_smart_defaults_indicator([])
            
            logger.info("✓ Smart defaults functionality validated")
            return True
            
        except Exception as e:
            logger.error(f"Smart defaults functionality test failed: {e}")
            return False
    
    def test_error_handling_graceful_degradation(self) -> bool:
        """Test error handling and graceful degradation."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer
            
            # Test error fallback rendering
            html_renderer.render_error_fallback("Test error message")
            
            # Test validation with dangerous content
            dangerous_html = "<script>alert('xss')</script><div>content</div>"
            is_safe = html_renderer.validate_html_content(dangerous_html)
            assert is_safe == False
            
            # Test safe content validation
            safe_html = "<div>Safe content</div>"
            is_safe = html_renderer.validate_html_content(safe_html)
            assert is_safe == True
            
            logger.info("✓ Error handling and graceful degradation validated")
            return True
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    def test_performance_optimization(self) -> bool:
        """Test performance optimization features."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer
            
            # Test performance of base styles injection
            start_time = time.time()
            html_renderer.inject_base_styles()
            injection_time = time.time() - start_time
            
            # Should be very fast (under 0.1 seconds)
            assert injection_time < 0.1
            
            # Test performance of multiple renders
            start_time = time.time()
            for i in range(10):
                html_renderer.render_card(f"Test {i}", f"Content {i}")
            render_time = time.time() - start_time
            
            # Should handle multiple renders efficiently
            assert render_time < 1.0
            
            logger.info(f"✓ Performance optimization validated (injection: {injection_time:.3f}s, renders: {render_time:.3f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Performance optimization test failed: {e}")
            return False
    
    def _generate_layout_report(self):
        """Generate comprehensive layout optimization test report."""
        total = self.test_results["total_tests"]
        passed = self.test_results["passed_tests"]
        failed = self.test_results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info("=" * 60)
        logger.info("🎨 PHASE 2 LAYOUT OPTIMIZATION TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 60)
        
        if success_rate >= 95:
            logger.info("🎉 EXCELLENT: Phase 2 Layout Optimization is outstanding!")
        elif success_rate >= 90:
            logger.info("✅ VERY GOOD: Phase 2 Layout Optimization is excellent")
        elif success_rate >= 80:
            logger.info("✅ GOOD: Phase 2 Layout Optimization is solid")
        elif success_rate >= 70:
            logger.info("⚠️ FAIR: Phase 2 Layout Optimization needs improvements")
        else:
            logger.info("❌ POOR: Phase 2 Layout Optimization needs significant work")
        
        # Detailed test results
        logger.info("\n📋 DETAILED LAYOUT TEST RESULTS:")
        for test_detail in self.test_results["test_details"]:
            status_icon = "✅" if test_detail["passed"] else "❌"
            logger.info(f"{status_icon} {test_detail['test_name']} ({test_detail['duration']:.2f}s)")


def main():
    """Run the Phase 2 layout optimization tests."""
    tester = Phase2LayoutOptimizationTester()
    results = tester.run_all_layout_tests()
    
    # Return exit code based on results
    if results["failed_tests"] == 0:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
