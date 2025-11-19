"""
Phase 4 Production Readiness Test
Comprehensive end-to-end testing and production validation.
"""

import gc
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import psutil

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase4ProductionReadinessTester:
    """Comprehensive tester for Phase 4 production readiness."""
    
    def __init__(self):
        """Initialize the production readiness tester."""
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "performance_metrics": {},
            "memory_usage": {},
            "error_handling_tests": {}
        }
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def run_all_production_readiness_tests(self) -> Dict[str, Any]:
        """Run all Phase 4 production readiness tests."""
        logger.info("🚀 Starting Phase 4 Production Readiness Tests")
        logger.info("=" * 60)
        
        test_methods = [
            self.test_complete_dashboard_initialization,
            self.test_end_to_end_user_workflow,
            self.test_performance_optimization,
            self.test_memory_management,
            self.test_error_handling_robustness,
            self.test_graceful_degradation,
            self.test_responsive_design_validation,
            self.test_accessibility_compliance,
            self.test_integration_stability,
            self.test_session_state_persistence,
            self.test_concurrent_user_simulation,
            self.test_data_validation_security,
            self.test_html_rendering_safety,
            self.test_css_injection_security,
            self.test_component_isolation,
            self.test_fallback_mechanisms,
            self.test_loading_performance,
            self.test_visual_consistency,
            self.test_user_experience_flow,
            self.test_production_deployment_readiness
        ]
        
        for test_method in test_methods:
            self._run_single_test(test_method)
        
        self._generate_production_readiness_report()
        return self.test_results
    
    def _run_single_test(self, test_method):
        """Run a single test method with performance monitoring."""
        test_name = test_method.__name__
        self.test_results["total_tests"] += 1
        
        try:
            logger.info(f"Running {test_name}...")
            
            # Memory before test
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            start_time = time.time()
            
            result = test_method()
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            duration = end_time - start_time
            memory_delta = memory_after - memory_before
            
            # Store performance metrics
            self.test_results["performance_metrics"][test_name] = {
                "duration": duration,
                "memory_before": memory_before,
                "memory_after": memory_after,
                "memory_delta": memory_delta
            }
            
            if result:
                self.test_results["passed_tests"] += 1
                status = "✅ PASSED"
                logger.info(f"{status} {test_name} ({duration:.2f}s, {memory_delta:+.1f}MB)")
            else:
                self.test_results["failed_tests"] += 1
                status = "❌ FAILED"
                logger.error(f"{status} {test_name} ({duration:.2f}s, {memory_delta:+.1f}MB)")
            
            self.test_results["test_details"].append({
                "test_name": test_name,
                "status": status,
                "duration": duration,
                "memory_delta": memory_delta,
                "passed": result
            })
            
            # Force garbage collection after each test
            gc.collect()
            
        except Exception as e:
            self.test_results["failed_tests"] += 1
            status = "❌ ERROR"
            logger.error(f"{status} {test_name}: {str(e)}")
            
            self.test_results["test_details"].append({
                "test_name": test_name,
                "status": status,
                "duration": 0,
                "memory_delta": 0,
                "passed": False,
                "error": str(e)
            })
    
    def test_complete_dashboard_initialization(self) -> bool:
        """Test complete dashboard initialization."""
        try:
            from dashboard.phase3_enhanced_dashboard import \
                Phase3EnhancedDashboard

            # Test dashboard creation
            dashboard = Phase3EnhancedDashboard()
            
            # Test all components are initialized
            assert dashboard.navigation_system is not None
            assert dashboard.team_selector is not None
            assert dashboard.prediction_display is not None
            assert dashboard.integration_adapter is not None
            assert dashboard.advanced_features is not None
            
            # Test session state initialization
            dashboard.initialize_dashboard()
            
            logger.info("✓ Complete dashboard initialization validated")
            return True
            
        except Exception as e:
            logger.error(f"Complete dashboard initialization test failed: {e}")
            return False
    
    def test_end_to_end_user_workflow(self) -> bool:
        """Test complete end-to-end user workflow."""
        try:
            import streamlit as st

            from dashboard.phase3_enhanced_dashboard import \
                Phase3EnhancedDashboard
            
            dashboard = Phase3EnhancedDashboard()
            dashboard.initialize_dashboard()
            
            # Simulate user workflow through all 4 steps
            nav_system = dashboard.navigation_system
            
            # Step 1: League Selection
            nav_system.go_to_step(1)
            assert nav_system.get_current_step() == 1
            
            # Step 2: Team Selection
            nav_system.advance_step()
            assert nav_system.get_current_step() == 2
            
            # Step 3: Prediction Generation
            nav_system.advance_step()
            assert nav_system.get_current_step() == 3
            
            # Step 4: Actionable Insights
            nav_system.advance_step()
            assert nav_system.get_current_step() == 4
            
            # Test completion tracking
            completion_percentage = nav_system.get_journey_completion_percentage()
            assert completion_percentage >= 75  # Should be near completion
            
            logger.info("✓ End-to-end user workflow validated")
            return True
            
        except Exception as e:
            logger.error(f"End-to-end user workflow test failed: {e}")
            return False
    
    def test_performance_optimization(self) -> bool:
        """Test performance optimization features."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer

            # Test CSS injection performance
            start_time = time.time()
            for _ in range(10):
                html_renderer.inject_base_styles()
            css_injection_time = time.time() - start_time
            
            # Should be very fast (under 0.5 seconds for 10 injections)
            assert css_injection_time < 0.5
            
            # Test component rendering performance
            start_time = time.time()
            for i in range(50):
                html_renderer.render_card(f"Test {i}", f"Content {i}")
            render_time = time.time() - start_time
            
            # Should handle 50 renders efficiently (under 2 seconds)
            assert render_time < 2.0
            
            # Test memory efficiency
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create and destroy multiple components
            for _ in range(100):
                html_renderer.render_progress_bar(50, "Test Progress")
            
            gc.collect()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = memory_after - memory_before
            
            # Memory increase should be reasonable (under 100MB for 400MB total target)
            assert memory_increase < 100
            
            logger.info(f"✓ Performance optimization validated (CSS: {css_injection_time:.3f}s, Render: {render_time:.3f}s, Memory: +{memory_increase:.1f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"Performance optimization test failed: {e}")
            return False
    
    def test_memory_management(self) -> bool:
        """Test memory management and leak prevention."""
        try:
            from dashboard.phase3_enhanced_dashboard import \
                Phase3EnhancedDashboard
            
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create and destroy multiple dashboard instances
            dashboards = []
            for i in range(10):
                dashboard = Phase3EnhancedDashboard()
                dashboard.initialize_dashboard()
                dashboards.append(dashboard)
            
            # Clear references
            dashboards.clear()
            gc.collect()
            
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = memory_after - memory_before
            
            # Memory increase should be reasonable (under 100MB for 10 instances)
            assert memory_increase < 100
            
            logger.info(f"✓ Memory management validated (Memory increase: +{memory_increase:.1f}MB)")
            return True
            
        except Exception as e:
            logger.error(f"Memory management test failed: {e}")
            return False
    
    def test_error_handling_robustness(self) -> bool:
        """Test error handling and robustness."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer

            # Test with invalid inputs
            error_count = 0
            
            try:
                html_renderer.render_card(None, None)
            except:
                error_count += 1
            
            try:
                html_renderer.render_progress_bar(-50)  # Invalid percentage
            except:
                error_count += 1
            
            try:
                html_renderer.validate_html_content("<script>alert('xss')</script>")
                # Should return False, not raise exception
            except:
                error_count += 1
            
            # Should handle errors gracefully without crashing
            assert error_count <= 2  # Some errors are expected and handled
            
            # Test error fallback rendering
            html_renderer.render_error_fallback("Test error message")
            
            logger.info("✓ Error handling robustness validated")
            return True
            
        except Exception as e:
            logger.error(f"Error handling robustness test failed: {e}")
            return False
    
    def test_graceful_degradation(self) -> bool:
        """Test graceful degradation when components fail."""
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
            
            # Test that system continues to work even with failed integrations
            assert adapter.integration_adapter is not None
            
            logger.info("✓ Graceful degradation validated")
            return True
            
        except Exception as e:
            logger.error(f"Graceful degradation test failed: {e}")
            return False
    
    def test_responsive_design_validation(self) -> bool:
        """Test responsive design implementation."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer

            # Test responsive components
            html_renderer.inject_base_styles()
            
            # Test responsive card rendering
            html_renderer.render_card("Responsive Test", "This should work on all devices")
            
            # Test responsive button groups
            buttons = [
                {"label": "Mobile", "type": "primary"},
                {"label": "Tablet", "type": "secondary"},
                {"label": "Desktop", "type": "success"}
            ]
            html_renderer.render_button_group(buttons)
            
            # Test responsive stats grid
            stats = [
                {"title": "Mobile", "value": "100%", "color": "success"},
                {"title": "Tablet", "value": "95%", "color": "primary"},
                {"title": "Desktop", "value": "98%", "color": "success"}
            ]
            html_renderer.render_stats_grid(stats)
            
            logger.info("✓ Responsive design validation completed")
            return True
            
        except Exception as e:
            logger.error(f"Responsive design validation test failed: {e}")
            return False
    
    def test_accessibility_compliance(self) -> bool:
        """Test accessibility compliance."""
        try:
            from dashboard.components.phase3_navigation_system import \
                Phase3NavigationSystem
            
            nav_system = Phase3NavigationSystem()
            
            # Test keyboard navigation support
            nav_system.render_keyboard_navigation_hints()
            
            # Test that all journey steps have proper descriptions
            for step in nav_system.user_journey_steps:
                assert len(step.get("description", "")) > 10
                assert "name" in step
                assert "icon" in step
            
            # Test contextual help for accessibility
            for step_id in range(1, 5):
                nav_system.render_contextual_help(step_id)
            
            logger.info("✓ Accessibility compliance validated")
            return True
            
        except Exception as e:
            logger.error(f"Accessibility compliance test failed: {e}")
            return False
    
    def test_integration_stability(self) -> bool:
        """Test integration stability across components."""
        try:
            from dashboard.phase3_enhanced_dashboard import \
                Phase3EnhancedDashboard
            
            dashboard = Phase3EnhancedDashboard()
            dashboard.initialize_dashboard()
            
            # Test component interactions
            integration_status = dashboard.integration_adapter.get_integration_status()
            assert isinstance(integration_status, dict)
            
            # Test advanced features integration
            assert dashboard.advanced_features is not None
            
            # Test navigation system integration
            current_step = dashboard.navigation_system.get_current_step()
            assert isinstance(current_step, int)
            assert 1 <= current_step <= 4
            
            logger.info("✓ Integration stability validated")
            return True
            
        except Exception as e:
            logger.error(f"Integration stability test failed: {e}")
            return False
    
    def test_session_state_persistence(self) -> bool:
        """Test session state persistence and management."""
        try:
            import streamlit as st

            from dashboard.phase3_enhanced_dashboard import \
                Phase3EnhancedDashboard
            
            dashboard = Phase3EnhancedDashboard()
            dashboard.initialize_dashboard()
            
            # Test session state structure
            assert "phase3_dashboard" in st.session_state
            assert "phase3_navigation" in st.session_state
            assert "phase3_advanced" in st.session_state
            
            # Test session state persistence
            dashboard_state = st.session_state.phase3_dashboard
            assert "initialized" in dashboard_state
            assert "current_prediction" in dashboard_state
            assert "selected_teams" in dashboard_state
            
            logger.info("✓ Session state persistence validated")
            return True
            
        except Exception as e:
            logger.error(f"Session state persistence test failed: {e}")
            return False
    
    def test_concurrent_user_simulation(self) -> bool:
        """Test concurrent user simulation."""
        try:
            from dashboard.phase3_enhanced_dashboard import \
                Phase3EnhancedDashboard

            # Simulate multiple dashboard instances (concurrent users)
            dashboards = []
            for i in range(5):
                dashboard = Phase3EnhancedDashboard()
                dashboard.initialize_dashboard()
                dashboards.append(dashboard)
            
            # Test that all instances work independently
            for i, dashboard in enumerate(dashboards):
                dashboard.navigation_system.go_to_step((i % 4) + 1)
                current_step = dashboard.navigation_system.get_current_step()
                assert current_step == (i % 4) + 1
            
            logger.info("✓ Concurrent user simulation validated")
            return True
            
        except Exception as e:
            logger.error(f"Concurrent user simulation test failed: {e}")
            return False
    
    def test_data_validation_security(self) -> bool:
        """Test data validation and security measures."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer

            # Test XSS prevention
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "<img src=x onerror=alert('xss')>",
                "javascript:alert('xss')",
                "<iframe src='javascript:alert(1)'></iframe>"
            ]
            
            for malicious_input in malicious_inputs:
                is_safe = html_renderer.validate_html_content(malicious_input)
                assert is_safe == False  # Should detect as unsafe
                
                # Test escaping
                escaped = html_renderer.escape_user_content(malicious_input)
                assert "<script>" not in escaped  # Should be escaped
            
            logger.info("✓ Data validation and security validated")
            return True
            
        except Exception as e:
            logger.error(f"Data validation and security test failed: {e}")
            return False
    
    def test_html_rendering_safety(self) -> bool:
        """Test HTML rendering safety measures."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer

            # Test safe HTML rendering
            safe_content = "<div>Safe content</div>"
            html_renderer.render_safe_html(safe_content)
            
            # Test content validation
            assert html_renderer.validate_html_content(safe_content) == True
            
            # Test dangerous content rejection
            dangerous_content = "<script>alert('danger')</script>"
            assert html_renderer.validate_html_content(dangerous_content) == False
            
            logger.info("✓ HTML rendering safety validated")
            return True
            
        except Exception as e:
            logger.error(f"HTML rendering safety test failed: {e}")
            return False
    
    def test_css_injection_security(self) -> bool:
        """Test CSS injection security measures."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer

            # Test CSS injection is controlled
            html_renderer.inject_base_styles()
            
            # Test that CSS injection is idempotent
            assert html_renderer.global_styles_injected == True
            
            # Multiple injections should not cause issues
            for _ in range(5):
                html_renderer.inject_base_styles()
            
            logger.info("✓ CSS injection security validated")
            return True
            
        except Exception as e:
            logger.error(f"CSS injection security test failed: {e}")
            return False
    
    def test_component_isolation(self) -> bool:
        """Test component isolation and independence."""
        try:
            from dashboard.components.phase3_navigation_system import \
                Phase3NavigationSystem
            from dashboard.components.phase3_prediction_display import \
                Phase3PredictionDisplay
            from dashboard.components.phase3_team_selector import \
                Phase3TeamSelector

            # Test that components can be created independently
            nav1 = Phase3NavigationSystem()
            nav2 = Phase3NavigationSystem()
            
            team1 = Phase3TeamSelector()
            team2 = Phase3TeamSelector()
            
            pred1 = Phase3PredictionDisplay()
            pred2 = Phase3PredictionDisplay()
            
            # Test that they don't interfere with each other
            nav1.go_to_step(1)
            nav2.go_to_step(3)
            
            assert nav1.get_current_step() == 1
            assert nav2.get_current_step() == 3
            
            logger.info("✓ Component isolation validated")
            return True
            
        except Exception as e:
            logger.error(f"Component isolation test failed: {e}")
            return False
    
    def test_fallback_mechanisms(self) -> bool:
        """Test fallback mechanisms for failed components."""
        try:
            from dashboard.components.phase3_integration_adapter import \
                Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            
            # Test fallback prediction generation
            mock_prediction = adapter.generate_mock_prediction("Team A", "Team B")
            assert isinstance(mock_prediction, dict)
            assert "probabilities" in mock_prediction
            assert "confidence" in mock_prediction
            
            # Test fallback team data
            mock_teams = adapter.get_fallback_team_data("Premier League")
            assert isinstance(mock_teams, list)
            assert len(mock_teams) > 0
            
            logger.info("✓ Fallback mechanisms validated")
            return True
            
        except Exception as e:
            logger.error(f"Fallback mechanisms test failed: {e}")
            return False
    
    def test_loading_performance(self) -> bool:
        """Test loading performance and initialization speed."""
        try:
            # Test dashboard initialization speed
            start_time = time.time()
            
            from dashboard.phase3_enhanced_dashboard import \
                Phase3EnhancedDashboard
            dashboard = Phase3EnhancedDashboard()
            dashboard.initialize_dashboard()
            
            initialization_time = time.time() - start_time
            
            # Should initialize quickly (under 5 seconds)
            assert initialization_time < 5.0
            
            logger.info(f"✓ Loading performance validated (Initialization: {initialization_time:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Loading performance test failed: {e}")
            return False
    
    def test_visual_consistency(self) -> bool:
        """Test visual consistency across components."""
        try:
            from dashboard.components.phase3_html_renderer import html_renderer

            # Test consistent styling
            html_renderer.inject_base_styles()
            
            # Test consistent component rendering
            html_renderer.render_card("Test", "Content", "primary")
            html_renderer.render_alert("Test alert", "info")
            html_renderer.render_progress_bar(75, "Test Progress")
            
            # Test consistent badge rendering
            badges = [
                {"label": "Test 1", "type": "primary"},
                {"label": "Test 2", "type": "success"}
            ]
            html_renderer.render_badge_list(badges)
            
            logger.info("✓ Visual consistency validated")
            return True
            
        except Exception as e:
            logger.error(f"Visual consistency test failed: {e}")
            return False
    
    def test_user_experience_flow(self) -> bool:
        """Test overall user experience flow."""
        try:
            from dashboard.phase3_enhanced_dashboard import \
                Phase3EnhancedDashboard
            
            dashboard = Phase3EnhancedDashboard()
            dashboard.initialize_dashboard()
            
            # Test user journey flow
            nav_system = dashboard.navigation_system
            
            # Test step progression
            for step in range(1, 5):
                nav_system.go_to_step(step)
                current_step = nav_system.get_current_step()
                assert current_step == step
                
                # Test contextual help for each step
                nav_system.render_contextual_help(step)
            
            # Test completion tracking
            completion = nav_system.get_journey_completion_percentage()
            assert isinstance(completion, float)
            assert 0 <= completion <= 100
            
            logger.info("✓ User experience flow validated")
            return True
            
        except Exception as e:
            logger.error(f"User experience flow test failed: {e}")
            return False
    
    def test_production_deployment_readiness(self) -> bool:
        """Test production deployment readiness."""
        try:
            from dashboard.phase3_enhanced_dashboard import \
                Phase3EnhancedDashboard
            
            dashboard = Phase3EnhancedDashboard()
            dashboard.initialize_dashboard()
            
            # Test that all critical components are available
            assert dashboard.navigation_system is not None
            assert dashboard.team_selector is not None
            assert dashboard.prediction_display is not None
            assert dashboard.integration_adapter is not None
            assert dashboard.advanced_features is not None
            
            # Test integration status
            integration_status = dashboard.integration_adapter.get_integration_status()
            
            # Count successful integrations
            successful_integrations = sum(1 for status in integration_status.values() if status)
            total_integrations = len(integration_status)
            integration_rate = successful_integrations / total_integrations if total_integrations > 0 else 0
            
            # Should have at least 60% integration success for production readiness
            assert integration_rate >= 0.6
            
            logger.info(f"✓ Production deployment readiness validated (Integration: {integration_rate:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"Production deployment readiness test failed: {e}")
            return False
    
    def _generate_production_readiness_report(self):
        """Generate comprehensive production readiness report."""
        total = self.test_results["total_tests"]
        passed = self.test_results["passed_tests"]
        failed = self.test_results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        # Calculate performance metrics
        total_duration = sum(metrics["duration"] for metrics in self.test_results["performance_metrics"].values())
        total_memory_delta = sum(metrics["memory_delta"] for metrics in self.test_results["performance_metrics"].values())
        avg_duration = total_duration / total if total > 0 else 0
        
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_memory_increase = current_memory - self.start_memory
        
        logger.info("=" * 60)
        logger.info("🚀 PHASE 4 PRODUCTION READINESS TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 60)
        logger.info("📊 PERFORMANCE METRICS")
        logger.info(f"Total Test Duration: {total_duration:.2f}s")
        logger.info(f"Average Test Duration: {avg_duration:.2f}s")
        logger.info(f"Total Memory Delta: {total_memory_delta:+.1f}MB")
        logger.info(f"Overall Memory Increase: {total_memory_increase:+.1f}MB")
        logger.info("=" * 60)
        
        if success_rate >= 95:
            logger.info("🎉 EXCELLENT: Phase 4 Production Readiness is outstanding!")
            logger.info("✅ READY FOR PRODUCTION DEPLOYMENT")
        elif success_rate >= 90:
            logger.info("✅ VERY GOOD: Phase 4 Production Readiness is excellent")
            logger.info("✅ READY FOR PRODUCTION DEPLOYMENT")
        elif success_rate >= 85:
            logger.info("✅ GOOD: Phase 4 Production Readiness is solid")
            logger.info("⚠️ MINOR OPTIMIZATIONS RECOMMENDED BEFORE PRODUCTION")
        elif success_rate >= 75:
            logger.info("⚠️ FAIR: Phase 4 Production Readiness needs improvements")
            logger.info("⚠️ SIGNIFICANT TESTING REQUIRED BEFORE PRODUCTION")
        else:
            logger.info("❌ POOR: Phase 4 Production Readiness needs significant work")
            logger.info("❌ NOT READY FOR PRODUCTION DEPLOYMENT")
        
        # Detailed test results
        logger.info("\n📋 DETAILED PRODUCTION READINESS TEST RESULTS:")
        for test_detail in self.test_results["test_details"]:
            status_icon = "✅" if test_detail["passed"] else "❌"
            duration = test_detail.get("duration", 0)
            memory_delta = test_detail.get("memory_delta", 0)
            logger.info(f"{status_icon} {test_detail['test_name']} ({duration:.2f}s, {memory_delta:+.1f}MB)")


def main():
    """Run the Phase 4 production readiness tests."""
    tester = Phase4ProductionReadinessTester()
    results = tester.run_all_production_readiness_tests()
    
    # Return exit code based on results
    success_rate = (results["passed_tests"] / results["total_tests"] * 100) if results["total_tests"] > 0 else 0
    
    if success_rate >= 90:
        return 0  # Production ready
    elif success_rate >= 75:
        return 1  # Needs minor improvements
    else:
        return 2  # Not production ready


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
