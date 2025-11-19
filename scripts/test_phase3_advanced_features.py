"""
Phase 3 Advanced Features Test
Comprehensive testing for advanced features implementation.
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

class Phase3AdvancedFeaturesTester:
    """Comprehensive tester for Phase 3 advanced features."""
    
    def __init__(self):
        """Initialize the advanced features tester."""
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
    
    def run_all_advanced_features_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 advanced features tests."""
        logger.info("🚀 Starting Phase 3 Advanced Features Tests")
        logger.info("=" * 60)
        
        test_methods = [
            self.test_advanced_features_initialization,
            self.test_real_time_feedback_system,
            self.test_personalization_features,
            self.test_user_preferences_management,
            self.test_notification_settings,
            self.test_actionable_insights_dashboard,
            self.test_confidence_scoring_system,
            self.test_value_betting_highlights,
            self.test_risk_assessment_visualization,
            self.test_ai_recommendations_engine,
            self.test_decision_making_tools,
            self.test_team_comparison_features,
            self.test_historical_performance_analysis,
            self.test_market_analysis_tools,
            self.test_performance_tracking_system,
            self.test_interactive_feedback_styles,
            self.test_loading_states_management,
            self.test_action_feedback_system,
            self.test_dashboard_integration,
            self.test_session_state_management
        ]
        
        for test_method in test_methods:
            self._run_single_test(test_method)
        
        self._generate_advanced_features_report()
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
    
    def test_advanced_features_initialization(self) -> bool:
        """Test advanced features initialization."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test initialization
            assert hasattr(advanced_features, 'initialize_session_state')
            
            # Test session state initialization
            advanced_features.initialize_session_state()
            
            logger.info("✓ Advanced features initialization validated")
            return True
            
        except Exception as e:
            logger.error(f"Advanced features initialization test failed: {e}")
            return False
    
    def test_real_time_feedback_system(self) -> bool:
        """Test real-time feedback system."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test feedback methods
            assert hasattr(advanced_features, 'render_real_time_feedback_system')
            assert hasattr(advanced_features, 'add_action_feedback')
            assert hasattr(advanced_features, 'update_loading_state')
            
            # Test adding feedback
            advanced_features.add_action_feedback("Test message", "success")
            advanced_features.add_action_feedback("Warning message", "warning")
            
            # Test loading state updates
            advanced_features.update_loading_state("test_operation", 50.0)
            advanced_features.update_loading_state("test_operation", 100.0)
            
            logger.info("✓ Real-time feedback system validated")
            return True
            
        except Exception as e:
            logger.error(f"Real-time feedback system test failed: {e}")
            return False
    
    def test_personalization_features(self) -> bool:
        """Test personalization features."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test personalization methods
            assert hasattr(advanced_features, 'render_personalization_features')
            assert hasattr(advanced_features, '_render_user_preferences')
            assert hasattr(advanced_features, '_render_favorite_teams_manager')
            assert hasattr(advanced_features, '_render_notification_settings')
            
            logger.info("✓ Personalization features validated")
            return True
            
        except Exception as e:
            logger.error(f"Personalization features test failed: {e}")
            return False
    
    def test_user_preferences_management(self) -> bool:
        """Test user preferences management."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            import streamlit as st
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test preferences structure
            preferences = st.session_state.phase3_advanced["user_preferences"]
            
            assert "favorite_leagues" in preferences
            assert "favorite_teams" in preferences
            assert "notification_settings" in preferences
            assert "dashboard_layout" in preferences
            assert "theme" in preferences
            
            # Test notification settings structure
            notifications = preferences["notification_settings"]
            assert "high_confidence_alerts" in notifications
            assert "value_betting_alerts" in notifications
            assert "favorite_team_alerts" in notifications
            
            logger.info("✓ User preferences management validated")
            return True
            
        except Exception as e:
            logger.error(f"User preferences management test failed: {e}")
            return False
    
    def test_notification_settings(self) -> bool:
        """Test notification settings functionality."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            import streamlit as st
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test notification settings
            notifications = st.session_state.phase3_advanced["user_preferences"]["notification_settings"]
            
            # Test default values
            assert isinstance(notifications["high_confidence_alerts"], bool)
            assert isinstance(notifications["value_betting_alerts"], bool)
            assert isinstance(notifications["favorite_team_alerts"], bool)
            
            logger.info("✓ Notification settings validated")
            return True
            
        except Exception as e:
            logger.error(f"Notification settings test failed: {e}")
            return False
    
    def test_actionable_insights_dashboard(self) -> bool:
        """Test actionable insights dashboard."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test insights methods
            assert hasattr(advanced_features, 'render_actionable_insights_dashboard')
            assert hasattr(advanced_features, '_render_confidence_scoring')
            assert hasattr(advanced_features, '_render_value_betting_highlights')
            assert hasattr(advanced_features, '_render_risk_assessment')
            assert hasattr(advanced_features, '_render_ai_recommendations')
            
            # Test with mock prediction
            mock_prediction = {
                "probabilities": {"home_win": 0.45, "away_win": 0.35, "draw": 0.20},
                "confidence": 0.75
            }
            
            # Should not raise exceptions
            advanced_features._render_confidence_scoring(mock_prediction)
            advanced_features._render_value_betting_highlights(mock_prediction)
            advanced_features._render_risk_assessment(mock_prediction)
            advanced_features._render_ai_recommendations(mock_prediction)
            
            logger.info("✓ Actionable insights dashboard validated")
            return True
            
        except Exception as e:
            logger.error(f"Actionable insights dashboard test failed: {e}")
            return False
    
    def test_confidence_scoring_system(self) -> bool:
        """Test confidence scoring system."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test confidence scoring with different levels
            test_predictions = [
                {"confidence": 0.9},  # Very High
                {"confidence": 0.7},  # High
                {"confidence": 0.55}, # Medium
                {"confidence": 0.3}   # Low
            ]
            
            for prediction in test_predictions:
                # Should handle all confidence levels without errors
                advanced_features._render_confidence_scoring(prediction)
            
            logger.info("✓ Confidence scoring system validated")
            return True
            
        except Exception as e:
            logger.error(f"Confidence scoring system test failed: {e}")
            return False
    
    def test_value_betting_highlights(self) -> bool:
        """Test value betting highlights."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test value betting highlights
            mock_prediction = {
                "probabilities": {"home_win": 0.45, "away_win": 0.35, "draw": 0.20},
                "confidence": 0.75
            }
            
            # Should render without errors
            advanced_features._render_value_betting_highlights(mock_prediction)
            
            logger.info("✓ Value betting highlights validated")
            return True
            
        except Exception as e:
            logger.error(f"Value betting highlights test failed: {e}")
            return False
    
    def test_risk_assessment_visualization(self) -> bool:
        """Test risk assessment visualization."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test risk assessment
            mock_prediction = {
                "probabilities": {"home_win": 0.45, "away_win": 0.35, "draw": 0.20},
                "confidence": 0.75
            }
            
            # Should render without errors
            advanced_features._render_risk_assessment(mock_prediction)
            
            logger.info("✓ Risk assessment visualization validated")
            return True
            
        except Exception as e:
            logger.error(f"Risk assessment visualization test failed: {e}")
            return False
    
    def test_ai_recommendations_engine(self) -> bool:
        """Test AI recommendations engine."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test AI recommendations
            mock_prediction = {
                "probabilities": {"home_win": 0.45, "away_win": 0.35, "draw": 0.20},
                "confidence": 0.75
            }
            
            # Should render without errors
            advanced_features._render_ai_recommendations(mock_prediction)
            
            logger.info("✓ AI recommendations engine validated")
            return True
            
        except Exception as e:
            logger.error(f"AI recommendations engine test failed: {e}")
            return False
    
    def test_decision_making_tools(self) -> bool:
        """Test decision-making tools."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test decision-making methods
            assert hasattr(advanced_features, 'render_decision_making_tools')
            assert hasattr(advanced_features, '_render_team_comparison')
            assert hasattr(advanced_features, '_render_historical_performance')
            assert hasattr(advanced_features, '_render_market_analysis')
            assert hasattr(advanced_features, '_render_performance_tracking')
            
            logger.info("✓ Decision-making tools validated")
            return True
            
        except Exception as e:
            logger.error(f"Decision-making tools test failed: {e}")
            return False
    
    def test_team_comparison_features(self) -> bool:
        """Test team comparison features."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test team comparison
            mock_home_team = {"name": "Test Home Team"}
            mock_away_team = {"name": "Test Away Team"}
            
            # Should render without errors
            advanced_features._render_team_comparison(mock_home_team, mock_away_team)
            
            logger.info("✓ Team comparison features validated")
            return True
            
        except Exception as e:
            logger.error(f"Team comparison features test failed: {e}")
            return False
    
    def test_historical_performance_analysis(self) -> bool:
        """Test historical performance analysis."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test historical performance
            mock_home_team = {"name": "Test Home Team"}
            mock_away_team = {"name": "Test Away Team"}
            
            # Should render without errors
            advanced_features._render_historical_performance(mock_home_team, mock_away_team)
            
            logger.info("✓ Historical performance analysis validated")
            return True
            
        except Exception as e:
            logger.error(f"Historical performance analysis test failed: {e}")
            return False
    
    def test_market_analysis_tools(self) -> bool:
        """Test market analysis tools."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test market analysis
            # Should render without errors
            advanced_features._render_market_analysis()
            
            logger.info("✓ Market analysis tools validated")
            return True
            
        except Exception as e:
            logger.error(f"Market analysis tools test failed: {e}")
            return False
    
    def test_performance_tracking_system(self) -> bool:
        """Test performance tracking system."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            import streamlit as st
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test performance metrics structure
            metrics = st.session_state.phase3_advanced["performance_metrics"]
            
            assert "total_predictions" in metrics
            assert "successful_predictions" in metrics
            assert "total_value_found" in metrics
            assert "average_confidence" in metrics
            
            # Test performance tracking rendering
            advanced_features._render_performance_tracking()
            
            logger.info("✓ Performance tracking system validated")
            return True
            
        except Exception as e:
            logger.error(f"Performance tracking system test failed: {e}")
            return False
    
    def test_interactive_feedback_styles(self) -> bool:
        """Test interactive feedback styles."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test interactive feedback styles injection
            advanced_features._inject_interactive_feedback_styles()
            
            logger.info("✓ Interactive feedback styles validated")
            return True
            
        except Exception as e:
            logger.error(f"Interactive feedback styles test failed: {e}")
            return False
    
    def test_loading_states_management(self) -> bool:
        """Test loading states management."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            import streamlit as st
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test loading state management
            advanced_features.update_loading_state("test_operation", 25.0)
            advanced_features.update_loading_state("test_operation", 50.0)
            advanced_features.update_loading_state("test_operation", 75.0)
            advanced_features.update_loading_state("test_operation", 100.0)
            
            # Test loading states structure
            loading_states = st.session_state.phase3_advanced["real_time_feedback"]["loading_states"]
            assert isinstance(loading_states, dict)
            
            logger.info("✓ Loading states management validated")
            return True
            
        except Exception as e:
            logger.error(f"Loading states management test failed: {e}")
            return False
    
    def test_action_feedback_system(self) -> bool:
        """Test action feedback system."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            import streamlit as st
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test action feedback
            advanced_features.add_action_feedback("Success message", "success")
            advanced_features.add_action_feedback("Info message", "info")
            advanced_features.add_action_feedback("Warning message", "warning")
            advanced_features.add_action_feedback("Error message", "danger")
            
            # Test feedback structure
            feedback_messages = st.session_state.phase3_advanced["real_time_feedback"]["action_feedback"]
            assert isinstance(feedback_messages, list)
            assert len(feedback_messages) > 0
            
            logger.info("✓ Action feedback system validated")
            return True
            
        except Exception as e:
            logger.error(f"Action feedback system test failed: {e}")
            return False
    
    def test_dashboard_integration(self) -> bool:
        """Test dashboard integration."""
        try:
            from dashboard.phase3_enhanced_dashboard import Phase3EnhancedDashboard
            
            dashboard = Phase3EnhancedDashboard()
            
            # Test advanced features integration
            assert hasattr(dashboard, 'advanced_features')
            assert dashboard.advanced_features is not None
            
            logger.info("✓ Dashboard integration validated")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard integration test failed: {e}")
            return False
    
    def test_session_state_management(self) -> bool:
        """Test session state management."""
        try:
            from dashboard.components.phase3_advanced_features import Phase3AdvancedFeatures
            import streamlit as st
            
            advanced_features = Phase3AdvancedFeatures()
            
            # Test session state structure
            assert "phase3_advanced" in st.session_state
            
            advanced_state = st.session_state.phase3_advanced
            assert "user_preferences" in advanced_state
            assert "betting_history" in advanced_state
            assert "performance_metrics" in advanced_state
            assert "real_time_feedback" in advanced_state
            
            logger.info("✓ Session state management validated")
            return True
            
        except Exception as e:
            logger.error(f"Session state management test failed: {e}")
            return False
    
    def _generate_advanced_features_report(self):
        """Generate comprehensive advanced features test report."""
        total = self.test_results["total_tests"]
        passed = self.test_results["passed_tests"]
        failed = self.test_results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info("=" * 60)
        logger.info("🚀 PHASE 3 ADVANCED FEATURES TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 60)
        
        if success_rate >= 95:
            logger.info("🎉 EXCELLENT: Phase 3 Advanced Features are outstanding!")
        elif success_rate >= 90:
            logger.info("✅ VERY GOOD: Phase 3 Advanced Features are excellent")
        elif success_rate >= 80:
            logger.info("✅ GOOD: Phase 3 Advanced Features are solid")
        elif success_rate >= 70:
            logger.info("⚠️ FAIR: Phase 3 Advanced Features need improvements")
        else:
            logger.info("❌ POOR: Phase 3 Advanced Features need significant work")
        
        # Detailed test results
        logger.info("\n📋 DETAILED ADVANCED FEATURES TEST RESULTS:")
        for test_detail in self.test_results["test_details"]:
            status_icon = "✅" if test_detail["passed"] else "❌"
            logger.info(f"{status_icon} {test_detail['test_name']} ({test_detail['duration']:.2f}s)")


def main():
    """Run the Phase 3 advanced features tests."""
    tester = Phase3AdvancedFeaturesTester()
    results = tester.run_all_advanced_features_tests()
    
    # Return exit code based on results
    if results["failed_tests"] == 0:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
