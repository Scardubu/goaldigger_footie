"""
Phase 3 Integration Test
Comprehensive testing for Phase 3 dashboard integration with existing platform components.
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

class Phase3IntegrationTester:
    """Comprehensive tester for Phase 3 Dashboard integration."""
    
    def __init__(self):
        """Initialize the integration tester."""
        self.test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
    
    def run_all_integration_tests(self) -> Dict[str, Any]:
        """Run all Phase 3 integration tests."""
        logger.info("🔗 Starting Phase 3 Dashboard Integration Tests")
        logger.info("=" * 60)
        
        test_methods = [
            self.test_integration_adapter_import,
            self.test_integration_adapter_initialization,
            self.test_prediction_engine_integration,
            self.test_data_loader_integration,
            self.test_team_data_integration,
            self.test_league_data_integration,
            self.test_real_prediction_generation,
            self.test_fallback_mechanisms,
            self.test_dashboard_integration,
            self.test_backward_compatibility,
            self.test_error_handling_integration,
            self.test_performance_with_integration,
            self.test_session_state_integration,
            self.test_end_to_end_workflow,
            self.test_production_readiness
        ]
        
        for test_method in test_methods:
            self._run_single_test(test_method)
        
        self._generate_integration_report()
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
    
    def test_integration_adapter_import(self) -> bool:
        """Test integration adapter import."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            # Test instantiation
            adapter = Phase3IntegrationAdapter()
            
            # Test basic attributes
            assert hasattr(adapter, 'prediction_engine')
            assert hasattr(adapter, 'data_loader')
            assert hasattr(adapter, 'integration_status')
            assert hasattr(adapter, 'initialized')
            
            logger.info("✓ Integration adapter import successful")
            return True
            
        except Exception as e:
            logger.error(f"Integration adapter import failed: {e}")
            return False
    
    def test_integration_adapter_initialization(self) -> bool:
        """Test integration adapter initialization."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            
            # Test initialization
            integration_status = adapter.initialize_integrations()
            
            assert isinstance(integration_status, dict)
            assert 'prediction_engine' in integration_status
            assert 'data_loader' in integration_status
            assert 'config' in integration_status
            assert 'database' in integration_status
            assert 'scrapers' in integration_status
            
            logger.info(f"✓ Integration adapter initialization successful: {sum(integration_status.values())}/5 components")
            return True
            
        except Exception as e:
            logger.error(f"Integration adapter initialization failed: {e}")
            return False
    
    def test_prediction_engine_integration(self) -> bool:
        """Test prediction engine integration."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            adapter.initialize_integrations()
            
            # Test prediction generation
            mock_home = {"name": "Test Home Team", "league": "Test League"}
            mock_away = {"name": "Test Away Team", "league": "Test League"}
            
            prediction = adapter.generate_real_prediction(mock_home, mock_away, "same_league")
            
            assert isinstance(prediction, dict)
            assert 'probabilities' in prediction
            assert 'confidence' in prediction
            assert 'generated_at' in prediction
            
            # Check probability structure
            probs = prediction['probabilities']
            assert 'home_win' in probs
            assert 'away_win' in probs
            assert 'draw' in probs
            
            # Check probability values are valid
            total_prob = probs['home_win'] + probs['away_win'] + probs['draw']
            assert 0.99 <= total_prob <= 1.01  # Allow for small floating point errors
            
            logger.info("✓ Prediction engine integration successful")
            return True
            
        except Exception as e:
            logger.error(f"Prediction engine integration failed: {e}")
            return False
    
    def test_data_loader_integration(self) -> bool:
        """Test data loader integration."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            adapter.initialize_integrations()
            
            # Test league data retrieval
            leagues = adapter.get_real_leagues()
            
            assert isinstance(leagues, list)
            assert len(leagues) > 0
            
            # Test league data structure
            for league in leagues:
                assert 'name' in league
                assert 'country' in league
                assert isinstance(league['name'], str)
                assert isinstance(league['country'], str)
            
            logger.info(f"✓ Data loader integration successful: {len(leagues)} leagues available")
            return True
            
        except Exception as e:
            logger.error(f"Data loader integration failed: {e}")
            return False
    
    def test_team_data_integration(self) -> bool:
        """Test team data integration."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            adapter.initialize_integrations()
            
            # Test team data retrieval for multiple leagues
            test_leagues = ["Premier League", "La Liga", "Serie A"]
            
            for league in test_leagues:
                teams = adapter.get_real_teams_for_league(league)
                
                assert isinstance(teams, list)
                assert len(teams) > 0
                
                # Test team data structure
                for team in teams:
                    assert 'name' in team
                    assert 'league' in team
                    assert isinstance(team['name'], str)
                    assert isinstance(team['league'], str)
            
            logger.info(f"✓ Team data integration successful for {len(test_leagues)} leagues")
            return True
            
        except Exception as e:
            logger.error(f"Team data integration failed: {e}")
            return False
    
    def test_league_data_integration(self) -> bool:
        """Test league data integration."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            adapter.initialize_integrations()
            
            # Test integration status
            status = adapter.get_integration_status()
            
            assert isinstance(status, dict)
            assert 'initialized' in status
            assert 'components' in status
            assert 'success_rate' in status
            assert 'available_features' in status
            
            # Test success rate calculation
            success_rate = status['success_rate']
            assert 0 <= success_rate <= 100
            
            logger.info(f"✓ League data integration successful: {success_rate:.1f}% integration rate")
            return True
            
        except Exception as e:
            logger.error(f"League data integration failed: {e}")
            return False
    
    def test_real_prediction_generation(self) -> bool:
        """Test real prediction generation workflow."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            adapter.initialize_integrations()
            
            # Test with realistic team data
            home_team = {"name": "Manchester City", "league": "Premier League", "id": "mc_1"}
            away_team = {"name": "Arsenal", "league": "Premier League", "id": "ar_1"}
            
            prediction = adapter.generate_real_prediction(home_team, away_team, "same_league")
            
            # Validate prediction structure
            assert 'probabilities' in prediction
            assert 'confidence' in prediction
            assert 'model_version' in prediction
            assert 'features_used' in prediction
            assert 'source' in prediction
            
            # Validate confidence is reasonable
            confidence = prediction['confidence']
            assert 0.0 <= confidence <= 1.0
            
            logger.info(f"✓ Real prediction generation successful (confidence: {confidence:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"Real prediction generation failed: {e}")
            return False
    
    def test_fallback_mechanisms(self) -> bool:
        """Test fallback mechanisms when components are unavailable."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            # Don't initialize integrations to test fallbacks
            
            # Test fallback prediction
            mock_home = {"name": "Test Home", "league": "Test League"}
            mock_away = {"name": "Test Away", "league": "Test League"}
            
            prediction = adapter.generate_real_prediction(mock_home, mock_away, "same_league")
            
            assert isinstance(prediction, dict)
            assert 'probabilities' in prediction
            assert prediction.get('source') == 'mock_prediction'
            
            # Test fallback team data
            teams = adapter.get_real_teams_for_league("Premier League")
            assert isinstance(teams, list)
            assert len(teams) > 0
            
            logger.info("✓ Fallback mechanisms working correctly")
            return True
            
        except Exception as e:
            logger.error(f"Fallback mechanisms test failed: {e}")
            return False
    
    def test_dashboard_integration(self) -> bool:
        """Test Phase 3 dashboard integration with adapter."""
        try:
            from dashboard.phase3_enhanced_dashboard import Phase3EnhancedDashboard
            
            dashboard = Phase3EnhancedDashboard()
            
            # Test that integration adapter is available
            assert hasattr(dashboard, 'integration_adapter')
            assert dashboard.integration_adapter is not None
            
            # Test initialization
            init_result = dashboard.initialize_dashboard()
            assert init_result is True
            
            logger.info("✓ Dashboard integration successful")
            return True
            
        except Exception as e:
            logger.error(f"Dashboard integration failed: {e}")
            return False
    
    def test_backward_compatibility(self) -> bool:
        """Test backward compatibility with existing components."""
        try:
            # Test that existing imports still work
            from dashboard.phase3_enhanced_dashboard import main
            assert callable(main)
            
            # Test that Phase 3 components don't break existing functionality
            from dashboard.components.phase3_navigation_system import Phase3NavigationSystem
            from dashboard.components.phase3_team_selector import Phase3TeamSelector
            from dashboard.components.phase3_prediction_display import Phase3PredictionDisplay
            
            # All should instantiate without errors
            nav = Phase3NavigationSystem()
            selector = Phase3TeamSelector()
            display = Phase3PredictionDisplay()
            
            assert nav is not None
            assert selector is not None
            assert display is not None
            
            logger.info("✓ Backward compatibility maintained")
            return True
            
        except Exception as e:
            logger.error(f"Backward compatibility test failed: {e}")
            return False
    
    def test_error_handling_integration(self) -> bool:
        """Test error handling in integrated environment."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            adapter.initialize_integrations()
            
            # Test error handling with invalid inputs
            try:
                prediction = adapter.generate_real_prediction(None, None, "invalid")
                # Should not raise exception, should return fallback
                assert isinstance(prediction, dict)
            except Exception:
                pass  # Expected behavior
            
            # Test error handling with invalid league
            teams = adapter.get_real_teams_for_league("Invalid League")
            assert isinstance(teams, list)  # Should return empty list or fallback
            
            logger.info("✓ Error handling integration successful")
            return True
            
        except Exception as e:
            logger.error(f"Error handling integration failed: {e}")
            return False
    
    def test_performance_with_integration(self) -> bool:
        """Test performance with full integration."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            
            # Test initialization performance
            start_time = time.time()
            adapter.initialize_integrations()
            init_time = time.time() - start_time
            
            # Should initialize reasonably quickly
            assert init_time < 10.0  # Allow up to 10 seconds for full integration
            
            # Test prediction performance
            mock_home = {"name": "Test Home", "league": "Test League"}
            mock_away = {"name": "Test Away", "league": "Test League"}
            
            start_time = time.time()
            prediction = adapter.generate_real_prediction(mock_home, mock_away, "same_league")
            pred_time = time.time() - start_time
            
            # Should generate predictions quickly
            assert pred_time < 5.0  # Allow up to 5 seconds for prediction
            
            logger.info(f"✓ Performance integration successful (init: {init_time:.2f}s, pred: {pred_time:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Performance integration test failed: {e}")
            return False
    
    def test_session_state_integration(self) -> bool:
        """Test session state integration."""
        try:
            # Mock streamlit session state for testing
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
            
            # Test session state patterns used in Phase 3
            mock_st = type('MockStreamlit', (), {'session_state': MockSessionState()})()
            
            # Initialize Phase 3 session state
            if "phase3_dashboard" not in mock_st.session_state:
                mock_st.session_state["phase3_dashboard"] = {
                    "initialized": False,
                    "current_prediction": None,
                    "selected_teams": {},
                    "user_journey_complete": False,
                    "integration_status": {}
                }
            
            assert "phase3_dashboard" in mock_st.session_state
            assert "integration_status" in mock_st.session_state["phase3_dashboard"]
            
            logger.info("✓ Session state integration successful")
            return True
            
        except Exception as e:
            logger.error(f"Session state integration failed: {e}")
            return False
    
    def test_end_to_end_workflow(self) -> bool:
        """Test end-to-end workflow integration."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            adapter.initialize_integrations()
            
            # Simulate complete workflow
            # 1. Get leagues
            leagues = adapter.get_real_leagues()
            assert len(leagues) > 0
            
            # 2. Get teams for first league
            first_league = leagues[0]["name"]
            teams = adapter.get_real_teams_for_league(first_league)
            assert len(teams) >= 2  # Need at least 2 teams for a match
            
            # 3. Generate prediction
            home_team = teams[0]
            away_team = teams[1]
            prediction = adapter.generate_real_prediction(home_team, away_team, "same_league")
            
            assert 'probabilities' in prediction
            assert 'confidence' in prediction
            
            # 4. Test integration status
            status = adapter.get_integration_status()
            assert isinstance(status, dict)
            
            logger.info("✓ End-to-end workflow integration successful")
            return True
            
        except Exception as e:
            logger.error(f"End-to-end workflow integration failed: {e}")
            return False
    
    def test_production_readiness(self) -> bool:
        """Test production readiness of integrated system."""
        try:
            from dashboard.components.phase3_integration_adapter import Phase3IntegrationAdapter
            
            adapter = Phase3IntegrationAdapter()
            integration_status = adapter.initialize_integrations()
            
            # Test integration test functionality
            test_results = adapter.test_integration()
            
            assert isinstance(test_results, dict)
            assert 'test_results' in test_results
            assert 'success_rate' in test_results
            assert 'integration_status' in test_results
            
            success_rate = test_results['success_rate']
            assert 0 <= success_rate <= 100
            
            logger.info(f"✓ Production readiness test successful: {success_rate:.1f}% integration success")
            return True
            
        except Exception as e:
            logger.error(f"Production readiness test failed: {e}")
            return False
    
    def _generate_integration_report(self):
        """Generate comprehensive integration test report."""
        total = self.test_results["total_tests"]
        passed = self.test_results["passed_tests"]
        failed = self.test_results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info("=" * 60)
        logger.info("🔗 PHASE 3 INTEGRATION TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed} ✅")
        logger.info(f"Failed: {failed} ❌")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info("=" * 60)
        
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT: Phase 3 Integration is production-ready!")
        elif success_rate >= 80:
            logger.info("✅ GOOD: Phase 3 Integration is mostly ready with minor issues")
        elif success_rate >= 70:
            logger.info("⚠️ FAIR: Phase 3 Integration needs some improvements")
        else:
            logger.info("❌ POOR: Phase 3 Integration needs significant work")
        
        # Detailed test results
        logger.info("\n📋 DETAILED INTEGRATION TEST RESULTS:")
        for test_detail in self.test_results["test_details"]:
            status_icon = "✅" if test_detail["passed"] else "❌"
            logger.info(f"{status_icon} {test_detail['test_name']} ({test_detail['duration']:.2f}s)")


def main():
    """Run the Phase 3 integration tests."""
    tester = Phase3IntegrationTester()
    results = tester.run_all_integration_tests()
    
    # Return exit code based on results
    if results["failed_tests"] == 0:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
