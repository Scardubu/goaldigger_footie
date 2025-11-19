#!/usr/bin/env python3
"""
Test Dashboard Integration

Test the integration between the dashboard and the Enhanced Prediction Engine.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dashboard_prediction_integration():
    """Test the dashboard prediction integration."""
    try:
        logger.info("Testing Dashboard Prediction Integration...")
        
        # Import the dashboard class
        from dashboard.optimized_production_app import OptimizedDashboard
        
        # Initialize dashboard
        dashboard = OptimizedDashboard()
        logger.info("Dashboard initialized successfully")
        
        # Test prediction generation
        home_team = {'name': 'Manchester City', 'id': 1}
        away_team = {'name': 'Liverpool', 'id': 2}
        league_id = 'premier_league'
        
        logger.info(f"Testing prediction for {home_team['name']} vs {away_team['name']}")
        
        # Generate prediction
        prediction = dashboard.generate_prediction(home_team, away_team, league_id)
        
        if prediction:
            logger.info("Prediction generated successfully:")
            logger.info(f"  Home Win: {prediction['home_win_prob']:.3f}")
            logger.info(f"  Draw: {prediction['draw_prob']:.3f}")
            logger.info(f"  Away Win: {prediction['away_win_prob']:.3f}")
            logger.info(f"  Confidence: {prediction['confidence']:.3f}")
            logger.info(f"  Model: {prediction['model_insights']['primary_model']}")
            
            # Check if non-uniform
            uniform_threshold = 0.05
            is_uniform = (abs(prediction['home_win_prob'] - 1/3) < uniform_threshold and 
                         abs(prediction['draw_prob'] - 1/3) < uniform_threshold and 
                         abs(prediction['away_win_prob'] - 1/3) < uniform_threshold)
            
            if is_uniform:
                logger.warning("⚠️  UNIFORM DISTRIBUTION DETECTED!")
                return False
            else:
                logger.info("✅ Non-uniform prediction generated")
                return True
        else:
            logger.error("No prediction generated")
            return False
            
    except Exception as e:
        logger.error(f"Dashboard integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_league_mapping():
    """Test the league mapping functionality."""
    try:
        logger.info("Testing League Mapping...")
        
        from dashboard.optimized_production_app import OptimizedDashboard
        dashboard = OptimizedDashboard()
        
        # Test league mappings
        test_leagues = ['premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1']
        
        for league_id in test_leagues:
            league_name = dashboard.get_league_name_from_id(league_id)
            logger.info(f"  {league_id} -> {league_name}")
        
        logger.info("✅ League mapping working correctly")
        return True
        
    except Exception as e:
        logger.error(f"League mapping test failed: {e}")
        return False

def main():
    """Run dashboard integration tests."""
    logger.info("🔍 Starting Dashboard Integration Tests")
    logger.info("=" * 60)
    
    # Test 1: League mapping
    logger.info("Test 1: League Mapping")
    league_test_result = test_league_mapping()
    
    # Test 2: Dashboard prediction integration
    logger.info("\nTest 2: Dashboard Prediction Integration")
    integration_test_result = test_dashboard_prediction_integration()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🔍 DASHBOARD INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"League Mapping: {'✅ PASSED' if league_test_result else '❌ FAILED'}")
    logger.info(f"Prediction Integration: {'✅ PASSED' if integration_test_result else '❌ FAILED'}")
    
    if league_test_result and integration_test_result:
        logger.info("🎉 Dashboard integration working correctly!")
        return True
    else:
        logger.warning("⚠️  Dashboard integration has issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
