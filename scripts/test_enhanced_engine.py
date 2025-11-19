#!/usr/bin/env python3
"""
Test Enhanced Prediction Engine

Simple test to verify the Enhanced Prediction Engine is working correctly.
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

def test_enhanced_engine():
    """Test the Enhanced Prediction Engine."""
    try:
        logger.info("Testing Enhanced Prediction Engine...")
        
        # Import the engine
        from enhanced_prediction_engine import EnhancedPredictionEngine
        
        # Initialize
        engine = EnhancedPredictionEngine()
        logger.info("Engine initialized successfully")
        
        # Test data
        match_data = {
            'home_team': 'Manchester City',
            'away_team': 'Liverpool',
            'home_league': 'Premier League',
            'away_league': 'Premier League',
            'date': '2024-01-15',
            'venue': 'Etihad Stadium'
        }
        
        # Make prediction
        logger.info("Making prediction...")
        result = engine.predict_match_outcome(match_data)
        
        # Check result
        if 'predictions' in result:
            predictions = result['predictions']
            home_win = predictions['home_win']
            draw = predictions['draw']
            away_win = predictions['away_win']
            
            logger.info(f"Predictions: Home={home_win:.3f}, Draw={draw:.3f}, Away={away_win:.3f}")
            
            # Check if uniform
            uniform_threshold = 0.05
            is_uniform = (abs(home_win - 1/3) < uniform_threshold and 
                         abs(draw - 1/3) < uniform_threshold and 
                         abs(away_win - 1/3) < uniform_threshold)
            
            if is_uniform:
                logger.warning("⚠️  UNIFORM DISTRIBUTION DETECTED!")
                return False
            else:
                logger.info("✅ Non-uniform prediction generated")
                return True
        else:
            logger.error("No predictions in result")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Run the test."""
    logger.info("🔍 Testing Enhanced Prediction Engine")
    logger.info("=" * 50)
    
    success = test_enhanced_engine()
    
    logger.info("=" * 50)
    if success:
        logger.info("✅ Enhanced Prediction Engine: PASSED")
    else:
        logger.error("❌ Enhanced Prediction Engine: FAILED")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
