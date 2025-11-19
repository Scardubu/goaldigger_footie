#!/usr/bin/env python3
"""
Prediction Accuracy Test Script

Tests the prediction system to identify and resolve uniform 33.3% probability issues.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_xgboost_predictor():
    """Test XGBoost predictor directly."""
    try:
        from models.xgboost_predictor import XGBoostPredictor
        from utils.feature_mapper import FeatureMapper
        
        logger.info("Testing XGBoost Predictor...")
        
        # Initialize predictor
        model_path = "models/predictor_model.joblib"
        predictor = XGBoostPredictor(model_path)
        
        if not predictor.model:
            logger.error("XGBoost model not loaded")
            return False
        
        # Initialize feature mapper
        feature_mapper = FeatureMapper()
        
        # Create test features
        test_features = {
            'home_team_elo': 1600,
            'away_team_elo': 1550,
            'home_recent_form': 8.0,
            'away_recent_form': 6.0,
            'home_attack_form': 2.1,
            'away_attack_form': 1.8,
            'home_defense_form': 1.2,
            'away_defense_form': 1.5,
            'h2h_home_wins': 3,
            'h2h_draws': 2,
            'h2h_away_wins': 1,
            'elo_difference': 50,
            'home_injury_severity': 0.1,
            'away_injury_severity': 0.2,
            'weather_temperature': 18.0,
            'weather_rain': 0.0,
            'weather_wind_speed': 8.0
        }
        
        # Map features to model format
        feature_df = feature_mapper.map_features(test_features)
        logger.info(f"Mapped features shape: {feature_df.shape}")
        logger.info(f"Feature columns: {list(feature_df.columns)}")
        
        # Test prediction
        predictions, explanations = predictor.predict(feature_df)
        
        logger.info(f"Prediction shape: {predictions.shape}")
        logger.info(f"Predictions: {predictions[0]}")
        
        # Check for uniform distribution
        pred_values = predictions[0]
        if len(pred_values) == 3:
            home_win, draw, away_win = pred_values[2], pred_values[1], pred_values[0]
            logger.info(f"Home Win: {home_win:.3f}, Draw: {draw:.3f}, Away Win: {away_win:.3f}")
            
            # Check if uniform (within tolerance)
            uniform_threshold = 0.05
            if abs(home_win - 1/3) < uniform_threshold and abs(draw - 1/3) < uniform_threshold and abs(away_win - 1/3) < uniform_threshold:
                logger.warning("⚠️  UNIFORM DISTRIBUTION DETECTED!")
                return False
            else:
                logger.info("✅ Non-uniform prediction generated")
                return True
        else:
            logger.error(f"Unexpected prediction shape: {predictions.shape}")
            return False
            
    except Exception as e:
        logger.error(f"XGBoost predictor test failed: {e}")
        return False

def test_enhanced_prediction_engine():
    """Test Enhanced Prediction Engine."""
    try:
        from enhanced_prediction_engine import EnhancedPredictionEngine
        
        logger.info("Testing Enhanced Prediction Engine...")
        
        # Initialize engine
        engine = EnhancedPredictionEngine()
        
        # Create test match data
        test_match = {
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'home_league': 'Premier League',
            'away_league': 'Premier League',
            'date': datetime.now(),
            'venue': 'Emirates Stadium'
        }
        
        # Test prediction
        result = engine.predict_match_outcome(test_match)
        
        predictions = result['predictions']
        logger.info(f"Enhanced Engine Predictions:")
        logger.info(f"Home Win: {predictions['home_win']:.3f}")
        logger.info(f"Draw: {predictions['draw']:.3f}")
        logger.info(f"Away Win: {predictions['away_win']:.3f}")
        logger.info(f"Confidence: {result['confidence']['overall']:.3f}")
        
        # Check for uniform distribution
        home_win = predictions['home_win']
        draw = predictions['draw']
        away_win = predictions['away_win']
        
        uniform_threshold = 0.05
        if abs(home_win - 1/3) < uniform_threshold and abs(draw - 1/3) < uniform_threshold and abs(away_win - 1/3) < uniform_threshold:
            logger.warning("⚠️  UNIFORM DISTRIBUTION DETECTED!")
            return False
        else:
            logger.info("✅ Non-uniform prediction generated")
            return True
            
    except Exception as e:
        logger.error(f"Enhanced prediction engine test failed: {e}")
        return False

def test_integration_adapter():
    """Test Integration Adapter."""
    try:
        from enhanced_integration_adapter import get_integration_adapter
        
        logger.info("Testing Integration Adapter...")
        
        # Get adapter
        adapter = get_integration_adapter()
        
        # Test different match scenarios
        test_matches = [
            {
                'home_team': 'Manchester City',
                'away_team': 'Liverpool',
                'home_league': 'Premier League',
                'away_league': 'Premier League',
                'date': datetime.now(),
                'venue': 'Etihad Stadium'
            },
            {
                'home_team': 'Barcelona',
                'away_team': 'Real Madrid',
                'home_league': 'La Liga',
                'away_league': 'La Liga',
                'date': datetime.now(),
                'venue': 'Camp Nou'
            },
            {
                'home_team': 'Arsenal',
                'away_team': 'Barcelona',
                'home_league': 'Premier League',
                'away_league': 'La Liga',
                'date': datetime.now(),
                'venue': 'Emirates Stadium'
            }
        ]
        
        all_passed = True
        
        for i, match in enumerate(test_matches, 1):
            logger.info(f"\nTesting Match {i}: {match['home_team']} vs {match['away_team']}")
            
            result = adapter.predict_match_outcome(match)
            
            predictions = result['predictions']
            logger.info(f"Predictions: Home={predictions['home_win']:.3f}, Draw={predictions['draw']:.3f}, Away={predictions['away_win']:.3f}")
            logger.info(f"Engine: {result['metadata']['engine']}")
            logger.info(f"Confidence: {result['confidence']['overall']:.3f}")
            
            # Check for uniform distribution
            home_win = predictions['home_win']
            draw = predictions['draw']
            away_win = predictions['away_win']
            
            uniform_threshold = 0.05
            if abs(home_win - 1/3) < uniform_threshold and abs(draw - 1/3) < uniform_threshold and abs(away_win - 1/3) < uniform_threshold:
                logger.warning(f"⚠️  UNIFORM DISTRIBUTION DETECTED in Match {i}!")
                all_passed = False
            else:
                logger.info(f"✅ Non-uniform prediction for Match {i}")
        
        return all_passed
        
    except Exception as e:
        logger.error(f"Integration adapter test failed: {e}")
        return False

def test_feature_mapping():
    """Test feature mapping functionality."""
    try:
        from utils.feature_mapper import FeatureMapper
        from enhanced_feature_engine import EnhancedFeatureEngine
        
        logger.info("Testing Feature Mapping...")
        
        # Initialize components
        feature_mapper = FeatureMapper()
        feature_engine = EnhancedFeatureEngine()
        
        # Generate enhanced features
        test_match = {
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'home_league': 'Premier League',
            'away_league': 'Premier League',
            'date': datetime.now(),
            'venue': 'Emirates Stadium'
        }
        
        enhanced_features = feature_engine.generate_advanced_features(test_match)
        logger.info(f"Generated {len(enhanced_features)} enhanced features")
        
        # Map to model features
        mapped_features = feature_mapper.map_features(enhanced_features)
        logger.info(f"Mapped to {len(mapped_features.columns)} model features")
        
        # Validate mapping
        is_valid = feature_mapper.validate_features(mapped_features)
        logger.info(f"Feature validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        # Check for missing or invalid values
        missing_values = mapped_features.isnull().sum().sum()
        logger.info(f"Missing values: {missing_values}")
        
        return is_valid and missing_values == 0
        
    except Exception as e:
        logger.error(f"Feature mapping test failed: {e}")
        return False

def main():
    """Run all prediction accuracy tests."""
    logger.info("🚀 Starting Prediction Accuracy Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Feature Mapping", test_feature_mapping),
        ("XGBoost Predictor", test_xgboost_predictor),
        ("Enhanced Prediction Engine", test_enhanced_prediction_engine),
        ("Integration Adapter", test_integration_adapter)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} Test...")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name} test error: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Prediction accuracy issues resolved.")
        return True
    else:
        logger.warning("⚠️  Some tests failed. Prediction accuracy issues may persist.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
