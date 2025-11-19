#!/usr/bin/env python3
"""
Debug Model Prediction

Direct test of the XGBoost model to understand why it's returning uniform predictions.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_raw_model():
    """Test the raw XGBoost model directly."""
    try:
        logger.info("Loading raw model...")
        
        # Load the model file
        model_path = "models/predictor_model.joblib"
        model_data = joblib.load(model_path)
        
        logger.info(f"Model data type: {type(model_data)}")
        logger.info(f"Model data keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dict'}")
        
        # Extract the actual model
        if isinstance(model_data, dict):
            model = model_data.get('model')
            logger.info(f"Extracted model type: {type(model)}")
            
            # Check if it's an XGBoost model
            if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                logger.info("✅ Model has both predict and predict_proba methods")
                
                # Get feature names from the model
                if hasattr(model, 'feature_names_in_'):
                    feature_names = list(model.feature_names_in_)
                    logger.info(f"Model feature names: {feature_names}")
                elif hasattr(model, 'get_booster'):
                    booster = model.get_booster()
                    if hasattr(booster, 'feature_names'):
                        feature_names = booster.feature_names
                        logger.info(f"Booster feature names: {feature_names}")
                    else:
                        logger.warning("No feature names found in booster")
                        feature_names = None
                else:
                    logger.warning("No feature names found in model")
                    feature_names = None
                
                # Create test data with the exact features the model expects
                if feature_names:
                    logger.info(f"Creating test data with {len(feature_names)} features")
                    
                    # Create realistic test data
                    test_data = {
                        'h2h_team1_wins': 3.0,
                        'h2h_draws': 2.0,
                        'h2h_team2_wins': 1.0,
                        'h2h_avg_goals': 2.5,
                        'home_formation': 1.0,  # 4-3-3 encoded
                        'away_formation': 2.0,  # 4-4-2 encoded
                        'formation_clash_score': 0.2,
                        'home_match_xg': 1.8,
                        'away_match_xg': 1.4,
                        'home_injury_impact': 0.1,
                        'away_injury_impact': 0.2,
                        'home_elo': 1650.0,
                        'away_elo': 1580.0,
                        'elo_diff': 70.0,
                        'home_form_points_last_5': 9.0,
                        'away_form_points_last_5': 6.0,
                        'home_avg_goals_scored_last_5': 2.2,
                        'home_avg_goals_conceded_last_5': 1.0,
                        'away_avg_goals_scored_last_5': 1.6,
                        'away_avg_goals_conceded_last_5': 1.4
                    }
                    
                    # Create DataFrame
                    df = pd.DataFrame([test_data])
                    
                    # Ensure correct order
                    df = df[feature_names]
                    
                    logger.info(f"Test data shape: {df.shape}")
                    logger.info(f"Test data columns: {list(df.columns)}")
                    logger.info(f"Test data values: {df.iloc[0].to_dict()}")
                    
                    # Test predict method
                    logger.info("\n--- Testing predict() method ---")
                    try:
                        pred_classes = model.predict(df)
                        logger.info(f"predict() output: {pred_classes}")
                        logger.info(f"predict() shape: {pred_classes.shape}")
                        logger.info(f"predict() type: {type(pred_classes)}")
                    except Exception as e:
                        logger.error(f"predict() failed: {e}")
                    
                    # Test predict_proba method
                    logger.info("\n--- Testing predict_proba() method ---")
                    try:
                        pred_proba = model.predict_proba(df)
                        logger.info(f"predict_proba() output: {pred_proba}")
                        logger.info(f"predict_proba() shape: {pred_proba.shape}")
                        logger.info(f"predict_proba() type: {type(pred_proba)}")
                        
                        if pred_proba.shape[1] == 3:
                            home_win, draw, away_win = pred_proba[0]
                            logger.info(f"Probabilities: Home={home_win:.4f}, Draw={draw:.4f}, Away={away_win:.4f}")
                            
                            # Check if uniform
                            uniform_threshold = 0.05
                            if abs(home_win - 1/3) < uniform_threshold and abs(draw - 1/3) < uniform_threshold and abs(away_win - 1/3) < uniform_threshold:
                                logger.warning("⚠️  UNIFORM DISTRIBUTION DETECTED!")
                                return False
                            else:
                                logger.info("✅ Non-uniform prediction generated")
                                return True
                        else:
                            logger.error(f"Unexpected probability shape: {pred_proba.shape}")
                            return False
                            
                    except Exception as e:
                        logger.error(f"predict_proba() failed: {e}")
                        return False
                
                else:
                    logger.error("Cannot test model without feature names")
                    return False
                    
            else:
                logger.error("Model doesn't have required prediction methods")
                return False
        else:
            logger.error("Model data is not a dictionary")
            return False
            
    except Exception as e:
        logger.error(f"Raw model test failed: {e}")
        return False

def test_different_inputs():
    """Test the model with different input scenarios."""
    try:
        logger.info("\n" + "="*60)
        logger.info("Testing model with different input scenarios")
        logger.info("="*60)
        
        # Load model
        model_path = "models/predictor_model.joblib"
        model_data = joblib.load(model_path)
        model = model_data['model']
        
        # Feature names
        feature_names = [
            'h2h_team1_wins', 'h2h_draws', 'h2h_team2_wins', 'h2h_avg_goals',
            'home_formation', 'away_formation', 'formation_clash_score',
            'home_match_xg', 'away_match_xg', 'home_injury_impact', 'away_injury_impact',
            'home_elo', 'away_elo', 'elo_diff', 'home_form_points_last_5', 'away_form_points_last_5',
            'home_avg_goals_scored_last_5', 'home_avg_goals_conceded_last_5',
            'away_avg_goals_scored_last_5', 'away_avg_goals_conceded_last_5'
        ]
        
        # Test scenarios
        scenarios = [
            {
                'name': 'Strong Home vs Weak Away',
                'data': {
                    'h2h_team1_wins': 5, 'h2h_draws': 1, 'h2h_team2_wins': 0, 'h2h_avg_goals': 3.0,
                    'home_formation': 1, 'away_formation': 2, 'formation_clash_score': 0.3,
                    'home_match_xg': 2.5, 'away_match_xg': 0.8, 'home_injury_impact': 0.0, 'away_injury_impact': 0.4,
                    'home_elo': 1800, 'away_elo': 1400, 'elo_diff': 400, 'home_form_points_last_5': 12, 'away_form_points_last_5': 3,
                    'home_avg_goals_scored_last_5': 3.0, 'home_avg_goals_conceded_last_5': 0.5,
                    'away_avg_goals_scored_last_5': 0.8, 'away_avg_goals_conceded_last_5': 2.5
                }
            },
            {
                'name': 'Even Teams',
                'data': {
                    'h2h_team1_wins': 2, 'h2h_draws': 2, 'h2h_team2_wins': 2, 'h2h_avg_goals': 2.5,
                    'home_formation': 1, 'away_formation': 1, 'formation_clash_score': 0.0,
                    'home_match_xg': 1.5, 'away_match_xg': 1.5, 'home_injury_impact': 0.1, 'away_injury_impact': 0.1,
                    'home_elo': 1500, 'away_elo': 1500, 'elo_diff': 0, 'home_form_points_last_5': 7, 'away_form_points_last_5': 7,
                    'home_avg_goals_scored_last_5': 1.5, 'home_avg_goals_conceded_last_5': 1.5,
                    'away_avg_goals_scored_last_5': 1.5, 'away_avg_goals_conceded_last_5': 1.5
                }
            },
            {
                'name': 'Weak Home vs Strong Away',
                'data': {
                    'h2h_team1_wins': 0, 'h2h_draws': 1, 'h2h_team2_wins': 5, 'h2h_avg_goals': 2.0,
                    'home_formation': 2, 'away_formation': 1, 'formation_clash_score': -0.2,
                    'home_match_xg': 0.8, 'away_match_xg': 2.5, 'home_injury_impact': 0.4, 'away_injury_impact': 0.0,
                    'home_elo': 1400, 'away_elo': 1800, 'elo_diff': -400, 'home_form_points_last_5': 3, 'away_form_points_last_5': 12,
                    'home_avg_goals_scored_last_5': 0.8, 'home_avg_goals_conceded_last_5': 2.5,
                    'away_avg_goals_scored_last_5': 3.0, 'away_avg_goals_conceded_last_5': 0.5
                }
            }
        ]
        
        all_uniform = True
        
        for scenario in scenarios:
            logger.info(f"\n--- Testing: {scenario['name']} ---")
            
            df = pd.DataFrame([scenario['data']])
            df = df[feature_names]  # Ensure correct order
            
            try:
                pred_proba = model.predict_proba(df)
                home_win, draw, away_win = pred_proba[0]
                
                logger.info(f"Probabilities: Home={home_win:.4f}, Draw={draw:.4f}, Away={away_win:.4f}")
                
                # Check if uniform
                uniform_threshold = 0.05
                is_uniform = (abs(home_win - 1/3) < uniform_threshold and 
                            abs(draw - 1/3) < uniform_threshold and 
                            abs(away_win - 1/3) < uniform_threshold)
                
                if is_uniform:
                    logger.warning("⚠️  UNIFORM DISTRIBUTION")
                else:
                    logger.info("✅ Non-uniform prediction")
                    all_uniform = False
                    
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
        
        return not all_uniform
        
    except Exception as e:
        logger.error(f"Different inputs test failed: {e}")
        return False

def main():
    """Run model debugging tests."""
    logger.info("🔍 Starting Model Debugging Tests")
    logger.info("=" * 60)
    
    # Test 1: Raw model functionality
    logger.info("Test 1: Raw Model Functionality")
    raw_test_result = test_raw_model()
    
    # Test 2: Different input scenarios
    logger.info("\nTest 2: Different Input Scenarios")
    scenarios_test_result = test_different_inputs()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🔍 DEBUG TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Raw Model Test: {'✅ PASSED' if raw_test_result else '❌ FAILED'}")
    logger.info(f"Scenarios Test: {'✅ PASSED' if scenarios_test_result else '❌ FAILED'}")
    
    if raw_test_result and scenarios_test_result:
        logger.info("🎉 Model is working correctly - issue is elsewhere!")
        return True
    else:
        logger.warning("⚠️  Model has fundamental issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
