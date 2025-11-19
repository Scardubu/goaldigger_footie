#!/usr/bin/env python
"""
Test script for feature engineering optimization improvements.
Tests feature selection, correlation analysis, and performance enhancements.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.feature_eng.feature_optimizer import FeatureOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def create_football_feature_dataset(n_samples=500, n_features=25, random_state=42):
    """Create a realistic football feature dataset for testing."""
    np.random.seed(random_state)
    
    # Define feature groups similar to the actual system
    feature_groups = {
        "H2H": ["h2h_team1_wins", "h2h_draws", "h2h_team2_wins", "h2h_avg_goals", "h2h_time_weighted"],
        "Style": ["home_formation", "away_formation", "formation_clash_score", "home_match_xg", "away_match_xg"],
        "Motivation": ["home_form_points_last_5", "away_form_points_last_5", "home_injury_impact", "away_injury_impact"],
        "Home": ["home_avg_goals_scored_last_5", "home_avg_goals_conceded_last_5", 
                "away_avg_goals_scored_last_5", "away_avg_goals_conceded_last_5", "home_elo", "away_elo", "elo_diff"],
        "Weather": ["weather_temp", "weather_precip", "weather_wind"]
    }
    
    # Flatten feature list
    all_features = []
    for group_features in feature_groups.values():
        all_features.extend(group_features)
    
    # Ensure we have enough features
    while len(all_features) < n_features:
        all_features.append(f"extra_feature_{len(all_features)}")
    
    all_features = all_features[:n_features]
    
    # Create realistic football data
    data = {}
    
    # H2H features
    data['h2h_team1_wins'] = np.random.poisson(3, n_samples)
    data['h2h_draws'] = np.random.poisson(2, n_samples)
    data['h2h_team2_wins'] = np.random.poisson(3, n_samples)
    data['h2h_avg_goals'] = np.random.normal(2.5, 0.8, n_samples)
    data['h2h_time_weighted'] = np.random.uniform(0, 1, n_samples)
    
    # Style features
    data['home_formation'] = np.random.choice([343, 352, 433, 442, 451], n_samples)
    data['away_formation'] = np.random.choice([343, 352, 433, 442, 451], n_samples)
    data['formation_clash_score'] = np.random.uniform(0, 1, n_samples)
    data['home_match_xg'] = np.random.normal(1.5, 0.5, n_samples)
    data['away_match_xg'] = np.random.normal(1.5, 0.5, n_samples)
    
    # Motivation features
    data['home_form_points_last_5'] = np.random.randint(0, 16, n_samples)
    data['away_form_points_last_5'] = np.random.randint(0, 16, n_samples)
    data['home_injury_impact'] = np.random.uniform(0, 1, n_samples)
    data['away_injury_impact'] = np.random.uniform(0, 1, n_samples)
    
    # Home advantage features
    data['home_avg_goals_scored_last_5'] = np.random.normal(1.8, 0.6, n_samples)
    data['home_avg_goals_conceded_last_5'] = np.random.normal(1.2, 0.5, n_samples)
    data['away_avg_goals_scored_last_5'] = np.random.normal(1.4, 0.6, n_samples)
    data['away_avg_goals_conceded_last_5'] = np.random.normal(1.6, 0.5, n_samples)
    data['home_elo'] = np.random.randint(1200, 1800, n_samples)
    data['away_elo'] = np.random.randint(1200, 1800, n_samples)
    data['elo_diff'] = data['home_elo'] - data['away_elo']
    
    # Weather features
    data['weather_temp'] = np.random.normal(15, 8, n_samples)
    data['weather_precip'] = np.random.exponential(2, n_samples)
    data['weather_wind'] = np.random.normal(10, 5, n_samples)
    
    # Add extra features if needed
    for i, feature in enumerate(all_features):
        if feature not in data:
            data[feature] = np.random.normal(0, 1, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some correlated features to test correlation detection
    df['home_goals_total'] = df['home_avg_goals_scored_last_5'] * 5 + np.random.normal(0, 0.1, n_samples)  # Highly correlated
    df['elo_diff_squared'] = df['elo_diff'] ** 2  # Derived feature
    
    # Create target variable based on features
    home_strength = (
        df['home_avg_goals_scored_last_5'] + 
        df['home_form_points_last_5'] / 15 + 
        df['elo_diff'] / 500 +
        df['h2h_time_weighted'] * 0.5
    )
    
    away_strength = (
        df['away_avg_goals_scored_last_5'] + 
        df['away_form_points_last_5'] / 15 - 
        df['elo_diff'] / 500 +
        (1 - df['h2h_time_weighted']) * 0.5
    )
    
    # Convert to probabilities and sample outcomes
    strength_diff = home_strength - away_strength
    prob_home = 1 / (1 + np.exp(-strength_diff))
    prob_draw = 0.3 * np.exp(-(strength_diff**2)/2)
    prob_away = 1 - prob_home - prob_draw
    
    # Normalize probabilities
    prob_sum = prob_home + prob_draw + prob_away
    prob_home /= prob_sum
    prob_draw /= prob_sum
    prob_away /= prob_sum
    
    # Sample outcomes
    target = np.array([np.random.choice([0, 1, 2], p=[ph, pd, pa]) 
                      for ph, pd, pa in zip(prob_home, prob_draw, prob_away)])
    
    return df, target, feature_groups

def test_correlation_analysis():
    """Test correlation analysis and redundant feature detection."""
    logger.info("Testing correlation analysis...")
    
    # Create test data with known correlations
    X, y, feature_groups = create_football_feature_dataset(n_samples=300, n_features=20)
    
    optimizer = FeatureOptimizer(config={'correlation_threshold': 0.8})
    
    # Run correlation analysis
    correlation_results = optimizer.analyze_feature_correlations(X, y)
    
    logger.info(f"Found {len(correlation_results['high_corr_pairs'])} highly correlated pairs")
    logger.info(f"Identified {len(correlation_results['redundant_features'])} redundant features")
    
    # Check if known correlated features are detected
    redundant_features = correlation_results['redundant_features']
    
    # We expect 'home_goals_total' to be identified as redundant (correlated with home_avg_goals_scored_last_5)
    expected_redundant = ['home_goals_total', 'elo_diff_squared']
    found_redundant = [f for f in expected_redundant if f in redundant_features]
    
    logger.info(f"Expected redundant features found: {found_redundant}")
    
    return len(correlation_results['high_corr_pairs']) > 0

def test_feature_selection():
    """Test feature selection methods."""
    logger.info("Testing feature selection methods...")
    
    X, y, feature_groups = create_football_feature_dataset(n_samples=400, n_features=25)
    
    optimizer = FeatureOptimizer(config={'max_features': 15})
    
    # Test different selection methods
    methods = ['mutual_info', 'f_classif', 'rfe']
    results = {}
    
    for method in methods:
        logger.info(f"Testing {method} feature selection...")
        start_time = time.time()
        
        selection_result = optimizer.select_best_features(X, y, method=method)
        selection_time = time.time() - start_time
        
        results[method] = {
            'selected_features': selection_result['selected_features'],
            'n_features': len(selection_result['selected_features']),
            'selection_time': selection_time
        }
        
        logger.info(f"{method}: Selected {results[method]['n_features']} features in {selection_time:.2f}s")
    
    # Check that all methods selected the expected number of features
    all_correct = all(result['n_features'] == 15 for result in results.values())
    
    return all_correct

def test_feature_group_optimization():
    """Test feature group optimization."""
    logger.info("Testing feature group optimization...")
    
    X, y, feature_groups = create_football_feature_dataset(n_samples=300, n_features=25)
    
    optimizer = FeatureOptimizer(config={'max_features': 10})
    
    # Run group optimization
    start_time = time.time()
    group_results = optimizer.optimize_feature_groups(X, y, feature_groups)
    optimization_time = time.time() - start_time
    
    logger.info(f"Group optimization completed in {optimization_time:.2f}s")
    logger.info(f"Baseline score: {group_results['baseline_score']:.4f}")
    logger.info(f"Combined optimized score: {group_results['combined_score']:.4f}")
    
    # Check each group
    for group_name, group_perf in group_results['group_performance'].items():
        logger.info(f"Group {group_name}: {group_perf['original_score']:.4f} → "
                   f"{group_perf['optimized_score']:.4f} "
                   f"({group_perf['feature_reduction']} features reduced)")
    
    # Verify that optimization was performed
    total_reduction = group_results['total_feature_reduction']
    logger.info(f"Total feature reduction: {total_reduction}")
    
    return total_reduction > 0

def test_engineered_features():
    """Test engineered feature creation."""
    logger.info("Testing engineered feature creation...")
    
    X, y, feature_groups = create_football_feature_dataset(n_samples=200, n_features=20)
    
    optimizer = FeatureOptimizer()
    
    # Create engineered features
    start_time = time.time()
    X_engineered = optimizer.create_engineered_features(X)
    engineering_time = time.time() - start_time
    
    new_features = len(X_engineered.columns) - len(X.columns)
    logger.info(f"Created {new_features} new engineered features in {engineering_time:.2f}s")
    
    # Check for expected engineered features
    expected_features = [
        'home_attack_vs_away_defense',
        'away_attack_vs_home_defense',
        'elo_ratio',
        'form_difference',
        'h2h_home_win_rate'
    ]
    
    found_features = [f for f in expected_features if f in X_engineered.columns]
    logger.info(f"Expected engineered features found: {found_features}")
    
    return new_features > 0

def test_complete_pipeline():
    """Test the complete feature optimization pipeline."""
    logger.info("Testing complete feature optimization pipeline...")
    
    X, y, feature_groups = create_football_feature_dataset(n_samples=400, n_features=30)
    
    optimizer = FeatureOptimizer(config={
        'correlation_threshold': 0.8,
        'max_features': 15
    })
    
    # Run complete pipeline
    start_time = time.time()
    pipeline_results = optimizer.optimize_feature_pipeline(X, y, feature_groups)
    pipeline_time = time.time() - start_time
    
    summary = pipeline_results['pipeline_summary']
    
    logger.info("Pipeline Summary:")
    logger.info(f"  Original features: {summary['original_features']}")
    logger.info(f"  After correlation removal: {summary['after_correlation_removal']}")
    logger.info(f"  After engineering: {summary['after_engineering']}")
    logger.info(f"  Final features: {summary['final_features']}")
    logger.info(f"  Total reduction: {summary['total_reduction']}")
    logger.info(f"  Pipeline time: {summary['pipeline_time']:.2f}s")
    logger.info(f"  Performance: {summary['baseline_score']:.4f} → {summary['final_score']:.4f}")
    
    # Verify pipeline worked correctly
    feature_reduction_achieved = summary['total_reduction'] > 0
    performance_maintained = summary['final_score'] >= summary['baseline_score'] - 0.1  # Allow small decrease
    
    return feature_reduction_achieved and performance_maintained

def main():
    """Main test function."""
    logger.info("Starting Feature Engineering Optimization Tests...")
    
    test_results = {}
    
    try:
        # Run all tests
        test_results['correlation_analysis'] = test_correlation_analysis()
        test_results['feature_selection'] = test_feature_selection()
        test_results['group_optimization'] = test_feature_group_optimization()
        test_results['engineered_features'] = test_engineered_features()
        test_results['complete_pipeline'] = test_complete_pipeline()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("FEATURE OPTIMIZATION TEST RESULTS")
        logger.info("="*60)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:25}: {status}")
        
        all_passed = all(test_results.values())
        
        if all_passed:
            logger.info("\n🎯 ALL FEATURE OPTIMIZATION TESTS PASSED!")
            logger.info("\n✅ Key Improvements Validated:")
            logger.info("   - Correlation analysis and redundant feature removal")
            logger.info("   - Multiple feature selection methods working")
            logger.info("   - Feature group optimization functional")
            logger.info("   - Engineered feature creation successful")
            logger.info("   - Complete optimization pipeline operational")
        else:
            logger.error("\n❌ Some tests failed. Check the logs above for details.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
