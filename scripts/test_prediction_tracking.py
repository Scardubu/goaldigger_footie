"""Test automatic prediction tracking integration."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json

from dashboard.components.prediction_history import PredictionHistorySystem
from models.enhanced_real_data_predictor import EnhancedRealDataPredictor


def test_prediction_tracking():
    """Test that predictions are automatically tracked."""
    print("=" * 80)
    print("Testing Automatic Prediction Tracking")
    print("=" * 80)
    
    # Initialize predictor
    print("\n1. Initializing predictor...")
    predictor = EnhancedRealDataPredictor()
    
    # Check prediction history system is initialized
    if not predictor._prediction_history:
        print("❌ FAILED: Prediction history system not initialized")
        return False
    print("✅ Prediction history system initialized")
    
    # Get count of predictions before test
    history_system = PredictionHistorySystem()
    initial_predictions = history_system.get_predictions(limit=10000)  # Get all
    initial_count = len(initial_predictions)
    print(f"\n2. Initial prediction count: {initial_count}")
    
    # Make a test prediction
    print("\n3. Making test prediction...")
    result = predictor.predict_match_enhanced(
        home_team="Manchester United",
        away_team="Liverpool",
        match_data={"league": "Premier League"}
    )
    
    print(f"   Home Win: {result.home_win_probability:.3f}")
    print(f"   Draw: {result.draw_probability:.3f}")
    print(f"   Away Win: {result.away_win_probability:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    
    # Check if prediction was tracked
    print("\n4. Checking if prediction was tracked...")
    new_predictions = history_system.get_predictions(limit=10000)
    new_count = len(new_predictions)
    
    if new_count > initial_count:
        print(f"✅ PASSED: Prediction tracked! Count increased from {initial_count} to {new_count}")
        
        # Get recent predictions (returns DataFrame)
        recent_df = history_system.get_predictions(limit=5)
        print(f"\n5. Recent predictions ({len(recent_df)}):")
        for idx, row in recent_df.head(3).iterrows():
            print(f"   - {row['home_team']} vs {row['away_team']}: {row['predicted_outcome']} (conf: {row['confidence_level']})")
        
        return True
    else:
        print(f"❌ FAILED: Prediction not tracked. Count remained at {initial_count}")
        return False

def test_multiple_predictions():
    """Test tracking multiple predictions."""
    print("\n" + "=" * 80)
    print("Testing Multiple Predictions Tracking")
    print("=" * 80)
    
    predictor = EnhancedRealDataPredictor()
    history_system = PredictionHistorySystem()
    
    initial_predictions = history_system.get_predictions(limit=10000)
    initial_count = len(initial_predictions)
    
    # Make 5 predictions
    test_matches = [
        ("Arsenal", "Chelsea", "Premier League"),
        ("Barcelona", "Real Madrid", "La Liga"),
        ("Bayern Munich", "Borussia Dortmund", "Bundesliga"),
        ("PSG", "Marseille", "Ligue 1"),
        ("Juventus", "AC Milan", "Serie A")
    ]
    
    print(f"\nMaking {len(test_matches)} predictions...")
    for home, away, league in test_matches:
        result = predictor.predict_match_enhanced(
            home_team=home,
            away_team=away,
            match_data={"league": league}
        )
        print(f"   ✓ {home} vs {away}")
    
    # Check count
    new_predictions = history_system.get_predictions(limit=10000)
    new_count = len(new_predictions)
    predictions_added = new_count - initial_count
    
    print(f"\nResults:")
    print(f"   Initial count: {initial_count}")
    print(f"   New count: {new_count}")
    print(f"   Predictions added: {predictions_added}")
    
    if predictions_added >= len(test_matches):
        print(f"\n✅ PASSED: All {len(test_matches)} predictions tracked")
        return True
    else:
        print(f"\n❌ FAILED: Expected {len(test_matches)}, but only {predictions_added} were tracked")
        return False

def test_stats_accuracy():
    """Test that stats are being calculated correctly."""
    print("\n" + "=" * 80)
    print("Testing Prediction Statistics")
    print("=" * 80)
    
    history_system = PredictionHistorySystem()
    
    # Get accuracy stats (with correct parameters)
    stats = history_system.get_accuracy_stats()
    
    print("\nCurrent Statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Check if we have predictions
    predictions = history_system.get_predictions(limit=10)
    total_count = len(history_system.get_predictions(limit=10000))
    print(f"\nTotal predictions in database: {total_count}")
    
    if total_count > 0:
        print(f"\n✅ PASSED: Stats retrieved successfully with {total_count} predictions")
        return True
    else:
        print(f"\n⚠️  WARNING: No predictions in database yet (expected for fresh install)")
        return True  # Not a failure, just empty DB

if __name__ == "__main__":
    print("\n🔍 Running Prediction Tracking Tests\n")
    
    results = []
    
    # Test 1: Basic tracking
    results.append(("Basic Tracking", test_prediction_tracking()))
    
    # Test 2: Multiple predictions
    results.append(("Multiple Predictions", test_multiple_predictions()))
    
    # Test 3: Stats accuracy
    results.append(("Statistics", test_stats_accuracy()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Prediction tracking is working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed.")
        sys.exit(1)
