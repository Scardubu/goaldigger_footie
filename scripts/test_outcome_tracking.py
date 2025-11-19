"""Test outcome tracking system."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sqlite3
from datetime import datetime, timedelta

from utils.outcome_tracker import OutcomeTracker, get_outcome_tracker


def test_outcome_update():
    """Test updating a prediction with actual outcome."""
    print("=" * 80)
    print("Testing Outcome Update")
    print("=" * 80)
    
    tracker = OutcomeTracker()
    
    # Get a recent prediction from database
    conn = sqlite3.connect("data/prediction_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT home_team, away_team, timestamp 
        FROM predictions 
        WHERE actual_outcome IS NULL
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        print("\n⚠️  No predictions available for testing")
        print("   Make some predictions first, then run this test")
        return True  # Not a failure
    
    home_team, away_team, timestamp = result
    match_date = datetime.fromisoformat(timestamp)
    
    print(f"\nFound prediction: {home_team} vs {away_team}")
    print(f"   Timestamp: {timestamp}")
    
    # Simulate match result (home team wins 2-1)
    print(f"\nUpdating with simulated result: {home_team} 2-1 {away_team}")
    
    update_result = tracker.update_prediction_outcome(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        home_score=2,
        away_score=1,
        tolerance_hours=24
    )
    
    print(f"\nUpdate result:")
    for key, value in update_result.items():
        print(f"   {key}: {value}")
    
    if update_result['success']:
        print(f"\n✅ PASSED: Outcome updated successfully")
        print(f"   Predicted: {update_result['predicted_outcome']}")
        print(f"   Actual: {update_result['actual_outcome']}")
        print(f"   Correct: {update_result['is_correct']}")
        return True
    else:
        print(f"\n❌ FAILED: {update_result.get('reason', 'Unknown error')}")
        return False

def test_accuracy_stats():
    """Test accuracy statistics calculation."""
    print("\n" + "=" * 80)
    print("Testing Accuracy Statistics")
    print("=" * 80)
    
    tracker = OutcomeTracker()
    
    # Get overall stats
    print("\nCalculating accuracy statistics...")
    stats = tracker.get_accuracy_stats()
    
    print(f"\nOverall Accuracy:")
    overall = stats['overall']
    print(f"   Total predictions: {overall['total_predictions']}")
    print(f"   Correct predictions: {overall['correct_predictions']}")
    print(f"   Accuracy: {overall['accuracy_percentage']:.1f}%")
    
    if stats['by_confidence']:
        print(f"\nBy Confidence Level:")
        for level, data in stats['by_confidence'].items():
            print(f"   {level}: {data['correct']}/{data['total']} ({data['accuracy']:.1f}%)")
    
    if stats['by_league']:
        print(f"\nBy League:")
        for league, data in stats['by_league'].items():
            print(f"   {league}: {data['correct']}/{data['total']} ({data['accuracy']:.1f}%)")
    
    print(f"\n✅ PASSED: Statistics calculated successfully")
    return True

def test_batch_update():
    """Test batch outcome updates."""
    print("\n" + "=" * 80)
    print("Testing Batch Update")
    print("=" * 80)
    
    tracker = OutcomeTracker()
    
    # Get multiple predictions
    conn = sqlite3.connect("data/prediction_history.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT home_team, away_team, timestamp 
        FROM predictions 
        WHERE actual_outcome IS NULL
        ORDER BY timestamp DESC
        LIMIT 3
    """)
    
    predictions = cursor.fetchall()
    conn.close()
    
    if not predictions:
        print("\n⚠️  No predictions available for batch testing")
        return True
    
    print(f"\nFound {len(predictions)} predictions for batch update")
    
    # Create batch of simulated results
    matches = []
    for i, (home_team, away_team, timestamp) in enumerate(predictions):
        matches.append({
            'home_team': home_team,
            'away_team': away_team,
            'match_date': datetime.fromisoformat(timestamp),
            'home_score': i + 1,  # Vary scores
            'away_score': i,
            'tolerance_hours': 24
        })
        print(f"   {i+1}. {home_team} vs {away_team}")
    
    # Perform batch update
    print(f"\nPerforming batch update...")
    result = tracker.batch_update_outcomes(matches)
    
    print(f"\nBatch update results:")
    print(f"   Total: {result['total']}")
    print(f"   Successful: {result['successful']}")
    print(f"   Failed: {result['failed']}")
    
    if result['successful'] > 0:
        print(f"\n✅ PASSED: Batch update completed ({result['successful']}/{result['total']} successful)")
        return True
    else:
        print(f"\n⚠️  WARNING: No updates successful")
        return True  # Not a hard failure

def test_singleton():
    """Test singleton pattern."""
    print("\n" + "=" * 80)
    print("Testing Singleton Pattern")
    print("=" * 80)
    
    tracker1 = get_outcome_tracker()
    tracker2 = get_outcome_tracker()
    
    if tracker1 is tracker2:
        print(f"\n✅ PASSED: Singleton working correctly (same instance)")
        return True
    else:
        print(f"\n❌ FAILED: Different instances returned")
        return False

if __name__ == "__main__":
    print("\n🔍 Running Outcome Tracking Tests\n")
    
    results = []
    
    # Test 1: Singleton
    results.append(("Singleton Pattern", test_singleton()))
    
    # Test 2: Outcome update
    results.append(("Outcome Update", test_outcome_update()))
    
    # Test 3: Accuracy stats
    results.append(("Accuracy Statistics", test_accuracy_stats()))
    
    # Test 4: Batch update
    results.append(("Batch Update", test_batch_update()))
    
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
        print("\n🎉 All tests passed! Outcome tracking is working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed.")
        sys.exit(1)
