#!/usr/bin/env python3
"""
Integration test for value betting visualization fixes.
Tests that all plotting functions handle edge cases correctly.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from dashboard.visualizations.value_bets import (
    _is_empty_input,
    _normalize_value_bets_to_df,
    plot_kelly_stakes,
    plot_probability_comparison,
    plot_value_bet_edge,
)


def test_empty_input_checks():
    """Test the _is_empty_input helper function"""
    print("\n✅ Testing empty input checks...")
    
    # Test None
    assert _is_empty_input(None), "None should be empty"
    
    # Test empty list
    assert _is_empty_input([]), "Empty list should be empty"
    
    # Test empty dict
    assert _is_empty_input({}), "Empty dict should be empty"
    
    # Test empty DataFrame
    assert _is_empty_input(pd.DataFrame()), "Empty DataFrame should be empty"
    
    # Test empty Series
    assert _is_empty_input(pd.Series(dtype=float)), "Empty Series should be empty"
    
    # Test non-empty list
    assert not _is_empty_input([1, 2, 3]), "Non-empty list should not be empty"
    
    # Test non-empty dict
    assert not _is_empty_input({"key": "value"}), "Non-empty dict should not be empty"
    
    # Test non-empty DataFrame
    assert not _is_empty_input(pd.DataFrame({"a": [1, 2]})), "Non-empty DataFrame should not be empty"
    
    print("   ✓ All empty input checks passed")


def test_normalize_value_bets():
    """Test the _normalize_value_bets_to_df function"""
    print("\n✅ Testing value bets normalization...")
    
    # Test with list of dicts
    bets_list = [
        {"bet_type": "Home Win", "odds": 2.5, "predicted_prob": 0.55, "edge": 0.175},
        {"bet_type": "Draw", "odds": 3.0, "predicted_prob": 0.30}
    ]
    df = _normalize_value_bets_to_df(bets_list)
    assert isinstance(df, pd.DataFrame), "Should return DataFrame"
    assert "edge" in df.columns, "Should have edge column"
    assert "implied_prob" in df.columns, "Should have implied_prob column"
    print("   ✓ List of dicts normalization passed")
    
    # Test with DataFrame missing edge
    bets_df = pd.DataFrame([
        {"bet_type": "Home Win", "odds": 2.5, "predicted_prob": 0.55},
    ])
    df = _normalize_value_bets_to_df(bets_df)
    assert "edge" in df.columns, "Should compute edge column"
    print("   ✓ DataFrame normalization with computed edge passed")
    
    # Test with Series (single bet)
    bets_series = pd.Series({"bet_type": "Home Win", "odds": 2.5, "predicted_prob": 0.55})
    df = _normalize_value_bets_to_df(bets_series)
    assert isinstance(df, pd.DataFrame), "Should convert Series to DataFrame"
    print("   ✓ Series normalization passed")


def test_plot_functions_with_empty_data():
    """Test that plotting functions handle empty data gracefully"""
    print("\n✅ Testing plot functions with empty data...")
    
    # Test plot_value_bet_edge with empty list
    fig = plot_value_bet_edge([])
    assert fig is not None, "Should return figure even with empty data"
    print("   ✓ plot_value_bet_edge with empty data passed")
    
    # Test plot_kelly_stakes with empty list
    fig = plot_kelly_stakes([])
    assert fig is not None, "Should return figure even with empty data"
    print("   ✓ plot_kelly_stakes with empty data passed")
    
    # Test plot_probability_comparison with empty list
    fig = plot_probability_comparison([])
    assert fig is not None, "Should return figure even with empty data"
    print("   ✓ plot_probability_comparison with empty data passed")


def test_plot_functions_with_valid_data():
    """Test that plotting functions work with valid data"""
    print("\n✅ Testing plot functions with valid data...")
    
    valid_bets = [
        {
            "bet_type": "Home Win",
            "odds": 2.5,
            "predicted_prob": 0.55,
            "edge": 0.175,
            "kelly_fraction": 0.05
        },
        {
            "bet_type": "Draw",
            "odds": 3.0,
            "predicted_prob": 0.30,
            "edge": 0.10,
            "kelly_fraction": 0.03
        }
    ]
    
    # Test plot_value_bet_edge
    fig = plot_value_bet_edge(valid_bets)
    assert fig is not None, "Should return figure"
    assert len(fig.data) > 0, "Should have plot traces"
    print("   ✓ plot_value_bet_edge with valid data passed")
    
    # Test plot_kelly_stakes
    fig = plot_kelly_stakes(valid_bets)
    assert fig is not None, "Should return figure"
    assert len(fig.data) > 0, "Should have plot traces"
    print("   ✓ plot_kelly_stakes with valid data passed")
    
    # Test plot_probability_comparison
    fig = plot_probability_comparison(valid_bets)
    assert fig is not None, "Should return figure"
    assert len(fig.data) > 0, "Should have plot traces"
    print("   ✓ plot_probability_comparison with valid data passed")


def test_plot_functions_with_series():
    """Test that plotting functions handle pandas Series input"""
    print("\n✅ Testing plot functions with Series input...")
    
    # Single bet as Series (what was causing the original error)
    single_bet = pd.Series({
        "bet_type": "Home Win",
        "odds": 2.5,
        "predicted_prob": 0.55,
        "edge": 0.175,
        "kelly_fraction": 0.05
    })
    
    # Test plot_kelly_stakes (this was failing with ambiguous truth value)
    try:
        fig = plot_kelly_stakes(single_bet)
        assert fig is not None, "Should return figure"
        print("   ✓ plot_kelly_stakes with Series passed")
    except ValueError as e:
        if "ambiguous" in str(e):
            print(f"   ✗ plot_kelly_stakes still has ambiguous truth value error")
            raise
        raise


def main():
    """Run all tests"""
    print("=" * 60)
    print("Value Betting Visualization Integration Tests")
    print("=" * 60)
    
    try:
        test_empty_input_checks()
        test_normalize_value_bets()
        test_plot_functions_with_empty_data()
        test_plot_functions_with_valid_data()
        test_plot_functions_with_series()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe value betting visualization fixes are working correctly.")
        print("You can now run the dashboard without KeyError or ValueError issues.")
        return 0
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
