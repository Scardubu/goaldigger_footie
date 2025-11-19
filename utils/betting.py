# utils/betting.py
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# Removed 'class AIDataValidator:' line
def calculate_implied_probability(decimal_odd: float) -> Optional[float]:
    """
    Calculates the implied probability from a decimal odd.

    Args:
        decimal_odd (float): The decimal odd (e.g., 2.50).

    Returns:
        Optional[float]: The implied probability (e.g., 0.40), or None if odd is invalid.
    """
    if decimal_odd is None or decimal_odd <= 1.0:  # Odds must be greater than 1
        logger.warning(
            f"Invalid decimal odd received: {decimal_odd}. Cannot calculate implied probability."
        )
        return None
    return 1.0 / decimal_odd


def check_overround(odds_h: float, odds_d: float, odds_a: float) -> Optional[float]:
    """
    Calculates the bookmaker's margin (overround).

    Args:
        odds_h (float): Decimal odd for Home win.
        odds_d (float): Decimal odd for Draw.
        odds_a (float): Decimal odd for Away win.

    Returns:
        Optional[float]: The overround percentage (e.g., 5.5 for 5.5%), or None if any odd is invalid.
    """
    prob_h = calculate_implied_probability(odds_h)
    prob_d = calculate_implied_probability(odds_d)
    prob_a = calculate_implied_probability(odds_a)

    if None in [prob_h, prob_d, prob_a]:
        return None

    total_probability = prob_h + prob_d + prob_a
    overround = (total_probability - 1.0) * 100
    logger.debug(
        f"Calculated overround: {overround:.2f}% (Total Implied Prob: {total_probability:.4f})"
    )
    return overround

# Removed misplaced _detect_missing_columns method

def find_value_bets(
    model_probs: Dict[str, float],
    bookmaker_odds: Dict[str, float],
    value_threshold: float = 0.05,
) -> Dict[str, bool]:
    # Removed erroneous duplicate check block
    """
    Compares model probabilities with bookmaker odds to identify potential value bets.

    Args:
        model_probs (Dict[str, float]): Dictionary with model probabilities {'home_win': prob, 'draw': prob, 'away_win': prob}.
        bookmaker_odds (Dict[str, float]): Dictionary with decimal odds {'home_win': odd, 'draw': odd, 'away_win': odd}.
        value_threshold (float): Minimum difference required between model probability and implied probability
                                 to flag as a value bet (e.g., 0.05 means model prob must be 5% higher).

    Returns:
        Dict[str, bool]: Dictionary indicating if value is found for each outcome {'home_win': bool, 'draw': bool, 'away_win': bool}.
    """
    value_flags = {"home_win": False, "draw": False, "away_win": False}

    odds_map = {
        "home_win": bookmaker_odds.get("home_win"),
        "draw": bookmaker_odds.get("draw"),
        "away_win": bookmaker_odds.get("away_win"),
    }

    for outcome, model_prob in model_probs.items():
        if outcome not in odds_map:
            logger.warning(
                f"Outcome '{outcome}' from model not found in bookmaker odds keys."
            )
            continue

        decimal_odd = odds_map[outcome]
        implied_prob = calculate_implied_probability(decimal_odd)

        if model_prob is None or implied_prob is None:
            logger.debug(
                f"Skipping value check for '{outcome}' due to missing probability (Model: {model_prob}, Implied: {implied_prob})"
            )
            continue

        # Check if model probability significantly exceeds implied probability
        if model_prob > implied_prob + value_threshold:
            logger.info(
                f"Potential value bet found for '{outcome}': Model Prob ({model_prob:.3f}) > Implied Prob ({implied_prob:.3f}) + Threshold ({value_threshold})"
            )
            value_flags[outcome] = True
        else:
            logger.debug(
                f"No value for '{outcome}': Model Prob ({model_prob:.3f}) vs Implied Prob ({implied_prob:.3f})"
            )

    return value_flags


# Example Usage
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#
#     model_predictions = {'home_win': 0.55, 'draw': 0.25, 'away_win': 0.20}
#     bookie_odds = {'home_win': 2.00, 'draw': 3.50, 'away_win': 4.00} # Implied: 0.50, 0.286, 0.25
#
#     overround = check_overround(bookie_odds['home_win'], bookie_odds['draw'], bookie_odds['away_win'])
#     if overround is not None:
#         print(f"Bookmaker Margin (Overround): {overround:.2f}%")
#
#     value = find_value_bets(model_predictions, bookie_odds, value_threshold=0.02)
#     print(f"Value Bets Found: {value}") # Expected: {'home_win': True, 'draw': False, 'away_win': False}
#
#     # Example with invalid odd
#     invalid_odds = {'home_win': 1.0, 'draw': 3.50, 'away_win': 4.00}
#     value_invalid = find_value_bets(model_predictions, invalid_odds)
#     print(f"Value Bets (Invalid Odds): {value_invalid}") # Expected: {'home_win': False, 'draw': False, 'away_win': False}
