"""
Feature generation utility for the GoalDiggers platform.
Provides functions to create mock features when real data is unavailable.
"""

import logging
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def create_mock_features(home_team: str, away_team: str, seed: Optional[int] = None) -> Dict[str, float]:
    """
    Create mock features for prediction when real features cannot be generated.
    This helps prevent errors when feature generation fails.
    
    Args:
        home_team: Name of home team
        away_team: Name of away team
        seed: Optional seed for random number generation
        
    Returns:
        Dictionary of mock features
    """
    logger.warning(f"Creating mock features for {home_team} vs {away_team}. This should not be used in production.")
    return {}
