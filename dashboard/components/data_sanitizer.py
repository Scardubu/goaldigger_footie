"""
Data sanitization utility for dashboard visualizations.
Ensures all data passed to visualizations is properly validated and sanitized.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd

from dashboard.error_log import log_error

logger = logging.getLogger(__name__)

def sanitize_prediction_data(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    Sanitize prediction data to ensure it contains valid probabilities.
    
    Args:
        data: Raw prediction data dictionary
        
    Returns:
        Sanitized prediction data or None if invalid
    """
    if not data or not isinstance(data, dict):
        return None
        
    required_keys = ["home_win", "draw", "away_win"]
    
    # Check if all required keys exist
    if not all(key in data for key in required_keys):
        logger.warning(f"Missing required keys in prediction data: {data}")
        return None
        
    # Create sanitized copy
    sanitized = {}
    
    # Validate and sanitize each probability value
    for key in required_keys:
        try:
            # Convert to float and validate range
            value = float(data[key])
            
            # Ensure value is a valid probability (0-1)
            if value < 0:
                logger.warning(f"Negative probability found for {key}: {value}, clamping to 0")
                value = 0
            elif value > 1:
                logger.warning(f"Probability > 1 found for {key}: {value}, clamping to 1")
                value = 1
                
            sanitized[key] = value
        except (ValueError, TypeError) as e:
            log_error(f"Invalid probability value for {key}: {data[key]}", e)
            sanitized[key] = 0.0
            
    # Ensure probabilities sum to 1 (or close to it)
    prob_sum = sum(sanitized.values())
    
    if abs(prob_sum - 1.0) > 0.01:
        logger.warning(f"Probabilities don't sum to 1: {prob_sum}, normalizing")
        
        # Normalize if sum is not zero
        if prob_sum > 0:
            for key in sanitized:
                sanitized[key] /= prob_sum
        else:
            # Equal distribution if all are zero
            for key in sanitized:
                sanitized[key] = 1.0 / len(required_keys)
                
    return sanitized


def sanitize_odds_data(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    Sanitize bookmaker odds data to ensure it contains valid values.
    
    Args:
        data: Raw odds data dictionary
        
    Returns:
        Sanitized odds data or None if invalid
    """
    if not data or not isinstance(data, dict):
        return None
        
    required_keys = ["home_win", "draw", "away_win"]
    
    # Check if all required keys exist
    if not all(key in data for key in required_keys):
        logger.warning(f"Missing required keys in odds data: {data}")
        return None
        
    # Create sanitized copy
    sanitized = {}
    
    # Validate and sanitize each odds value
    for key in required_keys:
        try:
            # Convert to float and validate range
            value = float(data[key])
            
            # Ensure value is a valid odds (> 1.0)
            if value < 1.0:
                logger.warning(f"Invalid odds found for {key}: {value}, setting to 1.01")
                value = 1.01
                
            sanitized[key] = value
        except (ValueError, TypeError) as e:
            log_error(f"Invalid odds value for {key}: {data[key]}", e)
            sanitized[key] = 2.0  # Default to 2.0 (50% implied probability)
            
    return sanitized


def sanitize_feature_importance(data: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Sanitize feature importance data to ensure it contains valid values.
    
    Args:
        data: Raw feature importance dictionary
        
    Returns:
        Sanitized feature importance data (empty dict if invalid)
    """
    if not data or not isinstance(data, dict):
        return {}
        
    # Create sanitized copy
    sanitized = {}
    
    # Validate and sanitize each feature importance value
    for key, value in data.items():
        try:
            # Convert to float
            float_value = float(value)
            
            # Skip features with zero importance
            if float_value == 0:
                continue
                
            sanitized[key] = float_value
        except (ValueError, TypeError) as e:
            log_error(f"Invalid feature importance value for {key}: {value}", e)
            # Skip invalid values
            
    # Limit to top 15 features if there are too many
    if len(sanitized) > 15:
        logger.info(f"Limiting feature importance to top 15 features (from {len(sanitized)})")
        
        # Sort by absolute importance value and take top 15
        sorted_items = sorted(sanitized.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        sanitized = dict(sorted_items)
        
    return sanitized


def sanitize_dataframe(
    df: Optional[pd.DataFrame],
    required_columns: List[str] = None,
    max_rows: int = 1000
) -> Optional[pd.DataFrame]:
    """
    Sanitize a DataFrame to ensure it contains required columns and is not too large.
    
    Args:
        df: DataFrame to sanitize
        required_columns: List of required column names
        max_rows: Maximum number of rows allowed
        
    Returns:
        Sanitized DataFrame or None if invalid
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return None
        
    # Check if DataFrame is empty
    if df.empty:
        return df
        
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns in DataFrame: {missing_columns}")
            return None
            
    # Limit number of rows if too large
    if len(df) > max_rows:
        logger.warning(f"DataFrame too large ({len(df)} rows), limiting to {max_rows} rows")
        df = df.head(max_rows)
        
    # Handle NaN values
    df = df.fillna(value=np.nan)
    
    return df


def sanitize_match_details(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Sanitize match details data to ensure it contains valid values for visualization.
    
    Args:
        data: Raw match details dictionary
        
    Returns:
        Sanitized match details or None if invalid
    """
    if not data or not isinstance(data, dict):
        return None
        
    # Create sanitized copy
    sanitized = {}
    
    # Sanitize fixture details
    fixture_details = data.get("fixture_details", {})
    if fixture_details and isinstance(fixture_details, dict):
        sanitized["fixture_details"] = {
            "home_team_name": str(fixture_details.get("home_team_name", "Home Team")),
            "away_team_name": str(fixture_details.get("away_team_name", "Away Team")),
            "date_utc": str(fixture_details.get("date_utc", "Unknown Date")),
            "competition_name": str(fixture_details.get("competition_name", "Unknown League"))
        }
    else:
        sanitized["fixture_details"] = {
            "home_team_name": "Home Team",
            "away_team_name": "Away Team",
            "date_utc": "Unknown Date",
            "competition_name": "Unknown League"
        }
        
    # Sanitize prediction
    prediction = data.get("prediction", {})
    sanitized["prediction"] = sanitize_prediction_data(prediction)
    
    # Sanitize odds
    odds = data.get("bookie_odds", {})
    sanitized["bookie_odds"] = sanitize_odds_data(odds)
    
    # Sanitize stats (if present)
    stats = data.get("stats", {})
    if stats and isinstance(stats, dict):
        sanitized["stats"] = stats  # No specific sanitization for stats yet
    else:
        sanitized["stats"] = {}
        
    # Sanitize value bet info (if present)
    value_bet_info = data.get("value_bet_info", {})
    if value_bet_info and isinstance(value_bet_info, dict):
        sanitized["value_bet_info"] = {
            "value_home": bool(value_bet_info.get("value_home", False)),
            "value_draw": bool(value_bet_info.get("value_draw", False)),
            "value_away": bool(value_bet_info.get("value_away", False)),
            "edge_home": float(value_bet_info.get("edge_home", 0)),
            "edge_draw": float(value_bet_info.get("edge_draw", 0)),
            "edge_away": float(value_bet_info.get("edge_away", 0))
        }
    else:
        sanitized["value_bet_info"] = {
            "value_home": False,
            "value_draw": False,
            "value_away": False,
            "edge_home": 0.0,
            "edge_draw": 0.0,
            "edge_away": 0.0
        }
        
    return sanitized


def sanitize_ai_analysis_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sanitize AI analysis configuration to ensure it contains valid values.
    
    Args:
        config: Raw AI analysis configuration dictionary
        
    Returns:
        Sanitized AI analysis configuration
    """
    if not config or not isinstance(config, dict):
        return {
            "provider": "gemini",
            "model": "gemini-pro",
            "analysis_aspects": []
        }
        
    # Create sanitized copy
    sanitized = {
        "provider": str(config.get("provider", "gemini")),
        "model": str(config.get("model", "gemini-pro")),
        "analysis_aspects": []
    }
    
    # Sanitize analysis aspects
    aspects = config.get("analysis_aspects", [])
    if aspects and isinstance(aspects, list):
        for aspect in aspects:
            if not isinstance(aspect, dict):
                continue
                
            key = aspect.get("key")
            name = aspect.get("name")
            
            if not key or not name:
                continue
                
            sanitized["analysis_aspects"].append({
                "key": str(key),
                "name": str(name)
            })
            
    return sanitized


def sanitize_ai_analysis_results(results: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    Sanitize AI analysis results to ensure they contain valid values.
    
    Args:
        results: Raw AI analysis results dictionary
        
    Returns:
        Sanitized AI analysis results
    """
    if not results or not isinstance(results, dict):
        return {}
        
    # Create sanitized copy
    sanitized = {}
    
    # Sanitize each analysis aspect
    for key, value in results.items():
        # Ensure value is a string
        if value is None:
            sanitized[key] = "No analysis available."
        else:
            sanitized[key] = str(value)
            
    return sanitized
