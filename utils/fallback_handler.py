"""
Fallback handler for the football betting insights platform.
Provides default predictions and graceful error handling when models or data are unavailable.
"""
import logging
import time
from typing import Dict, Any, Optional, List, Union
import traceback

import pandas as pd
import numpy as np

from dashboard.error_log import log_error

logger = logging.getLogger(__name__)

class FallbackHandler:
    """
    Provides fallback mechanisms for prediction and analysis failures.
    Ensures system resilience in production environments.
    """
    
    def __init__(self):
        """Initialize the fallback handler."""
        self.default_prediction = {
            "home_win": 0.33,
            "draw": 0.34,
            "away_win": 0.33,
            "confidence": 0.0,
            "status": "fallback"
        }
        
        self.default_features = {
            "home_form": 0.5,
            "away_form": 0.5,
            "home_goals_avg": 1.2,
            "away_goals_avg": 1.2,
            "league_position_home": 10,
            "league_position_away": 10,
            "home_win_rate": 0.5,
            "away_win_rate": 0.5
        }
        
        self.default_analysis = {
            "summary": "Analysis is currently unavailable. Please try again later.",
            "tactical": "Tactical analysis is unavailable.",
            "key_players": "Key player analysis is unavailable.",
            "historical": "Historical analysis is unavailable.",
            "betting": "Betting analysis is unavailable.",
            "form": "Form analysis is unavailable."
        }
    
    def get_fallback_prediction(self, home_edge: float = 0.0) -> Dict[str, Any]:
        """
        Get a fallback prediction with optional home advantage.
        
        Args:
            home_edge: Value to add to home win probability (-0.1 to 0.1)
            
        Returns:
            Dictionary with fallback prediction values
        """
        # Clamp home_edge to reasonable values
        home_edge = max(min(home_edge, 0.1), -0.1)
        
        # Create prediction with slight adjustment
        prediction = self.default_prediction.copy()
        
        if home_edge != 0:
            # Adjust probabilities with home edge
            prediction["home_win"] = max(0.0, min(1.0, prediction["home_win"] + home_edge))
            prediction["away_win"] = max(0.0, min(1.0, prediction["away_win"] - home_edge * 0.7))
            prediction["draw"] = max(0.0, min(1.0, prediction["draw"] - home_edge * 0.3))
            
            # Normalize to ensure sum = 1
            total = prediction["home_win"] + prediction["draw"] + prediction["away_win"]
            prediction["home_win"] /= total
            prediction["draw"] /= total
            prediction["away_win"] /= total
        
        return prediction
    
    def get_fallback_features(self) -> Dict[str, float]:
        """
        Get fallback feature values for prediction.
        
        Returns:
            Dictionary of default feature values
        """
        return self.default_features.copy()
    
    def get_fallback_analysis(self, home_team: str, away_team: str) -> Dict[str, str]:
        """
        Get fallback analysis text with team names incorporated.
        
        Args:
            home_team: Name of home team
            away_team: Name of away team
            
        Returns:
            Dictionary of analysis sections
        """
        analysis = self.default_analysis.copy()
        
        # Add team names to make it look more personalized
        analysis["summary"] = f"Analysis for {home_team} vs {away_team} is currently unavailable. Please try again later."
        analysis["betting"] = f"Betting analysis for {home_team} vs {away_team} is unavailable."
        
        return analysis
    
    def handle_prediction_error(
        self, 
        error: Exception, 
        context: str, 
        home_team: Optional[str] = None,
        away_team: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle prediction errors with detailed logging and fallback values.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            home_team: Optional home team name
            away_team: Optional away team name
            
        Returns:
            Fallback prediction dictionary
        """
        # Log the error with full traceback
        error_detail = f"Error in {context}: {str(error)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        
        # Log to error tracking system
        log_error(f"Prediction error in {context}", error)
        
        # Create prediction with slight home advantage if teams are provided
        home_edge = 0.05 if home_team and away_team else 0.0
        prediction = self.get_fallback_prediction(home_edge)
        
        # Add error information
        prediction["error"] = str(error)
        prediction["error_context"] = context
        
        return prediction
    
    def create_fallback_dataframe(self, template_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create a fallback DataFrame for feature generation or predictions.
        
        Args:
            template_df: Optional template DataFrame to match structure
            
        Returns:
            DataFrame with fallback values
        """
        features = self.get_fallback_features()
        
        if template_df is not None and not template_df.empty:
            # Match the structure of the template DataFrame
            df = pd.DataFrame(columns=template_df.columns)
            for col in df.columns:
                if col in features:
                    df[col] = [features[col]]
                else:
                    # Use a reasonable default for unknown columns
                    df[col] = [0.0]
        else:
            # Create a basic DataFrame with default features
            df = pd.DataFrame([features])
        
        return df
    
    def log_performance_issue(self, operation: str, duration: float, threshold: float = 1.0):
        """
        Log performance issues when operations take too long.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            threshold: Threshold in seconds to consider slow
        """
        if duration > threshold:
            logger.warning(f"Performance issue: {operation} took {duration:.2f}s (threshold: {threshold:.2f}s)")


# Global singleton instance
fallback_handler = FallbackHandler()


def get_fallback_handler() -> FallbackHandler:
    """
    Get the global fallback handler instance.
    
    Returns:
        FallbackHandler instance
    """
    global fallback_handler
    return fallback_handler
