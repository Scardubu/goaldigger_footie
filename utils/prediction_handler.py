"""
Prediction handler module to integrate XGBoost predictions with match analyses.
Serves as a bridge between the prediction models and the AI insights module.
"""
import logging
import os
import string
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dashboard.error_log import log_error
from models.xgboost_predictor import XGBoostPredictor
from utils.config import Config
from utils.fallback_handler import FallbackHandler, get_fallback_handler

try:  # Prefer rich team metadata when available
    from utils.team_data_enhancer import team_enhancer  # type: ignore
except Exception:  # pragma: no cover - optional dependency during tests
    team_enhancer = None  # type: ignore

_TEAM_ABBREVIATIONS: set[str] = {
    "FC",
    "CF",
    "SC",
    "AC",
    "AS",
    "SS",
    "AD",
    "UD",
    "CD",
    "BC",
    "BK",
    "FK",
    "IK",
    "SK",
    "SV",
    "AFC",
    "PSG",
    "QPR",
    "LAFC",
    "NYCFC",
    "TSG",
    "RB",
    "OGC",
    "BSC",
}


def _apply_abbreviation_fixes(name: str) -> str:
    """Ensure common football abbreviations remain uppercase."""

    def _normalize_token(token: str) -> str:
        if "-" in token:
            segments = token.split("-")
            patched = [seg.upper() if seg.upper() in _TEAM_ABBREVIATIONS else seg for seg in segments]
            return "-".join(patched)
        if token.upper() in _TEAM_ABBREVIATIONS:
            return token.upper()
        return token

    tokens = name.split()
    corrected = [
        token
        if not token or token.strip("&.").upper() not in _TEAM_ABBREVIATIONS
        else token.strip().upper()
        for token in tokens
    ]

    # Re-run normalization for tokens that include punctuation or hyphenation
    corrected = [_normalize_token(token) for token in corrected]
    return " ".join(corrected)


def _reconstruct_team_name(slug_fragment: str) -> str:
    """Rebuild a human-friendly team name from a cached slug fragment."""

    candidate = slug_fragment.replace("_", " ").strip()
    if not candidate:
        return candidate

    if team_enhancer:
        lookup_variants = {candidate, candidate.title(), candidate.upper()}
        for variant in lookup_variants:
            try:
                metadata = team_enhancer.get_team_data(variant)  # type: ignore[attr-defined]
                if metadata:
                    return (
                        metadata.get("full_name")
                        or metadata.get("display_name")
                        or metadata.get("name")
                        or candidate
                    )
            except Exception:
                continue

    title_case = string.capwords(candidate)
    return _apply_abbreviation_fixes(title_case)

logger = logging.getLogger(__name__)

class PredictionHandler:
    """
    Manages the integration between prediction models and match analysis.
    Provides a unified interface for getting predictions and explanations.
    """
    
    def __init__(self):
        """Initialize the prediction handler with required models and fallback mechanisms."""
        self.xgb_model = None
        self.config = Config.get("models.prediction", {})
        self.cache = {}
        self.fallback_handler = get_fallback_handler()
        self.performance_threshold = self.config.get("performance_threshold", 0.5)  # seconds
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all required prediction models using singleton pattern."""
        try:
            # Use singleton pattern for production performance
            from utils.model_singleton import get_model_manager
            model_manager = get_model_manager()

            # Get model path from config or use default
            model_path = self.config.get("xgboost_model_path")

            if not model_path:
                # Default model path
                model_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "models", "trained", "predictor_model.joblib"
                )

            # Get cached XGBoost predictor from singleton
            self.xgb_model = model_manager.get_xgboost_predictor(model_path)

            if not self.xgb_model or not self.xgb_model.model:
                logger.warning(f"Failed to load XGBoost model from {model_path}")
            else:
                logger.info(f"Successfully loaded XGBoost model with {len(self.xgb_model.feature_columns)} features")

        except Exception as e:
            log_error("Error loading prediction models", e)
            self.xgb_model = None
    
    def get_match_prediction(self, match_features: pd.DataFrame, match_id: Optional[str] = None,
                             home_team: Optional[str] = None, away_team: Optional[str] = None,
                             home_team_data: Optional[Dict] = None, away_team_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get predictions and explanations for a single match with robust error handling.
        
        Args:
            match_features: DataFrame with match features
            match_id: Optional match ID for caching
            home_team: Optional home team name for personalized fallbacks
            away_team: Optional away team name for personalized fallbacks
            
        Returns:
            Dict with predictions and explanations
        """
        # Extract team names from match_id if not provided
        if match_id and '_vs_' in match_id:
            try:
                slug_parts = match_id.split('_vs_', 1)
                if len(slug_parts) == 2:
                    if not home_team:
                        home_team = _reconstruct_team_name(slug_parts[0])
                    if not away_team:
                        away_team = _reconstruct_team_name(slug_parts[1])
                elif not home_team:
                    home_team = _reconstruct_team_name(slug_parts[0])
            except Exception:
                pass  # Ignore extraction errors
        
        # Check cache
        if match_id and match_id in self.cache:
            logger.debug(f"Using cached prediction for match {match_id}")
            return self.cache[match_id]
        
        start_time = time.time()
        
        # Use fallback as initial result
        result = self._get_fallback_prediction(home_team, away_team)
        
        try:
            # Ensure DataFrame is not empty
            if match_features is None or match_features.empty:
                logger.warning("Empty features provided for prediction")
                result["status"] = "error"
                result["error"] = "No features available for prediction"
                return result
            if self.xgb_model and self.xgb_model.model:
                # Fill missing features
                missing = [f for f in self.xgb_model.feature_columns if f not in match_features.columns]
                extra = [f for f in match_features.columns if f not in self.xgb_model.feature_columns]
                if missing:
                    logger.warning(f"Missing features for prediction: {missing}. Imputing with 0.0.")
                    for feature in missing:
                        match_features[feature] = 0.0
                if extra:
                    logger.warning(f"Extra features provided for prediction: {extra}. Dropping them.")
                    match_features = match_features.drop(columns=extra)
                match_features = match_features[self.xgb_model.feature_columns]
                predictions, explanations = self.xgb_model.predict(match_features)
                if len(predictions) > 0 and len(predictions[0]) == 3:
                    pred = predictions[0]
                    if np.isnan(pred).any() or np.isinf(pred).any():
                        logger.warning("Invalid prediction values detected (NaN or Inf). Using fallback.")
                        result = self._get_fallback_prediction(home_team, away_team)
                        result["status"] = "error"
                        result["error"] = "Invalid prediction values"
                    else:
                        base_confidence = float(np.max(pred))

                        # Apply cross-league adjustments if team data is available
                        if home_team_data and away_team_data:
                            adjusted_confidence = self._apply_cross_league_adjustments(
                                pred, base_confidence, home_team_data, away_team_data
                            )
                        else:
                            adjusted_confidence = base_confidence

                        result = {
                            "home_win": float(pred[0]),
                            "draw": float(pred[1]),
                            "away_win": float(pred[2]),
                            "confidence": adjusted_confidence,
                            "base_confidence": base_confidence,
                            "explanations": explanations,
                            "status": "success"
                        }

                        # Add cross-league insights if applicable
                        if home_team_data and away_team_data:
                            result = self._add_cross_league_insights(result, home_team_data, away_team_data)
                        logger.debug(f"Generated prediction: Home={result['home_win']:.3f}, Draw={result['draw']:.3f}, Away={result['away_win']:.3f}")
                else:
                    logger.warning("Invalid prediction output shape, using fallback.")
                    result = self._get_fallback_prediction(home_team, away_team)
                    result["status"] = "error"
                    result["error"] = "Invalid prediction output shape"
            else:
                logger.warning("No XGBoost model available for predictions")
                result["status"] = "not_available"
                result["error"] = "XGBoost model not loaded"
        except Exception as e:
            error_detail = f"Error generating match prediction: {str(e)}\n{traceback.format_exc()}"
            log_error(error_detail, e)
            result["status"] = "error"
            result["error"] = str(e)
        
        # Cache result
        if match_id:
            self.cache[match_id] = result
            
        # Monitor performance
        pred_time = time.time() - start_time
        logger.debug(f"Prediction completed in {pred_time:.4f}s")
        
        # Log performance issues
        if pred_time > self.performance_threshold:
            self.fallback_handler.log_performance_issue(
                "match_prediction", pred_time, threshold=self.performance_threshold)
        
        return result
    
    def get_batch_predictions(self, features_df: pd.DataFrame, match_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for multiple matches.
        
        Args:
            features_df: DataFrame with match features
            match_ids: Optional list of match IDs for indexing results
            
        Returns:
            Dict mapping match IDs to prediction results
        """
        results = {}
        
        # Default match IDs if not provided
        if not match_ids:
            match_ids = [f"match_{i}" for i in range(len(features_df))]
        
        # Ensure we have the right number of match IDs
        if len(match_ids) != len(features_df):
            logger.warning(f"Mismatch between features ({len(features_df)} rows) and match IDs ({len(match_ids)})")
            match_ids = match_ids[:len(features_df)] if len(match_ids) > len(features_df) else match_ids + [f"match_{i+len(match_ids)}" for i in range(len(features_df) - len(match_ids))]
        
        start_time = time.time()
        
        try:
            if self.xgb_model and self.xgb_model.model:
                # Get all predictions at once
                predictions, explanations = self.xgb_model.predict(features_df)
                
                # Process each prediction
                for i, match_id in enumerate(match_ids):
                    if i < len(predictions):
                        pred = predictions[i]
                        
                        # Create individual result
                        results[match_id] = {
                            "home_win": float(pred[0]),
                            "draw": float(pred[1]),
                            "away_win": float(pred[2]),
                            "confidence": float(np.max(pred)),
                            "explanations": explanations,  # Same explanations for all matches
                            "status": "success"
                        }
                        
                        # Cache result
                        self.cache[match_id] = results[match_id]
                    else:
                        # Fallback for any missing predictions
                        results[match_id] = self._get_fallback_prediction()
            else:
                # Use fallback for all matches
                for match_id in match_ids:
                    results[match_id] = self._get_fallback_prediction()
                    
        except Exception as e:
            log_error("Error generating batch predictions", e)
            # Use fallback for all matches
            for match_id in match_ids:
                results[match_id] = self._get_fallback_prediction()
                results[match_id]["status"] = "error"
                results[match_id]["error"] = str(e)
        
        # Log performance
        batch_time = time.time() - start_time
        logger.debug(f"Batch prediction for {len(match_ids)} matches completed in {batch_time:.4f}s")
        
        return results
    
    def _get_fallback_prediction(self, home_team: Optional[str] = None, away_team: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a fallback prediction when the model fails, with optional home advantage.
        
        Args:
            home_team: Optional home team name for personalizing fallback
            away_team: Optional away team name for personalizing fallback
            
        Returns:
            Dict with default prediction values
        """
        # Use the fallback handler for more sophisticated fallbacks
        home_edge = 0.05 if home_team and away_team else 0.0
        prediction = self.fallback_handler.get_fallback_prediction(home_edge)
        
        # Add explanations field required by the UI
        prediction["explanations"] = {
            "top_features": {
                "home_form": 0.5,
                "away_form": 0.45,
                "goals_scored_home": 0.4,
                "goals_scored_away": 0.38,
                "recent_performance_home": 0.3
            }
        }
        
        return prediction

    def _apply_cross_league_adjustments(self, pred: np.ndarray, base_confidence: float,
                                       home_team_data: Dict, away_team_data: Dict) -> float:
        """Apply confidence adjustments for cross-league matches."""
        try:
            from utils.cross_league_handler import CrossLeagueHandler
            cross_league_handler = CrossLeagueHandler()

            home_league = home_team_data.get('league_name', 'Unknown')
            away_league = away_team_data.get('league_name', 'Unknown')

            adjusted_confidence = cross_league_handler.calculate_cross_league_confidence_adjustment(
                home_league, away_league, base_confidence
            )

            logger.debug(f"Cross-league confidence adjustment: {base_confidence:.3f} -> {adjusted_confidence:.3f}")
            return adjusted_confidence

        except Exception as e:
            logger.error(f"Error applying cross-league adjustments: {e}")
            return base_confidence

    def _add_cross_league_insights(self, result: Dict[str, Any],
                                  home_team_data: Dict, away_team_data: Dict) -> Dict[str, Any]:
        """Add cross-league specific insights to prediction result."""
        try:
            from utils.cross_league_handler import CrossLeagueHandler
            cross_league_handler = CrossLeagueHandler()

            enhanced_result = cross_league_handler.generate_cross_league_insights(
                home_team_data, away_team_data, result
            )

            return enhanced_result

        except Exception as e:
            logger.error(f"Error adding cross-league insights: {e}")
            return result

    def get_top_features(self, prediction_result: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Extract the top features from a prediction result.
        
        Args:
            prediction_result: Prediction result from get_match_prediction
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if not prediction_result or "explanations" not in prediction_result:
            return []
            
        explanations = prediction_result.get("explanations", {})
        top_features = explanations.get("top_features", {})
        
        # Convert to list of tuples
        return [(k, v) for k, v in top_features.items()]
    
    def format_prediction_for_display(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format prediction for display in the UI.
        
        Args:
            prediction: Raw prediction from get_match_prediction
            
        Returns:
            Dict with formatted values for display
        """
        if not prediction:
            return self._get_fallback_prediction()
            
        # Round probabilities to percentage
        home_win = round(prediction.get("home_win", 0.33) * 100)
        draw = round(prediction.get("draw", 0.34) * 100)
        away_win = round(prediction.get("away_win", 0.33) * 100)
        
        # Get top features
        top_features = self.get_top_features(prediction)
        formatted_features = [
            {
                "name": self._format_feature_name(feature),
                "importance": round(importance * 100) / 100
            }
            for feature, importance in top_features[:5]  # Limit to top 5
        ]
        
        return {
            "home_win_pct": home_win,
            "draw_pct": draw,
            "away_win_pct": away_win,
            "confidence": round(prediction.get("confidence", 0) * 100),
            "top_features": formatted_features,
            "status": prediction.get("status", "unknown")
        }
    
    def _format_feature_name(self, feature_name: str) -> str:
        """
        Format feature names for display.
        
        Args:
            feature_name: Raw feature name
            
        Returns:
            Human-readable feature name
        """
        # Remove prefixes, replace underscores with spaces, and capitalize
        name = feature_name.split('.')[-1]  # Remove any prefixes
        name = name.replace('_', ' ').title()
        
        # Handle special abbreviations
        name = name.replace('Xg', 'xG')
        name = name.replace('Ht', 'HT')
        name = name.replace('Ft', 'FT')
        
        return name
