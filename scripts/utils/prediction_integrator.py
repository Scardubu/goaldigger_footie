"""
Prediction integrator for combining multiple model predictions.
Provides weighted ensemble of predictions and value betting calculations.
"""
import logging
import statistics
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class PredictionIntegrator:
    """
    Integrates predictions from multiple models and calculates value betting opportunities.
    Uses smart weighting based on model confidence and historical performance.
    """
    
    def __init__(self, model_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the prediction integrator.
        
        Args:
            model_weights: Optional dictionary mapping model names to weights
        """
        # Default model weights if not provided
        self.model_weights = model_weights or {
            "xgboost_main": 0.4,
            "lightgbm_main": 0.3,
            "ensemble": 0.3
        }
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {
                model: weight / total_weight 
                for model, weight in self.model_weights.items()
            }
        else:
            # Equal weights if total is 0
            count = len(self.model_weights)
            self.model_weights = {
                model: 1.0 / count 
                for model in self.model_weights
            }
            
        logger.info(f"PredictionIntegrator initialized with weights: {self.model_weights}")
        
    def integrate_predictions(self, 
                             predictions: Dict[str, Dict[str, float]],
                             confidence_scores: Optional[Dict[str, float]] = None
                             ) -> Dict[str, float]:
        """
        Integrate predictions from multiple models using weighted ensemble.
        
        Args:
            predictions: Dictionary mapping model names to prediction dictionaries
            confidence_scores: Optional dictionary mapping model names to confidence scores
            
        Returns:
            Integrated prediction dictionary
        """
        if not predictions:
            return {"home_win": 0.33, "draw": 0.34, "away_win": 0.33}
            
        # Initialize result
        result = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
        total_weight = 0.0
        
        # Integrate predictions with weights
        for model_name, prediction in predictions.items():
            # Skip if prediction is missing required values
            if not all(key in prediction for key in result.keys()):
                logger.warning(f"Skipping incomplete prediction from {model_name}")
                continue
                
            # Get model weight
            weight = self.model_weights.get(model_name, 0.1)  # Default weight for unknown models
            
            # Apply confidence score if provided
            if confidence_scores and model_name in confidence_scores:
                weight *= confidence_scores[model_name]
                
            # Add weighted prediction
            for key in result:
                result[key] += prediction[key] * weight
                
            total_weight += weight
            
        # Normalize to ensure probabilities sum to 1
        if total_weight > 0:
            for key in result:
                result[key] /= total_weight
        else:
            # Default to equal probabilities if total weight is 0
            for key in result:
                result[key] = 1.0 / len(result)
                
        return result
        
    def calculate_value_bets(self, 
                            prediction: Dict[str, float], 
                            odds: Dict[str, float],
                            kelly_fraction: float = 0.5,
                            min_edge: float = 0.05  # 5% minimum edge
                            ) -> Dict[str, Any]:
        """
        Calculate value betting opportunities based on prediction and odds.
        
        Args:
            prediction: Dictionary with predicted probabilities
            odds: Dictionary with bookmaker odds
            kelly_fraction: Fraction of Kelly criterion to use (0-1)
            min_edge: Minimum edge required to consider a bet valuable
            
        Returns:
            Dictionary with value betting information
        """
        if not prediction or not odds:
            return {"value_bets": [], "best_value": None}
            
        # Calculate implied probabilities from odds
        implied_probs = {}
        for outcome, odd in odds.items():
            if odd > 1.0:  # Valid odds
                implied_probs[outcome] = 1.0 / odd
            else:
                implied_probs[outcome] = 0.0
                
        # Calculate edges
        edges = {}
        for outcome in prediction:
            if outcome in odds and odds[outcome] > 1.0:
                # Edge = (Predicted probability * Odds) - 1
                edges[outcome] = (prediction[outcome] * odds[outcome]) - 1.0
            else:
                edges[outcome] = 0.0
                
        # Calculate Kelly stakes
        kelly_stakes = {}
        for outcome, edge in edges.items():
            if edge > 0:
                # Kelly formula: (edge * p) / odds  where p is predicted probability
                kelly = (prediction[outcome] * (odds[outcome] - 1) - (1 - prediction[outcome])) / (odds[outcome] - 1)
                # Apply Kelly fraction to be more conservative
                kelly_stakes[outcome] = max(0, kelly * kelly_fraction)
            else:
                kelly_stakes[outcome] = 0.0
                
        # Identify value bets
        value_bets = []
        for outcome in prediction:
            if outcome in edges and edges[outcome] >= min_edge:
                value_bets.append({
                    "outcome": outcome,
                    "prediction": prediction[outcome],
                    "implied_probability": implied_probs.get(outcome, 0),
                    "odds": odds.get(outcome, 0),
                    "edge": edges[outcome],
                    "kelly_stake": kelly_stakes.get(outcome, 0),
                    "confidence": self._calculate_confidence(
                        prediction[outcome], 
                        implied_probs.get(outcome, 0)
                    )
                })
                
        # Sort by edge
        value_bets.sort(key=lambda x: x["edge"], reverse=True)
        
        # Identify best value bet
        best_value = value_bets[0] if value_bets else None
        
        return {
            "value_bets": value_bets,
            "best_value": best_value,
            "total_edge": sum(edges.values()),
            "implied_overround": sum(implied_probs.values()),
            "model_overround": sum(prediction.values())
        }
        
    def _calculate_confidence(self, 
                             predicted_prob: float, 
                             implied_prob: float
                             ) -> str:
        """
        Calculate confidence level for a value bet.
        
        Args:
            predicted_prob: Predicted probability
            implied_prob: Implied probability from odds
            
        Returns:
            Confidence level string ("Low", "Medium", "High")
        """
        # Probability difference
        prob_diff = predicted_prob - implied_prob
        
        # Confidence based on probability difference
        if prob_diff >= 0.15:
            return "High"
        elif prob_diff >= 0.08:
            return "Medium"
        else:
            return "Low"
            
    def analyze_odds_movement(self, 
                             current_odds: Dict[str, float], 
                             historical_odds: List[Dict[str, float]]
                             ) -> Dict[str, Any]:
        """
        Analyze odds movement to detect market sentiment.
        
        Args:
            current_odds: Current bookmaker odds
            historical_odds: List of historical odds in chronological order
            
        Returns:
            Dictionary with odds movement analysis
        """
        if not current_odds or not historical_odds:
            return {"movement": "unknown", "steam": False, "details": {}}
            
        result = {"movement": {}, "steam": False, "details": {}}
        
        # Calculate movement for each outcome
        for outcome in current_odds:
            # Extract historical values for this outcome
            history = [odds.get(outcome, 0) for odds in historical_odds if odds.get(outcome, 0) > 0]
            
            if history and outcome in current_odds:
                # Calculate movement stats
                start_odds = history[0]
                current = current_odds[outcome]
                absolute_change = current - start_odds
                percentage_change = (current - start_odds) / start_odds * 100 if start_odds > 0 else 0
                
                # Determine direction and strength
                if absolute_change > 0.2:
                    direction = "strong_up"
                elif absolute_change > 0.05:
                    direction = "up"
                elif absolute_change < -0.2:
                    direction = "strong_down"
                elif absolute_change < -0.05:
                    direction = "down"
                else:
                    direction = "stable"
                    
                # Check for steam (rapid movement)
                is_steam = False
                if len(history) >= 3:
                    # Check for significant movement in last 3 data points
                    recent = history[-3:]
                    recent_change = recent[-1] - recent[0]
                    recent_pct = abs(recent_change) / recent[0] * 100 if recent[0] > 0 else 0
                    
                    # Consider it steam if recent movement is significant and rapid
                    is_steam = recent_pct >= 5.0
                    
                # Record results
                result["movement"][outcome] = direction
                result["details"][outcome] = {
                    "start": start_odds,
                    "current": current,
                    "absolute_change": absolute_change,
                    "percentage_change": percentage_change,
                    "is_steam": is_steam
                }
                
                # Update overall steam detection
                if is_steam:
                    result["steam"] = True
            else:
                result["movement"][outcome] = "unknown"
                
        return result
        
    def evaluate_prediction_consistency(self, 
                                      predictions: Dict[str, Dict[str, float]]
                                      ) -> Dict[str, Any]:
        """
        Evaluate consistency across multiple model predictions.
        
        Args:
            predictions: Dictionary mapping model names to prediction dictionaries
            
        Returns:
            Dictionary with consistency metrics
        """
        if not predictions or len(predictions) < 2:
            return {"consistency": "unknown", "agreement": 0.0, "details": {}}
            
        # Extract predictions for each outcome
        outcome_predictions = {}
        for outcome in ["home_win", "draw", "away_win"]:
            outcome_predictions[outcome] = [
                pred.get(outcome, 0)
                for model, pred in predictions.items()
                if outcome in pred
            ]
            
        # Calculate consistency metrics
        consistency = {}
        agreement_scores = []
        
        for outcome, preds in outcome_predictions.items():
            if len(preds) >= 2:
                mean = statistics.mean(preds)
                stdev = statistics.stdev(preds) if len(preds) > 1 else 0
                cv = stdev / mean if mean > 0 else float('inf')  # Coefficient of variation
                
                # Determine consistency level
                if cv < 0.15:
                    level = "high"
                elif cv < 0.3:
                    level = "medium"
                else:
                    level = "low"
                    
                consistency[outcome] = {
                    "level": level,
                    "mean": mean,
                    "stdev": stdev,
                    "cv": cv
                }
                
                # Calculate agreement score (0-1)
                agreement = max(0, 1 - min(1, cv))
                agreement_scores.append(agreement)
            else:
                consistency[outcome] = {"level": "unknown"}
                
        # Overall consistency evaluation
        overall_agreement = statistics.mean(agreement_scores) if agreement_scores else 0
        
        if overall_agreement > 0.8:
            overall_consistency = "high"
        elif overall_agreement > 0.6:
            overall_consistency = "medium"
        else:
            overall_consistency = "low"
            
        return {
            "consistency": overall_consistency,
            "agreement": overall_agreement,
            "details": consistency
        }
        
# Convenience function for value betting analysis
def analyze_value_bet(prediction: Dict[str, float], odds: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze a bet for value based on prediction and odds.
    
    Args:
        prediction: Dictionary with predicted probabilities
        odds: Dictionary with bookmaker odds
        
    Returns:
        Dictionary with value betting analysis
    """
    integrator = PredictionIntegrator()
    return integrator.calculate_value_bets(prediction, odds)
