"""
Value Bet Analyzer module for detecting value betting opportunities.
Compares model predictions against market odds to identify value bets.
Implements Kelly criterion and edge calculations for optimal bankroll management.
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sqlalchemy.exc import SQLAlchemyError

from database.db_manager import DatabaseManager
from database.schema import Match, Odds, Prediction, ValueBet
from utils.config import Config
from utils.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

class ValueBetAnalyzer:
    """
    Analyzes predictions and market odds to identify value betting opportunities.
    Uses Kelly criterion and other metrics to calculate optimal stake sizes.
    Supports multiple bet types (1X2, BTTS, Over/Under) with configurable parameters.
    """
    
    # Bet types supported by the analyzer
    BET_TYPES = {
        "1X2": ["HOME", "DRAW", "AWAY"],
        "BTTS": ["YES", "NO"],
        "OVER_UNDER_2_5": ["OVER", "UNDER"]
    }
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, min_edge: float = 0.05, 
                 min_confidence: float = 0.6, kelly_fraction: float = 0.5):
        """
        Initialize the value bet analyzer with enhanced configuration.
        
        Args:
            db_manager: Database manager instance
            min_edge: Minimum edge required for a value bet (default: 5%)
            min_confidence: Minimum model confidence required (default: 60%)
            kelly_fraction: Fraction of Kelly criterion to use (default: 0.5)
        """
        self.db_manager = db_manager or DatabaseManager()
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.kelly_fraction = kelly_fraction  # Using fractional Kelly for more conservative stakes
        self.system_monitor = SystemMonitor()
        
        logger.info(f"ValueBetAnalyzer initialized with min_edge={min_edge}, min_confidence={min_confidence}, "
                    f"kelly_fraction={kelly_fraction}")
    
    def analyze_match(self, match_id: str) -> List[Dict[str, Any]]:
        """
        Analyze a match for value betting opportunities across multiple bet types.
        
        Args:
            match_id: Match ID to analyze
            
        Returns:
            List of value bet opportunities
        """
    from datetime import timezone
    operation_id = self.system_monitor.start_operation("analyze_match_value_bets")
    logger.info(f"Analyzing match {match_id} for value betting opportunities")
        
        try:
            # Get latest prediction and odds for this match
            with self.db_manager.session_scope() as session:
                # Get match details
                match = session.query(Match).filter(Match.id == match_id).first()
                if not match:
                    logger.warning(f"Match not found: {match_id}")
                    return []
                
                # Get latest prediction
                prediction = session.query(Prediction)\
                    .filter(Prediction.match_id == match_id)\
                    .order_by(Prediction.created_at.desc())\
                    .first()
                    
                if not prediction:
                    logger.warning(f"No prediction found for match: {match_id}")
                    return []
                
                # Check prediction confidence
                if prediction.confidence < self.min_confidence:
                    logger.info(f"Prediction confidence {prediction.confidence} below threshold "
                                f"{self.min_confidence} for match {match_id}")
                    return []
                
                # Get latest odds
                odds = session.query(Odds)\
                    .filter(Odds.match_id == match_id)\
                    .order_by(Odds.timestamp.desc())\
                    .first()
                    
                if not odds:
                    logger.warning(f"No odds found for match: {match_id}")
                    return []
                
                # Analyze value bets for different bet types
                value_bets = []
                
                # Analyze 1X2 (Match Result) bets
                value_bets.extend(self._analyze_match_result_bets(match_id, prediction, odds))
                
                # Analyze Both Teams To Score (BTTS) bets
                value_bets.extend(self._analyze_btts_bets(match_id, prediction, odds))
                
                # Analyze Over/Under 2.5 goals bets
                value_bets.extend(self._analyze_over_under_bets(match_id, prediction, odds))
                
                # Save all value bets to database
                self._save_value_bets(session, value_bets)
                
                logger.info(f"Found {len(value_bets)} value betting opportunities for match {match_id}")
                return value_bets
                
        except Exception as e:
            logger.error(f"Error analyzing match {match_id} for value bets: {e}")
            return []
        finally:
            self.system_monitor.end_operation(operation_id)
    
    def _analyze_match_result_bets(self, match_id: str, prediction: Prediction, odds: Odds) -> List[Dict[str, Any]]:
        """
        Analyze match result (1X2) bets for value.
        
        Args:
            match_id: Match ID
            prediction: Match prediction
            odds: Match odds
            
        Returns:
            List of value bet opportunities
        """
        value_bets = []
        
        # Calculate implied probabilities from odds
        home_implied_prob = 1 / odds.home_win if odds.home_win else 0
        draw_implied_prob = 1 / odds.draw if odds.draw else 0
        away_implied_prob = 1 / odds.away_win if odds.away_win else 0
        
        # Get model probabilities
        home_model_prob = prediction.home_win_prob
        draw_model_prob = prediction.draw_prob
        away_model_prob = prediction.away_win_prob
        
        # Calculate edges
        home_edge = self._calculate_edge(home_model_prob, home_implied_prob)
        draw_edge = self._calculate_edge(draw_model_prob, draw_implied_prob)
        away_edge = self._calculate_edge(away_model_prob, away_implied_prob)
        
        # Calculate Kelly stakes
        home_kelly = self._calculate_kelly(home_model_prob, odds.home_win)
        draw_kelly = self._calculate_kelly(draw_model_prob, odds.draw)
        away_kelly = self._calculate_kelly(away_model_prob, odds.away_win)
        
        # Check for value bets with sufficient edge
        if home_edge >= self.min_edge and home_kelly > 0:
            value_bets.append({
                "id": f"{match_id}_1X2_HOME_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "match_id": match_id,
                "bet_type": "1X2",
                "selection": "HOME",
                "odds": odds.home_win,
                "fair_odds": 1 / home_model_prob if home_model_prob > 0 else 0,
                "edge": home_edge,
                "kelly_stake": home_kelly * self.kelly_fraction,  # Apply Kelly fraction
                "confidence": prediction.confidence * home_model_prob,
                "created_at": datetime.now(timezone.utc)
            })
        
        if draw_edge >= self.min_edge and draw_kelly > 0:
            value_bets.append({
                "id": f"{match_id}_1X2_DRAW_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "match_id": match_id,
                "bet_type": "1X2",
                "selection": "DRAW",
                "odds": odds.draw,
                "fair_odds": 1 / draw_model_prob if draw_model_prob > 0 else 0,
                "edge": draw_edge,
                "kelly_stake": draw_kelly * self.kelly_fraction,
                "confidence": prediction.confidence * draw_model_prob,
                "created_at": datetime.now(timezone.utc)
            })
        
        if away_edge >= self.min_edge and away_kelly > 0:
            value_bets.append({
                "id": f"{match_id}_1X2_AWAY_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "match_id": match_id,
                "bet_type": "1X2",
                "selection": "AWAY",
                "odds": odds.away_win,
                "fair_odds": 1 / away_model_prob if away_model_prob > 0 else 0,
                "edge": away_edge,
                "kelly_stake": away_kelly * self.kelly_fraction,
                "confidence": prediction.confidence * away_model_prob,
                "created_at": datetime.now(timezone.utc)
            })
        
        return value_bets
    
    def _analyze_btts_bets(self, match_id: str, prediction: Prediction, odds: Odds) -> List[Dict[str, Any]]:
        """
        Analyze Both Teams To Score (BTTS) bets for value.
        
        Args:
            match_id: Match ID
            prediction: Match prediction
            odds: Match odds
            
        Returns:
            List of value bet opportunities
        """
        value_bets = []
        
        # Check if we have the necessary odds and predictions
        if not hasattr(prediction, 'btts_prob') or prediction.btts_prob is None:
            return []
            
        if not hasattr(odds, 'both_teams_to_score_yes') or not hasattr(odds, 'both_teams_to_score_no'):
            return []
            
        # Calculate implied probabilities from odds
        btts_yes_implied_prob = 1 / odds.both_teams_to_score_yes if odds.both_teams_to_score_yes else 0
        btts_no_implied_prob = 1 / odds.both_teams_to_score_no if odds.both_teams_to_score_no else 0
        
        # Get model probabilities
        btts_yes_model_prob = prediction.btts_prob
        btts_no_model_prob = 1 - prediction.btts_prob if prediction.btts_prob is not None else 0
        
        # Calculate edges
        btts_yes_edge = self._calculate_edge(btts_yes_model_prob, btts_yes_implied_prob)
        btts_no_edge = self._calculate_edge(btts_no_model_prob, btts_no_implied_prob)
        
        # Calculate Kelly stakes
        btts_yes_kelly = self._calculate_kelly(btts_yes_model_prob, odds.both_teams_to_score_yes)
        btts_no_kelly = self._calculate_kelly(btts_no_model_prob, odds.both_teams_to_score_no)
        
        # Check for value bets with sufficient edge
        if btts_yes_edge >= self.min_edge and btts_yes_kelly > 0:
            value_bets.append({
                "id": f"{match_id}_BTTS_YES_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "match_id": match_id,
                "bet_type": "BTTS",
                "selection": "YES",
                "odds": odds.both_teams_to_score_yes,
                "fair_odds": 1 / btts_yes_model_prob if btts_yes_model_prob > 0 else 0,
                "edge": btts_yes_edge,
                "kelly_stake": btts_yes_kelly * self.kelly_fraction,
                "confidence": prediction.confidence * 0.9,  # Slightly lower confidence for BTTS
                "created_at": datetime.now(timezone.utc)
            })
        
        if btts_no_edge >= self.min_edge and btts_no_kelly > 0:
            value_bets.append({
                "id": f"{match_id}_BTTS_NO_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "match_id": match_id,
                "bet_type": "BTTS",
                "selection": "NO",
                "odds": odds.both_teams_to_score_no,
                "fair_odds": 1 / btts_no_model_prob if btts_no_model_prob > 0 else 0,
                "edge": btts_no_edge,
                "kelly_stake": btts_no_kelly * self.kelly_fraction,
                "confidence": prediction.confidence * 0.9,
                "created_at": datetime.now(timezone.utc)
            })
        
        return value_bets
    
    def _analyze_over_under_bets(self, match_id: str, prediction: Prediction, odds: Odds) -> List[Dict[str, Any]]:
        """
        Analyze Over/Under 2.5 goals bets for value.
        
        Args:
            match_id: Match ID
            prediction: Match prediction
            odds: Match odds
            
        Returns:
            List of value bet opportunities
        """
        value_bets = []
        
        # Check if we have the necessary odds and predictions
        if not hasattr(prediction, 'over_under_2_5_over_prob') or prediction.over_under_2_5_over_prob is None:
            return []
            
        if not hasattr(odds, 'over_under_2_5_over') or not hasattr(odds, 'over_under_2_5_under'):
            return []
        
        # Calculate implied probabilities from odds
        over_implied_prob = 1 / odds.over_under_2_5_over if odds.over_under_2_5_over else 0
        under_implied_prob = 1 / odds.over_under_2_5_under if odds.over_under_2_5_under else 0
        
        # Get model probabilities
        over_model_prob = prediction.over_under_2_5_over_prob
        under_model_prob = prediction.over_under_2_5_under_prob
        
        # Calculate edges
        over_edge = self._calculate_edge(over_model_prob, over_implied_prob)
        under_edge = self._calculate_edge(under_model_prob, under_implied_prob)
        
        # Calculate Kelly stakes
        over_kelly = self._calculate_kelly(over_model_prob, odds.over_under_2_5_over)
        under_kelly = self._calculate_kelly(under_model_prob, odds.over_under_2_5_under)
        
        # Check for value bets with sufficient edge
        if over_edge >= self.min_edge and over_kelly > 0:
            value_bets.append({
                "id": f"{match_id}_OVER_UNDER_OVER_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "match_id": match_id,
                "bet_type": "OVER_UNDER_2.5",
                "selection": "OVER",
                "odds": odds.over_under_2_5_over,
                "fair_odds": 1 / over_model_prob if over_model_prob > 0 else 0,
                "edge": over_edge,
                "kelly_stake": over_kelly * self.kelly_fraction,
                "confidence": prediction.confidence * 0.95,
                "created_at": datetime.now(timezone.utc)
            })
        
        if under_edge >= self.min_edge and under_kelly > 0:
            value_bets.append({
                "id": f"{match_id}_OVER_UNDER_UNDER_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "match_id": match_id,
                "bet_type": "OVER_UNDER_2.5",
                "selection": "UNDER",
                "odds": odds.over_under_2_5_under,
                "fair_odds": 1 / under_model_prob if under_model_prob > 0 else 0,
                "edge": under_edge,
                "kelly_stake": under_kelly * self.kelly_fraction,
                "confidence": prediction.confidence * 0.95,
                "created_at": datetime.now(timezone.utc)
            })
        
        return value_bets
    
    def _calculate_edge(self, model_prob: float, implied_prob: float) -> float:
        """
        Calculate the edge (value) of a bet.
        
        Args:
            model_prob: Probability from our model
            implied_prob: Implied probability from the odds
            
        Returns:
            Edge percentage (e.g., 0.05 for 5% edge)
        """
        if model_prob <= 0 or implied_prob <= 0:
            return 0
            
        # Edge formula: (model_prob * odds - 1) = model_prob / implied_prob - 1
        return model_prob / implied_prob - 1
    
    def _calculate_kelly(self, prob: float, odds: float) -> float:
        """
        Calculate the Kelly criterion stake.
        
        Args:
            prob: Probability of the outcome
            odds: Decimal odds offered
            
        Returns:
            Kelly stake as a percentage of bankroll
        """
        if prob <= 0 or odds <= 1:
            return 0
            
        # Kelly formula: (bp - q) / b
        # where b = odds - 1, p = probability of winning, q = probability of losing
        b = odds - 1  # Decimal odds minus 1 gives the profit on a winning bet
        q = 1 - prob  # Probability of losing
        kelly = (b * prob - q) / b
        
        # Ensure Kelly is non-negative
        return max(0, kelly)
    
    def _save_value_bets(self, session, value_bets: List[Dict[str, Any]]) -> None:
        """
        Save value bets to the database.
        
        Args:
            session: Database session
            value_bets: List of value bet dictionaries
        """
        try:
            for bet_dict in value_bets:
                # Check if this value bet already exists
                existing_bet = session.query(ValueBet).filter(
                    ValueBet.match_id == bet_dict["match_id"],
                    ValueBet.bet_type == bet_dict["bet_type"],
                    ValueBet.selection == bet_dict["selection"],
                ).order_by(ValueBet.created_at.desc()).first()
                
                # Only add if it doesn't exist or if odds have changed significantly
                if not existing_bet or abs(existing_bet.odds - bet_dict["odds"]) > 0.1:
                    # Create new value bet
                    new_bet = ValueBet(**bet_dict)
                    session.add(new_bet)
            
            # Commit the transaction
            session.commit()
        except SQLAlchemyError as e:
            logger.error(f"Database error saving value bets: {e}")
            session.rollback()
        except Exception as e:
            logger.error(f"Error saving value bets: {e}")
            session.rollback()
    
    def _analyze_prediction_vs_odds(self, prediction: Prediction, odds: Odds) -> List[Dict[str, Any]]:
        """
        Analyze a prediction against odds for value betting opportunities.
        
        Args:
            prediction: Prediction object
            odds: Odds object
            
        Returns:
            List of value bet opportunities
        """
        value_bets = []
        
        # Check if model confidence is high enough
        if prediction.confidence and prediction.confidence < self.min_confidence:
            logger.debug(f"Model confidence {prediction.confidence} below threshold {self.min_confidence}")
            return []
        
        # Analyze home win
        if odds.home_win > 0:
            implied_prob = 1 / odds.home_win  # Using formula directly to avoid confusion
            edge = (prediction.home_win_prob * odds.home_win) - 1
            
            if edge >= self.min_edge:
                kelly_stake = self._calculate_kelly(prediction.home_win_prob, odds.home_win)
                confidence = prediction.confidence * (0.5 + min(edge / 0.3, 0.5))  # Scale confidence by edge
                
                value_bets.append({
                    "bet_type": "home_win",
                    "predicted_prob": prediction.home_win_prob,
                    "implied_prob": implied_prob,
                    "odds": odds.home_win,
                    "edge": edge,
                    "kelly_stake": kelly_stake,
                    "confidence": confidence,
                    "bookmaker": odds.bookmaker
                })
        
        # Analyze draw
        if odds.draw > 0:
            implied_prob = 1 / odds.draw  # Using formula directly
            edge = (prediction.draw_prob * odds.draw) - 1
            
            if edge >= self.min_edge:
                kelly_stake = self._calculate_kelly(prediction.draw_prob, odds.draw)
                confidence = prediction.confidence * (0.5 + min(edge / 0.3, 0.5))  # Scale confidence by edge
                
                value_bets.append({
                    "bet_type": "draw",
                    "predicted_prob": prediction.draw_prob,
                    "implied_prob": implied_prob,
                    "odds": odds.draw,
                    "edge": edge,
                    "kelly_stake": kelly_stake,
                    "confidence": confidence,
                    "bookmaker": odds.bookmaker
                })
        
        # Analyze away win
        if odds.away_win > 0:
            implied_prob = 1 / odds.away_win  # Using formula directly
            edge = (prediction.away_win_prob * odds.away_win) - 1
            
            if edge >= self.min_edge:
                kelly_stake = self._calculate_kelly(prediction.away_win_prob, odds.away_win)
                confidence = prediction.confidence * (0.5 + min(edge / 0.3, 0.5))  # Scale confidence by edge
                
                value_bets.append({
                    "bet_type": "away_win",
                    "predicted_prob": prediction.away_win_prob,
                    "implied_prob": implied_prob,
                    "odds": odds.away_win,
                    "edge": edge,
                    "kelly_stake": kelly_stake,
                    "confidence": confidence,
                    "bookmaker": odds.bookmaker
                })
        
        # Analyze over/under and BTTS markets if available
        # This would require additional model predictions for these markets
        
        return value_bets
    
    # These methods are now unified with the ones above and removed to avoid duplication
    # The _calculate_kelly method is now used throughout the code
    # The previous usages of _decimal_odds_to_probability and _calculate_kelly_stake have been refactored
        
    def analyze_all_upcoming_matches(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze all upcoming matches for value betting opportunities.
        
        Returns:
            Dictionary of match IDs to value bet lists
        """
        logger.info("Analyzing all upcoming matches for value betting opportunities")
        
        # Get all upcoming matches with predictions and odds
        with db_manager.session_scope() as session:
            # Get matches with status 'scheduled'
            matches = session.query(Match).filter(
                Match.status == 'scheduled'
            ).all()
            
            if not matches:
                logger.warning("No upcoming matches found")
                return {}
            
            results = {}
            
            for match in matches:
                # Check if match has both predictions and odds
                predictions = session.query(Prediction).filter(
                    Prediction.match_id == match.id
                ).count()
                
                odds = session.query(Odds).filter(
                    Odds.match_id == match.id
                ).count()
                
                if predictions > 0 and odds > 0:
                    # Analyze match for value bets
                    value_bets = self.analyze_match(match.id)
                    
                    if value_bets:
                        results[match.id] = value_bets
            
            logger.info(f"Found value betting opportunities for {len(results)} matches")
            return results

# End of ValueBetAnalyzer class
