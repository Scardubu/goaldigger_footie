#!/usr/bin/env python3
"""
Cross-League Match Handler for GoalDiggers Platform
Handles normalization, validation, and adjustments for cross-league predictions.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class CrossLeagueHandler:
    """Handles cross-league match predictions with proper normalization and confidence adjustments."""
    
    # UEFA coefficient-inspired league strength ratings (higher = stronger)
    LEAGUE_STRENGTH_COEFFICIENTS = {
        'Premier League': 1.0,      # England - Reference league
        'La Liga': 0.95,            # Spain
        'Bundesliga': 0.90,         # Germany  
        'Serie A': 0.85,            # Italy
        'Ligue 1': 0.80,            # France
        'Champions League': 1.1,    # European competition
        'Europa League': 0.95,      # European competition
    }
    
    # League tier classifications
    LEAGUE_TIERS = {
        'Premier League': 1,
        'La Liga': 1, 
        'Bundesliga': 1,
        'Serie A': 1,
        'Ligue 1': 1,
        'Champions League': 0,  # International
        'Europa League': 0,     # International
    }
    
    def __init__(self):
        """Initialize the cross-league handler."""
        self.league_stats_cache = {}
        self.cross_league_history = {}
    
    def get_league_strength_coefficient(self, league_name: str) -> float:
        """Get the strength coefficient for a league."""
        return self.LEAGUE_STRENGTH_COEFFICIENTS.get(league_name, 0.75)  # Default for unknown leagues

    def is_cross_league_match(self, home_team: Dict[str, Any], away_team: Dict[str, Any]) -> bool:
        """
        Determine if a match is between teams from different leagues.

        Args:
            home_team: Home team data with league information
            away_team: Away team data with league information

        Returns:
            True if teams are from different leagues, False otherwise
        """
        home_league = home_team.get('league_name', '').strip()
        away_league = away_team.get('league_name', '').strip()

        # Handle empty or missing league names
        if not home_league or not away_league:
            logger.warning(f"Missing league information: home='{home_league}', away='{away_league}'")
            return False

        # Compare league names (case-insensitive)
        is_cross_league = home_league.lower() != away_league.lower()

        logger.debug(f"Cross-league check: {home_league} vs {away_league} = {is_cross_league}")

        return is_cross_league
    
    def calculate_cross_league_elo_adjustment(self, home_team_league: str, away_team_league: str, 
                                            home_elo: float, away_elo: float) -> Tuple[float, float]:
        """
        Adjust ELO ratings for cross-league matches based on league strength.
        
        Args:
            home_team_league: Home team's league name
            away_team_league: Away team's league name  
            home_elo: Home team's raw ELO rating
            away_elo: Away team's raw ELO rating
            
        Returns:
            Tuple of (adjusted_home_elo, adjusted_away_elo)
        """
        home_strength = self.get_league_strength_coefficient(home_team_league)
        away_strength = self.get_league_strength_coefficient(away_team_league)
        
        # Apply league strength adjustments
        adjusted_home_elo = home_elo * home_strength
        adjusted_away_elo = away_elo * away_strength
        
        logger.debug(f"ELO adjustment: {home_team_league}({home_strength}) vs {away_team_league}({away_strength})")
        logger.debug(f"ELO: {home_elo:.1f} -> {adjusted_home_elo:.1f}, {away_elo:.1f} -> {adjusted_away_elo:.1f}")
        
        return adjusted_home_elo, adjusted_away_elo
    
    def normalize_form_metrics_cross_league(self, home_form: Dict[str, float], away_form: Dict[str, float],
                                          home_league: str, away_league: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Normalize form metrics for cross-league comparison.
        
        Args:
            home_form: Home team form metrics
            away_form: Away team form metrics
            home_league: Home team's league
            away_league: Away team's league
            
        Returns:
            Tuple of (normalized_home_form, normalized_away_form)
        """
        home_strength = self.get_league_strength_coefficient(home_league)
        away_strength = self.get_league_strength_coefficient(away_league)
        
        # Normalize form metrics by league strength
        normalized_home_form = {}
        normalized_away_form = {}
        
        for key, value in home_form.items():
            if 'goals_scored' in key or 'points' in key:
                # Boost metrics for stronger leagues
                normalized_home_form[key] = value * home_strength
            elif 'goals_conceded' in key:
                # Defensive metrics - inverse relationship
                normalized_home_form[key] = value / home_strength
            else:
                normalized_home_form[key] = value
        
        for key, value in away_form.items():
            if 'goals_scored' in key or 'points' in key:
                normalized_away_form[key] = value * away_strength
            elif 'goals_conceded' in key:
                normalized_away_form[key] = value / away_strength
            else:
                normalized_away_form[key] = value
        
        return normalized_home_form, normalized_away_form
    
    def calculate_cross_league_confidence_adjustment(self, home_league: str, away_league: str,
                                                   base_confidence: float) -> float:
        """
        Adjust prediction confidence for cross-league matches.
        
        Args:
            home_league: Home team's league
            away_league: Away team's league
            base_confidence: Original prediction confidence
            
        Returns:
            Adjusted confidence score
        """
        if home_league == away_league:
            # Same league - no adjustment
            return base_confidence
        
        # Calculate league strength difference
        home_strength = self.get_league_strength_coefficient(home_league)
        away_strength = self.get_league_strength_coefficient(away_league)
        strength_diff = abs(home_strength - away_strength)
        
        # Reduce confidence based on league difference and rarity
        confidence_reduction = 0.1 + (strength_diff * 0.2)  # 10-30% reduction
        
        # Additional reduction for very rare matchups
        if self._is_rare_matchup(home_league, away_league):
            confidence_reduction += 0.15  # Additional 15% reduction
        
        adjusted_confidence = base_confidence * (1 - confidence_reduction)
        
        logger.debug(f"Confidence adjustment: {base_confidence:.3f} -> {adjusted_confidence:.3f} "
                    f"(reduction: {confidence_reduction:.3f})")
        
        return max(adjusted_confidence, 0.1)  # Minimum 10% confidence
    
    def _is_rare_matchup(self, home_league: str, away_league: str) -> bool:
        """Check if this is a rare cross-league matchup."""
        # Define rare matchups (could be based on historical data)
        rare_combinations = [
            ('Ligue 1', 'Premier League'),
            ('Serie A', 'Bundesliga'),
            # Add more rare combinations based on actual data
        ]
        
        matchup = (home_league, away_league)
        reverse_matchup = (away_league, home_league)
        
        return matchup in rare_combinations or reverse_matchup in rare_combinations
    
    def generate_cross_league_insights(self, home_team: Dict[str, Any], away_team: Dict[str, Any],
                                     prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate specific insights for cross-league matches.
        
        Args:
            home_team: Home team data including league info
            away_team: Away team data including league info
            prediction_data: Base prediction data
            
        Returns:
            Enhanced prediction data with cross-league insights
        """
        home_league = home_team.get('league_name', 'Unknown')
        away_league = away_team.get('league_name', 'Unknown')
        
        if home_league == away_league:
            return prediction_data  # No cross-league insights needed
        
        # Add cross-league specific insights
        cross_league_insights = {
            'match_type': 'cross_league',
            'league_comparison': {
                'home_league': home_league,
                'away_league': away_league,
                'home_league_strength': self.get_league_strength_coefficient(home_league),
                'away_league_strength': self.get_league_strength_coefficient(away_league)
            },
            'cross_league_factors': [],
            'confidence_notes': []
        }
        
        # Add specific insights based on league strengths
        home_strength = self.get_league_strength_coefficient(home_league)
        away_strength = self.get_league_strength_coefficient(away_league)
        
        if home_strength > away_strength:
            cross_league_insights['cross_league_factors'].append(
                f"{home_team['name']} benefits from playing in the stronger {home_league}"
            )
        elif away_strength > home_strength:
            cross_league_insights['cross_league_factors'].append(
                f"{away_team['name']} comes from the stronger {away_league}"
            )
        else:
            cross_league_insights['cross_league_factors'].append(
                f"Both teams from similarly competitive leagues ({home_league} vs {away_league})"
            )
        
        # Add confidence notes
        if abs(home_strength - away_strength) > 0.1:
            cross_league_insights['confidence_notes'].append(
                "Prediction confidence reduced due to significant league strength difference"
            )
        
        if self._is_rare_matchup(home_league, away_league):
            cross_league_insights['confidence_notes'].append(
                "Limited historical data for this league combination"
            )
        
        # Merge with existing prediction data
        enhanced_prediction = prediction_data.copy()
        enhanced_prediction['cross_league_insights'] = cross_league_insights
        
        return enhanced_prediction
    
    def validate_cross_league_match(self, home_team: Dict[str, Any], away_team: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a cross-league match and provide warnings if needed.
        
        Args:
            home_team: Home team data
            away_team: Away team data
            
        Returns:
            Validation result with warnings and recommendations
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'recommendations': [],
            'match_type': 'same_league'
        }
        
        home_league = home_team.get('league_name', 'Unknown')
        away_league = away_team.get('league_name', 'Unknown')
        
        if home_league != away_league:
            validation_result['match_type'] = 'cross_league'
            
            # Check for significant league strength differences
            home_strength = self.get_league_strength_coefficient(home_league)
            away_strength = self.get_league_strength_coefficient(away_league)
            strength_diff = abs(home_strength - away_strength)
            
            if strength_diff > 0.15:
                validation_result['warnings'].append(
                    f"Significant strength difference between {home_league} and {away_league}"
                )
                validation_result['recommendations'].append(
                    "Consider the league strength difference when interpreting predictions"
                )
            
            # Check for rare matchups
            if self._is_rare_matchup(home_league, away_league):
                validation_result['warnings'].append(
                    f"Rare matchup: {home_league} vs {away_league} teams"
                )
                validation_result['recommendations'].append(
                    "Prediction confidence may be lower due to limited historical data"
                )
        
        return validation_result
