#!/usr/bin/env python3
"""
Enhanced Personalization Integration Module

Provides utility functions for tracking user behavior and applying
adaptive interface configurations based on user preferences.

Phase 5B Integration: Advanced personalization with behavior tracking.
"""

import logging
from typing import Any, Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)


def track_prediction_view(home_team: str, away_team: str, league: str = None, 
                         confidence: float = None, metadata: Dict[str, Any] = None):
    """
    Track when user views a prediction.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        league: League name (optional)
        confidence: Prediction confidence (optional)
        metadata: Additional metadata (optional)
    """
    try:
        from dashboard.components.personalization_sidebar import track_user_interaction
        
        track_metadata = {
            'home_team': home_team,
            'away_team': away_team,
            'match': f"{home_team} vs {away_team}"
        }
        
        if league:
            track_metadata['league'] = league
        if confidence is not None:
            track_metadata['confidence'] = confidence
        if metadata:
            track_metadata.update(metadata)
        
        track_user_interaction(
            action_type='prediction_view',
            target=f"{home_team}_vs_{away_team}",
            metadata=track_metadata
        )
        
    except Exception as e:
        logger.debug(f"Prediction view tracking failed: {e}")


def track_prediction_interaction(home_team: str, away_team: str, interaction_type: str,
                                metadata: Dict[str, Any] = None):
    """
    Track user interactions with predictions (expand, analyze, bet consideration).
    
    Args:
        home_team: Home team name
        away_team: Away team name
        interaction_type: Type of interaction (expand, analyze, bet_consider, etc.)
        metadata: Additional metadata (optional)
    """
    try:
        from dashboard.components.personalization_sidebar import track_user_interaction
        
        track_metadata = {
            'home_team': home_team,
            'away_team': away_team,
            'match': f"{home_team} vs {away_team}",
            'interaction_type': interaction_type
        }
        
        if metadata:
            track_metadata.update(metadata)
        
        track_user_interaction(
            action_type='prediction_interaction',
            target=f"{home_team}_vs_{away_team}",
            metadata=track_metadata
        )
        
    except Exception as e:
        logger.debug(f"Prediction interaction tracking failed: {e}")


def track_team_interest(team_name: str, context: str = 'view', metadata: Dict[str, Any] = None):
    """
    Track user interest in specific teams.
    
    Args:
        team_name: Team name
        context: Context of interest (view, select, analyze, etc.)
        metadata: Additional metadata (optional)
    """
    try:
        from dashboard.components.personalization_sidebar import track_user_interaction
        
        track_metadata = {
            'team': team_name,
            'context': context
        }
        
        if metadata:
            track_metadata.update(metadata)
        
        track_user_interaction(
            action_type='team_interest',
            target=team_name,
            metadata=track_metadata
        )
        
    except Exception as e:
        logger.debug(f"Team interest tracking failed: {e}")


def track_league_interest(league_name: str, context: str = 'view', metadata: Dict[str, Any] = None):
    """
    Track user interest in specific leagues.
    
    Args:
        league_name: League name
        context: Context of interest (view, select, analyze, etc.)
        metadata: Additional metadata (optional)
    """
    try:
        from dashboard.components.personalization_sidebar import track_user_interaction
        
        track_metadata = {
            'league': league_name,
            'context': context
        }
        
        if metadata:
            track_metadata.update(metadata)
        
        track_user_interaction(
            action_type='league_interest',
            target=league_name,
            metadata=track_metadata
        )
        
    except Exception as e:
        logger.debug(f"League interest tracking failed: {e}")


def get_adaptive_match_filter() -> Dict[str, Any]:
    """
    Get adaptive match filtering configuration based on user preferences.
    
    Returns:
        Dictionary with filtering preferences:
        - preferred_teams: List of favorite teams
        - preferred_leagues: List of favorite leagues
        - risk_profile: User's risk tolerance
        - show_high_confidence_only: Whether to filter by confidence
    """
    try:
        from dashboard.components.personalization_sidebar import (
            get_adaptive_interface_config,
        )
        
        config = get_adaptive_interface_config()
        
        return {
            'preferred_teams': config.get('default_teams', []),
            'preferred_leagues': config.get('default_leagues', []),
            'risk_profile': config.get('risk_indicators', {}).get('show_risk_warnings', True),
            'show_high_confidence_only': config.get('risk_indicators', {}).get('highlight_safe_bets', False),
            'personalization_level': config.get('personalization_level', 0.0)
        }
        
    except Exception as e:
        logger.debug(f"Failed to get adaptive filter: {e}")
        return {
            'preferred_teams': [],
            'preferred_leagues': [],
            'risk_profile': True,
            'show_high_confidence_only': False,
            'personalization_level': 0.0
        }


def apply_adaptive_ui_styling() -> Dict[str, str]:
    """
    Get adaptive UI styling based on user preferences.
    
    Returns:
        Dictionary with CSS class overrides and styling preferences
    """
    try:
        from dashboard.components.personalization_sidebar import (
            get_adaptive_interface_config,
        )
        
        config = get_adaptive_interface_config()
        layout = config.get('layout', 'standard')
        
        if layout == 'simple':
            return {
                'card_style': 'minimal',
                'show_advanced_metrics': False,
                'highlight_essentials': True
            }
        elif layout == 'advanced':
            return {
                'card_style': 'detailed',
                'show_advanced_metrics': True,
                'highlight_essentials': False
            }
        else:
            return {
                'card_style': 'standard',
                'show_advanced_metrics': True,
                'highlight_essentials': False
            }
            
    except Exception as e:
        logger.debug(f"Failed to apply adaptive styling: {e}")
        return {
            'card_style': 'standard',
            'show_advanced_metrics': True,
            'highlight_essentials': False
        }


def get_personalized_match_recommendations(available_matches: list) -> list:
    """
    Filter and rank matches based on user preferences.
    
    Args:
        available_matches: List of all available matches
        
    Returns:
        Filtered and ranked list of matches with personalization scores
    """
    try:
        filter_config = get_adaptive_match_filter()
        
        if filter_config['personalization_level'] < 0.3:
            # Low personalization - return all matches
            return available_matches
        
        preferred_teams = set(filter_config['preferred_teams'])
        preferred_leagues = set(filter_config['preferred_leagues'])
        
        # Score each match
        scored_matches = []
        for match in available_matches:
            score = 0.0
            
            # Team preference scoring
            home_team = match.get('home_team', '')
            away_team = match.get('away_team', '')
            
            if home_team in preferred_teams or away_team in preferred_teams:
                score += 10.0  # Strong preference
            
            # League preference scoring
            league = match.get('league', '')
            if league in preferred_leagues:
                score += 5.0  # Moderate preference
            
            # Confidence filtering
            if filter_config['show_high_confidence_only']:
                confidence = match.get('confidence', 0.0)
                if confidence >= 0.75:
                    score += 3.0
                elif confidence < 0.60:
                    score -= 2.0  # Deprioritize low confidence
            
            match['personalization_score'] = score
            scored_matches.append(match)
        
        # Sort by personalization score (descending)
        scored_matches.sort(key=lambda x: x.get('personalization_score', 0.0), reverse=True)
        
        return scored_matches
        
    except Exception as e:
        logger.debug(f"Failed to personalize match recommendations: {e}")
        return available_matches


def render_personalization_indicator(personalization_level: float):
    """
    Render a visual indicator of personalization level.
    
    Args:
        personalization_level: Float between 0.0 and 1.0
    """
    try:
        if personalization_level >= 0.7:
            st.success(f"🎯 Highly Personalized ({personalization_level*100:.0f}%)")
        elif personalization_level >= 0.4:
            st.info(f"📊 Personalized ({personalization_level*100:.0f}%)")
        elif personalization_level > 0:
            st.warning(f"🌱 Learning Your Preferences ({personalization_level*100:.0f}%)")
        else:
            st.caption("💡 Interact with predictions to enable personalization")
            
    except Exception:
        pass


def get_risk_based_bet_suggestions(predictions: list, risk_tolerance: float = 0.5) -> list:
    """
    Filter predictions based on user's risk tolerance.
    
    Args:
        predictions: List of predictions
        risk_tolerance: User's risk tolerance (0.0 = very conservative, 1.0 = very aggressive)
        
    Returns:
        Filtered predictions matching risk profile with suggested stake sizes
    """
    try:
        suggested_bets = []
        
        for pred in predictions:
            confidence = pred.get('confidence', 0.0)
            edge = pred.get('edge', 0.0)  # Expected value edge
            
            # Calculate risk score (lower = safer)
            risk_score = 1.0 - confidence
            
            # Match risk profile
            if risk_tolerance < 0.3:
                # Conservative: Only high confidence, positive edge
                if confidence >= 0.80 and edge > 0.03:
                    suggested_stake = 0.02  # 2% of bankroll
                    pred['suggested_stake'] = suggested_stake
                    pred['risk_level'] = 'Low'
                    suggested_bets.append(pred)
                    
            elif risk_tolerance < 0.7:
                # Balanced: Medium to high confidence
                if confidence >= 0.65 and edge > 0.02:
                    suggested_stake = 0.03 + (confidence - 0.65) * 0.10  # 3-7% of bankroll
                    pred['suggested_stake'] = suggested_stake
                    pred['risk_level'] = 'Medium'
                    suggested_bets.append(pred)
                    
            else:
                # Aggressive: Accept higher risk for higher rewards
                if confidence >= 0.55 and edge > 0.01:
                    suggested_stake = 0.05 + (edge * 100) * 0.02  # 5-15% of bankroll
                    pred['suggested_stake'] = min(suggested_stake, 0.15)  # Cap at 15%
                    pred['risk_level'] = 'High'
                    suggested_bets.append(pred)
        
        # Sort by expected value (edge * confidence)
        suggested_bets.sort(
            key=lambda x: x.get('edge', 0) * x.get('confidence', 0),
            reverse=True
        )
        
        return suggested_bets[:10]  # Top 10 suggestions
        
    except Exception as e:
        logger.debug(f"Failed to generate risk-based suggestions: {e}")
        return predictions[:5]  # Fallback to top 5 predictions
