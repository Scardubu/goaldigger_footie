"""
Feature generator for the GoalDiggers prediction models.
This module provides functionality to extract and transform features for ML prediction.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    Generates features for match prediction models from raw match and team data.
    """
    
    def __init__(self, db_manager=None, lookback_matches=10, include_advanced_metrics=True):
        """
        Initialize feature generator.
        
        Args:
            db_manager: DatabaseManager instance for data access
            lookback_matches: Number of previous matches to consider for team form features
            include_advanced_metrics: Whether to include advanced metrics like xG
        """
        self.db_manager = db_manager
        self.lookback_matches = lookback_matches
        self.include_advanced_metrics = include_advanced_metrics
        self.feature_columns = self._get_feature_columns()
        logger.info("FeatureGenerator initialized with lookback of %s matches", lookback_matches)

    def _get_feature_columns(self) -> List[str]:
        """Define the feature columns used in the model."""
        basic_features = [
            # Team strength indicators
            'home_team_rank', 'away_team_rank',
            'home_win_rate', 'away_win_rate',
            
            # Recent form
            'home_recent_goals_scored', 'home_recent_goals_conceded',
            'away_recent_goals_scored', 'away_recent_goals_conceded',
            'home_points_last_5', 'away_points_last_5',
            
            # Historical matchup
            'historical_home_win_rate', 'historical_away_win_rate',
            
            # Home advantage
            'home_advantage',
        ]
        
        advanced_features = [
            # Advanced metrics
            'home_xG_avg', 'away_xG_avg',
            'home_xGA_avg', 'away_xGA_avg',
            'home_ppda', 'away_ppda',
            'home_deep_completions', 'away_deep_completions',
        ] if self.include_advanced_metrics else []
        
        return basic_features + advanced_features
    
    def generate_features_for_match(self, match_info: Dict, context_toggles: Optional[Dict[str, bool]] = None) -> Dict:
        """
        Generate features for a single match.
        
        Args:
            match_info: Dictionary containing match details (home_team, away_team, match_date, etc.)
            context_toggles: Optional feature toggles to include/exclude certain context
            
        Returns:
            Dictionary of features ready for model input
        """
        try:
            # Default toggles
            toggles = {
                'use_recent_form': True,
                'use_historical_h2h': True,
                'use_league_position': True,
                'use_advanced_metrics': self.include_advanced_metrics,
            }
            
            # Override with provided toggles
            if context_toggles:
                toggles.update(context_toggles)
            
            # Extract required information
            home_team = match_info['home_team']
            away_team = match_info['away_team']
            match_date = match_info.get('match_date', datetime.now())
            league = match_info.get('league')
            
            # Generate feature dictionary
            features = {col: 0.0 for col in self.feature_columns}
            
            if self.db_manager:
                with self.db_manager.session_scope() as session:
                    # Get team data
                    home_team_obj = self.db_manager.get_team_by_name(home_team, session)
                    away_team_obj = self.db_manager.get_team_by_name(away_team, session)
                    
                    if not home_team_obj or not away_team_obj:
                        logger.warning("Team not found: %s or %s", home_team, away_team)
                        return self._get_fallback_features(home_team, away_team)
                    
                    # Team rankings (normalized between 0-1, lower is better)
                    if toggles['use_league_position'] and league:
                        league_obj = self.db_manager.get_league_by_name(league, session)
                        if league_obj:
                            home_rank = self.db_manager.get_team_rank(home_team_obj.id, league_obj.id, match_date, session)
                            away_rank = self.db_manager.get_team_rank(away_team_obj.id, league_obj.id, match_date, session)
                            
                            total_teams = self.db_manager.count_teams_in_league(league_obj.id, session)
                            if total_teams > 0:
                                features['home_team_rank'] = home_rank / total_teams
                                features['away_team_rank'] = away_rank / total_teams
                    
                    # Recent form features
                    if toggles['use_recent_form']:
                        lookback_date = match_date - timedelta(days=180)  # 6 months lookback
                        
                        # Home team recent matches
                        home_matches = self.db_manager.get_team_matches(
                            home_team_obj.id, 
                            start_date=lookback_date, 
                            end_date=match_date,
                            limit=self.lookback_matches,
                            session=session
                        )
                        
                        if home_matches:
                            home_stats = self._calculate_team_stats(home_matches, home_team_obj.id)
                            features['home_win_rate'] = home_stats['win_rate']
                            features['home_recent_goals_scored'] = home_stats['avg_goals_scored']
                            features['home_recent_goals_conceded'] = home_stats['avg_goals_conceded']
                            features['home_points_last_5'] = home_stats['points_last_5'] / 15.0  # Normalize by max points
                            
                            if toggles['use_advanced_metrics'] and 'xG_avg' in home_stats:
                                features['home_xG_avg'] = home_stats['xG_avg']
                                features['home_xGA_avg'] = home_stats['xGA_avg']
                                features['home_ppda'] = home_stats.get('ppda', 0.0)
                                features['home_deep_completions'] = home_stats.get('deep_completions', 0.0)
                        
                        # Away team recent matches
                        away_matches = self.db_manager.get_team_matches(
                            away_team_obj.id, 
                            start_date=lookback_date, 
                            end_date=match_date,
                            limit=self.lookback_matches,
                            session=session
                        )
                        
                        if away_matches:
                            away_stats = self._calculate_team_stats(away_matches, away_team_obj.id)
                            features['away_win_rate'] = away_stats['win_rate']
                            features['away_recent_goals_scored'] = away_stats['avg_goals_scored']
                            features['away_recent_goals_conceded'] = away_stats['avg_goals_conceded']
                            features['away_points_last_5'] = away_stats['points_last_5'] / 15.0  # Normalize by max points
                            
                            if toggles['use_advanced_metrics'] and 'xG_avg' in away_stats:
                                features['away_xG_avg'] = away_stats['xG_avg']
                                features['away_xGA_avg'] = away_stats['xGA_avg']
                                features['away_ppda'] = away_stats.get('ppda', 0.0)
                                features['away_deep_completions'] = away_stats.get('deep_completions', 0.0)
                    
                    # Historical head-to-head
                    if toggles['use_historical_h2h']:
                        h2h_matches = self.db_manager.get_h2h_matches(
                            home_team_obj.id,
                            away_team_obj.id,
                            lookback_years=5,  # Last 5 years
                            session=session
                        )
                        
                        if h2h_matches:
                            home_wins = 0
                            away_wins = 0
                            
                            for match in h2h_matches:
                                if match.status == 'FINISHED':
                                    if match.home_team_id == home_team_obj.id and match.home_score > match.away_score:
                                        home_wins += 1
                                    elif match.away_team_id == away_team_obj.id and match.away_score > match.home_score:
                                        away_wins += 1
                            
                            features['historical_home_win_rate'] = home_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0.5
                            features['historical_away_win_rate'] = away_wins / len(h2h_matches) if len(h2h_matches) > 0 else 0.5
                    
                    # Home advantage - could be league specific
                    league_id = league_obj.id if league and hasattr(league_obj, 'id') else None
                    features['home_advantage'] = self._calculate_home_advantage(league_id, session)
                    
            else:
                # If no DB manager, return fallback features
                return self._get_fallback_features(home_team, away_team)
            
            return features
        
        except Exception as e:
            logger.error(f"Error generating features: {e}")
            return self._get_fallback_features(match_info.get('home_team', 'unknown'), 
                                               match_info.get('away_team', 'unknown'))
    
    def _calculate_team_stats(self, matches, team_id) -> Dict:
        """
        Calculate team statistics from a list of matches.
        
        Args:
            matches: List of match objects
            team_id: ID of the team to calculate stats for
            
        Returns:
            Dictionary of team statistics
        """
        stats = {
            'win_rate': 0.0,
            'avg_goals_scored': 0.0,
            'avg_goals_conceded': 0.0,
            'points_last_5': 0.0,
            'xG_avg': 0.0,
            'xGA_avg': 0.0,
            'ppda': 0.0,
            'deep_completions': 0.0
        }
        
        if not matches:
            return stats
        
        wins = 0
        draws = 0
        total_goals_scored = 0
        total_goals_conceded = 0
        total_xg = 0
        total_xga = 0
        total_ppda = 0
        total_deep = 0
        has_advanced = 0
        points = 0
        
        # Calculate stats from recent to oldest, to get points for last 5
        matches_sorted = sorted(matches, key=lambda m: m.match_date if hasattr(m, 'match_date') else datetime.now(), reverse=True)
        
        for i, match in enumerate(matches_sorted):
            is_home = match.home_team_id == team_id
            
            # Calculate goals
            if is_home:
                team_score = match.home_score if hasattr(match, 'home_score') else 0
                opponent_score = match.away_score if hasattr(match, 'away_score') else 0
            else:
                team_score = match.away_score if hasattr(match, 'away_score') else 0
                opponent_score = match.home_score if hasattr(match, 'home_score') else 0
            
            # Add to totals
            total_goals_scored += team_score
            total_goals_conceded += opponent_score
            
            # Calculate result
            if team_score > opponent_score:
                wins += 1
                if i < 5:  # Last 5 matches
                    points += 3
            elif team_score == opponent_score:
                draws += 1
                if i < 5:  # Last 5 matches
                    points += 1
            
            # Advanced metrics if available
            if hasattr(match, 'match_stats'):
                stats_obj = match.match_stats
                has_advanced += 1
                
                if is_home:
                    total_xg += getattr(stats_obj, 'home_xg', 0) or 0
                    total_xga += getattr(stats_obj, 'away_xg', 0) or 0
                    total_ppda += getattr(stats_obj, 'home_ppda', 0) or 0
                    total_deep += getattr(stats_obj, 'home_deep', 0) or 0
                else:
                    total_xg += getattr(stats_obj, 'away_xg', 0) or 0
                    total_xga += getattr(stats_obj, 'home_xg', 0) or 0
                    total_ppda += getattr(stats_obj, 'away_ppda', 0) or 0
                    total_deep += getattr(stats_obj, 'away_deep', 0) or 0
        
        # Calculate averages
        stats['win_rate'] = wins / len(matches)
        stats['avg_goals_scored'] = total_goals_scored / len(matches)
        stats['avg_goals_conceded'] = total_goals_conceded / len(matches)
        stats['points_last_5'] = points
        
        # Advanced stats if available
        if has_advanced > 0:
            stats['xG_avg'] = total_xg / has_advanced
            stats['xGA_avg'] = total_xga / has_advanced
            stats['ppda'] = total_ppda / has_advanced
            stats['deep_completions'] = total_deep / has_advanced
        
        return stats
    
    def _calculate_home_advantage(self, league_id, session) -> float:
        """
        Calculate the home advantage factor for a league.
        
        Args:
            league_id: ID of the league
            session: Database session
            
        Returns:
            Home advantage factor (0-1)
        """
        # Default home advantage
        default_advantage = 0.55
        
        if not self.db_manager or not session:
            return default_advantage
        
        try:
            # Get finished matches from the last 2 years for this league
            two_years_ago = datetime.now() - timedelta(days=730)
            
            if league_id:
                matches = self.db_manager.get_matches_by_league(
                    league_id,
                    start_date=two_years_ago,
                    status='FINISHED',
                    session=session
                )
            else:
                matches = self.db_manager.get_matches(
                    start_date=two_years_ago,
                    status='FINISHED',
                    limit=1000,  # Use reasonable limit to avoid overloading
                    session=session
                )
            
            if not matches:
                return default_advantage
            
            # Count home wins, draws, away wins
            home_wins = sum(1 for m in matches if m.home_score > m.away_score)
            
            # Calculate home win percentage
            home_advantage = home_wins / len(matches)
            
            # Normalize to reasonable range (0.4-0.7)
            home_advantage = max(0.4, min(0.7, home_advantage))
            
            return home_advantage
            
        except Exception as e:
            logger.error(f"Error calculating home advantage: {e}")
            return default_advantage
    
    def _get_fallback_features(self, home_team, away_team) -> Dict:
        """
        Generate fallback features when database access fails.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            
        Returns:
            Dictionary of fallback features
        """
        logger.warning(f"Using fallback features for match: {home_team} vs {away_team}")
        
        # Create default feature dictionary with balanced values
        features = {col: 0.5 for col in self.feature_columns}
        
        # Home advantage is slightly higher (55%)
        features['home_advantage'] = 0.55
        
        # Use team name strings to generate pseudo-random but consistent values
        import hashlib
        
        def get_team_strength(team_name):
            # Generate consistent pseudo-random value between 0.3 and 0.7
            hash_val = int(hashlib.md5(team_name.encode()).hexdigest(), 16)
            return 0.3 + (hash_val % 1000) / 2500  # Range 0.3-0.7
        
        home_strength = get_team_strength(home_team)
        away_strength = get_team_strength(away_team)
        
        # Adjust features based on pseudo-random team strength
        features['home_team_rank'] = 1.0 - home_strength  # Lower rank is better
        features['away_team_rank'] = 1.0 - away_strength
        features['home_win_rate'] = home_strength
        features['away_win_rate'] = away_strength
        features['home_recent_goals_scored'] = home_strength * 2.0  # 0.6-1.4 goals
        features['away_recent_goals_scored'] = away_strength * 2.0
        features['home_recent_goals_conceded'] = (1.0 - home_strength) * 1.5  # 0.45-1.05 goals
        features['away_recent_goals_conceded'] = (1.0 - away_strength) * 1.5
        
        # Balance historical h2h based on team strengths
        if home_strength > away_strength:
            features['historical_home_win_rate'] = 0.5 + (home_strength - away_strength)
            features['historical_away_win_rate'] = 0.5 - (home_strength - away_strength)
        else:
            features['historical_home_win_rate'] = 0.5 - (away_strength - home_strength)
            features['historical_away_win_rate'] = 0.5 + (away_strength - home_strength)
        
        # Points last 5 (normalized between 0-1, max 15 points)
        features['home_points_last_5'] = home_strength * 0.8  # 0-0.8 range
        features['away_points_last_5'] = away_strength * 0.8
        
        # Advanced metrics if included
        if 'home_xG_avg' in features:
            features['home_xG_avg'] = features['home_recent_goals_scored'] * 0.9  # Slightly lower than actual goals
            features['away_xG_avg'] = features['away_recent_goals_scored'] * 0.9
            features['home_xGA_avg'] = features['home_recent_goals_conceded'] * 0.9
            features['away_xGA_avg'] = features['away_recent_goals_conceded'] * 0.9
        
        return features
