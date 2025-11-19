#!/usr/bin/env python3
"""
Feature Mapper

Maps enhanced features (98+) to model features (28) to resolve prediction accuracy issues.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

class FeatureMapper:
    """Maps enhanced features to model-expected features."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize feature mapper with model configuration."""
        self.config_path = config_path or "config/model_params.yaml"
        self.model_features = []
        self.feature_mapping = {}
        self._load_model_features()
        self._create_feature_mapping()
    
    def _load_model_features(self):
        """Load expected model features from configuration."""
        # Always use the exact features the model expects (from model file analysis)
        # The config file has incorrect features, so we override it
        logger.info("Using exact model features from trained model analysis")
        self._use_default_features()
    
    def _use_default_features(self):
        """Use default model features if config loading fails."""
        # These are the exact features the trained model expects (from model file)
        self.model_features = [
            'h2h_team1_wins', 'h2h_draws', 'h2h_team2_wins', 'h2h_avg_goals',
            'home_formation', 'away_formation', 'formation_clash_score',
            'home_match_xg', 'away_match_xg', 'home_injury_impact', 'away_injury_impact',
            'home_elo', 'away_elo', 'elo_diff', 'home_form_points_last_5', 'away_form_points_last_5',
            'home_avg_goals_scored_last_5', 'home_avg_goals_conceded_last_5',
            'away_avg_goals_scored_last_5', 'away_avg_goals_conceded_last_5'
        ]
        logger.info(f"Using {len(self.model_features)} default model features")
    
    def _create_feature_mapping(self):
        """Create mapping from enhanced features to model features."""
        self.feature_mapping = {
            # Head-to-head features
            'h2h_team1_wins': ['home_h2h_wins', 'h2h_home_wins', 'home_historical_wins'],
            'h2h_draws': ['h2h_draws', 'historical_draws', 'h2h_draw_rate'],
            'h2h_team2_wins': ['away_h2h_wins', 'h2h_away_wins', 'away_historical_wins'],
            'h2h_avg_goals': ['h2h_avg_goals', 'historical_avg_goals', 'h2h_goals_avg'],
            'h2h_time_weighted': ['h2h_time_weight', 'h2h_recency', 'h2h_weighted_score'],
            
            # Formation features
            'home_formation': ['home_formation', 'home_tactical_setup', 'home_system'],
            'away_formation': ['away_formation', 'away_tactical_setup', 'away_system'],
            'formation_clash_score': ['tactical_clash', 'formation_advantage', 'tactical_mismatch'],
            
            # Injury features
            'home_injury_impact': ['home_injury_severity', 'home_player_availability', 'home_squad_strength'],
            'away_injury_impact': ['away_injury_severity', 'away_player_availability', 'away_squad_strength'],
            
            # ELO features
            'home_elo': ['home_team_elo', 'home_strength_rating', 'home_team_rating'],
            'away_elo': ['away_team_elo', 'away_strength_rating', 'away_team_rating'],
            'elo_diff': ['elo_difference', 'strength_difference', 'rating_gap'],
            
            # Form features
            'home_form_points_last_5': ['home_recent_form', 'home_form_5', 'home_recent_points'],
            'away_form_points_last_5': ['away_recent_form', 'away_form_5', 'away_recent_points'],
            'home_avg_goals_scored_last_5': ['home_attack_form', 'home_goals_recent', 'home_scoring_form'],
            'home_avg_goals_conceded_last_5': ['home_defense_form', 'home_conceded_recent', 'home_defensive_form'],
            'away_avg_goals_scored_last_5': ['away_attack_form', 'away_goals_recent', 'away_scoring_form'],
            'away_avg_goals_conceded_last_5': ['away_defense_form', 'away_conceded_recent', 'away_defensive_form'],
            
            # Expected goals
            'home_match_xg': ['home_expected_goals', 'home_xg', 'home_attack_quality'],
            'away_match_xg': ['away_expected_goals', 'away_xg', 'away_attack_quality'],
            
            # Weather features
            'weather_temp': ['temperature', 'weather_temperature', 'temp_celsius'],
            'weather_precip': ['precipitation', 'rainfall', 'weather_rain'],
            'weather_wind': ['wind_speed', 'weather_wind_speed', 'wind_mph'],
            
            # Substitution features
            'substitutions_home': ['home_subs', 'home_substitutions', 'home_tactical_changes'],
            'substitutions_away': ['away_subs', 'away_substitutions', 'away_tactical_changes'],
            
            # Opposition strength
            'home_recent_opp_elo': ['home_opposition_strength', 'home_schedule_difficulty', 'home_opp_rating'],
            'away_recent_opp_elo': ['away_opposition_strength', 'away_schedule_difficulty', 'away_opp_rating'],
            
            # Rest features
            'home_rest_days': ['home_rest', 'home_recovery_time', 'home_days_rest'],
            'away_rest_days': ['away_rest', 'away_recovery_time', 'away_days_rest']
        }
        
        logger.info(f"Created feature mapping for {len(self.feature_mapping)} model features")
    
    def map_features(self, enhanced_features: Dict[str, float]) -> pd.DataFrame:
        """
        Map enhanced features to model-expected features.
        
        Args:
            enhanced_features: Dictionary of enhanced features
            
        Returns:
            DataFrame with model-expected features
        """
        try:
            mapped_features = {}
            
            for model_feature in self.model_features:
                value = self._map_single_feature(model_feature, enhanced_features)
                mapped_features[model_feature] = value
            
            # Create DataFrame with single row
            df = pd.DataFrame([mapped_features])
            
            logger.debug(f"Mapped {len(enhanced_features)} enhanced features to {len(mapped_features)} model features")
            return df
            
        except Exception as e:
            logger.error(f"Feature mapping failed: {e}")
            return self._create_fallback_features()
    
    def _map_single_feature(self, model_feature: str, enhanced_features: Dict[str, float]) -> float:
        """Map a single model feature from enhanced features."""
        try:
            # Get possible enhanced feature names for this model feature
            possible_names = self.feature_mapping.get(model_feature, [model_feature])
            
            # Try to find matching enhanced feature
            for name in possible_names:
                if name in enhanced_features:
                    return float(enhanced_features[name])
            
            # If no direct match, try partial matching
            for enhanced_name, value in enhanced_features.items():
                for possible_name in possible_names:
                    if possible_name.lower() in enhanced_name.lower() or enhanced_name.lower() in possible_name.lower():
                        return float(value)
            
            # If still no match, use default value
            default_value = self._get_default_value(model_feature)
            logger.debug(f"No mapping found for {model_feature}, using default: {default_value}")
            return default_value
            
        except Exception as e:
            logger.error(f"Error mapping feature {model_feature}: {e}")
            return self._get_default_value(model_feature)
    
    def _get_default_value(self, feature_name: str) -> float:
        """Get default value for a feature."""
        # Default values based on feature type
        defaults = {
            'h2h_team1_wins': 0.0,
            'h2h_draws': 0.0,
            'h2h_team2_wins': 0.0,
            'h2h_avg_goals': 2.5,
            'home_formation': 0.0,  # Default formation encoding
            'away_formation': 0.0,  # Default formation encoding
            'formation_clash_score': 0.0,
            'home_match_xg': 1.5,
            'away_match_xg': 1.5,
            'home_injury_impact': 0.0,
            'away_injury_impact': 0.0,
            'home_elo': 1500.0,
            'away_elo': 1500.0,
            'elo_diff': 0.0,
            'home_form_points_last_5': 7.5,
            'away_form_points_last_5': 7.5,
            'home_avg_goals_scored_last_5': 1.5,
            'home_avg_goals_conceded_last_5': 1.5,
            'away_avg_goals_scored_last_5': 1.5,
            'away_avg_goals_conceded_last_5': 1.5
        }
        
        return defaults.get(feature_name, 0.0)
    
    def _create_fallback_features(self) -> pd.DataFrame:
        """Create fallback features when mapping fails."""
        fallback_data = {feature: self._get_default_value(feature) for feature in self.model_features}
        return pd.DataFrame([fallback_data])
    
    def get_model_features(self) -> List[str]:
        """Get list of expected model features."""
        return self.model_features.copy()
    
    def validate_features(self, features: pd.DataFrame) -> bool:
        """Validate that features match model expectations."""
        try:
            # Check if all required features are present
            missing_features = set(self.model_features) - set(features.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                return False
            
            # Check for non-numeric values
            for col in self.model_features:
                if col in features.columns:
                    if not pd.api.types.is_numeric_dtype(features[col]):
                        logger.warning(f"Non-numeric feature: {col}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Feature validation failed: {e}")
            return False
