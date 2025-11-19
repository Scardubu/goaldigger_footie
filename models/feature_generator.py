"""
Feature Generator for Football Match Prediction
Advanced feature engineering for football analytics and betting insights
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')

from utils.logging_config import get_logger

logger = get_logger(__name__)

class FeatureGenerator:
    """Advanced feature generator for football match prediction."""
    
    def __init__(self, feature_config: Optional[Dict] = None):
        """Initialize the feature generator."""
        self.feature_config = feature_config or self._default_config()
        self.scalers = {}
        self.feature_stats = {}
        
        logger.info("Feature Generator initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default feature generation configuration."""
        return {
            'temporal_features': {
                'enabled': True,
                'window_sizes': [5, 10, 15],  # Recent form windows
                'seasonal_adjustment': True
            },
            'team_strength_features': {
                'enabled': True,
                'attack_strength': True,
                'defense_strength': True,
                'overall_rating': True
            },
            'match_context_features': {
                'enabled': True,
                'home_advantage': True,
                'rivalry_factor': False,
                'importance_weight': False
            },
            'statistical_features': {
                'enabled': True,
                'rolling_averages': True,
                'form_trends': True,
                'head_to_head': True
            },
            'advanced_features': {
                'enabled': True,
                'momentum_indicators': True,
                'pressure_metrics': False,
                'weather_impact': False
            },
            'betting_features': {
                'enabled': True,
                'value_indicators': True,
                'market_sentiment': False,
                'odds_movements': False
            }
        }
    
    def generate_comprehensive_features(self, match_data: pd.DataFrame, 
                                      historical_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Generate comprehensive feature set for match prediction."""
        logger.info(f"Generating comprehensive features for {len(match_data)} matches")
        
        features_df = match_data.copy()
        
        # Basic team features
        features_df = self._add_basic_team_features(features_df)
        
        # Temporal features
        if self.feature_config['temporal_features']['enabled']:
            features_df = self._add_temporal_features(features_df, historical_data)
        
        # Team strength features
        if self.feature_config['team_strength_features']['enabled']:
            features_df = self._add_team_strength_features(features_df)
        
        # Match context features
        if self.feature_config['match_context_features']['enabled']:
            features_df = self._add_match_context_features(features_df)
        
        # Statistical features
        if self.feature_config['statistical_features']['enabled']:
            features_df = self._add_statistical_features(features_df)
        
        # Advanced features
        if self.feature_config['advanced_features']['enabled']:
            features_df = self._add_advanced_features(features_df)
        
        # Betting features
        if self.feature_config['betting_features']['enabled']:
            features_df = self._add_betting_features(features_df)
        
        # Clean and normalize features
        features_df = self._clean_and_normalize_features(features_df)
        
        logger.info(f"Feature generation completed: {len(features_df.columns)} total features")
        return features_df
    
    def _add_basic_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic team performance features."""
        # Mock team performance data
        np.random.seed(42)
        
        # Goals statistics
        df['home_goals_per_game'] = np.random.uniform(0.8, 2.5, len(df))
        df['away_goals_per_game'] = np.random.uniform(0.7, 2.2, len(df))
        df['home_goals_conceded_per_game'] = np.random.uniform(0.6, 2.0, len(df))
        df['away_goals_conceded_per_game'] = np.random.uniform(0.8, 2.3, len(df))
        
        # Goal differences
        df['home_goal_difference'] = df['home_goals_per_game'] - df['home_goals_conceded_per_game']
        df['away_goal_difference'] = df['away_goals_per_game'] - df['away_goals_conceded_per_game']
        df['goal_diff_comparison'] = df['home_goal_difference'] - df['away_goal_difference']
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame, historical_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Add temporal and form-based features."""
        np.random.seed(42)
        
        # Recent form (mock data)
        for window in self.feature_config['temporal_features']['window_sizes']:
            df[f'home_form_{window}'] = np.random.uniform(0.2, 0.9, len(df))
            df[f'away_form_{window}'] = np.random.uniform(0.2, 0.9, len(df))
            df[f'form_diff_{window}'] = df[f'home_form_{window}'] - df[f'away_form_{window}']
        
        # Seasonal adjustment
        if self.feature_config['temporal_features']['seasonal_adjustment']:
            # Mock seasonal factors (would be based on actual dates)
            df['season_progress'] = np.random.uniform(0.1, 0.9, len(df))
            df['home_seasonal_factor'] = 1 + np.random.uniform(-0.2, 0.2, len(df))
            df['away_seasonal_factor'] = 1 + np.random.uniform(-0.2, 0.2, len(df))
        
        return df
    
    def _add_team_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team strength and quality indicators."""
        np.random.seed(42)
        
        if self.feature_config['team_strength_features']['attack_strength']:
            df['home_attack_strength'] = np.random.uniform(0.3, 0.95, len(df))
            df['away_attack_strength'] = np.random.uniform(0.3, 0.95, len(df))
        
        if self.feature_config['team_strength_features']['defense_strength']:
            df['home_defense_strength'] = np.random.uniform(0.3, 0.95, len(df))
            df['away_defense_strength'] = np.random.uniform(0.3, 0.95, len(df))
        
        if self.feature_config['team_strength_features']['overall_rating']:
            df['home_overall_rating'] = (
                0.6 * df.get('home_attack_strength', 0.5) + 
                0.4 * df.get('home_defense_strength', 0.5)
            )
            df['away_overall_rating'] = (
                0.6 * df.get('away_attack_strength', 0.5) + 
                0.4 * df.get('away_defense_strength', 0.5)
            )
            df['quality_difference'] = df['home_overall_rating'] - df['away_overall_rating']
        
        return df
    
    def _add_match_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add match context and situational features."""
        np.random.seed(42)
        
        if self.feature_config['match_context_features']['home_advantage']:
            # Home advantage varies by team and league
            df['home_advantage_factor'] = np.random.uniform(0.1, 0.4, len(df))
            df['home_points_per_game'] = np.random.uniform(1.0, 2.5, len(df))
            df['away_points_per_game'] = np.random.uniform(0.8, 2.2, len(df))
        
        # Match importance (derby, relegation battle, etc.)
        if self.feature_config['match_context_features']['importance_weight']:
            df['match_importance'] = np.random.uniform(1.0, 1.5, len(df))
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical and analytical features."""
        np.random.seed(42)
        
        if self.feature_config['statistical_features']['rolling_averages']:
            # Rolling performance metrics
            df['home_shots_per_game'] = np.random.uniform(8, 20, len(df))
            df['away_shots_per_game'] = np.random.uniform(8, 20, len(df))
            df['home_shots_on_target_ratio'] = np.random.uniform(0.25, 0.55, len(df))
            df['away_shots_on_target_ratio'] = np.random.uniform(0.25, 0.55, len(df))
            
            # Possession and style metrics
            df['home_possession_avg'] = np.random.uniform(0.35, 0.65, len(df))
            df['away_possession_avg'] = np.random.uniform(0.35, 0.65, len(df))
            df['possession_difference'] = df['home_possession_avg'] - df['away_possession_avg']
        
        if self.feature_config['statistical_features']['head_to_head']:
            # Head-to-head statistics
            df['h2h_home_wins_last5'] = np.random.poisson(1.5, len(df))
            df['h2h_away_wins_last5'] = np.random.poisson(1.2, len(df))
            df['h2h_draws_last5'] = np.random.poisson(0.8, len(df))
            df['h2h_home_win_ratio'] = (
                df['h2h_home_wins_last5'] / 
                (df['h2h_home_wins_last5'] + df['h2h_away_wins_last5'] + df['h2h_draws_last5'] + 0.1)
            )
        
        return df
    
    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced analytical features."""
        np.random.seed(42)
        
        if self.feature_config['advanced_features']['momentum_indicators']:
            # Momentum and trend indicators
            df['home_momentum_3game'] = np.random.uniform(-1, 1, len(df))
            df['away_momentum_3game'] = np.random.uniform(-1, 1, len(df))
            df['momentum_difference'] = df['home_momentum_3game'] - df['away_momentum_3game']
            
            # Performance trends
            df['home_trend_goals'] = np.random.uniform(-0.5, 0.5, len(df))
            df['away_trend_goals'] = np.random.uniform(-0.5, 0.5, len(df))
        
        # Expected Goals (xG) features
        df['home_xg_per_game'] = np.random.uniform(0.8, 2.8, len(df))
        df['away_xg_per_game'] = np.random.uniform(0.7, 2.5, len(df))
        df['home_xga_per_game'] = np.random.uniform(0.6, 2.2, len(df))
        df['away_xga_per_game'] = np.random.uniform(0.8, 2.5, len(df))
        
        # xG differences and efficiency
        df['home_xg_difference'] = df['home_xg_per_game'] - df['home_xga_per_game']
        df['away_xg_difference'] = df['away_xg_per_game'] - df['away_xga_per_game']
        df['xg_diff_comparison'] = df['home_xg_difference'] - df['away_xg_difference']
        
        return df
    
    def _add_betting_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add betting-related features for value analysis."""
        np.random.seed(42)
        
        if self.feature_config['betting_features']['value_indicators']:
            # Mock betting odds and probabilities
            df['home_win_probability'] = np.random.uniform(0.2, 0.7, len(df))
            df['draw_probability'] = np.random.uniform(0.15, 0.35, len(df))
            df['away_win_probability'] = 1 - df['home_win_probability'] - df['draw_probability']
            
            # Implied odds
            df['home_implied_odds'] = 1 / df['home_win_probability']
            df['draw_implied_odds'] = 1 / df['draw_probability']
            df['away_implied_odds'] = 1 / df['away_win_probability']
            
            # Value indicators (comparing model prediction vs market)
            df['home_value_indicator'] = np.random.uniform(0.8, 1.3, len(df))
            df['draw_value_indicator'] = np.random.uniform(0.8, 1.3, len(df))
            df['away_value_indicator'] = np.random.uniform(0.8, 1.3, len(df))
            
            # Best value bet indicator
            value_cols = ['home_value_indicator', 'draw_value_indicator', 'away_value_indicator']
            df['best_value_bet'] = df[value_cols].idxmax(axis=1).str.replace('_value_indicator', '')
            df['best_value_score'] = df[value_cols].max(axis=1)
        
        # Over/Under features
        df['total_goals_prediction'] = df['home_xg_per_game'] + df['away_xg_per_game']
        df['over_2_5_probability'] = np.where(df['total_goals_prediction'] > 2.5, 
                                            np.random.uniform(0.6, 0.9, len(df)),
                                            np.random.uniform(0.1, 0.4, len(df)))
        
        return df
    
    def _clean_and_normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize features."""
        # Handle infinite values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with median
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Cap extreme values (winsorization)
        for col in numeric_columns:
            if df[col].std() > 0:  # Only process columns with variance
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        return df
    
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance scores (mock implementation)."""
        np.random.seed(42)
        
        # Mock feature importance based on feature type
        importance_scores = {}
        
        for feature in feature_names:
            if any(keyword in feature.lower() for keyword in ['goal', 'xg', 'strength']):
                importance_scores[feature] = np.random.uniform(0.6, 1.0)
            elif any(keyword in feature.lower() for keyword in ['form', 'momentum', 'trend']):
                importance_scores[feature] = np.random.uniform(0.4, 0.8)
            elif any(keyword in feature.lower() for keyword in ['h2h', 'advantage', 'quality']):
                importance_scores[feature] = np.random.uniform(0.3, 0.7)
            else:
                importance_scores[feature] = np.random.uniform(0.1, 0.5)
        
        # Normalize to sum to 1
        total = sum(importance_scores.values())
        importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        # Sort by importance
        importance_scores = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        return importance_scores
    
    def generate_betting_insights(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate betting insights from features."""
        logger.info("Generating betting insights...")
        
        insights = {
            'high_value_matches': [],
            'safe_bets': [],
            'risky_bets': [],
            'over_under_recommendations': [],
            'summary_stats': {}
        }
        
        if 'best_value_score' in features_df.columns:
            # High value matches (top 20%)
            high_value_threshold = features_df['best_value_score'].quantile(0.8)
            high_value_matches = features_df[features_df['best_value_score'] >= high_value_threshold]
            
            insights['high_value_matches'] = [
                {
                    'match_id': idx,
                    'value_score': row['best_value_score'],
                    'recommended_bet': row['best_value_bet'],
                    'confidence': row.get('home_win_probability', 0.5)
                }
                for idx, row in high_value_matches.iterrows()
            ]
        
        if 'over_2_5_probability' in features_df.columns:
            # Over/Under recommendations
            over_recommendations = features_df[features_df['over_2_5_probability'] > 0.7]
            under_recommendations = features_df[features_df['over_2_5_probability'] < 0.3]
            
            insights['over_under_recommendations'] = {
                'over_2_5_recommendations': len(over_recommendations),
                'under_2_5_recommendations': len(under_recommendations),
                'high_scoring_matches': over_recommendations.index.tolist()[:10],
                'low_scoring_matches': under_recommendations.index.tolist()[:10]
            }
        
        # Summary statistics
        insights['summary_stats'] = {
            'total_matches_analyzed': len(features_df),
            'high_confidence_predictions': len(features_df[features_df.get('home_win_probability', 0.5) > 0.7]),
            'balanced_matches': len(features_df[
                (features_df.get('home_win_probability', 0.5) > 0.3) & 
                (features_df.get('home_win_probability', 0.5) < 0.7)
            ]),
            'features_generated': len(features_df.columns)
        }
        
        logger.info("Betting insights generated successfully")
        return insights
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get current feature configuration."""
        return self.feature_config
    
    def update_feature_config(self, new_config: Dict[str, Any]):
        """Update feature configuration."""
        self.feature_config.update(new_config)
        logger.info("Feature configuration updated")
        self.feature_config.update(new_config)
        logger.info("Feature configuration updated")
