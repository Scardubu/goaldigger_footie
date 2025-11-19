#!/usr/bin/env python3
"""
Advanced Feature Engineering System for GoalDiggers Platform

Provides sophisticated feature engineering capabilities with automated
feature generation, selection, and transformation pipelines.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import streamlit as st

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering system with automated feature generation.
    """
    
    def __init__(self):
        """Initialize the advanced feature engineering system."""
        self.feature_generators = {}
        self.feature_transformers = {}
        self.feature_selectors = {}
        self.feature_history = []
        
        # Register built-in feature generators
        self._register_builtin_generators()
        
        logger.info("Advanced feature engineering system initialized")
    
    def _register_builtin_generators(self):
        """Register built-in feature generators."""
        self.feature_generators.update({
            'rolling_stats': self._generate_rolling_statistics,
            'team_form': self._generate_team_form_features,
            'head_to_head': self._generate_head_to_head_features,
            'league_position': self._generate_league_position_features,
            'player_stats': self._generate_player_statistics,
            'weather_impact': self._generate_weather_features,
            'temporal': self._generate_temporal_features,
            'interaction': self._generate_interaction_features
        })
    
    def generate_features(self, 
                         data: pd.DataFrame,
                         feature_types: List[str] = None,
                         target_column: str = None) -> pd.DataFrame:
        """
        Generate advanced features from input data.
        
        Args:
            data: Input DataFrame
            feature_types: List of feature types to generate
            target_column: Target column for supervised feature selection
            
        Returns:
            DataFrame with generated features
        """
        if feature_types is None:
            feature_types = list(self.feature_generators.keys())
        
        enhanced_data = data.copy()
        generated_features = []
        
        for feature_type in feature_types:
            if feature_type in self.feature_generators:
                try:
                    logger.info(f"Generating {feature_type} features...")
                    new_features = self.feature_generators[feature_type](enhanced_data)
                    
                    if isinstance(new_features, pd.DataFrame):
                        # Merge new features
                        enhanced_data = pd.concat([enhanced_data, new_features], axis=1)
                        generated_features.extend(new_features.columns.tolist())
                    
                    logger.info(f"Generated {len(new_features.columns) if isinstance(new_features, pd.DataFrame) else 0} {feature_type} features")
                    
                except Exception as e:
                    logger.error(f"Failed to generate {feature_type} features: {e}")
        
        # Record feature generation history
        self.feature_history.append({
            'timestamp': datetime.now(),
            'feature_types': feature_types,
            'generated_count': len(generated_features),
            'total_features': len(enhanced_data.columns),
            'generated_features': generated_features
        })
        
        return enhanced_data
    
    def _generate_rolling_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling statistics features."""
        features = pd.DataFrame(index=data.index)
        
        # Identify numeric columns for rolling stats
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Rolling windows
        windows = [3, 5, 10]
        
        for col in numeric_cols:
            if col in data.columns:
                for window in windows:
                    # Rolling mean
                    features[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling std
                    features[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window, min_periods=1).std()
                    
                    # Rolling min/max
                    features[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window, min_periods=1).min()
                    features[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window, min_periods=1).max()
                    
                    # Rolling trend (slope)
                    features[f'{col}_rolling_trend_{window}'] = data[col].rolling(window=window, min_periods=2).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )
        
        return features
    
    def _generate_team_form_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate team form and momentum features."""
        features = pd.DataFrame(index=data.index)
        
        # Mock team form features (in real implementation, would use match history)
        if 'home_team' in data.columns and 'away_team' in data.columns:
            # Recent form (last 5 matches)
            features['home_team_form_5'] = np.random.uniform(0, 1, len(data))
            features['away_team_form_5'] = np.random.uniform(0, 1, len(data))
            
            # Goal scoring form
            features['home_team_goals_per_game_5'] = np.random.uniform(0.5, 3.0, len(data))
            features['away_team_goals_per_game_5'] = np.random.uniform(0.5, 3.0, len(data))
            
            # Defensive form
            features['home_team_goals_conceded_per_game_5'] = np.random.uniform(0.3, 2.5, len(data))
            features['away_team_goals_conceded_per_game_5'] = np.random.uniform(0.3, 2.5, len(data))
            
            # Form difference
            features['form_difference'] = features['home_team_form_5'] - features['away_team_form_5']
            features['attack_vs_defense'] = features['home_team_goals_per_game_5'] - features['away_team_goals_conceded_per_game_5']
        
        return features
    
    def _generate_head_to_head_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate head-to-head historical features."""
        features = pd.DataFrame(index=data.index)
        
        if 'home_team' in data.columns and 'away_team' in data.columns:
            # Mock head-to-head features
            features['h2h_home_wins_last_5'] = np.random.randint(0, 6, len(data))
            features['h2h_away_wins_last_5'] = np.random.randint(0, 6, len(data))
            features['h2h_draws_last_5'] = 5 - features['h2h_home_wins_last_5'] - features['h2h_away_wins_last_5']
            
            # Goals in head-to-head
            features['h2h_avg_goals_home'] = np.random.uniform(0.5, 3.0, len(data))
            features['h2h_avg_goals_away'] = np.random.uniform(0.5, 3.0, len(data))
            features['h2h_total_goals_avg'] = features['h2h_avg_goals_home'] + features['h2h_avg_goals_away']
            
            # Home advantage in H2H
            features['h2h_home_advantage'] = features['h2h_home_wins_last_5'] / 5.0
        
        return features
    
    def _generate_league_position_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate league position and table-based features."""
        features = pd.DataFrame(index=data.index)
        
        if 'home_team' in data.columns and 'away_team' in data.columns:
            # Mock league positions
            features['home_team_position'] = np.random.randint(1, 21, len(data))
            features['away_team_position'] = np.random.randint(1, 21, len(data))
            
            # Position difference
            features['position_difference'] = features['away_team_position'] - features['home_team_position']
            
            # Points and goal difference (mock)
            features['home_team_points'] = np.random.randint(10, 90, len(data))
            features['away_team_points'] = np.random.randint(10, 90, len(data))
            features['points_difference'] = features['home_team_points'] - features['away_team_points']
            
            features['home_team_goal_diff'] = np.random.randint(-20, 50, len(data))
            features['away_team_goal_diff'] = np.random.randint(-20, 50, len(data))
            features['goal_diff_difference'] = features['home_team_goal_diff'] - features['away_team_goal_diff']
        
        return features
    
    def _generate_player_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate player-based statistical features."""
        features = pd.DataFrame(index=data.index)
        
        # Mock player statistics
        features['home_team_avg_player_rating'] = np.random.uniform(6.0, 8.5, len(data))
        features['away_team_avg_player_rating'] = np.random.uniform(6.0, 8.5, len(data))
        features['player_rating_difference'] = features['home_team_avg_player_rating'] - features['away_team_avg_player_rating']
        
        # Key player availability
        features['home_team_key_players_available'] = np.random.uniform(0.7, 1.0, len(data))
        features['away_team_key_players_available'] = np.random.uniform(0.7, 1.0, len(data))
        
        # Squad depth
        features['home_team_squad_depth'] = np.random.uniform(0.6, 1.0, len(data))
        features['away_team_squad_depth'] = np.random.uniform(0.6, 1.0, len(data))
        
        return features
    
    def _generate_weather_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate weather impact features."""
        features = pd.DataFrame(index=data.index)
        
        # Mock weather features
        features['temperature'] = np.random.uniform(-5, 35, len(data))
        features['humidity'] = np.random.uniform(30, 95, len(data))
        features['wind_speed'] = np.random.uniform(0, 25, len(data))
        features['precipitation'] = np.random.uniform(0, 10, len(data))
        
        # Weather impact scores
        features['weather_impact_score'] = (
            (features['temperature'] - 20).abs() * 0.1 +
            features['wind_speed'] * 0.05 +
            features['precipitation'] * 0.2
        )
        
        # Playing conditions
        features['ideal_conditions'] = (
            (features['temperature'].between(15, 25)) &
            (features['wind_speed'] < 10) &
            (features['precipitation'] < 1)
        ).astype(int)
        
        return features
    
    def _generate_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal and seasonal features."""
        features = pd.DataFrame(index=data.index)
        
        # Create mock datetime if not present
        if 'match_date' not in data.columns:
            base_date = datetime(2024, 1, 1)
            features['match_date'] = [base_date + timedelta(days=i*7) for i in range(len(data))]
        else:
            features['match_date'] = pd.to_datetime(data['match_date'])
        
        # Extract temporal features
        features['day_of_week'] = features['match_date'].dt.dayofweek
        features['month'] = features['match_date'].dt.month
        features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
        
        # Season phase
        features['season_phase'] = features['month'].apply(
            lambda x: 'early' if x in [8, 9, 10] else 'mid' if x in [11, 12, 1, 2] else 'late'
        )
        features['season_phase_encoded'] = features['season_phase'].map(
            {'early': 0, 'mid': 1, 'late': 2}
        )
        
        # Holiday effects
        features['is_holiday_period'] = features['month'].isin([12, 1]).astype(int)
        
        return features
    
    def _generate_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between existing variables."""
        features = pd.DataFrame(index=data.index)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:10]  # Limit to first 10 for performance
        
        # Generate polynomial features for key variables
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                if col1 in data.columns and col2 in data.columns:
                    # Interaction (multiplication)
                    features[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                    
                    # Ratio (avoid division by zero)
                    features[f'{col1}_div_{col2}'] = data[col1] / (data[col2] + 1e-8)
                    
                    # Difference
                    features[f'{col1}_minus_{col2}'] = data[col1] - data[col2]
        
        return features
    
    def select_features(self, 
                       data: pd.DataFrame,
                       target: pd.Series,
                       method: str = "mutual_info",
                       k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most important features.
        
        Args:
            data: Feature DataFrame
            target: Target variable
            method: Selection method ('mutual_info', 'f_classif', 'variance')
            k: Number of features to select
            
        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        try:
            if method == "mutual_info":
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            elif method == "f_classif":
                selector = SelectKBest(score_func=f_classif, k=k)
            else:
                # Default to mutual info
                selector = SelectKBest(score_func=mutual_info_classif, k=k)
            
            # Handle missing values
            data_clean = data.fillna(data.mean())
            
            # Fit and transform
            selected_features = selector.fit_transform(data_clean, target)
            selected_feature_names = data.columns[selector.get_support()].tolist()
            
            selected_df = pd.DataFrame(
                selected_features,
                columns=selected_feature_names,
                index=data.index
            )
            
            logger.info(f"Selected {len(selected_feature_names)} features using {method}")
            return selected_df, selected_feature_names
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return data, data.columns.tolist()
    
    def transform_features(self, 
                          data: pd.DataFrame,
                          method: str = "standard") -> pd.DataFrame:
        """
        Transform features using specified scaling method.
        
        Args:
            data: Input DataFrame
            method: Transformation method ('standard', 'minmax', 'robust')
            
        Returns:
            Transformed DataFrame
        """
        try:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {method}, using standard")
                scaler = StandardScaler()
            
            # Only transform numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return data
            
            transformed_data = data.copy()
            transformed_data[numeric_cols] = scaler.fit_transform(data[numeric_cols].fillna(0))
            
            logger.info(f"Transformed {len(numeric_cols)} features using {method} scaling")
            return transformed_data
            
        except Exception as e:
            logger.error(f"Feature transformation failed: {e}")
            return data
    
    def get_feature_importance_report(self, 
                                    data: pd.DataFrame,
                                    target: pd.Series) -> Dict[str, Any]:
        """
        Generate a comprehensive feature importance report.
        
        Args:
            data: Feature DataFrame
            target: Target variable
            
        Returns:
            Dictionary with feature importance analysis
        """
        try:
            # Calculate mutual information scores
            data_clean = data.fillna(data.mean())
            mi_scores = mutual_info_classif(data_clean, target)
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': data.columns,
                'mutual_info_score': mi_scores,
                'abs_mutual_info': np.abs(mi_scores)
            }).sort_values('abs_mutual_info', ascending=False)
            
            # Calculate correlations with target
            correlations = data_clean.corrwith(target).abs()
            importance_df['correlation'] = importance_df['feature'].map(correlations)
            
            # Feature statistics
            feature_stats = {
                'total_features': len(data.columns),
                'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(data.select_dtypes(include=['object', 'category']).columns),
                'missing_values': data.isnull().sum().sum(),
                'top_features': importance_df.head(10).to_dict('records')
            }
            
            return {
                'feature_importance': importance_df,
                'statistics': feature_stats,
                'generation_history': self.feature_history
            }
            
        except Exception as e:
            logger.error(f"Failed to generate feature importance report: {e}")
            return {'error': str(e)}

# Global instance for easy access
advanced_feature_engineer = AdvancedFeatureEngineer()
