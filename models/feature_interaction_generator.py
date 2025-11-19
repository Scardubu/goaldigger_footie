"""
Feature Interaction Generator

Generates interaction features from base features to capture non-linear relationships
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureInteractionGenerator:
    """
    Generates interaction features from base features
    
    Interactions capture:
    - Multiplicative relationships (A * B)
    - Ratio relationships (A / B)
    - Difference relationships (A - B)
    - Power relationships (A^2)
    """
    
    def __init__(self):
        """Initialize the feature interaction generator"""
        logger.info("🔧 Feature Interaction Generator initialized")
        
        # Define feature groups for targeted interactions
        self.feature_groups = {
            'form': [
                'home_form_points_per_game', 'away_form_points_per_game',
                'home_form_goals_scored_per_game', 'away_form_goals_scored_per_game',
                'home_form_goals_conceded_per_game', 'away_form_goals_conceded_per_game',
                'home_form_clean_sheets_pct', 'away_form_clean_sheets_pct',
                'home_form_win_pct', 'away_form_win_pct'
            ],
            'strength': [
                'home_elo', 'away_elo', 'elo_diff',
                'home_attack_strength', 'away_attack_strength',
                'home_defense_strength', 'away_defense_strength'
            ],
            'momentum': [
                'home_form_gradient', 'away_form_gradient',
                'home_win_streak', 'away_win_streak',
                'home_unbeaten_run', 'away_unbeaten_run'
            ],
            'h2h': [
                'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate',
                'h2h_recent_form', 'h2h_avg_goals', 'h2h_goal_differential'
            ]
        }
    
    def generate_interactions(
        self,
        X: pd.DataFrame,
        max_interactions: int = 15,
        enable: bool = False
    ) -> pd.DataFrame:
        """
        Generate interaction features from base features
        
        Args:
            X: Input DataFrame with base features
            max_interactions: Maximum number of interaction features to create
            enable: Whether to generate interactions (disabled by default due to accuracy decrease)
            
        Returns:
            DataFrame with original features + interactions (or just original if disabled)
        """
        if not enable:
            logger.info("⏭️  Feature interactions disabled (caused -5.5% accuracy drop)")
            return X
        
        logger.info("🔄 Generating feature interactions...")
        start_time = datetime.now()
        
        # Copy original data
        X_with_interactions = X.copy()
        interaction_count = 0
        
        # Only generate the most predictive interactions (very selective)
        # Focus on form*strength interactions which are theoretically sound
        interactions = []
        
        # 1. Attack strength * goals scored (offensive power)
        if 'home_attack_strength' in X.columns and 'home_form_goals_scored_per_game' in X.columns:
            interactions.append(
                X['home_attack_strength'] * X['home_form_goals_scored_per_game']
            )
            X_with_interactions['home_offensive_power'] = interactions[-1]
            
        if 'away_attack_strength' in X.columns and 'away_form_goals_scored_per_game' in X.columns:
            interactions.append(
                X['away_attack_strength'] * X['away_form_goals_scored_per_game']
            )
            X_with_interactions['away_offensive_power'] = interactions[-1]
        
        # 2. Defense strength * goals conceded (defensive vulnerability)
        if 'home_defense_strength' in X.columns and 'home_form_goals_conceded_per_game' in X.columns:
            interactions.append(
                X['home_defense_strength'] * X['home_form_goals_conceded_per_game']
            )
            X_with_interactions['home_defensive_exposure'] = interactions[-1]
            
        if 'away_defense_strength' in X.columns and 'away_form_goals_conceded_per_game' in X.columns:
            interactions.append(
                X['away_defense_strength'] * X['away_form_goals_conceded_per_game']
            )
            X_with_interactions['away_defensive_exposure'] = interactions[-1]
        
        interaction_count = len(interactions)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"✅ Generated {interaction_count} interaction features in {elapsed:.2f}s")
        logger.info(f"   Total features: {len(X.columns)} → {len(X_with_interactions.columns)}")
        
        return X_with_interactions
    
    def _generate_multiplicative_interactions(
        self,
        X: pd.DataFrame,
        features: List[str],
        prefix: str
    ) -> pd.DataFrame:
        """Generate multiplicative interactions between features"""
        interactions = pd.DataFrame(index=X.index)
        
        # Generate pairwise multiplications for related features
        for i, feat1 in enumerate(features):
            for feat2 in features[i+1:]:
                if feat1 in X.columns and feat2 in X.columns:
                    interaction_name = f"{prefix}mult_{feat1}_{feat2}"
                    interactions[interaction_name] = X[feat1] * X[feat2]
        
        return interactions
    
    def _generate_ratio_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate ratio-based interactions for strength comparisons"""
        interactions = pd.DataFrame(index=X.index)
        
        # Define safe ratio pairs (avoid division by zero)
        ratio_pairs = [
            ('home_form_points_per_game', 'away_form_points_per_game', 'form_ppg_ratio'),
            ('home_form_goals_scored_per_game', 'away_form_goals_scored_per_game', 'form_attack_ratio'),
            ('home_form_goals_conceded_per_game', 'away_form_goals_conceded_per_game', 'form_defense_ratio'),
            ('home_elo', 'away_elo', 'elo_ratio'),
            ('home_attack_strength', 'away_defense_strength', 'attack_vs_defense_ratio'),
            ('away_attack_strength', 'home_defense_strength', 'away_attack_vs_home_defense_ratio')
        ]
        
        for feat1, feat2, name in ratio_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                # Add small epsilon to avoid division by zero
                interactions[name] = X[feat1] / (X[feat2] + 1e-6)
        
        return interactions
    
    def _generate_difference_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate difference-based interactions for comparative features"""
        interactions = pd.DataFrame(index=X.index)
        
        # Define difference pairs (comparative features)
        difference_pairs = [
            ('home_form_points_per_game', 'away_form_points_per_game', 'form_ppg_diff'),
            ('home_form_goals_scored_per_game', 'away_form_goals_scored_per_game', 'form_attack_diff'),
            ('home_form_goals_conceded_per_game', 'away_form_goals_conceded_per_game', 'form_defense_diff'),
            ('home_form_win_pct', 'away_form_win_pct', 'form_win_pct_diff'),
            ('home_win_streak', 'away_win_streak', 'win_streak_diff'),
            ('home_unbeaten_run', 'away_unbeaten_run', 'unbeaten_run_diff'),
            ('home_form_gradient', 'away_form_gradient', 'form_momentum_diff')
        ]
        
        for feat1, feat2, name in difference_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                interactions[name] = X[feat1] - X[feat2]
        
        return interactions
    
    def _generate_polynomial_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial interactions (squared terms) for key features"""
        interactions = pd.DataFrame(index=X.index)
        
        # Define features to square (capture non-linear relationships)
        polynomial_features = [
            'elo_diff',
            'home_form_gradient',
            'away_form_gradient',
            'h2h_recent_form',
            'composite_score',
            'head_to_head'
        ]
        
        for feat in polynomial_features:
            if feat in X.columns:
                interactions[f"{feat}_squared"] = X[feat] ** 2
        
        return interactions
    
    def get_interaction_importance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model: Any
    ) -> pd.DataFrame:
        """
        Analyze importance of interaction features
        
        Args:
            X_train: Training data with interactions
            y_train: Training labels
            model: Trained model
            
        Returns:
            DataFrame with interaction feature importances
        """
        logger.info("📊 Analyzing interaction feature importance...")
        
        # Get feature importances from model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Filter to interaction features only
        interaction_keywords = ['mult_', 'ratio', 'diff', 'squared']
        interaction_df = importance_df[
            importance_df['feature'].str.contains('|'.join(interaction_keywords))
        ]
        
        logger.info(f"✅ Found {len(interaction_df)} interaction features")
        logger.info(f"   Top 5 interactions:\n{interaction_df.head().to_string()}")
        
        return interaction_df
