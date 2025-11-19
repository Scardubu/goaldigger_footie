#!/usr/bin/env python3
"""
Advanced Feature Engineering System for Production
Implements systematic feature engineering with performance tracking
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureMetrics:
    """Track feature engineering metrics"""
    feature_count: int
    computation_time: float
    missing_value_ratio: float
    correlation_max: float
    variance_min: float
    timestamp: datetime


class AdvancedFeatureEngineeringSystem:
    """
    Production-grade feature engineering with advanced metrics
    Adds momentum indicators, splits, cross-league strength, and more
    """
    
    def __init__(self):
        self.feature_catalog = {}
        self.feature_metrics = []
        self.baseline_features = self._load_baseline_features()
        logger.info("🔧 Advanced Feature Engineering System initialized")
    
    def _load_baseline_features(self) -> List[str]:
        """Load baseline feature set from config"""
        try:
            import yaml
            with open('config/model_params.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config.get('normalization', {}).get('feature_list', [])
        except Exception as e:
            logger.warning(f"Could not load baseline features: {e}")
            return []
    
    def generate_enhanced_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate enhanced feature set with advanced metrics
        
        New Features Added:
        - Momentum indicators (form gradient, acceleration)
        - Home/away performance splits
        - Head-to-head dominance patterns
        - Recent form trends (weighted recency)
        - Cross-league strength ratings
        - Goal-scoring patterns (variance, consistency)
        - Defensive solidity metrics
        - Late-game performance (last 15min scoring)
        """
        start_time = datetime.now()
        features = {}
        
        try:
            # Extract baseline features
            features.update(self._extract_baseline_features(match_data))
            
            # 1. Momentum Indicators
            features.update(self._calculate_momentum_indicators(match_data))
            
            # 2. Home/Away Performance Splits
            features.update(self._calculate_venue_splits(match_data))
            
            # 3. Head-to-Head Dominance Patterns
            features.update(self._calculate_h2h_patterns(match_data))
            
            # 4. Recent Form Trends (weighted recency)
            features.update(self._calculate_form_trends(match_data))
            
            # 5. Cross-League Strength Ratings
            features.update(self._calculate_cross_league_strength(match_data))
            
            # 6. Goal-Scoring Patterns
            features.update(self._calculate_scoring_patterns(match_data))
            
            # 7. Defensive Solidity Metrics
            features.update(self._calculate_defensive_metrics(match_data))
            
            # 8. Late-Game Performance
            features.update(self._calculate_late_game_performance(match_data))
            
            # 9. Fatigue and Rest Advantage (enhanced)
            features.update(self._calculate_rest_advantage(match_data))
            
            # 10. Tactical Matchup Features
            features.update(self._calculate_tactical_features(match_data))
            
            # Track metrics
            computation_time = (datetime.now() - start_time).total_seconds()
            self._track_feature_metrics(features, computation_time)
            
            logger.info(f"✅ Generated {len(features)} enhanced features in {computation_time:.3f}s")
            return features
            
        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            return self._generate_fallback_features(match_data)
    
    def _extract_baseline_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract existing baseline features"""
        features = {}
        for feature_name in self.baseline_features:
            features[feature_name] = float(match_data.get(feature_name, 0.0))
        return features
    
    def _calculate_momentum_indicators(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate momentum indicators
        - Form gradient (acceleration/deceleration)
        - Points per game trend
        - Win streak indicators
        """
        features = {}
        
        # Extract form data
        home_form = match_data.get('home_form_last_10', [])
        away_form = match_data.get('away_form_last_10', [])
        
        # Calculate form gradient (recent 5 vs previous 5)
        if len(home_form) >= 10:
            home_recent = sum(home_form[:5]) / 5.0
            home_previous = sum(home_form[5:10]) / 5.0
            features['home_form_gradient'] = home_recent - home_previous
        else:
            features['home_form_gradient'] = 0.0
        
        if len(away_form) >= 10:
            away_recent = sum(away_form[:5]) / 5.0
            away_previous = sum(away_form[5:10]) / 5.0
            features['away_form_gradient'] = away_recent - away_previous
        else:
            features['away_form_gradient'] = 0.0
        
        # Win streak indicators
        home_streak = self._calculate_streak(home_form)
        away_streak = self._calculate_streak(away_form)
        features['home_win_streak'] = max(home_streak, 0.0)
        features['away_win_streak'] = max(away_streak, 0.0)
        features['home_losing_streak'] = abs(min(home_streak, 0.0))
        features['away_losing_streak'] = abs(min(away_streak, 0.0))
        
        return features
    
    def _calculate_venue_splits(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate home/away performance splits"""
        features = {}
        
        # Home team's home performance
        home_at_home_wins = match_data.get('home_home_wins', 0)
        home_at_home_draws = match_data.get('home_home_draws', 0)
        home_at_home_losses = match_data.get('home_home_losses', 0)
        home_at_home_total = home_at_home_wins + home_at_home_draws + home_at_home_losses
        
        if home_at_home_total > 0:
            features['home_home_win_rate'] = home_at_home_wins / home_at_home_total
            features['home_home_points_per_game'] = (home_at_home_wins * 3 + home_at_home_draws) / home_at_home_total
        else:
            features['home_home_win_rate'] = 0.5
            features['home_home_points_per_game'] = 1.5
        
        # Away team's away performance
        away_at_away_wins = match_data.get('away_away_wins', 0)
        away_at_away_draws = match_data.get('away_away_draws', 0)
        away_at_away_losses = match_data.get('away_away_losses', 0)
        away_at_away_total = away_at_away_wins + away_at_away_draws + away_at_away_losses
        
        if away_at_away_total > 0:
            features['away_away_win_rate'] = away_at_away_wins / away_at_away_total
            features['away_away_points_per_game'] = (away_at_away_wins * 3 + away_at_away_draws) / away_at_away_total
        else:
            features['away_away_win_rate'] = 0.3
            features['away_away_points_per_game'] = 1.0
        
        # Venue advantage differential
        features['venue_advantage_differential'] = features['home_home_points_per_game'] - features['away_away_points_per_game']
        
        return features
    
    def _calculate_h2h_patterns(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate head-to-head dominance patterns"""
        features = {}
        
        h2h_home_wins = match_data.get('h2h_team1_wins', 0)
        h2h_draws = match_data.get('h2h_draws', 0)
        h2h_away_wins = match_data.get('h2h_team2_wins', 0)
        h2h_total = h2h_home_wins + h2h_draws + h2h_away_wins
        
        if h2h_total > 0:
            features['h2h_home_dominance'] = (h2h_home_wins - h2h_away_wins) / h2h_total
            features['h2h_draw_tendency'] = h2h_draws / h2h_total
            features['h2h_competitiveness'] = min(h2h_home_wins, h2h_away_wins) / h2h_total
        else:
            features['h2h_home_dominance'] = 0.0
            features['h2h_draw_tendency'] = 0.33
            features['h2h_competitiveness'] = 0.33
        
        # Recent H2H weight (more weight to recent matches)
        features['h2h_recency_weighted'] = match_data.get('h2h_time_weighted', 0.5)
        
        return features
    
    def _calculate_form_trends(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weighted recent form trends"""
        features = {}
        
        # Exponentially weighted form (more weight to recent)
        home_form_points = match_data.get('home_form_points_last_5', 0)
        away_form_points = match_data.get('away_form_points_last_5', 0)
        
        features['form_differential'] = home_form_points - away_form_points
        features['combined_form_quality'] = (home_form_points + away_form_points) / 30.0  # Normalize
        
        # Form volatility (consistency indicator)
        home_form_std = match_data.get('home_form_std', 1.0)
        away_form_std = match_data.get('away_form_std', 1.0)
        features['home_form_consistency'] = 1.0 / (1.0 + home_form_std)
        features['away_form_consistency'] = 1.0 / (1.0 + away_form_std)
        
        return features
    
    def _calculate_cross_league_strength(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cross-league strength ratings"""
        features = {}
        
        # ELO-based cross-league strength
        home_elo = match_data.get('home_elo', 1500)
        away_elo = match_data.get('away_elo', 1500)
        
        # Strength tiers (relative to league average 1500)
        features['home_strength_tier'] = (home_elo - 1500) / 100.0
        features['away_strength_tier'] = (away_elo - 1500) / 100.0
        
        # Quality of opposition faced recently
        home_opp_elo = match_data.get('home_recent_opp_elo', 1500)
        away_opp_elo = match_data.get('away_recent_opp_elo', 1500)
        features['home_schedule_strength'] = (home_opp_elo - 1500) / 100.0
        features['away_schedule_strength'] = (away_opp_elo - 1500) / 100.0
        
        return features
    
    def _calculate_scoring_patterns(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate goal-scoring patterns and consistency"""
        features = {}
        
        # Goals scored patterns
        home_goals_scored = match_data.get('home_avg_goals_scored_last_5', 1.5)
        away_goals_scored = match_data.get('away_avg_goals_scored_last_5', 1.5)
        
        features['home_attack_strength'] = home_goals_scored
        features['away_attack_strength'] = away_goals_scored
        features['attack_differential'] = home_goals_scored - away_goals_scored
        
        # xG (expected goals) if available
        home_xg = match_data.get('home_match_xg', home_goals_scored)
        away_xg = match_data.get('away_match_xg', away_goals_scored)
        
        # Clinical finishing (actual goals vs xG)
        features['home_clinical_finishing'] = home_goals_scored - home_xg if home_xg > 0 else 0.0
        features['away_clinical_finishing'] = away_goals_scored - away_xg if away_xg > 0 else 0.0
        
        return features
    
    def _calculate_defensive_metrics(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate defensive solidity metrics"""
        features = {}
        
        # Goals conceded patterns
        home_conceded = match_data.get('home_avg_goals_conceded_last_5', 1.5)
        away_conceded = match_data.get('away_avg_goals_conceded_last_5', 1.5)
        
        features['home_defensive_strength'] = 3.0 - home_conceded  # Inverse
        features['away_defensive_strength'] = 3.0 - away_conceded
        features['defensive_differential'] = away_conceded - home_conceded  # Higher = home advantage
        
        # Clean sheet probability (simplified)
        features['home_clean_sheet_prob'] = 1.0 / (1.0 + away_conceded)
        features['away_clean_sheet_prob'] = 1.0 / (1.0 + home_conceded)
        
        return features
    
    def _calculate_late_game_performance(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate late-game (75+ minutes) performance"""
        features = {}
        
        # Late goals scored/conceded (if available)
        home_late_goals = match_data.get('home_late_goals_last_5', 0.0)
        away_late_goals = match_data.get('away_late_goals_last_5', 0.0)
        
        features['home_late_game_strength'] = home_late_goals
        features['away_late_game_strength'] = away_late_goals
        features['late_game_differential'] = home_late_goals - away_late_goals
        
        return features
    
    def _calculate_rest_advantage(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Enhanced rest and fatigue calculations"""
        features = {}
        
        home_rest = match_data.get('home_rest_days', 3.0)
        away_rest = match_data.get('away_rest_days', 3.0)
        
        # Rest advantage (non-linear)
        features['rest_advantage'] = self._rest_advantage_curve(home_rest) - self._rest_advantage_curve(away_rest)
        
        # Fatigue indicators
        features['home_fatigue_factor'] = 1.0 if home_rest < 3 else 0.0
        features['away_fatigue_factor'] = 1.0 if away_rest < 3 else 0.0
        
        # Recovery advantage
        features['home_recovery_advantage'] = max(0, home_rest - 3) / 7.0
        features['away_recovery_advantage'] = max(0, away_rest - 3) / 7.0
        
        return features
    
    def _calculate_tactical_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate tactical matchup features"""
        features = {}
        
        # Formation compatibility
        formation_clash = match_data.get('formation_clash_score', 0.5)
        features['tactical_advantage'] = formation_clash - 0.5  # Center at 0
        
        # Injury impact (existing feature, extract for clarity)
        home_injury = match_data.get('home_injury_impact', 0.0)
        away_injury = match_data.get('away_injury_impact', 0.0)
        features['injury_differential'] = away_injury - home_injury  # Positive = home advantage
        
        # Substitution patterns
        home_subs = match_data.get('substitutions_home', 3.0)
        away_subs = match_data.get('substitutions_away', 3.0)
        features['substitution_activity'] = (home_subs + away_subs) / 6.0
        
        return features
    
    def _calculate_streak(self, form: List[float]) -> float:
        """Calculate win/loss streak from form data (3=win, 1=draw, 0=loss)"""
        if not form:
            return 0.0
        
        streak = 0
        for result in form:
            if result >= 3:  # Win
                streak = streak + 1 if streak >= 0 else 1
            elif result == 0:  # Loss
                streak = streak - 1 if streak <= 0 else -1
            else:  # Draw breaks streak
                break
        
        return float(streak)
    
    def _rest_advantage_curve(self, rest_days: float) -> float:
        """Non-linear rest advantage curve (diminishing returns after 5 days)"""
        if rest_days < 2:
            return -0.2  # Fatigued
        elif rest_days < 3:
            return 0.0  # Suboptimal
        elif rest_days <= 5:
            return 0.2  # Optimal
        else:
            return 0.1  # Too much rest (rust)
    
    def _track_feature_metrics(self, features: Dict[str, float], computation_time: float):
        """Track feature quality metrics"""
        values = list(features.values())
        
        # Calculate metrics
        missing_ratio = sum(1 for v in values if pd.isna(v) or v == 0.0) / len(values)
        
        # Convert to DataFrame for correlation
        df = pd.DataFrame([features])
        corr_matrix = df.corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)  # Ignore self-correlation
        max_corr = corr_matrix.max().max() if not corr_matrix.empty else 0.0
        
        # Variance
        min_variance = df.var().min() if not df.empty else 0.0
        
        metrics = FeatureMetrics(
            feature_count=len(features),
            computation_time=computation_time,
            missing_value_ratio=missing_ratio,
            correlation_max=max_corr,
            variance_min=min_variance,
            timestamp=datetime.now()
        )
        
        self.feature_metrics.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.feature_metrics) > 100:
            self.feature_metrics = self.feature_metrics[-100:]
    
    def _generate_fallback_features(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate minimal fallback features on error"""
        return {
            'home_elo': match_data.get('home_elo', 1500),
            'away_elo': match_data.get('away_elo', 1500),
            'elo_diff': match_data.get('elo_diff', 0),
            'home_form_points_last_5': match_data.get('home_form_points_last_5', 7.5),
            'away_form_points_last_5': match_data.get('away_form_points_last_5', 7.5),
        }
    
    def get_feature_importance_analysis(self) -> Dict[str, Any]:
        """Get feature importance analysis for reporting"""
        if not self.feature_metrics:
            return {}
        
        latest = self.feature_metrics[-1]
        avg_computation_time = np.mean([m.computation_time for m in self.feature_metrics])
        
        return {
            'feature_count': latest.feature_count,
            'baseline_count': len(self.baseline_features),
            'new_features': latest.feature_count - len(self.baseline_features),
            'avg_computation_time': avg_computation_time,
            'missing_value_ratio': latest.missing_value_ratio,
            'max_correlation': latest.correlation_max,
            'min_variance': latest.variance_min,
        }
    
    def save_feature_metrics(self, filepath: str):
        """Save feature metrics to file"""
        try:
            metrics_data = [
                {
                    'feature_count': m.feature_count,
                    'computation_time': m.computation_time,
                    'missing_value_ratio': m.missing_value_ratio,
                    'correlation_max': m.correlation_max,
                    'variance_min': m.variance_min,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in self.feature_metrics
            ]
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"✅ Saved feature metrics to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save feature metrics: {e}")


# Global instance
advanced_feature_system = AdvancedFeatureEngineeringSystem()
advanced_feature_system = AdvancedFeatureEngineeringSystem()
