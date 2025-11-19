"""
Enhanced Feature Engineering Optimizer for Football Betting Platform
Provides advanced feature selection, correlation analysis, and performance optimization.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, SelectKBest, f_classif,
                                       mutual_info_classif)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class FeatureOptimizer:
    """Advanced feature optimization and selection for football betting models."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the feature optimizer."""
        self.config = config or {}
        self.feature_importance_cache = {}
        self.correlation_cache = {}
        self.selection_results = {}
        
    def analyze_feature_correlations(self, features_df: pd.DataFrame, target: pd.Series = None) -> Dict[str, Any]:
        """Analyze feature correlations and identify redundant features."""
        logger.info("Analyzing feature correlations...")
        start_time = time.time()
        
        # Calculate correlation matrix
        correlation_matrix = features_df.corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        correlation_threshold = self.config.get('correlation_threshold', 0.8)
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Identify redundant features to remove
        redundant_features = set()
        for pair in high_corr_pairs:
            # Keep the feature with higher variance (more information)
            var1 = features_df[pair['feature1']].var()
            var2 = features_df[pair['feature2']].var()
            
            if var1 < var2:
                redundant_features.add(pair['feature1'])
            else:
                redundant_features.add(pair['feature2'])
        
        # Calculate feature-target correlations if target is provided
        target_correlations = {}
        if target is not None:
            # Convert target to pandas Series if it's a numpy array
            if isinstance(target, np.ndarray):
                target_series = pd.Series(target, index=features_df.index)
            else:
                target_series = target

            for col in features_df.columns:
                if features_df[col].dtype in ['int64', 'float64']:
                    try:
                        corr = features_df[col].corr(target_series)
                        if not np.isnan(corr):
                            target_correlations[col] = abs(corr)
                    except Exception as e:
                        logger.warning(f"Could not calculate correlation for {col}: {e}")
        
        analysis_time = time.time() - start_time
        logger.info(f"Correlation analysis completed in {analysis_time:.2f}s")
        logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs")
        logger.info(f"Identified {len(redundant_features)} redundant features")
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_corr_pairs': high_corr_pairs,
            'redundant_features': list(redundant_features),
            'target_correlations': target_correlations,
            'analysis_time': analysis_time
        }
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info') -> Dict[str, Any]:
        """Select the best features using various selection methods."""
        logger.info(f"Selecting best features using {method} method...")
        start_time = time.time()
        
        # Determine number of features to select
        n_features = min(self.config.get('max_features', 20), len(X.columns))
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        elif method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Fit selector
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature scores if available
        feature_scores = {}
        if hasattr(selector, 'scores_'):
            feature_scores = dict(zip(X.columns, selector.scores_))
        elif hasattr(selector, 'ranking_'):
            # For RFE, lower ranking is better
            feature_scores = dict(zip(X.columns, 1.0 / selector.ranking_))
        
        selection_time = time.time() - start_time
        logger.info(f"Feature selection completed in {selection_time:.2f}s")
        logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
        
        return {
            'selected_features': selected_features,
            'feature_scores': feature_scores,
            'selector': selector,
            'X_selected': pd.DataFrame(X_selected, columns=selected_features, index=X.index),
            'selection_time': selection_time
        }
    
    def optimize_feature_groups(self, X: pd.DataFrame, y: pd.Series, feature_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Optimize features within each group and across groups."""
        logger.info("Optimizing feature groups...")
        start_time = time.time()
        
        group_performance = {}
        optimized_groups = {}
        
        # Baseline performance with all features
        baseline_score = cross_val_score(
            RandomForestClassifier(n_estimators=50, random_state=42),
            X, y, cv=3, scoring='accuracy'
        ).mean()
        
        logger.info(f"Baseline accuracy with all features: {baseline_score:.4f}")
        
        # Evaluate each group individually
        for group_name, group_features in feature_groups.items():
            available_features = [f for f in group_features if f in X.columns]
            
            if not available_features:
                logger.warning(f"No features available for group {group_name}")
                continue
                
            X_group = X[available_features]
            
            # Test group performance
            group_score = cross_val_score(
                RandomForestClassifier(n_estimators=50, random_state=42),
                X_group, y, cv=3, scoring='accuracy'
            ).mean()
            
            # Select best features within group
            if len(available_features) > 5:  # Only optimize if group has many features
                selection_result = self.select_best_features(
                    X_group, y, method='mutual_info'
                )
                optimized_features = selection_result['selected_features']
                
                # Test optimized group performance
                optimized_score = cross_val_score(
                    RandomForestClassifier(n_estimators=50, random_state=42),
                    X[optimized_features], y, cv=3, scoring='accuracy'
                ).mean()
            else:
                optimized_features = available_features
                optimized_score = group_score
            
            group_performance[group_name] = {
                'original_features': available_features,
                'optimized_features': optimized_features,
                'original_score': group_score,
                'optimized_score': optimized_score,
                'improvement': optimized_score - group_score,
                'feature_reduction': len(available_features) - len(optimized_features)
            }
            
            optimized_groups[group_name] = optimized_features
            
            logger.info(f"Group {group_name}: {group_score:.4f} → {optimized_score:.4f} "
                       f"({len(available_features)} → {len(optimized_features)} features)")
        
        # Test combined optimized features
        all_optimized_features = []
        for features in optimized_groups.values():
            all_optimized_features.extend(features)
        
        if all_optimized_features:
            combined_score = cross_val_score(
                RandomForestClassifier(n_estimators=50, random_state=42),
                X[all_optimized_features], y, cv=3, scoring='accuracy'
            ).mean()
        else:
            combined_score = 0.0
        
        optimization_time = time.time() - start_time
        logger.info(f"Feature group optimization completed in {optimization_time:.2f}s")
        logger.info(f"Combined optimized score: {combined_score:.4f}")
        
        return {
            'baseline_score': baseline_score,
            'combined_score': combined_score,
            'group_performance': group_performance,
            'optimized_groups': optimized_groups,
            'all_optimized_features': all_optimized_features,
            'optimization_time': optimization_time,
            'total_feature_reduction': len(X.columns) - len(all_optimized_features)
        }
    
    def create_engineered_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create new engineered features from existing ones."""
        logger.info("Creating engineered features...")
        start_time = time.time()
        
        X_engineered = X.copy()
        
        # Feature interactions for football-specific insights
        try:
            # Goal-related interactions
            if 'home_avg_goals_scored_last_5' in X.columns and 'away_avg_goals_conceded_last_5' in X.columns:
                X_engineered['home_attack_vs_away_defense'] = (
                    X['home_avg_goals_scored_last_5'] - X['away_avg_goals_conceded_last_5']
                )
            
            if 'away_avg_goals_scored_last_5' in X.columns and 'home_avg_goals_conceded_last_5' in X.columns:
                X_engineered['away_attack_vs_home_defense'] = (
                    X['away_avg_goals_scored_last_5'] - X['home_avg_goals_conceded_last_5']
                )
            
            # ELO-based features
            if 'home_elo' in X.columns and 'away_elo' in X.columns:
                X_engineered['elo_ratio'] = X['home_elo'] / (X['away_elo'] + 1e-6)  # Avoid division by zero
                X_engineered['elo_sum'] = X['home_elo'] + X['away_elo']
                X_engineered['elo_product'] = X['home_elo'] * X['away_elo']
            
            # Form-based features
            if 'home_form_points_last_5' in X.columns and 'away_form_points_last_5' in X.columns:
                X_engineered['form_difference'] = X['home_form_points_last_5'] - X['away_form_points_last_5']
                X_engineered['form_ratio'] = X['home_form_points_last_5'] / (X['away_form_points_last_5'] + 1e-6)
            
            # H2H features
            if 'h2h_team1_wins' in X.columns and 'h2h_team2_wins' in X.columns:
                total_h2h = X['h2h_team1_wins'] + X['h2h_team2_wins'] + X.get('h2h_draws', 0)
                X_engineered['h2h_home_win_rate'] = X['h2h_team1_wins'] / (total_h2h + 1e-6)
                X_engineered['h2h_dominance'] = (X['h2h_team1_wins'] - X['h2h_team2_wins']) / (total_h2h + 1e-6)
            
            # Weather impact features
            if 'weather_temp' in X.columns and 'weather_precip' in X.columns:
                X_engineered['weather_impact'] = (
                    (X['weather_temp'] - 15) ** 2 + X['weather_precip'] * 2
                )  # Penalty for extreme temperatures and precipitation
            
            # Injury impact
            if 'home_injury_impact' in X.columns and 'away_injury_impact' in X.columns:
                X_engineered['injury_advantage'] = X['away_injury_impact'] - X['home_injury_impact']
            
        except Exception as e:
            logger.warning(f"Error creating some engineered features: {e}")
        
        engineering_time = time.time() - start_time
        new_features = len(X_engineered.columns) - len(X.columns)
        logger.info(f"Created {new_features} new engineered features in {engineering_time:.2f}s")
        
        return X_engineered
    
    def optimize_feature_pipeline(self, X: pd.DataFrame, y: pd.Series, feature_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Complete feature optimization pipeline."""
        logger.info("Starting complete feature optimization pipeline...")
        pipeline_start = time.time()
        
        results = {}
        
        # Step 1: Correlation analysis
        logger.info("Step 1: Analyzing correlations...")
        correlation_results = self.analyze_feature_correlations(X, y)
        results['correlation_analysis'] = correlation_results
        
        # Step 2: Remove redundant features
        logger.info("Step 2: Removing redundant features...")
        X_reduced = X.drop(columns=correlation_results['redundant_features'], errors='ignore')
        logger.info(f"Removed {len(correlation_results['redundant_features'])} redundant features")
        
        # Step 3: Create engineered features
        logger.info("Step 3: Creating engineered features...")
        X_engineered = self.create_engineered_features(X_reduced)
        results['engineered_features'] = list(set(X_engineered.columns) - set(X_reduced.columns))
        
        # Step 4: Optimize feature groups
        logger.info("Step 4: Optimizing feature groups...")
        group_optimization = self.optimize_feature_groups(X_engineered, y, feature_groups)
        results['group_optimization'] = group_optimization
        
        # Step 5: Final feature selection
        logger.info("Step 5: Final feature selection...")
        final_selection = self.select_best_features(
            X_engineered[group_optimization['all_optimized_features']], 
            y, 
            method='mutual_info'
        )
        results['final_selection'] = final_selection
        
        # Performance summary
        pipeline_time = time.time() - pipeline_start
        results['pipeline_summary'] = {
            'original_features': len(X.columns),
            'after_correlation_removal': len(X_reduced.columns),
            'after_engineering': len(X_engineered.columns),
            'final_features': len(final_selection['selected_features']),
            'total_reduction': len(X.columns) - len(final_selection['selected_features']),
            'pipeline_time': pipeline_time,
            'baseline_score': group_optimization['baseline_score'],
            'final_score': group_optimization['combined_score']
        }
        
        logger.info("Feature optimization pipeline completed!")
        logger.info(f"Features: {len(X.columns)} → {len(final_selection['selected_features'])} "
                   f"(reduction: {results['pipeline_summary']['total_reduction']})")
        logger.info(f"Performance: {group_optimization['baseline_score']:.4f} → "
                   f"{group_optimization['combined_score']:.4f}")
        logger.info(f"Total time: {pipeline_time:.2f}s")
        
        return results
