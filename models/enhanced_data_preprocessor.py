#!/usr/bin/env python3
"""
Enhanced Data Preprocessing Pipeline
Outlier detection, missing value imputation, feature scaling, temporal validation
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class EnhancedDataPreprocessor:
    """
    Production-grade data preprocessing with advanced techniques
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scalers = {}
        self.imputers = {}
        self.outlier_bounds = {}
        self.preprocessing_stats = {}
        logger.info("🔧 Enhanced Data Preprocessor initialized")
    
    def preprocess_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit: bool = True,
        detect_outliers: bool = True,
        impute_missing: bool = True,
        scale_features: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete data preprocessing pipeline
        
        Parameters:
            X: Input features
            y: Target variable (optional)
            fit: Whether to fit preprocessors or use existing
            detect_outliers: Whether to detect and handle outliers
            impute_missing: Whether to impute missing values
            scale_features: Whether to scale features
        
        Returns:
            Tuple of (preprocessed_data, preprocessing_stats)
        """
        logger.info("🔄 Starting data preprocessing pipeline")
        start_time = datetime.now()
        
        X_processed = X.copy()
        stats = {
            'original_shape': X.shape,
            'original_missing': X.isnull().sum().sum(),
            'steps_applied': []
        }
        
        # Step 1: Detect and handle outliers
        if detect_outliers:
            X_processed, outlier_stats = self._handle_outliers(X_processed, fit=fit)
            stats['outlier_detection'] = outlier_stats
            stats['steps_applied'].append('outlier_detection')
            logger.info(f"✅ Outlier detection: {outlier_stats['n_outliers']} outliers handled")
        
        # Step 2: Impute missing values
        if impute_missing:
            X_processed, imputation_stats = self._impute_missing_values(X_processed, fit=fit)
            stats['imputation'] = imputation_stats
            stats['steps_applied'].append('imputation')
            logger.info(f"✅ Imputation: {imputation_stats['values_imputed']} values imputed")
        
        # Step 3: Scale features
        if scale_features:
            X_processed, scaling_stats = self._scale_features(X_processed, fit=fit)
            stats['scaling'] = scaling_stats
            stats['steps_applied'].append('scaling')
            logger.info(f"✅ Scaling: {scaling_stats['scaler_type']} applied")
        
        # Step 4: Feature validation
        X_processed, validation_stats = self._validate_features(X_processed)
        stats['validation'] = validation_stats
        stats['steps_applied'].append('validation')
        
        stats['final_shape'] = X_processed.shape
        stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        self.preprocessing_stats = stats
        logger.info(f"✅ Preprocessing complete in {stats['processing_time']:.2f}s")
        
        return X_processed, stats
    
    def _handle_outliers(
        self,
        X: pd.DataFrame,
        fit: bool = True,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and handle outliers
        
        Methods:
        - iqr: Interquartile range (Q1 - threshold*IQR, Q3 + threshold*IQR)
        - zscore: Z-score (mean ± threshold*std)
        - isolation_forest: Isolation Forest algorithm
        """
        X_processed = X.copy()
        n_outliers = 0
        outlier_features = []
        
        if method == 'iqr':
            for col in X.select_dtypes(include=[np.number]).columns:
                if fit:
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    self.outlier_bounds[col] = (lower_bound, upper_bound)
                else:
                    lower_bound, upper_bound = self.outlier_bounds.get(col, (-np.inf, np.inf))
                
                # Clip outliers
                outliers_mask = (X_processed[col] < lower_bound) | (X_processed[col] > upper_bound)
                n_col_outliers = outliers_mask.sum()
                
                if n_col_outliers > 0:
                    X_processed[col] = X_processed[col].clip(lower_bound, upper_bound)
                    n_outliers += n_col_outliers
                    outlier_features.append(col)
        
        elif method == 'zscore':
            for col in X.select_dtypes(include=[np.number]).columns:
                if fit:
                    mean = X[col].mean()
                    std = X[col].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    self.outlier_bounds[col] = (lower_bound, upper_bound)
                else:
                    lower_bound, upper_bound = self.outlier_bounds.get(col, (-np.inf, np.inf))
                
                outliers_mask = (X_processed[col] < lower_bound) | (X_processed[col] > upper_bound)
                n_col_outliers = outliers_mask.sum()
                
                if n_col_outliers > 0:
                    X_processed[col] = X_processed[col].clip(lower_bound, upper_bound)
                    n_outliers += n_col_outliers
                    outlier_features.append(col)
        
        return X_processed, {
            'method': method,
            'threshold': threshold,
            'n_outliers': n_outliers,
            'outlier_features': outlier_features,
            'n_features_with_outliers': len(outlier_features)
        }
    
    def _impute_missing_values(
        self,
        X: pd.DataFrame,
        fit: bool = True,
        strategy: str = 'auto'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing values with intelligent strategy selection
        
        Strategies:
        - auto: Choose based on missing percentage and feature type
        - mean/median/mode: Simple imputation
        - knn: K-Nearest Neighbors imputation
        - forward_fill: Forward fill (for time series)
        """
        X_processed = X.copy()
        values_imputed = 0
        imputation_methods = {}
        
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            if missing_count == 0:
                continue
            
            missing_pct = missing_count / len(X)
            
            # Select imputation strategy
            if strategy == 'auto':
                if missing_pct > 0.5:
                    # Too many missing - drop or use constant
                    selected_strategy = 'constant'
                elif missing_pct > 0.1:
                    # Moderate missing - use KNN if numeric
                    selected_strategy = 'knn' if X[col].dtype in [np.float64, np.int64] else 'mode'
                else:
                    # Few missing - use mean/mode
                    selected_strategy = 'mean' if X[col].dtype in [np.float64, np.int64] else 'mode'
            else:
                selected_strategy = strategy
            
            # Apply imputation
            if selected_strategy in ['mean', 'median', 'mode', 'constant']:
                if fit:
                    if selected_strategy == 'mean':
                        imputer = SimpleImputer(strategy='mean')
                    elif selected_strategy == 'median':
                        imputer = SimpleImputer(strategy='median')
                    elif selected_strategy == 'mode':
                        imputer = SimpleImputer(strategy='most_frequent')
                    else:  # constant
                        imputer = SimpleImputer(strategy='constant', fill_value=0)
                    
                    X_processed[[col]] = imputer.fit_transform(X_processed[[col]])
                    self.imputers[col] = imputer
                else:
                    imputer = self.imputers.get(col)
                    if imputer:
                        X_processed[[col]] = imputer.transform(X_processed[[col]])
            
            elif selected_strategy == 'knn':
                if fit:
                    imputer = KNNImputer(n_neighbors=5)
                    X_processed[[col]] = imputer.fit_transform(X_processed[[col]])
                    self.imputers[col] = imputer
                else:
                    imputer = self.imputers.get(col)
                    if imputer:
                        X_processed[[col]] = imputer.transform(X_processed[[col]])
            
            values_imputed += missing_count
            imputation_methods[col] = selected_strategy
        
        return X_processed, {
            'strategy': strategy,
            'values_imputed': values_imputed,
            'imputation_methods': imputation_methods,
            'columns_imputed': list(imputation_methods.keys())
        }
    
    def _scale_features(
        self,
        X: pd.DataFrame,
        fit: bool = True,
        scaler_type: str = 'robust'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scale features using selected scaler
        
        Scalers:
        - standard: StandardScaler (mean=0, std=1)
        - robust: RobustScaler (median=0, IQR=1, robust to outliers)
        - minmax: MinMaxScaler (min=0, max=1)
        """
        X_processed = X.copy()
        
        # Select scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}, using standard")
            scaler = StandardScaler()
        
        # Apply scaling to numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if fit:
            X_processed[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            self.scalers['main'] = scaler
        else:
            scaler = self.scalers.get('main')
            if scaler:
                X_processed[numeric_cols] = scaler.transform(X[numeric_cols])
        
        return X_processed, {
            'scaler_type': scaler_type,
            'n_features_scaled': len(numeric_cols),
            'scaled_features': list(numeric_cols)
        }
    
    def _validate_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate features for quality issues"""
        validation_issues = []
        
        # Check for infinite values
        inf_cols = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if np.isinf(X[col]).any():
                inf_cols.append(col)
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                # Fill NaN with column mean
                X[col] = X[col].fillna(X[col].mean())
        
        if inf_cols:
            validation_issues.append(f"Infinite values in {len(inf_cols)} columns")
        
        # Check for constant features
        constant_cols = []
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].std() == 0:
                constant_cols.append(col)
        
        if constant_cols:
            validation_issues.append(f"Constant values in {len(constant_cols)} columns")
        
        # Check for high correlation (multicollinearity)
        corr_matrix = X.select_dtypes(include=[np.number]).corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        if high_corr_pairs:
            validation_issues.append(f"High correlation in {len(high_corr_pairs)} feature pairs")
        
        return X, {
            'validation_issues': validation_issues,
            'infinite_values_fixed': inf_cols,
            'constant_features': constant_cols,
            'highly_correlated_pairs': high_corr_pairs
        }
    
    def create_temporal_splits(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create temporal validation splits for time series data
        Uses TimeSeriesSplit to maintain temporal order
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(tscv.split(X))
        
        logger.info(f"✅ Created {n_splits} temporal validation splits")
        return splits
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing operations"""
        if not self.preprocessing_stats:
            return {'status': 'no_preprocessing_performed'}
        
        return {
            'original_shape': self.preprocessing_stats.get('original_shape'),
            'final_shape': self.preprocessing_stats.get('final_shape'),
            'original_missing': self.preprocessing_stats.get('original_missing'),
            'outliers_handled': self.preprocessing_stats.get('outlier_detection', {}).get('n_outliers', 0),
            'values_imputed': self.preprocessing_stats.get('imputation', {}).get('values_imputed', 0),
            'processing_time': self.preprocessing_stats.get('processing_time'),
            'steps_applied': self.preprocessing_stats.get('steps_applied', [])
        }
    
    def save_preprocessors(self, filepath: str):
        """Save fitted preprocessors"""
        try:
            import joblib
            
            preprocessors = {
                'scalers': self.scalers,
                'imputers': self.imputers,
                'outlier_bounds': self.outlier_bounds,
                'config': self.config
            }
            
            joblib.dump(preprocessors, filepath)
            logger.info(f"✅ Saved preprocessors to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save preprocessors: {e}")
    
    def load_preprocessors(self, filepath: str) -> bool:
        """Load previously saved preprocessors"""
        try:
            import joblib
            
            preprocessors = joblib.load(filepath)
            self.scalers = preprocessors.get('scalers', {})
            self.imputers = preprocessors.get('imputers', {})
            self.outlier_bounds = preprocessors.get('outlier_bounds', {})
            self.config = preprocessors.get('config', {})
            
            logger.info(f"✅ Loaded preprocessors from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load preprocessors: {e}")
            return False


# Global instance
enhanced_data_preprocessor = EnhancedDataPreprocessor()
enhanced_data_preprocessor = EnhancedDataPreprocessor()
