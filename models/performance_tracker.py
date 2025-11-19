#!/usr/bin/env python3
"""
Model Performance Tracking System for GoalDiggers Platform

Comprehensive tracking and monitoring of model performance with
real-time metrics, drift detection, and automated alerts.
"""

import json
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Data class for performance metrics."""
    model_id: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    dataset_type: str  # 'train', 'validation', 'test', 'production'
    data_size: int
    model_version: str = "v1.0"
    additional_info: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}

class ModelPerformanceTracker:
    """
    Comprehensive model performance tracking and monitoring system.
    """
    
    def __init__(self, db_path: str = "data/model_performance.db"):
        """
        Initialize the performance tracking system.
        
        Args:
            db_path: Path to SQLite database for storing metrics
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.alert_thresholds = {
            'accuracy_drop': 0.05,  # Alert if accuracy drops by 5%
            'drift_threshold': 0.1,  # Data drift threshold
            'performance_degradation': 0.1  # Performance degradation threshold
        }
        
        self._init_database()
        
        logger.info(f"Model performance tracker initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    dataset_type TEXT NOT NULL,
                    data_size INTEGER,
                    model_version TEXT,
                    additional_info TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT PRIMARY KEY,
                    model_type TEXT,
                    created_at TEXT,
                    last_updated TEXT,
                    status TEXT,
                    description TEXT
                )
            """)
            
            # Create indexes for better performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_timestamp 
                ON performance_metrics(model_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_name 
                ON performance_metrics(metric_name)
            """)
    
    def register_model(self, 
                      model_id: str,
                      model_type: str,
                      description: str = "") -> bool:
        """
        Register a new model for tracking.
        
        Args:
            model_id: Unique model identifier
            model_type: Type of model (e.g., 'xgboost', 'neural_network')
            description: Model description
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_metadata 
                    (model_id, model_type, created_at, last_updated, status, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    model_id,
                    model_type,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    'active',
                    description
                ))
            
            logger.info(f"Registered model: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
    
    def log_performance(self, metric: PerformanceMetric) -> bool:
        """
        Log a performance metric.
        
        Args:
            metric: Performance metric to log
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_metrics 
                    (model_id, timestamp, metric_name, metric_value, dataset_type, 
                     data_size, model_version, additional_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.model_id,
                    metric.timestamp.isoformat(),
                    metric.metric_name,
                    metric.metric_value,
                    metric.dataset_type,
                    metric.data_size,
                    metric.model_version,
                    json.dumps(metric.additional_info)
                ))
                
                # Update model metadata
                conn.execute("""
                    UPDATE model_metadata 
                    SET last_updated = ?, status = 'active'
                    WHERE model_id = ?
                """, (datetime.now().isoformat(), metric.model_id))
            
            logger.debug(f"Logged performance metric: {metric.model_id} - {metric.metric_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log performance metric: {e}")
            return False
    
    def log_batch_performance(self, 
                            model_id: str,
                            metrics: Dict[str, float],
                            dataset_type: str = "production",
                            data_size: int = None,
                            model_version: str = "v1.0") -> bool:
        """
        Log multiple performance metrics at once.
        
        Args:
            model_id: Model identifier
            metrics: Dictionary of metric names and values
            dataset_type: Type of dataset
            data_size: Size of dataset
            model_version: Model version
            
        Returns:
            True if successful
        """
        timestamp = datetime.now()
        success_count = 0
        
        for metric_name, metric_value in metrics.items():
            metric = PerformanceMetric(
                model_id=model_id,
                timestamp=timestamp,
                metric_name=metric_name,
                metric_value=metric_value,
                dataset_type=dataset_type,
                data_size=data_size or 0,
                model_version=model_version
            )
            
            if self.log_performance(metric):
                success_count += 1
        
        logger.info(f"Logged {success_count}/{len(metrics)} metrics for {model_id}")
        return success_count == len(metrics)
    
    def get_performance_history(self, 
                              model_id: str,
                              metric_name: str = None,
                              start_date: datetime = None,
                              end_date: datetime = None,
                              dataset_type: str = None) -> pd.DataFrame:
        """
        Get performance history for a model.
        
        Args:
            model_id: Model identifier
            metric_name: Specific metric name (optional)
            start_date: Start date for filtering
            end_date: End date for filtering
            dataset_type: Dataset type filter
            
        Returns:
            DataFrame with performance history
        """
        query = """
            SELECT * FROM performance_metrics 
            WHERE model_id = ?
        """
        params = [model_id]
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if dataset_type:
            query += " AND dataset_type = ?"
            params.append(dataset_type)
        
        query += " ORDER BY timestamp DESC"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['additional_info'] = df['additional_info'].apply(
                        lambda x: json.loads(x) if x else {}
                    )
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to get performance history: {e}")
            return pd.DataFrame()
    
    def detect_performance_drift(self, 
                               model_id: str,
                               metric_name: str = "accuracy",
                               window_size: int = 10,
                               threshold: float = None) -> Dict[str, Any]:
        """
        Detect performance drift for a model.
        
        Args:
            model_id: Model identifier
            metric_name: Metric to analyze for drift
            window_size: Size of rolling window for comparison
            threshold: Drift threshold (uses default if None)
            
        Returns:
            Dictionary with drift analysis results
        """
        if threshold is None:
            threshold = self.alert_thresholds['drift_threshold']
        
        # Get recent performance data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        df = self.get_performance_history(
            model_id=model_id,
            metric_name=metric_name,
            start_date=start_date,
            end_date=end_date,
            dataset_type="production"
        )
        
        if len(df) < window_size * 2:
            return {
                'drift_detected': False,
                'reason': 'Insufficient data for drift detection',
                'data_points': len(df)
            }
        
        # Calculate rolling statistics
        df = df.sort_values('timestamp')
        df['rolling_mean'] = df['metric_value'].rolling(window=window_size).mean()
        df['rolling_std'] = df['metric_value'].rolling(window=window_size).std()
        
        # Compare recent performance to baseline
        baseline_mean = df['rolling_mean'].iloc[:window_size].mean()
        recent_mean = df['rolling_mean'].iloc[-window_size:].mean()
        
        drift_magnitude = abs(recent_mean - baseline_mean) / baseline_mean
        drift_detected = drift_magnitude > threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_magnitude': drift_magnitude,
            'threshold': threshold,
            'baseline_mean': baseline_mean,
            'recent_mean': recent_mean,
            'data_points': len(df),
            'analysis_period': f"{start_date.date()} to {end_date.date()}"
        }
    
    def get_model_summary(self, model_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with model summary
        """
        try:
            # Get model metadata
            with sqlite3.connect(self.db_path) as conn:
                metadata_query = "SELECT * FROM model_metadata WHERE model_id = ?"
                metadata_df = pd.read_sql_query(metadata_query, conn, params=[model_id])
                
                if metadata_df.empty:
                    return {'error': f'Model {model_id} not found'}
                
                metadata = metadata_df.iloc[0].to_dict()
            
            # Get recent performance metrics
            recent_df = self.get_performance_history(
                model_id=model_id,
                start_date=datetime.now() - timedelta(days=7)
            )
            
            # Calculate summary statistics
            summary = {
                'model_info': metadata,
                'recent_performance': {},
                'total_predictions': 0,
                'last_updated': metadata.get('last_updated'),
                'status': metadata.get('status', 'unknown')
            }
            
            if not recent_df.empty:
                # Group by metric name and calculate latest values
                latest_metrics = recent_df.groupby('metric_name').agg({
                    'metric_value': 'last',
                    'timestamp': 'max',
                    'data_size': 'sum'
                }).to_dict('index')
                
                summary['recent_performance'] = latest_metrics
                summary['total_predictions'] = recent_df['data_size'].sum()
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get model summary: {e}")
            return {'error': str(e)}
    
    def render_performance_dashboard(self, model_id: str = None):
        """
        Render comprehensive performance dashboard.
        
        Args:
            model_id: Specific model to show (shows all if None)
        """
        st.markdown("## 📊 Model Performance Dashboard")
        
        # Model selector
        available_models = self._get_available_models()
        
        if not available_models:
            st.warning("No models found. Register models to start tracking performance.")
            return
        
        if model_id is None:
            model_id = st.selectbox("Select Model", available_models)
        
        if not model_id:
            return
        
        # Get model summary
        summary = self.get_model_summary(model_id)
        
        if 'error' in summary:
            st.error(summary['error'])
            return
        
        # Model info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", summary['model_info'].get('model_type', 'Unknown'))
        
        with col2:
            st.metric("Status", summary['status'].title())
        
        with col3:
            total_predictions = summary.get('total_predictions', 0)
            st.metric("Recent Predictions", f"{total_predictions:,}")
        
        with col4:
            last_updated = summary.get('last_updated')
            if last_updated:
                last_updated_dt = datetime.fromisoformat(last_updated)
                hours_ago = (datetime.now() - last_updated_dt).total_seconds() / 3600
                st.metric("Last Updated", f"{hours_ago:.1f}h ago")
        
        # Performance metrics over time
        self._render_performance_trends(model_id)
        
        # Drift detection
        self._render_drift_analysis(model_id)
        
        # Recent performance table
        self._render_recent_performance(model_id)
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT model_id FROM model_metadata ORDER BY model_id")
                return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []
    
    def _render_performance_trends(self, model_id: str):
        """Render performance trends charts."""
        st.markdown("### 📈 Performance Trends")
        
        # Get performance data for last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = self.get_performance_history(
            model_id=model_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if df.empty:
            st.info("No performance data available for the selected period")
            return
        
        # Group by metric name
        metrics = df['metric_name'].unique()
        
        if len(metrics) == 1:
            # Single metric chart
            metric_df = df[df['metric_name'] == metrics[0]]
            
            fig = px.line(
                metric_df,
                x='timestamp',
                y='metric_value',
                title=f"{metrics[0].title()} Over Time",
                labels={'metric_value': metrics[0].title(), 'timestamp': 'Date'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Multiple metrics subplots
            fig = make_subplots(
                rows=min(len(metrics), 3),
                cols=1,
                subplot_titles=[metric.title() for metric in metrics[:3]],
                vertical_spacing=0.1
            )
            
            for i, metric in enumerate(metrics[:3]):
                metric_df = df[df['metric_name'] == metric]
                
                fig.add_trace(
                    go.Scatter(
                        x=metric_df['timestamp'],
                        y=metric_df['metric_value'],
                        mode='lines+markers',
                        name=metric.title(),
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_drift_analysis(self, model_id: str):
        """Render drift analysis section."""
        st.markdown("### 🚨 Drift Detection")
        
        # Check for drift in key metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        drift_results = []
        for metric in key_metrics:
            drift_info = self.detect_performance_drift(model_id, metric)
            if 'drift_detected' in drift_info:
                drift_results.append({
                    'metric': metric,
                    'drift_detected': drift_info['drift_detected'],
                    'magnitude': drift_info.get('drift_magnitude', 0),
                    'threshold': drift_info.get('threshold', 0)
                })
        
        if drift_results:
            drift_df = pd.DataFrame(drift_results)
            
            # Show drift status
            col1, col2 = st.columns(2)
            
            with col1:
                drift_count = drift_df['drift_detected'].sum()
                total_metrics = len(drift_df)
                
                if drift_count > 0:
                    st.error(f"⚠️ Drift detected in {drift_count}/{total_metrics} metrics")
                else:
                    st.success(f"✅ No drift detected in {total_metrics} metrics")
            
            with col2:
                if not drift_df.empty:
                    max_drift = drift_df['magnitude'].max()
                    st.metric("Max Drift Magnitude", f"{max_drift:.3f}")
            
            # Detailed drift table
            if not drift_df.empty:
                st.dataframe(
                    drift_df,
                    use_container_width=True,
                    column_config={
                        'metric': 'Metric',
                        'drift_detected': st.column_config.CheckboxColumn('Drift Detected'),
                        'magnitude': st.column_config.NumberColumn('Magnitude', format="%.3f"),
                        'threshold': st.column_config.NumberColumn('Threshold', format="%.3f")
                    }
                )
        else:
            st.info("No drift analysis data available")
    
    def _render_recent_performance(self, model_id: str):
        """Render recent performance table."""
        st.markdown("### 📋 Recent Performance Metrics")
        
        # Get last 50 performance records
        df = self.get_performance_history(model_id)
        
        if df.empty:
            st.info("No performance data available")
            return
        
        # Format for display
        display_df = df.head(50).copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(
            display_df[['timestamp', 'metric_name', 'metric_value', 'dataset_type', 'data_size']],
            use_container_width=True,
            column_config={
                'timestamp': 'Timestamp',
                'metric_name': 'Metric',
                'metric_value': st.column_config.NumberColumn('Value', format="%.4f"),
                'dataset_type': 'Dataset',
                'data_size': 'Data Size'
            }
        )

# Global instance for easy access
performance_tracker = ModelPerformanceTracker()
