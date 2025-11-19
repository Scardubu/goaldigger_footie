#!/usr/bin/env python3
"""
Prediction History System for GoalDiggers Platform

Tracks and displays historical prediction accuracy with comprehensive
analytics and performance insights.
"""

import json
import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PredictionRecord:
    """Data class for prediction records."""
    prediction_id: str
    timestamp: datetime
    home_team: str
    away_team: str
    league: str
    predicted_outcome: str
    predicted_probability: float
    actual_outcome: Optional[str] = None
    is_correct: Optional[bool] = None
    confidence_level: str = "medium"
    model_version: str = "v1.0"
    features_used: List[str] = None
    
    def __post_init__(self):
        if self.features_used is None:
            self.features_used = []

class PredictionHistorySystem:
    """
    Comprehensive prediction history tracking and analytics system.
    """
    
    def __init__(self, db_path: str = "data/prediction_history.db"):
        """
        Initialize the prediction history system.
        
        Args:
            db_path: Path to SQLite database for storing predictions
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        
        logger.info(f"Prediction history system initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    league TEXT NOT NULL,
                    predicted_outcome TEXT NOT NULL,
                    predicted_probability REAL NOT NULL,
                    actual_outcome TEXT,
                    is_correct INTEGER,
                    confidence_level TEXT,
                    model_version TEXT,
                    features_used TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_league ON predictions(league)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_teams ON predictions(home_team, away_team)
            """)
    
    def add_prediction(self, prediction: PredictionRecord) -> bool:
        """
        Add a new prediction to the history.
        
        Args:
            prediction: Prediction record to add
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO predictions (
                        prediction_id, timestamp, home_team, away_team, league,
                        predicted_outcome, predicted_probability, actual_outcome,
                        is_correct, confidence_level, model_version, features_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.prediction_id,
                    prediction.timestamp.isoformat(),
                    prediction.home_team,
                    prediction.away_team,
                    prediction.league,
                    prediction.predicted_outcome,
                    prediction.predicted_probability,
                    prediction.actual_outcome,
                    prediction.is_correct,
                    prediction.confidence_level,
                    prediction.model_version,
                    json.dumps(prediction.features_used)
                ))
            
            logger.info(f"Added prediction: {prediction.prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add prediction: {e}")
            return False
    
    def update_prediction_outcome(self, 
                                prediction_id: str,
                                actual_outcome: str) -> bool:
        """
        Update a prediction with the actual outcome.
        
        Args:
            prediction_id: ID of the prediction to update
            actual_outcome: The actual match outcome
            
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get the predicted outcome
                cursor = conn.execute(
                    "SELECT predicted_outcome FROM predictions WHERE prediction_id = ?",
                    (prediction_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    logger.error(f"Prediction not found: {prediction_id}")
                    return False
                
                predicted_outcome = result[0]
                is_correct = predicted_outcome == actual_outcome
                
                # Update the prediction
                conn.execute("""
                    UPDATE predictions 
                    SET actual_outcome = ?, is_correct = ?
                    WHERE prediction_id = ?
                """, (actual_outcome, is_correct, prediction_id))
            
            logger.info(f"Updated prediction outcome: {prediction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update prediction outcome: {e}")
            return False
    
    def get_predictions(self, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       league: Optional[str] = None,
                       team: Optional[str] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """
        Get predictions with optional filtering.
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            league: League to filter by
            team: Team to filter by (home or away)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with prediction records
        """
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        if league:
            query += " AND league = ?"
            params.append(league)
        
        if team:
            query += " AND (home_team = ? OR away_team = ?)"
            params.extend([team, team])
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['features_used'] = df['features_used'].apply(
                        lambda x: json.loads(x) if x else []
                    )
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            return pd.DataFrame()
    
    def get_accuracy_stats(self, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get comprehensive accuracy statistics.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            Dictionary with accuracy statistics
        """
        df = self.get_predictions(start_date=start_date, end_date=end_date)
        
        if df.empty:
            return {
                'total_predictions': 0,
                'completed_predictions': 0,
                'overall_accuracy': 0,
                'accuracy_by_league': {},
                'accuracy_by_confidence': {},
                'accuracy_trend': []
            }
        
        # Filter completed predictions
        completed_df = df[df['is_correct'].notna()]
        
        stats = {
            'total_predictions': len(df),
            'completed_predictions': len(completed_df),
            'pending_predictions': len(df) - len(completed_df),
            'overall_accuracy': completed_df['is_correct'].mean() if not completed_df.empty else 0
        }
        
        # Accuracy by league
        if not completed_df.empty:
            stats['accuracy_by_league'] = completed_df.groupby('league')['is_correct'].agg([
                'count', 'mean'
            ]).to_dict('index')
            
            # Accuracy by confidence level
            stats['accuracy_by_confidence'] = completed_df.groupby('confidence_level')['is_correct'].agg([
                'count', 'mean'
            ]).to_dict('index')
            
            # Weekly accuracy trend
            completed_df['week'] = completed_df['timestamp'].dt.to_period('W')
            weekly_accuracy = completed_df.groupby('week')['is_correct'].agg([
                'count', 'mean'
            ]).reset_index()
            weekly_accuracy['week'] = weekly_accuracy['week'].astype(str)
            stats['accuracy_trend'] = weekly_accuracy.to_dict('records')
        
        return stats
    
    def render_history_dashboard(self):
        """Render a comprehensive prediction history dashboard."""
        st.markdown("## 📊 Prediction History & Analytics")
        
        # Date range selector
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30),
                max_value=datetime.now().date()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
        
        with col3:
            league_filter = st.selectbox(
                "League",
                options=["All"] + self._get_available_leagues(),
                index=0
            )
        
        # Get data
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        
        df = self.get_predictions(
            start_date=start_datetime,
            end_date=end_datetime,
            league=league_filter if league_filter != "All" else None
        )
        
        stats = self.get_accuracy_stats(start_datetime, end_datetime)
        
        # Key metrics
        self._render_key_metrics(stats)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_accuracy_trend_chart(stats)
        
        with col2:
            self._render_league_accuracy_chart(stats)
        
        # Detailed predictions table
        self._render_predictions_table(df)
    
    def _get_available_leagues(self) -> List[str]:
        """Get list of available leagues."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT league FROM predictions ORDER BY league")
                return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []
    
    def _render_key_metrics(self, stats: Dict[str, Any]):
        """Render key metrics cards."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Predictions",
                stats['total_predictions'],
                delta=None
            )
        
        with col2:
            st.metric(
                "Completed",
                stats['completed_predictions'],
                delta=f"{stats['pending_predictions']} pending"
            )
        
        with col3:
            accuracy_pct = stats['overall_accuracy'] * 100
            st.metric(
                "Overall Accuracy",
                f"{accuracy_pct:.1f}%",
                delta=f"{'🎯' if accuracy_pct >= 70 else '📈' if accuracy_pct >= 60 else '⚠️'}"
            )
        
        with col4:
            if stats['accuracy_trend']:
                recent_accuracy = stats['accuracy_trend'][-1]['mean'] * 100
                st.metric(
                    "Recent Accuracy",
                    f"{recent_accuracy:.1f}%",
                    delta=None
                )
    
    def _render_accuracy_trend_chart(self, stats: Dict[str, Any]):
        """Render accuracy trend chart."""
        st.markdown("### 📈 Accuracy Trend")
        
        if not stats['accuracy_trend']:
            st.info("No trend data available")
            return
        
        trend_df = pd.DataFrame(stats['accuracy_trend'])
        
        fig = px.line(
            trend_df,
            x='week',
            y='mean',
            title="Weekly Accuracy Trend",
            labels={'mean': 'Accuracy', 'week': 'Week'}
        )
        
        fig.update_traces(line_color='#1e40af')
        fig.update_layout(
            yaxis_tickformat='.1%',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_league_accuracy_chart(self, stats: Dict[str, Any]):
        """Render league accuracy chart."""
        st.markdown("### 🏆 Accuracy by League")
        
        if not stats['accuracy_by_league']:
            st.info("No league data available")
            return
        
        league_data = []
        for league, data in stats['accuracy_by_league'].items():
            league_data.append({
                'league': league,
                'accuracy': data['mean'],
                'count': data['count']
            })
        
        league_df = pd.DataFrame(league_data)
        
        fig = px.bar(
            league_df,
            x='league',
            y='accuracy',
            title="Accuracy by League",
            labels={'accuracy': 'Accuracy', 'league': 'League'},
            color='accuracy',
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            yaxis_tickformat='.1%',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_predictions_table(self, df: pd.DataFrame):
        """Render detailed predictions table."""
        st.markdown("### 📋 Recent Predictions")
        
        if df.empty:
            st.info("No predictions found for the selected criteria")
            return
        
        # Format the dataframe for display
        display_df = df.copy()
        display_df['Match'] = display_df['home_team'] + ' vs ' + display_df['away_team']
        display_df['Confidence'] = display_df['predicted_probability'].apply(lambda x: f"{x:.1%}")
        display_df['Status'] = display_df.apply(
            lambda row: (
                "✅ Correct" if row['is_correct'] == 1
                else "❌ Incorrect" if row['is_correct'] == 0
                else "⏳ Pending"
            ), axis=1
        )
        
        # Select columns for display
        display_columns = [
            'timestamp', 'Match', 'league', 'predicted_outcome',
            'Confidence', 'actual_outcome', 'Status'
        ]
        
        st.dataframe(
            display_df[display_columns].head(50),
            use_container_width=True,
            column_config={
                'timestamp': st.column_config.DatetimeColumn(
                    'Date',
                    format='DD/MM/YYYY HH:mm'
                ),
                'Match': 'Match',
                'league': 'League',
                'predicted_outcome': 'Prediction',
                'Confidence': 'Confidence',
                'actual_outcome': 'Actual',
                'Status': 'Result'
            }
        )

# Global instance for easy access
prediction_history = PredictionHistorySystem()
