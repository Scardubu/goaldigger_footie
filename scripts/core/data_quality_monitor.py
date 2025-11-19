"""
Advanced Data Quality Monitor with ML-based anomaly detection and real-time validation.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics for monitoring."""
    completeness: float  # Percentage of non-null values
    accuracy: float      # Percentage of values within expected ranges
    consistency: float   # Percentage of values consistent with historical patterns
    timeliness: float    # Percentage of data received within expected timeframe
    validity: float      # Percentage of values matching expected formats
    uniqueness: float    # Percentage of unique values where expected
    overall_score: float # Weighted overall quality score

@dataclass
class DataAnomalyAlert:
    """Data anomaly alert."""
    timestamp: datetime
    source: str
    data_type: str
    anomaly_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    affected_records: int
    confidence_score: float
    suggested_action: str

class MLAnomalyDetector:
    """ML-based anomaly detection for data quality."""
    
    def __init__(self):
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.historical_data: Dict[str, List[Dict]] = {}
        self.training_window = 30  # days
        
    def train_model(self, data_type: str, historical_data: List[Dict[str, Any]]):
        """Train anomaly detection model for specific data type."""
        if len(historical_data) < 10:
            logger.warning(f"Insufficient data to train model for {data_type}")
            return
        
        try:
            # Convert to DataFrame and extract numerical features
            df = pd.DataFrame(historical_data)
            numerical_features = df.select_dtypes(include=[np.number]).columns
            
            if len(numerical_features) == 0:
                logger.warning(f"No numerical features found for {data_type}")
                return
            
            # Prepare training data
            X = df[numerical_features].fillna(df[numerical_features].median())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train isolation forest
            model = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_estimators=100
            )
            model.fit(X_scaled)
            
            # Store model and scaler
            self.models[data_type] = model
            self.scalers[data_type] = scaler
            self.historical_data[data_type] = historical_data
            
            logger.info(f"Trained anomaly detection model for {data_type}")
            
        except Exception as e:
            logger.error(f"Failed to train model for {data_type}: {e}")
    
    def detect_anomalies(self, data_type: str, new_data: List[Dict[str, Any]]) -> List[int]:
        """Detect anomalies in new data."""
        if data_type not in self.models:
            logger.warning(f"No trained model for {data_type}")
            return []
        
        try:
            model = self.models[data_type]
            scaler = self.scalers[data_type]
            
            # Convert to DataFrame
            df = pd.DataFrame(new_data)
            
            # Get same features as training
            historical_df = pd.DataFrame(self.historical_data[data_type])
            numerical_features = historical_df.select_dtypes(include=[np.number]).columns
            
            # Prepare data
            X = df[numerical_features].fillna(df[numerical_features].median())
            X_scaled = scaler.transform(X)
            
            # Predict anomalies
            predictions = model.predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)
            
            # Return indices of anomalies (prediction = -1)
            anomaly_indices = [i for i, pred in enumerate(predictions) if pred == -1]
            
            return anomaly_indices
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies for {data_type}: {e}")
            return []

class DataQualityValidator:
    """Validates data quality based on predefined rules."""
    
    def __init__(self):
        self.validation_rules = {
            'match_data': {
                'required_fields': ['home_team', 'away_team', 'match_date'],
                'numeric_ranges': {
                    'home_score': (0, 20),
                    'away_score': (0, 20)
                },
                'date_formats': ['match_date'],
                'string_patterns': {
                    'status': ['scheduled', 'in_play', 'finished', 'postponed', 'cancelled']
                }
            },
            'team_stats': {
                'required_fields': ['team_id', 'season', 'matches_played'],
                'numeric_ranges': {
                    'wins': (0, 50),
                    'draws': (0, 50),
                    'losses': (0, 50),
                    'points': (0, 150)
                }
            },
            'odds_data': {
                'required_fields': ['match_id', 'bookmaker'],
                'numeric_ranges': {
                    'home_win': (1.0, 50.0),
                    'draw': (1.0, 50.0),
                    'away_win': (1.0, 50.0)
                }
            }
        }
    
    def validate_data(self, data_type: str, data: List[Dict[str, Any]]) -> DataQualityMetrics:
        """Validate data quality and return metrics."""
        if data_type not in self.validation_rules:
            logger.warning(f"No validation rules for {data_type}")
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        rules = self.validation_rules[data_type]
        total_records = len(data)
        
        if total_records == 0:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate completeness
        completeness_scores = []
        for field in rules.get('required_fields', []):
            non_null_count = sum(1 for record in data if record.get(field) is not None)
            completeness_scores.append(non_null_count / total_records)
        
        completeness = np.mean(completeness_scores) if completeness_scores else 1.0
        
        # Calculate accuracy (values within expected ranges)
        accuracy_scores = []
        for field, (min_val, max_val) in rules.get('numeric_ranges', {}).items():
            valid_count = sum(
                1 for record in data 
                if record.get(field) is not None and 
                min_val <= record.get(field, 0) <= max_val
            )
            field_total = sum(1 for record in data if record.get(field) is not None)
            if field_total > 0:
                accuracy_scores.append(valid_count / field_total)
        
        accuracy = np.mean(accuracy_scores) if accuracy_scores else 1.0
        
        # Calculate validity (format compliance)
        validity_scores = []
        for field in rules.get('date_formats', []):
            valid_count = sum(
                1 for record in data 
                if self._is_valid_date(record.get(field))
            )
            validity_scores.append(valid_count / total_records)
        
        for field, valid_values in rules.get('string_patterns', {}).items():
            valid_count = sum(
                1 for record in data 
                if record.get(field) in valid_values
            )
            field_total = sum(1 for record in data if record.get(field) is not None)
            if field_total > 0:
                validity_scores.append(valid_count / field_total)
        
        validity = np.mean(validity_scores) if validity_scores else 1.0
        
        # Calculate timeliness (assume recent data is more timely)
        timeliness = self._calculate_timeliness(data)
        
        # Calculate consistency (placeholder - would need historical comparison)
        consistency = 0.9  # Placeholder
        
        # Calculate uniqueness (for fields that should be unique)
        uniqueness = self._calculate_uniqueness(data, data_type)
        
        # Calculate overall score (weighted average)
        overall_score = (
            completeness * 0.25 +
            accuracy * 0.25 +
            validity * 0.20 +
            timeliness * 0.15 +
            consistency * 0.10 +
            uniqueness * 0.05
        )
        
        return DataQualityMetrics(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            timeliness=timeliness,
            validity=validity,
            uniqueness=uniqueness,
            overall_score=overall_score
        )
    
    def _is_valid_date(self, date_value: Any) -> bool:
        """Check if value is a valid date."""
        if date_value is None:
            return False
        
        try:
            if isinstance(date_value, str):
                datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            elif isinstance(date_value, datetime):
                return True
            return True
        except:
            return False
    
    def _calculate_timeliness(self, data: List[Dict[str, Any]]) -> float:
        """Calculate timeliness score based on data freshness."""
        now = datetime.now()
        recent_threshold = timedelta(hours=24)
        
        recent_count = 0
        total_with_timestamp = 0
        
        for record in data:
            timestamp_fields = ['created_at', 'updated_at', 'timestamp', 'match_date']
            
            for field in timestamp_fields:
                if field in record and record[field]:
                    try:
                        if isinstance(record[field], str):
                            record_time = datetime.fromisoformat(record[field].replace('Z', '+00:00'))
                        else:
                            record_time = record[field]
                        
                        total_with_timestamp += 1
                        if now - record_time <= recent_threshold:
                            recent_count += 1
                        break
                    except:
                        continue
        
        return recent_count / total_with_timestamp if total_with_timestamp > 0 else 1.0
    
    def _calculate_uniqueness(self, data: List[Dict[str, Any]], data_type: str) -> float:
        """Calculate uniqueness score for fields that should be unique."""
        unique_fields = {
            'match_data': ['id'],
            'team_stats': ['team_id', 'season'],
            'odds_data': ['match_id', 'bookmaker']
        }
        
        if data_type not in unique_fields:
            return 1.0
        
        uniqueness_scores = []
        for field in unique_fields[data_type]:
            values = [record.get(field) for record in data if record.get(field) is not None]
            if values:
                unique_ratio = len(set(values)) / len(values)
                uniqueness_scores.append(unique_ratio)
        
        return np.mean(uniqueness_scores) if uniqueness_scores else 1.0

class DataQualityMonitor:
    """Main data quality monitoring system."""
    
    def __init__(self):
        self.validator = DataQualityValidator()
        self.anomaly_detector = MLAnomalyDetector()
        self.quality_history: Dict[str, List[DataQualityMetrics]] = {}
        self.alerts: List[DataAnomalyAlert] = []
        self.alert_thresholds = {
            'completeness': 0.8,
            'accuracy': 0.85,
            'validity': 0.9,
            'overall_score': 0.8
        }
    
    async def monitor_data_quality(self, data_type: str, data: List[Dict[str, Any]]) -> DataQualityMetrics:
        """Monitor data quality and generate alerts if needed."""
        # Validate data quality
        metrics = self.validator.validate_data(data_type, data)
        
        # Store metrics history
        if data_type not in self.quality_history:
            self.quality_history[data_type] = []
        
        self.quality_history[data_type].append(metrics)
        
        # Keep only recent history (last 100 measurements)
        if len(self.quality_history[data_type]) > 100:
            self.quality_history[data_type] = self.quality_history[data_type][-100:]
        
        # Check for quality issues and generate alerts
        await self._check_quality_alerts(data_type, metrics, len(data))
        
        # Detect anomalies
        anomaly_indices = self.anomaly_detector.detect_anomalies(data_type, data)
        if anomaly_indices:
            await self._generate_anomaly_alert(data_type, anomaly_indices, len(data))
        
        logger.info(f"Data quality for {data_type}: {metrics.overall_score:.2f}")
        return metrics
    
    async def _check_quality_alerts(self, data_type: str, metrics: DataQualityMetrics, record_count: int):
        """Check quality metrics against thresholds and generate alerts."""
        alerts_to_generate = []
        
        if metrics.completeness < self.alert_thresholds['completeness']:
            alerts_to_generate.append(('completeness', metrics.completeness, 'HIGH'))
        
        if metrics.accuracy < self.alert_thresholds['accuracy']:
            alerts_to_generate.append(('accuracy', metrics.accuracy, 'HIGH'))
        
        if metrics.validity < self.alert_thresholds['validity']:
            alerts_to_generate.append(('validity', metrics.validity, 'MEDIUM'))
        
        if metrics.overall_score < self.alert_thresholds['overall_score']:
            alerts_to_generate.append(('overall_score', metrics.overall_score, 'CRITICAL'))
        
        for metric_name, value, severity in alerts_to_generate:
            alert = DataAnomalyAlert(
                timestamp=datetime.now(),
                source='data_quality_monitor',
                data_type=data_type,
                anomaly_type=f'low_{metric_name}',
                severity=severity,
                description=f'{metric_name.title()} below threshold: {value:.2f}',
                affected_records=record_count,
                confidence_score=1.0 - value,
                suggested_action=f'Review {data_type} data source and validation rules'
            )
            
            self.alerts.append(alert)
            logger.warning(f"Data quality alert: {alert.description}")
    
    async def _generate_anomaly_alert(self, data_type: str, anomaly_indices: List[int], total_records: int):
        """Generate alert for detected anomalies."""
        anomaly_count = len(anomaly_indices)
        anomaly_rate = anomaly_count / total_records
        
        severity = 'LOW'
        if anomaly_rate > 0.2:
            severity = 'CRITICAL'
        elif anomaly_rate > 0.1:
            severity = 'HIGH'
        elif anomaly_rate > 0.05:
            severity = 'MEDIUM'
        
        alert = DataAnomalyAlert(
            timestamp=datetime.now(),
            source='ml_anomaly_detector',
            data_type=data_type,
            anomaly_type='statistical_anomaly',
            severity=severity,
            description=f'Detected {anomaly_count} anomalies in {total_records} records ({anomaly_rate:.1%})',
            affected_records=anomaly_count,
            confidence_score=min(anomaly_rate * 5, 1.0),
            suggested_action='Review anomalous records and update validation rules if needed'
        )
        
        self.alerts.append(alert)
        logger.warning(f"Anomaly alert: {alert.description}")
    
    def get_quality_summary(self, data_type: Optional[str] = None) -> Dict[str, Any]:
        """Get quality summary for all or specific data types."""
        if data_type:
            history = self.quality_history.get(data_type, [])
            if not history:
                return {}
            
            latest = history[-1]
            avg_score = np.mean([m.overall_score for m in history])
            
            return {
                'data_type': data_type,
                'latest_score': latest.overall_score,
                'average_score': avg_score,
                'measurements': len(history),
                'latest_metrics': latest
            }
        else:
            summary = {}
            for dt in self.quality_history:
                summary[dt] = self.get_quality_summary(dt)
            return summary
    
    def get_recent_alerts(self, hours: int = 24) -> List[DataAnomalyAlert]:
        """Get recent alerts within specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff]
