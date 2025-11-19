#!/usr/bin/env python3
"""
Data Source Alerts System for GoalDiggers Platform

This module provides a comprehensive alert system for monitoring external data sources
and notifying relevant stakeholders when data integrity, availability or quality issues occur.

Features:
1. Real-time monitoring of data source availability
2. Integration with API monitor and data integrity checker
3. Multiple notification channels (email, Slack, PagerDuty)
4. Alert severity levels with configurable thresholds
5. Alerting logic for data quality, availability, and freshness
6. Failover mechanism recommendations for data source outages

Usage:
    python data_source_alerts.py [--config <config_file>] [--test-alert]
    
    Import:
    from utils.data_source_alerts import DataSourceAlertSystem, get_alert_system
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.api_monitor import ApiMonitor, ApiSource, DataQuality, get_api_monitor
from utils.data_integrity_checker import (
    CheckResult,
    CheckSeverity,
    get_data_integrity_checker,
)
from utils.load_test_alerts import LoadTestAlertSystem

# Central logging configuration unless under pytest
if not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        from utils.logging_config import configure_logging  # type: ignore
        configure_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("data_source_alerts")


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataSourceAlert:
    """Class representing a data source alert."""
    
    def __init__(
        self,
        source: str,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """Initialize a data source alert.
        
        Args:
            source: Name of the data source
            severity: Alert severity level
            message: Alert message
            details: Additional details about the alert
            timestamp: Alert timestamp (default: now)
        """
        self.source = source
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.now()
        self.alert_id = f"{self.source}_{int(self.timestamp.timestamp())}"
        self.status = "active"
        self.resolved_at = None
        self.resolution_message = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary representation.
        
        Returns:
            Dictionary representation of alert
        """
        return {
            "alert_id": self.alert_id,
            "source": self.source,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_message": self.resolution_message
        }
    
    def resolve(self, resolution_message: str) -> None:
        """Mark the alert as resolved.
        
        Args:
            resolution_message: Message explaining how the alert was resolved
        """
        self.status = "resolved"
        self.resolved_at = datetime.now()
        self.resolution_message = resolution_message
    
    def __str__(self) -> str:
        """String representation of the alert."""
        return f"[{self.severity.value.upper()}] {self.source}: {self.message}"


class DataSourceType(Enum):
    """Types of data sources."""
    
    FOOTBALL_DATA = "football_data"
    UNDERSTAT = "understat"
    ODDS_API = "odds_api"
    WEATHER_API = "weather_api"
    DATABASE = "database"
    CUSTOM = "custom"


class DataSourceAlertSystem:
    """Alert system for data source monitoring."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the data source alert system.
        
        Args:
            config_path: Path to alert system configuration
        """
        self.config_path = config_path or os.path.join(
            Path(__file__).parent.parent, "config", "data_source_alerts_config.json"
        )
        self.config = self._load_config()
        self.active_alerts = {}  # Dict of active alerts by alert_id
        self.alert_history = []  # List of all alerts (active and resolved)
        
        # Get references to other monitoring systems
        self.api_monitor = get_api_monitor()
        self.data_integrity_checker = get_data_integrity_checker()
        
        # Register with API monitor for alerts
        self.api_monitor.register_alert_callback(self._handle_api_monitor_alert)
        
        # Load test alert system for notification methods
        self.notification_system = LoadTestAlertSystem()
        
        logger.info("Data Source Alert System initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            "sources": {
                "football_data": {
                    "description": "Football Data API",
                    "api_source": "FOOTBALL_DATA",
                    "freshness_threshold_hours": 24,
                    "quality_threshold": 0.7,
                    "availability_threshold": 0.95
                },
                "understat": {
                    "description": "Understat API",
                    "api_source": "UNDERSTAT",
                    "freshness_threshold_hours": 24,
                    "quality_threshold": 0.7,
                    "availability_threshold": 0.95
                },
                "odds_api": {
                    "description": "Odds API",
                    "api_source": "ODDS_API",
                    "freshness_threshold_hours": 6,
                    "quality_threshold": 0.8,
                    "availability_threshold": 0.98
                },
                "weather_api": {
                    "description": "Weather API",
                    "api_source": "WEATHER_API",
                    "freshness_threshold_hours": 3,
                    "quality_threshold": 0.7,
                    "availability_threshold": 0.95
                },
                "database": {
                    "description": "Internal Database",
                    "api_source": None,
                    "freshness_threshold_hours": 24,
                    "quality_threshold": 0.9,
                    "availability_threshold": 0.99
                }
            },
            "notifications": {
                "email": ["alerts@goaldiggers.com"],
                "slack": "#goaldiggers-alerts",
                "pagerduty": {
                    "critical_only": True,
                    "service_key": "DUMMY_PD_SERVICE_KEY"
                }
            },
            "severity_mapping": {
                "data_quality_poor": "WARNING",
                "data_quality_invalid": "ERROR",
                "rate_limit_warning": "WARNING",
                "rate_limit_critical": "ERROR",
                "rate_limit_blocked": "CRITICAL",
                "data_integrity_warning": "WARNING",
                "data_integrity_error": "ERROR",
                "data_integrity_critical": "CRITICAL",
                "availability_warning": "WARNING",
                "availability_error": "ERROR",
                "freshness_warning": "WARNING",
                "freshness_error": "ERROR"
            },
            "cooldown": {
                "minutes": 30  # Don't send repeated alerts within this time
            },
            "failover_recommendations": {
                "football_data": [
                    "API-Football",
                    "SportMonks Football API"
                ],
                "understat": [
                    "StatsBomb",
                    "WhoScored"
                ],
                "odds_api": [
                    "Betfair API",
                    "Pinnacle API",
                    "OddsPortal"
                ],
                "weather_api": [
                    "AccuWeather API",
                    "Dark Sky API",
                    "WeatherBit"
                ]
            }
        }
        
        # Try to load from file
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    
                # Merge with default config
                for section in default_config:
                    if section in file_config:
                        if isinstance(default_config[section], dict) and isinstance(file_config[section], dict):
                            default_config[section].update(file_config[section])
                        else:
                            default_config[section] = file_config[section]
                
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found at {self.config_path}, using defaults")
                
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
        
        return default_config
    
    def _handle_api_monitor_alert(self, alert_type: str, message: str, data: Dict) -> None:
        """Handle alerts from the API monitor.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            data: Alert data
        """
        logger.info(f"Received API monitor alert: {alert_type} - {message}")
        
        source = data.get("source", "unknown")
        severity_name = self.config["severity_mapping"].get(alert_type, "WARNING")
        severity = getattr(AlertSeverity, severity_name)
        
        # Find the corresponding data source for this API source
        data_source = None
        for src_name, src_config in self.config["sources"].items():
            if src_config.get("api_source") == source:
                data_source = src_name
                break
        
        if not data_source:
            # Default to the source name from the alert
            data_source = source.lower() if isinstance(source, str) else "unknown"
        
        # Create alert
        alert = DataSourceAlert(
            source=data_source,
            severity=severity,
            message=message,
            details=data
        )
        
        # Process the alert
        self._process_alert(alert)
    
    def _process_alert(self, alert: DataSourceAlert) -> None:
        """Process a new alert.
        
        Args:
            alert: The alert to process
        """
        # Check if similar alert is already active (same source and similar message)
        for active_id, active_alert in list(self.active_alerts.items()):
            if (active_alert.source == alert.source and 
                active_alert.severity == alert.severity and
                self._similar_messages(active_alert.message, alert.message)):
                
                # Check cooldown period
                cooldown_minutes = self.config.get("cooldown", {}).get("minutes", 30)
                if (datetime.now() - active_alert.timestamp).total_seconds() < cooldown_minutes * 60:
                    logger.info(f"Skipping duplicate alert for {alert.source} (cooldown period active)")
                    return
                else:
                    # Resolve the old alert
                    self._resolve_alert(active_id, "Replaced by newer alert")
        
        # Add to active alerts
        self.active_alerts[alert.alert_id] = alert
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        logger.info(f"Processed new alert: {alert}")
    
    def _similar_messages(self, msg1: str, msg2: str) -> bool:
        """Check if two alert messages are similar.
        
        Args:
            msg1: First message
            msg2: Second message
            
        Returns:
            True if messages are similar
        """
        # Simple implementation - check if messages share significant words
        # Could be improved with more sophisticated text similarity algorithms
        
        # Convert to lowercase and split into words
        words1 = set(w.lower() for w in msg1.split() if len(w) > 3)
        words2 = set(w.lower() for w in msg2.split() if len(w) > 3)
        
        # Calculate similarity as intersection over union
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        
        # Consider similar if 70% or more words match
        return similarity >= 0.7
    
    def _resolve_alert(self, alert_id: str, resolution_message: str) -> None:
        """Resolve an active alert.
        
        Args:
            alert_id: ID of the alert to resolve
            resolution_message: Message explaining the resolution
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolve(resolution_message)
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Resolved alert {alert_id}: {resolution_message}")
    
    def _send_alert_notifications(self, alert: DataSourceAlert) -> None:
        """Send notifications for an alert.
        
        Args:
            alert: Alert to send notifications for
        """
        # Convert our alert to a format compatible with the load test alert system
        test_alert = {
            "type": f"data_source_{alert.severity.value}",
            "level": alert.severity.value,
            "timestamp": alert.timestamp.isoformat(),
            "message": alert.message,
            "scenario": alert.source,
            "num_users": "N/A",  # Not applicable for data source alerts
            "value": 0.0,  # Not applicable
            "threshold": 0.0,  # Not applicable
            "details": alert.details
        }
        
        # Send using the load test alert notification methods
        self.notification_system.send_alerts([test_alert])
    
    def check_data_sources(self) -> List[DataSourceAlert]:
        """Check all data sources for issues.
        
        Returns:
            List of new alerts generated
        """
        logger.info("Checking all data sources for issues...")
        
        new_alerts = []
        
        # Check API sources using API monitor
        api_status = self.api_monitor.get_all_sources_status()
        for source_name, source_config in self.config["sources"].items():
            api_source_name = source_config.get("api_source")
            
            if api_source_name and api_source_name in api_status:
                api_info = api_status[api_source_name]
                
                # Check availability based on error rate
                if "api_stats" in api_info and "error_rate" in api_info["api_stats"]:
                    error_rate = api_info["api_stats"]["error_rate"]
                    availability = 1.0 - error_rate
                    availability_threshold = source_config.get("availability_threshold", 0.95)
                    
                    if availability < availability_threshold:
                        severity = AlertSeverity.ERROR if availability < availability_threshold * 0.8 else AlertSeverity.WARNING
                        
                        alert = DataSourceAlert(
                            source=source_name,
                            severity=severity,
                            message=f"Data source availability below threshold: {availability:.2%} (threshold: {availability_threshold:.2%})",
                            details={
                                "availability": availability,
                                "threshold": availability_threshold,
                                "error_rate": error_rate,
                                "api_stats": api_info["api_stats"]
                            }
                        )
                        new_alerts.append(alert)
                
                # Check data quality
                if "quality" in api_info and "last_score" in api_info["quality"]:
                    quality_score = api_info["quality"]["last_score"]
                    quality_threshold = source_config.get("quality_threshold", 0.7)
                    
                    if quality_score is not None and quality_score < quality_threshold:
                        severity = AlertSeverity.ERROR if quality_score < quality_threshold * 0.8 else AlertSeverity.WARNING
                        
                        alert = DataSourceAlert(
                            source=source_name,
                            severity=severity,
                            message=f"Data quality below threshold: {quality_score:.2f} (threshold: {quality_threshold:.2f})",
                            details={
                                "quality_score": quality_score,
                                "threshold": quality_threshold,
                                "quality_metrics": api_info["quality"].get("last_metrics", {})
                            }
                        )
                        new_alerts.append(alert)
        
        # Check data freshness (would require additional data store in a real implementation)
        # This is a placeholder for demonstration purposes
        
        # Check data integrity using the data integrity checker
        integrity_results = self.data_integrity_checker.get_latest_results()
        for result in integrity_results:
            if not result.passed:
                source_name = result.check_name
                for prefix in ["data_consistency_", "database_schema_"]:
                    if source_name.startswith(prefix):
                        source_name = source_name[len(prefix):]
                
                # Map check severity to alert severity
                alert_severity = AlertSeverity.INFO
                if result.severity == CheckSeverity.CRITICAL:
                    alert_severity = AlertSeverity.CRITICAL
                elif result.severity == CheckSeverity.ERROR:
                    alert_severity = AlertSeverity.ERROR
                elif result.severity == CheckSeverity.WARNING:
                    alert_severity = AlertSeverity.WARNING
                
                alert = DataSourceAlert(
                    source=source_name,
                    severity=alert_severity,
                    message=result.message,
                    details=result.details
                )
                new_alerts.append(alert)
        
        # Process all new alerts
        for alert in new_alerts:
            self._process_alert(alert)
        
        if new_alerts:
            logger.info(f"Generated {len(new_alerts)} new alerts")
        else:
            logger.info("No new alerts generated")
        
        return new_alerts
    
    def get_active_alerts(self) -> List[DataSourceAlert]:
        """Get list of all active alerts.
        
        Returns:
            List of active alerts
        """
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[DataSourceAlert]:
        """Get alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts (newest first)
        """
        # Sort by timestamp (newest first)
        sorted_alerts = sorted(
            self.alert_history,
            key=lambda a: a.timestamp,
            reverse=True
        )
        
        return sorted_alerts[:limit]
    
    def export_alerts_to_json(self, filepath: str) -> None:
        """Export all alerts to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        alerts_dict = [a.to_dict() for a in self.alert_history]
        
        with open(filepath, 'w') as f:
            json.dump(alerts_dict, f, indent=2)
        
        logger.info(f"Exported {len(alerts_dict)} alerts to {filepath}")
    
    def get_failover_recommendations(self, source: str) -> List[str]:
        """Get recommendations for failover providers for a data source.
        
        Args:
            source: Name of the data source
            
        Returns:
            List of recommended failover providers
        """
        recommendations = self.config.get("failover_recommendations", {}).get(source, [])
        return recommendations
    
    def generate_test_alert(self, source: Optional[str] = None) -> DataSourceAlert:
        """Generate a test alert for debugging purposes.
        
        Args:
            source: Specific source to generate alert for (or random if None)
            
        Returns:
            Generated test alert
        """
        if source is None:
            # Pick a random source
            import random
            sources = list(self.config["sources"].keys())
            source = random.choice(sources)
        
        alert = DataSourceAlert(
            source=source,
            severity=AlertSeverity.WARNING,
            message=f"Test alert for {source}",
            details={
                "test": True,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Process the alert
        self._process_alert(alert)
        
        return alert


# Singleton instance
_data_source_alert_system = None

def get_alert_system() -> DataSourceAlertSystem:
    """Get the global data source alert system instance.
    
    Returns:
        DataSourceAlertSystem instance
    """
    global _data_source_alert_system
    if _data_source_alert_system is None:
        _data_source_alert_system = DataSourceAlertSystem()
    return _data_source_alert_system


def main():
    """Main entry point for the data source alert system."""
    parser = argparse.ArgumentParser(description="Data Source Alert System")
    parser.add_argument("--config", help="Path to alert configuration file")
    parser.add_argument("--test-alert", action="store_true", help="Generate a test alert")
    parser.add_argument("--source", help="Specific data source to test")
    parser.add_argument("--check-now", action="store_true", help="Check all data sources now")
    parser.add_argument("--monitor", action="store_true", help="Start continuous monitoring")
    parser.add_argument("--export", help="Export alerts to JSON file")
    
    args = parser.parse_args()
    
    # Create alert system
    alert_system = DataSourceAlertSystem(config_path=args.config)
    
    if args.test_alert:
        # Generate a test alert
        source = args.source
        logger.info(f"Generating test alert for {source or 'random source'}")
        alert = alert_system.generate_test_alert(source)
        logger.info(f"Generated test alert: {alert}")
    
    elif args.check_now:
        # Run a manual check
        logger.info("Running manual check of all data sources")
        alerts = alert_system.check_data_sources()
        logger.info(f"Check complete, generated {len(alerts)} alerts")
    
    elif args.monitor:
        # Start continuous monitoring
        logger.info("Starting continuous data source monitoring")
        interval_seconds = 60  # Check every minute
        
        try:
            while True:
                alert_system.check_data_sources()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    
    elif args.export:
        # Export alerts to JSON
        if not alert_system.alert_history:
            # Run a check to generate some data
            alert_system.check_data_sources()
        
        logger.info(f"Exporting alerts to {args.export}")
        alert_system.export_alerts_to_json(args.export)
    
    else:
        # No specific action requested
        logger.info("No action specified. Use --test-alert, --check-now, or --monitor")
        parser.print_help()


if __name__ == "__main__":
    main()
