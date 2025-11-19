#!/usr/bin/env python3
"""
Data Source Failover System for GoalDiggers Platform

This module provides an automated failover mechanism for external data sources
in the GoalDiggers platform. When primary data sources fail or provide poor quality data,
the system automatically switches to secondary data sources to ensure data availability.

Features:
1. Configuration-driven failover policies
2. Automatic switching between primary and secondary sources
3. Health checking and quality monitoring for all sources
4. Integration with data_source_alerts for notification
5. Self-healing capability when primary sources recover
6. Metrics tracking for failover events

Usage:
    python utils/data_source_failover.py [--config <config_file>] [--test-failover]
    
    Import:
    from utils.data_source_failover import DataSourceFailoverManager, get_failover_manager
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from utils.api_monitor import ApiSource, get_api_monitor
from utils.data_integrity_checker import get_data_integrity_checker
from utils.data_source_alerts import AlertSeverity, DataSourceAlert, get_alert_system

# Centralized logging configuration unless under pytest
if not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        from utils.logging_config import configure_logging  # type: ignore
        configure_logging()
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("data_source_failover")


class SourceStatus(Enum):
    """Status of a data source."""
    
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class FailoverEvent:
    """Class representing a failover event."""
    
    def __init__(
        self,
        source_type: str,
        primary_source: str,
        backup_source: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize a failover event.
        
        Args:
            source_type: Type of data source (football_data, odds, etc.)
            primary_source: Name of the primary source
            backup_source: Name of the backup source
            reason: Reason for failover
            details: Additional details about the failover
        """
        self.source_type = source_type
        self.primary_source = primary_source
        self.backup_source = backup_source
        self.reason = reason
        self.details = details or {}
        self.timestamp = datetime.now()
        self.recovered = False
        self.recovery_timestamp = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation
        """
        return {
            "source_type": self.source_type,
            "primary_source": self.primary_source,
            "backup_source": self.backup_source,
            "reason": self.reason,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "recovered": self.recovered,
            "recovery_timestamp": self.recovery_timestamp.isoformat() if self.recovery_timestamp else None
        }
    
    def mark_recovered(self) -> None:
        """Mark the event as recovered."""
        self.recovered = True
        self.recovery_timestamp = datetime.now()
    
    def __str__(self) -> str:
        """String representation of the failover event."""
        status = "RECOVERED" if self.recovered else "ACTIVE"
        return f"Failover [{status}] {self.source_type}: {self.primary_source} → {self.backup_source} ({self.reason})"


class DataSourceFailoverManager:
    """Manager for handling data source failovers."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the failover manager.
        
        Args:
            config_path: Path to failover configuration file
        """
        self.config_path = config_path or os.path.join(
            Path(__file__).parent.parent, "config", "failover_config.json"
        )
        self.config = self._load_config()
        
        # Get other system components
        self.api_monitor = get_api_monitor()
        self.data_integrity_checker = get_data_integrity_checker()
        self.alert_system = get_alert_system()
        
        # Track failover events and active sources
        self.failover_events = []
        self.active_sources = {}  # Dict mapping source type to active source name
        
        # Initialize active sources to primary sources
        for source_type, sources in self.config["sources"].items():
            primary = sources.get("primary")
            if primary:
                self.active_sources[source_type] = primary
        
        logger.info("Data Source Failover Manager initialized")
    
    def _load_config(self) -> Dict:
        """Load failover configuration from file.
        
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            "sources": {
                "football_data": {
                    "primary": "football-data.org",
                    "backups": [
                        "api-football.com",
                        "sportmonks.com"
                    ],
                    "thresholds": {
                        "availability": 0.95,
                        "quality": 0.7,
                        "error_rate": 0.05
                    },
                    "retry_interval_minutes": 30
                },
                "odds": {
                    "primary": "oddsapi.com",
                    "backups": [
                        "betfair.com",
                        "pinnacle.com"
                    ],
                    "thresholds": {
                        "availability": 0.98,
                        "quality": 0.8,
                        "error_rate": 0.02
                    },
                    "retry_interval_minutes": 15
                },
                "stats": {
                    "primary": "understat.com",
                    "backups": [
                        "statsbomb.com",
                        "whoscored.com"
                    ],
                    "thresholds": {
                        "availability": 0.95,
                        "quality": 0.7,
                        "error_rate": 0.05
                    },
                    "retry_interval_minutes": 30
                },
                "weather": {
                    "primary": "openweathermap.org",
                    "backups": [
                        "accuweather.com",
                        "darksky.net"
                    ],
                    "thresholds": {
                        "availability": 0.95,
                        "quality": 0.7,
                        "error_rate": 0.05
                    },
                    "retry_interval_minutes": 15
                }
            },
            "general": {
                "max_consecutive_failures": 3,
                "health_check_interval_seconds": 60,
                "fallback_retry_interval_minutes": 30,
                "notify_on_failover": True,
                "notify_on_recovery": True
            },
            "api_source_mapping": {
                "football-data.org": "FOOTBALL_DATA",
                "understat.com": "UNDERSTAT",
                "oddsapi.com": "ODDS_API",
                "openweathermap.org": "WEATHER_API"
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
    
    def check_sources_health(self) -> Dict[str, Dict[str, SourceStatus]]:
        """Check health of all configured data sources.
        
        Returns:
            Dictionary mapping source types to dictionaries of source names and status
        """
        logger.debug("Checking health of all data sources...")
        
        health_status = {}
        
        # Get API monitor status
        api_status = self.api_monitor.get_all_sources_status()
        
        # Check each source type
        for source_type, sources in self.config["sources"].items():
            health_status[source_type] = {}
            
            # Check primary and all backup sources
            all_sources = [sources["primary"]] + sources.get("backups", [])
            thresholds = sources.get("thresholds", {})
            
            for source_name in all_sources:
                # Map to API monitor source if possible
                api_source_name = self.config.get("api_source_mapping", {}).get(source_name)
                
                if api_source_name and api_source_name in api_status:
                    api_info = api_status[api_source_name]
                    
                    # Default to available
                    status = SourceStatus.AVAILABLE
                    
                    # Check error rate for availability
                    if "api_stats" in api_info and "error_rate" in api_info["api_stats"]:
                        error_rate = api_info["api_stats"]["error_rate"]
                        
                        if error_rate > thresholds.get("error_rate", 0.05):
                            if error_rate > 0.5:  # More than 50% errors is considered unavailable
                                status = SourceStatus.UNAVAILABLE
                            else:
                                status = SourceStatus.DEGRADED
                    
                    # Check data quality
                    if status != SourceStatus.UNAVAILABLE and "quality" in api_info and "last_score" in api_info["quality"]:
                        quality_score = api_info["quality"].get("last_score")
                        
                        if quality_score is not None:
                            if quality_score < thresholds.get("quality", 0.7):
                                status = SourceStatus.DEGRADED
                else:
                    # No monitoring information available
                    status = SourceStatus.UNKNOWN
                
                health_status[source_type][source_name] = status
        
        return health_status
    
    def handle_failovers(self) -> List[FailoverEvent]:
        """Check for needed failovers and handle them.
        
        Returns:
            List of new failover events
        """
        logger.info("Checking for needed failovers...")
        
        # Get current health status
        health_status = self.check_sources_health()
        new_events = []
        
        # Check each source type
        for source_type, sources_status in health_status.items():
            source_config = self.config["sources"].get(source_type, {})
            primary_source = source_config.get("primary")
            backup_sources = source_config.get("backups", [])
            
            # Skip if no primary or backups configured
            if not primary_source or not backup_sources:
                continue
            
            # Get current active source for this type
            active_source = self.active_sources.get(source_type, primary_source)
            
            # Check if we need to failover from primary
            if active_source == primary_source:
                primary_status = sources_status.get(primary_source, SourceStatus.UNKNOWN)
                
                if primary_status in [SourceStatus.UNAVAILABLE, SourceStatus.DEGRADED]:
                    # Need to failover to a backup
                    for backup in backup_sources:
                        backup_status = sources_status.get(backup, SourceStatus.UNKNOWN)
                        
                        if backup_status == SourceStatus.AVAILABLE:
                            # Found a good backup, switch to it
                            self.active_sources[source_type] = backup
                            
                            event = FailoverEvent(
                                source_type=source_type,
                                primary_source=primary_source,
                                backup_source=backup,
                                reason=f"Primary source {primary_status.value}",
                                details={
                                    "primary_status": primary_status.value,
                                    "backup_status": backup_status.value
                                }
                            )
                            
                            self.failover_events.append(event)
                            new_events.append(event)
                            
                            logger.warning(f"Failover initiated: {source_type} from {primary_source} to {backup}")
                            
                            # Send alert if configured
                            if self.config["general"].get("notify_on_failover", True):
                                self._send_failover_notification(event)
                            
                            # Found a backup, no need to check others
                            break
            
            # Check if we should failback to primary
            elif active_source != primary_source:
                primary_status = sources_status.get(primary_source, SourceStatus.UNKNOWN)
                
                if primary_status == SourceStatus.AVAILABLE:
                    # Check retry interval before switching back
                    can_retry = self._check_retry_interval(source_type)
                    
                    if can_retry:
                        # Switch back to primary
                        self.active_sources[source_type] = primary_source
                        
                        # Find the corresponding failover event and mark as recovered
                        for event in self.failover_events:
                            if (event.source_type == source_type and 
                                event.primary_source == primary_source and 
                                event.backup_source == active_source and 
                                not event.recovered):
                                event.mark_recovered()
                                
                                # Send recovery notification if configured
                                if self.config["general"].get("notify_on_recovery", True):
                                    self._send_recovery_notification(event)
                                
                                logger.info(f"Recovered from failover: {source_type} back to {primary_source}")
        
        return new_events
    
    def _check_retry_interval(self, source_type: str) -> bool:
        """Check if retry interval has passed for a source type.
        
        Args:
            source_type: Type of data source
            
        Returns:
            True if retry interval has passed
        """
        # Get retry interval from config
        retry_minutes = self.config["sources"].get(source_type, {}).get(
            "retry_interval_minutes", 
            self.config["general"].get("fallback_retry_interval_minutes", 30)
        )
        
        # Find the most recent failover event for this source type
        for event in sorted(self.failover_events, key=lambda e: e.timestamp, reverse=True):
            if event.source_type == source_type and not event.recovered:
                # Check if retry interval has passed
                elapsed = datetime.now() - event.timestamp
                return elapsed.total_seconds() >= (retry_minutes * 60)
        
        # No failover event found, allow retry
        return True
    
    def _send_failover_notification(self, event: FailoverEvent) -> None:
        """Send notification about a failover event.
        
        Args:
            event: Failover event
        """
        if not self.alert_system:
            logger.warning("Alert system not available, skipping failover notification")
            return
        
        alert = DataSourceAlert(
            source=event.source_type,
            severity=AlertSeverity.WARNING,
            message=f"Failover initiated: {event.source_type} from {event.primary_source} to {event.backup_source}",
            details={
                "event_type": "failover",
                "primary_source": event.primary_source,
                "backup_source": event.backup_source,
                "reason": event.reason,
                "details": event.details
            }
        )
        
        # Use the alert system to process this alert
        self.alert_system._process_alert(alert)
    
    def _send_recovery_notification(self, event: FailoverEvent) -> None:
        """Send notification about recovery from a failover.
        
        Args:
            event: Recovered failover event
        """
        if not self.alert_system:
            logger.warning("Alert system not available, skipping recovery notification")
            return
        
        # Calculate downtime duration
        downtime = event.recovery_timestamp - event.timestamp
        hours, remainder = divmod(downtime.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        downtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        alert = DataSourceAlert(
            source=event.source_type,
            severity=AlertSeverity.INFO,
            message=f"Recovered from failover: {event.source_type} back to {event.primary_source}",
            details={
                "event_type": "recovery",
                "primary_source": event.primary_source,
                "backup_source": event.backup_source,
                "downtime": downtime_str,
                "downtime_seconds": downtime.total_seconds()
            }
        )
        
        # Use the alert system to process this alert
        self.alert_system._process_alert(alert)
    
    def get_active_source(self, source_type: str) -> str:
        """Get the currently active source for a source type.
        
        Args:
            source_type: Type of data source
            
        Returns:
            Name of active source
        """
        return self.active_sources.get(
            source_type,
            self.config["sources"].get(source_type, {}).get("primary", "unknown")
        )
    
    def get_failover_events(self, active_only: bool = False) -> List[FailoverEvent]:
        """Get list of failover events.
        
        Args:
            active_only: If True, only return active (not recovered) events
            
        Returns:
            List of failover events
        """
        if active_only:
            return [event for event in self.failover_events if not event.recovered]
        else:
            return self.failover_events
    
    def export_events_to_json(self, filepath: str) -> None:
        """Export failover events to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        events_dict = [e.to_dict() for e in self.failover_events]
        
        with open(filepath, 'w') as f:
            json.dump(events_dict, f, indent=2)
        
        logger.info(f"Exported {len(events_dict)} failover events to {filepath}")
    
    def test_failover(self, source_type: str) -> Optional[FailoverEvent]:
        """Test failover for a specific source type.
        
        Args:
            source_type: Type of data source to test failover for
            
        Returns:
            Failover event if successful, None otherwise
        """
        source_config = self.config["sources"].get(source_type)
        if not source_config:
            logger.error(f"Source type not found in configuration: {source_type}")
            return None
        
        primary = source_config.get("primary")
        backups = source_config.get("backups", [])
        
        if not primary or not backups:
            logger.error(f"Source {source_type} does not have primary or backups configured")
            return None
        
        backup = backups[0]
        
        # Create a test failover event
        event = FailoverEvent(
            source_type=source_type,
            primary_source=primary,
            backup_source=backup,
            reason="Test failover",
            details={
                "test": True,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Update active source
        self.active_sources[source_type] = backup
        
        # Add to events
        self.failover_events.append(event)
        
        # Send notification if configured
        if self.config["general"].get("notify_on_failover", True):
            self._send_failover_notification(event)
        
        logger.info(f"Test failover initiated: {source_type} from {primary} to {backup}")
        
        return event


# Singleton instance
_failover_manager_instance = None

def get_failover_manager() -> DataSourceFailoverManager:
    """Get the global failover manager instance.
    
    Returns:
        DataSourceFailoverManager instance
    """
    global _failover_manager_instance
    if _failover_manager_instance is None:
        _failover_manager_instance = DataSourceFailoverManager()
    return _failover_manager_instance


def main():
    """Main entry point for the failover manager."""
    parser = argparse.ArgumentParser(description="Data Source Failover Manager")
    parser.add_argument("--config", help="Path to failover configuration file")
    parser.add_argument("--test-failover", help="Test failover for a specific source type")
    parser.add_argument("--check-health", action="store_true", help="Check health of all data sources")
    parser.add_argument("--handle-failovers", action="store_true", help="Check and handle needed failovers")
    parser.add_argument("--monitor", action="store_true", help="Start continuous failover monitoring")
    parser.add_argument("--export", help="Export failover events to JSON file")
    parser.add_argument("--active-sources", action="store_true", help="Show currently active sources")
    
    args = parser.parse_args()
    
    # Create failover manager
    failover_manager = DataSourceFailoverManager(config_path=args.config)
    
    if args.test_failover:
        # Test failover for a specific source type
        source_type = args.test_failover
        logger.info(f"Testing failover for source type: {source_type}")
        
        event = failover_manager.test_failover(source_type)
        if event:
            logger.info(f"Test failover successful: {event}")
        else:
            logger.error(f"Test failover failed for source type: {source_type}")
    
    elif args.check_health:
        # Check health of all data sources
        logger.info("Checking health of all data sources")
        
        health_status = failover_manager.check_sources_health()
        for source_type, sources in health_status.items():
            logger.info(f"{source_type}:")
            for source, status in sources.items():
                logger.info(f"  {source}: {status.value}")
    
    elif args.handle_failovers:
        # Check and handle needed failovers
        logger.info("Checking and handling needed failovers")
        
        events = failover_manager.handle_failovers()
        if events:
            logger.info(f"Handled {len(events)} new failover events")
            for event in events:
                logger.info(f"  {event}")
        else:
            logger.info("No new failover events needed")
    
    elif args.monitor:
        # Start continuous monitoring
        logger.info("Starting continuous failover monitoring")
        
        interval_seconds = failover_manager.config["general"].get("health_check_interval_seconds", 60)
        
        try:
            while True:
                failover_manager.handle_failovers()
                time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    
    elif args.export:
        # Export events to JSON
        logger.info(f"Exporting failover events to {args.export}")
        
        # Generate a test event if none exist
        if not failover_manager.failover_events:
            failover_manager.test_failover(list(failover_manager.config["sources"].keys())[0])
        
        failover_manager.export_events_to_json(args.export)
    
    elif args.active_sources:
        # Show currently active sources
        logger.info("Currently active sources:")
        
        for source_type in failover_manager.config["sources"]:
            active = failover_manager.get_active_source(source_type)
            primary = failover_manager.config["sources"][source_type].get("primary")
            
            status = "PRIMARY" if active == primary else "BACKUP"
            logger.info(f"{source_type}: {active} ({status})")
    
    else:
        # No specific action requested
        logger.info("No action specified. Use --check-health, --handle-failovers, or --monitor")
        parser.print_help()


if __name__ == "__main__":
    main()
