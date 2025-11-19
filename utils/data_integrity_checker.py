#!/usr/bin/env python3
"""
Automated Data Integrity Checker for GoalDiggers

This module provides automated data integrity checking functionality for
the GoalDiggers platform. It includes:

1. Regular validation of database integrity
2. Detection of inconsistent or corrupt data
3. Anomaly detection in sports statistics
4. Validation of match data consistency
5. Regular data schema validation

Usage:
    Run directly: python utils/data_integrity_checker.py
    Import: from utils.data_integrity_checker import DataIntegrityChecker
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text

from database.db_manager import DatabaseManager
from utils.api_monitor import ApiMonitor, get_api_monitor

# Configure logging
logger = logging.getLogger(__name__)


class CheckSeverity(Enum):
    """Enumeration for check severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CheckResult:
    """Class to represent a data integrity check result."""
    
    def __init__(
        self, 
        check_name: str, 
        passed: bool, 
        severity: CheckSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize a check result.
        
        Args:
            check_name: Name of the check
            passed: Whether the check passed
            severity: Severity level of the check
            message: Result message
            details: Additional details about the result
        """
        self.check_name = check_name
        self.passed = passed
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.
        
        Returns:
            Dictionary representation of the check result
        """
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation of the check result."""
        status = "PASSED" if self.passed else "FAILED"
        return f"{self.check_name} ({self.severity.value}): {status} - {self.message}"


class DataIntegrityChecker:
    """Automated data integrity checker for all platform data."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize data integrity checker.
        
        Args:
            db_manager: Database manager instance (optional)
        """
        self.db_manager = db_manager or DatabaseManager()
        self.api_monitor = get_api_monitor()
        self.check_results = []
        self.scheduled_checks = {}
        self.running = False
        self.last_run = {}
        
        # Load schema validation rules
        self.validation_rules = self._load_validation_rules()
        
        logger.info("Data Integrity Checker initialized")
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from configuration.
        
        Returns:
            Dictionary of validation rules
        """
        # Default rules
        default_rules = {
            "matches": {
                "required_fields": ["id", "home_team_id", "away_team_id", "match_date", "competition"],
                "numeric_fields": ["home_score", "away_score"],
                "date_fields": ["match_date"],
                "string_fields": ["home_team_id", "away_team_id", "competition"],
                "value_constraints": {
                    "home_score": {"min": 0, "max": 20},
                    "away_score": {"min": 0, "max": 20}
                }
            },
            "teams": {
                "required_fields": ["id", "name", "league_id"],
                "string_fields": ["name", "league_id"],
                # Removed unique_fields constraint - team names can legitimately duplicate
                # across different data sources (same team, different IDs)
            },
            "odds": {
                "required_fields": ["match_id", "home_win", "draw", "away_win"],
                "numeric_fields": ["home_win", "draw", "away_win"],
                "value_constraints": {
                    "home_win": {"min": 1.01, "max": 30.0},
                    "draw": {"min": 1.01, "max": 30.0},
                    "away_win": {"min": 1.01, "max": 30.0}
                }
            }
        }
        
        # Try to load from config file
        try:
            config_path = os.path.join(os.path.dirname(__file__), "../config/validation_rules.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    loaded_rules = json.load(f)
                
                # Merge with default rules
                for entity, rules in loaded_rules.items():
                    if entity in default_rules:
                        default_rules[entity].update(rules)
                    else:
                        default_rules[entity] = rules
                
                logger.info(f"Loaded validation rules from {config_path}")
        except Exception as e:
            logger.warning(f"Error loading validation rules from file: {e}")
        
        return default_rules
    
    def start_scheduled_checks(self) -> None:
        """Start scheduled integrity checks.
        
        This starts a background thread that runs scheduled checks.
        """
        import threading
        
        self.running = True
        
        def run_checks():
            while self.running:
                now = datetime.now()
                
                # Run any scheduled checks
                for check_name, schedule in list(self.scheduled_checks.items()):
                    interval = schedule.get('interval', 3600)  # Default 1 hour
                    last_run = self.last_run.get(check_name)
                    
                    if last_run is None or (now - last_run).total_seconds() >= interval:
                        try:
                            # Run the check
                            check_func = schedule.get('check_func')
                            if check_func:
                                logger.info(f"Running scheduled check: {check_name}")
                                result = check_func()
                                self.check_results.append(result)
                                self.last_run[check_name] = now
                                
                                # Check if we need to alert on failure
                                if schedule.get('alert_on_failure', True) and not result.passed:
                                    alert_type = f"data_integrity_{result.severity.value}"
                                    self.api_monitor.trigger_alert(
                                        alert_type,
                                        f"Data integrity check failed: {result.check_name}",
                                        result.to_dict()
                                    )
                        except Exception as e:
                            logger.error(f"Error running scheduled check {check_name}: {e}")
                
                # Sleep briefly to avoid high CPU usage
                time.sleep(1)
        
        # Start background thread
        check_thread = threading.Thread(target=run_checks, daemon=True)
        check_thread.start()
        
        logger.info("Scheduled data integrity checks started")
    
    def stop_scheduled_checks(self) -> None:
        """Stop scheduled integrity checks."""
        self.running = False
        logger.info("Scheduled data integrity checks stopped")
    
    def schedule_check(
        self, 
        check_name: str, 
        check_func: callable, 
        interval: int = 3600,
        alert_on_failure: bool = True
    ) -> None:
        """Schedule a regular integrity check.
        
        Args:
            check_name: Name of the check
            check_func: Function that performs the check (should return a CheckResult)
            interval: Check interval in seconds (default: 1 hour)
            alert_on_failure: Whether to alert when check fails
        """
        self.scheduled_checks[check_name] = {
            'check_func': check_func,
            'interval': interval,
            'alert_on_failure': alert_on_failure
        }
        logger.info(f"Scheduled check '{check_name}' to run every {interval} seconds")
    
    def validate_database_schema(self) -> CheckResult:
        """Validate database schema against expected structure.
        
        Returns:
            Check result for schema validation
        """
        try:
            # Expected tables and columns - using proper plural table names
            expected_tables = {
                "matches": {"id", "home_team_id", "away_team_id", "match_date", "competition", "home_score", "away_score"},
                "teams": {"id", "name", "league_id"},
                "leagues": {"id", "name", "country"},
                "odds": {"id", "match_id", "bookmaker", "home_win", "draw", "away_win", "timestamp"},
                "predictions": {"id", "match_id", "home_win_prob", "draw_prob", "away_win_prob", "confidence", "created_at"}
            }
            
            # Connect to database and inspect
            with self.db_manager.session_scope() as session:
                inspector = inspect(session.bind)
                actual_tables = inspector.get_table_names()
                
                # Check if expected tables exist
                missing_tables = [table for table in expected_tables.keys() if table not in actual_tables]
                
                # Check columns for each table
                column_issues = {}
                for table in expected_tables:
                    if table in actual_tables:
                        actual_columns = {col["name"] for col in inspector.get_columns(table)}
                        missing_columns = [col for col in expected_tables[table] if col not in actual_columns]
                        if missing_columns:
                            column_issues[table] = missing_columns
            
            if missing_tables or column_issues:
                details = {
                    "missing_tables": missing_tables,
                    "column_issues": column_issues
                }
                
                message = f"Schema validation issues found: {len(missing_tables)} missing tables, "
                message += f"{len(column_issues)} tables with column issues"
                
                return CheckResult(
                    check_name="database_schema_validation",
                    passed=False,
                    severity=CheckSeverity.ERROR,
                    message=message,
                    details=details
                )
            else:
                return CheckResult(
                    check_name="database_schema_validation",
                    passed=True,
                    severity=CheckSeverity.INFO,
                    message="Database schema validated successfully",
                    details={"tables_validated": list(expected_tables.keys())}
                )
                
        except Exception as e:
            logger.exception(f"Error validating database schema: {e}")
            return CheckResult(
                check_name="database_schema_validation",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"Error validating database schema: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_data_consistency(self, table_name: str) -> CheckResult:
        """Check data consistency for a specific table.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            Check result for data consistency
        """
        try:
            # Get validation rules for this entity
            rules = self.validation_rules.get(table_name)
            if not rules:
                return CheckResult(
                    check_name=f"data_consistency_{table_name}",
                    passed=True,
                    severity=CheckSeverity.INFO,
                    message=f"No validation rules defined for {table_name}",
                    details={}
                )
            
            # Query data from the table
            with self.db_manager.session_scope() as session:
                try:
                    # Get all data from the table
                    result = session.execute(text(f"SELECT * FROM {table_name}"))
                    columns = result.keys()
                    data = [{col: row[i] for i, col in enumerate(columns)} for row in result]
                    
                    if not data:
                        return CheckResult(
                            check_name=f"data_consistency_{table_name}",
                            passed=True,
                            severity=CheckSeverity.INFO,
                            message=f"No data found in table {table_name}",
                            details={"row_count": 0}
                        )
                except Exception as table_error:
                    # Return a specific error for this table
                    logger.error(f"Error accessing table {table_name}: {table_error}")
                    return CheckResult(
                        check_name=f"data_consistency_{table_name}",
                        passed=False,
                        severity=CheckSeverity.ERROR,
                        message=f"Error checking data consistency for {table_name}: {str(table_error)}",
                        details={"error": str(table_error)}
                    )
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(data)
            
            # Track issues
            issues = {
                "missing_required_fields": [],
                "invalid_numeric_values": [],
                "invalid_date_values": [],
                "invalid_string_values": [],
                "constraint_violations": [],
                "unique_violations": []
            }
            
            # Check required fields
            for field in rules.get("required_fields", []):
                if field not in df.columns:
                    issues["missing_required_fields"].append(field)
                elif df[field].isnull().any():
                    null_count = df[field].isnull().sum()
                    issues["missing_required_fields"].append(
                        f"{field}: {null_count} null values"
                    )
            
            # Check numeric fields
            for field in rules.get("numeric_fields", []):
                if field in df.columns:
                    # Check if values are numeric
                    try:
                        df[field].astype(float)
                    except:
                        issues["invalid_numeric_values"].append(field)
                    
                    # Check constraints
                    constraints = rules.get("value_constraints", {}).get(field, {})
                    if "min" in constraints and (df[field] < constraints["min"]).any():
                        below_min = (df[field] < constraints["min"]).sum()
                        issues["constraint_violations"].append(
                            f"{field}: {below_min} values below minimum ({constraints['min']})"
                        )
                    if "max" in constraints and (df[field] > constraints["max"]).any():
                        above_max = (df[field] > constraints["max"]).sum()
                        issues["constraint_violations"].append(
                            f"{field}: {above_max} values above maximum ({constraints['max']})"
                        )
            
            # Check date fields
            for field in rules.get("date_fields", []):
                if field in df.columns:
                    # Try to convert to datetime with mixed format support
                    try:
                        pd.to_datetime(df[field], format='mixed', errors='coerce')
                        # Check if any dates failed to parse
                        temp_parsed = pd.to_datetime(df[field], format='mixed', errors='coerce')
                        failed_parses = temp_parsed.isna().sum()
                        if failed_parses > 0:
                            issues["invalid_date_values"].append(f"{field}: {failed_parses} dates failed to parse")
                    except Exception as e:
                        issues["invalid_date_values"].append(f"{field}: parsing error - {str(e)}")
            
            # Check string fields
            for field in rules.get("string_fields", []):
                if field in df.columns:
                    non_string = df[~df[field].apply(lambda x: isinstance(x, str) or pd.isna(x))].shape[0]
                    if non_string > 0:
                        issues["invalid_string_values"].append(f"{field}: {non_string} non-string values")
            
            # Check unique constraints
            for field in rules.get("unique_fields", []):
                if field in df.columns:
                    duplicates = df[df.duplicated(subset=[field], keep=False)].shape[0]
                    if duplicates > 0:
                        issues["unique_violations"].append(f"{field}: {duplicates} duplicate values")
            
            # Remove empty issue categories
            issues = {k: v for k, v in issues.items() if v}
            
            if issues:
                return CheckResult(
                    check_name=f"data_consistency_{table_name}",
                    passed=False,
                    severity=CheckSeverity.WARNING,
                    message=f"Data consistency issues found in {table_name}",
                    details={
                        "issues": issues,
                        "row_count": len(df),
                        "column_count": len(df.columns)
                    }
                )
            else:
                return CheckResult(
                    check_name=f"data_consistency_{table_name}",
                    passed=True,
                    severity=CheckSeverity.INFO,
                    message=f"Data consistency validated for {table_name}",
                    details={
                        "row_count": len(df),
                        "column_count": len(df.columns)
                    }
                )
                
        except Exception as e:
            logger.exception(f"Error checking data consistency for {table_name}: {e}")
            return CheckResult(
                check_name=f"data_consistency_{table_name}",
                passed=False,
                severity=CheckSeverity.ERROR,
                message=f"Error checking data consistency for {table_name}: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_match_data_anomalies(self) -> CheckResult:
        """Check for anomalies in match data.
        
        Returns:
            Check result for match data anomalies
        """
        try:
            with self.db_manager.session_scope() as session:
                # Query recent matches
                query = """
                SELECT m.id, m.match_date, m.home_team_id, m.away_team_id, m.home_score, m.away_score,
                       ht.name as home_team_name, at.name as away_team_name
                FROM matches m
                JOIN teams ht ON m.home_team_id = ht.id
                JOIN teams at ON m.away_team_id = at.id
                WHERE m.match_date >= :cutoff_date
                """
                
                cutoff_date = datetime.now() - timedelta(days=30)  # Last 30 days
                result = session.execute(text(query), {"cutoff_date": cutoff_date})
                
                columns = result.keys()
                matches = [{col: row[i] for i, col in enumerate(columns)} for row in result]
                
                if not matches:
                    return CheckResult(
                        check_name="match_data_anomalies",
                        passed=True,
                        severity=CheckSeverity.INFO,
                        message="No recent match data found to check for anomalies",
                        details={"period": "last 30 days"}
                    )
            
            # Convert to DataFrame
            matches_df = pd.DataFrame(matches)
            anomalies = []
            
            # Check for unrealistic scores
            if 'home_score' in matches_df.columns and 'away_score' in matches_df.columns:
                high_score_matches = matches_df[
                    (matches_df['home_score'] > 7) | (matches_df['away_score'] > 7)
                ]
                
                if not high_score_matches.empty:
                    for _, match in high_score_matches.iterrows():
                        anomalies.append({
                            "type": "high_score",
                            "match_id": match['id'],
                            "teams": f"{match['home_team_name']} vs {match['away_team_name']}",
                            "score": f"{match['home_score']} - {match['away_score']}",
                            "date": match['match_date']
                        })
            
            # Check for duplicate matches (same teams on same day)
            if not matches_df.empty and 'home_team_id' in matches_df.columns and 'away_team_id' in matches_df.columns:
                matches_df['match_date_day'] = pd.to_datetime(matches_df['match_date'], format='ISO8601').dt.date
                duplicate_matches = matches_df[matches_df.duplicated(
                    subset=['home_team_id', 'away_team_id', 'match_date_day'], 
                    keep=False
                )]
                
                if not duplicate_matches.empty:
                    for _, match in duplicate_matches.iterrows():
                        anomalies.append({
                            "type": "duplicate_match",
                            "match_id": match['id'],
                            "teams": f"{match['home_team_name']} vs {match['away_team_name']}",
                            "date": match['match_date']
                        })
            
            # Check for future matches with scores
            if 'match_date' in matches_df.columns and 'home_score' in matches_df.columns:
                matches_df['match_date'] = pd.to_datetime(matches_df['match_date'], format='ISO8601')
                future_with_score = matches_df[
                    (matches_df['match_date'] > datetime.now()) & 
                    (~pd.isna(matches_df['home_score']) | ~pd.isna(matches_df['away_score']))
                ]
                
                if not future_with_score.empty:
                    for _, match in future_with_score.iterrows():
                        anomalies.append({
                            "type": "future_match_with_score",
                            "match_id": match['id'],
                            "teams": f"{match['home_team_name']} vs {match['away_team_name']}",
                            "score": f"{match['home_score']} - {match['away_score']}",
                            "date": match['match_date']
                        })
            
            # Return results
            if anomalies:
                return CheckResult(
                    check_name="match_data_anomalies",
                    passed=False,
                    severity=CheckSeverity.WARNING,
                    message=f"Found {len(anomalies)} anomalies in match data",
                    details={
                        "anomalies": anomalies,
                        "total_matches_checked": len(matches_df)
                    }
                )
            else:
                return CheckResult(
                    check_name="match_data_anomalies",
                    passed=True,
                    severity=CheckSeverity.INFO,
                    message="No anomalies found in match data",
                    details={"total_matches_checked": len(matches_df)}
                )
                
        except Exception as e:
            logger.exception(f"Error checking match data anomalies: {e}")
            return CheckResult(
                check_name="match_data_anomalies",
                passed=False,
                severity=CheckSeverity.ERROR,
                message=f"Error checking match data anomalies: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_betting_odds_anomalies(self) -> CheckResult:
        """Check for anomalies in betting odds data.
        
        Returns:
            Check result for betting odds anomalies
        """
        try:
            with self.db_manager.session_scope() as session:
                # Query recent odds
                query = """
                SELECT o.id, o.match_id, o.bookmaker, o.home_win, o.draw, o.away_win, o.timestamp,
                       m.match_date, ht.name as home_team, at.name as away_team
                FROM odds o
                JOIN matches m ON o.match_id = m.id
                JOIN teams ht ON m.home_team_id = ht.id
                JOIN teams at ON m.away_team_id = at.id
                WHERE o.timestamp >= :cutoff_date
                """
                
                cutoff_date = datetime.now() - timedelta(days=7)  # Last 7 days
                result = session.execute(text(query), {"cutoff_date": cutoff_date})
                
                columns = result.keys()
                odds_data = [{col: row[i] for i, col in enumerate(columns)} for row in result]
                
                if not odds_data:
                    return CheckResult(
                        check_name="betting_odds_anomalies",
                        passed=True,
                        severity=CheckSeverity.INFO,
                        message="No recent odds data found to check for anomalies",
                        details={"period": "last 7 days"}
                    )
            
            # Convert to DataFrame
            odds_df = pd.DataFrame(odds_data)
            anomalies = []
            
            # Check for unrealistic odds values
            unrealistic_odds = odds_df[
                (odds_df['home_win'] < 1.01) | (odds_df['home_win'] > 30) |
                (odds_df['draw'] < 1.01) | (odds_df['draw'] > 30) |
                (odds_df['away_win'] < 1.01) | (odds_df['away_win'] > 30)
            ]
            
            if not unrealistic_odds.empty:
                for _, odds in unrealistic_odds.iterrows():
                    anomalies.append({
                        "type": "unrealistic_odds",
                        "match_id": odds['match_id'],
                        "bookmaker": odds['bookmaker'],
                        "teams": f"{odds['home_team']} vs {odds['away_team']}",
                        "odds": f"H: {odds['home_win']}, D: {odds['draw']}, A: {odds['away_win']}",
                        "timestamp": odds['timestamp']
                    })
            
            # Check for arbitrage opportunities
            # Group by match_id to get all bookmakers for a match
            for match_id, group in odds_df.groupby('match_id'):
                # Calculate implied probabilities
                min_home = group['home_win'].min()
                min_draw = group['draw'].min()
                min_away = group['away_win'].min()
                
                # Calculate arbitrage sum (sum of best implied probabilities)
                arb_sum = (1 / min_home) + (1 / min_draw) + (1 / min_away)
                
                # If less than 1, it's an arbitrage opportunity
                if arb_sum < 0.98:  # With a small buffer for margin of error
                    match_info = group.iloc[0]
                    anomalies.append({
                        "type": "arbitrage_opportunity",
                        "match_id": match_id,
                        "teams": f"{match_info['home_team']} vs {match_info['away_team']}",
                        "best_odds": f"H: {min_home} ({match_info['bookmaker']}), D: {min_draw}, A: {min_away}",
                        "arbitrage_sum": arb_sum,
                        "profit_potential": f"{((1/arb_sum) - 1) * 100:.2f}%"
                    })
            
            # Check for drastic odds changes
            # Iterate through each match and bookmaker
            for (match_id, bookmaker), group in odds_df.groupby(['match_id', 'bookmaker']):
                # Sort by timestamp
                group = group.sort_values('timestamp')
                
                # Check if we have multiple entries
                if len(group) > 1:
                    # Calculate max change for each market
                    home_win_change = group['home_win'].pct_change().abs().max()
                    draw_change = group['draw'].pct_change().abs().max()
                    away_win_change = group['away_win'].pct_change().abs().max()
                    
                    # Flag drastic changes (more than 30%)
                    if home_win_change > 0.3 or draw_change > 0.3 or away_win_change > 0.3:
                        match_info = group.iloc[0]
                        anomalies.append({
                            "type": "drastic_odds_change",
                            "match_id": match_id,
                            "bookmaker": bookmaker,
                            "teams": f"{match_info['home_team']} vs {match_info['away_team']}",
                            "max_change": f"{max(home_win_change, draw_change, away_win_change)*100:.2f}%",
                            "timestamp": match_info['timestamp']
                        })
            
            # Return results
            if anomalies:
                return CheckResult(
                    check_name="betting_odds_anomalies",
                    passed=False,
                    severity=CheckSeverity.WARNING,
                    message=f"Found {len(anomalies)} anomalies in odds data",
                    details={
                        "anomalies": anomalies,
                        "total_odds_checked": len(odds_df)
                    }
                )
            else:
                return CheckResult(
                    check_name="betting_odds_anomalies",
                    passed=True,
                    severity=CheckSeverity.INFO,
                    message="No anomalies found in odds data",
                    details={"total_odds_checked": len(odds_df)}
                )
                
        except Exception as e:
            logger.exception(f"Error checking betting odds anomalies: {e}")
            return CheckResult(
                check_name="betting_odds_anomalies",
                passed=False,
                severity=CheckSeverity.ERROR,
                message=f"Error checking betting odds anomalies: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_prediction_model_consistency(self) -> CheckResult:
        """Check for consistency in prediction model outputs.
        
        Returns:
            Check result for prediction consistency
        """
        try:
            with self.db_manager.session_scope() as session:
                # Query recent predictions
                query = """
                SELECT p.id, p.match_id, p.home_win_prob, p.draw_prob, p.away_win_prob, 
                       p.confidence, p.created_at,
                       m.match_date, ht.name as home_team, at.name as away_team
                FROM predictions p
                JOIN matches m ON p.match_id = m.id
                JOIN teams ht ON m.home_team_id = ht.id
                JOIN teams at ON m.away_team_id = at.id
                WHERE p.created_at >= :cutoff_date
                """
                
                cutoff_date = datetime.now() - timedelta(days=7)  # Last 7 days
                result = session.execute(text(query), {"cutoff_date": cutoff_date})
                
                columns = result.keys()
                predictions = [{col: row[i] for i, col in enumerate(columns)} for row in result]
                
                if not predictions:
                    return CheckResult(
                        check_name="prediction_model_consistency",
                        passed=True,
                        severity=CheckSeverity.INFO,
                        message="No recent predictions found to check for consistency",
                        details={"period": "last 7 days"}
                    )
            
            # Convert to DataFrame
            predictions_df = pd.DataFrame(predictions)
            issues = []
            
            # Check for probability sum issues (should be close to 1)
            predictions_df['prob_sum'] = (
                predictions_df['home_win_prob'] + 
                predictions_df['draw_prob'] + 
                predictions_df['away_win_prob']
            )
            
            invalid_prob_sum = predictions_df[
                (predictions_df['prob_sum'] < 0.97) | 
                (predictions_df['prob_sum'] > 1.03)
            ]
            
            if not invalid_prob_sum.empty:
                for _, pred in invalid_prob_sum.iterrows():
                    issues.append({
                        "type": "invalid_probability_sum",
                        "match_id": pred['match_id'],
                        "teams": f"{pred['home_team']} vs {pred['away_team']}",
                        "probabilities": f"H: {pred['home_win_prob']:.2f}, D: {pred['draw_prob']:.2f}, A: {pred['away_win_prob']:.2f}",
                        "sum": pred['prob_sum'],
                        "created_at": pred['created_at']
                    })
            
            # Check for extreme probability values
            extreme_probs = predictions_df[
                (predictions_df['home_win_prob'] > 0.95) |
                (predictions_df['draw_prob'] > 0.70) |  # Draw probabilities are typically lower
                (predictions_df['away_win_prob'] > 0.95)
            ]
            
            if not extreme_probs.empty:
                for _, pred in extreme_probs.iterrows():
                    issues.append({
                        "type": "extreme_probability",
                        "match_id": pred['match_id'],
                        "teams": f"{pred['home_team']} vs {pred['away_team']}",
                        "probabilities": f"H: {pred['home_win_prob']:.2f}, D: {pred['draw_prob']:.2f}, A: {pred['away_win_prob']:.2f}",
                        "created_at": pred['created_at']
                    })
            
            # Check for inconsistent predictions for the same match
            for match_id, group in predictions_df.groupby('match_id'):
                if len(group) > 1:
                    # Calculate standard deviation of predictions
                    home_std = group['home_win_prob'].std()
                    draw_std = group['draw_prob'].std()
                    away_std = group['away_win_prob'].std()
                    
                    # Flag high variance predictions
                    if home_std > 0.15 or draw_std > 0.15 or away_std > 0.15:
                        match_info = group.iloc[0]
                        issues.append({
                            "type": "inconsistent_predictions",
                            "match_id": match_id,
                            "teams": f"{match_info['home_team']} vs {match_info['away_team']}",
                            "std_dev": f"H: {home_std:.3f}, D: {draw_std:.3f}, A: {away_std:.3f}",
                            "prediction_count": len(group),
                            "match_date": match_info['match_date']
                        })
            
            # Return results
            if issues:
                return CheckResult(
                    check_name="prediction_model_consistency",
                    passed=False,
                    severity=CheckSeverity.WARNING,
                    message=f"Found {len(issues)} consistency issues in prediction data",
                    details={
                        "issues": issues,
                        "total_predictions_checked": len(predictions_df)
                    }
                )
            else:
                return CheckResult(
                    check_name="prediction_model_consistency",
                    passed=True,
                    severity=CheckSeverity.INFO,
                    message="No consistency issues found in prediction data",
                    details={"total_predictions_checked": len(predictions_df)}
                )
                
        except Exception as e:
            logger.exception(f"Error checking prediction model consistency: {e}")
            return CheckResult(
                check_name="prediction_model_consistency",
                passed=False,
                severity=CheckSeverity.ERROR,
                message=f"Error checking prediction model consistency: {str(e)}",
                details={"error": str(e)}
            )
    
    def run_integrity_checks(self) -> Dict[str, Any]:
        """Run all integrity checks and return results in a format for the health check system.
        
        Returns:
            Dictionary with check results formatted for health check system
        """
        # Run all checks
        check_results = self.run_all_checks()
        
        # Format results for health check system
        formatted_results = {}
        for result in check_results:
            formatted_results[result.check_name] = {
                "passed": result.passed,
                "message": result.message,
                "severity": result.severity.value,
                "details": result.details
            }
        
        return formatted_results
    
    def run_all_checks(self) -> List[CheckResult]:
        """Run all available integrity checks.
        
        Returns:
            List of check results
        """
        results = []
        
        # Run schema validation
        results.append(self.validate_database_schema())
        
        # Run data consistency checks for each entity
        for entity in self.validation_rules.keys():
            results.append(self.check_data_consistency(entity))
        
        # Run anomaly checks
        results.append(self.check_match_data_anomalies())
        results.append(self.check_betting_odds_anomalies())
        results.append(self.check_prediction_model_consistency())
        
        # Store results
        self.check_results.extend(results)
        
        return results
    
    def get_latest_results(self, limit: int = 20) -> List[CheckResult]:
        """Get the most recent check results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of recent check results
        """
        # Sort by timestamp, most recent first
        sorted_results = sorted(
            self.check_results, 
            key=lambda r: r.timestamp, 
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def export_results_to_json(self, filepath: str) -> None:
        """Export check results to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        results_dict = [r.to_dict() for r in self.check_results]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        logger.info(f"Exported {len(results_dict)} check results to {filepath}")


# Singleton instance
_data_integrity_checker_instance = None

def get_data_integrity_checker() -> DataIntegrityChecker:
    """Get the global data integrity checker instance.
    
    Returns:
        DataIntegrityChecker instance
    """
    global _data_integrity_checker_instance
    if _data_integrity_checker_instance is None:
        _data_integrity_checker_instance = DataIntegrityChecker()
    return _data_integrity_checker_instance


# If running as a script
if __name__ == "__main__":
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from utils.logging_config import configure_logging  # type: ignore
            configure_logging()
        except Exception:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    
    # Get checker instance
    checker = get_data_integrity_checker()
    
    # Run all checks
    logger.info("Running all data integrity checks...")
    results = checker.run_all_checks()
    
    # Print results
    for result in results:
        status = "PASSED" if result.passed else "FAILED"
        logger.info(f"{result.check_name} [{result.severity.value.upper()}]: {status}")
        logger.info(f"  {result.message}")
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    logger.info(f"Summary: {passed}/{len(results)} checks passed")
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = f"data_integrity_report_{timestamp}.json"
    checker.export_results_to_json(export_path)
    logger.info(f"Report exported to {export_path}")
