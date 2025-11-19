import json
import logging
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import requests  # Added for robust HTTPS handling
from pydantic import BaseModel, ValidationError

from dashboard.error_log import log_error  # Import log_error

logger = logging.getLogger(__name__)


class MatchValidator(BaseModel):
    teams: List[str]
    odds: List[float]
    timestamp: int

    class Config:
        extra = "ignore"


class AdvancedDataValidator:
    def __init__(self):
        self.reference_odds = self._load_reference_data()
        self.session = requests.Session()  # Use a session for connection reuse
        self.session.headers.update({"X-Auth-Token": os.getenv("FOOTBALL_DATA_KEY")})
        # Initialize report counters
        self.report = {
            "processed_count": 0,
            "schema_failures": 0,
            "odds_failures": 0,
            "passed_count": 0,
            "start_time": datetime.now(),
            "end_time": None,
            "errors": [] # Store specific validation errors
        }

    def _load_reference_data(self) -> dict:
        """Load benchmark odds from trusted source with certificate verification"""
        try:
            response = self.session.get(
                "https://api.football-data.org/v4/matches", timeout=10
            )
            response.raise_for_status()  # Raise HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Reference data load failed: {str(e)}")
            return {}

    def _validate_schema(self, match: dict, match_identifier: str) -> bool:
        """Structural validation using Pydantic"""
        try:
            MatchValidator(**match)
            return True
        except ValidationError as e:
            error_detail = f"Schema validation failed for {match_identifier}: {e.json()}"
            logger.warning(error_detail)
            self.report["errors"].append(error_detail)
            self.report["schema_failures"] += 1
            return False

    def _validate_odds(self, match: dict, match_identifier: str) -> bool:
        """Statistical validation against reference data"""
        try:
            # Ensure 'teams' key exists before accessing
            if "teams" not in match or not isinstance(match["teams"], list) or len(match["teams"]) < 2:
                 error_detail = f"Odds validation skipped for {match_identifier}: Invalid or missing 'teams' key."
                 logger.warning(error_detail)
                 self.report["errors"].append(error_detail)
                 # Treat as failure? Or allow if schema passed? For now, count as failure.
                 self.report["odds_failures"] += 1
                 return False

            ref_odds = self.reference_odds.get(tuple(match["teams"]), [])
            if not ref_odds:
                logger.warning(f"No reference odds data found for match {match_identifier} ({match['teams']}). Skipping odds validation.")
                return True  # Allow unverified matches

            # Ensure 'odds' key exists and is a list
            if "odds" not in match or not isinstance(match["odds"], list):
                 error_detail = f"Odds validation skipped for {match_identifier}: Invalid or missing 'odds' key."
                 logger.warning(error_detail)
                 self.report["errors"].append(error_detail)
                 self.report["odds_failures"] += 1
                 return False

            # Handle potential division by zero if std is 0 (e.g., only one ref point)
            ref_std = np.std(ref_odds)
            if ref_std == 0:
                 # Compare directly if std is 0 (all ref odds are the same)
                 # This is a basic check, might need refinement based on desired logic
                 is_close = all(np.isclose(o, np.mean(ref_odds)) for o in match["odds"])
                 if not is_close:
                      logger.warning(f"Odds validation failed for {match_identifier}: Odds {match['odds']} differ significantly from single reference point {np.mean(ref_odds)}.")
                      self.report["odds_failures"] += 1
                      return False
                 else:
                      return True # Odds match the single reference point

            # Calculate Z-scores
            z_scores = [
                (o - np.mean(ref_odds)) / ref_std for o in match["odds"]
            ]
            if not all(abs(z) < 3 for z in z_scores):
                 error_detail = f"Odds validation failed for {match_identifier}: Z-scores {z_scores} outside threshold."
                 logger.warning(error_detail)
                 self.report["errors"].append(error_detail)
                 self.report["odds_failures"] += 1
                 return False

            return True # Passed odds validation
        except Exception as e:
            log_error("High-level operation failed", e)
            error_detail = f"Odds validation failed for {match_identifier}: {str(e)}" # Keep existing logic
            logger.error(error_detail, exc_info=True) # Keep existing logic
            self.report["errors"].append(error_detail) # Keep existing logic
            self.report["odds_failures"] += 1 # Keep existing logic
            return False # Keep existing logic

    def validate(self, raw_data: List[dict]) -> List[dict]:
        """Multi-stage validation pipeline"""
        valid_matches = []
        self.report["processed_count"] = len(raw_data)
        self.report["start_time"] = datetime.now() # Reset start time

        for i, match in enumerate(raw_data):
            # Create a simple identifier for logging/reporting
            match_identifier = f"Match index {i}"
            if isinstance(match, dict) and "teams" in match:
                 match_identifier += f" ({match.get('teams')})"

            if not self._validate_schema(match, match_identifier):
                continue # Schema failure already logged and counted

            if not self._validate_odds(match, match_identifier):
                continue # Odds failure already logged and counted

            # If both passed
            self.report["passed_count"] += 1
            valid_matches.append(match)

        self.report["end_time"] = datetime.now()
        duration = (self.report["end_time"] - self.report["start_time"]).total_seconds()
        logger.info(
             f"Validation complete in {duration:.2f}s: "
             f"{self.report['passed_count']}/{self.report['processed_count']} valid. "
             f"Schema Failures: {self.report['schema_failures']}, Odds Failures: {self.report['odds_failures']}"
        )
        return valid_matches

    def get_validation_report(self) -> dict:
         """Returns a summary report of the validation process."""
         report_copy = self.report.copy()
         # Convert datetimes to ISO strings for JSON serialization
         report_copy["start_time"] = report_copy["start_time"].isoformat() if report_copy["start_time"] else None
         report_copy["end_time"] = report_copy["end_time"].isoformat() if report_copy["end_time"] else None
         # Optionally calculate rates
         processed = report_copy["processed_count"]
         if processed > 0:
              report_copy["pass_rate"] = round(report_copy["passed_count"] / processed, 4)
              report_copy["schema_failure_rate"] = round(report_copy["schema_failures"] / processed, 4)
              report_copy["odds_failure_rate"] = round(report_copy["odds_failures"] / processed, 4)
         else:
              report_copy["pass_rate"] = 0.0
              report_copy["schema_failure_rate"] = 0.0
              report_copy["odds_failure_rate"] = 0.0

         # Limit the number of specific errors stored to avoid huge reports
         max_errors_in_report = 50
         if len(report_copy.get("errors", [])) > max_errors_in_report:
              report_copy["errors"] = report_copy["errors"][:max_errors_in_report] + [f"... (truncated, total errors: {len(self.report['errors'])})"]

         return report_copy

    def __del__(self):
        """Ensure session is closed"""
        self.session.close()


if __name__ == "__main__":
    with open("data/raw/fixtures.json") as f:
        raw_data = json.load(f)

    validator = AdvancedDataValidator()
    clean_data = validator.validate(raw_data)

    with open("data/processed/valid_matches.json", "w") as f:
        json.dump(clean_data, f)
