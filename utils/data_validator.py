"""
Data validation and sanitization utility for ensuring data integrity.
Provides schema validation and data cleaning for various data types.
"""
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import pandas as pd
from jsonschema import validate, ValidationError

from dashboard.error_log import log_error

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Data validation and sanitization utility for ensuring data integrity.
    Provides schema validation and data cleaning for various data types.
    """
    
    # Common schemas for reuse across validation methods
    MATCH_SCHEMA = {
        "type": "object",
        "required": ["id", "home_team", "away_team", "match_date", "competition"],
        "properties": {
            "id": {"type": ["string", "integer"]},
            "home_team": {"type": "string"},
            "away_team": {"type": "string"},
            "competition": {"type": "string"},
            "match_date": {"type": ["string", "object"]},
            "home_score": {"type": ["integer", "number", "null"]},
            "away_score": {"type": ["integer", "number", "null"]},
            "status": {"type": "string"}
        }
    }
    
    PREDICTION_SCHEMA = {
        "type": "object",
        "required": ["home_win", "draw", "away_win"],
        "properties": {
            "home_win": {"type": "number", "minimum": 0, "maximum": 1},
            "draw": {"type": "number", "minimum": 0, "maximum": 1},
            "away_win": {"type": "number", "minimum": 0, "maximum": 1}
        }
    }
    
    ODDS_SCHEMA = {
        "type": "object",
        "required": ["home_win", "draw", "away_win"],
        "properties": {
            "home_win": {"type": "number", "minimum": 1},
            "draw": {"type": "number", "minimum": 1},
            "away_win": {"type": "number", "minimum": 1}
        }
    }
    
    TEAM_SCHEMA = {
        "type": "object",
        "required": ["id", "name"],
        "properties": {
            "id": {"type": ["string", "integer"]},
            "name": {"type": "string"},
            "short_name": {"type": "string"},
            "league": {"type": "string"},
            "country": {"type": "string"}
        }
    }
    
    PLAYER_SCHEMA = {
        "type": "object",
        "required": ["id", "name"],
        "properties": {
            "id": {"type": ["string", "integer"]},
            "name": {"type": "string"},
            "position": {"type": "string"},
            "team_id": {"type": ["string", "integer"]},
            "team_name": {"type": "string"}
        }
    }
    
    @staticmethod
    def validate_match_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate match data against expected schema.
        
        Args:
            data: Match data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = [
            "id", "home_team", "away_team", "match_date", "competition"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate types
        if not isinstance(data["id"], (str, int)):
            return False, "Invalid type for 'id': must be string or integer"
        
        if not isinstance(data["home_team"], str):
            return False, "Invalid type for 'home_team': must be string"
        
        if not isinstance(data["away_team"], str):
            return False, "Invalid type for 'away_team': must be string"
        
        if not isinstance(data["competition"], str):
            return False, "Invalid type for 'competition': must be string"
        
        # Validate match_date (could be string or datetime)
        if not isinstance(data["match_date"], (str, pd.Timestamp, np.datetime64)):
            return False, "Invalid type for 'match_date': must be string or datetime"
        
        return True, None
    
    @staticmethod
    def validate_prediction_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate prediction data against expected schema.
        
        Args:
            data: Prediction data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = [
            "home_win", "draw", "away_win"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate probability values
        for field in required_fields:
            if not isinstance(data[field], (int, float)):
                return False, f"Invalid type for '{field}': must be numeric"
            
            if data[field] < 0 or data[field] > 1:
                return False, f"Invalid value for '{field}': must be between 0 and 1"
        
        # Check that probabilities sum to approximately 1
        prob_sum = sum(data[field] for field in required_fields)
        if abs(prob_sum - 1.0) > 0.01:
            return False, f"Probabilities do not sum to 1: {prob_sum}"
        
        return True, None
    
    @staticmethod
    def validate_odds_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate bookmaker odds data against expected schema.
        
        Args:
            data: Odds data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = [
            "home_win", "draw", "away_win"
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate odds values
        for field in required_fields:
            if not isinstance(data[field], (int, float)):
                return False, f"Invalid type for '{field}': must be numeric"
            
            if data[field] <= 1.0:
                return False, f"Invalid value for '{field}': must be greater than 1.0"
        
        return True, None
    
    @staticmethod
    def sanitize_string(value: Any) -> str:
        """
        Sanitize a value to ensure it's a valid string.
        
        Args:
            value: Value to sanitize
            
        Returns:
            Sanitized string
        """
        if value is None:
            return ""
        
        # Convert to string
        str_value = str(value)
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>]', '', str_value)
        
        return sanitized
    
    @staticmethod
    def sanitize_numeric(value: Any, default: float = 0.0) -> float:
        """
        Sanitize a value to ensure it's a valid number.
        
        Args:
            value: Value to sanitize
            default: Default value if conversion fails
            
        Returns:
            Sanitized numeric value
        """
        if value is None:
            return default
        
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert value to numeric: {value}")
            return default
    
    @staticmethod
    def sanitize_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        string_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Sanitize a DataFrame to ensure it has required columns and valid data.
        
        Args:
            df: DataFrame to sanitize
            required_columns: List of required column names
            numeric_columns: List of columns that should be numeric
            string_columns: List of columns that should be strings
            
        Returns:
            Sanitized DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Check required columns
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing required columns in DataFrame: {missing_columns}")
                # Add missing columns with default values
                for col in missing_columns:
                    df[col] = None
        
        # Sanitize numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: DataValidator.sanitize_numeric(x))
        
        # Sanitize string columns
        if string_columns:
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: DataValidator.sanitize_string(x))
        
        return df
    
    @staticmethod
    def validate_team_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate team data against expected schema.
        
        Args:
            data: Team data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return DataValidator.validate_schema(data, DataValidator.TEAM_SCHEMA)
    
    @staticmethod
    def validate_player_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate player data against expected schema.
        
        Args:
            data: Player data dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return DataValidator.validate_schema(data, DataValidator.PLAYER_SCHEMA)
    
    @staticmethod
    def sanitize_date(value: Any, format_str: str = "%Y-%m-%d") -> Optional[str]:
        """
        Sanitize a date value to ensure it's in a consistent format.
        
        Args:
            value: Date value to sanitize (string, datetime, timestamp)
            format_str: Output format string
            
        Returns:
            Formatted date string or None if invalid
        """
        if value is None:
            return None
        
        try:
            # Handle pandas Timestamp
            if isinstance(value, pd.Timestamp):
                return value.strftime(format_str)
            
            # Handle numpy datetime64
            if isinstance(value, np.datetime64):
                return pd.Timestamp(value).strftime(format_str)
            
            # Handle python datetime
            if hasattr(value, 'strftime'):
                return value.strftime(format_str)
            
            # Handle string dates in various formats
            if isinstance(value, str):
                # Try ISO format
                try:
                    date = pd.to_datetime(value)
                    return date.strftime(format_str)
                except:
                    pass
                
                # Try common formats
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y"]:
                    try:
                        from datetime import datetime
                        date = datetime.strptime(value, fmt)
                        return date.strftime(format_str)
                    except ValueError:
                        continue
            
            # Try converting numeric timestamp
            if isinstance(value, (int, float)):
                from datetime import datetime
                try:
                    date = datetime.fromtimestamp(value)
                    return date.strftime(format_str)
                except:
                    pass
            
            # Failed to parse the date
            logger.warning(f"Failed to convert value to date: {value}")
            return None
        except Exception as e:
            logger.error(f"Error sanitizing date: {value} - {str(e)}")
            return None
    
    @staticmethod
    def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate data against a JSON schema.
        
        Args:
            data: Data to validate
            schema: JSON schema
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            validate(instance=data, schema=schema)
            return True, None
        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            log_error("Error validating schema", e)
            return False, str(e)


# Create a singleton instance for easy access
validator = DataValidator()

# Convenience functions for direct use without creating a validator instance

def validate_match_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate match data against expected schema."""
    return validator.validate_match_data(data)

def validate_prediction_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate prediction data against expected schema."""
    return validator.validate_prediction_data(data)

def validate_odds_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate bookmaker odds data against expected schema."""
    return validator.validate_odds_data(data)

def sanitize_string(value: Any) -> str:
    """Sanitize a value to ensure it's a valid string."""
    return validator.sanitize_string(value)

def sanitize_numeric(value: Any, default: float = 0.0) -> float:
    """Sanitize a value to ensure it's a valid number."""
    return validator.sanitize_numeric(value, default)

def sanitize_dataframe(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Sanitize a DataFrame to ensure it has required columns and valid data."""
    return validator.sanitize_dataframe(df, **kwargs)

def validate_team_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate team data against expected schema."""
    return validator.validate_team_data(data)

def validate_player_data(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate player data against expected schema."""
    return validator.validate_player_data(data)

def sanitize_date(value: Any, format_str: str = "%Y-%m-%d") -> Optional[str]:
    """Sanitize a date value to ensure it's in a consistent format."""
    return validator.sanitize_date(value, format_str)

def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate data against a JSON schema."""
    return validator.validate_schema(data, schema)
