# models/feature_eng/feature_generator.py
import json  # For parsing JSON from DB
import logging
import math
from datetime import datetime, timedelta  # Import datetime class as well
from typing import Any, Dict, List, Optional

# Removed import elo_rating
import numpy as np
import pandas as pd

# Initialize logger early to avoid name errors in exception handling
logger = logging.getLogger(__name__)

# Graceful import handling for optional dependencies
try:
    from data.api_clients.openweather_api import OpenWeatherAPI
    OPENWEATHER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenWeatherAPI not available: {e}")
    OpenWeatherAPI = None
    OPENWEATHER_AVAILABLE = False

# Assuming DBManager is accessible, adjust import path as needed
from database.db_manager import DatabaseManager  # Use DatabaseManager
# Specific models - Added UnderstatStatsData, FBrefStatsData
# Import Pydantic models for parsing scraped data
from scripts.core.data_models import \
    ConsolidatedMatchData  # To parse the whole structure if needed
from scripts.core.data_models import validate_data  # Helper for validation
from scripts.core.data_models import (ConsolidatedStats, FBrefStatsData,
                                      FixtureDetails, H2HData, InjuryData,
                                      LineupData, OddsData, TeamFormData,
                                      UnderstatStatsData)

try:
    from utils.config import Config  # Import correct Config class
    CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Config not available: {e}")
    Config = None
    CONFIG_AVAILABLE = False

# Default values (used if config loading fails)
DEFAULT_INITIAL_ELO = 1500
DEFAULT_K_FACTOR = 30
DEFAULT_ROLLING_WINDOW = 5


# --- User-provided Elo Class Implementation ---
class Elo:
    # Accepts parameters instead of using global constants
    def __init__(self, initial_rating: int, k_factor: int):
        self.ratings = {}
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        logger.debug(f"Internal Elo class initialized with initial_rating={self.initial_rating}, k_factor={self.k_factor}")

    def get_rating(self, team_id):
        """Gets the rating for a team, returning initial if not found."""
        rating = self.ratings.get(team_id, self.initial_rating)
        # logger.debug(f"Getting rating for {team_id}: {rating}")
        return rating

    def recordMatch(self, home_id, away_id, result):
        """
        Record match result and update Elo ratings.

        Parameters:
        - home_id: ID of the home team
        - away_id: ID of the away team
        - result: 1 for home win, 0.5 for draw, 0 for away win
        """
        home_rating = self.get_rating(home_id)
        away_rating = self.get_rating(away_id)

        # Expected score calculation
        expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        expected_away = 1 - expected_home # This is actually expected_home, fixed below

        # Correct expected score calculation
        q_home = 10**(home_rating / 400)
        q_away = 10**(away_rating / 400)
        expected_home_score = q_home / (q_home + q_away)
        # expected_away_score = q_away / (q_home + q_away) # or 1 - expected_home_score

        # Update ratings
        new_home_rating = home_rating + self.k_factor * (result - expected_home_score)
        # Result for away is (1 - result) e.g. if home wins (1), away gets 0. If draw (0.5), away gets 0.5. If home loses (0), away gets 1.
        new_away_rating = away_rating + self.k_factor * ((1 - result) - (1 - expected_home_score))

        self.ratings[home_id] = new_home_rating
        self.ratings[away_id] = new_away_rating
        # logger.debug(f"Match Recorded: {home_id} ({home_rating:.1f}->{new_home_rating:.1f}) vs {away_id} ({away_rating:.1f}->{new_away_rating:.1f}), Result: {result}")

        # Return the updated ratings for these two teams
        return {
            home_id: self.ratings[home_id],
            away_id: self.ratings[away_id]
        }

    def getPlayerRating(self, team_id):
        """Alias for get_rating to maintain compatibility with previous attempts."""
        return self.get_rating(team_id)
# --- End User-provided Elo Class ---


def _add_team_perspective_cols(df: pd.DataFrame, team_id: str) -> pd.DataFrame:
    """
    Adds perspective-based columns (goals_scored, goals_conceded, result)
    to a match history DataFrame for a specific team.

    Args:
        df (pd.DataFrame): DataFrame of matches involving the team.
        team_id (str): The ID of the team for whom the perspective is calculated.

    Returns:
        pd.DataFrame: DataFrame with added 'goals_scored', 'goals_conceded', 'result' columns.
    """
    df = df.copy() # Avoid modifying the original DataFrame slice
    team_id = str(team_id) # Ensure team_id is string for comparison

    # Goals scored/conceded from the perspective of team_id
    df['goals_scored'] = np.where(df['home_team_id'] == team_id, df['home_score'], df['away_score'])
    df['goals_conceded'] = np.where(df['home_team_id'] == team_id, df['away_score'], df['home_score'])

    # Result from the perspective of team_id
    conditions = [
        (df['home_team_id'] == team_id) & (df['home_score'] > df['away_score']), # Team won at home
        (df['away_team_id'] == team_id) & (df['away_score'] > df['home_score']), # Team won away
        (df['home_score'] == df['away_score']), # Draw
    ]
    choices = ['win', 'win', 'draw']
    df['result'] = np.select(conditions, choices, default='loss')

    # Handle potential NaN scores affecting results - if scores are NaN, result should be NaN?
    # Or treat NaN scores as 0-0 draw? For now, NaN scores might lead to 'loss' if not caught by conditions.
    # Let's explicitly set result to NaN if either score is NaN
    df.loc[df['home_score'].isna() | df['away_score'].isna(), 'result'] = np.nan

    return df


class FeatureGenerator:
    def __init__(self, db_storage: 'DatabaseManager'): # Update type hint
        """Initialize the FeatureGenerator with database storage and feature group definitions."""
        self.db = db_storage # Store the DBManager instance

        # Load configuration with fallbacks using Config.get
        try:
            # No need to get intermediate dicts, access directly
            elo_config = Config.get('feature_engineering.elo', {})
            window_config = Config.get('feature_engineering.rolling_window', {})

            self.initial_elo = elo_config.get('initial_rating', DEFAULT_INITIAL_ELO)
            self.k_factor = elo_config.get('k_factor', DEFAULT_K_FACTOR)
            self.default_window = window_config.get('default', DEFAULT_ROLLING_WINDOW)

            logger.info(f"FeatureGenerator loaded config: ELO(initial={self.initial_elo}, k={self.k_factor}), Window(default={self.default_window})")

        except Exception as config_e:
            logger.error(f"Error loading feature_engineering config: {config_e}. Using defaults.")
            self.initial_elo = DEFAULT_INITIAL_ELO
            self.k_factor = DEFAULT_K_FACTOR
            self.default_window = DEFAULT_ROLLING_WINDOW

        # Define feature groups for context-aware filtering
        self.feature_groups = {
            "H2H": [
                "h2h_team1_wins", "h2h_draws", "h2h_team2_wins", "h2h_avg_goals", "h2h_time_weighted"
            ],
            "Style": [
                "home_formation", "away_formation", "formation_clash_score",
                "home_match_xg", "away_match_xg"
            ],
            "Motivation": [
                "home_form_points_last_5", "away_form_points_last_5",
                "home_injury_impact", "away_injury_impact"
            ],
            "Home": [
                "home_avg_goals_scored_last_5", "home_avg_goals_conceded_last_5",
                "away_avg_goals_scored_last_5", "away_avg_goals_conceded_last_5",
                "home_elo", "away_elo", "elo_diff"
            ],
            # Expanded Weather group
            "Weather": [
                "weather_temp", "weather_precip", "weather_wind"
            ],
            # New Tactical group
            "Tactical": [
                "home_formation", "away_formation", "formation_clash_score", "substitutions_home", "substitutions_away"
            ],
            # New OpponentStrength group
            "OpponentStrength": [
                "home_recent_opp_elo", "away_recent_opp_elo"
            ],
            # New Rest group
            "Rest": [
                "home_rest_days", "away_rest_days"
            ],
            # New Injuries group
            "Injuries": [
                "home_injury_impact", "away_injury_impact"
            ]
        }
        
        # Initialize weather API if available
        if OPENWEATHER_AVAILABLE and OpenWeatherAPI is not None:
            try:
                self.weather_api = OpenWeatherAPI()
                logger.info("OpenWeatherAPI initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenWeatherAPI: {e}")
                self.weather_api = None
        else:
            self.weather_api = None
            logger.info("OpenWeatherAPI not available - weather features disabled")
            
        logger.info("FeatureGenerator initialized with context-aware feature groups.")

    def _get_historical_matches_df(self, date_to: datetime, status: str = "FINISHED"):
        """
        Fetches historical matches from the database using DBManager and returns a DataFrame.
        Optimized with caching for better performance.
        """
        # Create a cache key based on date_to and status
        cache_key = f"hist_matches_{date_to.strftime('%Y%m%d')}_{status}"
        
        # Check if we have this data cached
        if hasattr(self, '_match_cache') and cache_key in self._match_cache:
            logger.debug(f"Using cached historical matches data for {date_to.strftime('%Y-%m-%d')}")
            return self._match_cache[cache_key].copy()
        
        start_time = datetime.now()
        try:
            # Use DBManager's get_matches_df method if available
            if hasattr(self.db, 'get_matches_df') and callable(getattr(self.db, 'get_matches_df')):
                df = self.db.get_matches_df(date_to=date_to, status=status)
            else:
                logger.warning("DatabaseManager.get_matches_df not available; returning empty DataFrame.")
                df = pd.DataFrame()

            # Ensure the DataFrame is not empty before proceeding
            if df.empty:
                logger.info(f"No historical matches found for date_to={date_to}, status={status}")
                # Store in cache and return an empty DataFrame
                if not hasattr(self, '_match_cache'):
                    self._match_cache = {}
                self._match_cache[cache_key] = pd.DataFrame()
                return pd.DataFrame()
            
            # Ensure datetime format (already handled by get_matches_df, but good for safety)
            if 'match_date' in df.columns and not df.empty:
                df['match_date'] = pd.to_datetime(df['match_date'])
        
            # Ensure numeric types for scores (other numeric types are typically handled by get_matches_df's optimization)
            score_cols = ['home_score', 'away_score']
            for col in score_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            duration = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Fetched {len(df)} historical matches in {duration:.2f}s for {date_to.strftime('%Y-%m-%d')}")

            # Store in cache before returning
            if not hasattr(self, '_match_cache'):
                self._match_cache = {}
            self._match_cache[cache_key] = df.copy() # Store a copy to prevent modification of cached df

            return df.copy() # Return a copy

        except Exception as e:
            logger.exception(f"Error fetching historical matches using DBManager: {e}")
            return pd.DataFrame() # Return empty DataFrame on error


    def _get_enabled_features(self, context_toggles: Optional[Dict[str, bool]] = None) -> List[str]:
        """Get list of feature names that should be included based on context toggles."""
        if context_toggles is None:
            # If no toggles provided, include all features
            return [feat for feats in self.feature_groups.values() for feat in feats]
        
        enabled_features = []
        for context, features in self.feature_groups.items():
            if context_toggles.get(context, True):  # Default to True if toggle not specified
                enabled_features.extend(features)
        return enabled_features

    def _calculate_rolling_stats(
    self, team_history: pd.DataFrame, team_id: str, window: int = 5
) -> pd.DataFrame:
        """
        Calculates rolling window statistics (e.g., goals scored/conceded).

        Args:
            team_history (pd.DataFrame): DataFrame of a team's past matches, sorted by date.
            team_id (str): ID of the team for whom the rolling stats are calculated.
            window (int): The rolling window size.

        Returns:
            pd.DataFrame: DataFrame with added rolling average features.
        """
        # Use the configured default window size
        window_size = window if window is not None else self.default_window
        logger.debug(
            f"Calculating rolling stats with window {window_size} for {len(team_history)} matches."
        )
        team_history[f'avg_goals_scored_last_{window_size}'] = team_history['goals_scored'].rolling(window=window_size, closed='left').mean()
        team_history[f'avg_goals_conceded_last_{window_size}'] = team_history['goals_conceded'].rolling(window=window_size, closed='left').mean()

        # Add rolling xG if columns exist (assuming they are added in _get_historical_matches_df or similar)
        if 'home_xg' in team_history.columns and 'away_xg' in team_history.columns:
             # Calculate xG from the team's perspective for each match
             team_history['xg_for'] = np.where(team_history['home_team_id'] == team_id, team_history['home_xg'], team_history['away_xg'])
             team_history['xg_against'] = np.where(team_history['home_team_id'] == team_id, team_history['away_xg'], team_history['home_xg'])
             # Calculate rolling averages
             team_history[f'avg_xg_for_last_{window_size}'] = team_history['xg_for'].rolling(window=window_size, closed='left').mean()
             team_history[f'avg_xg_against_last_{window_size}'] = team_history['xg_against'].rolling(window=window_size, closed='left').mean()
             # Drop intermediate perspective columns
             team_history = team_history.drop(columns=['xg_for', 'xg_against'], errors='ignore')
             logger.debug(f"Calculated rolling xG stats with window {window_size}.")
        else:
             logger.debug(f"Skipping rolling xG calculation: 'home_xg' or 'away_xg' columns not found in historical data.")

        return team_history

    def _calculate_form(
        self, team_history: pd.DataFrame, window: int = None # Allow override, default to config
    ) -> pd.DataFrame:
        """
        Calculates team form based on recent results.

        Args:
            team_history (pd.DataFrame): DataFrame of a team's past matches, sorted by date.
            window (int): The window size for form calculation.

        Returns:
            pd.DataFrame: DataFrame with added form features.
        """
        # Use the configured default window size
        window_size = window if window is not None else self.default_window
        logger.debug(
            f"Calculating form with window {window_size} for {len(team_history)} matches."
        )
        # Map results to points first
        points_map = {'win': 3, 'draw': 1, 'loss': 0}
        team_history['points'] = team_history['result'].map(points_map).fillna(0) # Fill NaN results with 0 points
        # Calculate rolling sum on points
        team_history[f'form_points_last_{window_size}'] = team_history['points'].rolling(window=window_size, closed='left').sum()
        # Drop the intermediate points column if not needed elsewhere
        team_history = team_history.drop(columns=['points'], errors='ignore') # Use errors='ignore'
        return team_history

    def _calculate_head_to_head(
        self, team1_id: str, team2_id: str, all_matches: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculates head-to-head statistics between two teams.

        Args:
            team1_id (str): ID of the first team.
            team2_id (str): ID of the second team.
            all_matches (pd.DataFrame): DataFrame containing historical matches.

        Returns:
            Dict[str, Any]: Dictionary containing H2H stats (e.g., wins, avg goals).
        """
        logger.debug(f"Calculating H2H stats between {team1_id} and {team2_id}.")
        h2h_matches = all_matches[
            ((all_matches["home_team_id"] == team1_id) & (all_matches["away_team_id"] == team2_id)) |
            ((all_matches["home_team_id"] == team2_id) & (all_matches["away_team_id"] == team1_id))
        ]
        h2h_stats = {
            "h2h_team1_wins": len(h2h_matches[h2h_matches["winner"] == team1_id]),
            "h2h_draws": len(h2h_matches[h2h_matches["winner"] == "draw"]),
            "h2h_team2_wins": len(h2h_matches[h2h_matches["winner"] == team2_id]),
            "h2h_avg_goals": pd.to_numeric(h2h_matches["total_goals"], errors='coerce').mean() if not h2h_matches.empty else None,
        }
        return h2h_stats

    def _calculate_elo_ratings(
        self, historical_matches: pd.DataFrame, current_match_date: datetime
    ) -> Dict[str, float]:
        """
        Calculates ELO ratings for all teams based on historical matches up to a certain date.

        Args:
            historical_matches (pd.DataFrame): DataFrame of historical matches, sorted by date.
            current_match_date (datetime): The date up to which ELO ratings should be calculated.

        Returns:
            Dict[str, float]: Dictionary mapping team_id to its ELO rating before the current_match_date.
        """
        logger.debug(f"Calculating ELO ratings up to {current_match_date} using internal Elo class...")
        # Initialize the internal Elo environment using configured parameters
        elo_env = Elo(initial_rating=self.initial_elo, k_factor=self.k_factor)
        # team_ratings will now be managed internally by the elo_env object's ratings dict

        # Ensure current_match_date is timezone-naive
        if hasattr(current_match_date, 'tzinfo') and current_match_date.tzinfo is not None:
            logger.debug(f"_calculate_elo_ratings: current_match_date was tz-aware ({current_match_date.tzinfo}), converting to naive.")
            current_match_date = current_match_date.replace(tzinfo=None)
        
        # Work on a copy of the DataFrame to avoid SettingWithCopyWarning and modify 'match_date'
        df_elo_calc = historical_matches.copy()

        # Ensure historical_matches['match_date'] is timezone-naive
        if 'match_date' in df_elo_calc.columns and pd.api.types.is_datetime64_any_dtype(df_elo_calc['match_date']):
            if df_elo_calc['match_date'].dt.tz is not None:
                logger.debug(f"_calculate_elo_ratings: historical_matches['match_date'] is tz-aware ({df_elo_calc['match_date'].dt.tz}), removing tz_info.")
                df_elo_calc['match_date'] = df_elo_calc['match_date'].dt.tz_localize(None)
            else:
                logger.debug("_calculate_elo_ratings: historical_matches['match_date'] is already tz-naive.")
        else:
            logger.warning("_calculate_elo_ratings: 'match_date' column not found or not datetime type in historical_matches.")
            # If 'match_date' is critical and missing/wrong, this could lead to errors or incorrect ELO.
            # Consider raising an error or returning empty ratings if this case is problematic.

        # Filter matches strictly before the current match date
        matches_for_elo = df_elo_calc[df_elo_calc['match_date'] < current_match_date].sort_values('match_date')

        # Collect all unique team IDs that have played up to this point
        all_teams = set(matches_for_elo['home_team_id'].astype(str)).union(set(matches_for_elo['away_team_id'].astype(str)))

        # Process matches chronologically to update ratings
        for _, row in matches_for_elo.iterrows():
            home_id = str(row['home_team_id'])
            away_id = str(row['away_team_id'])
            home_score = row['home_score']
            away_score = row['away_score']

            # Determine result for the internal Elo class (1=home win, 0.5=draw, 0=away win)
            result = None
            if pd.isna(home_score) or pd.isna(away_score):
                logger.warning(f"Skipping ELO update for match involving {home_id} vs {away_id} due to NaN scores.")
                continue # Skip this match if scores are missing

            if home_score > away_score:
                result = 1.0
            elif away_score > home_score:
                result = 0.0
            else:
                result = 0.5

            # Record the match result using the internal Elo class method
            elo_env.recordMatch(home_id, away_id, result=result)
            # Ratings are updated internally within elo_env.ratings

        # Return the final ratings dictionary directly from the internal Elo object
        # Ensure all teams that played are included, defaulting to initial if somehow missed (shouldn't happen)
        final_ratings = {str(team): elo_env.get_rating(str(team)) for team in all_teams}
        logger.debug(f"Finished ELO calculation using internal class. {len(final_ratings)} teams rated.")
        # logger.debug(f"Final ELO Ratings sample: {dict(list(final_ratings.items())[:5])}") # Log first 5 ratings
        return final_ratings


    def generate_features_for_match(
        self, match_info: Dict[str, Any], context_toggles: Optional[Dict[str, bool]] = None
    ) -> Optional[dict]:
        if context_toggles is None:
            context_toggles = {}
        """Generate features for a single match with context-aware filtering.

        Args:
            match_info (Dict[str, Any]): Dictionary containing match details
            context_toggles (Optional[Dict[str, bool]]): Dictionary of context toggles

        Returns:
            Optional[pd.DataFrame]: Single-row DataFrame with filtered features
        """
        # Validate and extract required information from match_info with type safety
        match_id = str(match_info.get("id", "")) if match_info.get("id") is not None else ""
        home_team_id = str(match_info.get("home_team_id", "")) if match_info.get("home_team_id") is not None else ""
        away_team_id = str(match_info.get("away_team_id", "")) if match_info.get("away_team_id") is not None else ""
        match_date = match_info.get("match_date")

        # Filter features based on context toggles (if provided)
        enabled_features = self._get_enabled_features(context_toggles)

        # Validate required fields are present
        if not all([match_id, home_team_id, away_team_id, match_date]):
            logger.error(
                f"Missing required info for feature generation in match: {match_info}"
            )
            return None

        logger.info(
            f"Generating features for match {match_id} ({home_team_id} vs {away_team_id}) on {match_date}"
        )

        try:
            # --- 1. Fetch Historical Match Results (for ELO, fallback form/H2H) ---
            logger.debug(f"Fetching historical match results before {match_date}...")
            
            # Ensure match_date is a proper datetime object with robust conversion
            if not isinstance(match_date, datetime):
                try:
                    if isinstance(match_date, str):
                        match_date = pd.to_datetime(match_date)
                    else:
                        # Handle other types (like pandas Timestamp)
                        match_date = pd.to_datetime(match_date)
                    logger.debug(f"Converted match_date to datetime: {match_date}")
                except Exception as e:
                    logger.error(f"Failed to convert match_date to datetime: {e} (type: {type(match_date)}). Original value: {match_date}")
                    return None
            
            # Get a clean datetime object without problematic timezone info if needed
            if hasattr(match_date, 'tzinfo') and match_date.tzinfo is not None:
                match_date = match_date.replace(tzinfo=None)
                logger.debug(f"Removed timezone info from match_date: {match_date}")
                    
            # Fetch matches finished strictly before the current match's date using helper
            historical_matches_df = self._get_historical_matches_df(
                date_to=match_date - timedelta(microseconds=1), # Ensure strictly before
                status="FINISHED"
            )

            if historical_matches_df.empty:
                logger.warning(f"No historical match data found before {match_date} for match {match_id}. ELO/Form/H2H fallbacks may use defaults.")
            else:
                logger.debug(f"Fetched {len(historical_matches_df)} historical matches.")


            # --- 2. Fetch Detailed Scraped Data for the Current Match ---
            # --- 2. Fetch Detailed Scraped Data for the Current Match ---
            logger.debug(f"Fetching detailed scraped data for match {match_id}...")
            scraped_data_dict = None
            try:
                # Only attempt direct SQL access if the required methods are available
                if hasattr(self.db, 'fetchone') and callable(getattr(self.db, 'fetchone')) \
                   and hasattr(self.db, 'execute') and callable(getattr(self.db, 'execute')):
                    # Query the scraped_data table for the specific match_id using DBManager
                    query = "SELECT * FROM scraped_data WHERE match_id = ? LIMIT 1"
                    # DBManager's fetchone returns a tuple or None directly
                    result_tuple = self.db.fetchone(query, (str(match_id),))

                    if result_tuple:
                        # Need column names to create the dictionary. Fetch them separately.
                        # WORKAROUND: Execute a query with LIMIT 0 to get columns without data.
                        cols_query = "SELECT * FROM scraped_data LIMIT 0"
                        cursor_for_cols = self.db.execute(cols_query) # Execute returns cursor
                        colnames = [desc[0] for desc in getattr(cursor_for_cols, 'description', [])] if cursor_for_cols else []

                        if colnames:
                            raw_scraped_data = dict(zip(colnames, result_tuple))
                            logger.info(f"Found scraped data entry for match {match_id}.")
                        else:
                             logger.error(f"Could not retrieve column names for scraped_data table. Cannot process scraped data for match {match_id}.")
                             raw_scraped_data = {} # Ensure it's a dict to avoid later errors

                        # Parse the JSON columns using Pydantic models
                        scraped_data_dict = {}
                        json_columns = {
                            "fixture_details": ("fixture_details_json", FixtureDetails),
                            "odds": ("odds_json", OddsData),
                            "stats": ("stats_json", ConsolidatedStats),
                            "lineups": ("lineups_json", LineupData), # Note: Model expects Dict[str, LineupData] after parsing
                            "form": ("form_json", TeamFormData),     # Note: Model expects Dict[str, TeamFormData] after parsing
                            "h2h": ("h2h_json", H2HData),
                            "injuries": ("injuries_json", InjuryData), # Note: Model expects Dict[str, InjuryData] after parsing
                            "sources_used": ("sources_used_json", None) # Raw list, no specific model needed here
                        }

                        for key, (json_col, pydantic_model) in json_columns.items():
                            json_str = raw_scraped_data.get(json_col)
                            if json_str:
                                try:
                                    parsed_json = json.loads(json_str)
                                    if pydantic_model:
                                        # Special handling for dict structures (lineups, form, injuries)
                                        if key in ["lineups", "form", "injuries"]:
                                             validated_obj = {}
                                             if isinstance(parsed_json, dict):
                                                  for team_key, team_data in parsed_json.items():
                                                       validated_team_data = validate_data(team_data, pydantic_model, f"Scraped_{key}_{team_key}")
                                                       if validated_team_data:
                                                            validated_obj[team_key] = validated_team_data
                                             else:
                                                  logger.warning(f"Expected dict for {key} JSON in scraped_data for match {match_id}, got {type(parsed_json)}")
                                        else:
                                             # Standard validation for other models
                                             validated_obj = validate_data(parsed_json, pydantic_model, f"Scraped_{key}")

                                        if validated_obj:
                                             scraped_data_dict[key] = validated_obj
                                        else:
                                             logger.warning(f"Validation failed for {key} from scraped_data for match {match_id}.")
                                    else:
                                         # For sources_used, just store the parsed list
                                         scraped_data_dict[key] = parsed_json
                                except json.JSONDecodeError:
                                    logger.error(f"Failed to decode JSON for {json_col} for match {match_id}.")
                                except Exception as parse_e:
                                    logger.error(f"Error parsing/validating {key} for match {match_id}: {parse_e}")
                            else:
                                logger.debug(f"No JSON data found for {json_col} for match {match_id}.")
                    else:
                        logger.warning(f"No scraped data found in scraped_data table for match {match_id}.")
                else:
                    logger.debug("DBManager does not support direct SQL access methods; skipping scraped_data fetch.")

            except Exception as e: # Catch generic exceptions, DBManager handles logging specifics
                logger.exception(f"Error fetching or processing scraped data for match {match_id} using DBManager: {e}")
                # Proceed without scraped data if fetch fails


            # --- 3. Calculate ELO Ratings ---
            logger.debug("Calculating ELO ratings...")
            elo_ratings = self._calculate_elo_ratings(historical_matches_df, match_date)
            home_elo = elo_ratings.get(str(home_team_id), self.initial_elo) # Use string IDs for consistency
            away_elo = elo_ratings.get(str(away_team_id), self.initial_elo)
            elo_diff = home_elo - away_elo


            # --- 4. Prepare Data for Feature Calculations ---
            # Filter historical results for home and away teams
            logger.debug("Preparing historical data views for home and away teams...")
            home_history = historical_matches_df[
                (historical_matches_df["home_team_id"] == home_team_id) |
                (historical_matches_df["away_team_id"] == home_team_id)
            ].sort_values(by="match_date", ascending=True) # Sort ascending for rolling

            away_history = historical_matches_df[
                (historical_matches_df["home_team_id"] == away_team_id) |
                (historical_matches_df["away_team_id"] == away_team_id)
            ].sort_values(by="match_date", ascending=True) # Sort ascending for rolling

            # Add perspective-based columns (goals, result) needed for feature calculation
            # Use the newly defined internal helper method
            home_history = _add_team_perspective_cols(home_history.copy(), home_team_id)
            away_history = _add_team_perspective_cols(away_history.copy(), away_team_id)


            # --- 5. Calculate Features ---
            logger.debug("Calculating features...")
            combined_features = {}

            # Only calculate features that are enabled based on context toggles
            if "H2H" in context_toggles:
                # Calculate H2H features
                h2h_data = scraped_data_dict.get("h2h") if scraped_data_dict else None
                if h2h_data and isinstance(h2h_data, H2HData):
                    combined_features.update({
                        "h2h_team1_wins": h2h_data.team1_wins,
                        "h2h_draws": h2h_data.draws,
                        "h2h_team2_wins": h2h_data.team2_wins,
                        "h2h_avg_goals": (h2h_data.avg_goals_team1 + h2h_data.avg_goals_team2) / 2
                    })

            if "Style" in context_toggles:
                # Calculate Style features
                home_lineup = scraped_data_dict.get("lineups", {}).get("home") if scraped_data_dict else None
                away_lineup = scraped_data_dict.get("lineups", {}).get("away") if scraped_data_dict else None
                if home_lineup and away_lineup:
                    combined_features.update({
                        "home_formation": home_lineup.formation,
                        "away_formation": away_lineup.formation,
                        "formation_clash_score": self._calculate_formation_clash(
                            home_lineup.formation, away_lineup.formation
                        )
                    })

            if "Motivation" in context_toggles:
                # Calculate Motivation features
                home_form = scraped_data_dict.get("form", {}).get("home") if scraped_data_dict else None
                away_form = scraped_data_dict.get("form", {}).get("away") if scraped_data_dict else None
                if home_form and away_form:
                    combined_features.update({
                        "home_form_points_last_5": self._calculate_form_points(home_form),
                        "away_form_points_last_5": self._calculate_form_points(away_form)
                    })

            if "Home" in context_toggles:
                # Calculate Home advantage features
                elo_ratings = self._calculate_elo_ratings(historical_matches_df, match_date)
                combined_features.update({
                    "home_elo": elo_ratings.get(str(home_team_id), self.initial_elo),
                    "away_elo": elo_ratings.get(str(away_team_id), self.initial_elo),
                    "elo_diff": elo_ratings.get(str(home_team_id), self.initial_elo) - 
                               elo_ratings.get(str(away_team_id), self.initial_elo)
                })

            if "Weather" in context_toggles:
                # Placeholder for weather features
                weather_features = self._fetch_weather_features(match_info, scraped_data_dict)
                for key in self.feature_groups.get("Weather", []):
                    combined_features[key] = weather_features.get(key, np.nan)

            # --- 5a. Form & Rolling Stats (Using default window from config) ---
            # Prioritize using parsed form data from scraped_data if available
            home_form_data = scraped_data_dict.get("form", {}).get("home") if scraped_data_dict else None
            away_form_data = scraped_data_dict.get("form", {}).get("away") if scraped_data_dict else None
            window_size = self.default_window # Use configured window size

            if home_form_data and isinstance(home_form_data, TeamFormData):
                 combined_features[f'home_avg_goals_scored_last_{window_size}'] = home_form_data.avg_goals_scored
                 combined_features[f'home_avg_goals_conceded_last_{window_size}'] = home_form_data.avg_goals_conceded
                 # Placeholder for form points - needs calculation within TeamFormData or here
                 # combined_features[f'home_form_points_last_{window_size}'] = calculate_points_from_form_string(home_form_data.form_string)
                 logger.debug(f"Using pre-scraped form data for home team {home_team_id}")
            elif not home_history.empty:
                 # Fallback: Calculate from historical matches
                 home_history = self._calculate_rolling_stats(home_history, team_id=home_team_id, window=window_size)
                 home_history = self._calculate_form(home_history, window=window_size)
                 home_latest = home_history.iloc[-1] # Last row before current match
                 combined_features[f'home_avg_goals_scored_last_{window_size}'] = home_latest.get(f'avg_goals_scored_last_{window_size}')
                 combined_features[f'home_avg_goals_conceded_last_{window_size}'] = home_latest.get(f'avg_goals_conceded_last_{window_size}')
                 combined_features[f'home_form_points_last_{window_size}'] = home_latest.get(f'form_points_last_{window_size}')
                 # Add rolling xG features if calculated
                 combined_features[f'home_avg_xg_for_last_{window_size}'] = home_latest.get(f'avg_xg_for_last_{window_size}')
                 combined_features[f'home_avg_xg_against_last_{window_size}'] = home_latest.get(f'avg_xg_against_last_{window_size}')
                 logger.debug(f"Calculating form and rolling stats from history for home team {home_team_id}")
            else:
                 logger.warning(f"No form data or history for home team {home_team_id}. Form/Rolling features will be null.")

            if away_form_data and isinstance(away_form_data, TeamFormData):
                 combined_features[f'away_avg_goals_scored_last_{window_size}'] = away_form_data.avg_goals_scored
                 combined_features[f'away_avg_goals_conceded_last_{window_size}'] = away_form_data.avg_goals_conceded
                 # combined_features[f'away_form_points_last_{window_size}'] = calculate_points_from_form_string(away_form_data.form_string)
                 logger.debug(f"Using pre-scraped form data for away team {away_team_id}")
            elif not away_history.empty:
                 # Fallback: Calculate from historical matches
                 away_history = self._calculate_rolling_stats(away_history, team_id=away_team_id, window=window_size)
                 away_history = self._calculate_form(away_history, window=window_size)
                 away_latest = away_history.iloc[-1]
                 combined_features[f'away_avg_goals_scored_last_{window_size}'] = away_latest.get(f'avg_goals_scored_last_{window_size}')
                 combined_features[f'away_avg_goals_conceded_last_{window_size}'] = away_latest.get(f'avg_goals_conceded_last_{window_size}')
                 combined_features[f'away_form_points_last_{window_size}'] = away_latest.get(f'form_points_last_{window_size}')
                 # Add rolling xG features if calculated
                 combined_features[f'away_avg_xg_for_last_{window_size}'] = away_latest.get(f'avg_xg_for_last_{window_size}')
                 combined_features[f'away_avg_xg_against_last_{window_size}'] = away_latest.get(f'avg_xg_against_last_{window_size}')
                 logger.debug(f"Calculating form and rolling stats from history for away team {away_team_id}")
            else:
                 logger.warning(f"No form data or history for away team {away_team_id}. Form/Rolling features will be null.")


            # --- 5b. Head-to-Head Features ---
            h2h_data = scraped_data_dict.get("h2h") if scraped_data_dict else None
            if h2h_data and isinstance(h2h_data, H2HData):
                 # Use pre-calculated H2H stats
                 # TODO: Implement time-weighted H2H calculation if needed
                 combined_features['h2h_team1_wins'] = h2h_data.team1_wins
                 combined_features['h2h_draws'] = h2h_data.draws
                 combined_features['h2h_team2_wins'] = h2h_data.team2_wins
                 combined_features['h2h_avg_goals'] = (h2h_data.avg_goals_team1 + h2h_data.avg_goals_team2) / 2 if h2h_data.avg_goals_team1 is not None and h2h_data.avg_goals_team2 is not None else None
                 logger.debug("Using pre-scraped H2H data.")
            elif not historical_matches_df.empty:
                 # Fallback: Calculate basic H2H from historical matches
                 logger.debug("Calculating basic H2H features from history.")
                 # Add necessary columns if missing
                 if 'total_goals' not in historical_matches_df.columns:
                      historical_matches_df['total_goals'] = historical_matches_df['home_score'].fillna(0) + historical_matches_df['away_score'].fillna(0)
                 if 'winner' not in historical_matches_df.columns:
                      conditions_winner = [
                           (historical_matches_df["home_score"] > historical_matches_df["away_score"]),
                           (historical_matches_df["home_score"] == historical_matches_df["away_score"]),
                           (historical_matches_df["home_score"] < historical_matches_df["away_score"]),
                      ]
                      choices_winner = [historical_matches_df["home_team_id"], "draw", historical_matches_df["away_team_id"]]
                      historical_matches_df["winner"] = np.select(conditions_winner, choices_winner, default=None)

                 h2h_features = self._calculate_head_to_head(
                      str(home_team_id), str(away_team_id), historical_matches_df
                 )
                 combined_features.update(h2h_features)
            else:
                 logger.warning(f"No H2H data or history for match {match_id}. H2H features will be null.")


            # --- 5c. ELO Features (Using configured initial ELO) ---
            try:
                combined_features['home_elo'] = elo_ratings.get(str(home_team_id), self.initial_elo)
                combined_features['away_elo'] = elo_ratings.get(str(away_team_id), self.initial_elo)
                combined_features['elo_diff'] = combined_features['home_elo'] - combined_features['away_elo']
            except Exception as elo_e:
                 logger.error(f"Error calculating ELO features for match {match_id}: {elo_e}")
                 combined_features['home_elo'] = self.initial_elo # Fallback
                 combined_features['away_elo'] = self.initial_elo # Fallback
                 combined_features['elo_diff'] = 0 # Fallback


            # --- 5d. Injury Impact Features (New) ---
            home_injuries = scraped_data_dict.get("injuries", {}).get("home") if scraped_data_dict else None
            away_injuries = scraped_data_dict.get("injuries", {}).get("away") if scraped_data_dict else None
            combined_features['home_injury_impact'] = self._calculate_advanced_injury_impact(home_injuries)
            combined_features['away_injury_impact'] = self._calculate_advanced_injury_impact(away_injuries)


            # --- 5e. Tactical Features (New) ---
            home_lineup = scraped_data_dict.get("lineups", {}).get("home") if scraped_data_dict else None
            away_lineup = scraped_data_dict.get("lineups", {}).get("away") if scraped_data_dict else None
            home_formation = home_lineup.formation if home_lineup else None
            away_formation = away_lineup.formation if away_lineup else None
            combined_features['home_formation'] = home_formation # Store raw string for now
            combined_features['away_formation'] = away_formation # Store raw string for now
            combined_features['formation_clash_score'] = self._calculate_formation_clash(home_formation, away_formation)
            # Substitutions (if available)
            combined_features["substitutions_home"] = getattr(home_lineup, "substitutions", np.nan) if home_lineup else np.nan
            combined_features["substitutions_away"] = getattr(away_lineup, "substitutions", np.nan) if away_lineup else np.nan


            # --- 5f. Detailed Stats Features (Optional - Example: xG) ---
            home_xg_total = None
            away_xg_total = None
            if scraped_data_dict and scraped_data_dict.get("stats", {}).get("understat"):
                 understat_data = scraped_data_dict["stats"]["understat"]
                 if isinstance(understat_data, UnderstatStatsData):
                      home_xg_total = sum(shot.xG for shot in understat_data.h if shot.xG is not None)
                      away_xg_total = sum(shot.xG for shot in understat_data.a if shot.xG is not None)
            combined_features['home_match_xg'] = home_xg_total # Example: Total xG in the specific match (if available)
            combined_features['away_match_xg'] = away_xg_total # Note: This is match-specific, not historical rolling xG

            # --- 5g. FBref Stats Features (Example: Possession) ---
            home_possession = None
            away_possession = None
            # Assuming FBref data is nested under 'stats' -> 'fbref' -> 'home'/'away'
            if scraped_data_dict and scraped_data_dict.get("stats", {}).get("fbref"):
                fbref_data = scraped_data_dict["stats"]["fbref"]
                # Validate structure - Assuming FBrefStatsData holds home/away stats dicts or similar
                if isinstance(fbref_data, FBrefStatsData):
                    # Accessing possession - Adjust field names based on actual FBrefStatsData model
                    # Example: Assuming fbref_data.home_stats['Possession']
                    try:
                        home_possession = fbref_data.home_stats.get("Possession") # Use .get for safety
                        away_possession = fbref_data.away_stats.get("Possession")
                        # Convert percentage string '55%' to float 55.0 if needed
                        if isinstance(home_possession, str) and '%' in home_possession:
                            home_possession = float(home_possession.strip('%'))
                        if isinstance(away_possession, str) and '%' in away_possession:
                            away_possession = float(away_possession.strip('%'))
                        logger.debug(f"Extracted FBref Possession: Home={home_possession}, Away={away_possession}")
                    except AttributeError:
                         logger.warning("Could not access expected FBref possession fields in FBrefStatsData.")
                    except ValueError:
                         logger.warning("Could not convert FBref possession string to float.")

            combined_features['home_fbref_possession'] = home_possession
            combined_features['away_fbref_possession'] = away_possession


            # --- Opponent Strength Features ---
            combined_features["home_recent_opp_elo"] = self._calculate_recent_opponent_strength(home_history, home_team_id, elo_ratings)
            combined_features["away_recent_opp_elo"] = self._calculate_recent_opponent_strength(away_history, away_team_id, elo_ratings, window=window_size)

            # --- Rest Features ---
            combined_features["home_rest_days"] = self._calculate_rest_days(home_history, match_date)
            combined_features["away_rest_days"] = self._calculate_rest_days(away_history, match_date)

            # --- Injuries Features ---
            combined_features["home_injury_impact"] = self._calculate_advanced_injury_impact(home_injuries)
            combined_features["away_injury_impact"] = self._calculate_advanced_injury_impact(away_injuries)

            # --- Ensure all features in feature_groups are present in output ---
            all_features = [f for feats in self.feature_groups.values() for f in feats]
            for feat in all_features:
                if feat not in combined_features:
                    combined_features[feat] = np.nan

            # --- 6. Finalize feature vector ---
            combined_features['match_id'] = match_id
            logger.debug(f"Raw combined features generated for match {match_id}: {len(combined_features)} features.")
            # Return strictly aligned dict for model prediction
            finalized_features = self._finalize_features(combined_features)
            return finalized_features
        except KeyError as ke:
            logger.error(f"KeyError generating features for match {match_id}: Missing key {ke}")
            return None
        except Exception as e:
            logger.exception(f"Unexpected error generating features for match {match_id}: {e}")
            return None

    # --- New Helper Methods for Feature Calculation ---

    def _calculate_injury_impact(self, injury_data: Optional[InjuryData]) -> float:
        """
        Calculates a simple injury impact score.
        Placeholder: Currently just counts missing players.
        Enhance with player value/importance later.
        """
        if not injury_data or not injury_data.players:
            return 0.0 # No impact if no data or no players listed

        # Simple count for now
        impact_score = len(injury_data.players)
        logger.debug(f"Calculated injury impact score: {impact_score} based on {len(injury_data.players)} players.")
        # TODO: Enhance with player value (market value, rating, position importance)
        return float(impact_score)

    def _calculate_formation_clash(self, home_formation: Optional[str], away_formation: Optional[str]) -> float:
        """
        Calculates a basic formation clash score based on predefined rules.
        Placeholder: Returns a simple heuristic score.
        Score > 0 favors home team tactically, < 0 favors away team.
        """
        if not home_formation or not away_formation:
            return 0.0 # Neutral score if formations are unknown

        logger.debug(f"Calculating formation clash: {home_formation} vs {away_formation}")

        # Simple heuristic example: Midfield battle
        # Count nominal midfielders (second number in X-Y-Z format)
        try:
            home_midfielders = int(home_formation.split('-')[1]) if '-' in home_formation and len(home_formation.split('-')) > 1 else 0
            away_midfielders = int(away_formation.split('-')[1]) if '-' in away_formation and len(away_formation.split('-')) > 1 else 0
        except (ValueError, IndexError):
             logger.warning(f"Could not parse midfielder count from formations: {home_formation}, {away_formation}. Returning neutral clash score.")
             return 0.0

        # Basic rule: More midfielders = slight advantage (positive score if home has more)
        midfield_diff = home_midfielders - away_midfielders

        # Scale the difference to a reasonable range (e.g., -1 to 1)
        # This is arbitrary and needs tuning based on observed impact
        clash_score = np.clip(midfield_diff * 0.2, -1.0, 1.0) # Example scaling

        # TODO: Add more sophisticated rules based on specific formation matchups (e.g., 442 vs 433)
        # Example: 3-5-2 might struggle against wide formations like 4-3-3?

        logger.debug(f"Calculated clash score based on midfield diff ({midfield_diff}): {clash_score:.2f}")
        return clash_score

    def _get_perf_monitor(self):
        """Get or create a performance monitor with robust error handling."""
        if not hasattr(self, '_perf_monitor'):
            try:
                from scripts.core.monitoring import PerformanceMonitor
                self._perf_monitor = PerformanceMonitor()
            except ImportError:
                # Create a fallback monitor that mimics the interface but does nothing
                logger.warning("PerformanceMonitor not available, using fallback")
                class FallbackMonitor:
                    def update(self, *args, **kwargs):
                        pass  # No-op implementation
                self._perf_monitor = FallbackMonitor()
        return self._perf_monitor

    def _generate_features_parallel(self, matches_df: pd.DataFrame, context_toggles: Optional[Dict[str, bool]], enabled_features: List[str]) -> List[pd.DataFrame]:
        """Generate features using parallel processing for large datasets."""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info("Using parallel feature generation...")
        start_time = time.time()

        all_features = []
        max_workers = min(4, len(matches_df) // 10)  # Limit workers based on dataset size

        def process_match_batch(batch_df):
            batch_features = []
            for _, match_row in batch_df.iterrows():
                match_features = self.generate_features_for_match(
                    match_row.to_dict(),
                    context_toggles=context_toggles
                )
                if match_features is not None:
                    match_features["match_id"] = match_row["id"]
                    cols_to_keep = ["match_id"] + [col for col in enabled_features if col in match_features.columns]
                    match_features = match_features[cols_to_keep]
                    batch_features.append(match_features)
            return batch_features

        # Split into batches for parallel processing
        batch_size = max(10, len(matches_df) // max_workers)
        batches = [matches_df[i:i+batch_size] for i in range(0, len(matches_df), batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(process_match_batch, batch): batch for batch in batches}

            for future in as_completed(future_to_batch):
                try:
                    batch_features = future.result()
                    all_features.extend(batch_features)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")

        parallel_time = time.time() - start_time
        logger.info(f"Parallel feature generation completed in {parallel_time:.2f}s")
        return all_features

    def _generate_features_sequential(self, matches_df: pd.DataFrame, context_toggles: Optional[Dict[str, bool]], enabled_features: List[str], batch_size: int) -> List[pd.DataFrame]:
        """Generate features using optimized sequential processing."""
        import time

        logger.info("Using sequential feature generation with batching...")
        start_time = time.time()

        all_features = []
        processed_count = 0

        # Process in batches to optimize memory usage
        for i in range(0, len(matches_df), batch_size):
            batch_start = time.time()
            batch_df = matches_df.iloc[i:i+batch_size]
            batch_features = []

            for _, match_row in batch_df.iterrows():
                match_features = self.generate_features_for_match(
                    match_row.to_dict(),
                    context_toggles=context_toggles
                )
                if match_features is not None:
                    match_features["match_id"] = match_row["id"]
                    cols_to_keep = ["match_id"] + [col for col in enabled_features if col in match_features.columns]
                    match_features = match_features[cols_to_keep]
                    batch_features.append(match_features)

            all_features.extend(batch_features)
            processed_count += len(batch_df)

            batch_time = time.time() - batch_start
            if i % (batch_size * 5) == 0:  # Log progress every 5 batches
                logger.debug(f"Processed {processed_count}/{len(matches_df)} matches "
                           f"(batch time: {batch_time:.2f}s)")

        sequential_time = time.time() - start_time
        logger.info(f"Sequential feature generation completed in {sequential_time:.2f}s")
        return all_features

    def generate_features_for_dataset(
        self, matches_df: pd.DataFrame, context_toggles: Optional[Dict[str, bool]] = None
    ) -> pd.DataFrame:
        """
        Enhanced feature generation with performance optimization and batch processing.
        """
        logger.info(f"Generating features for {len(matches_df)} matches with enhanced optimization...")

        enabled_features = self._get_enabled_features(context_toggles)
        logger.debug(f"Enabled features based on context toggles: {enabled_features}")

        perf_monitor = self._get_perf_monitor()
        import time
        start_time = time.time()

        # Performance optimization: batch processing
        batch_size = Config.get('feature_engineering.batch_size', 100)
        use_parallel = Config.get('feature_engineering.parallel', False)

        try:
            if use_parallel and len(matches_df) > batch_size:
                # Parallel processing for large datasets
                all_features = self._generate_features_parallel(matches_df, context_toggles, enabled_features)
            else:
                # Sequential processing with optimized batching
                all_features = self._generate_features_sequential(matches_df, context_toggles, enabled_features, batch_size)

            if not all_features:
                logger.warning("No features were generated for the dataset.")
                perf_monitor.update('feature_eng', False, time.time() - start_time)
                return pd.DataFrame()

            # Optimized concatenation
            features_dataset = pd.DataFrame(all_features)
            logger.info(f"Generated {len(enabled_features)} features for {len(features_dataset)} matches.")

            # --- NaN Imputation based on Config ---
            try:
                # Use Config.get for imputation settings
                num_strategy = Config.get('preprocessing.missing_data.numerical', 'median') # Default to median
                cat_strategy = Config.get('preprocessing.missing_data.categorical', 'mode') # Default to mode
                # dt_strategy = imputation_config.get('datetime', 'drop') # Datetime handled earlier or excluded

                logger.info(f"Applying NaN imputation: Numerical='{num_strategy}', Categorical='{cat_strategy}'")

                for col in features_dataset.columns:
                    if col == 'match_id': continue # Skip match_id

                    if features_dataset[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(features_dataset[col]):
                            if num_strategy == 'median':
                                fill_value = features_dataset[col].median()
                            elif num_strategy == 'mean':
                                fill_value = features_dataset[col].mean()
                            # Add other strategies like 'constant' if needed
                            else:
                                logger.warning(f"Unsupported numerical imputation strategy '{num_strategy}' for column '{col}'. Using median.")
                                fill_value = features_dataset[col].median()

                            if pd.isna(fill_value): # Handle case where median/mean is NaN (e.g., all NaNs)
                                fill_value = 0 # Fallback to 0 for numeric if median/mean is NaN
                                logger.warning(f"Median/Mean for column '{col}' is NaN. Filling NaNs with 0.")

                            features_dataset[col] = features_dataset[col].fillna(fill_value)
                            logger.debug(f"Imputed NaNs in numerical column '{col}' with {num_strategy} value: {fill_value}")

                        elif pd.api.types.is_categorical_dtype(features_dataset[col]) or features_dataset[col].dtype == 'object':
                             if cat_strategy == 'mode':
                                 fill_value = features_dataset[col].mode().iloc[0] if not features_dataset[col].mode().empty else 'Unknown'
                             # Add other strategies like 'constant' if needed
                             else:
                                 logger.warning(f"Unsupported categorical imputation strategy '{cat_strategy}' for column '{col}'. Using mode.")
                                 fill_value = features_dataset[col].mode().iloc[0] if not features_dataset[col].mode().empty else 'Unknown'

                             features_dataset[col] = features_dataset[col].fillna(fill_value)
                             logger.debug(f"Imputed NaNs in categorical column '{col}' with {cat_strategy} value: {fill_value}")

            except Exception as impute_e:
                 logger.error(f"Error during NaN imputation: {impute_e}. Returning dataset with potential NaNs.")


            perf_monitor.update('feature_eng', True, time.time() - start_time)
            logger.info(f"Finished feature generation and imputation for {len(features_dataset)} matches.")
            return features_dataset
        except Exception as e:
            logger.exception("Error during feature generation pipeline: %s", e)
            perf_monitor.update('feature_eng', False, time.time() - start_time)
            return pd.DataFrame()

    def _fetch_weather_features(self, match_info: Dict[str, Any], scraped_data_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch weather features using OpenWeather API if lat/lon are available.
        Falls back to scraped data or NaN if not available or API fails.
        Ensures all expected weather keys are present in the output, defaulting to np.nan.
        """
        weather_features = {
            "weather_temp": np.nan,
            "weather_precip": np.nan,
            "weather_wind": np.nan
        }

        lat = match_info.get("latitude")
        lon = match_info.get("longitude")

        # Try fetching from API if coordinates are valid
        # Check if weather API is available and coordinates are valid
        if (self.weather_api is not None and 
            lat is not None and lon is not None and 
            not (math.isnan(float(lat)) or math.isnan(float(lon)))):
            try:
                # Ensure lat/lon are floats for the API call
                api_lat = float(lat)
                api_lon = float(lon)
                weather_api_response = self.weather_api.get_weather(api_lat, api_lon)
                if weather_api_response and isinstance(weather_api_response, dict):
                    weather_features["weather_temp"] = weather_api_response.get("main", {}).get("temp", np.nan)
                    # OpenWeather uses '1h' for rain volume in the last hour
                    weather_features["weather_precip"] = weather_api_response.get("rain", {}).get("1h", 0.0) 
                    weather_features["weather_wind"] = weather_api_response.get("wind", {}).get("speed", np.nan)
                    logger.debug(f"Successfully fetched weather from API for match {match_info.get('id')}: {weather_features}")
                    return weather_features # Return if API call was successful
                else:
                    logger.warning(f"Weather API response was None or not a dict for match {match_info.get('id')}. Lat: {api_lat}, Lon: {api_lon}")
            except ValueError as ve:
                logger.warning(f"Invalid latitude/longitude format for weather API for match {match_info.get('id')}: Lat='{lat}', Lon='{lon}'. Error: {ve}")
            except Exception as e:
                logger.warning(f"Weather API call failed for match {match_info.get('id')}. Lat: {lat}, Lon: {lon}. Error: {e}")
        else:
            if self.weather_api is None:
                logger.debug(f"Weather API not available for match {match_info.get('id')}")
            else:
                logger.debug(f"Skipping weather API call due to invalid/missing coordinates for match {match_info.get('id')}: Lat='{lat}', Lon='{lon}'")

        # Fallback to scraped data if API call failed or was skipped, and scraped_data_dict is available
        if scraped_data_dict:
            fixture_details = scraped_data_dict.get("fixture_details")
            if isinstance(fixture_details, FixtureDetails):
                 # Accessing weather through the Pydantic model's structure if available
                scraped_weather = fixture_details.weather
                if scraped_weather and isinstance(scraped_weather, dict): # Assuming weather is a dict in FixtureDetails
                    weather_features["weather_temp"] = scraped_weather.get("temperature", np.nan)
                    weather_features["weather_precip"] = scraped_weather.get("precipitation", np.nan)
                    weather_features["weather_wind"] = scraped_weather.get("wind_speed", np.nan)
                    logger.debug(f"Fetched weather from scraped_data for match {match_info.get('id')}: {weather_features}")
                    return weather_features # Return if scraped data provided weather
                elif hasattr(fixture_details, 'conditions') and fixture_details.conditions: # Alternative common structure
                    conditions = fixture_details.conditions
                    weather_features["weather_temp"] = conditions.get("temp", np.nan)
                    weather_features["weather_precip"] = conditions.get("precipitation", np.nan) # Adjust key if different
                    weather_features["weather_wind"] = conditions.get("wind_speed", np.nan)   # Adjust key if different
                    logger.debug(f"Fetched weather conditions from scraped_data for match {match_info.get('id')}: {weather_features}")
                    return weather_features

        logger.debug(f"Using default NaN for weather features for match {match_info.get('id')} after checking API and scraped data.")
        return weather_features

    def _calculate_advanced_injury_impact(self, injury_data: Optional[InjuryData]) -> float:
        """
        Calculates injury impact weighted by player value/position if available.
        Placeholder: Currently uses market_value if available, otherwise counts players.
        """
        if not injury_data or not injury_data.players:
            return 0.0
        # Example: sum market_value if present, else count (using 1.0 as default value)
        impact = 0.0
        try:
            for player in injury_data.players:
                # Attempt to get market value, default to 1.0 if missing or not convertible
                player_value = 1.0 # Default value
                raw_market_value = player.get("market_value")
                if raw_market_value is not None:
                    try:
                        # Handle potential string values like '€1.5m' - needs parsing logic
                        # For now, assume it's numeric or directly convertible
                        player_value = float(raw_market_value)
                    except (ValueError, TypeError):
                        logger.debug(f"Could not convert market_value '{raw_market_value}' to float for player. Using default 1.0.")
                        player_value = 1.0
                impact += player_value
            logger.debug(f"Calculated advanced injury impact: {impact} (based on market_value or count)")
        except Exception as e:
            logger.error(f"Error calculating advanced injury impact: {e}. Returning 0.0")
            return 0.0
        return float(impact)

    def _calculate_recent_opponent_strength(self, team_history: pd.DataFrame, current_team_id: str, elo_ratings: dict, window: int = 5) -> float:
        """
        Calculates average ELO of recent opponents.
        """
        if team_history.empty:
            return np.nan
        recent = team_history.tail(window)
        opp_ids = np.where(recent['home_team_id'] == current_team_id, recent['away_team_id'], recent['home_team_id'])
        opp_elos = [elo_ratings.get(str(oid), self.initial_elo) for oid in opp_ids] # Use configured initial ELO
        return np.mean(opp_elos) if opp_elos else np.nan

    def _calculate_rest_days(self, team_history_or_match_data, current_match_date_or_team_side) -> float:
        """
        Overloaded method to calculate rest days between matches.
        
        Can be called in two ways:
        1. _calculate_rest_days(team_history_df, current_match_date) - For processing from historical data
        2. _calculate_rest_days(match_data_dict, team_side) - For processing from match data dictionary
        
        Returns:
            float: Number of rest days, or np.nan if data is missing
        """
        # Case 1: Called with DataFrame and datetime
        if isinstance(team_history_or_match_data, pd.DataFrame):
            team_history = team_history_or_match_data
            current_match_date = current_match_date_or_team_side
            if team_history.empty:
                return np.nan
            team_history['match_date'] = pd.to_datetime(team_history['match_date'])
            last_match_date = team_history['match_date'].iloc[-1]
            if last_match_date.tzinfo is not None and current_match_date.tzinfo is None:
                last_match_date = last_match_date.tz_localize(None)
            elif last_match_date.tzinfo is None and current_match_date.tzinfo is not None:
                current_match_date = current_match_date.tz_localize(None)
            delta_days = (current_match_date - last_match_date).total_seconds() / (24 * 3600)
            return float(delta_days)
        
        # Case 2: Called with match data dictionary and team side
        match_data = team_history_or_match_data
        team_side = current_match_date_or_team_side  # 'home' or 'away'
        try:
            if team_side == 'home':
                last_match_date_str = match_data.get('home_team_last_match_date')
            else:
                last_match_date_str = match_data.get('away_team_last_match_date')
            if not last_match_date_str:
                return 7.0
            last_match_date = datetime.strptime(last_match_date_str, '%Y-%m-%d') \
                              if isinstance(last_match_date_str, str) else last_match_date_str
            current_match_date = datetime.strptime(match_data.get('match_date', datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d')
            delta_days = (current_match_date - last_match_date).total_seconds() / (24 * 3600)
            return float(delta_days)
        except Exception as e:
            logger.warning(f"Error calculating rest days from match data: {e}")
            return 7.0

    def _calculate_time_weighted_h2h(self, team1_id: str, team2_id: str, h2h_matches: pd.DataFrame, current_match_date: datetime) -> float:
        """
        Calculates time-weighted H2H score (recent matches count more).
        Returns a score between 0 and 1, representing team1's historical success rate against team2, weighted by recency.
        """
        if h2h_matches.empty:
            logger.debug("No H2H matches found for time-weighted calculation. Returning NaN.")
            return np.nan # Return NaN if no matches

        logger.debug(f"Calculating time-weighted H2H for {len(h2h_matches)} matches between {team1_id} and {team2_id}.")

        h2h_matches = h2h_matches.copy() # Avoid modifying original slice

        # Ensure match_date is datetime
        h2h_matches['match_date'] = pd.to_datetime(h2h_matches['match_date'])

        # Ensure current_match_date is timezone-naive if h2h_matches['match_date'] is, or vice-versa
        if h2h_matches['match_date'].dt.tz is not None and current_match_date.tzinfo is None:
             h2h_matches['match_date'] = h2h_matches['match_date'].dt.tz_localize(None)
        elif h2h_matches['match_date'].dt.tz is None and current_match_date.tzinfo is not None:
             current_match_date = current_match_date.tz_localize(None)

        # Calculate time difference in days
        time_diff_days = (current_match_date - h2h_matches['match_date']).dt.days

        # Define decay constant (e.g., half-life of 1 year = 365 days)
        # This could be made configurable
        decay_constant = Config.get('feature_engineering.h2h.time_decay_days', 365)
        if decay_constant <= 0:
             logger.warning(f"Invalid decay_constant ({decay_constant}). Using 365.")
             decay_constant = 365

        # Calculate weights using exponential decay
        # Add small epsilon to avoid issues with matches on the same day (time_diff=0)
        weights = np.exp(-time_diff_days / decay_constant)

        # Determine result from team1's perspective (1=win, 0.5=draw, 0=loss)
        conditions = [
            (h2h_matches['home_team_id'] == team1_id) & (h2h_matches['home_score'] > h2h_matches['away_score']), # Team1 won at home
            (h2h_matches['away_team_id'] == team1_id) & (h2h_matches['away_score'] > h2h_matches['home_score']), # Team1 won away
            (h2h_matches['home_score'] == h2h_matches['away_score']), # Draw
        ]
        choices = [1.0, 1.0, 0.5]
        h2h_matches['team1_result_score'] = np.select(conditions, choices, default=0.0) # Default to loss (0.0)

        # Handle NaN scores - set result score to NaN if scores were NaN
        h2h_matches.loc[h2h_matches['home_score'].isna() | h2h_matches['away_score'].isna(), 'team1_result_score'] = np.nan

        # Drop rows where result score is NaN before calculating weighted average
        valid_h2h = h2h_matches.dropna(subset=['team1_result_score'])
        valid_weights = weights[valid_h2h.index] # Align weights

        if valid_h2h.empty or valid_weights.sum() == 0:
            logger.warning(f"No valid H2H matches or zero total weight for time-weighted calculation between {team1_id} and {team2_id}. Returning NaN.")
            return np.nan

        # Calculate weighted average
        try:
            weighted_avg_score = np.average(valid_h2h['team1_result_score'], weights=valid_weights)
            logger.debug(f"Calculated time-weighted H2H score: {weighted_avg_score:.4f}")
            return float(weighted_avg_score) # Ensure float return type
        except ZeroDivisionError:
             logger.warning(f"Total weight is zero for time-weighted H2H calculation between {team1_id} and {team2_id}. Returning NaN.")
             return np.nan
        except Exception as e:
             logger.error(f"Error calculating weighted average for H2H between {team1_id} and {team2_id}: {e}")
             return np.nan

    def _get_h2h_stats(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts head-to-head statistics from match data dictionary.
        
        Args:
            match_data: Dictionary with match information including H2H data
            
        Returns:
            Dict[str, float]: Dictionary with H2H features
        """
        h2h_stats = {
            "h2h_team1_wins": 0,
            "h2h_draws": 0,
            "h2h_team2_wins": 0,
            "h2h_avg_goals": 0.0,
            "h2h_time_weighted": 0.5  # Default to neutral value
        }
        
        try:
            # Try to get H2H data from match data
            h2h_data = match_data.get('h2h')
            
            if h2h_data and isinstance(h2h_data, H2HData):
                # Use the Pydantic model structure if available
                h2h_stats["h2h_team1_wins"] = h2h_data.team1_wins or 0
                h2h_stats["h2h_draws"] = h2h_data.draws or 0
                h2h_stats["h2h_team2_wins"] = h2h_data.team2_wins or 0
                
                # Calculate average goals if available
                if h2h_data.avg_goals_team1 is not None and h2h_data.avg_goals_team2 is not None:
                    h2h_stats["h2h_avg_goals"] = (h2h_data.avg_goals_team1 + h2h_data.avg_goals_team2) / 2
                
                # Calculate time-weighted H2H score if match history available
                if h2h_data.matches and len(h2h_data.matches) > 0:
                    # Get team IDs
                    team1_id = h2h_data.team1_id
                    team2_id = h2h_data.team2_id
                    
                    if team1_id and team2_id:
                        # Create a dataframe from matches to calculate time-weighted score
                        matches_data = []
                        for match in h2h_data.matches:
                            if match.date and match.goals_home is not None and match.goals_away is not None:
                                match_dict = {
                                    'date': match.date,
                                    'home_team_id': match.home_team_id,
                                    'away_team_id': match.away_team_id,
                                    'home_score': match.goals_home,
                                    'away_score': match.goals_away
                                }
                                matches_data.append(match_dict)
                        
                        if matches_data:
                            # Convert to dataframe and calculate time-weighted score
                            h2h_df = pd.DataFrame(matches_data)
                            h2h_stats["h2h_time_weighted"] = self._calculate_time_weighted_h2h_score(
                                h2h_df, str(team1_id), str(team2_id)
                            )
            
            # Alternative direct access if H2HData model not used
            elif isinstance(h2h_data, dict):
                h2h_stats["h2h_team1_wins"] = h2h_data.get('team1_wins', 0)
                h2h_stats["h2h_draws"] = h2h_data.get('draws', 0) 
                h2h_stats["h2h_team2_wins"] = h2h_data.get('team2_wins', 0)
                
                # Average goals
                avg_goals_team1 = h2h_data.get('avg_goals_team1')
                avg_goals_team2 = h2h_data.get('avg_goals_team2')
                if avg_goals_team1 is not None and avg_goals_team2 is not None:
                    h2h_stats["h2h_avg_goals"] = (avg_goals_team1 + avg_goals_team2) / 2
                
                # Calculate time-weighted score if matches available
                matches = h2h_data.get('matches', [])
                if matches and isinstance(matches, list):
                    h2h_df = pd.DataFrame(matches)
                    team1_id = h2h_data.get('team1_id', '')
                    team2_id = h2h_data.get('team2_id', '')
                    if team1_id and team2_id and 'date' in h2h_df.columns:
                        h2h_stats["h2h_time_weighted"] = self._calculate_time_weighted_h2h_score(
                            h2h_df, str(team1_id), str(team2_id)
                        )
            
            # Fill in nulls with default values
            for key, value in h2h_stats.items():
                if value is None:
                    h2h_stats[key] = 0 if key != 'h2h_time_weighted' else 0.5
            
            logger.debug(f"Extracted H2H stats: {h2h_stats}")
            
        except Exception as e:
            logger.error(f"Error extracting H2H stats: {e}")
            # Keep the default values
        
        return h2h_stats

    def apply_cross_league_normalization(self, features: Dict[str, float],
                                       home_team_data: Dict[str, Any],
                                       away_team_data: Dict[str, Any]) -> Dict[str, float]:
        """Apply cross-league normalization to features."""
        try:
            from utils.cross_league_handler import CrossLeagueHandler
            cross_league_handler = CrossLeagueHandler()

            home_league = home_team_data.get('league_name', 'Unknown')
            away_league = away_team_data.get('league_name', 'Unknown')

            if home_league == away_league:
                return features  # No normalization needed for same league

            # Extract ELO features for adjustment
            home_elo = features.get('home_elo', 1500)
            away_elo = features.get('away_elo', 1500)

            # Apply cross-league ELO adjustments
            adjusted_home_elo, adjusted_away_elo = cross_league_handler.calculate_cross_league_elo_adjustment(
                home_league, away_league, home_elo, away_elo
            )

            # Update features with adjusted ELO
            normalized_features = features.copy()
            normalized_features['home_elo'] = adjusted_home_elo
            normalized_features['away_elo'] = adjusted_away_elo
            normalized_features['elo_diff'] = adjusted_home_elo - adjusted_away_elo

            # Extract form metrics for normalization
            home_form = {k: v for k, v in features.items() if 'home_' in k and ('goals' in k or 'points' in k)}
            away_form = {k: v for k, v in features.items() if 'away_' in k and ('goals' in k or 'points' in k)}

            # Apply form normalization
            normalized_home_form, normalized_away_form = cross_league_handler.normalize_form_metrics_cross_league(
                home_form, away_form, home_league, away_league
            )

            # Update features with normalized form metrics
            normalized_features.update(normalized_home_form)
            normalized_features.update(normalized_away_form)

            logger.debug(f"Applied cross-league normalization: {home_league} vs {away_league}")
            return normalized_features

        except Exception as e:
            logger.error(f"Error applying cross-league normalization: {e}")
            return features

    def _finalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Align feature dict to the canonical feature list and coerce types.

        - Ensures all expected features exist (filled with 0 or NaN where appropriate)
        - Drops any unexpected keys
        - Preserves 'match_id'
        """
        try:
            canonical = Config.get("models.normalization.feature_list", []) or []
        except Exception:
            canonical = []

        # Always keep match_id if present
        result: Dict[str, Any] = {}
        if 'match_id' in features:
            result['match_id'] = features['match_id']

        if not canonical:
            # If no canonical list configured, return original features as-is
            return features

        # Fill in canonical order
        for name in canonical:
            val = features.get(name)
            # Coerce booleans/None to numeric defaults where sensible
            if isinstance(val, bool):
                val = float(val)
            if val is None:
                # Use NaN for numeric-like features, else 0.0 as safe default
                val = np.nan
            result[name] = val

        return result

    def generate_features(self, match_data: Dict[str, Any], session=None) -> Dict[str, float]:
        """
        Legacy method name for backward compatibility.
        Delegates to generate_features_for_match.
        """
        return self.generate_features_for_match(match_data)
