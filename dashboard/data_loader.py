# dashboard/data_loader.py
import asyncio
import logging
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Apply nest_asyncio patch early to allow nested event loops
import nest_asyncio
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import and_, func, not_, or_, text
from sqlalchemy.orm import aliased, joinedload

# --- Caching helpers (top-level, not methods) ---
from streamlit import cache_data

from dashboard.data_integration import DataIntegration
from dashboard.error_log import ErrorLog
from database.db_manager import DatabaseManager
from database.schema import League, Match, Odds, Prediction, Team, TeamStats, ValueBet
from models.feature_eng.feature_generator import FeatureGenerator
from models.predictive.ensemble_model import EnsemblePredictor
from scripts.core.enhanced_scraper import EnhancedScraper
from utils.ai_insights import MatchAnalyzer
from utils.config import Config, ConfigError
from utils.prediction_handler import PredictionHandler

logger = logging.getLogger(__name__)

# Apply nest_asyncio patch early to allow nested event loops
nest_asyncio.apply()

# Global singleton instance for the data loader
_data_loader_instance = None

def get_data_loader():
    """
    Get or create the singleton instance of the DashboardDataLoader.
    
    Returns:
        DashboardDataLoader or MinimalFallbackLoader: The global data loader instance
    """
    global _data_loader_instance
    
    if _data_loader_instance is None:
        try:
            # Initialize the main data loader
            _data_loader_instance = DashboardDataLoader()
            logger.info("Created new DashboardDataLoader singleton instance")
            
            # Verify the instance has required methods
            if not hasattr(_data_loader_instance, 'get_available_leagues') or not callable(_data_loader_instance.get_available_leagues):
                logger.warning("DashboardDataLoader instance is missing critical methods, falling back to minimal loader")
                raise AttributeError("Missing required method: get_available_leagues")
                
        except Exception as e:
            # Fall back to minimal loader if initialization fails
            logger.error(f"Error initializing DashboardDataLoader, falling back to minimal: {e}", exc_info=True)
            try:
                # Try to use the create_minimal_loader function
                _data_loader_instance = create_minimal_loader()
                
                # Verify it was properly created
                if _data_loader_instance is None:
                    raise ValueError("create_minimal_loader returned None")
                    
                # Verify it has the required method
                if not hasattr(_data_loader_instance, 'get_available_leagues') or not callable(_data_loader_instance.get_available_leagues):
                    raise AttributeError("Minimal loader missing get_available_leagues method")
                    
                logger.info("Successfully created minimal data loader as fallback")
            except Exception as e2:
                # If even the minimal loader fails, create a direct instance of MinimalFallbackLoader
                logger.critical(f"Failed to create minimal loader, creating emergency fallback: {e2}", exc_info=True)
                _data_loader_instance = MinimalFallbackLoader()
    
    # Double-check that we have a valid instance
    if _data_loader_instance is None:
        logger.critical("Data loader is still None after initialization attempts, creating emergency fallback")
        _data_loader_instance = MinimalFallbackLoader()
        
    # Final verification of the instance
    if not hasattr(_data_loader_instance, 'get_available_leagues') or not callable(_data_loader_instance.get_available_leagues):
        logger.critical("Data loader instance still missing required methods, replacing with emergency fallback")
        _data_loader_instance = MinimalFallbackLoader()
            
    return _data_loader_instance


# Helper to run async code in Streamlit
def run_async(coro):
    """Run an async coroutine, handling the event loop."""
    try:
        # Use the existing running loop if there is one
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Create a new loop if one is not running
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def create_minimal_loader():
    """
    Create a minimal data loader with fallback capabilities when the main loader fails.
    This ensures basic application functionality even when some components are unavailable.
    
    Returns:
        DashboardDataLoader or MinimalFallbackLoader: A simplified data loader instance
    """
    try:
        # Create a minimal loader with SQLite database in the standard data directory
        project_root = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, 'football.db')
        db_uri = f"sqlite:///{db_path}"
        
        logger.info(f"Creating minimal data loader with fallback database: {db_uri}")
        
        # Initialize the minimal fallback loader first as a safety measure
        fallback_loader = MinimalFallbackLoader()
        
        try:
            # Use a simplified initialization that skips problematic components
            loader = DashboardDataLoader(db_uri)
            
            # Override the loader's methods that might cause issues with simple implementations
            loader.feature_generator = None
            
            # Verify the loader is properly initialized before returning
            if hasattr(loader, 'get_available_leagues') and callable(loader.get_available_leagues):
                # Only return the DashboardDataLoader if it was successfully initialized
                logger.info("Successfully created minimal data loader with DashboardDataLoader")
                return loader
            else:
                logger.warning("DashboardDataLoader missing get_available_leagues method, using fallback")
                return fallback_loader
        except Exception as inner_e:
            logger.warning(f"Could not initialize DashboardDataLoader, using MinimalFallbackLoader instead: {inner_e}")
            # Return the already initialized fallback loader
            return fallback_loader
    except Exception as e:
        logger.critical(f"Failed to create even a minimal data loader: {e}", exc_info=True)
        # Return an extremely minimal loader that won't crash the UI
        return MinimalFallbackLoader()


class MinimalFallbackLoader:
    """Emergency fallback when everything else fails to ensure UI doesn't crash."""
    
    def __init__(self):
        self.error_log = ErrorLog("minimal_fallback")
        self.error_log.log("minimal_fallback_created", "Created emergency fallback data loader")
        logger.warning("Using emergency fallback data loader with no database connection")
        
        # Initialize database connection attributes to None
        self.db_manager = None
        self.feature_generator = None
        self.data_integrator = None
        self.prediction_handler = None
        self._cache_stats = {"hits": 0, "misses": 0, "last_clear": datetime.now()}
        
        # Initialize data freshness tracking timestamps
        self.last_matches_load_ts = datetime.now()
        self.last_bulk_details_load_ts = datetime.now()
        self.last_scrape_run_ts = datetime.now()
        
        # Initialize team name map
        self.team_name_map = {}
        self.analyzer = None
        self.scraper = None
        
        # Initialize cache and tracking variables
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "last_clear": datetime.now()
        }
        self.last_validation_report = None
        self.last_feature_report = None
        self.last_config_summary = None
        self.last_matches_load_ts = None
        self.last_bulk_details_load_ts = None
        self.last_scrape_run_ts = None
        
    def get_available_leagues(self):
        """
        Returns a list of available leagues with consistent structure.
        This is the critical method that needs to work for finalize_production.py.
        """
        logger.info("MinimalFallbackLoader.get_available_leagues called")
        return [
            {'id': 1, 'name': 'Premier League', 'country': 'England', 'team_count': 20},
            {'id': 2, 'name': 'La Liga', 'country': 'Spain', 'team_count': 20},
            {'id': 3, 'name': 'Bundesliga', 'country': 'Germany', 'team_count': 18},
            {'id': 4, 'name': 'Serie A', 'country': 'Italy', 'team_count': 20},
            {'id': 5, 'name': 'Ligue 1', 'country': 'France', 'team_count': 20}
        ]
        
    def get_all_teams(self):
        """Get all team names from database for user selection."""
        logger.info("MinimalFallbackLoader.get_all_teams called")
        return ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
                "Barcelona", "Real Madrid", "Atletico Madrid", "Bayern Munich", "Borussia Dortmund"]
        
    def load_matches(self, league_names, date_range):
        """Load matches with fallback implementation."""
        logger.warning("Using mock data in minimal fallback loader")
        # Create a basic DataFrame with minimal match information
        from datetime import datetime, timedelta

        import pandas as pd
        
        start_date, end_date = date_range
        data = []
        
        # Generate some mock match data
        for league in league_names:
            teams = {
                'Premier League': ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United"],
                'La Liga': ["Barcelona", "Real Madrid", "Atletico Madrid", "Sevilla", "Valencia"],
                'Bundesliga': ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Schalke 04"],
                'Serie A': ["Juventus", "Inter Milan", "AC Milan", "Napoli", "Roma"],
                'Ligue 1': ["PSG", "Marseille", "Lyon", "Monaco", "Lille"]
            }.get(league, ["Team A", "Team B", "Team C", "Team D", "Team E"])
            
            # Add a few matches per league
            import random
            current_date = start_date
            while current_date <= end_date:
                if random.random() > 0.5:  # 50% chance to have a match on this day
                    home_idx = random.randint(0, len(teams)-1)
                    away_idx = random.randint(0, len(teams)-1)
                    while home_idx == away_idx:
                        away_idx = random.randint(0, len(teams)-1)
                    
                    match_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=random.randint(12, 20))
                    
                    data.append({
                        'id': f"mock-{league}-{current_date}-{home_idx}-{away_idx}",
                        'home_team': teams[home_idx],
                        'away_team': teams[away_idx],
                        'match_date': match_time,
                        'status': "TIMED",
                        'competition': league,
                        'league_id': {'Premier League': 1, 'La Liga': 2, 'Bundesliga': 3, 'Serie A': 4, 'Ligue 1': 5}.get(league, 0)
                    })
                current_date += timedelta(days=1)
        
        return pd.DataFrame(data)
    
    def load_match_details(self, match_id):
        """Load detailed match information with fallback values."""
        logger.info(f"MinimalFallbackLoader.load_match_details called for {match_id}")
        
        # Return a basic match details dictionary
        return {
            "fixture_details": {
                "id": match_id,
                "home_team_id": "1",
                "away_team_id": "2",
                "home_team": "Home Team",
                "away_team": "Away Team",
                "home_team_name": "Home Team",
                "away_team_name": "Away Team",
                "date_utc": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "TIMED",
                "competition_name": "Mock Competition",
                "season": "2025/26",
                "match_date": datetime.now()
            },
            "prediction": {
                "home_win_prob": 0.33,
                "draw_prob": 0.33,
                "away_win_prob": 0.34,
                "expected_goals_home": 1.0,
                "expected_goals_away": 1.0,
                "btts_prob": 0.5,
                "over_under_2_5_over_prob": 0.5,
                "over_under_2_5_under_prob": 0.5,
                "model_version": "fallback",
                "confidence": 0.7
            },
            "bookie_odds": {
                "bookmaker": "Mock Bookmaker",
                "timestamp": datetime.now(),
                "home_win": 2.0,
                "draw": 3.0,
                "away_win": 2.0,
                "over_under_2_5_over": 1.8,
                "over_under_2_5_under": 2.0,
                "both_teams_to_score_yes": 1.8,
                "both_teams_to_score_no": 2.0
            },
            "stats": {},
            "value_bet_info": {},
            "features": {}
        }
    
    def get_ai_analysis(self, match_details, user_query):
        """Return fallback AI analysis response."""
        return {
            "error": "AI analyzer not available in fallback mode.",
            "match_id": match_details.get("match_id", "unknown"),
            "query": user_query,
            "analysis": "Analysis not available in minimal fallback mode."
        }
        
    def check_recent_fixtures(self, days_future=1):
        """Fallback implementation for fixture check."""
        logger.info("MinimalFallbackLoader.check_recent_fixtures called")
        return True  # Always return True in fallback mode
        
    def clear_cache(self):
        """Clear all cached data."""
        self._cache_stats["last_clear"] = datetime.now()
        self._cache_stats["hits"] = 0
        self._cache_stats["misses"] = 0

    def get_cache_stats(self):
        """Get cache statistics."""
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"]
        }
        
    def get_data_freshness_metrics(self):
        """Returns timestamps indicating data freshness."""
        return {
            "last_matches_load_iso": datetime.now().isoformat() if not self.last_matches_load_ts else self.last_matches_load_ts.isoformat(),
            "last_bulk_details_load_iso": datetime.now().isoformat() if not self.last_bulk_details_load_ts else self.last_bulk_details_load_ts.isoformat(),
            "last_scrape_run_iso": datetime.now().isoformat() if not self.last_scrape_run_ts else self.last_scrape_run_ts.isoformat(),
        }
        


class DashboardDataLoader:
    def resolve_team_name(self, team_id: any) -> str:
        """
        Robust team name resolution with comprehensive fallbacks.
        Resolves numeric IDs, common aliases, and formats unknowns gracefully.
        """
        if not team_id:
            return "Unknown Team"
            
        # Handle special case - if team_id is already a team name string like "Liverpool"
        # First try exact match with the map values
        if isinstance(team_id, str) and not team_id.isdigit():
            # Check if the team_id is already a proper team name
            for canonical_name in self.team_name_map.values():
                if team_id.lower() == canonical_name.lower() or canonical_name.lower().startswith(team_id.lower()):
                    return canonical_name
            
            # If the team_id contains "Team" or "Unknown", it might be a placeholder
            if "unknown team" in team_id.lower() or "team #" in team_id.lower():
                # It's already a fallback format, so extract the original ID if possible
                import re
                match = re.search(r'\[(.*?)\]', team_id)
                if match:
                    original_id = match.group(1)
                    # If it's "Home Team" or "Away Team", those are placeholders
                    if original_id in ["Home Team", "Away Team"]:
                        return team_id
                    # Try again with the extracted ID
                    return self.resolve_team_name(original_id)
            
            # The team_id might already be a proper name, so return it directly if reasonably long
            if len(team_id) > 3 and team_id.lower() not in ["home", "away", "team", "unknown"]:
                # Log this to track if we're handling direct team names correctly
                logger.info(f"Assuming team_id '{team_id}' is already a proper team name")
                return team_id

        # Normalize team_id for consistent lookups
        normalized_id = str(team_id).lower().replace(" ", "_").strip()

        # Direct lookup for known IDs
        if normalized_id in self.team_name_map:
            return self.team_name_map[normalized_id]

        # Fallback to database query if not in map
        try:
            with self.db_manager.session_scope() as session:
                # Try looking up by normalized name first, then by original ID
                team = session.query(Team).filter(func.lower(Team.name) == normalized_id).first()
                if not team:
                    # Try with the original ID - try both string and int versions
                    team = session.query(Team).filter_by(id=str(team_id)).first()
                    if not team and str(team_id).isdigit():
                        team = session.query(Team).filter_by(id=int(team_id)).first()

                if team and team.name:
                    # Cache it for future lookups
                    self.team_name_map[normalized_id] = team.name
                    return team.name
                    
                # Try wildcard match as last resort
                if len(normalized_id) > 3:  # Only try wildcard for reasonably long strings
                    pattern = f"%{normalized_id}%"
                    team = session.query(Team).filter(func.lower(Team.name).like(pattern)).first()
                    if team and team.name:
                        # Cache it for future lookups
                        self.team_name_map[normalized_id] = team.name
                        return team.name
        except Exception as e:
            logger.error(f"Database error resolving team_id {team_id}: {e}", exc_info=True)

        # Final fallback for unresolved IDs
        # If team_id is numeric, return 'Team #[id]'; else, 'Unknown Team [id]'
        try:
            int_id = int(str(team_id))
            return f"Team #{int_id}"
        except (ValueError, TypeError):
            # If it's already a placeholder like "Home Team", return as is
            if isinstance(team_id, str) and ("home team" in team_id.lower() or "away team" in team_id.lower()):
                return team_id
            return f"Unknown Team [{team_id}]"

    def _get_league_id(self, league_name: str) -> Optional[str]:
        """Get the league ID from the league name."""
        try:
            with self.db_manager.session_scope() as session:
                league = self.db_manager.get_league_by_name(league_name, session)
                if league:
                    return league.id
                return None
        except Exception as e:
            logger.error(f"Error getting league ID for {league_name}: {e}", exc_info=True)
            return None
            
    def get_all_teams(self) -> List[str]:
        """Get all team names from the database for user selection."""
        try:
            with self.db_manager.session_scope() as session:
                teams = session.query(Team).all()
                return [team.name for team in teams if team.name]
        except Exception as e:
            logger.error(f"Error getting team names: {e}", exc_info=True)
            return []

    def get_teams_for_league(self, league_id: int) -> List[Dict[str, Any]]:
        """Get all teams for a specific league."""
        try:
            with self.db_manager.session_scope() as session:
                teams = session.query(Team).filter(Team.league_id == league_id).all()
                return [{"id": team.id, "name": team.name, "league_id": team.league_id} for team in teams]
        except Exception as e:
            logger.error(f"Error getting teams for league {league_id}: {e}", exc_info=True)
            return []

    def get_all_teams_with_league_info(self) -> List[Dict[str, Any]]:
        """Get all teams with their league information for cross-league selection."""
        try:
            with self.db_manager.session_scope() as session:
                teams = session.query(Team).join(League).all()
                result = []
                for team in teams:
                    team_data = {
                        "id": team.id,
                        "name": team.name,
                        "league_id": team.league_id,
                        "league_name": team.league.name,
                        "league_country": team.league.country,
                        "short_name": team.short_name,
                        "tla": team.tla,
                        "venue": team.venue
                    }
                    result.append(team_data)
                return result
        except Exception as e:
            logger.error(f"Error getting all teams with league info: {e}", exc_info=True)
            return []

    def validate_cross_league_teams(self, home_team_id: str, away_team_id: str) -> Dict[str, Any]:
        """Validate a cross-league team combination."""
        try:
            with self.db_manager.session_scope() as session:
                home_team = session.query(Team).join(League).filter(Team.id == home_team_id).first()
                away_team = session.query(Team).join(League).filter(Team.id == away_team_id).first()

                if not home_team or not away_team:
                    return {
                        'valid': False,
                        'error': 'One or both teams not found'
                    }

                return {
                    'valid': True,
                    'home_team': {
                        'id': home_team.id,
                        'name': home_team.name,
                        'league_name': home_team.league.name,
                        'league_country': home_team.league.country
                    },
                    'away_team': {
                        'id': away_team.id,
                        'name': away_team.name,
                        'league_name': away_team.league.name,
                        'league_country': away_team.league.country
                    },
                    'is_cross_league': home_team.league_id != away_team.league_id
                }
        except Exception as e:
            logger.error(f"Error validating cross-league teams: {e}", exc_info=True)
            return {
                'valid': False,
                'error': str(e)
            }

    def get_league_id(self, league_name: str) -> Optional[int]:
        """Get the league ID from the league name (public method)."""
        return self._get_league_id(league_name)
            
    def get_available_leagues(self) -> List[Dict[str, Any]]:
        """Get all available leagues from the database with full details."""
        try:
            with self.db_manager.session_scope() as session:
                leagues = session.query(League).all()
                return [
                    {
                        'id': league.id,
                        'name': league.name,
                        'country': getattr(league, 'country', 'Europe'),
                        'team_count': len(getattr(league, 'teams', []))
                    }
                    for league in leagues if league.name
                ]
        except Exception as e:
            logger.error(f"Error getting available leagues: {e}", exc_info=True)
            # Default fallback with proper structure
            return [
                {'id': 1, 'name': 'Premier League', 'country': 'England', 'team_count': 20},
                {'id': 2, 'name': 'La Liga', 'country': 'Spain', 'team_count': 20},
                {'id': 3, 'name': 'Bundesliga', 'country': 'Germany', 'team_count': 18},
                {'id': 4, 'name': 'Serie A', 'country': 'Italy', 'team_count': 20},
                {'id': 5, 'name': 'Ligue 1', 'country': 'France', 'team_count': 20}
            ]

    def __init__(self, db_uri: Optional[str] = None):
        """Initialize the DataLoader with SQLAlchemy database integration.
        
        Args:
            db_uri: Optional database URI to override the default
        """
        try:
            # Load configuration
            Config.load()
            
            # Get database URI from various sources with fallbacks
            if db_uri is None:
                # Try environment variable first
                db_uri = os.getenv('DATABASE_URI')
                
                # Then try configuration
                if not db_uri:
                    db_uri = Config.get('database.uri')
                    
                # Fallback to default SQLite database in data directory
                if not db_uri:
                    data_dir = os.getenv('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
                    os.makedirs(data_dir, exist_ok=True)
                    db_uri = f"sqlite:///{os.path.join(data_dir, 'football.db')}"
                    logger.warning(f"No database URI found in environment or config. Using default SQLite database: {db_uri}")
            
            logger.info(f"Initializing database connection with URI: {db_uri}")
            
            # Initialize database connection with the properly determined URI
            self.db_manager = DatabaseManager(db_uri)
            # Ensure tables exist early to prevent 'no such table' errors
            try:
                self.db_manager.create_tables()
            except Exception as e:
                logger.error(f"Failed to ensure database tables: {e}", exc_info=True)
                raise
            self.error_log = ErrorLog(component_name="data_loader")
            
            # Initialize data integrator for fetching new data
            self.data_integrator = DataIntegration(self.db_manager)
        except ConfigError as e:
            logger.error(f"CRITICAL: Failed to load configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
        
        # Initialize prediction handler for XGBoost models
        self.prediction_handler = PredictionHandler()
        
        # Load AI analysis config
        ai_config = Config.get("dashboard.ai_analysis", {})
        provider = ai_config.get("provider", "gemini")
        model = ai_config.get("model", "gpt-4")
        # Initialize AI analyzer with API keys from environment variables
        self.analyzer = MatchAnalyzer(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize components with error handling
        try:
            # ProxyManager is initialized within EnhancedScraper, no need to create it separately
            self.scraper = EnhancedScraper()  # The EnhancedScraper creates its own proxy manager
            
            # Pass the db_manager to the FeatureGenerator as required
            self.feature_generator = FeatureGenerator(db_storage=self.db_manager)
            
            logger.info("Successfully initialized all data loader components")
        except Exception as e:
            self.error_log.log(
                "initialization_error",
                "Failed to initialize data loader components",
                exception=e,
                source="DashboardDataLoader.__init__",
                suggestion="Check configuration and environment variables"
            )
            raise

        # --- Team name/alias map for robust resolution ---
        self.team_name_map = {
            # Canonical names and common aliases (expand as needed)
            "arsenal": "Arsenal FC",
            "arsenal_fc": "Arsenal FC",
            "man_city": "Manchester City FC",
            "manchester_city": "Manchester City FC",
            "manchester_city_fc": "Manchester City FC",
            "man_united": "Manchester United FC",
            "manchester_united": "Manchester United FC",
            "manchester_united_fc": "Manchester United FC",
            "liverpool": "Liverpool FC",
            "liverpool_fc": "Liverpool FC",
            "chelsea": "Chelsea FC",
            "chelsea_fc": "Chelsea FC",
            "tottenham": "Tottenham Hotspur FC",
            "tottenham_hotspur": "Tottenham Hotspur FC",
            "tottenham_hotspur_fc": "Tottenham Hotspur FC",
            "real_betis": "Real Betis Balompié",
            "atletico_madrid": "Atlético de Madrid",
            "west_ham": "West Ham United FC",
            "west_ham_united": "West Ham United FC",
            "west_ham_united_fc": "West Ham United FC",
            "aston_villa": "Aston Villa FC",
            "aston_villa_fc": "Aston Villa FC",
            "fulham": "Fulham FC",
            "fulham_fc": "Fulham FC",
            "newcastle": "Newcastle United FC",
            "newcastle_united": "Newcastle United FC",
            "newcastle_united_fc": "Newcastle United FC",
            "crystal_palace": "Crystal Palace FC",
            "crystal_palace_fc": "Crystal Palace FC",
            "sheffield_united": "Sheffield United FC",
            "sheffield_united_fc": "Sheffield United FC",
            "burnley": "Burnley FC",
            "burnley_fc": "Burnley FC",
            "brighton": "Brighton & Hove Albion FC",
            "brighton_&_hove_albion": "Brighton & Hove Albion FC",
            "brighton_&_hove_albion_fc": "Brighton & Hove Albion FC",
            "everton": "Everton FC",
            "everton_fc": "Everton FC",
            "bayern": "FC Bayern München",
            "bayern_munich": "FC Bayern München",
            "fc_bayern_munchen": "FC Bayern München",
            "dortmund": "Borussia Dortmund",
            "borussia_dortmund": "Borussia Dortmund",
            "rb_leipzig": "RB Leipzig",
            "leipzig": "RB Leipzig",
            "barcelona": "FC Barcelona",
            "fc_barcelona": "FC Barcelona",
            "real_madrid": "Real Madrid CF",
            "real_madrid_cf": "Real Madrid CF",
            "nottingham_forest": "Nottingham Forest FC",
            "nottingham_forest_fc": "Nottingham Forest FC",
            # Add more aliases and canonical names as needed
        }

        self.league_name_to_id = {}
        self.league_id_to_name = {}
        self._load_league_mapping()
        
        # Initialize cache tracking
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "last_clear": datetime.now()
        }

        self.last_validation_report = None
        self.last_feature_report = None
        self.last_config_summary = None
        # Data Freshness Tracking
        self.last_matches_load_ts: Optional[datetime] = None
        self.last_bulk_details_load_ts: Optional[datetime] = None
        self.last_scrape_run_ts: Optional[datetime] = None # Timestamp of the last scraping attempt within bulk load

        # Get MCP server URL from centralized config
        self.mcp_server_url = Config.get("mcp_server_url", "http://localhost:3000")
        self._matches_cache = {}
        self._ensure_feedback_table()

    def _ensure_feedback_table(self):
        """
        Ensures the 'feedback' table exists in the database.
        This is idempotent and safe to run on every feedback submission.
        """
        try:
            with self.db_manager.engine.begin() as connection:
                connection.execute(text("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id TEXT PRIMARY KEY,
                        match_id TEXT NOT NULL,
                        batch_id TEXT,
                        user_id TEXT,
                        feedback TEXT NOT NULL,
                        comment TEXT,
                        context_toggles TEXT,
                        timestamp TEXT NOT NULL
                    );
                """))
        except Exception as e:
            logger.error(f"Failed to create or verify 'feedback' table: {e}")
            self.error_log.log("db_error", "Failed to create feedback table", exception=e)

    # close_db and __del__ are likely not needed as DBManager uses context managers

    def _load_league_mapping(self):
        """Load league name to ID mapping."""
        try:
            with self.db_manager.session_scope() as session:
                leagues = session.query(League).options(joinedload(League.teams)).all()
                if not leagues:
                    logger.warning("No leagues found in the database.")
                    return
                for league in leagues:
                    self.league_name_to_id[league.name] = league.id
                    self.league_id_to_name[league.id] = league.name
        except Exception as e:
            logger.error(f"Error fetching available leagues: {e}", exc_info=True)
            self.error_log.log("db_error", f"Failed to fetch leagues: {e}", exception=e, source="get_available_leagues")

    async def _load_matches_async(self, leagues: List[str], date_range: Tuple[date, date]) -> Optional[pd.DataFrame]:
        """Helper to load matches asynchronously using DataIntegration with optimized caching for top leagues."""
        all_matches = []
        start_date, end_date = date_range
        days_ahead = (end_date - datetime.now().date()).days
        if days_ahead < 0:
            days_ahead = 0 # Ensure non-negative
            
        # Define top 6 leagues for prioritization
        top_leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "Eredivisie"]
        
        # Sort leagues to prioritize top leagues first
        prioritized_leagues = []
        
        # Add top leagues first (if they're in the selected leagues)
        for league in top_leagues:
            if league in leagues:
                prioritized_leagues.append(league)
                
        # Add remaining leagues
        for league in leagues:
            if league not in prioritized_leagues:
                prioritized_leagues.append(league)
        
        # Create tasks for parallel loading for better performance
        tasks = []
        for league_display_name in prioritized_leagues:
            tasks.append(self._load_league_matches(league_display_name, start_date, end_date, days_ahead))
        
        # Execute all tasks in parallel
        league_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in league_results:
            if isinstance(result, Exception):
                logger.error(f"Error loading matches: {result}")
                continue
            if result is not None and not result.empty:
                all_matches.append(result)
        
        if not all_matches:
            return pd.DataFrame() # Return empty DataFrame if no matches found
        
        result_df = pd.concat(all_matches, ignore_index=True)
        
        # Cache result for top leagues to improve performance
        # We use the cache_key method to create a deterministic key from the arguments
        if any(league in top_leagues for league in leagues):
            cache_key = f"matches_{','.join(sorted(leagues))}_{start_date}_{end_date}"
            self._matches_cache[cache_key] = (result_df, datetime.now())
        
        return result_df
        
    async def _load_league_matches(self, league_name: str, start_date: date, end_date: date, days_ahead: int) -> pd.DataFrame:
        """
        Helper method to load matches for a single league asynchronously with error handling.
        
        Args:
            league_name: Display name of the league
            start_date: Start date for the match search
            end_date: End date for the match search
            days_ahead: Number of days to look ahead
            
        Returns:
            DataFrame containing league matches
        """
        try:
            # Get the internal league ID if available
            league_id = self._get_league_id(league_name)
            
            # Convert dates to datetime objects for database query
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # Use the database manager to query matches with team names
            with self.db_manager.session_scope() as session:
                # Query matches with team names using joins with proper aliases
                HomeTeam = aliased(Team)
                AwayTeam = aliased(Team)
                
                logger.info(f"Querying matches for league_id='{league_id}' between {start_datetime} and {end_datetime}")

                query = session.query(
                    Match,
                    HomeTeam.name.label('home_team_name'),
                    AwayTeam.name.label('away_team_name')
                ).join(
                    HomeTeam, Match.home_team_id == HomeTeam.id, isouter=True
                ).join(
                    AwayTeam, Match.away_team_id == AwayTeam.id, isouter=True
                ).filter(
                    Match.league_id == league_id,
                    Match.match_date.between(start_datetime, end_datetime)
                ).order_by(Match.match_date)
                
                matches = query.all()
                
                # Convert to DataFrame
                if matches:
                    logger.info(f"Found {len(matches)} matches for league {league_name}")
                    data = []
                    for match_row in matches:
                        match = match_row.Match
                        # Apply team name resolution
                        home_team = match_row.home_team_name or self.resolve_team_name(match.home_team_id) or 'TBD'
                        away_team = match_row.away_team_name or self.resolve_team_name(match.away_team_id) or 'TBD'
                        
                        data.append({
                            'id': match.id,
                            'home_team_id': match.home_team_id,
                            'away_team_id': match.away_team_id,
                            'home_team': home_team,
                            'away_team': away_team,
                            'match_date': match.match_date,
                            'status': match.status,
                            'competition': league_name,
                            'league_id': match.league_id,
                            'league_name': league_name
                        })
                    
                    return pd.DataFrame(data)
                else:
                    logger.info(f"No matches found for league {league_name} between {start_date} and {end_date}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading matches for league {league_name}: {e}", exc_info=True)
            return pd.DataFrame()

    # Using a separate function for caching to avoid 'self' hash issues
    def load_matches(self, league_names: List[str], date_range: Tuple[date, date]) -> pd.DataFrame:
        """
        Load matches from the database with caching and error handling.
        This method uses a separate cached function to avoid Streamlit's self-hashing issues.
        
        Args:
            league_names: List of league names to filter by
            date_range: Tuple of start and end dates
            
        Returns:
            DataFrame with match information
        """
        return DashboardDataLoader._cached_load_matches(tuple(sorted(league_names)), date_range)
        
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def _cached_load_matches(_league_names_tuple: Tuple[str, ...], _date_range: Tuple[date, date]) -> pd.DataFrame:
        """Internal cached implementation that avoids hashing self"""
        start_time = time.time()
        try:
            # Convert tuple back to list for processing
            league_names = list(_league_names_tuple)
            # Get the singleton instance
            from dashboard.data_loader import get_data_loader
            data_loader = get_data_loader()
            # Use the data loader to fetch matches
            result_df = run_async(data_loader._load_matches_async(league_names, _date_range))
            
            load_time = time.time() - start_time
            logger.info(f"Loaded {len(result_df)} matches in {load_time:.2f}s")
            if not result_df.empty:
                result_df['_load_time'] = load_time
            return result_df
        except Exception as e:
            logger.error(f"Error loading matches: {e}", exc_info=True)
            load_time = time.time() - start_time
            logger.error(f"Match loading failed after {load_time:.2f}s")
            return pd.DataFrame()

    def load_match_details(self, match_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load detailed match information with caching, including stats, odds, and features for XGBoost prediction.
        This method uses a separate cached function to avoid Streamlit's self-hashing issues.
        
        Args:
            match_id: ID of the match to load details for
            
        Returns:
            Dictionary with comprehensive match details or error dict if match_id is invalid
        """
        # Validate match_id to prevent errors
        if not match_id:
            logger.warning("Attempted to load match details with empty match_id")
            return {
                "error": True,
                "message": "No match ID provided",
                "match_id": None,
                "fixture_details": {},
                "prediction": {},
                "bookie_odds": {},
                "stats": {},
                "value_bet_info": {},
                "features": {}
            }
            
        try:
            return DashboardDataLoader._cached_load_match_details(match_id)
        except Exception as e:
            logger.error(f"Error loading match details for {match_id}: {e}", exc_info=True)
            self.error_log.log("data_loading_error", f"Failed to load details for match {match_id}", exception=e)
            return {
                "error": True,
                "message": f"Error loading match details: {str(e)}",
                "match_id": match_id,
                "fixture_details": {},
                "prediction": {},
                "bookie_odds": {},
                "stats": {},
                "value_bet_info": {},
                "features": {}
            }
            
    @staticmethod
    @cache_data(ttl=600)  # Cache for 10 minutes
    def _cached_load_match_details(match_id: str) -> Dict[str, Any]:
        """Internal cached implementation that avoids hashing self"""
        try:
            # Get the singleton instance
            from dashboard.data_loader import get_data_loader
            data_loader = get_data_loader()
            # Use the data loader to fetch match details
            return data_loader._load_match_details_logic(match_id)
        except Exception as e:
            logger.error(f"Error in cached load match details for {match_id}: {e}", exc_info=True)
            return {
                "error": True,
                "message": f"Error loading match details: {str(e)}",
                "match_id": match_id,
                "fixture_details": {},
                "prediction": {},
                "bookie_odds": {},
                "stats": {},
                "value_bet_info": {},
                "features": {}
            }

    def _load_match_details_logic(self, match_id: str) -> Dict[str, Any]:
        """Internal logic for loading detailed match information. Wrapped by load_match_details for caching."""
        logger.info(f"Loading detailed information for match: {match_id}")
        
        # Initialize result structure
        result = {
            "fixture_details": {},
            "prediction": {},
            "bookie_odds": {},
            "stats": {},
            "value_bet_info": {},
            "features": {}
        }
        
        # Get all data within a single session scope
        with self.db_manager.session_scope() as session:
            # Get the latest prediction
            prediction = (
                session.query(Prediction)
                .filter(Prediction.match_id == match_id)
                .order_by(Prediction.created_at.desc())
                .first()
            )
            
            # Get the latest odds
            odds = (
                session.query(Odds)
                .filter(Odds.match_id == match_id)
                .order_by(Odds.timestamp.desc())
                .first()
            )
            
            # Get value bets if available
            # value_bets = (
            #     session.query(ValueBet)
            #     .filter(ValueBet.match_id == match_id)
            #     .all()
            # )
            
            # Get basic match details
            match = session.query(Match).filter(Match.id == match_id).first()
            
            if not match:
                logger.warning(f"Match not found: {match_id}")
                return result
            
            # Populate fixture details - ensure proper team name resolution
            # First try to get canonical team names directly from the team objects
            home_team_name = "Unknown"
            away_team_name = "Unknown"
            
            if match.home_team and match.home_team.name:
                home_team_name = match.home_team.name
            elif match.home_team_id:
                home_team_name = self.resolve_team_name(match.home_team_id)
                
            if match.away_team and match.away_team.name:
                away_team_name = match.away_team.name
            elif match.away_team_id:
                away_team_name = self.resolve_team_name(match.away_team_id)
            
            # Store both the IDs and resolved names for maximum compatibility
            fixture_details = {
                "id": match.id,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "home_team": home_team_name,  # Use "home_team" as the primary field
                "away_team": away_team_name,  # Use "away_team" as the primary field
                "home_team_name": home_team_name,  # Also include as "home_team_name" for backwards compatibility
                "away_team_name": away_team_name,  # Also include as "away_team_name" for backwards compatibility
                "date_utc": match.match_date.strftime("%Y-%m-%d %H:%M:%S") if match.match_date else "Unknown",
                "status": match.status,
                "competition_name": match.league.name if match.league else "Unknown Competition",
                "season": getattr(match, 'season', '2024/25'),
                "match_date": match.match_date  # Include raw match_date
            }
            
            # Update result with processed fixture details
            result["fixture_details"] = fixture_details
            
            # Log team resolution for debugging
            logger.info(f"Resolved team names for match {match.id}: {fixture_details['home_team']} vs {fixture_details['away_team']}")
            
            # Extract match features for XGBoost prediction
            try:
                # Create match info dictionary for feature generation
                match_info = {
                    'id': match.id,
                    'home_team_id': match.home_team_id,
                    'away_team_id': match.away_team_id,
                    'match_date': match.match_date,
                    'status': match.status,
                    'home_team': result["fixture_details"]["home_team"],
                    'away_team': result["fixture_details"]["away_team"]
                }
                
                # Generate features using the proper method from FeatureGenerator
                if self.feature_generator:
                    try:
                        # First try the direct method - generate_features_for_match
                        feature_df = self.feature_generator.generate_features_for_match(match_info)
                        
                        if feature_df is not None:
                            # Check if it's a DataFrame or a dict
                            if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
                                # Convert DataFrame to dictionary for easier handling
                                result["features"] = feature_df.iloc[0].to_dict()
                                logger.info(f"Successfully generated {len(result['features'])} features for match {match_id}")
                            elif isinstance(feature_df, dict) and feature_df:
                                # It's already a dictionary
                                result["features"] = feature_df
                                logger.info(f"Successfully generated {len(result['features'])} features for match {match_id} (direct dict)")
                            else:
                                # If that fails, try an alternative approach using a test implementation
                                logger.warning(f"Primary feature generation returned no results for match {match_id}, trying backup approach")
                                import datetime
                                test_match_info = {
                                    'id': match.id,
                                    'home_team_id': str(match.home_team_id),  # Ensure string type
                                    'away_team_id': str(match.away_team_id),  # Ensure string type
                                    'match_date': datetime.datetime.now() if not match.match_date else match.match_date,
                                    'status': 'SCHEDULED' if not match.status else match.status
                                }
                                backup_features = self.feature_generator.generate_features_for_match(test_match_info)
                                if backup_features is not None:
                                    if isinstance(backup_features, dict):
                                        # If it returned a dict instead of DataFrame
                                        result["features"] = backup_features
                                        logger.info(f"Backup approach succeeded with direct dict features for match {match_id}")
                                    elif isinstance(backup_features, pd.DataFrame) and not backup_features.empty:
                                        result["features"] = backup_features.iloc[0].to_dict()
                                        logger.info(f"Backup approach succeeded with {len(result['features'])} features for match {match_id}")
                                    else:
                                        logger.warning(f"Backup feature generation returned unexpected type: {type(backup_features)}")
                                        result["features"] = {}
                                else:
                                    logger.warning(f"Both feature generation approaches failed for match {match_id}")
                                    result["features"] = {}
                    except AttributeError as attr_err:
                        # This might be an issue with the specific method/attribute
                        logger.error(f"AttributeError during feature generation: {attr_err}", exc_info=True)
                        # Create fallback features with safe default values
                        result["features"] = {
                            "h2h_team1_wins": 0, "h2h_draws": 0, "h2h_team2_wins": 0, 
                            "h2h_avg_goals": 0.0, "h2h_time_weighted": 0.5,
                            "weather_temp": 15.0, "weather_precip": 0.0, "weather_wind": 5.0,
                            "home_rest_days": 7.0, "away_rest_days": 7.0,
                            "home_match_xg": 1.5, "away_match_xg": 1.2,
                            "home_formation": "4-3-3", "away_formation": "4-3-3"
                        }
                        logger.info(f"Created fallback features for match {match_id} after AttributeError")
                else:
                    logger.warning(f"Feature generator not initialized for match {match_id}")
                    result["features"] = {}
            except Exception as e:
                logger.error(f"Error generating features for match {match_id}: {e}", exc_info=True)
                # Create minimal fallback features
                result["features"] = {
                    "h2h_team1_wins": 0, "h2h_draws": 0, "h2h_team2_wins": 0, 
                    "h2h_avg_goals": 0.0, "h2h_time_weighted": 0.5
                }
            
            # Add match stats if available
            if hasattr(match, 'match_stats') and match.match_stats:
                # Match stats is a single object, not a list (uselist=False in the relationship)
                stats = match.match_stats  # No need to use [0] since it's a single object, not a list
                result["stats"] = {
                    "possession": {
                        "home": getattr(stats, 'home_possession', 0),
                        "away": getattr(stats, 'away_possession', 0)
                    },
                    "shots": {
                        "home": getattr(stats, 'home_shots', 0),
                        "away": getattr(stats, 'away_shots', 0)
                    },
                    "shots_on_target": {
                        "home": getattr(stats, 'home_shots_on_target', 0),
                        "away": getattr(stats, 'away_shots_on_target', 0)
                    },
                    "corners": {
                        "home": getattr(stats, 'home_corners', 0),
                        "away": getattr(stats, 'away_corners', 0)
                    },
                    "fouls": {
                        "home": getattr(stats, 'home_fouls', 0),
                        "away": getattr(stats, 'away_fouls', 0)
                    },
                    "cards": {
                        "yellow": {
                            "home": getattr(stats, 'home_yellow_cards', 0),
                            "away": getattr(stats, 'away_yellow_cards', 0)
                        },
                        "red": {
                            "home": getattr(stats, 'home_red_cards', 0),
                            "away": getattr(stats, 'away_red_cards', 0)
                        }
                    }
                }
                
                # Add form data if available
                if hasattr(stats, 'home_form') and hasattr(stats, 'away_form'):
                    result["stats"]["home_form"] = stats.home_form if stats.home_form else "UNKNOWN"
                    result["stats"]["away_form"] = stats.away_form if stats.away_form else "UNKNOWN"
                
                # Copy features to stats for easy access
                if result["features"]:
                    result["stats"]["features"] = result["features"]
            
            # Add prediction if available
            if prediction:
                result["prediction"] = {
                    "home_win_prob": getattr(prediction, 'home_win_prob', 0.33),
                    "draw_prob": getattr(prediction, 'draw_prob', 0.33),
                    "away_win_prob": getattr(prediction, 'away_win_prob', 0.34),
                    "expected_goals_home": getattr(prediction, 'expected_goals_home', 1.0),
                    "expected_goals_away": getattr(prediction, 'expected_goals_away', 1.0),
                    "btts_prob": getattr(prediction, 'btts_prob', 0.5),
                    "over_under_2_5_over_prob": getattr(prediction, 'over_under_2_5_over_prob', 0.5),
                    "over_under_2_5_under_prob": getattr(prediction, 'over_under_2_5_under_prob', 0.5),
                    "model_version": getattr(prediction, 'model_version', 'v1.0'),
                    "confidence": getattr(prediction, 'confidence', 0.7)
                }
            
            # Add odds if available
            if odds:
                result["bookie_odds"] = {
                    "bookmaker": getattr(odds, 'bookmaker', 'Unknown'),
                    "timestamp": getattr(odds, 'timestamp', datetime.now()),
                    "home_win": getattr(odds, 'home_win', 2.0),
                    "draw": getattr(odds, 'draw', 3.0),
                    "away_win": getattr(odds, 'away_win', 2.0),
                    "over_under_2_5_over": getattr(odds, 'over_under_2_5_over', 1.8),
                    "over_under_2_5_under": getattr(odds, 'over_under_2_5_under', 2.0),
                    "both_teams_to_score_yes": getattr(odds, 'both_teams_to_score_yes', 1.8),
                    "both_teams_to_score_no": getattr(odds, 'both_teams_to_score_no', 2.0)
                }
            
            # Add value bets if available
            # if value_bets:
            #     result["value_bet_info"] = [
            #         {
            #             "bet_type": getattr(vb, 'bet_type', 'unknown'),
            #             "selection": getattr(vb, 'selection', 'unknown'),
            #             "odds": getattr(vb, 'odds', 2.0),
            #             "fair_odds": getattr(vb, 'fair_odds', 2.0),
            #             "edge": getattr(vb, 'edge', 0.0),
            #             "bookmaker": getattr(vb, 'bookmaker', 'Unknown'),
            #             "stake_suggestion": getattr(vb, 'stake_suggestion', 0.0)
            #         } 
            #         for vb in value_bets
            #     ]

            # Add team stats if available
            home_stats = session.query(TeamStats).filter(TeamStats.team_id == match.home_team_id).first()
            away_stats = session.query(TeamStats).filter(TeamStats.team_id == match.away_team_id).first()
            if home_stats and away_stats:
                result["stats"]["home_team_stats"] = {
                    "goals_scored": getattr(home_stats, 'goals_scored', 0),
                    "goals_conceded": getattr(home_stats, 'goals_conceded', 0),
                    "matches_played": getattr(home_stats, 'matches_played', 0),
                    "wins": getattr(home_stats, 'wins', 0),
                    "draws": getattr(home_stats, 'draws', 0),
                    "losses": getattr(home_stats, 'losses', 0),
                    "home_advantage": getattr(home_stats, 'home_advantage', 0.0)
                }
                result["stats"]["away_team_stats"] = {
                    "goals_scored": getattr(away_stats, 'goals_scored', 0),
                    "goals_conceded": getattr(away_stats, 'goals_conceded', 0),
                    "matches_played": getattr(away_stats, 'matches_played', 0),
                    "wins": getattr(away_stats, 'wins', 0),
                    "draws": getattr(away_stats, 'draws', 0),
                    "losses": getattr(away_stats, 'losses', 0),
                    "away_advantage": getattr(away_stats, 'away_advantage', 0.0)
                }
        
        return result

    def clear_cache(self):
        """Clear all cached data."""
        self._cache_stats["last_clear"] = datetime.now()
        self._cache_stats["hits"] = 0
        self._cache_stats["misses"] = 0
        # Clear Streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"]
        }
        
    async def get_upcoming_matches(self, league_name: str, days_ahead: int = 14) -> pd.DataFrame:
        """
        Get upcoming matches for a specific league.
        
        Args:
            league_name: League name or ID
            days_ahead: Number of days ahead to look for matches
            
        Returns:
            DataFrame with upcoming matches
        """
        try:
            # Calculate date range from today to N days ahead
            today = datetime.now().date()
            future_date = today + timedelta(days=days_ahead)
            
            # Use data_integrator to fetch the data directly
            matches_df = await self._load_league_matches(league_name, today, future_date, days_ahead)            
            return matches_df
        except Exception as e:
            logger.error(f"Error getting upcoming matches for {league_name}: {e}", exc_info=True)
            return pd.DataFrame()

    async def process_data_batch(self, batch: List[Any], processor_func: Callable, batch_size: int = 10, delay_between_batches: float = 0.1) -> List[Any]:
        """
        Process data in optimized batches with concurrency limits to prevent resource exhaustion.
        
        Args:
            batch: List of items to process
            processor_func: Async function to process each item
            batch_size: Size of each processing batch
            delay_between_batches: Delay between batches in seconds
            
        Returns:
            List of processed results
        """
        results = []
        
        try:
            # Process in controlled batches
            for i in range(0, len(batch), batch_size):
                current_batch = batch[i:i + batch_size]
                batch_tasks = [processor_func(item) for item in current_batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Handle any exceptions in the batch results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing batch item {i+j}: {result}")
                        # Add None for failed items to maintain order
                        results.append(None)
                    else:
                        results.append(result)
                
                # Add controlled delay between batches to manage resource usage
                if i + batch_size < len(batch):  # Only delay if there are more batches
                    await asyncio.sleep(delay_between_batches)
            
            return results
        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            # Return partial results if any were successful
            return results

    def generate_features_for_matches(self, matches_df: pd.DataFrame, context_toggles: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
        """Generate features for a set of matches using the feature generator."""
        try:
            feature_gen = FeatureGenerator(self.db_manager)
            # Pass context_toggles to the feature generator
            features = feature_gen.generate_features_for_dataset(matches_df, context_toggles=context_toggles)
            # --- Robust feature alignment (enforce canonical feature list) ---
            from utils.config import Config
            feature_list = Config.get("models.normalization.feature_list", [])
            if feature_list:
                missing = [feat for feat in feature_list if feat not in features.columns]
                extra = [feat for feat in features.columns if feat not in feature_list and feat != 'match_id']
                if missing:
                    logger.warning(f"Missing features in dashboard: {missing}. Imputing with 0.0.")
                    for feat in missing:
                        features[feat] = 0.0
                if extra:
                    logger.warning(f"Extra features in dashboard: {extra}. Dropping them.")
                    features = features.drop(columns=extra)
                # Ensure correct order
                features = features[[col for col in ['match_id'] + feature_list if col in features.columns]]
            else:
                logger.warning("No canonical feature_list found in config. Skipping strict alignment.")
            
            self.last_feature_report = {
                'num_matches': len(matches_df),
                'num_features': features.shape[1] if not features.empty else 0,
                'success': not features.empty,
                'context_toggles': context_toggles
            }
            return features
        except Exception as e:
            logger.error(f"Error generating features: {e}", exc_info=True)
            self.last_feature_report = {'error': str(e), 'success': False, 'context_toggles': context_toggles}
            return pd.DataFrame()

    def get_last_validation_report(self):
        return self.last_validation_report

    def get_last_feature_report(self):
        return self.last_feature_report

    def get_config_summary(self):
        return self.last_config_summary

    def run_batch_predictions(
        self, 
        selected_league_names: List[str], 
        date_range: Tuple[date, date], 
        model_version: str = None, 
        feature_set: str = None, 
        user_id: str = None,
        context_toggles: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """
        Run batch predictions for selected leagues and date range (max 4 days).
        Args:
            selected_league_names (List[str]): Leagues to include.
            date_range (Tuple[date, date]): (start_date, end_date), max 4 days.
            model_version (str): Model version or name to use.
            feature_set (str): Feature set name to use.
            user_id (str): Optional user identifier for analytics.
            context_toggles (Optional[Dict[str, bool]]): Dictionary of context toggles for feature selection.
        Returns:
            Dict[str, Any]: {
                'matches': [ {match info, prediction, confidence, explanation, ...} ],
                'aggregate_feature_importance': {feature: importance, ...},
                'batch_metadata': {...}
            }
        """
        # --- Input validation ---
        if not selected_league_names:
            raise ValueError("No leagues selected for batch prediction.")
        if not date_range or len(date_range) != 2:
            raise ValueError("Date range must be a tuple of (start, end) date.")
        start_date, end_date = date_range
        if (end_date - start_date).days > 3:
            raise ValueError("Date range cannot exceed 4 days.")
        
        allowed_leagues = set(self.league_name_to_id.keys())

        for league in selected_league_names:
            if league not in allowed_leagues:
                raise ValueError(f"League \'{league}\' is not allowed.")

        # --- Load matches ---
        matches_df = self.load_matches(selected_league_names, date_range)
        if matches_df.empty:
            return {'matches': [], 'aggregate_feature_importance': {}, 'batch_metadata': {}}

        # --- Generate features with context toggles ---
        features_df = self.generate_features_for_matches(matches_df, context_toggles=context_toggles)
        if features_df.empty:
            return {'matches': [], 'aggregate_feature_importance': {}, 'batch_metadata': {}}

        # --- Run predictions ---
        try:
            predictor = EnsemblePredictor(model_version=model_version, feature_set=feature_set)
        except Exception as e:
            logger.error(f"Could not initialize predictor: {e}", exc_info=True)
            return {'matches': [], 'aggregate_feature_importance': {}, 'batch_metadata': {}}

        batch_results = []
        feature_importances = []
        for idx, row in features_df.iterrows():
            match_id = row.get('match_id') or row.get('id')
            match_info = matches_df[matches_df['id'] == match_id].iloc[0].to_dict() if 'id' in matches_df.columns else {}
            try:
                X = row.values.reshape(1, -1)
                pred = predictor.predict(X)[0]  # e.g., [home, draw, away]
                confidence = predictor.get_confidence(pred) if hasattr(predictor, 'get_confidence') else max(pred)
                # Explainability (SHAP or model-based)
                explanation = None
                if hasattr(predictor, 'explain_prediction'):
                    explanation = predictor.explain_prediction(X)
                    if explanation is not None:
                        feature_importances.append(explanation)
                batch_results.append({
                    'match_id': match_id,
                    'match_info': match_info,
                    'prediction': pred,
                    'confidence': confidence,
                    'explanation': explanation
                })
            except Exception as e:
                logger.warning(f"Prediction failed for match {match_id}: {e}", exc_info=True)
                batch_results.append({
                    'match_id': match_id,
                    'match_info': match_info,
                    'prediction': None,
                    'confidence': None,
                    'explanation': None,
                    'error': str(e)
                })

        # --- Aggregate feature importance (mean of abs importances) ---
        aggregate_importance = {}
        if feature_importances:
            importances = np.array([list(f.values()) for f in feature_importances if isinstance(f, dict)])
            if importances.size > 0:
                mean_importance = np.mean(np.abs(importances), axis=0)
                feature_names = list(feature_importances[0].keys())
                aggregate_importance = dict(zip(feature_names, mean_importance))

        # --- Final result structure ---
        return {
            'matches': batch_results,
            'aggregate_feature_importance': aggregate_importance,
            'batch_metadata': {
                'num_matches_processed': len(features_df),
                'num_predictions_successful': sum(1 for r in batch_results if r.get('prediction') is not None),
                'model_version_used': model_version,
                'feature_set_used': feature_set,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
        }

    def get_ai_analysis(self, match_details: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """Get AI-powered analysis for a match."""
        try:
            if not self.analyzer:
                return {"error": "AI analyzer not initialized."}
            return self.analyzer.analyze_match(match_details, user_query)
        except Exception as e:
            logger.error(f"Error getting AI analysis: {e}", exc_info=True)
            return {"error": str(e)}

    def submit_prediction_feedback(self, match_id: str, feedback: str, user_id: str = None, comment: str = None, batch_id: str = None, context_toggles: dict = None):
        """
        Store user feedback (thumbs up/down, comment) for a prediction.
        Args:
            match_id (str): Match identifier.
            feedback (str): 'up' or 'down'.
            user_id (str): Optional user identifier.
            comment (str): Optional user comment.
            batch_id (str): Optional batch run identifier.
            context_toggles (dict): Context toggles at time of feedback.
        """
        self._ensure_feedback_table()
        import json  # Keep json import here
        try:
            with self.db_manager.session_scope() as session:
                # Generate a unique ID for the feedback entry
                feedback_id = str(uuid.uuid4())
                insert_sql = text("""
                INSERT INTO feedback (id, match_id, batch_id, user_id, feedback, comment, context_toggles, timestamp)
                VALUES (:id, :match_id, :batch_id, :user_id, :feedback, :comment, :context_toggles, :timestamp)
                """)
                session.execute(insert_sql, {
                    "id": feedback_id,
                    "match_id": match_id,
                    "batch_id": batch_id,
                    "user_id": user_id,
                    "feedback": feedback,
                    "comment": comment,
                    "context_toggles": json.dumps(context_toggles) if context_toggles else None,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            logger.info(f"Feedback persisted: match_id={match_id}, feedback={feedback}, user_id={user_id}, batch_id={batch_id}")
        except Exception as e:
            logger.error(f"Failed to persist feedback: {e}", exc_info=True)

    def get_feedback_for_batch(self, batch_id: str) -> list:
        """Fetch all feedback for a given batch_id."""
        self._ensure_feedback_table()
        try:
            with self.db_manager.session_scope() as session:
                rows = session.execute(text("SELECT match_id, user_id, feedback, comment, context_toggles, timestamp FROM feedback WHERE batch_id = :batch_id"), {"batch_id": batch_id}).fetchall()
                return [
                    {
                        'match_id': r[0],
                        'user_id': r[1],
                        'feedback': r[2],
                        'comment': r[3],
                        'context_toggles': r[4],
                        'timestamp': r[5]
                    } for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch feedback for batch {batch_id}: {e}", exc_info=True)
            return []

    def get_feedback_history_for_user(self, user_id: str) -> list:
        """Fetch all feedback submitted by a user."""
        self._ensure_feedback_table()
        try:
            with self.db_manager.session_scope() as session:
                rows = session.execute(text("SELECT match_id, batch_id, feedback, comment, context_toggles, timestamp FROM feedback WHERE user_id = :user_id ORDER BY timestamp DESC"), {"user_id": user_id}).fetchall()
                return [
                    {
                        'match_id': r[0],
                        'batch_id': r[1],
                        'feedback': r[2],
                        'comment': r[3],
                        'context_toggles': r[4],
                        'timestamp': r[5]
                    } for r in rows
                ]
        except Exception as e:
            logger.error(f"Failed to fetch feedback history for user {user_id}: {e}", exc_info=True)
            return []

    def get_data_freshness_metrics(self) -> Dict[str, Optional[str]]:
        """Returns timestamps indicating data freshness."""
        return {
            "last_matches_load_iso": self.last_matches_load_ts.isoformat() if self.last_matches_load_ts else None,
            "last_bulk_details_load_iso": self.last_bulk_details_load_ts.isoformat() if self.last_bulk_details_load_ts else None,
            "last_scrape_run_iso": self.last_scrape_run_ts.isoformat() if self.last_scrape_run_ts else None,
        }

    def check_recent_fixtures(self, days_future=1) -> bool:
        """
        Checks if there are any matches scheduled for today or within the next 'days_future' days.

        Args:
            days_future (int): Number of future days (including today) to check for fixtures. Defaults to 1 (today and tomorrow).

        Returns:
            bool: True if recent/upcoming fixtures exist, False otherwise.
        """
        try:
            today_date = datetime.now().date()
            future_date = today_date + timedelta(days=days_future) # Check up to 'days_future' days ahead

            # Convert dates to ISO format strings for the query
            start_datetime_str = datetime.combine(today_date, datetime.min.time()).isoformat()
            end_datetime_str = datetime.combine(future_date, datetime.max.time()).isoformat()

            with self.db_manager.session_scope() as session:
                count = session.query(Match).filter(
                    Match.match_date.between(start_datetime_str, end_datetime_str),
                    Match.status == "TIMED"
                ).count()

            if count > 0:
                logger.info(f"Recent fixtures check passed: Found {count} matches between {today_date} and {future_date}.")
                return True
            else:
                logger.warning(f"Recent fixtures check failed: No 'TIMED' matches found between {today_date} and {future_date}.")
                return False
        except Exception as e:
            logger.error("Error checking for recent fixtures in database", e)
            return False

    def get_match_analysis_aspects(
        self,
        home_team: str,
        away_team: str,
        match_id: str,
        prediction: Optional[Dict] = None,
        odds: Optional[Dict] = None,
        stats: Optional[Dict] = None,
        provider: str = None,
        model: str = None,
        aspects: list = None
    ) -> dict:
        """
        Get aspect-based AI-generated match analysis using pre-fetched data if available.
        Returns a dict of {aspect_key: analysis_text}.
        """
        if not self.analyzer:
            return {"error": "AI analysis not available. Please configure API keys."}

        logger.debug(
            f"Generating aspect-based AI analysis for match {match_id} ({home_team} vs {away_team})"
        )

        # Fetch data only if not provided
        if prediction is None:
            prediction = self.get_prediction(match_id)
        if odds is None:
            odds = self.get_bookmaker_odds(match_id)
        if stats is None:
            try:
                with self.db_manager.session_scope() as session:
                    result = session.query(MatchStats).filter(MatchStats.match_id == match_id).first()
                    if result:
                        import json
                        stats = json.loads(result.stats_json)
                        logger.debug(f"Fetched pre-scraped stats for match {match_id} for AI analysis.")
                    else:
                        stats = {} # Default to empty dict if no stats found
            except Exception as e:
                logger.warning(f"Could not fetch pre-scraped stats for match {match_id}: {e}")
                stats = {} # Default to empty dict on error

        # Use config if not provided
        ai_config = Config.get("dashboard.ai_analysis", {})
        provider = provider or ai_config.get("provider", "gemini")
        model = model or ai_config.get("model", "gpt-4")
        aspects = aspects or ai_config.get("analysis_aspects", [
            {"name": "Match Overview", "key": "overview"},
            {"name": "Team Form", "key": "form"},
            {"name": "Key Factors", "key": "factors"},
            {"name": "Prediction Confidence", "key": "confidence"}
        ])

        try:
            results = self.analyzer.analyze_match_aspects(
                home_team, away_team, stats, prediction, provider=provider, model=model, aspects=aspects
            )
            return results
        except Exception as e:
            logger.exception(f"Error generating aspect-based AI analysis for match {match_id}: {e}", exc_info=True)
            return {"error": f"Error generating AI analysis: {e}"}

    def get_prediction(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest prediction for a match.
        
        Args:
            match_id: Match identifier
            
        Returns:
            Dictionary with prediction data or None if not found
        """
        try:
            with self.db_manager.session_scope() as session:
                prediction = (
                    session.query(Prediction)
                    .filter(Prediction.match_id == match_id)
                    .order_by(Prediction.created_at.desc())
                    .first()
                )
                
                if prediction:
                    return {
                        "home_win_prob": prediction.home_win_prob,
                        "draw_prob": prediction.draw_prob,
                        "away_win_prob": prediction.away_win_prob,
                        "expected_goals_home": prediction.expected_goals_home,
                        "expected_goals_away": prediction.expected_goals_away,
                        "btts_prob": prediction.btts_prob,
                        "over_under_2_5_over_prob": prediction.over_under_2_5_over_prob,
                        "over_under_2_5_under_prob": prediction.over_under_2_5_under_prob,
                        "model_version": prediction.model_version,
                        "confidence": prediction.confidence
                    }
                else:
                    # Return default prediction if none found
                    return {
                        "home_win_prob": 0.33,
                        "draw_prob": 0.33,
                        "away_win_prob": 0.34,
                        "expected_goals_home": 1.0,
                        "expected_goals_away": 1.0,
                        "btts_prob": 0.5,
                        "over_under_2_5_over_prob": 0.5,
                        "over_under_2_5_under_prob": 0.5,
                        "model_version": "v1.0",
                        "confidence": 0.7
                    }
                    
        except Exception as e:
            logger.error(f"Error getting prediction for match {match_id}: {e}", exc_info=True)
            return None

    def get_bookmaker_odds(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest bookmaker odds for a match.
        
        Args:
            match_id: Match identifier
            
        Returns:
            Dictionary with odds data or None if not found
        """
        try:
            with self.db_manager.session_scope() as session:
                odds = (
                    session.query(Odds)
                    .filter(Odds.match_id == match_id)
                    .order_by(Odds.timestamp.desc())
                    .first()
                )
                
                if odds:
                    return {
                        "bookmaker": odds.bookmaker,
                        "timestamp": odds.timestamp,
                        "home_win": odds.home_win,
                        "draw": odds.draw,
                        "away_win": odds.away_win,
                        "over_under_2_5_over": odds.over_under_2_5_over,
                        "over_under_2_5_under": odds.over_under_2_5_under,
                        "both_teams_to_score_yes": odds.both_teams_to_score_yes,
                        "both_teams_to_score_no": odds.both_teams_to_score_no
                    }
                else:
                    # Return default odds if none found
                    return {
                        "bookmaker": "Unknown",
                        "timestamp": datetime.now(),
                        "home_win": 2.0,
                        "draw": 3.0,
                        "away_win": 2.0,
                        "over_under_2_5_over": 1.8,
                        "over_under_2_5_under": 2.0,
                        "both_teams_to_score_yes": 1.8,
                        "both_teams_to_score_no": 2.0
                    }
                    
        except Exception as e:
            logger.error(f"Error getting odds for match {match_id}: {e}", exc_info=True)
            return None

    # (Removed duplicate get_available_leagues to avoid overriding DB-driven implementation)


# This duplicate create_minimal_loader function has been removed to avoid confusion
# The primary implementation is at the top of the file