import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import (BaseModel, Field, HttpUrl, ValidationError,
                      field_validator)

logger = logging.getLogger(__name__)

# --- Base Models ---

class BaseSourceData(BaseModel):
    """Base model for data parsed from a specific source."""
    source: str
    data: Optional[Union[Dict[str, Any], str]] = None # Allow string for parsing errors
    parsing_error: Optional[str] = None # Field to store parsing errors

# --- Component Models (for Consolidated Data) ---

class FixtureDetails(BaseModel):
    """Model for core fixture information."""
    fixture_id: Optional[int] = None
    status: Optional[str] = None
    status_long: Optional[str] = None
    timestamp: Optional[int] = None
    date_utc: Optional[datetime] = None # Pydantic handles datetime parsing
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    league_name: Optional[str] = None
    league_round: Optional[str] = None
    home_team_id: Optional[int] = None
    home_team_name: Optional[str] = None
    home_team_winner: Optional[bool] = None
    away_team_id: Optional[int] = None
    away_team_name: Optional[str] = None
    away_team_winner: Optional[bool] = None
    goals_home: Optional[int] = None
    goals_away: Optional[int] = None
    score_halftime_home: Optional[int] = None
    score_halftime_away: Optional[int] = None
    score_fulltime_home: Optional[int] = None
    score_fulltime_away: Optional[int] = None
    # Fields from other sources if needed
    match_date: Optional[str] = None # From FBref scorebox
    kickoff_time: Optional[str] = None # From FBref scorebox
    home_team_crest: Optional[HttpUrl] = None # From Football-Data
    away_team_crest: Optional[HttpUrl] = None # From Football-Data
    competition_name: Optional[str] = None # From Football-Data
    competition_code: Optional[str] = None # From Football-Data
    season_matchday: Optional[int] = None # From Football-Data

    # Add temporal validation
    @field_validator('date_utc')
    def validate_date_not_future(cls, v):
        if v is not None and v > datetime.now(): # Check if v is not None before comparison
            raise ValueError("Fixture date cannot be in the future")
        return v

    class Config:
        extra = 'ignore' # Ignore extra fields during validation

class OddsData(BaseModel):
    """Model for betting odds."""
    home_win: Optional[float] = None
    draw: Optional[float] = None
    away_win: Optional[float] = None
    source: Optional[str] = None

    class Config:
        extra = 'ignore'

class PlayerStatsDetail(BaseModel):
    """Generic model for player stats tables (can be refined)."""
    # Example fields - adjust based on actual FBref table columns
    Player: Optional[str] = None
    Min: Optional[Union[int, str]] = None # Can be string like '90'
    # Add other common stats fields...
    # Goals: Optional[int] = Field(None, alias='Gls') # Example alias
    # Assists: Optional[int] = Field(None, alias='Ast')

    class Config:
        extra = 'allow' # Allow extra columns from FBref

class TeamStatsDetail(BaseModel):
    """Generic model for team stats tables."""
    # Example fields
    Possession: Optional[Union[float, str]] = Field(None, alias='%') # Example alias
    # Add other common stats fields...

    class Config:
        extra = 'allow'

class FBrefStatsData(BaseModel):
    """Model for parsed FBref data structure."""
    summary_stats: Optional[Dict[str, Any]] = {}
    team_stats: Optional[Dict[str, Dict[str, List[TeamStatsDetail]]]] = {} # home/away -> stat_type -> list
    player_stats: Optional[Dict[str, Dict[str, List[PlayerStatsDetail]]]] = {} # home/away -> stat_type -> list
    lineups: Optional[Dict[str, List[str]]] = {} # home/away -> list of names
    shots: Optional[List[Dict[str, Any]]] = None # List of shot events
    other_tables: Optional[Dict[str, List[Dict[str, Any]]]] = {}

    class Config:
        extra = 'ignore'

class UnderstatShotData(BaseModel):
    """Model for a single shot in Understat data."""
    id: Optional[str] = None
    minute: Optional[int] = None
    result: Optional[str] = None
    X: Optional[float] = None
    Y: Optional[float] = None
    xG: Optional[float] = None
    player: Optional[str] = None
    h_a: Optional[str] = None # Home 'h' or Away 'a'
    player_id: Optional[int] = None
    situation: Optional[str] = None
    season: Optional[str] = None
    shotType: Optional[str] = None
    match_id: Optional[int] = None
    h_team: Optional[str] = None
    a_team: Optional[str] = None
    h_goals: Optional[int] = None
    a_goals: Optional[int] = None
    date: Optional[datetime] = None
    player_assisted: Optional[str] = None
    lastAction: Optional[str] = None

    class Config:
        extra = 'ignore'

class UnderstatStatsData(BaseModel):
    """Model for parsed Understat data structure."""
    h: Optional[List[UnderstatShotData]] = [] # Home shots
    a: Optional[List[UnderstatShotData]] = [] # Away shots
    # Add other potential top-level keys if needed (e.g., matchData)

    class Config:
        extra = 'ignore'

class TransfermarktData(BaseModel):
    """Model for parsed Transfermarkt data."""
    match_details: Optional[Dict[str, Any]] = {}
    lineups: Optional[Dict[str, List[str]]] = {}
    substitutions: Optional[Dict[str, List[Any]]] = {} # Structure might vary
    goals: Optional[List[Dict[str, Any]]] = []
    market_values: Optional[Dict[str, Optional[str]]] = {}

    class Config:
        extra = 'ignore'


class ConsolidatedStats(BaseModel):
    """Model for the consolidated stats section."""
    understat: Optional[UnderstatStatsData] = None
    fbref: Optional[FBrefStatsData] = None
    transfermarkt: Optional[TransfermarktData] = None
    other: Optional[List[BaseSourceData]] = [] # For sources like SportsDataIO

    class Config:
        extra = 'ignore'

# --- New Models for Scope 2 ---

class PlayerInfo(BaseModel):
    """Basic player identification."""
    id: Optional[int] = None
    name: Optional[str] = None
    number: Optional[int] = None
    pos: Optional[str] = None # Position (e.g., 'G', 'D', 'M', 'F')
    grid: Optional[str] = None # Position on grid (e.g., '4:2')

class LineupData(BaseModel):
    """Model for team lineup and formation."""
    team_id: Optional[int] = None
    team_name: Optional[str] = None
    formation: Optional[str] = None
    startXI: Optional[List[PlayerInfo]] = []
    substitutes: Optional[List[PlayerInfo]] = []
    coach_id: Optional[int] = None
    coach_name: Optional[str] = None
    source: Optional[str] = None # e.g., API-Football

    class Config:
        extra = 'ignore'

class RecentMatchResult(BaseModel):
    """Model for a single recent match result for form calculation."""
    fixture_id: Optional[int] = None
    date: Optional[datetime] = None
    opponent_id: Optional[int] = None
    opponent_name: Optional[str] = None
    venue: Optional[str] = Field(None, pattern="^(Home|Away)$") # 'Home' or 'Away'
    result: Optional[str] = Field(None, pattern="^(W|D|L)$") # 'W', 'D', 'L'
    goals_for: Optional[int] = None
    goals_against: Optional[int] = None
    league_id: Optional[int] = None

    class Config:
        extra = 'ignore'

class TeamFormData(BaseModel):
    """Model for a team's recent form."""
    team_id: Optional[int] = None
    team_name: Optional[str] = None
    matches: Optional[List[RecentMatchResult]] = []
    form_string: Optional[str] = None # e.g., "WWLDW"
    avg_goals_scored: Optional[float] = None
    avg_goals_conceded: Optional[float] = None
    clean_sheets: Optional[int] = None
    failed_to_score: Optional[int] = None
    source: Optional[str] = None

    class Config:
        extra = 'ignore'

class H2HMatchResult(BaseModel):
    """Model for a single past H2H match result."""
    fixture_id: Optional[int] = None
    date: Optional[datetime] = None
    home_team_id: Optional[int] = None
    home_team_name: Optional[str] = None
    away_team_id: Optional[int] = None
    away_team_name: Optional[str] = None
    goals_home: Optional[int] = None
    goals_away: Optional[int] = None
    league_id: Optional[int] = None
    league_name: Optional[str] = None

    class Config:
        extra = 'ignore'

class H2HData(BaseModel):
    """Model for head-to-head statistics."""
    team1_id: Optional[int] = None
    team2_id: Optional[int] = None
    total_matches: Optional[int] = None
    team1_wins: Optional[int] = None
    team2_wins: Optional[int] = None
    draws: Optional[int] = None
    avg_goals_team1: Optional[float] = None
    avg_goals_team2: Optional[float] = None
    matches: Optional[List[H2HMatchResult]] = []
    source: Optional[str] = None

    class Config:
        extra = 'ignore'

class InjuredPlayer(BaseModel):
    """Model for an injured or suspended player."""
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    type: Optional[str] = None # e.g., 'Injury', 'Suspension'
    reason: Optional[str] = None # e.g., 'Knee injury', 'Red card'
    fixture_id: Optional[int] = None # Link to specific fixture if applicable

    class Config:
        extra = 'ignore'

class InjuryData(BaseModel):
    """Model for team injuries and suspensions."""
    team_id: Optional[int] = None
    team_name: Optional[str] = None
    players: Optional[List[InjuredPlayer]] = []
    source: Optional[str] = None

    class Config:
        extra = 'ignore'


# --- Main Consolidated Model ---

class ConsolidatedMatchData(BaseModel):
    """Top-level model for the final consolidated data saved to the database."""
    fixture_details: FixtureDetails = Field(default_factory=FixtureDetails)
    odds: Optional[OddsData] = None
    stats: Optional[ConsolidatedStats] = Field(default_factory=ConsolidatedStats)
    # --- Added fields for Scope 2 ---
    lineups: Optional[Dict[str, LineupData]] = None # {'home': LineupData, 'away': LineupData}
    form: Optional[Dict[str, TeamFormData]] = None # {'home': TeamFormData, 'away': TeamFormData}
    h2h: Optional[H2HData] = None
    injuries: Optional[Dict[str, InjuryData]] = None # {'home': InjuryData, 'away': InjuryData}
    # --- End added fields ---
    prediction: Optional[Any] = None # Keep flexible for now
    # Add prediction confidence fields
    prediction_confidence: Optional[float] = Field(None, ge=0, le=1)
    sources_used: List[str] = []

    @field_validator('stats')
    def validate_stats(cls, v):
        """Ensure at least one stats source present"""
        if v is not None and not any([v.understat, v.fbref, v.transfermarkt]):
            # Allow other stats if primary ones are missing, but log warning
            if not v.other:
                 raise ValueError("At least one stats source (understat, fbref, transfermarkt, or other) required")
            else:
                 logger.warning("Primary stats sources (understat, fbref, transfermarkt) are missing, relying on 'other'.")
        return v

    class Config:
        validate_assignment = True # Re-validate on attribute assignment
        extra = 'ignore'

# --- Helper Function for Validation ---

def validate_data(data: Dict[str, Any], model: BaseModel, source_name: str = "Consolidated") -> Optional[BaseModel]:
    """Validates dictionary data against a Pydantic model."""
    try:
        validated_model = model.model_validate(data)
        # logger.debug(f"Data validation successful for {source_name} using {model.__name__}")
        return validated_model
    except ValidationError as e:
        logger.warning(f"Data validation failed for {source_name} using {model.__name__}: {e}")
        # Optionally log the problematic data (be careful with large data)
        # logger.debug(f"Invalid data for {source_name}: {data}")
        return None

class DataModel:
    """Main data model class for handling various data operations."""
    
    def __init__(self):
        self.models = {
            'consolidated_match': ConsolidatedMatchData,
            'fixture_details': FixtureDetails,
            'odds_data': OddsData,
            'lineup_data': LineupData,
            'team_form': TeamFormData,
            'h2h_data': H2HData,
            'injury_data': InjuryData,
            'player_stats': PlayerStatsDetail,
            'team_stats': TeamStatsDetail,
            'fbref_stats': FBrefStatsData,
            'understat_stats': UnderstatStatsData,
            'transfermarkt_data': TransfermarktData
        }
    
    def validate(self, data: Dict[str, Any], model_type: str, source_name: str = "Unknown") -> Optional[BaseModel]:
        """Validate data against specified model type."""
        if model_type not in self.models:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        model_class = self.models[model_type]
        return validate_data(data, model_class, source_name)
    
    def get_model_class(self, model_type: str) -> Optional[BaseModel]:
        """Get model class by type name."""
        return self.models.get(model_type)
    
    def create_consolidated_match(self, **kwargs) -> ConsolidatedMatchData:
        """Create a new ConsolidatedMatchData instance."""
        return ConsolidatedMatchData(**kwargs)
    
    def create_fixture_details(self, **kwargs) -> FixtureDetails:
        """Create a new FixtureDetails instance."""
        return FixtureDetails(**kwargs)
    
    def create_odds_data(self, **kwargs) -> OddsData:
        """Create a new OddsData instance."""
        return OddsData(**kwargs)
    
    def list_available_models(self) -> List[str]:
        """List all available model types."""
        return list(self.models.keys())
