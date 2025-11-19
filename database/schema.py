"""
Database schema definitions for the football betting insights platform.
Provides SQLAlchemy ORM models for storing match data, predictions, and other information.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class League(Base):
    """League information table."""
    __tablename__ = 'leagues'
    
    id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50), nullable=False)
    tier = Column(Integer, default=1)  # League tier (1 = top tier)
    api_id = Column(String(20))  # ID in external API
    season_start = Column(DateTime)
    season_end = Column(DateTime)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    
    # Relationships
    teams = relationship("Team", back_populates="league")
    matches = relationship("Match", back_populates="league")

class Team(Base):
    """Team information table."""
    __tablename__ = 'teams'
    
    id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    short_name = Column(String(20))
    tla = Column(String(3))  # Three-letter abbreviation/code (e.g., ARS for Arsenal)
    league_id = Column(String(20), ForeignKey('leagues.id'))
    venue = Column(String(100))
    venue_capacity = Column(Integer)
    api_id = Column(String(20))  # ID in external API
    crest_url = Column(String(255))  # URL to team logo/crest
    country = Column(String(50))  # Team's country
    country_code = Column(String(2))  # ISO country code
    team_flag = Column(String(10))  # Team flag emoji
    country_flag = Column(String(10))  # Country flag emoji
    primary_color = Column(String(7))  # Primary team color hex code
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    
    # Relationships
    league = relationship("League", back_populates="teams")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    team_stats = relationship("TeamStats", back_populates="team")

class Match(Base):
    """Match information table."""
    __tablename__ = 'matches'
    
    id = Column(String(50), primary_key=True, nullable=False)
    league_id = Column(String(20), ForeignKey('leagues.id'))
    home_team_id = Column(String(20), ForeignKey('teams.id'))
    away_team_id = Column(String(20), ForeignKey('teams.id'))
    match_date = Column(DateTime, nullable=False)
    status = Column(String(20), default='scheduled')  # scheduled, in_play, finished, postponed, cancelled
    matchday = Column(Integer)
    venue = Column(String(100))
    referee = Column(String(100))
    home_score = Column(Integer)
    away_score = Column(Integer)
    api_id = Column(String(50))  # ID in external API
    competition = Column(String(100))  # Added competition field for compatibility
    last_synced_at = Column(DateTime, default=_utcnow)  # Track when data was last refreshed
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    
    # Relationships
    league = relationship("League", back_populates="matches")
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    match_stats = relationship("MatchStats", back_populates="match", uselist=False)
    predictions = relationship("Prediction", back_populates="match")
    odds = relationship("Odds", back_populates="match")
    
    def to_dict(self):
        """Convert match to dictionary for API responses and UI rendering."""
        result = {
            "match_id": self.id,  # Ensure match_id is always available
            "id": self.id,
            "league_id": self.league_id,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "match_date": self.match_date.isoformat() if self.match_date else None,
            "status": self.status,
            "matchday": self.matchday,
            "venue": self.venue,
            "referee": self.referee,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "api_id": self.api_id,
            "competition": self.competition
        }
        
        # Add team names if relationships are loaded
        if hasattr(self, 'home_team') and self.home_team:
            result["home_team"] = self.home_team.name
        if hasattr(self, 'away_team') and self.away_team:
            result["away_team"] = self.away_team.name
        if hasattr(self, 'league') and self.league:
            result["competition"] = self.league.name
            
        return result

class MatchStats(Base):
    """Match statistics table."""
    __tablename__ = 'match_stats'
    
    id = Column(String(50), primary_key=True)
    match_id = Column(String(50), ForeignKey('matches.id'))
    home_possession = Column(Float)
    away_possession = Column(Float)
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_yellow_cards = Column(Integer)
    away_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    away_red_cards = Column(Integer)
    extra_stats = Column(JSON)  # Additional stats in JSON format
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    
    # Relationships
    match = relationship("Match", back_populates="match_stats")

class TeamStats(Base):
    """Team statistics table for season/competition."""
    __tablename__ = 'team_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(String(20), ForeignKey('teams.id'))
    season = Column(String(20))
    league_id = Column(String(20), ForeignKey('leagues.id'))
    matches_played = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    goals_for = Column(Integer, default=0)
    goals_against = Column(Integer, default=0)
    points = Column(Integer, default=0)
    position = Column(Integer)
    form_last_5 = Column(String(5))  # e.g., "WDLWW"
    home_wins = Column(Integer, default=0)
    home_draws = Column(Integer, default=0)
    home_losses = Column(Integer, default=0)
    away_wins = Column(Integer, default=0)
    away_draws = Column(Integer, default=0)
    away_losses = Column(Integer, default=0)
    xg_for = Column(Float, default=0.0)
    xg_against = Column(Float, default=0.0)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    
    # Relationships
    team = relationship("Team", back_populates="team_stats")

class Prediction(Base):
    """Match outcome prediction table."""
    __tablename__ = 'predictions'

    id = Column(String(50), primary_key=True)
    match_id = Column(String(50), ForeignKey('matches.id'))
    model_version = Column(String(100))
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    home_score_pred = Column(Float)  # Predicted home team score
    away_score_pred = Column(Float)  # Predicted away team score
    predicted_outcome = Column(String(20))  # home_win, draw, away_win
    confidence = Column(Float)
    insights_json = Column(Text)
    feature_importance = Column(JSON)  # Feature importance data
    timestamp = Column(DateTime, default=_utcnow)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationships
    match = relationship("Match", back_populates="predictions")

class Odds(Base):
    """Match odds table."""
    __tablename__ = 'odds'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(50), ForeignKey('matches.id'))
    bookmaker = Column(String(50))
    home_win = Column(Float)
    draw = Column(Float)
    away_win = Column(Float)
    over_under_2_5_over = Column(Float)
    over_under_2_5_under = Column(Float)
    both_teams_to_score_yes = Column(Float)
    both_teams_to_score_no = Column(Float)
    timestamp = Column(DateTime, default=_utcnow)
    
    # Relationships
    match = relationship("Match", back_populates="odds")

class ValueBet(Base):
    """Value betting opportunities identified by the system."""
    __tablename__ = 'value_bets'
    
    id = Column(String(50), primary_key=True)
    match_id = Column(String(50), ForeignKey('matches.id'))
    bet_type = Column(String(50))  # home_win, draw, away_win, over_2.5, etc.
    odds = Column(Float)
    expected_value = Column(Float)
    confidence = Column(Float)
    reason = Column(Text)
    recommended_stake = Column(Float)  # Optional recommended stake size
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)
    
    # Relationships
    match = relationship("Match")


class APICache(Base):
    """Cache for API responses to minimize external API calls."""
    __tablename__ = 'api_cache'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    endpoint = Column(String(255), nullable=False)  # API endpoint or URL
    parameters = Column(Text)  # Serialized parameters (e.g., JSON)
    response = Column(Text, nullable=False)  # Cached response data
    status_code = Column(Integer)  # HTTP status code
    created_at = Column(DateTime, default=_utcnow)
    expires_at = Column(DateTime)  # When this cache entry expires
    source = Column(String(50))  # API source name (e.g., 'football-data', 'odds-api')
    
    # Index to speed up lookups by endpoint and parameters
    __table_args__ = (
        # Index for faster lookup
        # Note: SQLite doesn't support functional indexes, so this is a standard index
        {'sqlite_autoincrement': True},
    )


class ScrapedData(Base):
    """Table to store scraped data from various sources."""
    __tablename__ = 'scraped_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(100), nullable=False)  # Source website or name
    data_type = Column(String(50), nullable=False)  # Type of data (e.g., 'match', 'odds', 'team_stats')
    external_id = Column(String(100))  # ID from the source if available
    url = Column(String(500))  # URL from which data was scraped
    match_id = Column(String(50), ForeignKey('matches.id'), nullable=True)  # Related match if applicable
    team_id = Column(String(20), ForeignKey('teams.id'), nullable=True)  # Related team if applicable
    content = Column(JSON, nullable=False)  # Actual scraped data in JSON format
    meta_info = Column(JSON)  # Additional metadata about the scrape (e.g., coordinates for weather data)
    timestamp = Column(DateTime, default=_utcnow)  # When the data was scraped
    last_updated = Column(DateTime, default=_utcnow, onupdate=_utcnow)

def init_db(db_uri: str) -> Tuple[Any, Any]:
    """
    Initialize the database with the defined schema.
    
    Args:
        db_uri: Database URI (e.g., 'sqlite:///football.db')
        
    Returns:
        Tuple of (engine, session_factory)
    """
    engine = create_engine(db_uri)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    
    logger.info(f"Initialized database at {db_uri}")
    
    return engine, session_factory

# Performance optimization indexes
# Match table indexes
Index('idx_match_date', Match.match_date)
Index('idx_match_league', Match.league_id)
Index('idx_match_teams', Match.home_team_id, Match.away_team_id)

# Prediction table indexes
Index('idx_prediction_match', Prediction.match_id)

# Odds table indexes
Index('idx_odds_match', Odds.match_id)

# Team form indexes - Commented out as TeamForm table is not defined
# Index('idx_form_team', TeamForm.team_id)
# Index('idx_form_date', TeamForm.date)
