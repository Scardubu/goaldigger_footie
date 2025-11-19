"""
Database utilities for the Streamlit dashboard.
Provides functions to initialize the database and manage data for the dashboard.
"""
import logging
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session

from database import (
    initialize_database,  # db_manager instance removed, class is imported below if needed
)
from database.db_manager import DatabaseManager
from database.schema import (
    League,
    Match,
    MatchStats,
    Odds,
    Prediction,
    Team,
    TeamStats,
    ValueBet,
)
from utils.config import Config, ConfigError

logger = logging.getLogger(__name__)

# Type variable for generic caching decorator
T = TypeVar('T')

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get or create the global database manager instance using the centralized
    initialize_database function from the database package.
    
    Returns:
        DatabaseManager: Database manager instance
    """
    global _db_manager
    if _db_manager is None:
        try:
            # Ensure environment variables are loaded for database configuration
            from dotenv import load_dotenv
            load_dotenv()
            
            # Determine database URI with fallbacks
            # First check environment variables
            db_uri = os.getenv('DATABASE_URI')
            
            # Then check from configuration
            if not db_uri:
                try:
                    db_uri = Config.get('database.uri')
                except Exception as config_err:
                    logger.warning(f"Could not get database URI from config: {config_err}")
            
            # Set up default SQLite database as final fallback
            if not db_uri:
                data_dir = os.getenv('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
                os.makedirs(data_dir, exist_ok=True)
                db_uri = f"sqlite:///{os.path.join(data_dir, 'football.db')}"
                logger.warning(f"Using default SQLite database URI: {db_uri}")
            
            # Now create database manager with explicit URI
            logger.info(f"Initializing database manager with URI: {db_uri}")
            _db_manager = DatabaseManager(db_uri)
            # Also ensure tables exist
            _db_manager.create_tables()
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Critical error obtaining database manager: {e}", exc_info=True)
            # Create a fallback in-memory SQLite database for minimal functionality
            try:
                logger.warning("Attempting to create fallback in-memory database")
                _db_manager = DatabaseManager("sqlite:///:memory:")
                _db_manager.create_tables()
                logger.info("Fallback in-memory database created")
            except Exception as fallback_err:
                logger.critical(f"Failed to create even a fallback database: {fallback_err}")
                raise
    return _db_manager


def get_database_session() -> Session:
    """
    Get a database session from the manager.
    
    Returns:
        Session: SQLAlchemy session
    """
    db_manager = get_db_manager()
    return db_manager.session_factory()


def db_cache(ttl_seconds: int = 300):
    """
    Decorator for caching database query results.
    
    Args:
        ttl_seconds: Time-to-live in seconds for cached results
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Use LRU cache with maxsize of 128 entries
        cached_func = lru_cache(maxsize=128)(func)
        
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Add timestamp to invalidate cache after TTL
            timestamp = int(time.time() / ttl_seconds)
            return cast(T, cached_func(*args, timestamp=timestamp, **kwargs))
        
        # Add cache clear method to the wrapper
        wrapper.cache_clear = cached_func.cache_clear  # type: ignore
        return wrapper
    
    return decorator


def get_available_leagues() -> List[Dict[str, Any]]:
    """
    Get all available leagues from the database.
    
    Returns:
        List of leagues with id, name, and country
    """
    db_manager = get_db_manager()
    
    try:
        with db_manager.session_scope() as session:
            leagues = session.query(League).order_by(League.country, League.name).all()
            
            return [{
                'id': league.id,
                'name': league.name,
                'country': league.country,
                'tier': league.tier
            } for league in leagues]
    
    except Exception as e:
        logger.error(f"Error getting available leagues: {e}")
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_value_bets(match_id: Optional[str] = None, min_edge: float = 0.05, 
                  min_confidence: float = 0.6, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get value betting opportunities from the database.
    
    Args:
        match_id: Optional match ID to filter by
        min_edge: Minimum edge percentage (0.05 = 5%)
        min_confidence: Minimum confidence score (0-1)
        limit: Maximum number of value bets to return
        
    Returns:
        List of value betting opportunities with match details
    """
    db_manager = get_db_manager()
    
    try:
        with db_manager.session_scope() as session:
            # Build query with joins to get all required data
            query = (
                session.query(
                    ValueBet,
                    Match,
                    Team.name.label("home_team"),
                    session.query(Team.name)
                    .filter(Team.id == Match.away_team_id)
                    .label('away_team'),
                    League.name.label('league_name')
                )
                .join(Match, ValueBet.match_id == Match.id)
                .join(League, Match.league_id == League.id)
                .join(Team, Match.home_team_id == Team.id)
                .filter(
                    ValueBet.edge >= min_edge,
                    ValueBet.confidence >= min_confidence,
                    Match.status == 'scheduled',  # Only upcoming matches
                    Match.match_date >= datetime.now()  # Only future matches
                )
                .order_by(ValueBet.edge.desc())
            )
            
            # Apply match filter if provided
            if match_id:
                query = query.filter(ValueBet.match_id == match_id)
            
            # Apply limit
            query = query.limit(limit)
            
            # Execute query
            results = query.all()
            
            # Format results
            value_bets = []
            for vb, match, home_team, away_team, league in results:
                value_bets.append({
                    'id': vb.id,
                    'match_id': vb.match_id,
                    'match_date': match.match_date,
                    'league': league,
                    'home_team': home_team,
                    'away_team': away_team,
                    'bet_type': vb.bet_type,
                    'selection': vb.selection,
                    'odds': vb.odds,
                    'fair_odds': vb.fair_odds,
                    'edge': vb.edge,
                    'kelly_stake': vb.kelly_stake,
                    'confidence': vb.confidence,
                    'created_at': vb.created_at
                })
            
            return value_bets
    
    except Exception as e:
        logger.error(f"Error getting value bets: {e}")
        return []


def update_database_from_api(force_update: bool = False) -> bool:
    """
    Update database with fresh data from APIs.
    
    Args:
        force_update: Whether to force update regardless of last update time
        
    Returns:
        Success status
    """
    try:
        # Import here to avoid circular imports
        import asyncio

        import nest_asyncio

        from scripts.data_pipeline.db_integrator import DataIntegrator

        # Apply nest_asyncio to allow nested event loops in Streamlit environment
        try:
            nest_asyncio.apply()
        except Exception as nest_err:
            logger.warning(f"Failed to apply nest_asyncio: {nest_err}, continuing anyway")
        
        # Check if database needs updating
        if not force_update:
            db_manager = get_db_manager()
            with db_manager.session_scope() as session:
                # Check when last update was performed
                latest_match = session.query(Match).order_by(Match.updated_at.desc()).first()
                if latest_match:
                    current_time = datetime.now(timezone.utc)
                    last_updated = latest_match.updated_at
                    if last_updated and last_updated.tzinfo is None:
                        last_updated = last_updated.replace(tzinfo=timezone.utc)
                    if last_updated and (current_time - last_updated).total_seconds() < 3600:
                        # Less than 1 hour since last update, skip
                        logger.info("Database was updated recently, skipping update")
                        return True
        
        # Create a robust database manager instance
        db_manager = get_db_manager()
        if not db_manager:
            logger.error("Failed to get database manager")
            return False
            
        # Run database update asynchronously
        data_integrator = DataIntegrator(db_manager)
        
        # Set a reasonable timeout for the operation
        try:
            # Use compatibility helper for robust loop acquisition
            try:
                from utils.asyncio_compat import ensure_loop
                loop = ensure_loop()
            except Exception:
                # Fallback legacy behavior
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            coro = asyncio.wait_for(
                data_integrator.update_upcoming_matches_all_leagues(),
                timeout=60.0  # 60 second timeout
            )
            if loop.is_running():
                # If we're already in an event loop (Streamlit async context), create a task and await via nest_asyncio applied loop
                result = loop.run_until_complete(coro) if hasattr(loop, 'run_until_complete') else asyncio.run(coro)
            else:
                result = loop.run_until_complete(coro)
            
            # Validate result is a dictionary
            if not isinstance(result, dict):
                logger.warning(f"Unexpected result type: {type(result)}, expected dict")
                result = {}
                
            success_count = sum(1 for r in result.values() if r)
            logger.info(f"Updated match data for {success_count}/{len(result) if result else 0} leagues")
            
            return success_count > 0
        except asyncio.TimeoutError:
            logger.error("Database update timed out after 60 seconds")
            return False
    
    except Exception as e:
        logger.error(f"Error updating database from API: {e}")
        return False


def get_upcoming_matches_count() -> int:
    """
    Get count of upcoming matches in the database.
    
    Returns:
        Count of upcoming matches
    """
    db_manager = get_db_manager()
    
    try:
        with db_manager.session_scope() as session:
            from datetime import timezone
            count = session.query(func.count(Match.id)).\
                filter(Match.status == 'scheduled', 
                       Match.match_date > datetime.now(timezone.utc)).scalar()
            return count or 0
    
    except Exception as e:
        logger.error(f"Error getting upcoming matches count: {e}")
        return 0


# Import time and datetime here to avoid circular imports
import time
from datetime import datetime, timedelta


@db_cache(ttl_seconds=300)  # Cache for 5 minutes
def get_cached_upcoming_matches(
    league_id: Optional[str] = None,
    team_id: Optional[str] = None,
    days_ahead: int = 7
) -> pd.DataFrame:
    """
    Get upcoming matches from the database with caching.
    
    Args:
        league_id: Optional league ID filter
        team_id: Optional team ID filter
        days_ahead: Number of days ahead to look for matches
        
    Returns:
        DataFrame of upcoming matches
    """
    db_manager = get_db_manager()
    
    try:
        with db_manager.session_scope() as session:
            from datetime import timezone
            query = session.query(Match).filter(
                Match.status == 'scheduled',
                Match.match_date <= datetime.now(timezone.utc) + timedelta(days=days_ahead)
            )
            
            if league_id:
                query = query.filter(Match.league_id == league_id)
            
            if team_id:
                query = query.filter(
                    (Match.home_team_id == team_id) | (Match.away_team_id == team_id)
                )
            
            matches = query.all()
            
            if not matches:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame([match.to_dict() for match in matches])
            
            # Add prediction columns if available
            if not df.empty and 'prediction' in df.columns:
                for idx, row in df.iterrows():
                    if row['prediction']:
                        for key, value in row['prediction'].items():
                            df.at[idx, f'prediction_{key}'] = value
                
                # Drop the original prediction dict column to avoid confusion
                if 'prediction' in df.columns:
                    df.drop('prediction', axis=1, inplace=True)
            
            # Add odds columns if available
            if not df.empty and 'odds' in df.columns:
                for idx, row in df.iterrows():
                    if row['odds']:
                        for key, value in row['odds'].items():
                            df.at[idx, f'odds_{key}'] = value
                
                # Drop the original odds dict column to avoid confusion
                if 'odds' in df.columns:
                    df.drop('odds', axis=1, inplace=True)
            
            # Ensure match_date is datetime
            if 'match_date' in df.columns:
                df['match_date'] = pd.to_datetime(df['match_date'])
            
            return df
    
    except Exception as e:
        logger.error(f"Error getting upcoming matches: {e}")
        return pd.DataFrame()
    
    # Add prediction columns if available
    if not df.empty and 'prediction' in df.iloc[0]:
        for idx, row in df.iterrows():
            if 'prediction' in row and row['prediction']:
                for key, value in row['prediction'].items():
                    df.at[idx, f'prediction_{key}'] = value
        
        # Drop the original prediction dict column to avoid confusion
        if 'prediction' in df.columns:
            df.drop('prediction', axis=1, inplace=True)
    
    # Add odds columns if available
    if not df.empty and 'odds' in df.iloc[0]:
        for idx, row in df.iterrows():
            if 'odds' in row and row['odds']:
                for key, value in row['odds'].items():
                    df.at[idx, f'odds_{key}'] = value
        
        # Drop the original odds dict column to avoid confusion
        if 'odds' in df.columns:
            df.drop('odds', axis=1, inplace=True)
    
    # Ensure match_date is datetime
    if 'match_date' in df.columns:
        df['match_date'] = pd.to_datetime(df['match_date'])
    
    return df

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_match_details(match_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific match with caching.
    
    Args:
        match_id: Match ID
        
    Returns:
        Dictionary with match details
    """
    return get_match_details(match_id)

def analyze_match_value_bets(match_id: str) -> List[Dict[str, Any]]:
    """
    Analyze a match for value betting opportunities.
    This is not cached to ensure fresh analysis.
    
    Args:
        match_id: Match ID
        
    Returns:
        List of value bet dictionaries
    """
    return analyze_match_for_value_bets(match_id)

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_cached_db_stats() -> Dict[str, Any]:
    """
    Get database statistics for the dashboard with caching.
    
    Returns:
        Dictionary with database statistics
    """
    return get_db_stats()

def convert_match_details_to_dataframe(match_details: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert match details dictionary to a DataFrame format for compatibility with existing code.
    
    Args:
        match_details: Match details dictionary
        
    Returns:
        DataFrame with match details
    """
    # Create a single-row DataFrame with match details
    df = pd.DataFrame([match_details])
    
    return df
