"""
Database integration module for the Streamlit dashboard.
Provides functions to access and manipulate database data for presentation in the UI.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np

from database import db_manager
from database.schema import League, Team, Match, MatchStats, TeamStats, Prediction, Odds, ValueBet
from scripts.models.value_bet_analyzer import ValueBetAnalyzer
from utils.system_monitor import SystemMonitor, start_operation, end_operation

logger = logging.getLogger(__name__)

def get_available_leagues() -> List[Dict[str, Any]]:
    """
    Get all available leagues from the database.
    
    Returns:
        List of league dictionaries with id, name, and country
    """
    start_operation("get_available_leagues")
    
    try:
        with db_manager.session_scope() as session:
            leagues = session.query(League).order_by(League.name).all()
            
            result = []
            for league in leagues:
                result.append({
                    "id": league.id,
                    "name": league.name,
                    "country": league.country,
                    "tier": league.tier
                })
                
            return result
    except Exception as e:
        logger.error(f"Error getting available leagues: {e}")
        return []
    finally:
        end_operation("get_available_leagues")

def get_league_teams(league_id: str) -> List[Dict[str, Any]]:
    """
    Get all teams in a league from the database.
    
    Args:
        league_id: League ID
        
    Returns:
        List of team dictionaries with id and name
    """
    start_operation("get_league_teams")
    
    try:
        with db_manager.session_scope() as session:
            teams = session.query(Team).filter(
                Team.league_id == league_id
            ).order_by(Team.name).all()
            
            result = []
            for team in teams:
                result.append({
                    "id": team.id,
                    "name": team.name,
                    "short_name": team.short_name or team.name
                })
                
            return result
    except Exception as e:
        logger.error(f"Error getting teams for league {league_id}: {e}")
        return []
    finally:
        end_operation("get_league_teams")

def get_upcoming_matches(
    league_id: Optional[str] = None,
    team_id: Optional[str] = None,
    days_ahead: int = 7
) -> List[Dict[str, Any]]:
    """
    Get upcoming matches from the database with optional filters.
    
    Args:
        league_id: Optional league ID filter
        team_id: Optional team ID filter
        days_ahead: Number of days ahead to look for matches
        
    Returns:
        List of match dictionaries with details
    """
    start_operation("get_upcoming_matches")
    
    try:
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_ahead)
        
        with db_manager.session_scope() as session:
            query = session.query(Match).filter(
                Match.match_date >= start_date,
                Match.match_date <= end_date,
                Match.status == 'scheduled'
            )
            
            if league_id:
                query = query.filter(Match.league_id == league_id)
                
            if team_id:
                query = query.filter(
                    (Match.home_team_id == team_id) | (Match.away_team_id == team_id)
                )
                
            matches = query.order_by(Match.match_date).all()
            
            result = []
            for match in matches:
                # Get home and away team names
                home_team = session.query(Team).filter(Team.id == match.home_team_id).first()
                away_team = session.query(Team).filter(Team.id == match.away_team_id).first()
                league = session.query(League).filter(League.id == match.league_id).first()
                
                # Get latest prediction
                prediction = session.query(Prediction).filter(
                    Prediction.match_id == match.id
                ).order_by(Prediction.created_at.desc()).first()
                
                # Get average odds
                odds_query = session.query(
                    func.avg(Odds.home_win).label('avg_home_win'),
                    func.avg(Odds.draw).label('avg_draw'),
                    func.avg(Odds.away_win).label('avg_away_win')
                ).filter(Odds.match_id == match.id)
                
                avg_odds = odds_query.first()
                
                # Get value bets
                value_bets = session.query(ValueBet).filter(
                    ValueBet.match_id == match.id,
                    ValueBet.confidence.in_(['Medium', 'High'])
                ).all()
                
                match_dict = {
                    "id": match.id,
                    "match_date": match.match_date,
                    "home_team": home_team.name if home_team else match.home_team_id,
                    "away_team": away_team.name if away_team else match.away_team_id,
                    "league": league.name if league else match.league_id,
                    "status": match.status,
                    "venue": match.venue
                }
                
                # Add prediction if available
                if prediction:
                    match_dict["prediction"] = {
                        "home_win": prediction.home_win_prob,
                        "draw": prediction.draw_prob,
                        "away_win": prediction.away_win_prob,
                        "home_score": prediction.home_score_pred,
                        "away_score": prediction.away_score_pred,
                        "confidence": prediction.confidence,
                        "model": prediction.model_name
                    }
                
                # Add odds if available
                if avg_odds and avg_odds.avg_home_win:
                    match_dict["odds"] = {
                        "home_win": avg_odds.avg_home_win,
                        "draw": avg_odds.avg_draw,
                        "away_win": avg_odds.avg_away_win
                    }
                
                # Add value bets if available
                if value_bets:
                    match_dict["value_bets"] = []
                    for vb in value_bets:
                        match_dict["value_bets"].append({
                            "bet_type": vb.bet_type,
                            "edge": vb.edge,
                            "confidence": vb.confidence
                        })
                
                result.append(match_dict)
            
            return result
    except Exception as e:
        logger.error(f"Error getting upcoming matches: {e}")
        return []
    finally:
        end_operation("get_upcoming_matches")

def get_match_details(match_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific match.
    
    Args:
        match_id: Match ID
        
    Returns:
        Dictionary with match details
    """
    start_operation("get_match_details")
    
    try:
        match_details = db_manager.get_match_with_details(match_id)
        
        if not match_details:
            logger.warning(f"No details found for match {match_id}")
            return {}
        
        # Convert match details to DataFrame format for compatibility with existing code
        match_df = pd.DataFrame([match_details["match"]])
        
        # Add stats if available
        if "stats" in match_details:
            for key, value in match_details["stats"].items():
                match_df[key] = value
        
        # Add prediction if available
        if "prediction" in match_details:
            for key, value in match_details["prediction"].items():
                match_df[f"prediction_{key}"] = value
        
        # Add odds if available
        if "odds" in match_details:
            # Find the best odds for each market
            best_odds = {
                "home_win": max([odds.get("home_win", 0) for odds in match_details["odds"].values()]),
                "draw": max([odds.get("draw", 0) for odds in match_details["odds"].values()]),
                "away_win": max([odds.get("away_win", 0) for odds in match_details["odds"].values()])
            }
            
            for key, value in best_odds.items():
                match_df[f"odds_{key}"] = value
        
        # Add value bets if available
        if "value_bets" in match_details:
            value_bets_list = []
            for vb in match_details["value_bets"]:
                value_bets_list.append({
                    "bet_type": vb["bet_type"],
                    "edge": vb["edge"],
                    "kelly_stake": vb["kelly_stake"],
                    "confidence": vb["confidence"]
                })
            
            match_df["value_bets"] = [value_bets_list]
        
        return match_df.to_dict('records')[0]
    except Exception as e:
        logger.error(f"Error getting match details for {match_id}: {e}")
        return {}
    finally:
        end_operation("get_match_details")

def analyze_match_for_value_bets(match_id: str) -> List[Dict[str, Any]]:
    """
    Analyze a match for value betting opportunities.
    
    Args:
        match_id: Match ID
        
    Returns:
        List of value bet dictionaries
    """
    start_operation("analyze_match_for_value_bets")
    
    try:
        analyzer = ValueBetAnalyzer()
        value_bets = analyzer.analyze_match(match_id)
        return value_bets
    except Exception as e:
        logger.error(f"Error analyzing match {match_id} for value bets: {e}")
        return []
    finally:
        end_operation("analyze_match_for_value_bets")

def get_db_stats() -> Dict[str, Any]:
    """
    Get database statistics for the dashboard.
    
    Returns:
        Dictionary with database statistics
    """
    start_operation("get_db_stats")
    
    try:
        with db_manager.session_scope() as session:
            # Count entities
            league_count = session.query(League).count()
            team_count = session.query(Team).count()
            match_count = session.query(Match).count()
            prediction_count = session.query(Prediction).count()
            odds_count = session.query(Odds).count()
            value_bet_count = session.query(ValueBet).count()
            
            # Get upcoming match count
            upcoming_match_count = session.query(Match).filter(
                Match.match_date >= datetime.now(),
                Match.status == 'scheduled'
            ).count()
            
            # Get matches with predictions count
            matches_with_predictions = session.query(Match.id).join(
                Prediction, Match.id == Prediction.match_id
            ).distinct().count()
            
            # Get matches with odds count
            matches_with_odds = session.query(Match.id).join(
                Odds, Match.id == Odds.match_id
            ).distinct().count()
            
            # Get matches with value bets count
            matches_with_value_bets = session.query(Match.id).join(
                ValueBet, Match.id == ValueBet.match_id
            ).distinct().count()
            
            return {
                "leagues": league_count,
                "teams": team_count,
                "matches": match_count,
                "upcoming_matches": upcoming_match_count,
                "predictions": prediction_count,
                "odds": odds_count,
                "value_bets": value_bet_count,
                "matches_with_predictions": matches_with_predictions,
                "matches_with_odds": matches_with_odds,
                "matches_with_value_bets": matches_with_value_bets,
                "last_updated": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting database statistics: {e}")
        return {}
    finally:
        end_operation("get_db_stats")

# Import at the bottom to avoid circular imports
from sqlalchemy import func
