#!/usr/bin/env python
"""
Database Population Script for GoalDiggers Platform

This script populates the database with high-quality historical match data
from the top six football leagues to ensure comprehensive training data.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path for robust imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.setup_env import setup_environment

# Configure environment first
logger = setup_environment()

from api.understat_client import UnderstatAPIClient
from database.db_manager import DatabaseManager
from database.schema import League, Match, MatchStats, Team, TeamStats
from scripts.scrapers.football_data_scraper import FootballDataScraper
from scripts.scrapers.transfermarkt_scraper import TransfermarktScraper
from utils.config import Config

# Configure logging
logger = logging.getLogger(__name__)


def populate_leagues():
    """Populate the database with the top six European football leagues."""
    db = DatabaseManager()
    
    top_leagues = [
        {
            "id": "PL",
            "name": "Premier League",
            "country": "England",
            "tier": 1,
            "api_id": "PL",
            "season_start": datetime(2024, 8, 9),
            "season_end": datetime(2025, 5, 25)
        },
        {
            "id": "PD",
            "name": "LaLiga",
            "country": "Spain",
            "tier": 1,
            "api_id": "PD",
            "season_start": datetime(2024, 8, 16),
            "season_end": datetime(2025, 5, 25)
        },
        {
            "id": "BL1",
            "name": "Bundesliga",
            "country": "Germany",
            "tier": 1,
            "api_id": "BL1",
            "season_start": datetime(2024, 8, 23),
            "season_end": datetime(2025, 5, 17)
        },
        {
            "id": "SA",
            "name": "Serie A",
            "country": "Italy",
            "tier": 1,
            "api_id": "SA",
            "season_start": datetime(2024, 8, 17),
            "season_end": datetime(2025, 5, 25)
        },
        {
            "id": "FL1",
            "name": "Ligue 1",
            "country": "France",
            "tier": 1,
            "api_id": "FL1",
            "season_start": datetime(2024, 8, 16),
            "season_end": datetime(2025, 5, 17)
        },
        {
            "id": "DED",
            "name": "Eredivisie",
            "country": "Netherlands",
            "tier": 1,
            "api_id": "DED",
            "season_start": datetime(2024, 8, 10),
            "season_end": datetime(2025, 5, 11)
        }
    ]
    
    logger.info(f"Populating database with {len(top_leagues)} top football leagues")
    
    with db.session_scope() as session:
        for league_data in top_leagues:
            # Check if league already exists
            existing = session.query(League).filter(League.id == league_data["id"]).first()
            if existing:
                logger.info(f"League {league_data['name']} already exists, updating")
                for key, value in league_data.items():
                    setattr(existing, key, value)
            else:
                logger.info(f"Adding league: {league_data['name']}")
                league = League(**league_data)
                session.add(league)
    
    logger.info("Leagues successfully populated")


async def populate_teams():
    """Populate the database with teams from the top six leagues using multiple sources."""
    db = DatabaseManager()
    
    # Initialize scrapers
    football_data_scraper = FootballDataScraper()
    transfermarkt_scraper = TransfermarktScraper()
    
    # Get leagues from the database
    with db.session_scope() as session:
        leagues = session.query(League).all()
        
        for league in leagues:
            logger.info(f"Fetching teams for league: {league.name}")
            
            # First try football-data.org API
            try:
                teams_data = await football_data_scraper.get_teams(league.id)
                logger.info(f"Found {len(teams_data)} teams from Football-Data API")
            except Exception as e:
                logger.error(f"Error fetching teams from Football-Data API: {e}")
                teams_data = []
            
            # Supplement with Transfermarkt data for richer team information
            try:
                transfermarkt_teams = await transfermarkt_scraper.get_teams(league.id)
                logger.info(f"Found {len(transfermarkt_teams)} teams from Transfermarkt")
                
                # Merge data sources
                teams_enriched = enrich_teams_data(teams_data, transfermarkt_teams)
            except Exception as e:
                logger.error(f"Error fetching teams from Transfermarkt: {e}")
                teams_enriched = teams_data
            
            # Save to database
            for team_data in teams_enriched:
                # Ensure team_data has required fields
                if "id" not in team_data or "name" not in team_data:
                    logger.warning(f"Skipping team with missing required fields: {team_data}")
                    continue
                    
                # Set league_id
                team_data["league_id"] = league.id
                
                # Check if team exists
                existing = session.query(Team).filter(Team.id == team_data["id"]).first()
                if existing:
                    logger.info(f"Updating team: {team_data['name']}")
                    for key, value in team_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                else:
                    logger.info(f"Adding team: {team_data['name']}")
                    # Ensure TLA is set (fallback to empty string if missing)
                    if "tla" not in team_data or not team_data["tla"]:
                        team_data["tla"] = team_data.get("short_name", "")[:3].upper() if team_data.get("short_name") else ""
                    team = Team(**team_data)
                    session.add(team)
    
    logger.info("Teams successfully populated")


def enrich_teams_data(primary_teams, secondary_teams):
    """Merge team data from multiple sources, prioritizing primary source."""
    team_map = {team.get("name", ""): team for team in primary_teams if "name" in team}
    
    # Add data from secondary source if team exists in primary
    for sec_team in secondary_teams:
        if "name" not in sec_team:
            continue
            
        name = sec_team["name"]
        if name in team_map:
            # For each field in secondary team, add it to primary if not already present
            for key, value in sec_team.items():
                if key not in team_map[name] or not team_map[name][key]:
                    team_map[name][key] = value
    
    return list(team_map.values())


async def populate_historical_matches(days=365):
    """
    Populate the database with historical matches from the past year.
    
    Args:
        days: Number of days of historical data to fetch
    """
    db = DatabaseManager()
    football_data_scraper = FootballDataScraper()
    understat_client = UnderstatAPIClient()
    
    # Get date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Fetching match data from {start_date.date()} to {end_date.date()}")
    
    # Get leagues from the database
    with db.session_scope() as session:
        leagues = session.query(League).all()
        
        for league in leagues:
            logger.info(f"Fetching matches for league: {league.name}")
            
            # Get matches from Football-Data API
            try:
                matches = await football_data_scraper.get_matches(
                    league_code=league.id,
                    date_from=start_date,
                    date_to=end_date
                )
                logger.info(f"Found {len(matches)} matches from Football-Data API")
            except Exception as e:
                logger.error(f"Error fetching matches from Football-Data API: {e}")
                matches = []
            
            # Get additional stats from Understat
            try:
                league_mapping = {
                    "PL": "premier_league",
                    "PD": "la_liga",
                    "BL1": "bundesliga",
                    "SA": "serie_a",
                    "FL1": "ligue_1",
                    "DED": "eredivisie"
                }

                understat_league_code = league_mapping.get(league.id)
                matches_enriched = matches

                if understat_league_code:
                    season = UnderstatAPIClient._default_season(end_date)
                    understat_df = await understat_client.get_league_matches(
                        understat_league_code,
                        season
                    )

                    if understat_df is not None and not understat_df.empty:
                        understat_matches = []
                        for _, row in understat_df.iterrows():
                            match_record = {
                                "id": str(row.get("id")),
                                "match_id": str(row.get("id")),
                                "home_team": row.get("home_team"),
                                "away_team": row.get("away_team"),
                                "home_team_id": str(row.get("home_team_id", "")),
                                "away_team_id": str(row.get("away_team_id", "")),
                                "match_date": row.get("match_date"),
                                "stats": {
                                    "home_xg": row.get("home_xG"),
                                    "away_xg": row.get("away_xG"),
                                    "home_goals": row.get("home_goals"),
                                    "away_goals": row.get("away_goals")
                                }
                            }
                            understat_matches.append(match_record)

                        logger.info(
                            "Found %s matches from Understat for %s",
                            len(understat_matches),
                            league.id
                        )

                        matches_enriched = enrich_matches_data(matches, understat_matches)
                    else:
                        logger.warning("No Understat matches returned for league %s", league.id)
                else:
                    logger.debug("No Understat mapping for league %s", league.id)
            except Exception as e:
                logger.error(f"Error fetching matches from Understat: {e}")
                matches_enriched = matches
            
            # Save matches to database
            for match_data in matches_enriched:
                # Ensure match has the required fields
                if "id" not in match_data or "home_team_id" not in match_data or "away_team_id" not in match_data:
                    logger.warning(f"Skipping match with missing required fields: {match_data}")
                    continue
                
                # Set league_id
                match_data["league_id"] = league.id
                
                # Extract match stats if available
                match_stats_data = match_data.pop("stats", None)
                
                # Check if match exists
                existing = session.query(Match).filter(Match.id == match_data["id"]).first()
                if existing:
                    logger.debug(f"Updating match: {match_data.get('id')}")
                    for key, value in match_data.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                    match_obj = existing
                else:
                    logger.debug(f"Adding match: {match_data.get('id')}")
                    match_obj = Match(**match_data)
                    session.add(match_obj)
                
                # Ensure session has match_obj before adding stats
                session.flush()
                
                # Add match stats if available
                if match_stats_data:
                    # Check if stats already exist
                    existing_stats = session.query(MatchStats).filter(
                        MatchStats.match_id == match_obj.id
                    ).first()
                    
                    if existing_stats:
                        logger.debug(f"Updating match stats for match: {match_obj.id}")
                        for key, value in match_stats_data.items():
                            if hasattr(existing_stats, key):
                                setattr(existing_stats, key, value)
                    else:
                        logger.debug(f"Adding match stats for match: {match_obj.id}")
                        stats_data = {
                            "id": f"stats_{match_obj.id}",
                            "match_id": match_obj.id,
                            **match_stats_data
                        }
                        match_stats = MatchStats(**stats_data)
                        session.add(match_stats)
    
    logger.info("Historical matches successfully populated")
    
    # Update team statistics based on match results
    await update_team_statistics()

    try:
        await understat_client.close()
    except Exception:
        pass


def enrich_matches_data(primary_matches, secondary_matches):
    """Merge match data from multiple sources, prioritizing primary source."""
    # Create a map of matches by a composite key of teams and date
    match_map = {}
    for match in primary_matches:
        if "home_team_id" not in match or "away_team_id" not in match or "match_date" not in match:
            continue
        
        key = f"{match['home_team_id']}_{match['away_team_id']}_{match['match_date']}"
        match_map[key] = match
    
    # Enrich with secondary source
    for sec_match in secondary_matches:
        if "home_team_id" not in sec_match or "away_team_id" not in sec_match or "match_date" not in sec_match:
            continue
        
        key = f"{sec_match['home_team_id']}_{sec_match['away_team_id']}_{sec_match['match_date']}"
        if key in match_map:
            # For each field in secondary match, add it to primary if not already present
            for field, value in sec_match.items():
                if field not in match_map[key] or not match_map[key][field]:
                    match_map[key][field] = value
            
            # Special handling for stats - merge them
            if "stats" in sec_match:
                if "stats" not in match_map[key]:
                    match_map[key]["stats"] = {}
                
                for stat_key, stat_value in sec_match["stats"].items():
                    if stat_key not in match_map[key]["stats"] or not match_map[key]["stats"][stat_key]:
                        match_map[key]["stats"][stat_key] = stat_value
    
    return list(match_map.values())


async def update_team_statistics():
    """Update team statistics based on match results."""
    db = DatabaseManager()
    
    with db.session_scope() as session:
        # Get all teams
        teams = session.query(Team).all()
        
        for team in teams:
            logger.info(f"Updating statistics for team: {team.name}")
            
            # Get all finished matches for this team
            home_matches = session.query(Match).filter(
                Match.home_team_id == team.id,
                Match.status == "FINISHED"
            ).all()
            
            away_matches = session.query(Match).filter(
                Match.away_team_id == team.id,
                Match.status == "FINISHED"
            ).all()
            
            # Calculate statistics
            matches_played = len(home_matches) + len(away_matches)
            home_wins = sum(1 for m in home_matches if m.home_score > m.away_score)
            home_draws = sum(1 for m in home_matches if m.home_score == m.away_score)
            home_losses = sum(1 for m in home_matches if m.home_score < m.away_score)
            
            away_wins = sum(1 for m in away_matches if m.away_score > m.home_score)
            away_draws = sum(1 for m in away_matches if m.away_score == m.home_score)
            away_losses = sum(1 for m in away_matches if m.away_score < m.home_score)
            
            wins = home_wins + away_wins
            draws = home_draws + away_draws
            losses = home_losses + away_losses
            
            goals_for = sum(m.home_score for m in home_matches if m.home_score is not None) + \
                        sum(m.away_score for m in away_matches if m.away_score is not None)
            
            goals_against = sum(m.away_score for m in home_matches if m.away_score is not None) + \
                            sum(m.home_score for m in away_matches if m.home_score is not None)
            
            points = wins * 3 + draws
            
            # Get last 5 matches for form
            all_matches = sorted(
                home_matches + away_matches,
                key=lambda m: m.match_date if m.match_date else datetime.min,
                reverse=True
            )
            
            form_matches = all_matches[:5]
            form_last_5 = ""
            for match in form_matches:
                if match.home_team_id == team.id:
                    if match.home_score > match.away_score:
                        form_last_5 += "W"
                    elif match.home_score == match.away_score:
                        form_last_5 += "D"
                    else:
                        form_last_5 += "L"
                else:  # away team
                    if match.away_score > match.home_score:
                        form_last_5 += "W"
                    elif match.away_score == match.home_score:
                        form_last_5 += "D"
                    else:
                        form_last_5 += "L"
            
            # Create or update team stats
            stats_data = {
                "team_id": team.id,
                "season": str(datetime.now().year),
                "league_id": team.league_id,
                "matches_played": matches_played,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "points": points,
                "form_last_5": form_last_5,
                "home_wins": home_wins,
                "home_draws": home_draws,
                "home_losses": home_losses,
                "away_wins": away_wins,
                "away_draws": away_draws,
                "away_losses": away_losses
            }
            
            # Get existing stats or create new one
            existing = session.query(TeamStats).filter(
                TeamStats.team_id == team.id,
                TeamStats.season == str(datetime.now().year)
            ).first()
            
            if existing:
                logger.info(f"Updating team stats for {team.name}")
                for key, value in stats_data.items():
                    setattr(existing, key, value)
            else:
                logger.info(f"Creating team stats for {team.name}")
                team_stats = TeamStats(**stats_data)
                session.add(team_stats)
    
    logger.info("Team statistics updated successfully")


async def generate_reference_data():
    """Generate reference data CSV for ML training."""
    from generate_reference_data import generate_reference_csv
    
    logger.info("Generating reference data CSV for ML training")
    generate_reference_csv()
    logger.info("Reference data generated successfully")


async def main():
    """Main function to populate the database."""
    parser = argparse.ArgumentParser(description="Populate the database with high-quality historical match data.")
    parser.add_argument("--days", type=int, default=365, help="Number of days of historical data to fetch (default: 365)")
    parser.add_argument("--leagues-only", action="store_true", help="Only populate leagues data")
    parser.add_argument("--teams-only", action="store_true", help="Only populate teams data")
    parser.add_argument("--matches-only", action="store_true", help="Only populate matches data")
    parser.add_argument("--generate-reference", action="store_true", help="Generate reference data CSV for ML training")
    
    args = parser.parse_args()
    
    logger.info("Starting database population process")
    
    # Determine which steps to run
    run_all = not (args.leagues_only or args.teams_only or args.matches_only or args.generate_reference)
    
    try:
        # Initialize database structure
        db = DatabaseManager()
        db.create_tables()
        
        if run_all or args.leagues_only:
            populate_leagues()
        
        if run_all or args.teams_only:
            await populate_teams()
        
        if run_all or args.matches_only:
            await populate_historical_matches(days=args.days)
        
        if run_all or args.generate_reference:
            await generate_reference_data()
        
        logger.info("Database population completed successfully")
    
    except Exception as e:
        logger.error(f"Error during database population: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
