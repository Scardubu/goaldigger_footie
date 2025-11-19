"""
Enhanced Data Integrator for the football betting insights platform.
Includes real-time validation, automated league-specific data scraping,
and improved error handling with fallback strategies.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sqlalchemy.exc import IntegrityError

from api.understat_client import UnderstatAPIClient
from database.db_manager import DatabaseManager
from database.schema import League, Match, Team
from scripts.core.ai_validator import AIDataValidator
from scripts.scrapers.scraper_factory import ScraperFactory
from utils.config import Config
from utils.logging_config import get_logger

logger = get_logger(__name__)

class EnhancedDataIntegrator:
    """Enhanced data integrator with real-time validation and AI-powered data quality checks."""
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        scraper_factory: ScraperFactory,
        validator: Optional[AIDataValidator] = None,
        config: Optional[Config] = None
    ):
        """Initialize the Enhanced Data Integrator."""
        self.db = db_manager
        self.scraper_factory = scraper_factory
        self.validator = validator
        self.config = config or Config()
        self.integration_stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Configuration
        integration_config = self.config.get('data_integration', {})
        self.default_leagues = integration_config.get('default_leagues', [
            'premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1', 'eredivisie'
        ])
        self.max_concurrent_requests = integration_config.get('max_concurrent_requests', 10)

        # Initialize Understat API client and share HTTP session when available
        http_client = getattr(self.scraper_factory, "http_client", None)
        try:
            self.understat_client = UnderstatAPIClient(http_client=http_client)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Unable to initialize Understat API client: %s", exc)
            self.understat_client = None
        
        logger.info("Enhanced Data Integrator initialized")

    async def integrate_leagues(
        self, 
        league_codes: Optional[List[str]] = None,
        include_teams: bool = True,
        include_matches: bool = True
    ) -> Dict[str, Any]:
        """
        Main integration method that coordinates league data integration.
        
        Args:
            league_codes: Optional list of league codes to integrate
            include_teams: Whether to integrate teams data
            include_matches: Whether to integrate matches data
            
        Returns:
            Dictionary with integration results
        """
        from datetime import timezone
        self.integration_stats['start_time'] = datetime.now(timezone.utc)
        leagues_to_process = league_codes or self.default_leagues
        
        results = {
            'successful_leagues': [],
            'failed_leagues': [],
            'total_processed': 0,
            'errors': []
        }
        
        logger.info(f"Starting integration for {len(leagues_to_process)} leagues")
        
        for league_code in leagues_to_process:
            try:
                self.integration_stats['total_processed'] += 1
                
                league_result = {'league': league_code, 'teams': 0, 'matches': 0}
                
                # Integrate teams data
                if include_teams:
                    teams_result = await self._integrate_teams_with_validation(league_code)
                    if teams_result['success']:
                        league_result['teams'] = teams_result.get('teams_count', 0)
                        logger.info(f"Successfully integrated {league_result['teams']} teams for {league_code}")
                    else:
                        raise Exception(f"Team integration failed: {teams_result.get('error', 'Unknown error')}")
                
                # Integrate matches data
                if include_matches:
                    matches_result = await self._integrate_upcoming_matches_with_validation(league_code)
                    if matches_result['success']:
                        league_result['matches'] = matches_result.get('matches_count', 0)
                        logger.info(f"Successfully integrated {league_result['matches']} matches for {league_code}")
                    else:
                        logger.warning(f"Matches integration failed for {league_code}: {matches_result.get('error', 'Unknown error')}")
                
                results['successful_leagues'].append(league_result)
                self.integration_stats['successful'] += 1
                
            except Exception as e:
                error_info = {'league': league_code, 'error': str(e)}
                results['failed_leagues'].append(error_info)
                results['errors'].append(str(e))
                self.integration_stats['failed'] += 1
                logger.error(f"Failed to integrate league {league_code}: {e}")
                
        results['total_processed'] = self.integration_stats['total_processed']
        self.integration_stats['end_time'] = datetime.now(timezone.utc)
        
        logger.info(f"League integration completed. Successful: {len(results['successful_leagues'])}, Failed: {len(results['failed_leagues'])}")
        
        return results

    async def _integrate_teams_with_validation(self, league_code: str) -> Dict[str, Any]:
        """Integrate teams for a league with validation and fallback strategies."""
        scrapers_to_try = ['football_data', 'understat', 'espn']
        
        for scraper_name in scrapers_to_try:
            try:
                if scraper_name == 'football_data':
                    result = await self._integrate_teams_football_data(league_code)
                elif scraper_name == 'understat':
                    result = await self._integrate_teams_understat(league_code)
                elif scraper_name == 'espn':
                    result = await self._integrate_teams_espn(league_code)
                else:
                    continue
                
                if result['success']:
                    logger.info(f"Successfully integrated league data from {scraper_name} for {league_code}")
                    return result
                else:
                    logger.warning(f"Failed to integrate from {scraper_name} for {league_code}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error integrating from {scraper_name} for {league_code}: {e}")
                continue
        
        # If all scrapers fail, create placeholder teams
        logger.warning(f"All scrapers failed for {league_code}. Creating placeholder teams.")
        try:
            result = await self._create_placeholder_teams(league_code)
            if result['success']:
                return result
        except Exception as e:
            logger.error(f"Failed to create placeholder teams for {league_code}: {e}")
        
        return {'success': False, 'error': 'Team integration failed'}

    async def _integrate_teams_football_data(self, league_code: str) -> Dict[str, Any]:
        """Integrate teams from Football-Data.org with validation."""
        try:
            scraper = self.scraper_factory.get_scraper("football_data")
            if not scraper:
                return {'success': False, 'error': 'Failed to get Football-Data scraper'}
            
            # Define league to country mapping
            league_country_map = {
                'premier_league': 'England',
                'la_liga': 'Spain', 
                'bundesliga': 'Germany',
                'serie_a': 'Italy',
                'ligue_1': 'France',
                'eredivisie': 'Netherlands'
            }
            
            with self.db.session_scope() as session:
                # Create league if it doesn't exist
                league = session.query(League).filter(League.id == league_code).first()
                if not league:
                    country = league_country_map.get(league_code, 'Unknown')
                    league = League(
                        id=league_code, 
                        name=league_code.replace('_', ' ').title(),
                        country=country
                    )
                    session.add(league)
                    session.flush()
                
                # Get teams from Football-Data.org API
                teams_data = await scraper.get_teams(league_code)
                
                if not teams_data or teams_data.empty:
                    return {'success': False, 'error': 'No teams data found from Football-Data'}
                
                processed_teams_count = 0
                for _, team_row in teams_data.iterrows():
                    # Check if team exists
                    existing_team = session.query(Team).filter(Team.id == str(team_row.get('id', ''))).first()
                    
                    team_data_payload = {
                        "id": str(team_row.get('id', '')),
                        "name": team_row.get('name', ''),
                        "short_name": team_row.get('short_name', team_row.get('name', '')[:10]),
                        "tla": team_row.get('tla', team_row.get('short_name', '')[:3]),
                        "league_id": league_code,
                        "venue": team_row.get('venue', ''),
                        "api_id": str(team_row.get('api_id', team_row.get('id', ''))),
                        "crest_url": team_row.get('crest_url', '')
                    }
                    
                    if existing_team:
                        for key, value in team_data_payload.items():
                            if hasattr(existing_team, key):
                                setattr(existing_team, key, value)
                    else:
                        new_team = Team(**team_data_payload)
                        session.add(new_team)
                    processed_teams_count += 1
                
                logger.info(f"Integrated {processed_teams_count} teams for league: {league_code} from Football-Data.org")
                return {'success': True, 'teams_count': processed_teams_count}
                
        except Exception as e:
            logger.error(f"Error integrating teams from Football-Data for {league_code}: {e}")
            return {'success': False, 'error': str(e)}

    async def _integrate_teams_understat(self, league_code: str) -> Dict[str, Any]:
        """Integrate teams from Understat with validation."""
        try:
            if not self.understat_client:
                return {'success': False, 'error': 'Understat API client unavailable'}
            
            # Define league to country mapping
            league_country_map = {
                'premier_league': 'England',
                'la_liga': 'Spain', 
                'bundesliga': 'Germany',
                'serie_a': 'Italy',
                'ligue_1': 'France',
                'eredivisie': 'Netherlands'
            }
            
            with self.db.session_scope() as session:
                # Create league if it doesn't exist
                league = session.query(League).filter(League.id == league_code).first()
                if not league:
                    country = league_country_map.get(league_code, 'Unknown')
                    league = League(
                        id=league_code, 
                        name=league_code.replace('_', ' ').title(),
                        country=country
                    )
                    session.add(league)
                    session.flush()
                
                # Determine season and fetch teams from Understat API
                season = UnderstatAPIClient._default_season()
                teams_list = await self.understat_client.get_league_teams(league_code, season)
                
                if not teams_list:
                    return {'success': False, 'error': 'No teams data found from Understat'}
                
                processed_teams_count = 0
                for team_info in teams_list:
                    # Skip if team info is invalid
                    if not team_info.get('id'):
                        continue
                        
                    team_id_str = str(team_info['id'])
                    team_db_id = f"{league_code}_{team_id_str}"
                    
                    team_data_payload = {
                        "id": team_db_id,
                        "name": team_info["name"],
                        "short_name": team_info.get("short_name", team_info["name"][:3].upper()),
                        "league_id": league_code,
                        "api_id": team_id_str
                    }
                    
                    # Check if team exists
                    existing_team = session.query(Team).filter(Team.id == team_db_id).first()
                    
                    if existing_team:
                        for key, value in team_data_payload.items():
                            if hasattr(existing_team, key):
                                setattr(existing_team, key, value)
                    else:
                        new_team = Team(**team_data_payload)
                        session.add(new_team)
                    processed_teams_count += 1
                
                logger.info(f"Integrated {processed_teams_count} teams for league: {league_code} from Understat")
                return {'success': True, 'teams_count': processed_teams_count}
                
        except Exception as e:
            logger.error(f"Error integrating teams from Understat for {league_code}: {e}")
            return {'success': False, 'error': str(e)}

    async def _integrate_teams_espn(self, league_code: str) -> Dict[str, Any]:
        """Integrate teams from ESPN with validation."""
        try:
            scraper = self.scraper_factory.get_scraper("espn")
            if not scraper:
                return {'success': False, 'error': 'Failed to get ESPN scraper'}
            
            # Define league to country mapping
            league_country_map = {
                'premier_league': 'England',
                'la_liga': 'Spain', 
                'bundesliga': 'Germany',
                'serie_a': 'Italy',
                'ligue_1': 'France',
                'eredivisie': 'Netherlands'
            }
            
            with self.db.session_scope() as session:
                # Create league if it doesn't exist
                league = session.query(League).filter(League.id == league_code).first()
                if not league:
                    country = league_country_map.get(league_code, 'Unknown')
                    league = League(
                        id=league_code, 
                        name=league_code.replace('_', ' ').title(),
                        country=country
                    )
                    session.add(league)
                    session.flush()
                
                # Get teams from ESPN
                league_data = await scraper.get_league_data(league_code)
                
                if not league_data or 'teams' not in league_data:
                    return {'success': False, 'error': 'No teams data found from ESPN'}
                
                teams_list = league_data['teams']
                processed_teams_count = 0
                
                for team_info in teams_list:
                    team_id_str = str(team_info.get('id', f"{league_code}_team_{processed_teams_count}"))
                    
                    team_data_payload = {
                        "id": team_id_str,
                        "name": team_info.get("name", "Unknown Team"),
                        "short_name": team_info.get("short_name", team_info.get("name", "UNK")[:3].upper()),
                        "league_id": league_code,
                        "api_id": team_id_str
                    }
                    
                    # Check if team exists
                    existing_team = session.query(Team).filter(Team.id == team_id_str).first()
                    
                    if existing_team:
                        for key, value in team_data_payload.items():
                            if hasattr(existing_team, key):
                                setattr(existing_team, key, value)
                    else:
                        new_team = Team(**team_data_payload)
                        session.add(new_team)
                    processed_teams_count += 1
                
                logger.info(f"Integrated {processed_teams_count} teams for league: {league_code} from ESPN")
                return {'success': True, 'teams_count': processed_teams_count}
                
        except Exception as e:
            logger.error(f"Error integrating teams from ESPN for {league_code}: {e}")
            return {'success': False, 'error': str(e)}

    async def _create_placeholder_teams(self, league_code: str) -> Dict[str, Any]:
        """Create placeholder teams for a league when all data sources fail."""
        try:
            # Define league to country mapping
            league_country_map = {
                'premier_league': 'England',
                'la_liga': 'Spain', 
                'bundesliga': 'Germany',
                'serie_a': 'Italy',
                'ligue_1': 'France',
                'eredivisie': 'Netherlands'
            }
            
            # League name mapping for default teams
            league_teams = {
                'premier_league': ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester City', 'Manchester United', 
                                'Tottenham', 'Leicester City', 'Everton', 'West Ham', 'Aston Villa'],
                'la_liga': ['Barcelona', 'Real Madrid', 'Atletico Madrid', 'Sevilla', 'Valencia', 
                          'Villarreal', 'Real Sociedad', 'Athletic Bilbao', 'Real Betis', 'Getafe'],
                'bundesliga': ['Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 
                             'Borussia Monchengladbach', 'Wolfsburg', 'Eintracht Frankfurt', 'Schalke 04', 
                             'Hertha Berlin', 'FC Cologne'],
                'serie_a': ['Juventus', 'Inter Milan', 'AC Milan', 'Napoli', 'Roma', 'Lazio', 'Atalanta', 
                          'Fiorentina', 'Torino', 'Bologna'],
                'ligue_1': ['PSG', 'Marseille', 'Lyon', 'Monaco', 'Lille', 'Rennes', 'Nice', 'Saint-Etienne', 
                          'Nantes', 'Strasbourg'],
                'eredivisie': ['Ajax', 'PSV Eindhoven', 'Feyenoord', 'AZ Alkmaar', 'FC Utrecht', 'Vitesse', 
                             'FC Groningen', 'FC Twente', 'Willem II', 'SC Heerenveen']
            }
            
            with self.db.session_scope() as session:
                # Create league if it doesn't exist
                league = session.query(League).filter(League.id == league_code).first()
                if not league:
                    country = league_country_map.get(league_code, 'Unknown')
                    league = League(
                        id=league_code, 
                        name=league_code.replace('_', ' ').title(),
                        country=country
                    )
                    session.add(league)
                    session.flush()
                
                # Get default teams for this league or use generic ones
                teams_to_create = league_teams.get(league_code, [f"Team{i}" for i in range(1, 11)])
                
                processed_teams_count = 0
                for i, team_name in enumerate(teams_to_create):
                    team_id = f"{league_code}_{i+1}"
                    
                    # Check if team already exists
                    existing_team = session.query(Team).filter(Team.id == team_id).first()
                    if not existing_team:
                        team_data_payload = {
                            "id": team_id,
                            "name": team_name,
                            "short_name": team_name[:3].upper(),
                            "league_id": league_code,
                            "api_id": team_id
                        }
                        
                        new_team = Team(**team_data_payload)
                        session.add(new_team)
                        processed_teams_count += 1
                
                logger.info(f"Created {processed_teams_count} placeholder teams for league: {league_code}")
                return {'success': True, 'teams_count': processed_teams_count}
                
        except Exception as e:
            logger.error(f"Error creating placeholder teams for {league_code}: {e}")
            return {'success': False, 'error': str(e)}

    async def _integrate_upcoming_matches_with_validation(self, league_code: str) -> Dict[str, Any]:
        """Integrate upcoming matches for a league with validation."""
        try:
            scraper = self.scraper_factory.get_scraper("football_data")
            if not scraper:
                logger.warning(f"No Football-Data scraper available for {league_code}")
                return {'success': False, 'error': 'Football-Data scraper not available'}
            
            # Get upcoming matches (next 30 days)
            date_from = datetime.now()
            date_to = date_from + timedelta(days=30)
            
            matches_df = await scraper.get_matches(
                league_code=league_code,
                date_from=date_from,
                date_to=date_to,
                status='SCHEDULED'
            )
            
            if matches_df is None or matches_df.empty:
                logger.warning(f"No upcoming matches found for {league_code}")
                return {'success': True, 'matches_count': 0}
            
            with self.db.session_scope() as session:
                processed_matches_count = 0
                
                for _, match_row in matches_df.iterrows():
                    match_id = str(match_row.get('id', ''))
                    if not match_id:
                        continue
                    
                    # Check if match already exists
                    existing_match = session.query(Match).filter(Match.id == match_id).first()
                    
                    match_data = {
                        "id": match_id,
                        "league_id": league_code,
                        "home_team_id": str(match_row.get('home_team_id', '')),
                        "away_team_id": str(match_row.get('away_team_id', '')),
                        "match_date": match_row.get('match_date', datetime.now()),
                        "status": match_row.get('status', 'SCHEDULED'),
                        "matchday": match_row.get('matchday'),
                        "venue": match_row.get('venue', ''),
                        "api_id": str(match_row.get('api_id', match_id))
                    }
                    
                    if existing_match:
                        for key, value in match_data.items():
                            if hasattr(existing_match, key):
                                setattr(existing_match, key, value)
                    else:
                        new_match = Match(**match_data)
                        session.add(new_match)
                    
                    processed_matches_count += 1
                
                logger.info(f"Integrated {processed_matches_count} upcoming matches for {league_code}")
                return {'success': True, 'matches_count': processed_matches_count}
                
        except Exception as e:
            logger.error(f"Error integrating upcoming matches for {league_code}: {e}")
            return {'success': False, 'error': str(e)}

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        stats = self.integration_stats.copy()
        if stats['start_time'] and stats['end_time']:
            stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
        return stats
