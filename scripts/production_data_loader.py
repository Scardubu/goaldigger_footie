"""
Production Data Loader - Loads essential data for platform operation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from database.db_manager import DatabaseManager
from database.schema import League, Match, Team, TeamStats
from utils.logging_config import get_logger

logger = get_logger(__name__)

class ProductionDataLoader:
    """Loads essential data for production operation when APIs are limited."""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """Initialize the production data loader."""
        self.db = db_manager or DatabaseManager()
        self.leagues_data = self._get_default_leagues()
        self.teams_data = self._get_default_teams()
        
    def _get_default_leagues(self) -> Dict[str, Dict[str, Any]]:
        """Get default league data for production."""
        return {
            'premier_league': {
                'name': 'Premier League',
                'country': 'England',
                'tier': 1,
                'api_id': 'PL',
                'teams_count': 20
            },
            'la_liga': {
                'name': 'La Liga',
                'country': 'Spain', 
                'tier': 1,
                'api_id': 'PD',
                'teams_count': 20
            },
            'bundesliga': {
                'name': 'Bundesliga',
                'country': 'Germany',
                'tier': 1,
                'api_id': 'BL1',
                'teams_count': 18
            },
            'serie_a': {
                'name': 'Serie A',
                'country': 'Italy',
                'tier': 1,
                'api_id': 'SA',
                'teams_count': 20
            },
            'ligue_1': {
                'name': 'Ligue 1',
                'country': 'France',
                'tier': 1,
                'api_id': 'FL1',
                'teams_count': 18
            },
            'eredivisie': {
                'name': 'Eredivisie',
                'country': 'Netherlands',
                'tier': 1,
                'api_id': 'DED',
                'teams_count': 18
            }
        }
    
    def _get_default_teams(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get default team data for each league."""
        return {
            'premier_league': [
                {'name': 'Manchester City', 'short_name': 'MCI', 'tla': 'MCI'},
                {'name': 'Arsenal', 'short_name': 'ARS', 'tla': 'ARS'},
                {'name': 'Liverpool', 'short_name': 'LIV', 'tla': 'LIV'},
                {'name': 'Aston Villa', 'short_name': 'AVL', 'tla': 'AVL'},
                {'name': 'Tottenham Hotspur', 'short_name': 'TOT', 'tla': 'TOT'},
                {'name': 'Chelsea', 'short_name': 'CHE', 'tla': 'CHE'},
                {'name': 'Newcastle United', 'short_name': 'NEW', 'tla': 'NEW'},
                {'name': 'Manchester United', 'short_name': 'MUN', 'tla': 'MUN'},
                {'name': 'West Ham United', 'short_name': 'WHU', 'tla': 'WHU'},
                {'name': 'Crystal Palace', 'short_name': 'CRY', 'tla': 'CRY'},
                {'name': 'Brighton & Hove Albion', 'short_name': 'BHA', 'tla': 'BHA'},
                {'name': 'Bournemouth', 'short_name': 'BOU', 'tla': 'BOU'},
                {'name': 'Fulham', 'short_name': 'FUL', 'tla': 'FUL'},
                {'name': 'Wolverhampton Wanderers', 'short_name': 'WOL', 'tla': 'WOL'},
                {'name': 'Everton', 'short_name': 'EVE', 'tla': 'EVE'},
                {'name': 'Brentford', 'short_name': 'BRE', 'tla': 'BRE'},
                {'name': 'Nottingham Forest', 'short_name': 'NFO', 'tla': 'NFO'},
                {'name': 'Luton Town', 'short_name': 'LUT', 'tla': 'LUT'},
                {'name': 'Burnley', 'short_name': 'BUR', 'tla': 'BUR'},
                {'name': 'Sheffield United', 'short_name': 'SHU', 'tla': 'SHU'}
            ],
            'la_liga': [
                {'name': 'Real Madrid', 'short_name': 'RMA', 'tla': 'RMA'},
                {'name': 'Barcelona', 'short_name': 'BAR', 'tla': 'BAR'},
                {'name': 'Atletico Madrid', 'short_name': 'ATM', 'tla': 'ATM'},
                {'name': 'Athletic Bilbao', 'short_name': 'ATH', 'tla': 'ATH'},
                {'name': 'Real Sociedad', 'short_name': 'RSO', 'tla': 'RSO'},
                {'name': 'Real Betis', 'short_name': 'BET', 'tla': 'BET'},
                {'name': 'Valencia', 'short_name': 'VAL', 'tla': 'VAL'},
                {'name': 'Villarreal', 'short_name': 'VIL', 'tla': 'VIL'},
                {'name': 'Osasuna', 'short_name': 'OSA', 'tla': 'OSA'},
                {'name': 'Getafe', 'short_name': 'GET', 'tla': 'GET'},
                {'name': 'Girona', 'short_name': 'GIR', 'tla': 'GIR'},
                {'name': 'Sevilla', 'short_name': 'SEV', 'tla': 'SEV'},
                {'name': 'Las Palmas', 'short_name': 'LPA', 'tla': 'LPA'},
                {'name': 'Celta Vigo', 'short_name': 'CEL', 'tla': 'CEL'},
                {'name': 'Rayo Vallecano', 'short_name': 'RAY', 'tla': 'RAY'},
                {'name': 'Mallorca', 'short_name': 'MAL', 'tla': 'MAL'},
                {'name': 'Cadiz', 'short_name': 'CAD', 'tla': 'CAD'},
                {'name': 'Granada', 'short_name': 'GRA', 'tla': 'GRA'},
                {'name': 'Almeria', 'short_name': 'ALM', 'tla': 'ALM'},
                {'name': 'Alaves', 'short_name': 'ALA', 'tla': 'ALA'}
            ],
            # Add other leagues...
            'bundesliga': [
                {'name': 'Bayern Munich', 'short_name': 'BAY', 'tla': 'BAY'},
                {'name': 'RB Leipzig', 'short_name': 'RBL', 'tla': 'RBL'},
                {'name': 'Borussia Dortmund', 'short_name': 'BVB', 'tla': 'BVB'},
                {'name': 'Union Berlin', 'short_name': 'UNI', 'tla': 'UNI'},
                {'name': 'SC Freiburg', 'short_name': 'FRE', 'tla': 'FRE'},
                {'name': 'Bayer Leverkusen', 'short_name': 'B04', 'tla': 'B04'},
                {'name': 'Eintracht Frankfurt', 'short_name': 'SGE', 'tla': 'SGE'},
                {'name': 'VfL Wolfsburg', 'short_name': 'WOB', 'tla': 'WOB'},
                {'name': 'Mainz 05', 'short_name': 'M05', 'tla': 'M05'},
                {'name': 'Borussia Monchengladbach', 'short_name': 'BMG', 'tla': 'BMG'},
                {'name': 'FC Koln', 'short_name': 'KOE', 'tla': 'KOE'},
                {'name': 'Hoffenheim', 'short_name': 'HOF', 'tla': 'HOF'},
                {'name': 'VfB Stuttgart', 'short_name': 'VFB', 'tla': 'VFB'},
                {'name': 'FC Augsburg', 'short_name': 'AUG', 'tla': 'AUG'},
                {'name': 'Werder Bremen', 'short_name': 'BRE', 'tla': 'BRE'},
                {'name': 'VfL Bochum', 'short_name': 'BOC', 'tla': 'BOC'},
                {'name': 'FC Heidenheim', 'short_name': 'HEI', 'tla': 'HEI'},
                {'name': 'SV Darmstadt 98', 'short_name': 'DAR', 'tla': 'DAR'}
            ],
            'serie_a': [
                {'name': 'Inter Milan', 'short_name': 'INT', 'tla': 'INT'},
                {'name': 'AC Milan', 'short_name': 'MIL', 'tla': 'MIL'},
                {'name': 'Juventus', 'short_name': 'JUV', 'tla': 'JUV'},
                {'name': 'Atalanta', 'short_name': 'ATA', 'tla': 'ATA'},
                {'name': 'Bologna', 'short_name': 'BOL', 'tla': 'BOL'},
                {'name': 'Roma', 'short_name': 'ROM', 'tla': 'ROM'},
                {'name': 'Lazio', 'short_name': 'LAZ', 'tla': 'LAZ'},
                {'name': 'Fiorentina', 'short_name': 'FIO', 'tla': 'FIO'},
                {'name': 'Torino', 'short_name': 'TOR', 'tla': 'TOR'},
                {'name': 'Napoli', 'short_name': 'NAP', 'tla': 'NAP'},
                {'name': 'Genoa', 'short_name': 'GEN', 'tla': 'GEN'},
                {'name': 'Monza', 'short_name': 'MON', 'tla': 'MON'},
                {'name': 'Hellas Verona', 'short_name': 'VER', 'tla': 'VER'},
                {'name': 'Lecce', 'short_name': 'LEC', 'tla': 'LEC'},
                {'name': 'Udinese', 'short_name': 'UDI', 'tla': 'UDI'},
                {'name': 'Cagliari', 'short_name': 'CAG', 'tla': 'CAG'},
                {'name': 'Frosinone', 'short_name': 'FRO', 'tla': 'FRO'},
                {'name': 'Empoli', 'short_name': 'EMP', 'tla': 'EMP'},
                {'name': 'Sassuolo', 'short_name': 'SAS', 'tla': 'SAS'},
                {'name': 'Salernitana', 'short_name': 'SAL', 'tla': 'SAL'}
            ],
            'ligue_1': [
                {'name': 'Paris Saint-Germain', 'short_name': 'PSG', 'tla': 'PSG'},
                {'name': 'AS Monaco', 'short_name': 'MON', 'tla': 'MON'},
                {'name': 'Brest', 'short_name': 'BRE', 'tla': 'BRE'},
                {'name': 'Lille', 'short_name': 'LIL', 'tla': 'LIL'},
                {'name': 'Nice', 'short_name': 'NIC', 'tla': 'NIC'},
                {'name': 'Lyon', 'short_name': 'LYO', 'tla': 'LYO'},
                {'name': 'Lens', 'short_name': 'LEN', 'tla': 'LEN'},
                {'name': 'Marseille', 'short_name': 'MAR', 'tla': 'MAR'},
                {'name': 'Rennes', 'short_name': 'REN', 'tla': 'REN'},
                {'name': 'Reims', 'short_name': 'REI', 'tla': 'REI'},
                {'name': 'Montpellier', 'short_name': 'MON', 'tla': 'MON'},
                {'name': 'Strasbourg', 'short_name': 'STR', 'tla': 'STR'},
                {'name': 'Nantes', 'short_name': 'NAN', 'tla': 'NAN'},
                {'name': 'Le Havre', 'short_name': 'HAV', 'tla': 'HAV'},
                {'name': 'Toulouse', 'short_name': 'TOU', 'tla': 'TOU'},
                {'name': 'Lorient', 'short_name': 'LOR', 'tla': 'LOR'},
                {'name': 'Metz', 'short_name': 'MET', 'tla': 'MET'},
                {'name': 'Clermont Foot', 'short_name': 'CLE', 'tla': 'CLE'}
            ],
            'eredivisie': [
                {'name': 'PSV Eindhoven', 'short_name': 'PSV', 'tla': 'PSV'},
                {'name': 'Feyenoord', 'short_name': 'FEY', 'tla': 'FEY'},
                {'name': 'Ajax', 'short_name': 'AJA', 'tla': 'AJA'},
                {'name': 'AZ Alkmaar', 'short_name': 'AZA', 'tla': 'AZA'},
                {'name': 'FC Twente', 'short_name': 'TWE', 'tla': 'TWE'},
                {'name': 'Sparta Rotterdam', 'short_name': 'SPA', 'tla': 'SPA'},
                {'name': 'Go Ahead Eagles', 'short_name': 'GAE', 'tla': 'GAE'},
                {'name': 'Fortuna Sittard', 'short_name': 'FOR', 'tla': 'FOR'},
                {'name': 'NEC Nijmegen', 'short_name': 'NEC', 'tla': 'NEC'},
                {'name': 'Heerenveen', 'short_name': 'HEE', 'tla': 'HEE'},
                {'name': 'PEC Zwolle', 'short_name': 'PEC', 'tla': 'PEC'},
                {'name': 'Utrecht', 'short_name': 'UTR', 'tla': 'UTR'},
                {'name': 'Excelsior', 'short_name': 'EXC', 'tla': 'EXC'},
                {'name': 'Willem II', 'short_name': 'WIL', 'tla': 'WIL'},
                {'name': 'NAC Breda', 'short_name': 'NAC', 'tla': 'NAC'},
                {'name': 'Almere City', 'short_name': 'ALM', 'tla': 'ALM'},
                {'name': 'Groningen', 'short_name': 'GRO', 'tla': 'GRO'},
                {'name': 'RKC Waalwijk', 'short_name': 'RKC', 'tla': 'RKC'}
            ]
        }
        
    async def load_essential_data(self) -> Dict[str, Any]:
        """Load essential data for production operation."""
        logger.info("Starting production data loading...")
        
        results = {
            'leagues_loaded': 0,
            'teams_loaded': 0,
            'sample_matches_created': 0,
            'errors': []
        }
        
        try:
            # Load leagues
            results['leagues_loaded'] = await self._load_leagues()
            
            # Load teams for each league
            total_teams = 0
            for league_code in self.leagues_data.keys():
                team_count = await self._load_teams(league_code)
                total_teams += team_count
                
            results['teams_loaded'] = total_teams
                
            # Create sample matches for demonstration
            results['sample_matches_created'] = await self._create_sample_matches()
            
            # If no new data was loaded but data exists, report existing counts
            with self.db.session_scope() as session:
                if results['leagues_loaded'] == 0:
                    existing_leagues = session.query(League).count()
                    results['leagues_loaded'] = existing_leagues
                    
                if results['teams_loaded'] == 0:
                    existing_teams = session.query(Team).count()
                    results['teams_loaded'] = existing_teams
                    
                if results['sample_matches_created'] == 0:
                    existing_matches = session.query(Match).filter(Match.match_date > datetime.now()).count()
                    results['sample_matches_created'] = existing_matches
            
            logger.info(f"Production data loading completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in production data loading: {e}")
            results['errors'].append(str(e))
            return results
    
    async def _load_leagues(self) -> int:
        """Load league data into database."""
        loaded_count = 0
        
        with self.db.session_scope() as session:
            for league_id, league_info in self.leagues_data.items():
                try:
                    # Check if league already exists
                    existing_league = session.query(League).filter(League.id == league_id).first()
                    
                    if not existing_league:
                        league = League(
                            id=league_id,
                            name=league_info['name'],
                            country=league_info['country'],
                            tier=league_info['tier'],
                            api_id=league_info['api_id']
                        )
                        session.add(league)
                        loaded_count += 1
                        logger.info(f"Added league: {league_info['name']}")
                    else:
                        logger.info(f"League already exists: {league_info['name']}")
                        
                except Exception as e:
                    logger.error(f"Error loading league {league_id}: {e}")
                    
        return loaded_count
    
    async def _load_teams(self, league_code: str) -> int:
        """Load team data for a specific league."""
        if league_code not in self.teams_data:
            logger.warning(f"No team data available for league: {league_code}")
            return 0
            
        loaded_count = 0
        teams = self.teams_data[league_code]
        
        with self.db.session_scope() as session:
            # First check if teams already exist for this league
            existing_teams_count = session.query(Team).filter(Team.league_id == league_code).count()
            
            if existing_teams_count > 0:
                logger.info(f"Teams already exist for {league_code}: {existing_teams_count} teams")
                return existing_teams_count
            
            for i, team_info in enumerate(teams, 1):
                try:
                    team_id = f"{league_code}_{i:02d}"
                    
                    team = Team(
                        id=team_id,
                        name=team_info['name'],
                        short_name=team_info['short_name'],
                        tla=team_info['tla'],
                        league_id=league_code,
                        api_id=str(i)
                    )
                    session.add(team)
                    loaded_count += 1
                    logger.debug(f"Added team: {team_info['name']} to {league_code}")
                    
                except Exception as e:
                    logger.error(f"Error loading team {team_info.get('name', 'unknown')}: {e}")
                    
        logger.info(f"Loaded {loaded_count} new teams for {league_code}")
        return loaded_count
    
    async def _create_sample_matches(self) -> int:
        """Create sample upcoming matches for demonstration."""
        sample_count = 0
        
        with self.db.session_scope() as session:
            # Check if sample matches already exist
            existing_matches = session.query(Match).filter(Match.match_date > datetime.now()).count()
            if existing_matches > 0:
                logger.info(f"Sample matches already exist: {existing_matches} matches")
                return existing_matches
            
            # Get teams for each league to create sample matches
            for league_code in self.leagues_data.keys():
                teams = session.query(Team).filter(Team.league_id == league_code).limit(10).all()
                
                if len(teams) < 4:
                    continue
                    
                # Create 5 sample matches per league
                for i in range(5):
                    try:
                        home_team = teams[i * 2 % len(teams)]
                        away_team = teams[(i * 2 + 1) % len(teams)]
                        
                        match_date = datetime.now() + timedelta(days=i + 1)
                        match_id = f"{league_code}_sample_{i+1}"
                        
                        match = Match(
                            id=match_id,
                            league_id=league_code,
                            home_team_id=home_team.id,
                            away_team_id=away_team.id,
                            match_date=match_date,
                            status='scheduled',
                            matchday=i + 1,
                            venue=f"{home_team.name} Stadium"
                        )
                        session.add(match)
                        sample_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error creating sample match: {e}")
        
        logger.info(f"Created {sample_count} sample matches")
        return sample_count


async def main():
    """Main function to run production data loader."""
    loader = ProductionDataLoader()
    results = await loader.load_essential_data()
    print("Production data loading results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
