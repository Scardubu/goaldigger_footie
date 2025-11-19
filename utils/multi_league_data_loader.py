#!/usr/bin/env python3
"""
Multi-League Data Loader for GoalDiggers Platform

Enhanced data loading system for cross-league analysis:
- Seamless data fetching across top-6 leagues
- Consistent feature engineering and normalization
- League-specific data processing and validation
- Integration with unified configuration system
- Performance optimization for multi-league scenarios

Supported Leagues:
- Premier League (England)
- La Liga (Spain)
- Bundesliga (Germany)
- Serie A (Italy)
- Ligue 1 (France)
- Eredivisie (Netherlands)
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

class MultiLeagueDataLoader:
    """Enhanced data loader for multi-league cross-analysis."""
    
    def __init__(self, config_manager=None):
        """Initialize multi-league data loader."""
        self.config_manager = config_manager
        self.league_configs = self._initialize_league_configs()
        self.data_cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        self.last_update = {}
        
        # Performance tracking
        self.load_times = {}
        self.success_rates = {}
        
        logger.info("🌍 Multi-League Data Loader initialized")
    
    def _initialize_league_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize configuration for each supported league."""
        return {
            'Premier League': {
                'country': 'England',
                'code': 'PL',
                'api_id': 39,
                'teams_count': 20,
                'season_format': '2024-25',
                'timezone': 'Europe/London',
                'typical_kickoff_times': ['15:00', '17:30', '20:00'],
                'data_sources': ['football-data', 'api-football', 'understat'],
                'quality_weight': 1.0
            },
            'La Liga': {
                'country': 'Spain',
                'code': 'PD',
                'api_id': 140,
                'teams_count': 20,
                'season_format': '2024-25',
                'timezone': 'Europe/Madrid',
                'typical_kickoff_times': ['14:00', '16:15', '18:30', '21:00'],
                'data_sources': ['football-data', 'api-football', 'understat'],
                'quality_weight': 0.95
            },
            'Bundesliga': {
                'country': 'Germany',
                'code': 'BL1',
                'api_id': 78,
                'teams_count': 18,
                'season_format': '2024-25',
                'timezone': 'Europe/Berlin',
                'typical_kickoff_times': ['15:30', '18:30'],
                'data_sources': ['football-data', 'api-football', 'understat'],
                'quality_weight': 0.90
            },
            'Serie A': {
                'country': 'Italy',
                'code': 'SA',
                'api_id': 135,
                'teams_count': 20,
                'season_format': '2024-25',
                'timezone': 'Europe/Rome',
                'typical_kickoff_times': ['15:00', '18:00', '20:45'],
                'data_sources': ['football-data', 'api-football', 'understat'],
                'quality_weight': 0.85
            },
            'Ligue 1': {
                'country': 'France',
                'code': 'FL1',
                'api_id': 61,
                'teams_count': 20,
                'season_format': '2024-25',
                'timezone': 'Europe/Paris',
                'typical_kickoff_times': ['15:00', '17:00', '21:00'],
                'data_sources': ['football-data', 'api-football'],
                'quality_weight': 0.80
            },
            'Eredivisie': {
                'country': 'Netherlands',
                'code': 'DED',
                'api_id': 88,
                'teams_count': 18,
                'season_format': '2024-25',
                'timezone': 'Europe/Amsterdam',
                'typical_kickoff_times': ['14:30', '16:45', '20:00'],
                'data_sources': ['football-data', 'api-football'],
                'quality_weight': 0.75
            }
        }
    
    async def load_multi_league_data(self, leagues: List[str] = None, data_types: List[str] = None) -> Dict[str, Any]:
        """
        Load data across multiple leagues with parallel processing.
        
        Args:
            leagues: List of league names to load (default: all supported)
            data_types: Types of data to load ['teams', 'matches', 'standings', 'stats']
            
        Returns:
            Dictionary containing multi-league data
        """
        start_time = time.time()
        
        if leagues is None:
            leagues = list(self.league_configs.keys())
        
        if data_types is None:
            data_types = ['teams', 'matches', 'standings']
        
        logger.info(f"🔄 Loading data for {len(leagues)} leagues: {leagues}")
        
        try:
            # Create tasks for parallel loading
            tasks = []
            for league in leagues:
                if league in self.league_configs:
                    task = asyncio.create_task(
                        self._load_league_data(league, data_types)
                    )
                    tasks.append((league, task))
                else:
                    logger.warning(f"⚠️ Unsupported league: {league}")
            
            # Wait for all tasks to complete
            results = {}
            for league, task in tasks:
                try:
                    league_data = await task
                    results[league] = league_data
                    logger.info(f"✅ {league} data loaded successfully")
                except Exception as e:
                    logger.error(f"❌ Failed to load {league} data: {e}")
                    results[league] = self._get_fallback_league_data(league)
            
            # Normalize data across leagues
            normalized_results = self._normalize_multi_league_data(results)
            
            total_time = time.time() - start_time
            logger.info(f"🌍 Multi-league data loading completed in {total_time:.2f}s")
            
            return {
                'leagues': normalized_results,
                'metadata': {
                    'load_time': total_time,
                    'leagues_loaded': len(results),
                    'data_types': data_types,
                    'timestamp': datetime.now().isoformat(),
                    'cache_status': self._get_cache_status()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-league data loading failed: {e}")
            return self._get_fallback_multi_league_data(leagues)
    
    async def _load_league_data(self, league: str, data_types: List[str]) -> Dict[str, Any]:
        """Load data for a specific league."""
        league_config = self.league_configs[league]
        
        # Check cache first
        cache_key = f"{league}_{'-'.join(data_types)}"
        if self._is_cache_valid(cache_key):
            logger.debug(f"📋 Using cached data for {league}")
            return self.data_cache[cache_key]
        
        league_data = {
            'config': league_config,
            'data': {}
        }
        
        # Load each data type
        for data_type in data_types:
            try:
                if data_type == 'teams':
                    league_data['data']['teams'] = await self._load_teams_data(league)
                elif data_type == 'matches':
                    league_data['data']['matches'] = await self._load_matches_data(league)
                elif data_type == 'standings':
                    league_data['data']['standings'] = await self._load_standings_data(league)
                elif data_type == 'stats':
                    league_data['data']['stats'] = await self._load_stats_data(league)
                else:
                    logger.warning(f"⚠️ Unknown data type: {data_type}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {data_type} for {league}: {e}")
                league_data['data'][data_type] = self._get_fallback_data(league, data_type)
        
        # Cache the result
        self.data_cache[cache_key] = league_data
        self.last_update[cache_key] = datetime.now()
        
        return league_data
    
    async def _load_teams_data(self, league: str) -> List[Dict[str, Any]]:
        """Load teams data for a specific league."""
        league_config = self.league_configs[league]
        
        # Mock teams data with realistic structure
        teams_data = []
        team_names = self._get_league_teams(league)
        
        for i, team_name in enumerate(team_names):
            team_data = {
                'id': f"{league_config['code']}_{i+1}",
                'name': team_name,
                'league': league,
                'country': league_config['country'],
                'founded': 1900 + (i * 5),  # Mock founding years
                'venue': f"{team_name} Stadium",
                'capacity': 30000 + (i * 2000),  # Mock capacities
                'current_form': self._generate_mock_form(),
                'league_position': i + 1,
                'points': max(0, 50 - (i * 2)),  # Mock points
                'goals_for': max(10, 40 - (i * 1)),
                'goals_against': min(50, 20 + (i * 1)),
                'last_updated': datetime.now().isoformat()
            }
            teams_data.append(team_data)
        
        return teams_data
    
    async def _load_matches_data(self, league: str) -> List[Dict[str, Any]]:
        """Load matches data for a specific league."""
        # Mock recent matches data
        matches_data = []
        teams = self._get_league_teams(league)
        
        for i in range(10):  # Mock 10 recent matches
            home_team = teams[i % len(teams)]
            away_team = teams[(i + 1) % len(teams)]
            
            match_data = {
                'id': f"{league}_{i+1}",
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'date': (datetime.now() - timedelta(days=i)).isoformat(),
                'status': 'completed',
                'home_score': np.random.randint(0, 4),
                'away_score': np.random.randint(0, 4),
                'venue': f"{home_team} Stadium",
                'attendance': np.random.randint(20000, 60000),
                'last_updated': datetime.now().isoformat()
            }
            matches_data.append(match_data)
        
        return matches_data
    
    async def _load_standings_data(self, league: str) -> Dict[str, Any]:
        """Load standings data for a specific league."""
        teams = self._get_league_teams(league)
        league_config = self.league_configs[league]
        
        standings = []
        for i, team in enumerate(teams):
            standing = {
                'position': i + 1,
                'team': team,
                'played': 20,
                'won': max(0, 15 - i),
                'drawn': 3,
                'lost': min(15, 2 + i),
                'goals_for': max(20, 50 - (i * 2)),
                'goals_against': min(60, 25 + i),
                'goal_difference': max(-20, 25 - (i * 3)),
                'points': max(0, 48 - (i * 2))
            }
            standings.append(standing)
        
        return {
            'league': league,
            'season': league_config['season_format'],
            'standings': standings,
            'last_updated': datetime.now().isoformat()
        }
    
    async def _load_stats_data(self, league: str) -> Dict[str, Any]:
        """Load statistical data for a specific league."""
        league_config = self.league_configs[league]
        
        return {
            'league': league,
            'season': league_config['season_format'],
            'avg_goals_per_game': 2.5 + np.random.uniform(-0.3, 0.3),
            'avg_cards_per_game': 4.2 + np.random.uniform(-0.5, 0.5),
            'home_win_percentage': 0.45 + np.random.uniform(-0.05, 0.05),
            'draw_percentage': 0.25 + np.random.uniform(-0.03, 0.03),
            'away_win_percentage': 0.30 + np.random.uniform(-0.05, 0.05),
            'clean_sheet_percentage': 0.35 + np.random.uniform(-0.05, 0.05),
            'both_teams_score_percentage': 0.55 + np.random.uniform(-0.05, 0.05),
            'last_updated': datetime.now().isoformat()
        }
    
    def _normalize_multi_league_data(self, league_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data across multiple leagues for consistent analysis."""
        normalized_data = {}
        
        for league, data in league_data.items():
            try:
                league_config = self.league_configs[league]
                quality_weight = league_config['quality_weight']
                
                normalized_league_data = {
                    'config': data['config'],
                    'normalized_data': {},
                    'quality_weight': quality_weight
                }
                
                # Normalize teams data
                if 'teams' in data['data']:
                    normalized_league_data['normalized_data']['teams'] = self._normalize_teams_data(
                        data['data']['teams'], quality_weight
                    )
                
                # Normalize matches data
                if 'matches' in data['data']:
                    normalized_league_data['normalized_data']['matches'] = self._normalize_matches_data(
                        data['data']['matches'], quality_weight
                    )
                
                # Normalize standings data
                if 'standings' in data['data']:
                    normalized_league_data['normalized_data']['standings'] = self._normalize_standings_data(
                        data['data']['standings'], quality_weight
                    )
                
                normalized_data[league] = normalized_league_data
                
            except Exception as e:
                logger.error(f"❌ Failed to normalize data for {league}: {e}")
                normalized_data[league] = data  # Use original data as fallback
        
        return normalized_data
    
    def _normalize_teams_data(self, teams_data: List[Dict[str, Any]], quality_weight: float) -> List[Dict[str, Any]]:
        """Normalize teams data with quality weighting."""
        normalized_teams = []
        
        for team in teams_data:
            normalized_team = team.copy()
            
            # Apply quality weighting to performance metrics
            if 'points' in team:
                normalized_team['normalized_points'] = team['points'] * quality_weight
            if 'goals_for' in team:
                normalized_team['normalized_goals_for'] = team['goals_for'] * quality_weight
            if 'goals_against' in team:
                normalized_team['normalized_goals_against'] = team['goals_against'] / quality_weight
            
            normalized_teams.append(normalized_team)
        
        return normalized_teams
    
    def _normalize_matches_data(self, matches_data: List[Dict[str, Any]], quality_weight: float) -> List[Dict[str, Any]]:
        """Normalize matches data with quality weighting."""
        normalized_matches = []
        
        for match in matches_data:
            normalized_match = match.copy()
            
            # Apply quality weighting to match outcomes
            if 'home_score' in match and 'away_score' in match:
                normalized_match['normalized_home_score'] = match['home_score'] * quality_weight
                normalized_match['normalized_away_score'] = match['away_score'] * quality_weight
                normalized_match['normalized_total_goals'] = (match['home_score'] + match['away_score']) * quality_weight
            
            normalized_matches.append(normalized_match)
        
        return normalized_matches
    
    def _normalize_standings_data(self, standings_data: Dict[str, Any], quality_weight: float) -> Dict[str, Any]:
        """Normalize standings data with quality weighting."""
        normalized_standings = standings_data.copy()
        
        if 'standings' in standings_data:
            normalized_table = []
            for team_standing in standings_data['standings']:
                normalized_standing = team_standing.copy()
                
                # Apply quality weighting
                if 'points' in team_standing:
                    normalized_standing['normalized_points'] = team_standing['points'] * quality_weight
                if 'goals_for' in team_standing:
                    normalized_standing['normalized_goals_for'] = team_standing['goals_for'] * quality_weight
                if 'goals_against' in team_standing:
                    normalized_standing['normalized_goals_against'] = team_standing['goals_against'] / quality_weight
                
                normalized_table.append(normalized_standing)
            
            normalized_standings['normalized_standings'] = normalized_table
        
        return normalized_standings
    
    def _get_league_teams(self, league: str) -> List[str]:
        """Get list of teams for a specific league."""
        teams_by_league = {
            'Premier League': [
                'Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 'Manchester United',
                'Tottenham', 'Newcastle', 'Brighton', 'West Ham', 'Aston Villa',
                'Crystal Palace', 'Fulham', 'Brentford', 'Wolves', 'Everton',
                'Nottingham Forest', 'Bournemouth', 'Sheffield United', 'Burnley', 'Luton Town'
            ],
            'La Liga': [
                'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Real Sociedad', 'Villarreal',
                'Real Betis', 'Athletic Bilbao', 'Valencia', 'Getafe', 'Sevilla',
                'Osasuna', 'Las Palmas', 'Girona', 'Alaves', 'Mallorca',
                'Rayo Vallecano', 'Celta Vigo', 'Cadiz', 'Granada', 'Almeria'
            ],
            'Bundesliga': [
                'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Union Berlin', 'SC Freiburg',
                'Bayer Leverkusen', 'Eintracht Frankfurt', 'Wolfsburg', 'Mainz', 'Borussia Monchengladbach',
                'FC Koln', 'Werder Bremen', 'Augsburg', 'VfB Stuttgart', 'Hoffenheim',
                'VfL Bochum', 'Heidenheim', 'Darmstadt'
            ],
            'Serie A': [
                'Inter Milan', 'AC Milan', 'Juventus', 'Atalanta', 'Roma',
                'Lazio', 'Napoli', 'Fiorentina', 'Bologna', 'Torino',
                'Monza', 'Genoa', 'Lecce', 'Udinese', 'Frosinone',
                'Empoli', 'Verona', 'Cagliari', 'Sassuolo', 'Salernitana'
            ],
            'Ligue 1': [
                'PSG', 'Monaco', 'Lille', 'Nice', 'Rennes',
                'Lyon', 'Marseille', 'Lens', 'Strasbourg', 'Nantes',
                'Montpellier', 'Brest', 'Reims', 'Toulouse', 'Le Havre',
                'Metz', 'Lorient', 'Clermont', 'Angers', 'Ajaccio'
            ],
            'Eredivisie': [
                'PSV', 'Ajax', 'Feyenoord', 'AZ Alkmaar', 'Twente',
                'Utrecht', 'Go Ahead Eagles', 'Fortuna Sittard', 'NEC', 'Heerenveen',
                'PEC Zwolle', 'Sparta Rotterdam', 'Almere City', 'Excelsior', 'Waalwijk',
                'Vitesse', 'Volendam', 'Emmen'
            ]
        }
        
        return teams_by_league.get(league, ['Team A', 'Team B', 'Team C'])
    
    def _generate_mock_form(self) -> List[str]:
        """Generate mock recent form."""
        results = ['W', 'D', 'L']
        return [np.random.choice(results) for _ in range(5)]
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.data_cache:
            return False
        
        last_update = self.last_update.get(cache_key)
        if not last_update:
            return False
        
        return (datetime.now() - last_update).total_seconds() < self.cache_ttl
    
    def _get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status."""
        return {
            'cached_items': len(self.data_cache),
            'cache_ttl': self.cache_ttl,
            'last_updates': {k: v.isoformat() for k, v in self.last_update.items()}
        }
    
    def _get_fallback_league_data(self, league: str) -> Dict[str, Any]:
        """Get fallback data for a league."""
        return {
            'config': self.league_configs.get(league, {}),
            'data': {
                'teams': [],
                'matches': [],
                'standings': {'standings': []},
                'stats': {}
            },
            'status': 'fallback'
        }
    
    def _get_fallback_multi_league_data(self, leagues: List[str]) -> Dict[str, Any]:
        """Get fallback data for multiple leagues."""
        return {
            'leagues': {league: self._get_fallback_league_data(league) for league in leagues},
            'metadata': {
                'load_time': 0,
                'leagues_loaded': 0,
                'status': 'fallback',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _get_fallback_data(self, league: str, data_type: str) -> Any:
        """Get fallback data for specific data type."""
        fallbacks = {
            'teams': [],
            'matches': [],
            'standings': {'standings': []},
            'stats': {}
        }
        return fallbacks.get(data_type, {})

# Global instance
_multi_league_loader_instance = None

def get_multi_league_data_loader(config_manager=None) -> MultiLeagueDataLoader:
    """Get global multi-league data loader instance."""
    global _multi_league_loader_instance
    if _multi_league_loader_instance is None:
        _multi_league_loader_instance = MultiLeagueDataLoader(config_manager)
    return _multi_league_loader_instance
