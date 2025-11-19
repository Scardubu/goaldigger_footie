#!/usr/bin/env python3
"""
Enhanced Team Data Manager for GoalDiggers Platform

Comprehensive team name mapping, validation, and metadata integration across all 6 supported leagues:
- Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie
- Standardized team identification and display
- Team metadata integration (badges, colors, stadium info, form indicators)
- Cross-league team recognition and validation
- Performance-optimized team data handling
"""

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from utils.safe_logging import safe_log

logger = logging.getLogger(__name__)

@dataclass
class TeamMetadata:
    """Comprehensive team metadata structure."""
    id: str
    name: str
    short_name: str
    display_name: str
    league: str
    country: str
    founded: Optional[int]
    stadium: Optional[str]
    capacity: Optional[int]
    colors: Dict[str, str]
    badge_url: Optional[str]
    website: Optional[str]
    aliases: List[str]
    current_form: List[str]
    league_position: Optional[int]

class EnhancedTeamDataManager:
    """Enhanced team data management with comprehensive validation and metadata."""
    
    def __init__(self):
        """Initialize enhanced team data manager."""
        self.team_database = self._initialize_team_database()
        self.league_mappings = self._initialize_league_mappings()
        self.name_variations = self._initialize_name_variations()
        self.metadata_cache = {}
        
        safe_log(logger, "info", "[TROPHY] Enhanced Team Data Manager initialized")
    
    def _initialize_team_database(self) -> Dict[str, Dict[str, TeamMetadata]]:
        """Initialize comprehensive team database for all 6 leagues."""
        return {
            'Premier League': self._get_premier_league_teams(),
            'La Liga': self._get_la_liga_teams(),
            'Bundesliga': self._get_bundesliga_teams(),
            'Serie A': self._get_serie_a_teams(),
            'Ligue 1': self._get_ligue_1_teams(),
            'Eredivisie': self._get_eredivisie_teams()
        }
    
    def _get_premier_league_teams(self) -> Dict[str, TeamMetadata]:
        """Get Premier League teams with comprehensive metadata."""
        teams = {}
        
        premier_league_data = [
            {
                'id': 'man_city', 'name': 'Manchester City', 'short_name': 'Man City',
                'founded': 1880, 'stadium': 'Etihad Stadium', 'capacity': 55017,
                'colors': {'primary': '#6CABDD', 'secondary': '#FFFFFF'},
                'badge_url': 'https://resources.premierleague.com/premierleague/badges/t43.png',
                'aliases': ['Manchester City FC', 'City', 'MCFC', 'Man City']
            },
            {
                'id': 'arsenal', 'name': 'Arsenal', 'short_name': 'Arsenal',
                'founded': 1886, 'stadium': 'Emirates Stadium', 'capacity': 60704,
                'colors': {'primary': '#EF0107', 'secondary': '#FFFFFF'},
                'badge_url': 'https://resources.premierleague.com/premierleague/badges/t3.png',
                'aliases': ['Arsenal FC', 'The Gunners', 'AFC']
            },
            {
                'id': 'liverpool', 'name': 'Liverpool', 'short_name': 'Liverpool',
                'founded': 1892, 'stadium': 'Anfield', 'capacity': 53394,
                'colors': {'primary': '#C8102E', 'secondary': '#FFFFFF'},
                'badge_url': 'https://resources.premierleague.com/premierleague/badges/t14.png',
                'aliases': ['Liverpool FC', 'The Reds', 'LFC']
            },
            {
                'id': 'chelsea', 'name': 'Chelsea', 'short_name': 'Chelsea',
                'founded': 1905, 'stadium': 'Stamford Bridge', 'capacity': 40834,
                'colors': {'primary': '#034694', 'secondary': '#FFFFFF'},
                'badge_url': 'https://resources.premierleague.com/premierleague/badges/t8.png',
                'aliases': ['Chelsea FC', 'The Blues', 'CFC']
            },
            {
                'id': 'man_united', 'name': 'Manchester United', 'short_name': 'Man Utd',
                'founded': 1878, 'stadium': 'Old Trafford', 'capacity': 74140,
                'colors': {'primary': '#DA020E', 'secondary': '#FFFFFF'},
                'badge_url': 'https://resources.premierleague.com/premierleague/badges/t1.png',
                'aliases': ['Manchester United FC', 'United', 'MUFC', 'Man Utd', 'Red Devils']
            },
            {
                'id': 'tottenham', 'name': 'Tottenham Hotspur', 'short_name': 'Spurs',
                'founded': 1882, 'stadium': 'Tottenham Hotspur Stadium', 'capacity': 62850,
                'colors': {'primary': '#132257', 'secondary': '#FFFFFF'},
                'badge_url': 'https://resources.premierleague.com/premierleague/badges/t6.png',
                'aliases': ['Tottenham', 'Spurs', 'THFC']
            }
        ]
        
        for team_data in premier_league_data:
            team_id = team_data['id']
            teams[team_id] = TeamMetadata(
                id=team_id,
                name=team_data['name'],
                short_name=team_data['short_name'],
                display_name=team_data['name'],
                league='Premier League',
                country='England',
                founded=team_data.get('founded'),
                stadium=team_data.get('stadium'),
                capacity=team_data.get('capacity'),
                colors=team_data.get('colors', {}),
                badge_url=team_data.get('badge_url'),
                website=f"https://www.{team_data['name'].lower().replace(' ', '')}.com",
                aliases=team_data.get('aliases', []),
                current_form=['W', 'W', 'D', 'W', 'L'],  # Mock form
                league_position=None
            )
        
        return teams
    
    def _get_la_liga_teams(self) -> Dict[str, TeamMetadata]:
        """Get La Liga teams with comprehensive metadata."""
        teams = {}
        
        la_liga_data = [
            {
                'id': 'real_madrid', 'name': 'Real Madrid', 'short_name': 'Real Madrid',
                'founded': 1902, 'stadium': 'Santiago Bernabéu', 'capacity': 81044,
                'colors': {'primary': '#FFFFFF', 'secondary': '#000000'},
                'aliases': ['Real Madrid CF', 'Los Blancos', 'Madrid']
            },
            {
                'id': 'barcelona', 'name': 'Barcelona', 'short_name': 'Barcelona',
                'founded': 1899, 'stadium': 'Camp Nou', 'capacity': 99354,
                'colors': {'primary': '#A50044', 'secondary': '#004D98'},
                'aliases': ['FC Barcelona', 'Barça', 'Barca', 'FCB']
            },
            {
                'id': 'atletico_madrid', 'name': 'Atlético Madrid', 'short_name': 'Atlético',
                'founded': 1903, 'stadium': 'Wanda Metropolitano', 'capacity': 68456,
                'colors': {'primary': '#CE3524', 'secondary': '#FFFFFF'},
                'aliases': ['Atletico Madrid', 'Atleti', 'ATM']
            },
            {
                'id': 'real_sociedad', 'name': 'Real Sociedad', 'short_name': 'Real Sociedad',
                'founded': 1909, 'stadium': 'Reale Arena', 'capacity': 39500,
                'colors': {'primary': '#0066CC', 'secondary': '#FFFFFF'},
                'aliases': ['Real Sociedad de Fútbol', 'La Real']
            },
            {
                'id': 'villarreal', 'name': 'Villarreal', 'short_name': 'Villarreal',
                'founded': 1923, 'stadium': 'Estadio de la Cerámica', 'capacity': 23008,
                'colors': {'primary': '#FFFF00', 'secondary': '#000000'},
                'aliases': ['Villarreal CF', 'Yellow Submarine']
            },
            {
                'id': 'real_betis', 'name': 'Real Betis', 'short_name': 'Betis',
                'founded': 1907, 'stadium': 'Benito Villamarín', 'capacity': 60721,
                'colors': {'primary': '#00A651', 'secondary': '#FFFFFF'},
                'aliases': ['Real Betis Balompié', 'Betis']
            }
        ]
        
        for team_data in la_liga_data:
            team_id = team_data['id']
            teams[team_id] = TeamMetadata(
                id=team_id,
                name=team_data['name'],
                short_name=team_data['short_name'],
                display_name=team_data['name'],
                league='La Liga',
                country='Spain',
                founded=team_data.get('founded'),
                stadium=team_data.get('stadium'),
                capacity=team_data.get('capacity'),
                colors=team_data.get('colors', {}),
                badge_url=None,  # Would be populated from API
                website=f"https://www.{team_data['name'].lower().replace(' ', '').replace('é', 'e')}.com",
                aliases=team_data.get('aliases', []),
                current_form=['W', 'D', 'W', 'W', 'D'],  # Mock form
                league_position=None
            )
        
        return teams
    
    def _get_bundesliga_teams(self) -> Dict[str, TeamMetadata]:
        """Get Bundesliga teams with comprehensive metadata."""
        teams = {}
        
        bundesliga_data = [
            {
                'id': 'bayern_munich', 'name': 'Bayern Munich', 'short_name': 'Bayern',
                'founded': 1900, 'stadium': 'Allianz Arena', 'capacity': 75024,
                'colors': {'primary': '#DC052D', 'secondary': '#FFFFFF'},
                'aliases': ['FC Bayern München', 'Bayern München', 'FCB']
            },
            {
                'id': 'borussia_dortmund', 'name': 'Borussia Dortmund', 'short_name': 'Dortmund',
                'founded': 1909, 'stadium': 'Signal Iduna Park', 'capacity': 81365,
                'colors': {'primary': '#FDE100', 'secondary': '#000000'},
                'aliases': ['BVB', 'Dortmund', 'Borussia']
            },
            {
                'id': 'rb_leipzig', 'name': 'RB Leipzig', 'short_name': 'Leipzig',
                'founded': 2009, 'stadium': 'Red Bull Arena', 'capacity': 47069,
                'colors': {'primary': '#DC143C', 'secondary': '#FFFFFF'},
                'aliases': ['Leipzig', 'RBL']
            },
            {
                'id': 'union_berlin', 'name': 'Union Berlin', 'short_name': 'Union',
                'founded': 1966, 'stadium': 'Stadion An der Alten Försterei', 'capacity': 22012,
                'colors': {'primary': '#DC143C', 'secondary': '#FFFFFF'},
                'aliases': ['1. FC Union Berlin', 'Union']
            },
            {
                'id': 'bayer_leverkusen', 'name': 'Bayer Leverkusen', 'short_name': 'Leverkusen',
                'founded': 1904, 'stadium': 'BayArena', 'capacity': 30210,
                'colors': {'primary': '#E32221', 'secondary': '#000000'},
                'aliases': ['Bayer 04 Leverkusen', 'Leverkusen']
            },
            {
                'id': 'eintracht_frankfurt', 'name': 'Eintracht Frankfurt', 'short_name': 'Frankfurt',
                'founded': 1899, 'stadium': 'Deutsche Bank Park', 'capacity': 51500,
                'colors': {'primary': '#E1000F', 'secondary': '#000000'},
                'aliases': ['Frankfurt', 'SGE']
            }
        ]
        
        for team_data in bundesliga_data:
            team_id = team_data['id']
            teams[team_id] = TeamMetadata(
                id=team_id,
                name=team_data['name'],
                short_name=team_data['short_name'],
                display_name=team_data['name'],
                league='Bundesliga',
                country='Germany',
                founded=team_data.get('founded'),
                stadium=team_data.get('stadium'),
                capacity=team_data.get('capacity'),
                colors=team_data.get('colors', {}),
                badge_url=None,
                website=f"https://www.{team_data['name'].lower().replace(' ', '').replace('ü', 'u')}.de",
                aliases=team_data.get('aliases', []),
                current_form=['W', 'W', 'W', 'D', 'W'],  # Mock form
                league_position=None
            )
        
        return teams
    
    def _get_serie_a_teams(self) -> Dict[str, TeamMetadata]:
        """Get Serie A teams with comprehensive metadata."""
        teams = {}
        
        serie_a_data = [
            {
                'id': 'inter_milan', 'name': 'Inter Milan', 'short_name': 'Inter',
                'founded': 1908, 'stadium': 'San Siro', 'capacity': 75923,
                'colors': {'primary': '#0068A8', 'secondary': '#000000'},
                'aliases': ['FC Internazionale Milano', 'Inter', 'Internazionale']
            },
            {
                'id': 'ac_milan', 'name': 'AC Milan', 'short_name': 'Milan',
                'founded': 1899, 'stadium': 'San Siro', 'capacity': 75923,
                'colors': {'primary': '#FB090B', 'secondary': '#000000'},
                'aliases': ['Milan', 'AC Milan', 'Associazione Calcio Milan']
            },
            {
                'id': 'juventus', 'name': 'Juventus', 'short_name': 'Juventus',
                'founded': 1897, 'stadium': 'Allianz Stadium', 'capacity': 41507,
                'colors': {'primary': '#000000', 'secondary': '#FFFFFF'},
                'aliases': ['Juventus FC', 'Juve', 'La Vecchia Signora']
            },
            {
                'id': 'atalanta', 'name': 'Atalanta', 'short_name': 'Atalanta',
                'founded': 1907, 'stadium': 'Gewiss Stadium', 'capacity': 21300,
                'colors': {'primary': '#1E90FF', 'secondary': '#000000'},
                'aliases': ['Atalanta BC', 'La Dea']
            },
            {
                'id': 'roma', 'name': 'Roma', 'short_name': 'Roma',
                'founded': 1927, 'stadium': 'Stadio Olimpico', 'capacity': 70634,
                'colors': {'primary': '#CC0000', 'secondary': '#FFCC00'},
                'aliases': ['AS Roma', 'I Giallorossi']
            },
            {
                'id': 'lazio', 'name': 'Lazio', 'short_name': 'Lazio',
                'founded': 1900, 'stadium': 'Stadio Olimpico', 'capacity': 70634,
                'colors': {'primary': '#87CEEB', 'secondary': '#FFFFFF'},
                'aliases': ['SS Lazio', 'I Biancocelesti']
            }
        ]
        
        for team_data in serie_a_data:
            team_id = team_data['id']
            teams[team_id] = TeamMetadata(
                id=team_id,
                name=team_data['name'],
                short_name=team_data['short_name'],
                display_name=team_data['name'],
                league='Serie A',
                country='Italy',
                founded=team_data.get('founded'),
                stadium=team_data.get('stadium'),
                capacity=team_data.get('capacity'),
                colors=team_data.get('colors', {}),
                badge_url=None,
                website=f"https://www.{team_data['name'].lower().replace(' ', '')}.it",
                aliases=team_data.get('aliases', []),
                current_form=['D', 'W', 'W', 'D', 'W'],  # Mock form
                league_position=None
            )
        
        return teams
    
    def _get_ligue_1_teams(self) -> Dict[str, TeamMetadata]:
        """Get Ligue 1 teams with comprehensive metadata."""
        teams = {}
        
        ligue_1_data = [
            {
                'id': 'psg', 'name': 'PSG', 'short_name': 'PSG',
                'founded': 1970, 'stadium': 'Parc des Princes', 'capacity': 47929,
                'colors': {'primary': '#004170', 'secondary': '#FF0000'},
                'aliases': ['Paris Saint-Germain', 'Paris SG', 'Paris']
            },
            {
                'id': 'monaco', 'name': 'Monaco', 'short_name': 'Monaco',
                'founded': 1924, 'stadium': 'Stade Louis II', 'capacity': 18523,
                'colors': {'primary': '#FF0000', 'secondary': '#FFFFFF'},
                'aliases': ['AS Monaco', 'AS Monaco FC']
            },
            {
                'id': 'lille', 'name': 'Lille', 'short_name': 'Lille',
                'founded': 1944, 'stadium': 'Stade Pierre-Mauroy', 'capacity': 50186,
                'colors': {'primary': '#FF0000', 'secondary': '#000000'},
                'aliases': ['LOSC Lille', 'LOSC']
            },
            {
                'id': 'nice', 'name': 'Nice', 'short_name': 'Nice',
                'founded': 1904, 'stadium': 'Allianz Riviera', 'capacity': 35624,
                'colors': {'primary': '#FF0000', 'secondary': '#000000'},
                'aliases': ['OGC Nice', 'Les Aiglons']
            },
            {
                'id': 'rennes', 'name': 'Rennes', 'short_name': 'Rennes',
                'founded': 1901, 'stadium': 'Roazhon Park', 'capacity': 29778,
                'colors': {'primary': '#FF0000', 'secondary': '#000000'},
                'aliases': ['Stade Rennais FC', 'Stade Rennais']
            },
            {
                'id': 'lyon', 'name': 'Lyon', 'short_name': 'Lyon',
                'founded': 1950, 'stadium': 'Groupama Stadium', 'capacity': 59186,
                'colors': {'primary': '#FFFFFF', 'secondary': '#FF0000'},
                'aliases': ['Olympique Lyonnais', 'OL']
            }
        ]
        
        for team_data in ligue_1_data:
            team_id = team_data['id']
            teams[team_id] = TeamMetadata(
                id=team_id,
                name=team_data['name'],
                short_name=team_data['short_name'],
                display_name=team_data['name'],
                league='Ligue 1',
                country='France',
                founded=team_data.get('founded'),
                stadium=team_data.get('stadium'),
                capacity=team_data.get('capacity'),
                colors=team_data.get('colors', {}),
                badge_url=None,
                website=f"https://www.{team_data['name'].lower().replace(' ', '')}.fr",
                aliases=team_data.get('aliases', []),
                current_form=['W', 'D', 'D', 'W', 'W'],  # Mock form
                league_position=None
            )
        
        return teams
    
    def _get_eredivisie_teams(self) -> Dict[str, TeamMetadata]:
        """Get Eredivisie teams with comprehensive metadata."""
        teams = {}
        
        eredivisie_data = [
            {
                'id': 'psv', 'name': 'PSV', 'short_name': 'PSV',
                'founded': 1913, 'stadium': 'Philips Stadion', 'capacity': 35000,
                'colors': {'primary': '#FF0000', 'secondary': '#FFFFFF'},
                'aliases': ['PSV Eindhoven', 'Eindhoven']
            },
            {
                'id': 'ajax', 'name': 'Ajax', 'short_name': 'Ajax',
                'founded': 1900, 'stadium': 'Johan Cruyff Arena', 'capacity': 54990,
                'colors': {'primary': '#FF0000', 'secondary': '#FFFFFF'},
                'aliases': ['AFC Ajax', 'Ajax Amsterdam']
            },
            {
                'id': 'feyenoord', 'name': 'Feyenoord', 'short_name': 'Feyenoord',
                'founded': 1908, 'stadium': 'De Kuip', 'capacity': 51117,
                'colors': {'primary': '#FF0000', 'secondary': '#FFFFFF'},
                'aliases': ['Feyenoord Rotterdam']
            },
            {
                'id': 'az_alkmaar', 'name': 'AZ Alkmaar', 'short_name': 'AZ',
                'founded': 1967, 'stadium': 'AFAS Stadion', 'capacity': 17023,
                'colors': {'primary': '#FF0000', 'secondary': '#FFFFFF'},
                'aliases': ['AZ', 'Alkmaar Zaanstreek']
            },
            {
                'id': 'twente', 'name': 'Twente', 'short_name': 'Twente',
                'founded': 1965, 'stadium': 'De Grolsch Veste', 'capacity': 30205,
                'colors': {'primary': '#FF0000', 'secondary': '#FFFFFF'},
                'aliases': ['FC Twente', 'FC Twente Enschede']
            },
            {
                'id': 'utrecht', 'name': 'Utrecht', 'short_name': 'Utrecht',
                'founded': 1970, 'stadium': 'Stadion Galgenwaard', 'capacity': 23750,
                'colors': {'primary': '#FF0000', 'secondary': '#FFFFFF'},
                'aliases': ['FC Utrecht']
            }
        ]
        
        for team_data in eredivisie_data:
            team_id = team_data['id']
            teams[team_id] = TeamMetadata(
                id=team_id,
                name=team_data['name'],
                short_name=team_data['short_name'],
                display_name=team_data['name'],
                league='Eredivisie',
                country='Netherlands',
                founded=team_data.get('founded'),
                stadium=team_data.get('stadium'),
                capacity=team_data.get('capacity'),
                colors=team_data.get('colors', {}),
                badge_url=None,
                website=f"https://www.{team_data['name'].lower().replace(' ', '')}.nl",
                aliases=team_data.get('aliases', []),
                current_form=['W', 'W', 'D', 'W', 'D'],  # Mock form
                league_position=None
            )
        
        return teams
    
    def _initialize_league_mappings(self) -> Dict[str, str]:
        """Initialize league code mappings."""
        return {
            'PL': 'Premier League',
            'EPL': 'Premier League',
            'PREMIER_LEAGUE': 'Premier League',
            'LA_LIGA': 'La Liga',
            'LALIGA': 'La Liga',
            'PRIMERA_DIVISION': 'La Liga',
            'BUNDESLIGA': 'Bundesliga',
            'BL1': 'Bundesliga',
            'SERIE_A': 'Serie A',
            'SA': 'Serie A',
            'LIGUE_1': 'Ligue 1',
            'FL1': 'Ligue 1',
            'EREDIVISIE': 'Eredivisie',
            'DED': 'Eredivisie'
        }
    
    def _initialize_name_variations(self) -> Dict[str, str]:
        """Initialize common team name variations and mappings."""
        variations = {}
        
        # Build variations from all teams
        for league_teams in self.team_database.values():
            for team_id, team_metadata in league_teams.items():
                # Add main name
                variations[team_metadata.name.lower()] = team_id
                variations[team_metadata.short_name.lower()] = team_id
                
                # Add aliases
                for alias in team_metadata.aliases:
                    variations[alias.lower()] = team_id
                
                # Add common variations
                name_lower = team_metadata.name.lower()
                variations[name_lower.replace(' ', '_')] = team_id
                variations[name_lower.replace(' ', '')] = team_id
        
        return variations
    
    def resolve_team(self, team_input: str, league: str = None) -> Optional[TeamMetadata]:
        """
        Resolve team input to standardized team metadata.
        
        Args:
            team_input: Team name, ID, or alias
            league: Optional league filter
            
        Returns:
            TeamMetadata if found, None otherwise
        """
        if not team_input:
            return None
        
        team_input_lower = str(team_input).lower().strip()
        
        # Try direct lookup in name variations
        if team_input_lower in self.name_variations:
            team_id = self.name_variations[team_input_lower]
            
            # Find the team in the database
            for league_name, teams in self.team_database.items():
                if league and league != league_name:
                    continue
                if team_id in teams:
                    return teams[team_id]
        
        # Try fuzzy matching
        return self._fuzzy_match_team(team_input, league)
    
    def _fuzzy_match_team(self, team_input: str, league: str = None) -> Optional[TeamMetadata]:
        """Perform fuzzy matching for team names."""
        team_input_lower = team_input.lower()
        best_match = None
        best_score = 0
        
        leagues_to_search = [league] if league else self.team_database.keys()
        
        for league_name in leagues_to_search:
            if league_name not in self.team_database:
                continue
                
            for team_id, team_metadata in self.team_database[league_name].items():
                # Check main name
                score = self._calculate_similarity(team_input_lower, team_metadata.name.lower())
                if score > best_score:
                    best_score = score
                    best_match = team_metadata
                
                # Check aliases
                for alias in team_metadata.aliases:
                    score = self._calculate_similarity(team_input_lower, alias.lower())
                    if score > best_score:
                        best_score = score
                        best_match = team_metadata
        
        # Return match if similarity is above threshold
        return best_match if best_score > 0.7 else None
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using simple algorithm."""
        if str1 == str2:
            return 1.0
        
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Simple character overlap calculation
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_team_display_info(self, team_metadata: TeamMetadata) -> Dict[str, Any]:
        """Get formatted team display information."""
        return {
            'name': team_metadata.display_name,
            'short_name': team_metadata.short_name,
            'league': team_metadata.league,
            'country': team_metadata.country,
            'colors': team_metadata.colors,
            'badge_url': team_metadata.badge_url,
            'stadium': team_metadata.stadium,
            'capacity': team_metadata.capacity,
            'founded': team_metadata.founded,
            'current_form': team_metadata.current_form,
            'form_display': ''.join(team_metadata.current_form[-5:]) if team_metadata.current_form else 'N/A'
        }
    
    def validate_cross_league_teams(self, home_team: str, away_team: str, 
                                  home_league: str = None, away_league: str = None) -> Dict[str, Any]:
        """Validate teams for cross-league analysis."""
        home_metadata = self.resolve_team(home_team, home_league)
        away_metadata = self.resolve_team(away_team, away_league)
        
        if not home_metadata or not away_metadata:
            return {
                'valid': False,
                'error': 'One or both teams could not be resolved',
                'home_resolved': home_metadata is not None,
                'away_resolved': away_metadata is not None
            }
        
        return {
            'valid': True,
            'home_team': self.get_team_display_info(home_metadata),
            'away_team': self.get_team_display_info(away_metadata),
            'is_cross_league': home_metadata.league != away_metadata.league,
            'leagues': [home_metadata.league, away_metadata.league]
        }
    
    def get_league_teams(self, league: str) -> List[str]:
        """Get list of team names for a specific league."""
        if league not in self.team_database:
            return []
        
        return [team.display_name for team in self.team_database[league].values()]
    
    def get_all_supported_leagues(self) -> List[str]:
        """Get list of all supported leagues."""
        return list(self.team_database.keys())

# Global instance
_enhanced_team_data_manager = None

def get_enhanced_team_data_manager() -> EnhancedTeamDataManager:
    """Get global enhanced team data manager instance."""
    global _enhanced_team_data_manager
    if _enhanced_team_data_manager is None:
        _enhanced_team_data_manager = EnhancedTeamDataManager()
    return _enhanced_team_data_manager
