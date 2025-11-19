#!/usr/bin/env python3
"""
Enhanced Match Data Validator - GoalDiggers Platform

Provides comprehensive validation and correction for:
- Match dates accuracy and consistency
- Team name standardization and verification
- League assignments and cross-league validation
- Fixture data integrity and real-time updates
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

from database.db_manager import DatabaseManager
from database.schema import League, Match, Team

logger = logging.getLogger(__name__)


class EnhancedMatchDataValidator:
    """Enhanced match data validator with comprehensive verification."""
    
    def __init__(self):
        """Initialize the match data validator."""
        self.db_manager = DatabaseManager()
        self.logger = logger
        
        # Standard team name mappings
        self.team_name_mappings = self._initialize_team_mappings()
        
        # League configurations
        self.league_configs = self._initialize_league_configs()
        
        # Common date patterns
        self.date_patterns = [
            r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'[A-Za-z]+ \d{1,2}, \d{4}',  # Month DD, YYYY
        ]
    
    def validate_and_correct_match_data(self, home_team: str, away_team: str, 
                                      match_date: Optional[datetime] = None,
                                      league: Optional[str] = None) -> Dict:
        """
        Validate and correct match data comprehensively.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Optional match date
            league: Optional league specification
            
        Returns:
            Dict containing validated and corrected match data
        """
        validation_result = {
            'original': {
                'home_team': home_team,
                'away_team': away_team,
                'match_date': match_date,
                'league': league
            },
            'validated': {},
            'corrections': [],
            'warnings': [],
            'errors': [],
            'confidence': 1.0,
            'is_valid': True
        }
        
        try:
            # Validate and correct team names
            validated_teams = self._validate_team_names(home_team, away_team)
            validation_result['validated'].update(validated_teams)
            
            if validated_teams.get('corrections'):
                validation_result['corrections'].extend(validated_teams['corrections'])
                validation_result['confidence'] *= 0.9
            
            # Validate and correct match date
            validated_date = self._validate_match_date(match_date)
            validation_result['validated'].update(validated_date)
            
            if validated_date.get('corrections'):
                validation_result['corrections'].extend(validated_date['corrections'])
                validation_result['confidence'] *= 0.95
            
            # Validate league assignment
            validated_league = self._validate_league_assignment(
                validation_result['validated']['home_team'],
                validation_result['validated']['away_team'],
                league
            )
            validation_result['validated'].update(validated_league)
            
            if validated_league.get('warnings'):
                validation_result['warnings'].extend(validated_league['warnings'])
            
            # Cross-validate team compatibility
            compatibility_check = self._check_team_compatibility(
                validation_result['validated']['home_team'],
                validation_result['validated']['away_team'],
                validation_result['validated']['league']
            )
            
            if compatibility_check.get('warnings'):
                validation_result['warnings'].extend(compatibility_check['warnings'])
                validation_result['confidence'] *= 0.85
            
            # Final validation status
            validation_result['is_valid'] = (
                len(validation_result['errors']) == 0 and
                validation_result['confidence'] > 0.5
            )
            
            self.logger.info(f"Validation complete: {validation_result['confidence']:.2%} confidence")
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            validation_result['errors'].append(f"Validation error: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['confidence'] = 0.0
        
        return validation_result
    
    def _validate_team_names(self, home_team: str, away_team: str) -> Dict:
        """Validate and standardize team names."""
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'corrections': []
        }
        
        try:
            # Standardize home team
            standardized_home = self._standardize_team_name(home_team)
            if standardized_home != home_team:
                result['home_team'] = standardized_home
                result['corrections'].append(f"Home team name standardized: {home_team} → {standardized_home}")
            
            # Standardize away team
            standardized_away = self._standardize_team_name(away_team)
            if standardized_away != away_team:
                result['away_team'] = standardized_away
                result['corrections'].append(f"Away team name standardized: {away_team} → {standardized_away}")
            
            # Check for same team
            if result['home_team'].lower() == result['away_team'].lower():
                raise ValueError(f"Same team cannot play itself: {result['home_team']}")
            
            # Verify teams exist in database
            self._verify_teams_in_database(result['home_team'], result['away_team'])
            
        except Exception as e:
            self.logger.warning(f"Team name validation warning: {e}")
        
        return result
    
    def _validate_match_date(self, match_date: Optional[datetime]) -> Dict:
        """Validate and correct match date."""
        result = {
            'match_date': match_date,
            'corrections': []
        }
        
        try:
            current_date = datetime.now()
            
            if match_date is None:
                # Default to tomorrow for new predictions
                result['match_date'] = current_date + timedelta(days=1)
                result['corrections'].append("Match date defaulted to tomorrow")
            
            elif isinstance(match_date, str):
                # Parse string date
                parsed_date = self._parse_date_string(match_date)
                result['match_date'] = parsed_date
                result['corrections'].append(f"Date parsed from string: {match_date} → {parsed_date}")
            
            elif isinstance(match_date, datetime):
                # Validate datetime
                if match_date < current_date - timedelta(days=365):
                    # Date too far in past
                    result['match_date'] = current_date + timedelta(days=1)
                    result['corrections'].append("Match date too far in past, adjusted to tomorrow")
                
                elif match_date > current_date + timedelta(days=365):
                    # Date too far in future
                    result['match_date'] = current_date + timedelta(days=7)
                    result['corrections'].append("Match date too far in future, adjusted to next week")
            
            # Ensure realistic match time
            if result['match_date']:
                # Typical match times: weekend afternoon/evening, midweek evening
                if result['match_date'].weekday() < 5:  # Monday-Friday
                    # Midweek match - typically evening
                    result['match_date'] = result['match_date'].replace(hour=20, minute=0, second=0, microsecond=0)
                else:  # Weekend
                    # Weekend match - typically afternoon
                    result['match_date'] = result['match_date'].replace(hour=15, minute=0, second=0, microsecond=0)
        
        except Exception as e:
            self.logger.warning(f"Date validation warning: {e}")
            # Fallback to tomorrow
            result['match_date'] = datetime.now() + timedelta(days=1)
            result['corrections'].append(f"Date validation failed, defaulted to tomorrow: {str(e)}")
        
        return result
    
    def _validate_league_assignment(self, home_team: str, away_team: str, 
                                  league: Optional[str]) -> Dict:
        """Validate league assignment for teams."""
        result = {
            'league': league,
            'warnings': []
        }
        
        try:
            # Get team league information from database
            home_league = self._get_team_league(home_team)
            away_league = self._get_team_league(away_team)
            
            if home_league and away_league:
                if home_league == away_league:
                    # Same league match
                    if not league:
                        result['league'] = home_league
                    elif league != home_league:
                        result['warnings'].append(f"League mismatch: specified {league}, teams are in {home_league}")
                        result['league'] = home_league
                
                else:
                    # Cross-league match
                    if not league:
                        result['league'] = f"{home_league} vs {away_league}"
                    result['warnings'].append(f"Cross-league match: {home_team} ({home_league}) vs {away_team} ({away_league})")
            
            else:
                # Unknown leagues
                if not league:
                    result['league'] = "Mixed / Unknown"
                result['warnings'].append("Unable to determine team leagues from database")
        
        except Exception as e:
            self.logger.warning(f"League validation warning: {e}")
            if not league:
                result['league'] = "Unknown"
        
        return result
    
    def _check_team_compatibility(self, home_team: str, away_team: str, 
                                league: str) -> Dict:
        """Check if teams are compatible for prediction."""
        result = {
            'compatible': True,
            'warnings': []
        }
        
        try:
            # Check realistic matchup
            unrealistic_combinations = [
                # Add specific unrealistic team combinations if needed
                # e.g., teams that never play each other
            ]
            
            team_pair = (home_team.lower(), away_team.lower())
            reverse_pair = (away_team.lower(), home_team.lower())
            
            if team_pair in unrealistic_combinations or reverse_pair in unrealistic_combinations:
                result['warnings'].append(f"Potentially unrealistic matchup: {home_team} vs {away_team}")
                result['compatible'] = False
            
            # Check league compatibility
            if "vs" in league and "Unknown" not in league:
                result['warnings'].append("Cross-league match may have limited historical data")
            
        except Exception as e:
            self.logger.warning(f"Compatibility check warning: {e}")
        
        return result
    
    def _standardize_team_name(self, team_name: str) -> str:
        """Standardize team name using mappings."""
        if not team_name:
            return team_name
        
        # Clean the name
        cleaned_name = team_name.strip()
        
        # Check direct mappings
        for standard_name, variations in self.team_name_mappings.items():
            if cleaned_name.lower() in [v.lower() for v in variations]:
                return standard_name
        
        # Check fuzzy matching for common abbreviations
        cleaned_lower = cleaned_name.lower()
        
        # Common patterns
        if cleaned_lower.endswith(' fc'):
            base_name = cleaned_name[:-3].strip()
            return f"{base_name} FC"
        elif cleaned_lower.endswith(' united'):
            base_name = cleaned_name[:-7].strip()
            return f"{base_name} United"
        elif cleaned_lower.endswith(' city'):
            base_name = cleaned_name[:-5].strip()
            return f"{base_name} City"
        
        # Return as-is if no standardization found
        return cleaned_name
    
    def _parse_date_string(self, date_string: str) -> datetime:
        """Parse various date string formats."""
        # Try common formats
        formats = [
            '%d/%m/%Y',
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y'
        ]
        
        for format_str in formats:
            try:
                return datetime.strptime(date_string.strip(), format_str)
            except ValueError:
                continue
        
        # If no format matches, try regex patterns
        for pattern in self.date_patterns:
            match = re.search(pattern, date_string)
            if match:
                # Extract and try to parse the matched portion
                date_part = match.group(0)
                for format_str in formats:
                    try:
                        return datetime.strptime(date_part, format_str)
                    except ValueError:
                        continue
        
        # Fallback: assume it's tomorrow
        raise ValueError(f"Unable to parse date string: {date_string}")
    
    def _get_team_league(self, team_name: str) -> Optional[str]:
        """Get team's league from database."""
        try:
            with self.db_manager.session_scope() as session:
                team = session.query(Team).filter(Team.name == team_name).first()
                if team and team.league:
                    return team.league.name
                return None
        except Exception as e:
            self.logger.debug(f"Could not get league for {team_name}: {e}")
            return None
    
    def _verify_teams_in_database(self, home_team: str, away_team: str):
        """Verify that teams exist in the database."""
        try:
            with self.db_manager.session_scope() as session:
                home_exists = session.query(Team).filter(Team.name == home_team).first() is not None
                away_exists = session.query(Team).filter(Team.name == away_team).first() is not None
                
                if not home_exists:
                    self.logger.info(f"Home team not in database: {home_team}")
                
                if not away_exists:
                    self.logger.info(f"Away team not in database: {away_team}")
        
        except Exception as e:
            self.logger.debug(f"Database verification warning: {e}")
    
    def _initialize_team_mappings(self) -> Dict[str, List[str]]:
        """Initialize team name standardization mappings."""
        return {
            # Premier League
            'Arsenal': ['Arsenal FC', 'Arsenal F.C.', 'The Gunners'],
            'Chelsea': ['Chelsea FC', 'Chelsea F.C.', 'The Blues'],
            'Liverpool': ['Liverpool FC', 'Liverpool F.C.', 'The Reds'],
            'Manchester City': ['Manchester City FC', 'Man City', 'City', 'MCFC'],
            'Manchester United': ['Manchester United FC', 'Man United', 'Man Utd', 'United', 'MUFC'],
            'Tottenham': ['Tottenham Hotspur', 'Spurs', 'Tottenham Hotspur FC'],
            'Newcastle United': ['Newcastle', 'Newcastle FC', 'The Magpies'],
            'Brighton': ['Brighton & Hove Albion', 'Brighton FC', 'The Seagulls'],
            'Aston Villa': ['Villa', 'Aston Villa FC'],
            'West Ham': ['West Ham United', 'West Ham FC', 'The Hammers'],
            
            # La Liga
            'Barcelona': ['FC Barcelona', 'Barca', 'FCB'],
            'Real Madrid': ['Real Madrid CF', 'Madrid', 'Los Blancos'],
            'Atletico Madrid': ['Atlético Madrid', 'Atletico', 'ATM'],
            'Sevilla': ['Sevilla FC', 'Sevilla F.C.'],
            'Valencia': ['Valencia CF', 'Valencia F.C.'],
            'Real Betis': ['Betis', 'Real Betis Balompié'],
            'Athletic Bilbao': ['Athletic Club', 'Athletic', 'Bilbao'],
            
            # Bundesliga
            'Bayern Munich': ['FC Bayern München', 'Bayern', 'FCB'],
            'Borussia Dortmund': ['BVB', 'Dortmund', 'BVB 09'],
            'RB Leipzig': ['Leipzig', 'Red Bull Leipzig'],
            'Bayer Leverkusen': ['Leverkusen', 'Bayer 04'],
            'Borussia Mönchengladbach': ['Mönchengladbach', 'Gladbach', 'BMG'],
            'Eintracht Frankfurt': ['Frankfurt', 'SGE'],
            
            # Serie A
            'Juventus': ['Juventus FC', 'Juve', 'Juventus F.C.'],
            'AC Milan': ['Milan', 'AC Milan'],
            'Inter Milan': ['Inter', 'FC Internazionale', 'Internazionale'],
            'Napoli': ['SSC Napoli', 'Società Sportiva Calcio Napoli'],
            'AS Roma': ['Roma', 'AS Roma'],
            'Lazio': ['SS Lazio', 'Società Sportiva Lazio'],
            
            # Ligue 1
            'Paris Saint-Germain': ['PSG', 'Paris SG', 'Paris Saint Germain'],
            'Olympique Marseille': ['Marseille', 'OM', 'Olympique de Marseille'],
            'Lyon': ['Olympique Lyonnais', 'OL'],
            'AS Monaco': ['Monaco', 'AS Monaco FC'],
        }
    
    def _initialize_league_configs(self) -> Dict[str, Dict]:
        """Initialize league-specific configurations."""
        return {
            'Premier League': {
                'country': 'England',
                'season_start': 8,  # August
                'season_end': 5,    # May
                'typical_matchdays': [5, 6, 0],  # Saturday, Sunday, Monday
                'teams_count': 20
            },
            'La Liga': {
                'country': 'Spain',
                'season_start': 8,
                'season_end': 5,
                'typical_matchdays': [5, 6],  # Saturday, Sunday
                'teams_count': 20
            },
            'Bundesliga': {
                'country': 'Germany',
                'season_start': 8,
                'season_end': 5,
                'typical_matchdays': [5, 6],  # Saturday, Sunday
                'teams_count': 18
            },
            'Serie A': {
                'country': 'Italy',
                'season_start': 8,
                'season_end': 5,
                'typical_matchdays': [6, 0],  # Sunday, Monday
                'teams_count': 20
            },
            'Ligue 1': {
                'country': 'France',
                'season_start': 8,
                'season_end': 5,
                'typical_matchdays': [5, 6],  # Saturday, Sunday
                'teams_count': 20
            }
        }
    
    def get_corrected_match_preview(self, home_team: str, away_team: str,
                                  match_date: Optional[datetime] = None,
                                  league: Optional[str] = None) -> Dict:
        """Get a corrected match preview with all validations applied."""
        validation_result = self.validate_and_correct_match_data(
            home_team, away_team, match_date, league
        )
        
        if not validation_result['is_valid']:
            return {
                'error': True,
                'message': 'Match data validation failed',
                'details': validation_result['errors']
            }
        
        validated = validation_result['validated']
        
        return {
            'home_team': validated['home_team'],
            'away_team': validated['away_team'],
            'match_date': validated['match_date'],
            'league': validated['league'],
            'venue': f"{validated['home_team']} Stadium",
            'validation_confidence': validation_result['confidence'],
            'corrections_applied': validation_result['corrections'],
            'warnings': validation_result['warnings'],
            'is_cross_league': 'vs' in validated['league'],
            'formatted_date': validated['match_date'].strftime('%d/%m/%Y %H:%M'),
            'match_day': validated['match_date'].strftime('%A'),
        }


def validate_match_data(home_team: str, away_team: str,
                       match_date: Optional[datetime] = None,
                       league: Optional[str] = None) -> Dict:
    """
    Convenience function to validate match data.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Optional match date
        league: Optional league
        
    Returns:
        Dict with validation results
    """
    validator = EnhancedMatchDataValidator()
    return validator.validate_and_correct_match_data(home_team, away_team, match_date, league)


def get_match_preview(home_team: str, away_team: str,
                     match_date: Optional[datetime] = None,
                     league: Optional[str] = None) -> Dict:
    """
    Get corrected match preview with validation.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        match_date: Optional match date
        league: Optional league
        
    Returns:
        Dict with corrected match preview
    """
    validator = EnhancedMatchDataValidator()
    return validator.get_corrected_match_preview(home_team, away_team, match_date, league)


if __name__ == "__main__":
    # Test the validator
    validator = EnhancedMatchDataValidator()
    
    # Test cases
    test_cases = [
        ("Arsenal", "Chelsea", None, None),
        ("Man City", "Liverpool FC", "2024-12-25", "Premier League"),
        ("Barcelona", "Real Madrid CF", datetime(2024, 12, 15), "La Liga"),
        ("Bayern", "Dortmund", "invalid-date", "Bundesliga")
    ]
    
    for home, away, date, league in test_cases:
        print(f"\n=== Testing: {home} vs {away} ===")
        result = validator.get_corrected_match_preview(home, away, date, league)
        print(f"Result: {result}")
