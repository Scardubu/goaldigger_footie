"""
Data quality checks and consistency validators for the football data pipeline.

Provides validation functions to ensure data integrity, completeness,
and consistency across the normalized schema.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from database.db_manager import DatabaseManager
from database.schema import League, Match, Team

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Data quality validation and consistency checks."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def validate_match_data(self, session: Session) -> Dict[str, List[str]]:
        """Validate match data quality and return issues found."""
        issues = {
            'missing_teams': [],
            'missing_leagues': [],
            'invalid_dates': [],
            'duplicate_matches': [],
            'inconsistent_scores': []
        }
        
        # Check for matches with missing team references
        matches_without_home = session.query(Match).filter(
            ~Match.home_team_id.in_(session.query(Team.id))
        ).all()
        for match in matches_without_home:
            issues['missing_teams'].append(f"Match {match.id}: missing home team {match.home_team_id}")
            
        matches_without_away = session.query(Match).filter(
            ~Match.away_team_id.in_(session.query(Team.id))
        ).all()
        for match in matches_without_away:
            issues['missing_teams'].append(f"Match {match.id}: missing away team {match.away_team_id}")
            
        # Check for matches with missing league references
        matches_without_league = session.query(Match).filter(
            ~Match.league_id.in_(session.query(League.id))
        ).all()
        for match in matches_without_league:
            issues['missing_leagues'].append(f"Match {match.id}: missing league {match.league_id}")
            
        # Check for invalid dates (too far in past/future)
    now = datetime.now(timezone.utc)
    too_old = (now - timedelta(days=365 * 5)).replace(tzinfo=None)  # 5 years ago
    too_future = (now + timedelta(days=365)).replace(tzinfo=None)    # 1 year ahead
        
        invalid_dates = session.query(Match).filter(
            (Match.match_date < too_old) | (Match.match_date > too_future)
        ).all()
        for match in invalid_dates:
            issues['invalid_dates'].append(f"Match {match.id}: date {match.match_date} out of range")
            
        # Check for duplicate matches (same teams, same date)
        duplicates = session.query(Match).group_by(
            Match.home_team_id, Match.away_team_id, Match.match_date
        ).having(func.count(Match.id) > 1).all()
        for match in duplicates:
            issues['duplicate_matches'].append(f"Duplicate match: {match.home_team_id} vs {match.away_team_id} on {match.match_date}")
            
        # Check for inconsistent scores (negative scores, impossible scores)
        invalid_scores = session.query(Match).filter(
            (Match.home_score < 0) | (Match.away_score < 0) |
            (Match.home_score > 20) | (Match.away_score > 20)
        ).all()
        for match in invalid_scores:
            issues['inconsistent_scores'].append(f"Match {match.id}: invalid scores {match.home_score}-{match.away_score}")
            
        return issues
        
    def validate_team_data(self, session: Session) -> Dict[str, List[str]]:
        """Validate team data quality."""
        issues = {
            'missing_leagues': [],
            'duplicate_teams': [],
            'invalid_names': []
        }
        
        # Check for teams with missing league references
        teams_without_league = session.query(Team).filter(
            ~Team.league_id.in_(session.query(League.id))
        ).all()
        for team in teams_without_league:
            issues['missing_leagues'].append(f"Team {team.id}: missing league {team.league_id}")
            
        # Check for duplicate team names within same league
        duplicates = session.query(Team).group_by(Team.name, Team.league_id).having(
            func.count(Team.id) > 1
        ).all()
        for team in duplicates:
            issues['duplicate_teams'].append(f"Duplicate team: {team.name} in league {team.league_id}")
            
        # Check for invalid team names (empty, too short, too long)
        invalid_names = session.query(Team).filter(
            (Team.name == '') | (func.length(Team.name) < 2) | (func.length(Team.name) > 100)
        ).all()
        for team in invalid_names:
            issues['invalid_names'].append(f"Team {team.id}: invalid name '{team.name}'")
            
        return issues
        
    def run_comprehensive_check(self) -> Dict[str, Dict[str, List[str]]]:
        """Run all data quality checks and return comprehensive report."""
        with self.db.session_scope() as session:
            match_issues = self.validate_match_data(session)
            team_issues = self.validate_team_data(session)
            
            return {
                'matches': match_issues,
                'teams': team_issues,
                'summary': self._generate_summary(match_issues, team_issues)
            }
            
    def _generate_summary(self, match_issues: Dict, team_issues: Dict) -> Dict[str, int]:
        """Generate summary statistics of issues found."""
        total_issues = 0
        for category in match_issues.values():
            total_issues += len(category)
        for category in team_issues.values():
            total_issues += len(category)
            
        return {
            'total_issues': total_issues,
            'critical_issues': len(match_issues['missing_teams']) + len(match_issues['missing_leagues']),
            'data_quality_score': max(0, 100 - min(100, total_issues * 2))  # Simple scoring
        }


def validate_data_quality(db_uri: Optional[str] = None) -> Dict:
    """Convenience function to run data quality validation."""
    db = DatabaseManager(db_uri)
    checker = DataQualityChecker(db)
    return checker.run_comprehensive_check()
