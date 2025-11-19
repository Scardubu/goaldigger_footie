"""
Enhanced data migration script to populate database with proper team metadata.

This script ensures all teams have proper names, flags, country information,
and other metadata for a professional display.
"""
import logging
from datetime import datetime
from typing import Dict, List

from database.db_manager import DatabaseManager
from database.schema import League, Team
from utils.team_data_enhancer import team_enhancer

logger = logging.getLogger(__name__)


class EnhancedDataMigration:
    """Enhanced data migration with proper team metadata."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
    def migrate_team_metadata(self):
        """Migrate and enhance team metadata in the database."""
        logger.info("Starting enhanced team metadata migration...")
        
        with self.db.session_scope() as session:
            # Get all existing teams
            teams = session.query(Team).all()
            logger.info(f"Found {len(teams)} teams to migrate")
            
            migrated_count = 0
            for team in teams:
                try:
                    # Get enhanced team data
                    enhanced_data = team_enhancer.get_team_data(team.name)
                    
                    # Update team with enhanced metadata
                    team.country = enhanced_data.get('country', 'Unknown')
                    team.country_code = enhanced_data.get('country_code', 'XX')
                    team.team_flag = enhanced_data.get('flag', '⚽')
                    team.country_flag = enhanced_data.get('country_flag', '🏳️')
                    team.primary_color = enhanced_data.get('color', '#667eea')
                    
                    # Update other fields if they're missing
                    if not team.short_name:
                        team.short_name = enhanced_data.get('short_name', team.name[:3].upper())
                    if not team.tla:
                        team.tla = enhanced_data.get('tla', team.name[:3].upper())
                    if not team.venue:
                        team.venue = enhanced_data.get('venue', f"{team.name} Stadium")
                    if not team.venue_capacity:
                        team.venue_capacity = enhanced_data.get('capacity', 30000)
                    
                    migrated_count += 1
                    
                except Exception as e:
                    logger.error(f"Error migrating team {team.name}: {e}")
                    continue
            
            session.commit()
            logger.info(f"Successfully migrated {migrated_count} teams with enhanced metadata")
    
    def ensure_leagues_have_metadata(self):
        """Ensure all leagues have proper metadata."""
        logger.info("Ensuring leagues have proper metadata...")
        
        with self.db.session_scope() as session:
            leagues = session.query(League).all()
            
            for league in leagues:
                # Get league info from team enhancer
                league_info = team_enhancer.get_league_info(league.api_id or league.id)
                
                # Update league metadata
                if not league.country:
                    league.country = league_info.get('country', 'Unknown')
                
                session.commit()
            
            logger.info("League metadata migration completed")
    
    def create_missing_teams_from_enhanced_data(self):
        """Create missing teams based on enhanced team data."""
        logger.info("Creating missing teams from enhanced data...")
        with self.db.session_scope() as session:
            created_count = 0
            for team_name, team_data in team_enhancer.team_database.items():
                # Normalize team name and TLA
                team_name_norm = team_name.strip().title()
                tla_norm = team_data['tla'].strip().upper()
                # Check if team already exists by id (TLA) or name
                existing_team_by_id = session.query(Team).filter_by(id=tla_norm).first()
                existing_team_by_name = session.query(Team).filter_by(name=team_name_norm).first()
                if existing_team_by_id or existing_team_by_name:
                    continue
                # Get or create league
                league_code = team_data['league_code'].strip().upper()
                league = session.query(League).filter_by(api_id=league_code).first()
                if not league:
                    league_info = team_enhancer.get_league_info(league_code)
                    league = League(
                        id=league_code,
                        name=league_info['name'].strip().title(),
                        country=league_info['country'].strip().title(),
                        api_id=league_code,
                        tier=1
                    )
                    session.add(league)
                    session.flush()
                # Create team with normalized data
                team = Team(
                    id=tla_norm,
                    name=team_name_norm,
                    short_name=team_data['short_name'].strip().title(),
                    tla=tla_norm,
                    league_id=league.id,
                    venue=team_data['venue'].strip().title() if team_data.get('venue') else None,
                    venue_capacity=team_data['capacity'],
                    country=team_data['country'].strip().title(),
                    country_code=team_data['country_code'].strip().upper(),
                    team_flag=team_data['flag'],
                    country_flag=team_data['country_flag'],
                    primary_color=team_data['color']
                )
                session.add(team)
                created_count += 1
            session.commit()
            logger.info(f"Created {created_count} new teams with enhanced metadata")
    
    def run_full_migration(self):
        """Run the complete enhanced data migration."""
        logger.info("Starting full enhanced data migration...")
        
        try:
            # Ensure leagues have metadata
            self.ensure_leagues_have_metadata()
            
            # Create missing teams
            self.create_missing_teams_from_enhanced_data()
            
            # Migrate existing team metadata
            self.migrate_team_metadata()
            
            logger.info("Enhanced data migration completed successfully")
            
        except Exception as e:
            logger.error(f"Error during enhanced data migration: {e}")
            raise


def run_enhanced_migration(db_uri: str = None):
    """Convenience function to run enhanced data migration."""
    db_manager = DatabaseManager(db_uri)
    migration = EnhancedDataMigration(db_manager)
    migration.run_full_migration()


if __name__ == "__main__":
    # Run migration when script is executed directly
    run_enhanced_migration()
