#!/usr/bin/env python
"""
Team Name Mapping Fixer

This script ensures all teams referenced in the application have proper 
database entries and fixes team name mapping issues.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path for robust imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("team_name_fixer")

try:
    from scripts.setup_env import setup_environment
    logger = setup_environment()
except ImportError:
    logger.warning("Couldn't import setup_environment, using default logging")

# Import necessary components (these imports assume the correct structure)
try:
    from database.db_manager import DatabaseManager
    from database.schema import Team
except ImportError as e:
    logger.error(f"Failed to import database components: {e}")
    logger.error("Make sure you are running this script from the project root")
    sys.exit(1)

# Common team name variants that might cause mapping issues
TEAM_NAME_VARIANTS = {
    "Arsenal": ["Arsenal FC", "Arsenal F.C.", "The Gunners"],
    "Chelsea": ["Chelsea FC", "Chelsea F.C.", "The Blues"],
    "Liverpool": ["Liverpool FC", "Liverpool F.C.", "The Reds"],
    "Manchester United": ["Man Utd", "Manchester Utd", "Man United", "Manchester Utd FC", "Manchester United FC"],
    "Manchester City": ["Man City", "Manchester City FC", "Man City FC"],
    "Tottenham Hotspur": ["Tottenham", "Spurs", "Tottenham FC", "Tottenham Hotspur FC"],
    "Leicester City": ["Leicester", "Leicester City FC", "The Foxes"],
    "West Ham United": ["West Ham", "West Ham FC", "The Hammers"],
    "Real Madrid": ["Real Madrid CF", "R. Madrid", "Real"],
    "Barcelona": ["FC Barcelona", "Barça", "FCB"],
    "Atletico Madrid": ["Atlético Madrid", "Atletico", "Atlético", "Atletico de Madrid"],
    "Bayern Munich": ["Bayern", "FC Bayern", "FC Bayern Munich", "Bayern München"],
    "Borussia Dortmund": ["Dortmund", "BVB", "BVB 09"],
    "Paris Saint-Germain": ["PSG", "Paris SG", "Paris Saint Germain"],
    "Juventus": ["Juventus FC", "Juve", "Old Lady"],
    "AC Milan": ["Milan", "A.C. Milan", "Rossoneri"],
    "Inter Milan": ["Inter", "FC Internazionale Milano", "Nerazzurri"],
}

def fix_team_name_mappings(create_missing=True, dry_run=False):
    """
    Fix team name mappings in the database.
    
    Args:
        create_missing (bool): Whether to create entries for missing teams
        dry_run (bool): If True, don't make actual changes to the database
    
    Returns:
        tuple: (teams_updated, teams_created, teams_skipped)
    """
    db = DatabaseManager()
    
    teams_updated = 0
    teams_created = 0
    teams_skipped = 0
    
    # Get all team name variants
    all_variants = {}
    for canonical, variants in TEAM_NAME_VARIANTS.items():
        all_variants[canonical] = canonical  # Map canonical name to itself
        for variant in variants:
            all_variants[variant] = canonical  # Map each variant to canonical name
    
    with db.session_scope() as session:
        # Get all existing teams
        existing_teams = session.query(Team).all()
        existing_names = {team.name: team for team in existing_teams}
        
        # Find canonical names missing from the database
        missing_canonical = set(TEAM_NAME_VARIANTS.keys()) - set(existing_names.keys())
        
        if missing_canonical and create_missing:
            logger.info(f"Found {len(missing_canonical)} missing canonical team names")
            
            for canonical_name in missing_canonical:
                if dry_run:
                    logger.info(f"[DRY RUN] Would create team: {canonical_name}")
                    teams_skipped += 1
                    continue
                
                # Create team with canonical name
                try:
                    # Generate a simple ID based on the name
                    team_id = canonical_name.lower().replace(" ", "_")
                    
                    # Get the default league_id from existing teams or fallback to "PL"
                    default_league = None
                    if existing_teams:
                        default_league = existing_teams[0].league_id
                    
                    # Create basic team entry
                    new_team = Team(
                        id=team_id,
                        name=canonical_name,
                        short_name=canonical_name.split(" ")[0],
                        tla=canonical_name[:3].upper(),
                        league_id=default_league or "PL"  # Fallback to Premier League
                    )
                    
                    session.add(new_team)
                    logger.info(f"Created team: {canonical_name}")
                    teams_created += 1
                except Exception as e:
                    logger.error(f"Error creating team {canonical_name}: {e}")
        
        # Create aliases for existing teams
        for variant, canonical in all_variants.items():
            # Skip if the variant is already a team name in the database
            if variant in existing_names:
                continue
            
            # Find the canonical team in the database
            canonical_team = next((team for team in existing_teams if team.name == canonical), None)
            
            if canonical_team is None:
                logger.warning(f"Canonical team '{canonical}' not found for variant '{variant}'")
                teams_skipped += 1
                continue
            
            if dry_run:
                logger.info(f"[DRY RUN] Would add alias '{variant}' for team '{canonical}'")
                continue
            
            # Add the variant as an alias (assuming there's an aliases field)
            try:
                if hasattr(canonical_team, 'aliases'):
                    if canonical_team.aliases:
                        # Append if existing aliases
                        current_aliases = canonical_team.aliases.split(',')
                        if variant not in current_aliases:
                            canonical_team.aliases = f"{canonical_team.aliases},{variant}"
                            teams_updated += 1
                    else:
                        # Set if no aliases yet
                        canonical_team.aliases = variant
                        teams_updated += 1
                else:
                    logger.warning(f"Team table has no 'aliases' field, can't add alias '{variant}' for '{canonical}'")
            except Exception as e:
                logger.error(f"Error updating aliases for {canonical}: {e}")
    
    return teams_updated, teams_created, teams_skipped

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Fix team name mappings in the database")
    parser.add_argument("--no-create", action="store_true", help="Don't create missing teams, only update existing ones")
    parser.add_argument("--dry-run", action="store_true", help="Don't make actual changes, just show what would be done")
    
    args = parser.parse_args()
    
    mode = "DRY RUN" if args.dry_run else "PRODUCTION"
    create_mode = "WITHOUT CREATION" if args.no_create else "WITH CREATION"
    
    logger.info(f"🚀 Starting Team Name Mapping Fixer - {mode} MODE {create_mode}")
    
    try:
        teams_updated, teams_created, teams_skipped = fix_team_name_mappings(
            create_missing=not args.no_create,
            dry_run=args.dry_run
        )
        
        logger.info("=" * 50)
        logger.info("Team Name Mapping Fix Complete!")
        logger.info(f"Teams updated: {teams_updated}")
        logger.info(f"Teams created: {teams_created}")
        logger.info(f"Teams skipped: {teams_skipped}")
        logger.info("=" * 50)
        
        return 0
    except Exception as e:
        logger.error(f"Error fixing team name mappings: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
