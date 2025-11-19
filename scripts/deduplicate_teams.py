#!/usr/bin/env python3
"""
Team Deduplication Script - GoalDiggers Platform

This script identifies and removes duplicate team entries from the database,
keeping the most complete/reliable record for each team.

Strategy:
1. Identify all duplicate teams (same name + same league)
2. For each duplicate group, select the "best" record to keep
3. Update all references (matches, stats) to point to the kept record
4. Delete the duplicate records
5. Add database constraints to prevent future duplicates
"""

import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List

# Add workspace root to Python path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from database.db_manager import DatabaseManager
from database.schema import Match, Team, TeamStats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TeamDeduplicator:
    """Handles team deduplication with data integrity preservation."""

    def __init__(self, dry_run: bool = True):
        """
        Initialize the deduplicator.
        
        Args:
            dry_run: If True, only report what would be done without making changes
        """
        self.db = DatabaseManager()
        self.dry_run = dry_run
        self.stats = {
            'duplicates_found': 0,
            'records_kept': 0,
            'records_deleted': 0,
            'matches_updated': 0,
            'stats_updated': 0
        }

    def _select_best_record(self, duplicates: List[Team]) -> Team:
        """
        Select the best record to keep from a list of duplicates.
        
        Priority:
        1. Record with most complete data (non-null fields)
        2. Record with shortest/cleanest ID (prefer league_code prefixed IDs)
        3. Record created first (if timestamps available)
        
        Args:
            duplicates: List of duplicate Team objects
            
        Returns:
            The Team object to keep
        """
        def score_record(team: Team) -> tuple:
            # Count non-null fields using fields that exist on the Team model
            completeness = sum([
                bool(getattr(team, 'short_name', None)),
                bool(getattr(team, 'tla', None)),
                bool(getattr(team, 'venue', None)),
                bool(getattr(team, 'venue_capacity', None)),
                bool(getattr(team, 'country', None)),
                bool(getattr(team, 'crest_url', None)),
                bool(getattr(team, 'api_id', None)),
                bool(getattr(team, 'country_code', None)),
                bool(getattr(team, 'primary_color', None)),
            ])
            
            # Prefer IDs that start with league code (e.g., "PL_ARS" over "ARS")
            id_starts_with_league = team.id.startswith(f"{team.league_id}_")
            
            # Prefer shorter, cleaner IDs
            id_length = len(team.id)
            
            # Return tuple for sorting (higher completeness = better, league-prefixed = better, shorter = better)
            return (completeness, id_starts_with_league, -id_length)
        
        # Sort by score (descending) and return the best
        best = max(duplicates, key=score_record)
        logger.debug(f"Selected {best.id} as best record for {best.name} (score: {score_record(best)})")
        return best

    def find_duplicates(self) -> Dict[tuple, List[str]]:
        """
        Find all duplicate team entries.
        
        Returns:
            Dictionary mapping (name, league_id) to list of duplicate Team IDs
        """
        with self.db.session_scope() as session:
            teams = session.query(Team).all()
            
            duplicates_map = defaultdict(list)
            for team in teams:
                key = (team.name.strip() if team.name else '', team.league_id)
                duplicates_map[key].append(team.id)  # Store just the ID
            
            # Filter to only groups with more than one entry
            duplicates = {k: v for k, v in duplicates_map.items() if len(v) > 1}
            
            self.stats['duplicates_found'] = len(duplicates)
            logger.info(f"Found {len(duplicates)} teams with duplicates")
            
            return duplicates

    def deduplicate_team_group(self, duplicate_ids: List[str], session) -> None:
        """
        Deduplicate a single group of duplicate teams.

        Args:
            duplicate_ids: List of duplicate Team IDs
            session: Database session to use
        """
        if len(duplicate_ids) <= 1:
            return

        # Load Team objects inside the active session to avoid detached instances
        duplicates = session.query(Team).filter(Team.id.in_(duplicate_ids)).all()
        if not duplicates:
            return

        # Select best record to keep
        keep_record = self._select_best_record(duplicates)
        delete_records = [t for t in duplicates if t.id != keep_record.id]

        logger.info(f"\n{'='*80}")
        logger.info(f"Deduplicating: {keep_record.name} in {keep_record.league_id}")
        logger.info(f"  Keeping: ID={keep_record.id}, TLA={keep_record.tla}")
        logger.info(f"  Deleting {len(delete_records)} duplicate(s):")
        for record in delete_records:
            logger.info(f"    - ID={record.id}, TLA={record.tla}")

        if self.dry_run:
            logger.info("  [DRY RUN] No changes made")
            return

        # Update all references to point to the kept record
        for delete_record in delete_records:
            # Update home team references in matches
            home_matches = session.query(Match).filter(
                Match.home_team_id == delete_record.id
            ).all()
            for match in home_matches:
                match.home_team_id = keep_record.id
                self.stats['matches_updated'] += 1

            # Update away team references in matches
            away_matches = session.query(Match).filter(
                Match.away_team_id == delete_record.id
            ).all()
            for match in away_matches:
                match.away_team_id = keep_record.id
                self.stats['matches_updated'] += 1

            # Update team stats references
            team_stats = session.query(TeamStats).filter(
                TeamStats.team_id == delete_record.id
            ).all()
            for stats in team_stats:
                # Check if stats already exist for keep_record in same season
                existing_stats = session.query(TeamStats).filter(
                    TeamStats.team_id == keep_record.id,
                    TeamStats.season == stats.season
                ).first()

                if existing_stats:
                    # Delete the duplicate stats
                    session.delete(stats)
                else:
                    # Transfer to keep_record
                    stats.team_id = keep_record.id
                    self.stats['stats_updated'] += 1

            # Delete the duplicate team record
            session.delete(delete_record)
            self.stats['records_deleted'] += 1
            logger.info(f"    ✓ Deleted {delete_record.id}")

        self.stats['records_kept'] += 1
        # Commit is handled by the caller's session scope
        logger.info(f"  ✓ Kept {keep_record.id}")

    def run_deduplication(self) -> Dict:
        """
        Run the full deduplication process.
        
        Returns:
            Dictionary with deduplication statistics
        """
        logger.info("=" * 80)
        logger.info("TEAM DEDUPLICATION PROCESS")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE - MAKING CHANGES'}")
        logger.info("=" * 80)
        
        # Find all duplicates
        duplicates = self.find_duplicates()
        
        if not duplicates:
            logger.info("\n✅ No duplicates found! Database is clean.")
            return self.stats
        
        # Process each duplicate group
        with self.db.session_scope() as session:
            for (name, league_id), duplicate_list in sorted(duplicates.items()):
                try:
                    self.deduplicate_team_group(duplicate_list, session)
                except Exception as e:
                    logger.error(f"Error deduplicating {name} in {league_id}: {e}")
                    if not self.dry_run:
                        session.rollback()
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("DEDUPLICATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Duplicate groups found: {self.stats['duplicates_found']}")
        logger.info(f"Records kept: {self.stats['records_kept']}")
        logger.info(f"Records deleted: {self.stats['records_deleted']}")
        logger.info(f"Match references updated: {self.stats['matches_updated']}")
        logger.info(f"Team stats updated: {self.stats['stats_updated']}")
        
        if self.dry_run:
            logger.info("\n⚠️  This was a DRY RUN - no actual changes were made")
            logger.info("   Run with --apply to make actual changes")
        else:
            logger.info("\n✅ Deduplication complete!")
        
        logger.info("=" * 80)
        
        return self.stats


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deduplicate team entries in the database")
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Actually apply changes (default is dry-run mode)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run deduplication
    deduplicator = TeamDeduplicator(dry_run=not args.apply)
    stats = deduplicator.run_deduplication()
    
    # Exit with error code if in dry-run and duplicates were found
    if deduplicator.dry_run and stats['duplicates_found'] > 0:
        exit(1)
    
    exit(0)


if __name__ == "__main__":
    main()
