#!/usr/bin/env python3
"""Fix team ID mapping inconsistencies in GoalDiggers database"""

import logging
import sqlite3
import sys

from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_team_id_mapping():
    """Fix the team ID mapping between matches and teams tables"""
    logger.info("🔧 Starting team ID mapping fix...")
    
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        
        # Create mapping between placeholder IDs and real team names
        team_mapping = {
            # Premier League teams
            'PL_01': ('57', 'Arsenal FC'),
            'PL_02': ('61', 'Chelsea FC'), 
            'PL_03': ('64', 'Liverpool FC'),
            'PL_04': ('65', 'Manchester City FC'),
            'PL_05': ('66', 'Manchester United FC'),
            'PL_06': ('73', 'Tottenham Hotspur FC'),
            'PL_07': ('67', 'Newcastle United FC'),
            'PL_08': ('63', 'Fulham FC'),
            'PL_09': ('58', 'Aston Villa FC'),
            'PL_10': ('62', 'Everton FC'),
            # Add more mappings as needed
        }
        
        with db.session_scope() as session:
            # Step 1: Update matches table with real team IDs
            logger.info("📊 Updating matches with real team IDs...")
            
            updates_made = 0
            for placeholder_id, (real_id, team_name) in team_mapping.items():
                # Update home_team_id
                result = session.execute(text(
                    "UPDATE matches SET home_team_id = :real_id WHERE home_team_id = :placeholder_id"
                ), {"real_id": real_id, "placeholder_id": placeholder_id})
                updates_made += result.rowcount
                
                # Update away_team_id  
                result = session.execute(text(
                    "UPDATE matches SET away_team_id = :real_id WHERE away_team_id = :placeholder_id"
                ), {"real_id": real_id, "placeholder_id": placeholder_id})
                updates_made += result.rowcount
                
                logger.info(f"✅ Mapped {placeholder_id} -> {real_id} ({team_name})")
            
            session.commit()
            logger.info(f"🎯 Updated {updates_made} team ID references")
            
            # Step 2: Verify the fix
            logger.info("🔍 Verifying team ID mapping fix...")
            verify_sql = text("""
                SELECT 
                    m.id, m.home_team_id, m.away_team_id,
                    ht.name as home_team_name, 
                    at.name as away_team_name
                FROM matches m
                LEFT JOIN teams ht ON m.home_team_id = ht.id
                LEFT JOIN teams at ON m.away_team_id = at.id
                WHERE ht.name IS NOT NULL AND at.name IS NOT NULL
                LIMIT 5
            """)
            result = session.execute(verify_sql)
            verified_matches = result.fetchall()
            logger.info(f"✅ Successfully verified {len(verified_matches)} matches with proper team names")
            for match in verified_matches:
                logger.info(f"  • {match.home_team_name} vs {match.away_team_name}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Error fixing team ID mapping: {e}")
        return False

if __name__ == "__main__":
    success = fix_team_id_mapping()
    sys.exit(0 if success else 1)
