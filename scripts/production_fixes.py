#!/usr/bin/env python3
"""
GoalDiggers Production Readiness Fixes
Execute: python scripts/production_fixes.py
"""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_critical_fixes():
    """Apply critical fixes for production readiness"""
    logger.info("🚀 Starting GoalDiggers production fixes...")
    
    try:
        # Import database components
        from database.db_manager import DatabaseManager

        # Initialize database manager (following pattern from reset_database.py)
        db = DatabaseManager()
        
        # Apply database indexes
        logger.info("📊 Creating database performance indexes...")
        
        with db.get_session() as session:
            # Create indexes for better query performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)",
                "CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_id)", 
                "CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id)",
                "CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status)",
                "CREATE INDEX IF NOT EXISTS idx_teams_league ON teams(league_id)"
            ]
            
            from sqlalchemy import text
            for index_sql in indexes:
                try:
                    session.execute(text(index_sql))
                    logger.info(f"✅ Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
                except Exception as e:
                    logger.warning(f"⚠️  Index might already exist: {e}")
            session.commit()
        
        logger.info("🎯 Verifying database schema...")
        
        # Verify table structure
        from sqlalchemy import text
        with db.get_session() as session:
            result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]
            logger.info(f"📋 Found tables: {', '.join(tables)}")
            if 'matches' in tables:
                result = session.execute(text("PRAGMA table_info(matches)"))
                columns = [row[1] for row in result.fetchall()]
                logger.info(f"🏗️  Match table columns: {', '.join(columns)}")
        
        print("✅ All critical fixes applied successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("💡 Make sure you're running from the project root directory")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error applying fixes: {e}")
        logger.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    success = apply_critical_fixes()
    sys.exit(0 if success else 1)
