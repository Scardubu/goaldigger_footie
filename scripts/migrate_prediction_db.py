"""
Database migration script to add outcome tracking fields.

Adds home_score, away_score, and outcome_updated_at columns to predictions table.
"""

import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_prediction_history_db(db_path: str = "data/prediction_history.db"):
    """Add outcome tracking columns to predictions table."""
    
    db_file = Path(db_path)
    if not db_file.exists():
        logger.warning(f"Database {db_path} does not exist yet - will be created with new schema")
        return
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Check existing columns
        cursor.execute("PRAGMA table_info(predictions)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        logger.info(f"Existing columns: {existing_columns}")
        
        # Add new columns if they don't exist
        new_columns = [
            ("home_score", "INTEGER"),
            ("away_score", "INTEGER"),
            ("outcome_updated_at", "TEXT")
        ]
        
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                logger.info(f"Adding column: {col_name} ({col_type})")
                cursor.execute(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}")
            else:
                logger.info(f"Column {col_name} already exists")
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Migration complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("Prediction History Database Migration")
    print("=" * 80)
    print("\nAdding outcome tracking fields...")
    
    success = migrate_prediction_history_db()
    
    if success:
        print("\n✅ Migration completed successfully")
        print("\nNew fields added:")
        print("  - home_score: INTEGER (home team final score)")
        print("  - away_score: INTEGER (away team final score)")  
        print("  - outcome_updated_at: TEXT (timestamp when outcome was recorded)")
    else:
        print("\n❌ Migration failed - check logs for details")
