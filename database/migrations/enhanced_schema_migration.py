"""
Enhanced Database Schema Migration for Additional Data Sources
"""
import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedSchemaMigration:
    """Handles database schema migrations for enhanced data sources."""
    
    def __init__(self, db_path: str = "data/football.db"):
        self.db_path = db_path
        self.migration_version = "2.0.0"
        
    def execute_migration(self) -> bool:
        """Execute the complete schema migration."""
        try:
            # Ensure database directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check current schema version
                current_version = self._get_schema_version(cursor)
                logger.info(f"Current schema version: {current_version}")
                
                if current_version >= self.migration_version:
                    logger.info("Schema already up to date")
                    return True
                
                # Execute migrations in order
                migrations = [
                    self._create_players_table,
                    self._create_player_stats_table,
                    self._create_match_weather_table,
                    self._create_referees_table,
                    self._create_referee_assignments_table,
                    self._create_historical_performance_table,
                    self._create_market_sentiment_table,
                    self._create_injury_reports_table,
                    self._create_suspension_records_table,
                    self._create_venue_details_table,
                    self._create_enhanced_indexes,
                    self._update_schema_version
                ]
                
                for migration in migrations:
                    try:
                        migration(cursor)
                        conn.commit()
                        logger.info(f"Executed migration: {migration.__name__}")
                    except Exception as e:
                        logger.error(f"Migration {migration.__name__} failed: {e}")
                        conn.rollback()
                        raise
                
                logger.info("Schema migration completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Schema migration failed: {e}")
            return False
    
    def _get_schema_version(self, cursor: sqlite3.Cursor) -> str:
        """Get current schema version."""
        try:
            cursor.execute("SELECT version FROM schema_version ORDER BY created_at DESC LIMIT 1")
            result = cursor.fetchone()
            return result[0] if result else "1.0.0"
        except sqlite3.OperationalError:
            # Table doesn't exist, create it
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            return "1.0.0"
    
    def _create_players_table(self, cursor: sqlite3.Cursor):
        """Create players table for detailed player information."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                team_id VARCHAR(20),
                position VARCHAR(20),
                market_value DECIMAL(12,2),
                nationality VARCHAR(50),
                age INTEGER,
                height_cm INTEGER,
                weight_kg INTEGER,
                preferred_foot VARCHAR(10),
                contract_expires DATE,
                injury_status VARCHAR(50) DEFAULT 'fit',
                suspension_status VARCHAR(50) DEFAULT 'available',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES teams(id)
            )
        """)
    
    def _create_player_stats_table(self, cursor: sqlite3.Cursor):
        """Create player statistics table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id VARCHAR(50) NOT NULL,
                match_id VARCHAR(50),
                season VARCHAR(20),
                minutes_played INTEGER DEFAULT 0,
                goals INTEGER DEFAULT 0,
                assists INTEGER DEFAULT 0,
                shots INTEGER DEFAULT 0,
                shots_on_target INTEGER DEFAULT 0,
                passes_completed INTEGER DEFAULT 0,
                passes_attempted INTEGER DEFAULT 0,
                tackles INTEGER DEFAULT 0,
                interceptions INTEGER DEFAULT 0,
                clearances INTEGER DEFAULT 0,
                yellow_cards INTEGER DEFAULT 0,
                red_cards INTEGER DEFAULT 0,
                rating DECIMAL(3,1),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(id),
                FOREIGN KEY (match_id) REFERENCES matches(id)
            )
        """)
    
    def _create_match_weather_table(self, cursor: sqlite3.Cursor):
        """Create weather data table for matches."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_weather (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id VARCHAR(50) NOT NULL,
                temperature_celsius DECIMAL(4,1),
                humidity_percent INTEGER,
                wind_speed_kmh DECIMAL(4,1),
                wind_direction VARCHAR(10),
                precipitation_mm DECIMAL(4,1),
                visibility_km INTEGER,
                weather_condition VARCHAR(50),
                pressure_hpa DECIMAL(6,1),
                uv_index INTEGER,
                forecast_accuracy DECIMAL(3,2),
                data_source VARCHAR(50),
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches(id)
            )
        """)
    
    def _create_referees_table(self, cursor: sqlite3.Cursor):
        """Create referees table for referee performance tracking."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS referees (
                id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                nationality VARCHAR(50),
                birth_date DATE,
                experience_years INTEGER,
                fifa_badge BOOLEAN DEFAULT FALSE,
                cards_per_game DECIMAL(3,1),
                penalties_per_game DECIMAL(3,2),
                var_usage_rate DECIMAL(3,2),
                home_bias_factor DECIMAL(3,2),
                big_game_experience INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_referee_assignments_table(self, cursor: sqlite3.Cursor):
        """Create referee assignments table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS referee_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id VARCHAR(50) NOT NULL,
                referee_id VARCHAR(50) NOT NULL,
                role VARCHAR(20) DEFAULT 'main', -- main, assistant1, assistant2, fourth, var
                performance_rating DECIMAL(3,1),
                cards_issued INTEGER DEFAULT 0,
                penalties_awarded INTEGER DEFAULT 0,
                var_decisions INTEGER DEFAULT 0,
                controversial_decisions INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches(id),
                FOREIGN KEY (referee_id) REFERENCES referees(id)
            )
        """)
    
    def _create_historical_performance_table(self, cursor: sqlite3.Cursor):
        """Create historical performance tracking table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id VARCHAR(20) NOT NULL,
                season VARCHAR(20) NOT NULL,
                league_id VARCHAR(20),
                elo_rating DECIMAL(6,2),
                attack_strength DECIMAL(4,2),
                defense_strength DECIMAL(4,2),
                home_advantage DECIMAL(3,2),
                away_performance DECIMAL(3,2),
                form_index DECIMAL(4,2),
                momentum_score DECIMAL(4,2),
                pressure_handling DECIMAL(3,2),
                big_game_performance DECIMAL(3,2),
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (team_id) REFERENCES teams(id),
                FOREIGN KEY (league_id) REFERENCES leagues(id)
            )
        """)
    
    def _create_market_sentiment_table(self, cursor: sqlite3.Cursor):
        """Create market sentiment tracking table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id VARCHAR(50) NOT NULL,
                social_sentiment_home DECIMAL(3,2),
                social_sentiment_away DECIMAL(3,2),
                media_coverage_intensity INTEGER,
                betting_volume_indicator DECIMAL(4,2),
                public_betting_percentage_home DECIMAL(3,2),
                public_betting_percentage_away DECIMAL(3,2),
                sharp_money_percentage_home DECIMAL(3,2),
                sharp_money_percentage_away DECIMAL(3,2),
                line_movement_significance DECIMAL(3,2),
                market_efficiency_score DECIMAL(3,2),
                sentiment_source VARCHAR(50),
                confidence_score DECIMAL(3,2),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches(id)
            )
        """)
    
    def _create_injury_reports_table(self, cursor: sqlite3.Cursor):
        """Create injury reports table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS injury_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id VARCHAR(50) NOT NULL,
                injury_type VARCHAR(100),
                injury_severity VARCHAR(20), -- minor, moderate, major, career_threatening
                injury_date DATE,
                expected_return_date DATE,
                actual_return_date DATE,
                matches_missed INTEGER DEFAULT 0,
                recovery_progress DECIMAL(3,2), -- 0.0 to 1.0
                medical_notes TEXT,
                data_source VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(id)
            )
        """)
    
    def _create_suspension_records_table(self, cursor: sqlite3.Cursor):
        """Create suspension records table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS suspension_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id VARCHAR(50) NOT NULL,
                suspension_type VARCHAR(50), -- yellow_cards, red_card, disciplinary, other
                suspension_reason TEXT,
                suspension_date DATE,
                matches_suspended INTEGER,
                matches_served INTEGER DEFAULT 0,
                appeal_status VARCHAR(20), -- none, pending, successful, rejected
                reinstatement_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players(id)
            )
        """)
    
    def _create_venue_details_table(self, cursor: sqlite3.Cursor):
        """Create detailed venue information table."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS venue_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue_name VARCHAR(100) NOT NULL,
                city VARCHAR(100),
                country VARCHAR(100),
                capacity INTEGER,
                pitch_length_m INTEGER,
                pitch_width_m INTEGER,
                surface_type VARCHAR(50), -- grass, artificial, hybrid
                altitude_m INTEGER,
                latitude DECIMAL(10,8),
                longitude DECIMAL(11,8),
                roof_type VARCHAR(20), -- open, retractable, closed
                atmosphere_rating DECIMAL(3,1),
                home_advantage_factor DECIMAL(3,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_enhanced_indexes(self, cursor: sqlite3.Cursor):
        """Create performance indexes for new tables."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_players_team_position ON players(team_id, position)",
            "CREATE INDEX IF NOT EXISTS idx_players_nationality ON players(nationality)",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_player_match ON player_stats(player_id, match_id)",
            "CREATE INDEX IF NOT EXISTS idx_player_stats_season ON player_stats(season)",
            "CREATE INDEX IF NOT EXISTS idx_weather_match_date ON match_weather(match_id, recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_referees_nationality ON referees(nationality)",
            "CREATE INDEX IF NOT EXISTS idx_referee_assignments_match ON referee_assignments(match_id)",
            "CREATE INDEX IF NOT EXISTS idx_historical_team_season ON historical_performance(team_id, season)",
            "CREATE INDEX IF NOT EXISTS idx_sentiment_match_time ON market_sentiment(match_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_injuries_player_date ON injury_reports(player_id, injury_date)",
            "CREATE INDEX IF NOT EXISTS idx_suspensions_player_date ON suspension_records(player_id, suspension_date)",
            "CREATE INDEX IF NOT EXISTS idx_venue_details_name ON venue_details(venue_name)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def _update_schema_version(self, cursor: sqlite3.Cursor):
        """Update schema version."""
        cursor.execute("""
            INSERT INTO schema_version (version) VALUES (?)
        """, (self.migration_version,))

def run_migration():
    """Run the enhanced schema migration."""
    migration = EnhancedSchemaMigration()
    success = migration.execute_migration()
    
    if success:
        logger.info("Enhanced schema migration completed successfully")
    else:
        logger.error("Enhanced schema migration failed")
    
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migration()
