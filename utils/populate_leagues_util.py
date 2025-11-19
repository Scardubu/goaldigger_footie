import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker

try:
    if 'PYTEST_CURRENT_TEST' not in os.environ:
        from utils.logging_config import configure_logging
        configure_logging(level=os.getenv('LOG_LEVEL', 'INFO'))
except Exception:
    pass
logger = logging.getLogger(__name__)

# Define project root and paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'football.db')
DB_URI = f'sqlite:///{DB_PATH}'
LEAGUE_DATA_CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'reference', 'league_data.csv')

# Define the League model (subset of the main schema.py for this utility)
Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class League(Base):
    __tablename__ = 'leagues'
    id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    country = Column(String(50), nullable=False)
    tier = Column(Integer, default=1)
    # Add other fields if they are non-nullable and you want to provide defaults
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

def populate_leagues():
    logger.info(f"Starting league population utility.")
    logger.info(f"Database URI: {DB_URI}")
    logger.info(f"League data CSV: {LEAGUE_DATA_CSV_PATH}")

    if not os.path.exists(LEAGUE_DATA_CSV_PATH):
        logger.error(f"League data CSV file not found at {LEAGUE_DATA_CSV_PATH}. Cannot populate leagues.")
        logger.error("Please ensure 'data/reference/league_data.csv' exists. You might need to run the main application once to generate it if it's missing.")
        return

    try:
        league_df = pd.read_csv(LEAGUE_DATA_CSV_PATH)
        if league_df.empty:
            logger.warning(f"The file {LEAGUE_DATA_CSV_PATH} is empty. No leagues to populate.")
            return
    except pd.errors.EmptyDataError:
        logger.warning(f"The file {LEAGUE_DATA_CSV_PATH} is empty (pandas EmptyDataError). No leagues to populate.")
        return
    except Exception as e:
        logger.error(f"Error reading {LEAGUE_DATA_CSV_PATH}: {e}")
        return

    # Ensure required columns are present
    required_cols = ['league_id', 'league_name', 'country', 'tier']
    missing_cols = [col for col in required_cols if col not in league_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in {LEAGUE_DATA_CSV_PATH}: {missing_cols}. Required: {required_cols}")
        return

    engine = create_engine(DB_URI)
    Base.metadata.create_all(engine) # Ensure leagues table exists
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()

    added_count = 0
    skipped_count = 0

    try:
        for _, row in league_df.iterrows():
            league_id = str(row['league_id'])
            league_name = str(row['league_name'])
            country = str(row['country'])
            tier = int(row['tier']) if pd.notna(row['tier']) else 1 # Default tier to 1 if NaN

            # Check if league already exists by id or name
            exists = session.query(League).filter((League.id == league_id) | (League.name == league_name)).first()
            if exists:
                logger.info(f"League '{league_name}' (ID: {league_id}) already exists. Skipping.")
                skipped_count += 1
                continue
            
            new_league = League(
                id=league_id,
                name=league_name,
                country=country,
                tier=tier
            )
            session.add(new_league)
            added_count += 1
            logger.info(f"Adding new league: ID={league_id}, Name='{league_name}', Country='{country}', Tier={tier}")

        session.commit()
        logger.info(f"Successfully committed changes to the database.")

    except IntegrityError as e:
        session.rollback()
        logger.error(f"Database integrity error: {e}. Changes rolled back.")
        logger.error("This might be due to a unique constraint violation (e.g., league name already exists if 'unique=True' was missed in this script's model but present in the main schema).")
    except Exception as e:
        session.rollback()
        logger.error(f"An error occurred during database population: {e}. Changes rolled back.")
    finally:
        session.close()
        logger.info("Database session closed.")

    logger.info(f"League population process finished. Added: {added_count}, Skipped: {skipped_count}.")

if __name__ == "__main__":
    # Add project root to Python path to allow imports from database.schema if needed for full model
    # For this standalone script, we define a minimal League model.
    # If you were to import from database.schema, ensure PROJECT_ROOT is in sys.path
    # sys.path.insert(0, PROJECT_ROOT)
    populate_leagues()
