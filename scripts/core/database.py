import os  # Added for environment variable access

from dotenv import load_dotenv
from sqlalchemy import (  # Added func for server_default
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables from .env file
load_dotenv()

Base = declarative_base()

class MatchPrediction(Base):
    __tablename__ = 'match_predictions'

    id = Column(Integer, primary_key=True)
    match_id = Column(String(50), unique=True, nullable=False) # Added nullable=False
    features = Column(JSON) # Store input features
    raw_prediction = Column(JSON) # Store model output probabilities
    analysis_text = Column(String, nullable=True) # Store AI analysis text, allow null
    # Use func.now() for database-generated timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now()) # Correct onupdate usage

def init_db(connection_string: str = None):
    """
    Initializes the database connection and creates tables if they don't exist.

    Args:
        connection_string (str, optional): Database connection URL.
                                           Defaults to POSTGRES_URL from environment variables.

    Returns:
        sessionmaker: A configured sessionmaker instance.
    """
    if connection_string is None:
        connection_string = os.getenv("POSTGRES_URL")
        if not connection_string:
            raise ValueError("Database connection string not provided and POSTGRES_URL environment variable is not set.")

    print(f"Initializing database connection to: {connection_string.split('@')[-1]}") # Avoid logging credentials
    engine = create_engine(connection_string)

    print("Creating database tables if they don't exist...")
    Base.metadata.create_all(engine)
    print("Tables created (or already exist).")

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal

# Allow running this script directly to initialize the DB
if __name__ == "__main__":
    print("Running database initialization...")
    try:
        # Initialize using environment variable
        init_db()
        print("Database initialization complete.")
    except Exception as e:
        print(f"Database initialization failed: {e}")
