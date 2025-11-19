"""
Database configuration module for simplified access to database sessions.
Provides compatibility functions for accessing the database.
"""
import logging
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy.orm import Session

from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Global database manager instance
_db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get or create the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        # Initialize database manager with default SQLite database
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_root, 'football.db')
        db_uri = f"sqlite:///{db_path}"
        
        _db_manager = DatabaseManager(db_uri=db_uri)
        _db_manager.create_tables()
        logger.info(f"Initialized database manager with URI: {db_uri}")
    
    return _db_manager

@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Get a database session context manager.
    
    Yields:
        SQLAlchemy session
    """
    db_manager = get_database_manager()
    with db_manager.session_scope() as session:
        yield session

def get_session_factory():
    """Get the session factory from the database manager."""
    db_manager = get_database_manager()
    return db_manager.session_factory
