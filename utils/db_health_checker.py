"""Database health check utilities"""
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)

def safe_execute_query(session, query_text: str):
    """Safely execute a query with proper text() wrapper"""
    try:
        return session.execute(text(query_text))
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise

def test_database_connection(db_manager):
    """Test database connection with proper error handling"""
    try:
        with db_manager.get_session() as session:
            result = safe_execute_query(session, "SELECT 1")
            return result.fetchone() is not None
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
