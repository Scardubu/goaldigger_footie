"""
Database Retry Manager for GoalDiggers Platform
Provides robust database connection management with retry mechanisms and initialization sequences.
"""
import logging
import time
import threading
from typing import Any, Callable, Optional, Dict, List
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, OperationalError, DisconnectionError
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker, scoped_session

from utils.comprehensive_error_handler import (
    DatabaseException,
    ConnectionError,
    QueryError,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    get_error_handler
)

logger = logging.getLogger(__name__)

class DatabaseRetryManager:
    """
    Manages database connections with retry mechanisms and proper initialization sequences.
    """

    def __init__(
        self,
        db_uri: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600
    ):
        """
        Initialize the database retry manager.

        Args:
            db_uri: Database URI
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            retry_backoff: Exponential backoff multiplier
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Pool timeout in seconds
            pool_recycle: Connection recycle time in seconds
        """
        self.db_uri = db_uri
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle

        self.engine = None
        self.session_factory = None
        self._connection_lock = threading.Lock()
        self._health_check_interval = 60  # Health check every 60 seconds
        self._last_health_check = 0
        self._is_healthy = False

        # Initialize error handler
        self.error_handler = get_error_handler()

    def initialize_database(self) -> bool:
        """
        Initialize the database with proper connection pooling and retry mechanisms.

        Returns:
            True if initialization successful, False otherwise
        """
        logger.info("Initializing database connection with retry mechanisms...")

        try:
            # Create engine with connection pooling
            self.engine = create_engine(
                self.db_uri,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                pool_pre_ping=True,  # Enable connection health checks
                echo=False  # Disable SQL echoing in production
            )

            # Create session factory
            self.session_factory = scoped_session(sessionmaker(bind=self.engine))

            # Test connection with retries
            if self._test_connection_with_retries():
                logger.info("Database initialization completed successfully")
                self._is_healthy = True
                return True
            else:
                logger.error("Database initialization failed - connection test unsuccessful")
                return False

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self._handle_initialization_error(e)
            return False

    def _test_connection_with_retries(self) -> bool:
        """Test database connection with retry mechanism."""
        for attempt in range(self.max_retries + 1):
            try:
                with self.engine.connect() as connection:
                    result = connection.execute(text("SELECT 1")).fetchone()
                    if result and result[0] == 1:
                        logger.info("Database connection test successful")
                        return True
            except Exception as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    logger.warning(f"Database connection test failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    logger.info(f"Retrying connection test in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"Database connection test failed after {self.max_retries + 1} attempts: {e}")
                    return False

        return False

    def _handle_initialization_error(self, error: Exception):
        """Handle database initialization errors."""
        context = ErrorContext(
            component="DatabaseRetryManager",
            operation="initialize_database",
            metadata={
                "db_uri": self.db_uri.replace("://", "://[REDACTED]@") if "://" in self.db_uri else self.db_uri,
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow
            }
        )

        db_error = DatabaseException(
            f"Database initialization failed: {str(error)}",
            severity=ErrorSeverity.CRITICAL,
            context=context,
            cause=error
        )

        if self.error_handler:
            self.error_handler.handle_error(db_error, context)

    @contextmanager
    def get_session_with_retry(self):
        """
        Get a database session with automatic retry on connection failures.

        Yields:
            SQLAlchemy session
        """
        session = None
        try:
            session = self.session_factory()
            yield session
            session.commit()
        except (OperationalError, DisconnectionError) as e:
            if session:
                session.rollback()
            logger.warning(f"Database operation failed due to connection error: {e}")

            # Try to recover the connection
            if self._recover_connection():
                # Retry the operation once after recovery
                try:
                    if session:
                        session.commit()  # Retry commit
                    else:
                        # If no session, caller will need to retry their operation
                        pass
                except Exception as retry_error:
                    logger.error(f"Retry after connection recovery failed: {retry_error}")
                    raise ConnectionError(
                        "Database operation failed even after connection recovery",
                        severity=ErrorSeverity.HIGH,
                        context=ErrorContext(component="DatabaseRetryManager", operation="get_session_with_retry"),
                        cause=retry_error
                    ) from retry_error
            else:
                raise ConnectionError(
                    "Database connection recovery failed",
                    severity=ErrorSeverity.HIGH,
                    context=ErrorContext(component="DatabaseRetryManager", operation="get_session_with_retry"),
                    cause=e
                ) from e
        except SQLAlchemyError as e:
            if session:
                session.rollback()
            raise QueryError(
                f"Database query error: {str(e)}",
                severity=ErrorSeverity.MEDIUM,
                context=ErrorContext(component="DatabaseRetryManager", operation="get_session_with_retry"),
                cause=e
            ) from e
        except Exception as e:
            if session:
                session.rollback()
            raise DatabaseException(
                f"Unexpected database error: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(component="DatabaseRetryManager", operation="get_session_with_retry"),
                cause=e
            ) from e
        finally:
            if session:
                session.close()

    def _recover_connection(self) -> bool:
        """
        Attempt to recover database connection.

        Returns:
            True if recovery successful, False otherwise
        """
        logger.info("Attempting database connection recovery...")

        try:
            # Dispose of the current engine
            if self.engine:
                self.engine.dispose()

            # Reinitialize the database
            return self.initialize_database()

        except Exception as e:
            logger.error(f"Database connection recovery failed: {e}")
            return False

    def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute a database operation with retry mechanism.

        Args:
            operation: Function to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation

        Raises:
            Last exception encountered if all retries fail
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Check connection health before executing
                if not self._check_connection_health():
                    if not self._recover_connection():
                        raise ConnectionError("Connection health check and recovery failed")

                return operation(*args, **kwargs)

            except (OperationalError, DisconnectionError) as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    logger.warning(f"Database operation failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    logger.info(f"Retrying in {delay:.1f} seconds...")

                    # Try to recover connection
                    if not self._recover_connection():
                        logger.warning("Connection recovery failed, continuing with retry")

                    time.sleep(delay)
                else:
                    logger.error(f"Database operation failed after {self.max_retries + 1} attempts: {e}")
                    break
            except Exception as e:
                # Don't retry for non-connection errors
                logger.error(f"Non-retryable database error: {e}")
                raise

        # If we get here, all retries failed
        if last_exception:
            raise ConnectionError(
                f"Database operation failed after {self.max_retries + 1} attempts",
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(component="DatabaseRetryManager", operation="execute_with_retry"),
                cause=last_exception
            ) from last_exception

    def _check_connection_health(self) -> bool:
        """
        Check the health of the database connection.

        Returns:
            True if connection is healthy, False otherwise
        """
        current_time = time.time()

        # Only check health if enough time has passed since last check
        if current_time - self._last_health_check < self._health_check_interval:
            return self._is_healthy

        self._last_health_check = current_time

        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1")).fetchone()
                self._is_healthy = True
                return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            self._is_healthy = False
            return False

    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get current connection pool statistics.

        Returns:
            Dictionary with connection statistics
        """
        if not self.engine:
            return {"status": "not_initialized"}

        try:
            return {
                "status": "healthy" if self._is_healthy else "unhealthy",
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_timeout": self.pool_timeout,
                "pool_recycle": self.pool_recycle,
                "connections_in_use": self.engine.pool.checkedout() if hasattr(self.engine.pool, 'checkedout') else 0,
                "connections_available": self.engine.pool.checkedin() if hasattr(self.engine.pool, 'checkedin') else 0,
                "overflow": self.engine.pool.overflow() if hasattr(self.engine.pool, 'overflow') else 0,
                "last_health_check": self._last_health_check
            }
        except Exception as e:
            logger.error(f"Failed to get connection stats: {e}")
            return {"status": "error", "error": str(e)}

    def shutdown(self):
        """Shutdown the database manager and clean up resources."""
        logger.info("Shutting down database retry manager...")

        try:
            if self.session_factory:
                self.session_factory.remove()

            if self.engine:
                self.engine.dispose()
                logger.info("Database engine disposed successfully")

        except Exception as e:
            logger.error(f"Error during database manager shutdown: {e}")

# Global instance
_database_retry_manager: Optional[DatabaseRetryManager] = None

def get_database_retry_manager() -> Optional[DatabaseRetryManager]:
    """Get the global database retry manager instance."""
    return _database_retry_manager

def set_database_retry_manager(manager: DatabaseRetryManager):
    """Set the global database retry manager instance."""
    global _database_retry_manager
    _database_retry_manager = manager

def initialize_database_retry_manager(db_uri: str, **kwargs) -> DatabaseRetryManager:
    """
    Initialize the global database retry manager.

    Args:
        db_uri: Database URI
        **kwargs: Additional arguments for DatabaseRetryManager

    Returns:
        Initialized DatabaseRetryManager instance
    """
    manager = DatabaseRetryManager(db_uri, **kwargs)

    if manager.initialize_database():
        set_database_retry_manager(manager)
        logger.info("Global database retry manager initialized successfully")
        return manager
    else:
        logger.error("Failed to initialize global database retry manager")
        return None