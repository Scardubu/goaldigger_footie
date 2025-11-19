"""
Database manager for the football betting insights platform.
Handles database connections, session management, and CRUD operations.
"""
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Generator, List, Optional, Tuple, Type, Union
from urllib.parse import quote_plus

from sqlalchemy import and_, create_engine, func, inspect, not_, or_, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import DisconnectionError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from sqlalchemy.pool import QueuePool

# Import custom PostgreSQL configuration validator
try:
    from database.pg_config_validator import validate_postgres_config
except ImportError:
    # Define a simple fallback if the module is not available
    def validate_postgres_config():
        return True, {"config_valid": True}

from database.schema import (
    Base,
    League,
    Match,
    MatchStats,
    Odds,
    Prediction,
    Team,
    TeamStats,
    ValueBet,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler to output logs to a file
log_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db_manager.log')
file_handler = logging.FileHandler(log_file_path, mode='a') # Append mode
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger if it hasn't been added already
# This check prevents adding multiple handlers if the module is reloaded or logger is accessed multiple times
if not any(isinstance(h, logging.FileHandler) and h.baseFilename == file_handler.baseFilename for h in logger.handlers):
    logger.addHandler(file_handler)


def _mask_db_uri(uri: str) -> str:
    """Mask sensitive components of a database URI for logging."""
    try:
        parsed = make_url(uri)
        if getattr(parsed, "password", None):
            parsed = parsed.set(password="***")
        return str(parsed)
    except Exception:  # pragma: no cover - defensive logging helper
        if "@" in uri:
            user_part, host_part = uri.split("@", 1)
            if ":" in user_part:
                user_part = user_part.rsplit(":", 1)[0] + ":***"
            return f"{user_part}@{host_part}"
        return uri


def _connect_args_for_postgres_from_env() -> Dict[str, Any]:
    """Build optional PostgreSQL connection arguments from environment variables."""
    connect_args: Dict[str, Any] = {}
    env_to_arg = {
        "POSTGRES_SSLMODE": "sslmode",
        "DATABASE_SSLMODE": "sslmode",
        "POSTGRES_SSLROOTCERT": "sslrootcert",
        "POSTGRES_SSLCERT": "sslcert",
        "POSTGRES_SSLKEY": "sslkey",
        "POSTGRES_TARGET_SESSION_ATTRS": "target_session_attrs",
        "POSTGRES_APPLICATION_NAME": "application_name",
        "POSTGRES_OPTIONS": "options",
    }
    for env_name, arg_name in env_to_arg.items():
        value = os.getenv(env_name)
        if value:
            connect_args[arg_name] = value
    return connect_args


def _build_postgres_uri_from_env() -> Tuple[Optional[str], Dict[str, Any]]:
    """Construct a PostgreSQL URI from discrete POSTGRES_* environment variables."""
    host = os.getenv("POSTGRES_HOST") or os.getenv("PGHOST")
    db_name = (
        os.getenv("POSTGRES_DB")
        or os.getenv("POSTGRES_DATABASE")
        or os.getenv("PGDATABASE")
    )
    user = os.getenv("POSTGRES_USER") or os.getenv("PGUSER")
    password = os.getenv("POSTGRES_PASSWORD") or os.getenv("PGPASSWORD")
    port = os.getenv("POSTGRES_PORT") or os.getenv("PGPORT") or "5432"

    if not (host and db_name and user):
        return None, {}

    safe_user = quote_plus(user)
    if password is not None and password != "":
        safe_password = quote_plus(password)
        auth_part = f"{safe_user}:{safe_password}"
    else:
        auth_part = safe_user

    # Allow host to already contain a port specification
    if host.count(":") == 1 and host.split(":")[1].isdigit():
        host_part = host
    else:
        host_part = f"{host}:{port}" if port else host

    uri = f"postgresql+psycopg2://{auth_part}@{host_part}/{db_name}"
    return uri, _connect_args_for_postgres_from_env()


def _merge_connect_args(uri: str, connect_args: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure PostgreSQL-specific connect args are present when needed."""
    if uri.startswith("postgresql"):
        merged = dict(_connect_args_for_postgres_from_env())
        merged.update(connect_args)
        return merged
    return connect_args


def _sanitize_failure_reason(failure: Exception) -> str:
    """Convert exception information into a concise single-line message."""
    text = " ".join(str(failure).split())
    return text if len(text) <= 300 else f"{text[:297]}..."

class DatabaseManager:
    """
    Database manager for handling database connections and CRUD operations.
    Uses SQLAlchemy ORM for database operations.
    
    Implements singleton pattern to avoid redundant database connections.
    """

    _global_fallback_active: bool = False
    _global_fallback_uri: Optional[str] = None
    _global_fallback_reason: Optional[str] = None
    _global_fallback_lock: Lock = Lock()
    
    # Singleton pattern - cache instances by db_uri
    _instances: Dict[str, 'DatabaseManager'] = {}
    _instance_lock: Lock = Lock()
    
    def __new__(cls, db_uri: str = None):
        """
        Singleton implementation: return existing instance if available.
        
        Args:
            db_uri: Database URI (e.g., 'sqlite:///football.db')
            
        Returns:
            Cached or new DatabaseManager instance
        """
        # Determine the effective URI for cache key
        effective_uri = db_uri or os.getenv("DATABASE_URI", os.getenv("DATABASE_URL", ""))
        
        # Use global fallback URI if active
        if not effective_uri and cls._global_fallback_active and cls._global_fallback_uri:
            effective_uri = cls._global_fallback_uri
        
        # Default cache key for empty URI
        cache_key = effective_uri or "default"
        
        with cls._instance_lock:
            # Return existing instance if available
            if cache_key in cls._instances:
                logger.debug(f"Reusing cached DatabaseManager instance for: {_mask_db_uri(cache_key)}")
                return cls._instances[cache_key]
            
            # Create new instance
            logger.debug(f"Creating new DatabaseManager instance for: {_mask_db_uri(cache_key)}")
            instance = super().__new__(cls)
            cls._instances[cache_key] = instance
            return instance
    
    def __init__(self, db_uri: str = None):
        """
        Initialize the database manager.
        
        Args:
            db_uri: Database URI (e.g., 'sqlite:///football.db')
        """
        # Skip initialization if already initialized (singleton pattern)
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # If no URI provided, try to create a default one
        connect_args: Dict[str, Any] = {}

        fallback_pref = os.getenv("DATABASE_SQLITE_FALLBACK", "true").lower()
        self._allow_sqlite_fallback = fallback_pref not in {"false", "0", "no", "off"}
        self.using_sqlite_fallback = False
        self._primary_db_uri: Optional[str] = None
        self._engine_kwargs_base: Dict[str, Any] = {}
        self.fallback_reason: Optional[str] = None
        self._pg_config_valid = False
        self._initialized = False  # Will be set to True at the end

        primary_uri = db_uri

        if not primary_uri:
            # Common environment variable names in priority order
            for candidate in ("DATABASE_URI", "DATABASE_URL", "POSTGRES_URL"):
                env_uri = os.getenv(candidate)
                if env_uri:
                    primary_uri = env_uri
                    break

        # Validate PostgreSQL configuration if no URI is provided directly
        if not primary_uri:
            is_pg_valid, pg_validation = validate_postgres_config()
            self._pg_config_valid = is_pg_valid
            
            if is_pg_valid:
                env_uri, env_connect_args = _build_postgres_uri_from_env()
                if env_uri:
                    primary_uri = env_uri
                    connect_args = env_connect_args
                    logger.info("Using PostgreSQL configuration from environment variables")
                else:
                    logger.warning("PostgreSQL configuration is valid but URI could not be constructed")
            else:
                logger.warning("PostgreSQL configuration validation failed, missing required variables")
                if pg_validation.get("missing_required"):
                    logger.warning(f"Missing required variables: {pg_validation['missing_required']}")

        cached_fallback_active = (
            self._allow_sqlite_fallback
            and DatabaseManager._global_fallback_active
            and DatabaseManager._global_fallback_uri
        )

        if cached_fallback_active:
            db_uri = DatabaseManager._global_fallback_uri
            connect_args = {}
            self.using_sqlite_fallback = True
            logger.info(
                "Using cached SQLite fallback URI %s (primary DB previously unreachable)",
                _mask_db_uri(db_uri),
            )
        else:
            db_uri = primary_uri

        if not db_uri:
            db_uri = self._prepare_sqlite_uri()

        if not db_uri:
            db_uri = "sqlite:///:memory:"
            logger.warning("No valid database URI could be determined. Using in-memory SQLite database.")

        if not connect_args:
            connect_args = _merge_connect_args(primary_uri or db_uri, connect_args)

        self.db_uri = db_uri
        self._primary_db_uri = primary_uri
        self._connect_args = connect_args
        self.pool_size = 10  # Default pool size
        self.max_overflow = 5  # Default max overflow
        self.pool_timeout = 30  # Default timeout in seconds
        self.pool_recycle = 3600  # Recycle connections every hour
        
        # Ensure the directory exists for SQLite database if it's a file-based SQLite URI
        if self.db_uri.startswith('sqlite:///') and not self.db_uri.endswith(':memory:'):
            path_part = self.db_uri.replace('sqlite:///', '', 1)
            # Adjust for Windows absolute paths starting with a drive letter after 'sqlite:///'
            if os.name == 'nt' and ':' in path_part.split('/')[0]: # e.g. C:/... or /C:/...
                if path_part.startswith('/') and len(path_part) > 1 and path_part[1].isalpha() and path_part[2] == ':':
                     path_part = path_part[1:] # Converts '/C:/path' to 'C:/path'
        
            db_file_path = os.path.abspath(path_part)
            db_dir = os.path.dirname(db_file_path)
            
            try:
                if db_dir: # Ensure dirname is not empty
                    os.makedirs(db_dir, exist_ok=True)
                    logger.info(f"Ensured directory exists for database: {db_dir}")
            except Exception as e: # pylint: disable=broad-except
                logger.error(f"Failed to create directory {db_dir} for database: {e}")
                # Don't pass through - we'll let create_engine handle it
        
        # Initialize engine and session factory with appropriate pool settings
        try:
            # Use a connection pool with reasonable defaults, allowing env overrides
            pool_size = int(os.getenv('DATABASE_POOL_SIZE', os.getenv('DB_POOL_SIZE', '5')))
            max_overflow = int(os.getenv('DATABASE_MAX_OVERFLOW', os.getenv('DB_MAX_OVERFLOW', '10')))
            timeout = int(os.getenv('DATABASE_POOL_TIMEOUT', os.getenv('DB_POOL_TIMEOUT', '30')))
            recycle = int(os.getenv('DATABASE_POOL_RECYCLE', os.getenv('DB_POOL_RECYCLE', '3600')))

            # Store for later reference
            self.pool_size = pool_size
            self.max_overflow = max_overflow
            self.pool_timeout = timeout
            self.pool_recycle = recycle

            self._engine_kwargs_base = {
                'poolclass': QueuePool,
                'pool_size': pool_size,
                'max_overflow': max_overflow,
                'pool_timeout': timeout,
                'pool_recycle': recycle,
                'pool_pre_ping': True,
            }

            engine_kwargs: Dict[str, Any] = dict(self._engine_kwargs_base)
            if self._connect_args:
                engine_kwargs['connect_args'] = self._connect_args
            if self.db_uri.startswith('sqlite'):
                engine_kwargs.setdefault('connect_args', {})
                engine_kwargs['connect_args'].setdefault('check_same_thread', False)

            self.engine = create_engine(self.db_uri, **engine_kwargs)
            self.session_factory = scoped_session(sessionmaker(bind=self.engine))
            logger.info(f"DatabaseManager initialized with URI: {_mask_db_uri(self.db_uri)}")
            self._verify_initial_connection()
            self._initialized = True  # Mark as successfully initialized
        except Exception as e:
            # Last resort fallback to in-memory SQLite if even the engine creation fails
            logger.error(f"Failed to create database engine with URI {self.db_uri}: {e}")
            logger.warning("Falling back to in-memory SQLite database")
            self.db_uri = "sqlite:///:memory:"
            self.engine = create_engine(self.db_uri)
            self.session_factory = scoped_session(sessionmaker(bind=self.engine))
            logger.info("Initialized fallback in-memory SQLite database")
            self._initialized = True  # Mark as initialized even with fallback
        
    def _compute_sqlite_target(self, ensure_dir: bool = True) -> Tuple[str, Optional[str]]:
        """Resolve the SQLite URI and optionally ensure its directory exists."""
        data_dir = os.getenv('DATA_DIR')
        if not data_dir:
            project_root = os.getenv('PROJECT_ROOT') or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            data_dir = os.path.join(project_root, 'data')

        if ensure_dir:
            try:
                os.makedirs(data_dir, exist_ok=True)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(f"Failed to create data directory at {data_dir}: {exc}")
                return "sqlite:///:memory:", None

        db_file_path = os.path.join(data_dir, 'football.db')
        return f"sqlite:///{db_file_path}", db_file_path

    def _prepare_sqlite_uri(self, ensure_dir: bool = True) -> str:
        """Prepare a SQLite URI for default usage, logging the resolved path."""
        sqlite_uri, db_file_path = self._compute_sqlite_target(ensure_dir=ensure_dir)
        if db_file_path:
            logger.info(f"Using default SQLite database at: {db_file_path}")
        else:
            logger.warning("Using in-memory SQLite database as fallback")
        return sqlite_uri

    def _verify_initial_connection(self):
        """Test connectivity to the configured database and apply fallback if needed."""
        if self.db_uri.startswith('sqlite'):
            return

        try:
            # Use a shorter timeout for the initial connection check (1 second)
            # to fail fast if PostgreSQL is unavailable
            connect_args_with_timeout = dict(self._connect_args or {})
            connect_args_with_timeout['connect_timeout'] = 1  # Fast-fail: 1 second
            # Add additional psycopg2-specific timeouts to prevent IPv4/IPv6 retry delays
            if 'postgresql' in self.db_uri.lower() or 'postgres' in self.db_uri.lower():
                connect_args_with_timeout['keepalives'] = 1
                connect_args_with_timeout['keepalives_idle'] = 1
                connect_args_with_timeout['keepalives_interval'] = 1
                connect_args_with_timeout['keepalives_count'] = 1
            
            # Create a temporary engine with fast-fail timeout
            test_engine = create_engine(
                self.db_uri,
                connect_args=connect_args_with_timeout,
                pool_pre_ping=False,  # Disable for speed
                pool_size=1,
                max_overflow=0
            )
            
            # Test connection with fast timeout
            with test_engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            # Clean up test engine
            test_engine.dispose()
            logger.debug("PostgreSQL connection verified successfully")
            
        except (OperationalError, DisconnectionError, TimeoutError) as db_err:
            self._handle_connection_failure(db_err)
        except SQLAlchemyError as sa_err:
            self._handle_connection_failure(sa_err)
        except Exception as e:
            # Catch any other connection errors
            logger.warning(f"Unexpected error during connection verification: {e}")
            self._handle_connection_failure(e)

    def _handle_connection_failure(self, failure: Exception):
        masked_uri = _mask_db_uri(self._primary_db_uri or self.db_uri)
        if not self._allow_sqlite_fallback:
            logger.error(f"Unable to connect to primary database {masked_uri}: {failure}")
            raise failure

        reason = _sanitize_failure_reason(failure)
        logger.warning(
            "Primary database %s is unreachable (%s). Switching to SQLite fallback.",
            masked_uri,
            reason,
        )
        self._activate_sqlite_fallback(failure)

    def _activate_sqlite_fallback(self, failure: Exception):
        """Switch the manager to a local SQLite database when the primary fails."""
        if self.using_sqlite_fallback:
            logger.error("SQLite fallback already active; propagating database failure.")
            raise failure

        reason = _sanitize_failure_reason(failure)
        sqlite_uri, db_file_path = self._compute_sqlite_target(ensure_dir=True)
        if db_file_path:
            logger.warning(
                "Falling back to SQLite database at %s (reason: %s)",
                db_file_path,
                reason,
            )
        else:
            logger.warning(
                "Falling back to in-memory SQLite database (reason: %s)",
                reason,
            )

        # Dispose old engine if it exists before replacing it
        try:
            if hasattr(self, 'engine') and self.engine:
                self.engine.dispose()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("Engine dispose during fallback raised but was suppressed.")

        self.db_uri = sqlite_uri
        self._connect_args = {}

        engine_kwargs = dict(self._engine_kwargs_base or {})
        if not engine_kwargs:
            engine_kwargs = {'pool_pre_ping': True}
        if self.db_uri.startswith('sqlite'):
            engine_kwargs.setdefault('connect_args', {})
            engine_kwargs['connect_args'].setdefault('check_same_thread', False)

        self.engine = create_engine(self.db_uri, **engine_kwargs)
        self.session_factory = scoped_session(sessionmaker(bind=self.engine))
        self.using_sqlite_fallback = True
        self.fallback_reason = reason
        logger.info(f"SQLite fallback initialized with URI: {_mask_db_uri(self.db_uri)}")

        try:
            Base.metadata.create_all(self.engine)
        except Exception as exc:  # pragma: no cover - defensive schema init
            logger.warning(f"Automatic table creation on SQLite fallback encountered an issue: {exc}")

        if self._allow_sqlite_fallback:
            with DatabaseManager._global_fallback_lock:
                DatabaseManager._global_fallback_active = True
                DatabaseManager._global_fallback_uri = self.db_uri
                DatabaseManager._global_fallback_reason = reason
                logger.info(
                    "Cached SQLite fallback globally (reason: %s)",
                    self.fallback_reason,
                )

    @classmethod
    def get_global_fallback_info(cls) -> Optional[Dict[str, str]]:
        """Return shared fallback metadata if a global fallback is active."""
        if not cls._global_fallback_active:
            return None
        return {
            "uri": cls._global_fallback_uri,
            "reason": cls._global_fallback_reason,
        }

    def connection_info(self) -> Dict[str, Any]:
        """Expose connection metadata for UI diagnostics."""
        global_info = self.get_global_fallback_info()
        configured_uri = self._primary_db_uri or self.db_uri
        active_uri = self.db_uri
        
        # Basic connection info
        info = {
            "configured_uri": configured_uri,
            "active_uri": active_uri,
            "masked_configured_uri": _mask_db_uri(configured_uri) if configured_uri else None,
            "masked_active_uri": _mask_db_uri(active_uri) if active_uri else None,
            "using_fallback": self.using_sqlite_fallback,
            "fallback_allowed": self._allow_sqlite_fallback,
            "fallback_reason": self.fallback_reason or (global_info or {}).get("reason"),
            "global_fallback": global_info,
            "database_type": "sqlite" if self.db_uri.startswith("sqlite") else "postgresql",
            "is_production_ready": False
        }
        
        # Add connection pool information if available
        if hasattr(self, 'engine') and self.engine:
            pool_stats = self.get_pool_stats()
            if pool_stats:
                info["pool_stats"] = pool_stats
        
        # Add PostgreSQL-specific production readiness check
        if active_uri and active_uri.startswith("postgresql"):
            # PostgreSQL is being used - this is good for production
            info["is_production_ready"] = True
            
            # Check for optimal production pool settings
            pool_size = getattr(self, 'pool_size', 0)
            max_overflow = getattr(self, 'max_overflow', 0)
            
            if pool_size < 5:
                info["production_warning"] = "Pool size is smaller than recommended for production (min 5)"
                info["is_production_ready"] = False
            
            # Add PostgreSQL version if available
            try:
                with self.engine.connect() as conn:
                    version = conn.execute(text("SELECT version()")).scalar()
                    if version:
                        info["postgresql_version"] = version
            except Exception as e:
                info["version_check_error"] = str(e)
        
        return info

    def create_tables(self):
        """Create all tables defined in the schema."""
        logger.info("Ensuring database schema is up to date.")

        try:
            Base.metadata.create_all(self.engine)

            inspector = inspect(self.engine)
            tables_in_db = sorted(inspector.get_table_names())
            logger.info("Schema synchronization complete (%d tables).", len(tables_in_db))

            expected_tables = ['api_cache', 'scraped_data', 'matches', 'leagues', 'teams']
            missing_core = [name for name in expected_tables if name not in tables_in_db]
            if missing_core:
                logger.warning("Missing expected tables after create_all: %s", missing_core)

            # Update schema to handle any new columns that have been added to models
            self.update_schema()

        except SQLAlchemyError as e:
            logger.error(f"Error during create_tables: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during create_tables: {e}", exc_info=True)
            raise
        
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        Handles commit/rollback and session cleanup.
        
        Yields:
            SQLAlchemy session
        """
        attempts = 0
        while True:
            try:
                session = self.session_factory()
                break
            except (OperationalError, DisconnectionError) as op_err:
                attempts += 1
                logger.error(f"Database session factory error: {op_err}")
                if attempts > 1 or not self._allow_sqlite_fallback:
                    raise op_err
                self._handle_connection_failure(op_err)
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        This is an alias for session_scope to maintain compatibility with other code.
        
        Yields:
            SQLAlchemy session
        """
        with self.session_scope() as session:
            yield session
    
    def update_schema(self):
        """
        Update the database schema to reflect the latest model changes.
        This is useful when new columns are added to existing tables.
        """
        logger.info("Checking for schema updates...")
        inspector = inspect(self.engine)
        tables_in_db = inspector.get_table_names()
        
        # Check and update each model's table schema
        for table_name, table in Base.metadata.tables.items():
            if table_name not in tables_in_db:
                logger.info(f"Table '{table_name}' not found, will be created")
                continue
                
            # Get existing columns in the database table
            existing_columns = set(column['name'] for column in inspector.get_columns(table_name))
            # Get columns defined in the model
            model_columns = set(column.name for column in table.columns)
            
            # Find columns to add
            missing_columns = model_columns - existing_columns
            if missing_columns:
                logger.info(f"Found missing columns in table '{table_name}': {missing_columns}")
                self._add_missing_columns(table_name, missing_columns, table)
    
    def _add_missing_columns(self, table_name, missing_columns, table):
        """Add missing columns to an existing table."""
        with self.engine.connect() as connection:
            for column_name in missing_columns:
                column = next(col for col in table.columns if col.name == column_name)
                column_type = column.type.compile(self.engine.dialect)
                nullable = "" if column.nullable else " NOT NULL"
                default = f" DEFAULT {column.default.arg}" if column.default is not None and not callable(column.default.arg) else ""
                
                # Create ALTER TABLE statement - syntax varies by DB type
                if self.db_uri.startswith('sqlite://'):
                    # SQLite has limited ALTER TABLE support
                    sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}{nullable}{default}"
                else:
                    # PostgreSQL, MySQL, etc.
                    sql = f"ALTER TABLE {table_name} ADD {column_name} {column_type}{nullable}{default}"
                
                try:
                    logger.info(f"Adding column '{column_name}' to table '{table_name}'")
                    connection.execute(text(sql))
                    connection.commit()
                except Exception as e:
                    logger.error(f"Failed to add column '{column_name}' to table '{table_name}': {e}")
    
    # --- League Operations ---
    
    def get_leagues(self, session: Optional[Session] = None) -> List[League]:
        """
        Get all leagues.
        
        Args:
            session: Optional session to use
            
        Returns:
            List of League objects
        """
        if session:
            return session.query(League).all()
        
        with self.session_scope() as s:
            return s.query(League).all()
            
    def get_league_by_id(self, league_id: str, session: Optional[Session] = None) -> Optional[League]:
        """
        Get league by ID.
        
        Args:
            league_id: League ID
            session: Optional session to use
            
        Returns:
            League object or None if not found
        """
        if session:
            return session.query(League).filter(League.id == league_id).first()
        
        with self.session_scope() as s:
            return s.query(League).filter(League.id == league_id).first()
    
    def get_league_by_name(self, name: str, session: Optional[Session] = None) -> Optional[League]:
        """
        Get league by name.
        
        Args:
            name: League name
            session: Optional session to use
            
        Returns:
            League object or None if not found
        """
        if session:
            return session.query(League).filter(League.name == name).first()
        
        with self.session_scope() as s:
            return s.query(League).filter(League.name == name).first()
    
    # --- Team Operations ---
    
    def get_teams(self, session: Optional[Session] = None) -> List[Team]:
        """
        Get all teams.
        
        Args:
            session: Optional session to use
            
        Returns:
            List of Team objects
        """
        if session:
            return session.query(Team).all()
        
        with self.session_scope() as s:
            return s.query(Team).all()
    
    def get_team_by_id(self, team_id: str, session: Optional[Session] = None) -> Optional[Team]:
        """
        Get team by ID.
        
        Args:
            team_id: Team ID
            session: Optional session to use
            
        Returns:
            Team object if found, None otherwise
        """
        if session:
            return session.query(Team).filter(Team.id == team_id).first()
        
        with self.session_scope() as s:
            return s.query(Team).filter(Team.id == team_id).first()
    
    def get_team_by_name(self, team_name: str, session: Optional[Session] = None) -> Optional[Team]:
        """
        Get team by name (case-insensitive).
        
        Args:
            team_name: Team name
            session: Optional session to use
            
        Returns:
            Team object if found, None otherwise
        """
        if session:
            return session.query(Team).filter(func.lower(Team.name) == func.lower(team_name)).first()
        
        with self.session_scope() as s:
            return s.query(Team).filter(func.lower(Team.name) == func.lower(team_name)).first()
    
    def get_teams_by_league(self, league_id: str, session: Optional[Session] = None) -> List[Team]:
        """
        Get teams by league ID.
        
        Args:
            league_id: League ID
            session: Optional session to use
            
        Returns:
            List of Team objects
        """
        if session:
            return session.query(Team).filter(Team.league_id == league_id).all()
        
        with self.session_scope() as s:
            return s.query(Team).filter(Team.league_id == league_id).all()
    
    def add_team(self, team_data: Dict, session: Optional[Session] = None) -> Team:
        """
        Add a team to the database.
        
        Args:
            team_data: Team data dictionary
            session: Optional session to use
            
        Returns:
            Created Team object
        """
        # Handle fields not in Team model
        valid_fields = {column.name for column in Team.__table__.columns}
        team_dict = {k: v for k, v in team_data.items() if k in valid_fields}
        
        team = Team(**team_dict)
        
        if session:
            session.add(team)
            session.flush()
            return team
        
        with self.session_scope() as s:
            s.add(team)
            s.flush()
            return team
            
    def bulk_add_teams(self, team_data_list: List[Dict], session: Optional[Session] = None) -> List[Team]:
        """
        Add multiple teams to the database.
        
        Args:
            team_data_list: List of team data dictionaries
            session: Optional session to use
            
        Returns:
            List of created Team objects
        """
        # Filter fields not in Team model for each team
        valid_fields = {column.name for column in Team.__table__.columns}
        team_dicts = [{k: v for k, v in team_data.items() if k in valid_fields} 
                     for team_data in team_data_list]
        
        teams = [Team(**team_dict) for team_dict in team_dicts]
        
        if session:
            session.add_all(teams)
            session.flush()
            return teams
        
        with self.session_scope() as s:
            s.add_all(teams)
            s.flush()
            return teams
    
    # --- Match Operations ---
    
    def get_matches(
        self,
        league_id: Optional[str] = None,
        team_id: Optional[str] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        status: Optional[str] = None,
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Match]:
        """
        Get matches with various filters.
        
        Args:
            league_id: Optional league ID filter
            team_id: Optional team ID filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            status: Optional match status filter
            limit: Maximum number of matches to return
            session: Optional session to use
            
        Returns:
            List of Match objects
        """
        if session:
            query = session.query(Match)
        else:
            with self.session_scope() as s:
                query = s.query(Match)
        
        # Apply filters
        if league_id:
            query = query.filter(Match.league_id == league_id)
        
        if team_id:
            query = query.filter(or_(Match.home_team_id == team_id, Match.away_team_id == team_id))
        
        if start_date:
            query = query.filter(Match.match_date >= start_date)
        
        if end_date:
            query = query.filter(Match.match_date <= end_date)
        
        if status:
            query = query.filter(Match.status == status)
        
        # Order by date and limit
        query = query.order_by(Match.match_date).limit(limit)
        
        if session:
            return query.all()
        
        with self.session_scope() as s:
            return query.all()
    
    def get_match_by_id(self, match_id: str, session: Optional[Session] = None) -> Optional[Match]:
        """
        Get match by ID.
        
        Args:
            match_id: Match ID
            session: Optional session to use
            
        Returns:
            Match object or None if not found
        """
        if session:
            return session.query(Match).filter(Match.id == match_id).first()
        
        with self.session_scope() as s:
            return s.query(Match).filter(Match.id == match_id).first()
    
    def get_match_with_details(self, match_id: str, session: Optional[Session] = None) -> Dict[str, Any]:
        """
        Get match with all related details (stats, predictions, odds).
        
        Args:
            match_id: Match ID
            session: Optional session to use
            
        Returns:
            Dictionary with match details
        """
        result = {}
        
        with self.session_scope() as s:
            # Get match
            match = s.query(Match).filter(Match.id == match_id).first()
            if not match:
                return {}
                
            # Basic match info
            result["match"] = {
                "id": match.id,
                "league_id": match.league_id,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "match_date": match.match_date,
                "status": match.status,
                "venue": match.venue,
                "home_score": match.home_score,
                "away_score": match.away_score
            }
            
            # Add team names
            result["match"]["home_team_name"] = match.home_team.name
            result["match"]["away_team_name"] = match.away_team.name
            result["match"]["league_name"] = match.league.name
            
            # Get stats
            if match.match_stats:
                result["stats"] = {
                    "home_possession": match.match_stats.home_possession,
                    "away_possession": match.match_stats.away_possession,
                    "home_shots": match.match_stats.home_shots,
                    "away_shots": match.match_stats.away_shots,
                    "home_shots_on_target": match.match_stats.home_shots_on_target,
                    "away_shots_on_target": match.match_stats.away_shots_on_target,
                    "home_corners": match.match_stats.home_corners,
                    "away_corners": match.match_stats.away_corners,
                    "home_fouls": match.match_stats.home_fouls,
                    "away_fouls": match.match_stats.away_fouls,
                    "home_yellow_cards": match.match_stats.home_yellow_cards,
                    "away_yellow_cards": match.match_stats.away_yellow_cards,
                    "home_red_cards": match.match_stats.home_red_cards,
                    "away_red_cards": match.match_stats.away_red_cards
                }
                
                # Add extra stats if available
                if match.match_stats.extra_stats:
                    result["stats"].update(match.match_stats.extra_stats)
            
            # Get latest prediction
            latest_prediction = s.query(Prediction).filter(
                Prediction.match_id == match_id
            ).order_by(Prediction.created_at.desc()).first()
            
            if latest_prediction:
                result["prediction"] = {
                    "home_win": latest_prediction.home_win_prob,
                    "draw": latest_prediction.draw_prob,
                    "away_win": latest_prediction.away_win_prob,
                    "home_score": latest_prediction.home_score_pred,
                    "away_score": latest_prediction.away_score_pred,
                    "confidence": latest_prediction.confidence,
                    "model": latest_prediction.model_name,
                    "feature_importance": latest_prediction.feature_importance
                }
            
            # Get latest odds from each bookmaker
            latest_odds = {}
            odds_query = s.query(Odds).filter(Odds.match_id == match_id).order_by(Odds.timestamp.desc())
            
            for odds in odds_query:
                if odds.bookmaker not in latest_odds:
                    latest_odds[odds.bookmaker] = {
                        "home_win": odds.home_win,
                        "draw": odds.draw,
                        "away_win": odds.away_win,
                        "over_2_5": odds.over_under_2_5_over,
                        "under_2_5": odds.over_under_2_5_under,
                        "btts_yes": odds.both_teams_to_score_yes,
                        "btts_no": odds.both_teams_to_score_no,
                        "timestamp": odds.timestamp
                    }
            
            result["odds"] = latest_odds
            
            # Get value bets
            value_bets = s.query(ValueBet).filter(ValueBet.match_id == match_id).all()
            if value_bets:
                result["value_bets"] = []
                for vb in value_bets:
                    result["value_bets"].append({
                        "bet_type": vb.bet_type,
                        "predicted_prob": vb.predicted_prob,
                        "implied_prob": vb.implied_prob,
                        "edge": vb.edge,
                        "kelly_stake": vb.kelly_stake,
                        "confidence": vb.confidence
                    })
            
            return result
    
    # --- Prediction Operations ---
    
    def get_predictions(
        self,
        match_id: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
        session: Optional[Session] = None
    ) -> List[Prediction]:
        """
        Get predictions with various filters.
        
        Args:
            match_id: Optional match ID filter
            model_name: Optional model name filter
            limit: Maximum number of predictions to return
            session: Optional session to use
            
        Returns:
            List of Prediction objects
        """
        if session:
            query = session.query(Prediction)
        else:
            with self.session_scope() as s:
                query = s.query(Prediction)
        
        # Apply filters
        if match_id:
            query = query.filter(Prediction.match_id == match_id)
        
        if model_name:
            query = query.filter(Prediction.model_name == model_name)
        
        # Order by creation date and limit
        query = query.order_by(Prediction.created_at.desc()).limit(limit)
        
        if session:
            return query.all()
        
        with self.session_scope() as s:
            return query.all()
    
    def save_prediction(
        self,
        match_id: str,
        model_name: str,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        home_score_pred: Optional[float] = None,
        away_score_pred: Optional[float] = None,
        confidence: Optional[float] = None,
        feature_importance: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None
    ) -> Prediction:
        """
        Save a new prediction.
        
        Args:
            match_id: Match ID
            model_name: Model name
            home_win_prob: Home win probability
            draw_prob: Draw probability
            away_win_prob: Away win probability
            home_score_pred: Optional predicted home score
            away_score_pred: Optional predicted away score
            confidence: Optional model confidence
            feature_importance: Optional feature importance dictionary
            session: Optional session to use
            
        Returns:
            Saved Prediction object
        """
        prediction = Prediction(
            match_id=match_id,
            model_name=model_name,
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            home_score_pred=home_score_pred,
            away_score_pred=away_score_pred,
            confidence=confidence,
            feature_importance=feature_importance
        )
        
        if session:
            session.add(prediction)
            session.flush()
            return prediction
        
        with self.session_scope() as s:
            s.add(prediction)
            s.flush()
            s.refresh(prediction)
            return prediction
    
    # --- Odds Operations ---
    
    def save_odds(
        self,
        match_id: str,
        bookmaker: str,
        home_win: float,
        draw: float,
        away_win: float,
        over_under_2_5_over: Optional[float] = None,
        over_under_2_5_under: Optional[float] = None,
        both_teams_to_score_yes: Optional[float] = None,
        both_teams_to_score_no: Optional[float] = None,
        session: Optional[Session] = None
    ) -> Odds:
        """
        Save new odds.
        
        Args:
            match_id: Match ID
            bookmaker: Bookmaker name
            home_win: Home win odds
            draw: Draw odds
            away_win: Away win odds
            over_under_2_5_over: Optional over 2.5 goals odds
            over_under_2_5_under: Optional under 2.5 goals odds
            both_teams_to_score_yes: Optional BTTS yes odds
            both_teams_to_score_no: Optional BTTS no odds
            session: Optional session to use
            
        Returns:
            Saved Odds object
        """
        odds = Odds(
            match_id=match_id,
            bookmaker=bookmaker,
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            over_under_2_5_over=over_under_2_5_over,
            over_under_2_5_under=over_under_2_5_under,
            both_teams_to_score_yes=both_teams_to_score_yes,
            both_teams_to_score_no=both_teams_to_score_no
        )
        
        if session:
            session.add(odds)
            session.flush()
            return odds
        
        with self.session_scope() as s:
            s.add(odds)
            s.flush()
            s.refresh(odds)
            return odds
    
    # --- Value Bet Operations ---
    
    def save_value_bet(
        self,
        match_id: str,
        prediction_id: int,
        odds_id: int,
        bet_type: str,
        predicted_prob: float,
        implied_prob: float,
        edge: float,
        kelly_stake: float,
        confidence: str,
        session: Optional[Session] = None
    ) -> ValueBet:
        """
        Save a new value bet.
        
        Args:
            match_id: Match ID
            prediction_id: Prediction ID
            odds_id: Odds ID
            bet_type: Bet type (e.g., "home_win", "draw", "away_win")
            predicted_prob: Predicted probability
            implied_prob: Implied probability from odds
            edge: Edge value ((predicted_prob * odds) - 1)
            kelly_stake: Kelly criterion stake
            confidence: Confidence level ("Low", "Medium", "High")
            session: Optional session to use
            
        Returns:
            Saved ValueBet object
        """
        value_bet = ValueBet(
            match_id=match_id,
            prediction_id=prediction_id,
            odds_id=odds_id,
            bet_type=bet_type,
            predicted_prob=predicted_prob,
            implied_prob=implied_prob,
            edge=edge,
            kelly_stake=kelly_stake,
            confidence=confidence
        )
        
        if session:
            session.add(value_bet)
            session.flush()
            return value_bet
        
        with self.session_scope() as s:
            s.add(value_bet)
            s.flush()
            s.refresh(value_bet)
            return value_bet

    def _setup_connection_pool(self, db_uri, pool_size=None, max_overflow=None, timeout=None, recycle=None):
        """
        Set up a connection pool for better database performance.
        
        Args:
            db_uri: Database URI
            pool_size: Size of the connection pool (None uses instance default)
            max_overflow: Maximum overflow connections (None uses instance default)
            timeout: Pool timeout in seconds (None uses instance default)
            recycle: Connection recycle time in seconds (None uses instance default)
        """
        from sqlalchemy.pool import QueuePool

        # Use instance defaults if not specified
        pool_size = pool_size if pool_size is not None else self.pool_size
        max_overflow = max_overflow if max_overflow is not None else self.max_overflow
        timeout = timeout if timeout is not None else self.pool_timeout
        recycle = recycle if recycle is not None else self.pool_recycle
        
        # Optimize pool settings for PostgreSQL production environments
        is_postgres = db_uri.startswith('postgresql')
        is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
        
        if is_postgres and is_production:
            # In production PostgreSQL, use larger pools with optimized settings
            prod_pool_size = int(os.getenv('POSTGRES_POOL_SIZE', str(max(10, pool_size))))
            prod_max_overflow = int(os.getenv('POSTGRES_MAX_OVERFLOW', str(max(20, max_overflow))))
            
            logger.info(f"Using production PostgreSQL pool settings: size={prod_pool_size}, max_overflow={prod_max_overflow}")
            
            pool_size = prod_pool_size
            max_overflow = prod_max_overflow
            
            # Enable pre-ping for production PostgreSQL to detect stale connections
            engine_kwargs = {
                'poolclass': QueuePool,
                'pool_size': pool_size,
                'max_overflow': max_overflow,
                'pool_timeout': timeout,
                'pool_recycle': recycle,
                'pool_pre_ping': True,  # Always check connection validity before use
            }
            
            # Add health checks for production PostgreSQL
            if os.getenv('POSTGRES_POOL_HEALTH_CHECK', 'true').lower() not in {'false', '0', 'no', 'off'}:
                engine_kwargs['pool_use_lifo'] = True  # Use Last-In-First-Out for better connection reuse
                
            self.engine = create_engine(db_uri, **engine_kwargs)
        else:
            # Standard configuration for non-production or non-PostgreSQL
            self.engine = create_engine(
                db_uri,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=timeout,
                pool_recycle=recycle,
                pool_pre_ping=True
            )
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        self.scoped_session = scoped_session(self.Session)
        
        logger.info(f"Configured database connection pool: size={pool_size}, max_overflow={max_overflow}, recycle={recycle}s, uri={_mask_db_uri(db_uri)}")
        
    def get_pool_stats(self):
        """
        Get current statistics about the connection pool.
        
        Returns:
            Dictionary with pool statistics
        """
        if hasattr(self, 'engine') and self.engine:
            return {
                "pool_size": self.pool_size,
                "connections_in_use": self.engine.pool.checkedout(),
                "connections_available": self.engine.pool.checkedin(),
                "max_overflow": self.max_overflow,
                "overflow": self.engine.pool.overflow(),
                "pool_timeout": self.pool_timeout,
                "pool_recycle": self.pool_recycle
            }
        return None
        
    @contextmanager
    def get_connection(self):
        """
        Get a raw database connection with cursor() method for direct SQL execution.
        This is a wrapper around SQLAlchemy's connection to provide a standard interface.
        
        Yields:
            A connection object with cursor() method
        """
        class ConnectionWrapper:
            def __init__(self, sa_connection):
                self.sa_connection = sa_connection
                self._cursor = None

            def cursor(self):
                class CursorWrapper:
                    def __init__(self, sa_connection):
                        self.sa_connection = sa_connection
                        self._last_result = None
                    def execute(self, sql, params=None):
                        if params is None or params == {}:
                            self._last_result = self.sa_connection.execute(text(sql))
                        else:
                            # Support sqlite style positional parameters ("?") by using driver-level execution
                            if isinstance(params, (list, tuple)):
                                self._last_result = self.sa_connection.exec_driver_sql(sql, params)
                            elif isinstance(params, dict):
                                self._last_result = self.sa_connection.execute(text(sql), params)
                            else:
                                # Fallback to driver execution for other iterables
                                self._last_result = self.sa_connection.exec_driver_sql(sql, params)
                        # Auto-commit for non-session operations when needed
                        if self.sa_connection.in_transaction():
                            try:
                                self.sa_connection.commit()
                            except Exception:
                                pass
                        return self._last_result
                    def fetchall(self):
                        if self._last_result is None:
                            return []
                        rows = self._last_result.fetchall()
                        return rows
                    def fetchone(self):
                        if self._last_result is None:
                            return None
                        return self._last_result.fetchone()
                    def close(self):
                        pass
                if not self._cursor:
                    self._cursor = CursorWrapper(self.sa_connection)
                return self._cursor

            def close(self):
                self.sa_connection.close()

        with self.engine.connect() as connection:
            yield ConnectionWrapper(connection)
    
    # --- Match Operations ---
    
    def add_match(self, match_data: Dict, session: Optional[Session] = None) -> Match:
        """
        Add a match to the database.
        
        Args:
            match_data: Match data dictionary
            session: Optional session to use
            
        Returns:
            Created Match object
        """
        # Handle fields not in Match model
        valid_fields = {column.name for column in Match.__table__.columns}
        match_dict = {k: v for k, v in match_data.items() if k in valid_fields}
        
        match = Match(**match_dict)
        
        if session:
            session.add(match)
            session.flush()
            return match
        
        with self.session_scope() as s:
            s.add(match)
            s.flush()
            return match
    
    def add_match_stats(self, stats_data: Dict, session: Optional[Session] = None) -> MatchStats:
        """
        Add match statistics to the database.
        
        Args:
            stats_data: Match stats data dictionary
            session: Optional session to use
            
        Returns:
            Created MatchStats object
        """
        # Handle fields not in MatchStats model
        valid_fields = {column.name for column in MatchStats.__table__.columns}
        stats_dict = {k: v for k, v in stats_data.items() if k in valid_fields}
        
        match_stats = MatchStats(**stats_dict)
        
        if session:
            session.add(match_stats)
            session.flush()
            return match_stats
        
        with self.session_scope() as s:
            s.add(match_stats)
            s.flush()
            return match_stats
    
    def add_league(self, league_data: Dict, session: Optional[Session] = None) -> League:
        """
        Add a league to the database.
        
        Args:
            league_data: League data dictionary
            session: Optional session to use
            
        Returns:
            Created League object
        """
        # Handle fields not in League model
        valid_fields = {column.name for column in League.__table__.columns}
        league_dict = {k: v for k, v in league_data.items() if k in valid_fields}
        
        league = League(**league_dict)
        
        if session:
            session.add(league)
            session.flush()
            return league
        
        with self.session_scope() as s:
            s.add(league)
            s.flush()
            return league
    
    def test_connection(self) -> bool:
        """
        Test the database connection.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            with self.session_scope() as session:
                # Simple query to test connection
                result = session.execute(text("SELECT 1")).fetchone()
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def check_connection(self) -> bool:
        """
        Check if the database connection is working.
        This method is used by the system health check.

        Returns:
            True if connection is successful, False otherwise
        """
        return self.test_connection()

    def get_table_names(self) -> List[str]:
        """
        Get list of all table names in the database.

        Returns:
            List of table names
        """
        try:
            from sqlalchemy import inspect
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            logger.info(f"Found {len(tables)} tables in database: {tables}")
            return tables
        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            return []
    
    def get_tables(self) -> List[str]:
        """
        Get list of all table names in the database.
        This is an alias for get_table_names used by the health check.

        Returns:
            List of table names
        """
        return self.get_table_names()

    def start_operation(self, operation_name: str) -> str:
        """
        Start a database operation and return an operation ID.
        This is used for tracking and monitoring database operations.

        Args:
            operation_name: Name of the operation to start

        Returns:
            Operation ID string for tracking purposes
        """
        operation_id = f"{operation_name}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        logger.info(f"Starting database operation: {operation_name} (ID: {operation_id})")
        return operation_id
        
    def end_operation(self, operation_id: str, success: bool = True, metadata: Dict[str, Any] = None, status: str = None):
        """
        End a database operation and log results.
        
        Args:
            operation_id: ID of the operation to end
            success: Whether the operation was successful (deprecated, use status instead)
            metadata: Additional metadata about the operation
            status: Status of the operation (success, failure, partial_success, partial_failure, warning)
        """
        # Handle the case where status is provided directly
        if status is not None:
            result_status = status
        else:
            # Legacy support for success parameter
            result_status = "success" if success else "failure"
            
        logger.info(f"Database operation {operation_id} ended with status: {result_status}")
        if metadata:
            logger.debug(f"Operation {operation_id} metadata: {json.dumps(metadata)}")
    
    def close(self):
        """
        Close any open connections and clean up resources.
        Should be called when the application is shutting down.
        """
        logger.info("Closing database manager and releasing resources.")
        if hasattr(self, 'Session'):
            self.Session.remove()
        
        if hasattr(self, 'engine'):
            self.engine.dispose()
            logger.info("Engine disposed and connections closed.")
    
    @classmethod
    def clear_instance_cache(cls, db_uri: str = None):
        """
        Clear the singleton instance cache.
        Use this to force re-initialization of DatabaseManager instances.
        
        Args:
            db_uri: If provided, only clear the instance for this specific URI.
                   If None, clear all cached instances.
        """
        with cls._instance_lock:
            if db_uri:
                cache_key = db_uri or "default"
                if cache_key in cls._instances:
                    logger.info(f"Clearing cached DatabaseManager instance for: {_mask_db_uri(cache_key)}")
                    del cls._instances[cache_key]
            else:
                logger.info("Clearing all cached DatabaseManager instances")
                cls._instances.clear()
    
    @classmethod
    def get_cached_instance_count(cls) -> int:
        """
        Get the number of cached DatabaseManager instances.
        Useful for monitoring and debugging.
        
        Returns:
            Number of cached instances
        """
        return len(cls._instances)
