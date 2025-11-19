"""
Data integration module for the football betting insights platform.
Integrates data from various sources and provides a unified interface for the dashboard.
"""
import asyncio
import json
import logging
import os
import re
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from api.understat_client import UnderstatAPIClient
from scripts.scrapers.base_scraper import BaseScraper
from scripts.scrapers.football_data_scraper import FootballDataScraper
from scripts.scrapers.scraper_factory import get_scraper
from utils.data_validator import (
    sanitize_dataframe,
    validate_match_data,
    validate_player_data,
    validate_team_data,
)
from utils.http_client_async import HttpClientAsync
from utils.system_monitor import SystemMonitor, end_operation, start_operation

logger = logging.getLogger(__name__)

def get_scraper_status() -> Dict[str, Dict[str, Any]]:
    """
    Get the status of all available scrapers.
    
    Returns:
        Dictionary with scraper status information
    """
    try:
        # Initialize scrapers to check status
        scrapers = {
            "football_data": {
                "name": "Football-Data API",
                "status": "unavailable",
                "last_check": datetime.now().isoformat(),
                "details": {"error": "Not initialized"}
            },
            "understat": {
                "name": "Understat API",
                "status": "unavailable",
                "last_check": datetime.now().isoformat(),
                "details": {"error": "Not initialized"}
            }
        }

        # Check Football-Data API status
        football_data_scraper = get_scraper("football_data")
        if football_data_scraper:
            scrapers["football_data"]["status"] = "available"
            scrapers["football_data"]["details"] = {
                "base_url": getattr(football_data_scraper, "base_url", "Unknown"),
                "rate_limit": getattr(football_data_scraper, "rate_limit", None),
                "api_key_valid": bool(getattr(football_data_scraper, "api_key", None)),
            }

        # Check Understat API client status
        try:
            understat_client = UnderstatAPIClient()
            scrapers["understat"]["status"] = "available"
            scrapers["understat"]["details"] = {
                "base_url": understat_client.base_url,
                "cache_enabled": understat_client.use_cache,
                "max_retries": understat_client.max_retries,
                "api_key_present": bool(understat_client.api_key),
            }
            # Ensure underlying session is closed if we created it
            if hasattr(understat_client, "close"):
                try:
                    asyncio.run(understat_client.close())
                except RuntimeError:
                    # Likely running inside existing event loop; schedule closing later
                    pass
        except Exception as exc:
            scrapers["understat"]["details"] = {"error": str(exc)}

        return scrapers

    except Exception as e:
        logger.error(f"Error getting scraper status: {e}")
        return {
            "error": {
                "name": "Error",
                "status": "error",
                "last_check": datetime.now().isoformat(),
                "details": {"error": str(e)}
            }
        }

class DataIntegration:
    """
    Data integration for the football betting insights platform.
    Fetches, processes, and combines data from various sources.
    """
    
    # League mapping for consistency across scrapers
    LEAGUE_MAPPING = {
        "Premier League": "premier_league",
        "La Liga": "la_liga",
        "Bundesliga": "bundesliga",
        "Serie A": "serie_a",
        "Ligue 1": "ligue_1",
        "Eredivisie": "eredivisie"
    }
    
    # Reverse league mapping for display
    DISPLAY_LEAGUE_MAPPING = {v: k for k, v in LEAGUE_MAPPING.items()}
    
    def __init__(
        self,
        system_monitor: Optional[SystemMonitor] = None,
        use_cache: bool = True,
        cache_expiry: int = 3600,
        understat_client: Optional[UnderstatAPIClient] = None,
    ):
        """
        Initialize the data integration module with enhanced performance settings.
        
        Args:
            system_monitor: Optional system monitor for tracking performance
            use_cache: Whether to use caching (default: True)
            cache_expiry: Default cache expiration time in seconds (default: 3600 = 1 hour)
            understat_client: Optional pre-configured Understat API client instance
        """
        self.system_monitor = system_monitor or SystemMonitor()
        self.use_cache = use_cache
        self.cache = {}
        self.cache_expiry = {}
        self.default_cache_duration = cache_expiry
        
        # Cache control - remember frequent user queries for faster response
        self.frequent_queries = {}
        self.query_frequency = {}
        self.prefetch_threshold = 3  # Prefetch data after 3 queries of the same type
        
        # Tiered cache system for performance optimization
        # L1: In-memory cache for fastest access (Python dict)
        # L2: File-based cache for persistence across restarts (JSON files)
        self.l1_cache = {}
        self.l2_cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(self.l2_cache_dir, exist_ok=True)
        
        # Initialize HTTP client
        # Note: HttpClientAsync loads its configuration from the Config system
        # rather than through constructor parameters
        self.http_client = HttpClientAsync()  # No parameters needed as it will use default config
        
        # Initialize Understat API client (reuse HTTP client session when possible)
        self.understat_client: Optional[UnderstatAPIClient]
        self.understat_enabled = False
        if understat_client is not None:
            self.understat_client = understat_client
            self.understat_enabled = True
        else:
            try:
                self.understat_client = UnderstatAPIClient(http_client=self.http_client)
                self.understat_enabled = True
            except Exception as exc:
                logger.warning("Failed to initialize Understat API client: %s", exc)
                self.understat_client = None
                self.understat_enabled = False

        # Cache for Understat metadata to avoid redundant lookups
        self._understat_team_index: Dict[Tuple[str, str], Dict[str, str]] = {}

        # Initialize scrapers with our enhanced infrastructure
        self._scrapers = {}
        self._initialize_scrapers()
        
        # Track failed data sources to avoid repeated failures
        self.failed_sources = set()
        self.source_health = {}  # Track health of data sources
        
        # Enhanced performance metrics for monitoring and optimization
        self.performance_metrics = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "avg_response_time": 0,
            "request_times": [],     # Track last 100 request times for analysis
            "success_rate": 1.0,      # Success rate (0.0-1.0)
            "bandwidth_usage": 0,      # Approximate bandwidth usage in bytes
            "last_optimization": datetime.now()
        }
        
        # Load any previously cached data from L2 cache
        self._load_persistent_cache()
        
        logger.info(f"Enhanced DataIntegration initialized with {'enabled' if use_cache else 'disabled'} "
                   f"caching (expiry: {cache_expiry}s) and performance tracking")
    
    def _load_persistent_cache(self) -> None:
        """
        Load data from L2 cache (file system) into L1 cache (memory).
        This provides persistence across application restarts.
        """
        if not self.use_cache:
            return
            
        try:
            # Track loading time for performance metrics
            start_time = datetime.now()
            cache_files = [f for f in os.listdir(self.l2_cache_dir) if f.endswith('.json')]
            loaded_count = 0
            expired_count = 0
            
            for cache_file in cache_files:
                try:
                    file_path = os.path.join(self.l2_cache_dir, cache_file)
                    # Skip files older than our cache expiry
                    file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if (datetime.now() - file_mod_time).total_seconds() > self.default_cache_duration:
                        os.remove(file_path)  # Clean up expired cache file
                        expired_count += 1
                        continue
                        
                    with open(file_path, 'r') as f:
                        cache_data = json.load(f)
                        
                    # Extract key and metadata
                    cache_key = cache_file.replace('.json', '')
                    expiry_time = cache_data.get('expiry_time')
                    data = cache_data.get('data')
                    
                    # Check if data has expired
                    if expiry_time and datetime.fromisoformat(expiry_time) > datetime.now():
                        # Convert DataFrame data back to DataFrame if needed
                        if isinstance(data, dict) and cache_data.get('data_type') == 'dataframe':
                            data = pd.DataFrame(data)
                            
                        # Store in L1 cache
                        self.l1_cache[cache_key] = data
                        # Set expiry in cache_expiry
                        self.cache_expiry[cache_key] = datetime.fromisoformat(expiry_time)
                        loaded_count += 1
                    else:
                        # Remove expired cache file
                        os.remove(file_path)
                        expired_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {str(e)}")
                    
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"L2 cache loaded: {loaded_count} items loaded, {expired_count} expired items cleaned up in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load persistent cache: {str(e)}")
            # Continue without persistent cache
    
    def _save_to_persistent_cache(self, cache_key: str, data: Any, expiry_time: datetime) -> None:
        """
        Save data to L2 cache (file system) for persistence across restarts.
        
        Args:
            cache_key: Unique identifier for the cached data
            data: The data to cache (must be JSON serializable)
            expiry_time: When the cache entry expires
        """
        if not self.use_cache:
            return
            
        try:
            # Prepare data for serialization
            cache_data = {
                'expiry_time': expiry_time.isoformat(),
                'creation_time': datetime.now().isoformat(),
                'data_type': 'dataframe' if isinstance(data, pd.DataFrame) else 'other',
                'data': data.to_dict('records') if isinstance(data, pd.DataFrame) else data
            }
            
            # Create a safe filename from the cache key
            safe_key = ''.join(c if c.isalnum() else '_' for c in cache_key)
            if len(safe_key) > 100:  # Limit filename length
                # Use hash for long keys to avoid filename length issues
                import hashlib
                safe_key = hashlib.md5(cache_key.encode()).hexdigest()
                
            file_path = os.path.join(self.l2_cache_dir, f"{safe_key}.json")
            
            with open(file_path, 'w') as f:
                json.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save to persistent cache: {str(e)}")
            # Continue without saving to persistent cache
    
    def _get_cache_key(self, method_name: str, *args, **kwargs) -> str:
        """
        Generate a cache key for a method call.
        
        Args:
            method_name: Name of the method
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Convert args and kwargs to strings
        args_str = '_'.join(str(arg) for arg in args)
        kwargs_str = '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        
        return f"{method_name}_{args_str}_{kwargs_str}"
    
    def _initialize_scrapers(self) -> None:
        """
        Initialize the scrapers with our enhanced async infrastructure.
        Makes the initialization more resilient by initializing each scraper independently.
        """
        operation_id = self.system_monitor.start_operation("initialize_scrapers")
        success_count = 0
        
        try:
            # Create shared HTTP client for efficient connection pooling
            http_client = self.http_client
            
            # Initialize each scraper separately to prevent one failure from stopping all scrapers
            scrapers_to_init = {
                "football_data": FootballDataScraper,
                # Additional scrapers can be registered here
            }

            total_targets = len(scrapers_to_init) + (1 if self.understat_client else 0)
            
            for scraper_name, scraper_class in scrapers_to_init.items():
                try:
                    # Try to initialize with system_monitor first
                    try:
                        self._scrapers[scraper_name] = scraper_class(http_client=http_client)
                        success_count += 1
                        logger.info(f"Initialized {scraper_name} scraper successfully")
                    except TypeError as type_error:
                        # If TypeError occurs (likely due to unexpected arguments), try without system_monitor
                        if "unexpected keyword argument" in str(type_error):
                            logger.warning(f"Could not initialize {scraper_name} with all parameters: {type_error}")
                            # Try again without the problematic parameter
                            if "system_monitor" in str(type_error):
                                self._scrapers[scraper_name] = scraper_class(http_client=http_client)
                                success_count += 1
                                logger.info(f"Initialized {scraper_name} scraper with limited parameters")
                            else:
                                # Some other unexpected keyword error
                                raise
                        else:
                            # Not a keyword argument error
                            raise
                except Exception as e:
                    logger.error(f"Failed to initialize {scraper_name} scraper: {e}")
                    # Continue with other scrapers

            # Register Understat API client separately so we can leverage richer data
            if self.understat_client:
                self._scrapers["understat"] = self.understat_client
                success_count += 1
                logger.info("Registered Understat API client for advanced metrics")
            else:
                logger.warning("Understat API client unavailable; advanced metrics disabled")
            
            logger.info(f"Successfully initialized {success_count}/{total_targets or 1} data sources")
        except Exception as e:
            logger.error(f"Overall error in scraper initialization process: {e}")
            self.system_monitor.end_operation(operation_id, status="partial_failure")
        else:
            status = "success" if success_count == total_targets else "partial_success"
            self.system_monitor.end_operation(operation_id, status=status)

    def _normalize_team_name(self, name: Optional[str]) -> str:
        """Normalize team names for consistent mapping."""
        if not name:
            return ""
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def _resolve_league_code(self, competition: Optional[Union[str, Dict[str, Any]]]) -> Optional[str]:
        """Resolve display competition information to an Understat league code."""
        if not competition:
            return None

        candidates: List[str] = []
        if isinstance(competition, dict):
            for key in ("code", "id", "name", "shortName"):
                value = competition.get(key)
                if isinstance(value, str):
                    candidates.append(value)
        elif isinstance(competition, str):
            candidates.append(competition)

        for candidate in candidates:
            if not candidate:
                continue
            if candidate in self.LEAGUE_MAPPING.values():
                return candidate
            mapped = self.LEAGUE_MAPPING.get(candidate)
            if mapped:
                return mapped
            # Case-insensitive match against display mapping
            for display_name, league_code in self.LEAGUE_MAPPING.items():
                if candidate.lower() == display_name.lower():
                    return league_code
        return None

    def _infer_understat_season(self, match_date: Optional[Union[str, datetime]]) -> str:
        """Infer the Understat season string (e.g. '2024') from a match date."""
        anchor: datetime
        if isinstance(match_date, datetime):
            anchor = match_date
        elif isinstance(match_date, str) and match_date:
            try:
                anchor = datetime.fromisoformat(match_date.replace("Z", "+00:00"))
            except ValueError:
                parsed = pd.to_datetime(match_date, errors="coerce")
                if isinstance(parsed, pd.Timestamp):
                    anchor = parsed.to_pydatetime()
                else:
                    anchor = datetime.now(timezone.utc)
        else:
            anchor = datetime.now(timezone.utc)

        if anchor.month >= 7:
            season_year = anchor.year
        else:
            season_year = anchor.year - 1
        return str(season_year)

    async def _get_understat_team_index(self, league_code: str, season: str) -> Dict[str, str]:
        """Build or retrieve a mapping of normalized team names to Understat team IDs."""
        cache_key = (league_code, season)
        if cache_key in self._understat_team_index:
            return self._understat_team_index[cache_key]

        if not self.understat_enabled or not self.understat_client:
            self._understat_team_index[cache_key] = {}
            return {}

        mapping: Dict[str, str] = {}
        try:
            league_matches = await self.understat_client.get_league_matches(league_code, season)
            if league_matches is not None and not league_matches.empty:
                for _, row in league_matches.iterrows():
                    home_name = row.get("home_team")
                    away_name = row.get("away_team")
                    home_id = row.get("home_team_id")
                    away_id = row.get("away_team_id")
                    if home_name and home_id is not None:
                        mapping[self._normalize_team_name(str(home_name))] = str(home_id)
                    if away_name and away_id is not None:
                        mapping[self._normalize_team_name(str(away_name))] = str(away_id)
        except Exception as exc:
            logger.warning("Failed to build Understat team index for %s %s: %s", league_code, season, exc)

        self._understat_team_index[cache_key] = mapping
        return mapping

    async def _get_understat_team_id(self, team_name: str, league_code: str, season: str) -> Optional[str]:
        """Resolve a team name to an Understat team ID."""
        if not team_name:
            return None
        index = await self._get_understat_team_index(league_code, season)
        normalized = self._normalize_team_name(team_name)
        team_id = index.get(normalized)
        if team_id is None and team_name:
            logger.debug("Understat team ID not found for %s in %s %s", team_name, league_code, season)
        return team_id

    def _extract_stat(self, stats: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[float]:
        """Safely extract numeric statistics from the Understat payload."""
        for key in keys:
            value = stats.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _summarize_understat_team_stats(self, team_stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Summarize relevant Understat team statistics for dashboard consumption."""
        if not team_stats:
            return None

        stats_block = team_stats.get("stats") or {}
        matches_recorded = stats_block.get("matches_played") or stats_block.get("matches")
        if matches_recorded is None:
            matches_recorded = len(team_stats.get("matches", []))

        summary: Dict[str, Any] = {
            "team_name": team_stats.get("team_name") or team_stats.get("name"),
            "season": team_stats.get("season"),
            "matches_recorded": matches_recorded,
            "goals_scored": self._extract_stat(stats_block, ("goals", "scored", "goals_for")),
            "goals_conceded": self._extract_stat(stats_block, ("goals_against", "conceded", "goalsAgainst")),
            "xg": self._extract_stat(stats_block, ("xg", "xG")),
            "xga": self._extract_stat(stats_block, ("xga", "xGA")),
            "npxg": self._extract_stat(stats_block, ("npxg", "npxG")),
            "npxga": self._extract_stat(stats_block, ("npxga", "npxGA")),
            "deep": self._extract_stat(stats_block, ("deep",)),
            "deep_allowed": self._extract_stat(stats_block, ("deep_allowed",)),
        }

        matches = max(matches_recorded or 0, 1)
        if summary["xg"] is not None:
            summary["xg_per_match"] = round(summary["xg"] / matches, 3)
        else:
            summary["xg_per_match"] = None
        if summary["xga"] is not None:
            summary["xga_per_match"] = round(summary["xga"] / matches, 3)
        else:
            summary["xga_per_match"] = None

        return summary

    async def _get_understat_team_summary(
        self,
        team_name: str,
        league_code: str,
        season: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch and summarize Understat information for a specific team."""
        if not self.understat_enabled or not self.understat_client:
            return None

        team_id = await self._get_understat_team_id(team_name, league_code, season)
        if not team_id:
            return None

        try:
            team_stats = await self.understat_client.get_team_stats(team_id, season)
        except Exception as exc:
            logger.warning(
                "Error retrieving Understat stats for %s (%s %s): %s",
                team_name,
                league_code,
                season,
                exc,
            )
            return None

        summary = self._summarize_understat_team_stats(team_stats)
        if summary is None:
            return None

        summary["understat_team_id"] = team_id
        return summary

    async def _build_understat_context(self, match_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Construct Understat context for a match, including team summaries and xG gap."""
        if not self.understat_enabled or not self.understat_client or not match_details:
            return None

        competition = match_details.get("competition")
        league_code = self._resolve_league_code(competition)
        if not league_code:
            return None

        match_date = (
            match_details.get("match_date")
            or match_details.get("utc_date")
            or match_details.get("utcDate")
        )
        season = self._infer_understat_season(match_date)

        home_team_name = match_details.get("home_team") or match_details.get("homeTeam", {}).get("name")
        away_team_name = match_details.get("away_team") or match_details.get("awayTeam", {}).get("name")

        home_summary = await self._get_understat_team_summary(home_team_name or "", league_code, season)
        away_summary = await self._get_understat_team_summary(away_team_name or "", league_code, season)

        if not home_summary and not away_summary:
            return None

        context: Dict[str, Any] = {
            "league_code": league_code,
            "season": season,
            "home": home_summary,
            "away": away_summary,
        }

        if home_summary and away_summary:
            home_xg = home_summary.get("xg_per_match")
            away_xg = away_summary.get("xg_per_match")
            if isinstance(home_xg, (int, float)) and isinstance(away_xg, (int, float)):
                context["xg_gap"] = round(home_xg - away_xg, 3)

        return context
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Get data from cache if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached data or None if not found or expired
        """
        if cache_key in self.cache:
            # Check if cache is expired
            if cache_key in self.cache_expiry:
                if datetime.now() < self.cache_expiry[cache_key]:
                    # Track cache hit
                    self.performance_metrics["cache_hits"] += 1
                    return self.cache[cache_key]
                else:
                    # Cache expired, remove it
                    del self.cache[cache_key]
                    del self.cache_expiry[cache_key]
        
        # Track cache miss
        self.performance_metrics["cache_misses"] += 1
        return None
    
    def _store_in_cache(self, cache_key: str, data: Any, duration: int = None) -> None:
        """
        Store data in cache with expiry.
        
        Args:
            cache_key: Cache key
            data: Data to store
            duration: Cache duration in seconds (default: self.default_cache_duration)
        """
        if duration is None:
            duration = self.default_cache_duration
        
        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = datetime.now() + timedelta(seconds=duration)
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.cache_expiry.clear()
        logger.info("Cache cleared")
    
    async def get_upcoming_matches(
        self,
        league: str,
        days_ahead: int = 7,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get upcoming matches for a specific league with optimized data flow.

        Args:
            league: League name or code
            days_ahead: Number of days ahead to fetch matches
            use_cache: Whether to use cached data
            force_refresh: Force refresh data from source even if cached

        Returns:
            DataFrame with upcoming matches
        """
        # Track performance and increment request counter
        operation_id = self.system_monitor.start_operation("get_upcoming_matches")
        self.performance_metrics["requests"] += 1

        try:
            # Convert league name to code if needed
            league_code = self.LEAGUE_MAPPING.get(league, league)

            # Check cache first (unless force refresh)
            if use_cache and not force_refresh:
                cache_key = self._get_cache_key("get_upcoming_matches", league_code, days_ahead)
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.info(f"Using cached upcoming matches for {league_code}")
                    self.system_monitor.end_operation(operation_id)
                    return cached_data

            # Initialize scraper (prefer injected test double if present)
            football_data_scraper = None
            if hasattr(self, "_scrapers"):
                football_data_scraper = self._scrapers.get("football_data")
            if not football_data_scraper:
                football_data_scraper = get_scraper("football_data")

            if not football_data_scraper:
                logger.error("Failed to initialize Football-Data scraper")
                self.system_monitor.end_operation(operation_id, status="failure")
                return pd.DataFrame()

            # Prefer a dedicated get_upcoming_matches method if the scraper offers one
            if hasattr(football_data_scraper, "get_upcoming_matches"):
                try:
                    matches_df = await football_data_scraper.get_upcoming_matches(league_code=league_code, days_ahead=days_ahead)
                except Exception as e:
                    logger.error(f"Scraper get_upcoming_matches failed for {league_code}: {e}")
                    matches_df = pd.DataFrame()
            else:
                # Fallback: derive using get_matches over the date window
                date_from = datetime.now()
                date_to = date_from + timedelta(days=days_ahead)
                try:
                    matches_df = await football_data_scraper.get_matches(
                        league_code=league_code,
                        date_from=date_from,
                        date_to=date_to,
                        status="SCHEDULED"
                    )
                except Exception as e:
                    logger.error(f"Scraper get_matches fallback failed for {league_code}: {e}")
                    matches_df = pd.DataFrame()

            if matches_df is None or (isinstance(matches_df, pd.DataFrame) and matches_df.empty):
                logger.warning(f"No upcoming matches found for {league_code}")
                self.system_monitor.end_operation(operation_id, status="warning")
                return pd.DataFrame()

            # Process and normalize data
            matches_df = self._process_matches_dataframe(matches_df)

            # Cache the result
            if use_cache:
                cache_key = self._get_cache_key("get_upcoming_matches", league_code, days_ahead)
                self._store_in_cache(cache_key, matches_df, duration=600)  # 10 minutes

            self.system_monitor.end_operation(operation_id)
            return matches_df

        except Exception as e:
            logger.error(f"Error getting upcoming matches for {league}: {e}")
            self.performance_metrics["errors"] += 1
            self.system_monitor.end_operation(operation_id, status="failure")
            return pd.DataFrame()
    
    async def get_match_predictions(
        self,
        match_id: Union[str, int],
        use_cache: bool = True,
        include_features: bool = True,
        xg_model: str = "default"
    ) -> Dict[str, Any]:
        """
        Get predictions for a specific match.
        
        Args:
            match_id: Match ID
            use_cache: Whether to use cached data
            include_features: Whether to include feature data used in prediction
            xg_model: Which xG model to use ("default", "advanced", or "ensemble")
            
        Returns:
            Dictionary with match predictions
        """
        # Track performance
        operation_id = self.system_monitor.start_operation("get_match_predictions")
        self.performance_metrics["requests"] += 1
        # Check cache
        cache_key = self._get_cache_key("get_match_predictions", match_id)
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached predictions for match {match_id}")
                return cached_data
        
        # Initialize Football-Data scraper (Understat handled via API client)
        football_data_scraper = None
        if hasattr(self, "_scrapers"):
            football_data_scraper = self._scrapers.get("football_data")
        if not football_data_scraper:
            football_data_scraper = get_scraper("football_data")

        if not football_data_scraper:
            logger.error("Failed to initialize Football-Data scraper for match predictions")
            return {}

        understat_available = self.understat_enabled and self.understat_client is not None
        if not understat_available:
            logger.debug("Understat API unavailable; proceeding with limited prediction context")

        # Fetch match details early so downstream consumers have richer metadata
        match_details = await self.get_match_details(match_id)
        
        try:
            # Start performance monitoring if available
            if self.system_monitor:
                self.system_monitor.start_operation("get_match_predictions")
            
            # Fetch match odds from Football-Data
            odds_data = await football_data_scraper.get_match_odds(match_id)
            
            # Initialize predictions dictionary
            predictions = {
                "match_id": str(match_id),  # Ensure match_id is string
                "bookmaker_odds": {},
                "model_predictions": {},
                "value_bets": {},
                "insights": [],
                "home_win_probability": 0.0,
                "draw_probability": 0.0,
                "away_win_probability": 0.0
            }
            
            # Process odds data if available
            if odds_data and "bookmakers" in odds_data:
                # Extract odds from the first bookmaker with 1X2 market
                for bookmaker in odds_data["bookmakers"]:
                    for market in bookmaker["markets"]:
                        if market["name"] == "WINNER":  # 1X2 market
                            predictions["bookmaker_odds"][bookmaker["name"]] = {
                                outcome["name"]: outcome["price"]
                                for outcome in market["outcomes"]
                            }
                            break
                    
                    # Break after finding the first bookmaker with 1X2 market
                    if predictions["bookmaker_odds"]:
                        break
            
            # TODO: Implement model predictions using ML models
            # For now, use dummy predictions based on odds
            if predictions["bookmaker_odds"]:
                first_bookmaker = list(predictions["bookmaker_odds"].keys())[0]
                odds = predictions["bookmaker_odds"][first_bookmaker]
                
                # Convert odds to probabilities
                total_prob = sum(1/float(odd) for odd in odds.values())
                model_predictions = {
                    outcome: round((1/float(odd))/total_prob, 2)
                    for outcome, odd in odds.items()
                }
                predictions["model_predictions"] = model_predictions
                
                # Set standardized probability keys
                predictions["home_win_probability"] = model_predictions.get("HOME_TEAM", model_predictions.get("1", 0.0))
                predictions["draw_probability"] = model_predictions.get("DRAW", model_predictions.get("X", 0.0)) 
                predictions["away_win_probability"] = model_predictions.get("AWAY_TEAM", model_predictions.get("2", 0.0))
                
                # Generate value bets
                predictions["value_bets"] = {
                    outcome: {
                        "odds": float(odd),
                        "fair_odds": round(1/predictions["model_predictions"][outcome], 2),
                        "value": round((float(odd) * predictions["model_predictions"][outcome]) - 1, 2)
                    }
                    for outcome, odd in odds.items()
                }
                
                # Generate insights
                for outcome, value_bet in predictions["value_bets"].items():
                    if value_bet["value"] > 0.1:  # 10% value threshold
                        predictions["insights"].append(
                            f"Value bet on {outcome}: Odds of {value_bet['odds']} offer {value_bet['value']*100:.1f}% value"
                        )
            
            # Add Understat context when available
            understat_context = await self._build_understat_context(match_details)
            if understat_context:
                predictions["understat_context"] = understat_context
                xg_gap = understat_context.get("xg_gap")
                if isinstance(xg_gap, (int, float)):
                    predictions.setdefault("model_predictions", {})["understat_xg_gap"] = xg_gap

            # Remove features if not requested to reduce payload size
            if not include_features and "features" in predictions:
                del predictions["features"]
                
            # Add model version information
            predictions["model_info"] = {
                "xg_model": xg_model,
                "version": "1.0.4",
                "generated_at": datetime.now().isoformat(),
                "includes_features": include_features
            }
                
            # Cache the result
            self._store_in_cache(cache_key, predictions, duration=1800)  # 30 minutes
            
            # Track successful completion
            self.system_monitor.end_operation(operation_id)
            return predictions
        
        except Exception as e:
            logger.error(f"Error generating predictions for match {match_id}: {e}")
            self.performance_metrics["errors"] += 1
            self.system_monitor.end_operation(operation_id, status="failure")
            return {}
        
        finally:
            # End performance monitoring if available
            if self.system_monitor:
                self.system_monitor.end_operation("get_match_predictions")
    
    async def get_team_form(
        self,
        team_id: Union[str, int],
        matches_count: int = 5,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        # Track performance
        operation_id = self.system_monitor.start_operation("get_team_form")
        self.performance_metrics["requests"] += 1
        """
        Get recent form for a specific team.
        
        Args:
            team_id: Team ID
            matches_count: Number of recent matches to include
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with team form data
        """
        # Check cache
        cache_key = self._get_cache_key("get_team_form", team_id, matches_count)
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached form data for team {team_id}")
                return cached_data
        
        # Initialize scrapers
        football_data_scraper = get_scraper("football_data")
        
        if not football_data_scraper:
            logger.error("Failed to initialize Football-Data scraper")
            return {}
        
        try:
            # Start performance monitoring if available
            if self.system_monitor:
                self.system_monitor.start_operation("get_team_form")
            
            # Fetch recent matches from Football-Data
            matches_df = await football_data_scraper.get_team_matches(
                team_id=team_id,
                status="FINISHED",
                limit=matches_count
            )
            
            if matches_df is None or matches_df.empty:
                logger.warning(f"No recent matches found for team {team_id}")
                return {}
            
            # Calculate form metrics
            form_data = {
                "team_id": team_id,
                "matches": [],
                "summary": {
                    "played": len(matches_df),
                    "won": 0,
                    "drawn": 0,
                    "lost": 0,
                    "goals_for": 0,
                    "goals_against": 0,
                    "points": 0,
                    "form_string": ""
                }
            }
            
            # Process each match
            for _, match in matches_df.iterrows():
                # Skip matches without scores
                if pd.isna(match.get("home_score")) or pd.isna(match.get("away_score")):
                    continue
                
                # Determine result from team's perspective
                is_home = match.get("is_home", match.get("home_team_id") == team_id)
                team_score = match["home_score"] if is_home else match["away_score"]
                opponent_score = match["away_score"] if is_home else match["home_score"]
                
                if team_score > opponent_score:
                    result = "W"
                    points = 3
                    form_data["summary"]["won"] += 1
                elif team_score == opponent_score:
                    result = "D"
                    points = 1
                    form_data["summary"]["drawn"] += 1
                else:
                    result = "L"
                    points = 0
                    form_data["summary"]["lost"] += 1
                
                # Update summary stats
                form_data["summary"]["goals_for"] += team_score
                form_data["summary"]["goals_against"] += opponent_score
                form_data["summary"]["points"] += points
                form_data["summary"]["form_string"] = result + form_data["summary"]["form_string"]
                
                # Add match details
                form_data["matches"].append({
                    "match_id": match.get("id"),
                    "date": match.get("match_date").strftime("%Y-%m-%d") if not pd.isna(match.get("match_date")) else "",
                    "opponent": match.get("away_team") if is_home else match.get("home_team"),
                    "is_home": is_home,
                    "team_score": team_score,
                    "opponent_score": opponent_score,
                    "result": result
                })
            
            # Cache the result
            self._store_in_cache(cache_key, form_data, duration=3600)  # 1 hour
            
            return form_data
        
        except Exception as e:
            logger.error(f"Error getting form data for team {team_id}: {e}")
            return {}
        
        finally:
            # End performance monitoring if available
            if self.system_monitor:
                self.system_monitor.end_operation("get_team_form")
    
    async def get_league_standings(
        self,
        league: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get current standings for a specific league.
        
        Args:
            league: League name or code
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with league standings
        """
        # Track performance
        operation_id = self.system_monitor.start_operation("get_league_standings") if self.system_monitor else None
        self.performance_metrics["requests"] += 1
        
        try:
            # Convert league name to code if needed
            league_code = self.LEAGUE_MAPPING.get(league, league)
            
            # Check cache
            cache_key = self._get_cache_key("get_league_standings", league_code)
            if use_cache:
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    logger.info(f"Using cached standings for {league_code}")
                    return cached_data
            
            # Initialize scrapers
            # Use instance _scrapers if available (for tests), else factory
            football_data_scraper = self._scrapers.get("football_data") or get_scraper("football_data")
            if not football_data_scraper:
                logger.error("Failed to initialize Football-Data scraper")
                return pd.DataFrame()
            
            # Fetch standings from Football-Data
            standings_df = await football_data_scraper.get_league_standings(league_code)
            
            if standings_df is None or standings_df.empty:
                logger.warning(f"No standings found for {league_code}")
                return pd.DataFrame()
            
            # Add league display name
            standings_df["league_display"] = self.DISPLAY_LEAGUE_MAPPING.get(league_code, league_code)
            
            # Cache the result
            self._store_in_cache(cache_key, standings_df, duration=3600)  # 1 hour
            
            return standings_df
            
        except Exception as e:
            logger.error(f"Error getting standings for {league_code}: {e}")
            self.performance_metrics["errors"] += 1
            if self.system_monitor and operation_id:
                self.system_monitor.end_operation(operation_id, status="failure")
            return pd.DataFrame()
            
        finally:
            # End performance monitoring if available
            if self.system_monitor and operation_id:
                self.system_monitor.end_operation(operation_id)
    
    async def get_match_details(
        self,
        match_id: Union[str, int],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed information for a specific match.
        
        Args:
            match_id: Match ID
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with match details
        """
        # Track performance
        operation_id = self.system_monitor.start_operation("get_match_details") if self.system_monitor else None
        self.performance_metrics["requests"] += 1
        
        # Check cache
        cache_key = self._get_cache_key("get_match_details", match_id)
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached details for match {match_id}")
                if self.system_monitor and operation_id:
                    self.system_monitor.end_operation(operation_id)
                return cached_data
        
        try:
            # Get scraper for match details
            football_data_scraper = self._scrapers.get("football_data")
            
            if not football_data_scraper:
                logger.error("Failed to access Football-Data scraper")
                self.system_monitor.end_operation(operation_id, status="failure")
                return {}
                
            # We already started operation tracking at the beginning of this method
            # No need to start it again here
            
            # Fetch match details from Football-Data
            match_data = await football_data_scraper.get_match_details(match_id)
            
            if not match_data or not isinstance(match_data, dict):
                logger.warning(f"No details found for match {match_id}")
                # Return basic structure with match_id even when no data is found
                return {"match_id": str(match_id), "error": "No match data found"}
            
            # Process match data
            match_details = {
                "match_id": str(match_id),  # Ensure match_id is string
                "competition": match_data.get("competition", {}).get("name", ""),
                "season": match_data.get("season", {}).get("startDate", "")[:4],
                "match_date": match_data.get("utcDate", ""),
                "status": match_data.get("status", ""),
                "matchday": match_data.get("matchday", ""),
                "stage": match_data.get("stage", ""),
                "group": match_data.get("group", ""),
                "home_team": {
                    "id": match_data.get("homeTeam", {}).get("id", ""),
                    "name": match_data.get("homeTeam", {}).get("name", ""),
                    "short_name": match_data.get("homeTeam", {}).get("shortName", ""),
                    "tla": match_data.get("homeTeam", {}).get("tla", ""),
                    "crest": match_data.get("homeTeam", {}).get("crest", "")
                },
                "away_team": {
                    "id": match_data.get("awayTeam", {}).get("id", ""),
                    "name": match_data.get("awayTeam", {}).get("name", ""),
                    "short_name": match_data.get("awayTeam", {}).get("shortName", ""),
                    "tla": match_data.get("awayTeam", {}).get("tla", ""),
                    "crest": match_data.get("awayTeam", {}).get("crest", "")
                },
                "score": {
                    "winner": match_data.get("score", {}).get("winner", ""),
                    "duration": match_data.get("score", {}).get("duration", ""),
                    "full_time": {
                        "home": match_data.get("score", {}).get("fullTime", {}).get("home", None),
                        "away": match_data.get("score", {}).get("fullTime", {}).get("away", None)
                    },
                    "half_time": {
                        "home": match_data.get("score", {}).get("halfTime", {}).get("home", None),
                        "away": match_data.get("score", {}).get("halfTime", {}).get("away", None)
                    }
                },
                "referees": [
                    {
                        "id": referee.get("id", ""),
                        "name": referee.get("name", ""),
                        "type": referee.get("type", ""),
                        "nationality": referee.get("nationality", "")
                    }
                    for referee in match_data.get("referees", [])
                ],
                "venue": match_data.get("venue", ""),
                "odds": {}
            }
            
            # Add odds if available
            if "odds" in match_data:
                for bookmaker in match_data["odds"].get("bookmakers", []):
                    bookmaker_name = bookmaker.get("name", "")
                    match_details["odds"][bookmaker_name] = {}
                    
                    for market in bookmaker.get("markets", []):
                        market_name = market.get("name", "")
                        match_details["odds"][bookmaker_name][market_name] = {
                            outcome.get("name", ""): outcome.get("price", 0)
                            for outcome in market.get("outcomes", [])
                        }
            
            # Cache the result
            self._store_in_cache(cache_key, match_details, duration=3600)  # 1 hour
            
            # End performance tracking
            self.system_monitor.end_operation(operation_id)
            return match_details
        
        except Exception as e:
            logger.error(f"Error getting details for match {match_id}: {e}")
            return {}
        
        finally:
            # This is handled by the initial operation tracking
            pass
    
    async def get_team_stats(
        self,
        team_id: Union[str, int],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific team.
        
        Args:
            team_id: Team ID
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with team statistics
        """
        # Check cache
        cache_key = self._get_cache_key("get_team_stats", team_id)
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Using cached stats for team {team_id}")
                return cached_data
        
        # Initialize scrapers
        football_data_scraper = get_scraper("football_data")
        
        if not football_data_scraper:
            logger.error("Failed to initialize Football-Data scraper")
            return {}
        
        try:
            # Start performance monitoring if available
            if self.system_monitor:
                self.system_monitor.start_operation("get_team_stats")
            
            # Fetch team info from Football-Data
            team_info = await football_data_scraper.get_team_info(team_id)
            
            if not team_info:
                logger.warning(f"No info found for team {team_id}")
                return {}
            
            # Fetch recent matches
            matches_df = await football_data_scraper.get_team_matches(
                team_id=team_id,
                status="FINISHED",
                limit=10
            )
            
            # Process team statistics
            team_stats = {
                "team_id": team_id,
                "name": team_info.get("name", ""),
                "short_name": team_info.get("short_name", ""),
                "tla": team_info.get("tla", ""),
                "crest": team_info.get("crest", ""),
                "address": team_info.get("address", ""),
                "website": team_info.get("website", ""),
                "founded": team_info.get("founded", ""),
                "club_colors": team_info.get("club_colors", ""),
                "venue": team_info.get("venue", ""),
                "squad_size": team_info.get("squad_size", 0),
                "form": {
                    "recent_matches": [],
                    "summary": {
                        "played": 0,
                        "won": 0,
                        "drawn": 0,
                        "lost": 0,
                        "goals_for": 0,
                        "goals_against": 0,
                        "clean_sheets": 0,
                        "failed_to_score": 0
                    }
                }
            }
            
            # Process recent matches if available
            if matches_df is not None and not matches_df.empty:
                # Calculate form metrics
                for _, match in matches_df.iterrows():
                    # Skip matches without scores
                    if pd.isna(match.get("home_score")) or pd.isna(match.get("away_score")):
                        continue
                    
                    # Determine result from team's perspective
                    is_home = match.get("is_home", match.get("home_team_id") == team_id)
                    team_score = match["home_score"] if is_home else match["away_score"]
                    opponent_score = match["away_score"] if is_home else match["home_score"]
                    
                    if team_score > opponent_score:
                        result = "W"
                        team_stats["form"]["summary"]["won"] += 1
                    elif team_score == opponent_score:
                        result = "D"
                        team_stats["form"]["summary"]["drawn"] += 1
                    else:
                        result = "L"
                        team_stats["form"]["summary"]["lost"] += 1
                    
                    # Update summary stats
                    team_stats["form"]["summary"]["played"] += 1
                    team_stats["form"]["summary"]["goals_for"] += team_score
                    team_stats["form"]["summary"]["goals_against"] += opponent_score
                    
                    if opponent_score == 0:
                        team_stats["form"]["summary"]["clean_sheets"] += 1
                    
                    if team_score == 0:
                        team_stats["form"]["summary"]["failed_to_score"] += 1
                    
                    # Add match details
                    team_stats["form"]["recent_matches"].append({
                        "match_id": match.get("id"),
                        "date": match.get("match_date").strftime("%Y-%m-%d") if not pd.isna(match.get("match_date")) else "",
                        "opponent": match.get("away_team") if is_home else match.get("home_team"),
                        "is_home": is_home,
                        "team_score": team_score,
                        "opponent_score": opponent_score,
                        "result": result
                    })
            
            # Add squad information if available
            if "squad" in team_info:
                team_stats["squad"] = team_info["squad"]
            
            # Enrich with Understat metrics if available
            league_code = None
            season = None
            if matches_df is not None and not matches_df.empty:
                league_code = self._resolve_league_code(matches_df.iloc[0].get("competition"))
                season = self._infer_understat_season(matches_df.iloc[0].get("match_date"))

            if league_code and season and team_stats.get("name"):
                understat_summary = await self._get_understat_team_summary(team_stats["name"], league_code, season)
                if understat_summary:
                    team_stats["advanced_metrics"] = understat_summary

            # Cache the result
            self._store_in_cache(cache_key, team_stats, duration=3600)  # 1 hour
            
            return team_stats
        
        except Exception as e:
            logger.error(f"Error getting stats for team {team_id}: {e}")
            return {}
        
        finally:
            # End performance monitoring if available
            if self.system_monitor:
                self.system_monitor.end_operation("get_team_stats")
    
    def get_scraper_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all scrapers with detailed information.
        
        Returns:
            Dictionary mapping scraper names to their status details
        """
        operation_id = self.system_monitor.start_operation("get_scraper_status")
        
        try:
            result = {}
            logger.debug(f"Getting status for {len(self._scrapers)} scrapers: {list(self._scrapers.keys())}")
            
            for name, scraper in self._scrapers.items():
                operation_key = f"{name}_fetch_html"
                # Check if scraper is UnderstatAPIClient using type name check
                is_understat = scraper.__class__.__name__ == 'UnderstatAPIClient'
                if is_understat:
                    operation_key = f"{name}_api_request"

                scraper_stats = self.system_monitor.get_operation_stats(operation_key)
                stats_payload: Dict[str, Any] = {}
                if scraper_stats:
                    total_calls = max(scraper_stats.get("count", 0), 1)
                    success_count = scraper_stats.get("success_count", 0)
                    stats_payload = {
                        "success_rate": success_count / total_calls if total_calls else 0,
                        "avg_response_time": scraper_stats.get("avg_duration", 0),
                        "requests_count": scraper_stats.get("count", 0),
                    }

                info: Dict[str, Any] = {
                    "available": True,
                    "last_request": getattr(scraper, "last_request_time", None),
                    "stats": stats_payload,
                }

                if hasattr(scraper, "min_delay") and hasattr(scraper, "max_delay"):
                    info["rate_limit"] = f"{getattr(scraper, 'min_delay')}-{getattr(scraper, 'max_delay')}s"

                if is_understat:
                    info.update({
                        "type": "api_client",
                        "base_url": getattr(scraper, "base_url", "N/A"),
                        "max_retries": getattr(scraper, "max_retries", 0),
                        "cache_enabled": getattr(scraper, "use_cache", False),
                        "rate_limit_remaining": getattr(scraper, "rate_limit_remaining", None),
                        "rate_limit_reset": getattr(scraper, "rate_limit_reset", None),
                    })

                result[name] = info
            
            logger.debug(f"Returning status for {len(result)} scrapers: {list(result.keys())}")
                
            # Add system status
            self.system_monitor.end_operation(operation_id)
            return result
        except Exception as e:
            logger.error(f"Error getting scraper status: {e}")
            self.system_monitor.end_operation(operation_id, status="failure")
            return {scraper: {"available": False, "error": str(e)} for scraper in self._scrapers}

    def get_available_leagues(self) -> List[str]:
        """Get a list of available league display names."""
        return list(self.DISPLAY_LEAGUE_MAPPING.keys())

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the internal cache and performance metrics.
        
        Returns detailed information about cache usage, hit rates, memory consumption,
        and operation performance. Useful for monitoring and optimizing the data
        integration system.
        
        Returns:
            Dict containing:
                - cached_items: Number of items currently in cache
                - memory_usage_bytes: Estimated memory usage in bytes
                - memory_usage_mb: Memory usage in megabytes
                - hit_rate: Cache hit rate as percentage
                - cache_hits: Total cache hits
                - cache_misses: Total cache misses
                - total_requests: Total API requests made
                - errors: Total error count
                - expiring_soon: Number of items expiring within 5 minutes
                - operation_stats: Detailed performance statistics per operation
                - cache_types: Breakdown of cached content types
        """
        operation_id = self.system_monitor.start_operation("get_cache_stats")
        
        try:
            # Calculate cache hit rate
            total_requests = self.performance_metrics["cache_hits"] + self.performance_metrics["cache_misses"]
            hit_rate = (self.performance_metrics["cache_hits"] / max(total_requests, 1)) * 100 if total_requests > 0 else 0.0
            
            # Get expiring soon items
            now = datetime.now()
            expiring_soon = [k for k, v in self.cache_expiry.items() if (v - now).total_seconds() < 300]  # Items expiring in next 5 minutes
            
            # Get memory usage estimate (rough approximation)
            import sys
            memory_usage = sum(sys.getsizeof(v) for v in self.cache.values())
            
            result = {
                "cached_items": len(self.cache),
                "memory_usage_bytes": memory_usage,
                "memory_usage_mb": round(memory_usage / (1024 * 1024), 2),
                "hit_rate": (self.performance_metrics["cache_hits"] / max(total_requests, 1)) * 100 if total_requests > 0 else 0.0,
                "cache_hits": self.performance_metrics["cache_hits"],
                "cache_misses": self.performance_metrics["cache_misses"],
                "total_requests": self.performance_metrics["requests"],
                "errors": self.performance_metrics["errors"],
                "expiring_soon": len(expiring_soon),
                "operation_stats": self.system_monitor.get_operation_stats(),
                "cache_types": self._get_cache_content_types()
            }
            
            self.system_monitor.end_operation(operation_id)
            return result
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            self.system_monitor.end_operation(operation_id, status="failure")
            return {
                "error": str(e),
                "cached_items": len(self.cache),
                "cache_keys": list(self.cache.keys()),
                "message": "Error getting detailed cache statistics."
            }
    
    def _get_cache_content_types(self) -> Dict[str, int]:
        """Analyze types of content in cache."""
        types = {}
        for k, v in self.cache.items():
            content_type = type(v).__name__
            types[content_type] = types.get(content_type, 0) + 1
        return types
    
    def _process_matches_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and normalize matches DataFrame.
        
        Args:
            df: Raw matches DataFrame
            
        Returns:
            Processed DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame()
        
        try:
            # Ensure match_date is datetime
            if 'match_date' in df.columns:
                df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
            
            # Add derived columns
            if 'match_date' in df.columns:
                # Handle timezone-aware vs timezone-naive datetime comparison
                now = pd.Timestamp.now()
                if hasattr(df['match_date'].dt, 'tz') and df['match_date'].dt.tz is not None:
                    # If match_date is timezone-aware, localize now to UTC
                    if now.tzinfo is None:
                        now = now.tz_localize('UTC')
                else:
                    # If match_date is naive, ensure now is naive
                    if now.tzinfo is not None:
                        now = now.tz_convert(None)
                df['days_until_match'] = (df['match_date'] - now).dt.days
            
            # Sort by match date
            if 'match_date' in df.columns:
                df = df.sort_values('match_date')
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error processing matches DataFrame: {e}")
            return df
        
    async def cleanup(self) -> None:
        """Cleanup resources like HTTP connections."""
        if self.http_client:
            await self.http_client.close()
        if self.understat_client and hasattr(self.understat_client, "close"):
            try:
                await self.understat_client.close()
            except Exception as exc:
                logger.debug("Error closing Understat client: %s", exc)
        logger.info("DataIntegration resources cleaned up")


# Create a singleton instance for easy access
data_integration = DataIntegration()

# Convenience functions for direct use without creating an integration instance

async def get_upcoming_matches(league: str, days_ahead: int = 7, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
    """Get upcoming matches for a specific league."""
    return await data_integration.get_upcoming_matches(league, days_ahead, use_cache, force_refresh)

async def get_match_predictions(match_id: Union[str, int], use_cache: bool = True, include_features: bool = True, xg_model: str = "default") -> Dict[str, Any]:
    """Get predictions for a specific match."""
    return await data_integration.get_match_predictions(match_id, use_cache, include_features, xg_model)

async def get_team_form(team_id: Union[str, int], matches_count: int = 5, use_cache: bool = True) -> Dict[str, Any]:
    """Get recent form for a specific team."""
    return await data_integration.get_team_form(team_id, matches_count, use_cache)

async def get_league_standings(league: str, use_cache: bool = True) -> pd.DataFrame:
    """Get current standings for a specific league."""
    return await data_integration.get_league_standings(league, use_cache)

async def get_match_details(match_id: Union[str, int], use_cache: bool = True) -> Dict[str, Any]:
    """Get detailed information for a specific match."""
    return await data_integration.get_match_details(match_id, use_cache)

async def get_team_stats(team_id: Union[str, int], use_cache: bool = True) -> Dict[str, Any]:
    """Get detailed statistics for a specific team."""
    return await data_integration.get_team_stats(team_id, use_cache)

def get_scraper_status() -> Dict[str, Dict[str, Any]]:
    """Get the status of all scrapers with detailed information."""
    return data_integration.get_scraper_status()

def clear_cache() -> None:
    """Clear all cached data."""
    data_integration.clear_cache()
    
async def cleanup() -> None:
    """Clean up resources."""
    await data_integration.cleanup()
