"""Understat API Integration Client.

Provides a lightweight asynchronous client for interacting with the
Understat API so advanced expected-goals (xG) analytics can be surfaced
throughout the GoalDiggers platform.
"""

import asyncio
import copy
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from aiohttp import ClientConnectionError, ClientError, ContentTypeError

from database.db_manager import DatabaseManager
from utils.http_client_async import HttpClientAsync

# Optional import for API keys manager
try:
    from utils.api_keys_manager import APIKeysManager
except ImportError:
    APIKeysManager = None

logger = logging.getLogger(__name__)

class UnderstatAPIClient:
    """
    Client for Understat API integration.
    
    This client provides methods to fetch xG data, player statistics,
    and team performance metrics from Understat.
    """
    
    LEAGUE_MAPPING = {
        "premier_league": "EPL",
        "la_liga": "La_liga", 
        "bundesliga": "Bundesliga",
        "serie_a": "Serie_A",
        "ligue_1": "Ligue_1",
        "eredivisie": "Eredivisie"
    }
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 db_manager: Optional[DatabaseManager] = None,
                 http_client: Optional[HttpClientAsync] = None,
                 api_keys_manager: Optional["APIKeysManager"] = None,
                 base_url: str = "https://api.understat.com/v1"):
        """
        Initialize the Understat API client.
        
        Args:
            api_key: Understat API key (optional, can be loaded from environment)
            db_manager: Database manager for caching (optional)
            http_client: Asynchronous HTTP client for API requests (optional)
            api_keys_manager: API keys manager for retrieving keys (optional)
            base_url: Understat API base URL
        """
        # Initialize API key - try from parameter, then environment, then manager
        self.api_key = api_key or os.getenv("UNDERSTAT_API_KEY")
        self.api_keys_manager = api_keys_manager
        
        if not self.api_key and self.api_keys_manager:
            self.api_key = self.api_keys_manager.get_api_key("understat")
            
        self.base_url = base_url.rstrip("/")
        self.db_manager = db_manager

        # Initialize HTTP client or use the provided one
        self._owns_http_client = http_client is None
        self.http_client = http_client or HttpClientAsync()

        # Track API rate limiting
        self.request_count = 0
        self.last_request_time: Optional[datetime] = None
        self.rate_limit_reset: Optional[int] = None
        self.rate_limit_remaining: Optional[int] = None
        self.max_retries = max(1, int(os.getenv("UNDERSTAT_MAX_RETRIES", "3")))
        self.rate_limit_backoff = max(1, int(os.getenv("UNDERSTAT_RATE_LIMIT_BACKOFF", "60")))

        # Set up cache configuration
        self.use_cache = True
        self.cache_expiry = 3600  # 1 hour default cache expiry
        self._memory_cache_max_entries = max(1, int(os.getenv("UNDERSTAT_CACHE_MAX_ENTRIES", "128")))
        self._memory_cache: "OrderedDict[str, Tuple[datetime, Any]]" = OrderedDict()

        logger.info(
            "Initialized Understat API client with API key: %s",
            "Available" if self.api_key else "Not Available",
        )

    @classmethod
    def _normalize_league_code(cls, league_code: str) -> Optional[str]:
        """Translate platform league identifiers to Understat codes."""
        if not league_code:
            return None
        normalized = league_code.strip().lower()
        return cls.LEAGUE_MAPPING.get(normalized)

    @staticmethod
    def _default_season(seed_date: Optional[datetime] = None) -> str:
        """Infer the most likely Understat season string for the given date."""
        reference = seed_date or datetime.utcnow()
        # Understat seasons typically start mid-year; before July is prior season
        if reference.month < 7:
            return str(reference.year - 1)
        return str(reference.year)

    @staticmethod
    def _normalize_team_name(name: str) -> str:
        """Normalize team names for lookup consistency."""
        return " ".join(
            (name or "")
            .lower()
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make an Understat API request with retries and caching."""

        normalized_endpoint = endpoint.lstrip("/")
        url = f"{self.base_url}/{normalized_endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        cache_key, params_json = self._build_cache_tokens(normalized_endpoint, params)

        if self.use_cache:
            cached = await self._get_cached_response(cache_key, normalized_endpoint, params_json)
            if cached is not None:
                logger.debug("Understat cache hit for %s", url)
                return cached

        backoff_delay = 1
        for attempt in range(1, self.max_retries + 1):
            self.request_count += 1
            self.last_request_time = datetime.utcnow()

            data, response_headers, retry_after = await self._make_http_request(
                url=url,
                headers=headers,
                params=params,
            )

            if data is not None:
                self._update_rate_limit(response_headers)
                if self.use_cache:
                    self._store_in_memory_cache(cache_key, data)
                    if self.db_manager:
                        await asyncio.to_thread(
                            self._store_db_cache_sync,
                            normalized_endpoint,
                            params_json,
                            data,
                        )
                return data

            self._update_rate_limit(response_headers)

            if attempt == self.max_retries:
                break

            sleep_for = retry_after or min(backoff_delay, self.rate_limit_backoff)
            logger.debug(
                "Retrying Understat request '%s' in %s seconds (attempt %s/%s)",
                normalized_endpoint,
                sleep_for,
                attempt + 1,
                self.max_retries,
            )
            await asyncio.sleep(sleep_for)
            backoff_delay = min(backoff_delay * 2, self.rate_limit_backoff)

        logger.warning(
            "Failed to retrieve Understat data for endpoint '%s' after %s attempts",
            normalized_endpoint,
            self.max_retries,
        )
        return None

    async def _make_http_request(
        self,
        url: str,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, str], Optional[int]]:
        """Perform a single HTTP request to the Understat API."""

        try:
            session = await self.http_client.get_session()
            async with session.get(url, headers=headers, params=params) as response:
                headers_map = dict(response.headers)

                if response.status == 429:
                    retry_after = int(headers_map.get("Retry-After", str(self.rate_limit_backoff)))
                    logger.warning(
                        "Understat API rate limit exceeded for %s. Retrying in %s seconds",
                        url,
                        retry_after,
                    )
                    return None, headers_map, retry_after

                if response.status >= 400:
                    body = await response.text()
                    logger.error(
                        "Understat API error %s for %s: %s",
                        response.status,
                        url,
                        body[:500],
                    )
                    return None, headers_map, None

                try:
                    payload = await response.json()
                except ContentTypeError:
                    body = await response.text()
                    logger.error(
                        "Unexpected content type received from Understat API for %s: %s",
                        url,
                        body[:500],
                    )
                    return None, headers_map, None

                return payload, headers_map, None

        except (ClientConnectionError, ClientError) as exc:
            logger.error("Understat API request error for %s: %s", url, exc)
            return None, {}, None
        except asyncio.TimeoutError:
            logger.error("Understat API request timed out for %s", url)
            return None, {}, None
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unexpected error during Understat API request to %s", url)
            return None, {}, None

    def _build_cache_tokens(
        self, endpoint: str, params: Optional[Dict[str, Any]]
    ) -> Tuple[str, str]:
        params_json = json.dumps(params or {}, sort_keys=True, default=str)
        cache_key = f"{endpoint}|{params_json}"
        return cache_key, params_json

    async def _get_cached_response(
        self, cache_key: str, endpoint: str, params_json: str
    ) -> Optional[Dict[str, Any]]:
        cached = self._get_from_memory_cache(cache_key)
        if cached is not None:
            return cached

        if not self.db_manager:
            return None

        db_result = await asyncio.to_thread(
            self._get_from_db_cache_sync,
            endpoint,
            params_json,
        )

        if db_result is not None:
            self._store_in_memory_cache(cache_key, db_result)

        return db_result

    def _get_from_memory_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        entry = self._memory_cache.get(cache_key)
        if not entry:
            return None

        expires_at, payload = entry
        if expires_at <= datetime.utcnow():
            self._memory_cache.pop(cache_key, None)
            return None

        self._memory_cache.move_to_end(cache_key, last=True)
        return copy.deepcopy(payload)

    def _store_in_memory_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        expires_at = datetime.utcnow() + timedelta(seconds=self.cache_expiry)
        self._memory_cache[cache_key] = (expires_at, copy.deepcopy(data))
        self._memory_cache.move_to_end(cache_key, last=True)

        while len(self._memory_cache) > self._memory_cache_max_entries:
            self._memory_cache.popitem(last=False)

    def _get_from_db_cache_sync(
        self, endpoint: str, params_json: str
    ) -> Optional[Dict[str, Any]]:
        if not self.db_manager:
            return None

        try:
            from database.schema import APICache

            with self.db_manager.session_scope() as session:
                record = (
                    session.query(APICache)
                    .filter(
                        APICache.endpoint == endpoint,
                        APICache.parameters == params_json,
                        APICache.source == "understat",
                    )
                    .order_by(APICache.created_at.desc())
                    .first()
                )

                if not record:
                    return None

                if record.expires_at and record.expires_at <= datetime.utcnow():
                    return None

                if not record.response:
                    return None

                return json.loads(record.response)

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Understat DB cache retrieval error: %s", exc)
            return None

    def _store_db_cache_sync(
        self, endpoint: str, params_json: str, data: Dict[str, Any]
    ) -> bool:
        if not self.db_manager:
            return False

        try:
            from database.schema import APICache

            payload = json.dumps(data)
            expires_at = datetime.utcnow() + timedelta(seconds=self.cache_expiry)

            with self.db_manager.session_scope() as session:
                record = (
                    session.query(APICache)
                    .filter(
                        APICache.endpoint == endpoint,
                        APICache.parameters == params_json,
                        APICache.source == "understat",
                    )
                    .order_by(APICache.created_at.desc())
                    .first()
                )

                if record:
                    record.response = payload
                    record.status_code = 200
                    record.expires_at = expires_at
                else:
                    session.add(
                        APICache(
                            endpoint=endpoint,
                            parameters=params_json,
                            response=payload,
                            status_code=200,
                            source="understat",
                            created_at=datetime.utcnow(),
                            expires_at=expires_at,
                        )
                    )

            return True

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug("Understat DB cache storage error: %s", exc)
            return False

    def _update_rate_limit(self, headers: Dict[str, str]) -> None:
        if not headers:
            return

        remaining = headers.get("X-RateLimit-Remaining")
        if remaining is not None:
            try:
                self.rate_limit_remaining = int(remaining)
            except ValueError:
                pass

        reset = headers.get("X-RateLimit-Reset")
        if reset is not None:
            try:
                self.rate_limit_reset = int(reset)
            except ValueError:
                pass
    
    async def get_league_matches(self, league_code: str, season: str = "2023") -> Optional[pd.DataFrame]:
        """
        Get all matches for a specific league and season.
        
        Args:
            league_code: League code (e.g., "premier_league", "la_liga")
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            DataFrame with match data or None if failed
        """
        # Convert league code to Understat format
        understat_league = self._normalize_league_code(league_code)
        if not understat_league:
            logger.error(f"Invalid league code: {league_code}")
            return None
        
        try:
            # Call the API
            endpoint = f"league/{understat_league}/{season}/matches"
            response_data = await self._make_request(endpoint)
            
            if not response_data or "matches" not in response_data:
                logger.warning(f"No matches data found for {league_code} season {season}")
                return None
            
            # Process matches data
            all_matches = []
            for match in response_data["matches"]:
                match_data = {
                    "id": match["id"],
                    "home_team": match["home_team"]["name"],
                    "away_team": match["away_team"]["name"],
                    "match_date": datetime.fromisoformat(match["datetime"].replace("Z", "+00:00")),
                    "league": league_code,
                    "season": season,
                    "home_goals": int(match["home_goals"]),
                    "away_goals": int(match["away_goals"]),
                    "home_xG": float(match["home_xg"]),
                    "away_xG": float(match["away_xg"]),
                    "home_team_id": match["home_team"]["id"],
                    "away_team_id": match["away_team"]["id"],
                    "status": match["status"]
                }
                all_matches.append(match_data)
            
            # Create DataFrame
            df = pd.DataFrame(all_matches)
            
            return df
        
        except Exception as e:
            logger.error(f"Error getting league matches for {league_code}: {e}")
            return None

    async def get_league_teams(
        self,
        league_code: str,
        season: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Retrieve team metadata for a league/season."""

        understat_league = self._normalize_league_code(league_code)
        if not understat_league:
            logger.error("Invalid league code for team retrieval: %s", league_code)
            return None

        target_season = season or self._default_season()
        endpoint = f"league/{understat_league}/{target_season}/teams"

        teams: List[Dict[str, Any]] = []

        try:
            response_data = await self._make_request(endpoint)

            if response_data and "teams" in response_data:
                for team in response_data.get("teams", []):
                    team_id = str(team.get("id") or team.get("team_id") or "").strip()
                    name = team.get("name") or team.get("title")
                    if not team_id or not name:
                        continue

                    teams.append(
                        {
                            "id": team_id,
                            "name": name,
                            "short_name": team.get("short_name")
                            or (name[:3].upper() if name else ""),
                            "league_code": league_code,
                            "season": target_season,
                            "stats": team.get("stats"),
                        }
                    )

            if teams:
                return teams

            # Fallback: build team list from league matches if dedicated endpoint missing
            matches_df = await self.get_league_matches(league_code, target_season)
            if matches_df is None or matches_df.empty:
                return None

            seen: Dict[str, Dict[str, Any]] = {}
            for row in matches_df.itertuples(index=False):
                home_id = str(getattr(row, "home_team_id", ""))
                away_id = str(getattr(row, "away_team_id", ""))
                home_name = getattr(row, "home_team", None)
                away_name = getattr(row, "away_team", None)

                if home_id and home_name and home_id not in seen:
                    seen[home_id] = {
                        "id": home_id,
                        "name": home_name,
                        "short_name": home_name[:3].upper(),
                        "league_code": league_code,
                        "season": target_season,
                    }

                if away_id and away_name and away_id not in seen:
                    seen[away_id] = {
                        "id": away_id,
                        "name": away_name,
                        "short_name": away_name[:3].upper(),
                        "league_code": league_code,
                        "season": target_season,
                    }

            if seen:
                return list(seen.values())

        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Error retrieving Understat teams for %s %s: %s",
                league_code,
                target_season,
                exc,
            )

        return None

    async def resolve_team_id(
        self, team_name: str, league_code: str, season: Optional[str] = None
    ) -> Optional[str]:
        """Resolve a human-readable team name to an Understat team id."""

        if not team_name:
            return None

        teams = await self.get_league_teams(league_code, season)
        if not teams:
            return None

        target = self._normalize_team_name(team_name)

        for team in teams:
            candidate_name = self._normalize_team_name(team.get("name", ""))
            if candidate_name == target or target in {candidate_name.replace(" fc", ""), candidate_name.replace(" afc", "")}:
                return team.get("id")

        # Try partial matches as last resort
        for team in teams:
            candidate_name = self._normalize_team_name(team.get("name", ""))
            if target in candidate_name or candidate_name in target:
                return team.get("id")

        return None
    
    async def get_team_stats(self, team_id: Union[str, int], season: str = "2023") -> Optional[Dict[str, Any]]:
        """
        Get detailed team statistics.
        
        Args:
            team_id: Team ID
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            Dictionary with team statistics or None if failed
        """
        try:
            # Call the API
            endpoint = f"team/{team_id}/{season}"
            response_data = await self._make_request(endpoint)
            
            if not response_data:
                logger.warning(f"No team data found for team ID {team_id}")
                return None
            
            # Process team statistics
            team_stats = {
                "team_id": team_id,
                "team_name": response_data["name"],
                "season": season,
                "matches": response_data.get("matches", []),
                "players": response_data.get("players", []),
                "stats": response_data.get("stats", {})
            }
            
            return team_stats
        
        except Exception as e:
            logger.error(f"Error getting team stats for team ID {team_id}: {e}")
            return None
    
    async def get_match_stats(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a specific match.
        
        Args:
            match_id: Match ID
            
        Returns:
            Dictionary with match statistics or None if failed
        """
        try:
            # Call the API
            endpoint = f"match/{match_id}"
            response_data = await self._make_request(endpoint)
            
            if not response_data:
                logger.warning(f"No match data found for match ID {match_id}")
                return None
            
            # Process match statistics
            match_stats = {
                "match_id": match_id,
                "home_team": response_data["home_team"]["name"],
                "away_team": response_data["away_team"]["name"],
                "match_date": datetime.fromisoformat(response_data["datetime"].replace("Z", "+00:00")),
                "league": response_data.get("league", ""),
                "season": response_data.get("season", ""),
                "home_score": int(response_data["home_goals"]),
                "away_score": int(response_data["away_goals"]),
                "home_xg": float(response_data["home_xg"]),
                "away_xg": float(response_data["away_xg"]),
                "shots": response_data.get("shots", []),
                "home_players": response_data.get("home_players", []),
                "away_players": response_data.get("away_players", [])
            }
            
            return match_stats
        
        except Exception as e:
            logger.error(f"Error getting match stats for match ID {match_id}: {e}")
            return None
    
    async def get_player_stats(self, player_id: str, season: str = "2023") -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a specific player.
        
        Args:
            player_id: Player ID
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            Dictionary with player statistics or None if failed
        """
        try:
            # Call the API
            endpoint = f"player/{player_id}/{season}"
            response_data = await self._make_request(endpoint)
            
            if not response_data:
                logger.warning(f"No player data found for player ID {player_id}")
                return None
            
            # Process player statistics
            player_stats = {
                "player_id": player_id,
                "player_name": response_data["name"],
                "team_name": response_data["team"],
                "position": response_data.get("position", ""),
                "season": season,
                "stats": response_data.get("stats", {}),
                "matches": response_data.get("matches", []),
                "shots": response_data.get("shots", [])
            }
            
            return player_stats
        
        except Exception as e:
            logger.error(f"Error getting player stats for player ID {player_id}: {e}")
            return None
    
    async def get_player_shot_map(self, player_id: str, season: str = "2023") -> Optional[pd.DataFrame]:
        """
        Get shot map data for a specific player.
        
        Args:
            player_id: Player ID
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            DataFrame with shot map data or None if failed
        """
        try:
            # First get player stats which includes shots
            player_data = await self.get_player_stats(player_id, season)
            
            if not player_data or "shots" not in player_data or not player_data["shots"]:
                logger.warning(f"No shot data found for player ID {player_id}")
                return None
            
            # Convert shots to DataFrame
            shots_df = pd.DataFrame(player_data["shots"])
            
            return shots_df
        
        except Exception as e:
            logger.error(f"Error getting player shot map for player ID {player_id}: {e}")
            return None
    
    async def get_team_shot_map(self, team_id: Union[str, int], season: str = "2023") -> Optional[pd.DataFrame]:
        """
        Get shot map data for a specific team.
        
        Args:
            team_id: Team ID
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            DataFrame with shot map data or None if failed
        """
        try:
            # Call the API
            endpoint = f"team/{team_id}/{season}/shots"
            response_data = await self._make_request(endpoint)
            
            if not response_data or "shots" not in response_data:
                logger.warning(f"No shot data found for team ID {team_id}")
                return None
            
            # Convert shots to DataFrame
            shots_df = pd.DataFrame(response_data["shots"])
            
            return shots_df
        
        except Exception as e:
            logger.error(f"Error getting team shot map for team ID {team_id}: {e}")
            return None
    
    def set_api_key(self, api_key: str):
        """
        Set or update the API key.
        
        Args:
            api_key: New API key
        """
        self.api_key = api_key
        logger.info("Updated Understat API key")
    
    def configure_cache(self, use_cache: bool = True, cache_expiry: int = 3600):
        """
        Configure cache settings.
        
        Args:
            use_cache: Whether to use cache
            cache_expiry: Cache expiry time in seconds
        """
        self.use_cache = use_cache
        self.cache_expiry = cache_expiry
        logger.info(f"Updated cache settings: use_cache={use_cache}, expiry={cache_expiry}s")

    async def close(self) -> None:
        """Clean up owned HTTP resources."""
        if self._owns_http_client and hasattr(self.http_client, "close"):
            await self.http_client.close()
    
    async def test_connection(self) -> bool:
        """
        Test the connection to the Understat API.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try a simple API call
            endpoint = "leagues"
            response_data = await self._make_request(endpoint)
            
            if response_data and "leagues" in response_data:
                logger.info("Understat API connection test successful")
                return True
            else:
                logger.warning("Understat API connection test failed: Invalid response")
                return False
        
        except Exception as e:
            logger.error(f"Understat API connection test failed: {e}")
            return False