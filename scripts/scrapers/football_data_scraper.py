"""
Football-Data.org API scraper for retrieving match data, team statistics, and odds.
"""
import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp  # Added for exception handling
import pandas as pd

from scripts.scrapers.base_scraper import BaseScraper
from utils.data_validator import sanitize_dataframe, validate_match_data
from utils.http_client_async import HttpClientAsync  # Corrected import


# Backwards compatibility: tests patch scripts.scrapers.football_data_scraper.HttpClient
# so provide an alias that subclasses the async client. This mirrors the pattern in base_scraper.
class HttpClient(HttpClientAsync):  # type: ignore
    """Compatibility subclass so test patches against HttpClient work without modifying test code."""
    pass

logger = logging.getLogger(__name__)

class FootballDataScraper(BaseScraper):
    """
    Scraper for the Football-Data.org API.
    Retrieves match data, team statistics, and odds.
    """
    
    LEAGUE_CODES = {
        "premier_league": "PL",
        "la_liga": "PD",
        "bundesliga": "BL1",
        "serie_a": "SA",
        "ligue_1": "FL1",
        "eredivisie": "DED"
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        http_client: Optional[HttpClientAsync] = None, # Corrected type hint
        rate_limit_delay: Tuple[float, float] = (1.0, 2.0),
        system_monitor: Optional['SystemMonitor'] = None,
        proxy_manager=None,
        playwright_manager=None,
        use_proxies=False
    ):
        # If no client supplied, instantiate the compatibility HttpClient so tests can patch it
        if http_client is None:
            try:
                http_client = HttpClient()
            except Exception:
                http_client = HttpClientAsync()

        super().__init__(
            base_url="https://api.football-data.org/v4",
            name="FootballData",
            http_client=http_client,
            use_proxies=use_proxies,
            use_playwright=False,
            rate_limit_delay=rate_limit_delay,
            system_monitor=system_monitor,
            proxy_manager=proxy_manager,
            playwright_manager=playwright_manager
        )
        
        # Try multiple sources for API key
        self.api_key = api_key or os.environ.get("FOOTBALL_DATA_API_KEY") or os.environ.get("FOOTBALL_DATA_TOKEN")
        
        if not self.api_key:
            try:
                from utils.config import Config

                # Look in the correct location in config based on config.yaml structure
                self.api_key = Config.get("api_keys.football_data", "")
            except ImportError:
                pass
            
        # Try to get from environment if .env file is loaded but not in environment
        if not self.api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                self.api_key = os.environ.get("FOOTBALL_DATA_API_KEY") or os.environ.get("FOOTBALL_DATA_TOKEN")
            except ImportError:
                pass
        
        # Validate API key format
        if self.api_key:
            # Check if it looks like a valid Football-Data API key (should be alphanumeric, not RapidAPI format)
            if self.api_key.startswith('b87a700ad8') or 'msh' in self.api_key or len(self.api_key) > 50:
                logger.warning("API key appears to be invalid (RapidAPI format detected). Please get a valid Football-Data API key from https://www.football-data.org/client/register")
                self.api_key = None
            elif len(self.api_key) < 10:
                logger.warning("API key appears to be too short. Please check your Football-Data API key.")
                self.api_key = None
            # Remove any environment variable syntax that might be present
            elif self.api_key.startswith("${") and self.api_key.endswith("}"):
                env_var = self.api_key[2:-1]
                logger.warning(f"API key contains environment variable syntax: {self.api_key}. Attempting to resolve from environment...")
                self.api_key = os.environ.get(env_var)
                if not self.api_key:
                    logger.error(f"Could not resolve API key from environment variable {env_var}")
        
        # Add rate limiting attributes
        self.rate_limit_remaining = 10  # Default limit per minute
        self.rate_limit_reset = datetime.now() + timedelta(seconds=60)
        self.use_fallback = False
                
        if not self.api_key:
            logger.warning("No valid API key provided for Football-Data.org. Using limited API functionality.")
            self.has_api_key = False
        else:
            # Log a masked version of the API key for debugging
            masked_key = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "***"
            logger.info(f"Football-Data.org API key configured successfully. Key: {masked_key}")
            self.has_api_key = True
        
        # Rate limiting parameters - more conservative for production
        self.request_count = 0
        self.minute_limit = 6  # More conservative: 6 requests per minute (down from 10)
        self.min_request_interval = 12.0  # Minimum 12 seconds between requests (increased from 6)
        self.last_request_time = 0
        self.rate_limit_lock = asyncio.Lock()
        
        # Connection management
        self._session_lock = asyncio.Lock()
        self._session_created_at = None
        self._max_session_age = 300  # 5 minutes max session age
    
    async def _rate_limit_wait(self):
        """Wait if necessary to respect rate limits with exponential backoff"""
        async with self.rate_limit_lock:
            # Calculate time since last request
            from utils.asyncio_compat import (
                loop_time,  # lightweight import inside method
            )
            current_time = loop_time()
            time_since_last = current_time - self.last_request_time
            
            # If we've made a request recently, wait to respect rate limits
            if time_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last
                # Add some jitter to avoid thundering herd
                jitter = wait_time * 0.1 * (0.5 - random.random())
                wait_time += jitter
                logger.debug(f"Rate limit: Waiting {wait_time:.2f} seconds before next request")
                await asyncio.sleep(wait_time)
            
            # Update last request time
            self.last_request_time = loop_time()
            self.request_count += 1
            
            # Log rate limiting status every 10 requests
            if self.request_count % 10 == 0:
                logger.info(f"Rate limiting: Made {self.request_count} requests, last interval: {time_since_last:.2f}s")

    async def _ensure_session_health(self):
        """Ensure HTTP session is healthy and not too old"""
        async with self._session_lock:
            from utils.asyncio_compat import loop_time
            current_time = loop_time()
            
            # Check if session is too old or closed
            if (self._session_created_at is None or 
                current_time - self._session_created_at > self._max_session_age or
                (hasattr(self.http_client, 'session') and self.http_client.session and self.http_client.session.closed)):
                
                logger.debug("Refreshing HTTP session for better connection health")
                # Close old session if it exists
                if hasattr(self.http_client, 'session') and self.http_client.session:
                    if not self.http_client.session.closed:
                        await self.http_client.session.close()
                
                # Force creation of new session
                self.http_client.session = None
                self._session_created_at = current_time

    def _get_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get headers with API key for API requests"""
        headers = {"X-Auth-Token": self.api_key, "Accept": "application/json"}
        if additional_headers:
            headers.update(additional_headers)
        return headers
    
    async def _batch_date_ranges(self, start_date: datetime, end_date: datetime, days_per_batch: int = 7):
        """Yield (date_from, date_to) tuples for batching requests by date range."""
        current = start_date
        while current <= end_date:
            batch_end = min(current + timedelta(days=days_per_batch - 1), end_date)
            yield current.strftime("%Y-%m-%d"), batch_end.strftime("%Y-%m-%d")
            current = batch_end + timedelta(days=1)
    async def _make_request_with_backoff(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Make HTTP request with exponential backoff and enhanced error handling."""
        for attempt in range(4):  # 0, 1, 2, 3 (4 total attempts)
            try:
                await self._ensure_session_health()
                await self._rate_limit_wait()
                
                headers = self._get_headers()
                logger.debug(f"Making request to {endpoint} with params: {params}")
                response = await self.http_client.get(
                    endpoint,
                    params=params,
                    headers=headers
                )
                
                if response.status == 200:
                    return response
                elif response.status == 429:
                    # Rate limit exceeded
                    wait_time = self.min_request_interval * (2 ** attempt)
                    jitter = wait_time * 0.1 * (0.5 - random.random())
                    total_wait = wait_time + jitter
                    logger.warning(f"Rate limit hit (429). Waiting {total_wait:.2f}s before retry {attempt+1}/4")
                    await asyncio.sleep(total_wait)
                    continue
                elif response.status == 400:
                    # Bad request - try to get more details
                    try:
                        error_data = await response.text()
                        logger.error(f"Bad request (400) for {endpoint} with params {params}")
                        logger.error(f"Error response: {error_data}")
                        
                        # If this is a date parameter issue, try to recover
                        if "dateFrom" in params or "dateTo" in params:
                            if "dateFrom" in params:
                                logger.warning(f"Possible issue with dateFrom format: {params['dateFrom']}")
                            if "dateTo" in params:
                                logger.warning(f"Possible issue with dateTo format: {params['dateTo']}")
                                
                            # Don't retry date format issues
                            return None
                    except:
                        logger.error(f"Bad request (400) for {endpoint} with params {params} - couldn't parse error")
                    
                    return None
                else:
                    # Other HTTP errors
                    try:
                        error_data = await response.text()
                        logger.error(f"HTTP {response.status} error for {endpoint}: {error_data}")
                    except:
                        logger.error(f"HTTP {response.status} error for {endpoint}")
                        
                    if attempt < 3:
                        wait_time = 2.0 * (2 ** attempt)
                        await asyncio.sleep(wait_time)
                        continue
                    return None
                    
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
                if attempt < 3:
                    wait_time = 3.0 * (2 ** attempt)
                    logger.warning(f"Connection error (attempt {attempt+1}/4): {e}. Retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Final connection error after 4 attempts: {e}")
                    return None
        
        return None

    async def get_matches(
        self, 
        league_code: str, 
        status: str = None,
        date_from: str = None,
        date_to: str = None,
        matchday: int = None,
        season: str = None,
        limit: int = 100,
        retry_count: int = 3,  # Increased retry count
        batch_days: int = 7,    # Reduced batch size to avoid date range issues
        min_wait: float = 12.0  # Increased wait time
    ) -> Optional[pd.DataFrame]:
        """
        Get matches for a specific league, batching requests by date range to avoid rate limits.
        Enhanced with better connection management and error handling.
        """
        api_code = self.LEAGUE_CODES.get(league_code) if league_code in self.LEAGUE_CODES else league_code
        endpoint = f"{self.base_url}/competitions/{api_code}/matches"
        all_matches = []
        
        # Debug log the API key (masked for security)
        if self.api_key:
            masked_key = self.api_key[:4] + "..." + self.api_key[-4:] if len(self.api_key) > 8 else "***"
            logger.debug(f"Using API key: {masked_key}")
        else:
            logger.error("No API key available for Football-Data.org")
            return pd.DataFrame()
            
        # Get the current season if not provided
        if not season:
            current_year = datetime.now().year
            current_month = datetime.now().month
            # Football seasons typically start in August and end in May of the next year
            # So from August onwards, it's the current year season, before August it's the previous year season
            if current_month >= 8:
                season = str(current_year)
            else:
                season = str(current_year - 1)
            
        # For 2025, the API might not have data yet. Try 2024 instead
        if season == "2025":
            logger.warning("Season 2025 data may not be available. Trying 2024 season.")
            season = "2024"
            
        # Fix for future date issue: if date_to is in the future beyond 1 year, restrict it
        current_date = datetime.now().date()
        max_future_date = current_date + timedelta(days=365)  # Maximum 1 year in future
            
        # If both date_from and date_to are provided, batch requests
        if date_from and date_to:
            try:
                # Convert any datetime objects to strings
                if isinstance(date_from, datetime):
                    date_from = date_from.strftime("%Y-%m-%d")
                if isinstance(date_to, datetime):
                    date_to = date_to.strftime("%Y-%m-%d")
                
                # Parse and validate dates
                try:
                    start_date = datetime.strptime(date_from, "%Y-%m-%d").date()
                except ValueError:
                    logger.error(f"Invalid date_from format: {date_from}. Expected YYYY-MM-DD")
                    return pd.DataFrame()
                    
                try:
                    end_date = datetime.strptime(date_to, "%Y-%m-%d").date()
                except ValueError:
                    logger.error(f"Invalid date_to format: {date_to}. Expected YYYY-MM-DD")
                    return pd.DataFrame()
                
                # Ensure dates are not too far in the future (API limitation)
                if end_date > max_future_date:
                    logger.warning(f"Date range too far in future. Limiting end date to {max_future_date}")
                    end_date = max_future_date
                    
                # Ensure date range is not too large (API limitation)
                max_date_range = timedelta(days=90)  # 3 months
                if end_date - start_date > max_date_range:
                    logger.warning(f"Date range too large. Football-Data API limits to 3 months. Adjusting...")
                    # Keep the end date but adjust start date
                    start_date = end_date - max_date_range
                    
                # Format dates for API
                date_from_fmt = start_date.strftime("%Y-%m-%d")
                date_to_fmt = end_date.strftime("%Y-%m-%d")
                
                logger.info(f"Using date range: {date_from_fmt} to {date_to_fmt} for {league_code}")
                
                # Get match data with a single request (simplify to avoid batching issues)
                params = {
                    "dateFrom": date_from_fmt,
                    "dateTo": date_to_fmt
                }
                
                if status:
                    params["status"] = status
                if matchday:
                    params["matchday"] = matchday
                if season:
                    params["season"] = season
                    
                # Make a single request instead of batching
                logger.debug(f"Making request to {endpoint} with params: {params}")
                response = await self._make_request_with_backoff(endpoint, params)
                if response is None:
                    logger.error(f"Failed to get matches for {league_code}")
                    return pd.DataFrame()
                
                try:
                    matches_data = await response.json()
                    if not matches_data or "matches" not in matches_data:
                        logger.warning(f"No match data found for {league_code}")
                        return pd.DataFrame()
                        
                    for match in matches_data["matches"]:
                        match_data = {
                            "match_id": match["id"],
                            "competition": match["competition"].get("name", ""),
                            "match_date": match["utcDate"],
                            "status": match["status"],
                            "matchday": match["matchday"],
                            "home_team": match["homeTeam"]["name"],
                            "away_team": match["awayTeam"]["name"],
                            "home_team_id": match["homeTeam"]["id"],
                            "away_team_id": match["awayTeam"]["id"],
                            "venue": match.get("venue", ""),
                        }
                        if "score" in match and match["score"].get("fullTime"):
                            match_data["home_score"] = match["score"]["fullTime"].get("home", None)
                            match_data["away_score"] = match["score"]["fullTime"].get("away", None)
                        all_matches.append(match_data)
                        
                    logger.info(f"Successfully retrieved {len(matches_data['matches'])} matches for {league_code}")
                    
                except Exception as e:
                    logger.error(f"Error parsing matches for {league_code}: {e}")
                    return pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Error in get_matches: {e}")
                return pd.DataFrame()
        else:
            # If no date range specified, get current season data
            # This is a simplified approach - just get current season
            params = {}
            if status:
                params["status"] = status
            if season:
                params["season"] = season
            if matchday:
                params["matchday"] = matchday
                
            # Handle single date parameters if provided
            if date_from:
                try:
                    if isinstance(date_from, datetime):
                        date_from = date_from.strftime("%Y-%m-%d")
                    # Validate format
                    datetime.strptime(date_from, "%Y-%m-%d")
                    params["dateFrom"] = date_from
                except ValueError:
                    logger.error(f"Invalid date_from format: {date_from}. Expected YYYY-MM-DD")
                    # Continue without date_from parameter
            
            if date_to:
                try:
                    if isinstance(date_to, datetime):
                        date_to = date_to.strftime("%Y-%m-%d")
                    # Validate format
                    datetime.strptime(date_to, "%Y-%m-%d")
                    
                    # Check if date is too far in future
                    date_to_obj = datetime.strptime(date_to, "%Y-%m-%d").date()
                    if date_to_obj > max_future_date:
                        logger.warning(f"Date too far in future. Limiting to {max_future_date}")
                        date_to = max_future_date.strftime("%Y-%m-%d")
                        
                    params["dateTo"] = date_to
                except ValueError:
                    logger.error(f"Invalid date_to format: {date_to}. Expected YYYY-MM-DD")
                    # Continue without date_to parameter
                
            logger.debug(f"Making request to {endpoint} with params: {params}")
            response = await self._make_request_with_backoff(endpoint, params)
            if response is None:
                logger.error(f"Failed to get matches for {league_code}")
                return pd.DataFrame()
            
            try:
                matches_data = await response.json()
                if not matches_data or "matches" not in matches_data:
                    logger.warning(f"No match data found for {league_code}")
                    # Log the full response for debugging
                    try:
                        logger.debug(f"API response: {matches_data}")
                    except:
                        pass
                    return pd.DataFrame()
                    
                for match in matches_data["matches"]:
                    match_data = {
                        "match_id": match["id"],
                        "competition": match["competition"].get("name", ""),
                        "match_date": match["utcDate"],
                        "status": match["status"],
                        "matchday": match["matchday"],
                        "home_team": match["homeTeam"]["name"],
                        "away_team": match["awayTeam"]["name"],
                        "home_team_id": match["homeTeam"]["id"],
                        "away_team_id": match["awayTeam"]["id"],
                        "venue": match.get("venue", ""),
                    }
                    if "score" in match and match["score"].get("fullTime"):
                        match_data["home_score"] = match["score"]["fullTime"].get("home", None)
                        match_data["away_score"] = match["score"]["fullTime"].get("away", None)
                    all_matches.append(match_data)
                    
            except Exception as e:
                logger.error(f"Error parsing matches for {league_code}: {e}")
                return pd.DataFrame()
        
        if not all_matches:
            logger.info(f"No matches found for {league_code}")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_matches)
        if "match_date" in df.columns:
            df["match_date"] = pd.to_datetime(df["match_date"])
        df = self.sanitize_data(df)
        logger.info(f"Successfully retrieved {len(df)} matches for {league_code}")
        return df

    async def get_team_info(self, team_id: int) -> Optional[Dict[str, Any]]:
        endpoint = f"{self.base_url}/teams/{team_id}"
        try:
            response = await self.http_client.get(endpoint, headers=self._get_headers())
            
            # Check if response is successful
            if response.status != 200:
                logger.error(f"HTTP {response.status} error getting team info for ID {team_id}")
                return None
                
            data = await response.json()
            if not data:
                logger.warning(f"No team information found for ID {team_id}")
                return None
            
            team_info = {
                "id": data["id"],
                "name": data["name"],
                "short_name": data["shortName"],
                "tla": data["tla"],
                "crest": data["crest"],
                "address": data.get("address", ""),
                "website": data.get("website", ""),
                "founded": data.get("founded", None),
                "club_colors": data.get("clubColors", ""),
                "venue": data.get("venue", ""),
                "last_updated": data.get("lastUpdated", "")
            }
            if "squad" in data:
                team_info["squad_size"] = len(data["squad"])
                team_info["squad"] = [
                    {
                        "id": player["id"],
                        "name": player["name"],
                        "position": player.get("position", ""),
                    }
                    for player in data["squad"]
                ]
            return team_info
        except Exception as e:
            logger.error(f"Error getting team info for ID {team_id}: {e}")
            return None
    
    async def get_teams(self, league_id: str) -> List[Dict[str, Any]]:
        """
        Get teams for a specific league.
        
        Args:
            league_id: League ID/code (e.g., 'PL', 'BL1')
            
        Returns:
            List of team data dictionaries
        """
        logger.info(f"Getting teams for league: {league_id}")
        endpoint = f"{self.base_url}/competitions/{league_id}/teams"
        
        try:
            response = await self.http_client.get(endpoint, headers=self._get_headers())
            
            # Check if response is successful
            if response.status != 200:
                logger.error(f"HTTP {response.status} error getting teams for league {league_id}")
                # Try to get error details
                try:
                    error_data = await response.json()
                    logger.error(f"API Error details: {error_data}")
                except:
                    logger.error(f"Could not parse error response for {league_id}")
                return []
                
            data = await response.json()
            if not data or "teams" not in data:
                logger.warning(f"No teams found for league {league_id}")
                return []
            
            teams = []
            for team in data["teams"]:
                team_data = {
                    "id": str(team["id"]),
                    "name": team["name"],
                    "short_name": team.get("shortName", ""),
                    "tla": team.get("tla", ""),
                    "crest_url": team.get("crest", ""),
                    "venue": team.get("venue", ""),
                    "website": team.get("website", ""),
                    "founded": team.get("founded", None),
                    "club_colors": team.get("clubColors", ""),
                    "api_id": str(team["id"]),  # Store the API ID
                    "league_id": league_id
                }
                teams.append(team_data)
            
            logger.info(f"Found {len(teams)} teams for league {league_id}")
            return teams
        except Exception as e:
            logger.error(f"Error getting teams for league {league_id}: {e}")
            return []
    
    async def get_team_matches(
        self,
        team_id: int,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        status: str = "FINISHED",
        limit: int = 10
    ) -> Optional[pd.DataFrame]:
        if date_to is None:
            date_to = datetime.now()
        if date_from is None:
            date_from = date_to - timedelta(days=30)
        
        date_from_str = date_from.strftime("%Y-%m-%d")
        date_to_str = date_to.strftime("%Y-%m-%d")
        
        endpoint = f"{self.base_url}/teams/{team_id}/matches"
        params = {
            "dateFrom": date_from_str,
            "dateTo": date_to_str,
            "status": status,
            "limit": limit
        }
        
        try:
            await self._rate_limit_wait()  # Wait if necessary to respect rate limits
            # CORRECTED LINE FOR get_team_matches
            response = await self.http_client.get(endpoint, params=params, headers=self._get_headers())
            
            # Check if response is successful
            if response.status != 200:
                logger.error(f"HTTP {response.status} error getting team matches for ID {team_id}")
                return None
                
            data = await response.json()
            if not data or "matches" not in data:
                logger.warning(f"No matches found for team ID {team_id}")
                return pd.DataFrame()
            
            matches = []
            for match in data["matches"]:
                match_data = {
                    "id": match["id"],
                    "competition": match["competition"]["name"],
                    "match_date": match["utcDate"],
                    "status": match["status"],
                    "matchday": match["matchday"],
                    "home_team": match["homeTeam"]["name"],
                    "away_team": match["awayTeam"]["name"],
                    "home_team_id": match["homeTeam"]["id"],
                    "away_team_id": match["awayTeam"]["id"],
                    "is_home": match["homeTeam"]["id"] == team_id,
                }
                if "score" in match and match["score"].get("fullTime"):
                    match_data["home_score"] = match["score"]["fullTime"].get("home", None)
                    match_data["away_score"] = match["score"]["fullTime"].get("away", None)
                    if match_data["is_home"]:
                        match_data["team_score"] = match_data["home_score"]
                        match_data["opponent_score"] = match_data["away_score"]
                    else:
                        match_data["team_score"] = match_data["away_score"]
                        match_data["opponent_score"] = match_data["home_score"]
                matches.append(match_data)
            
            df = pd.DataFrame(matches)
            if "match_date" in df.columns:
                df["match_date"] = pd.to_datetime(df["match_date"])
            df = df.sort_values("match_date", ascending=False)
            df = self.sanitize_data(df)
            return df
        except Exception as e:
            logger.error(f"Error getting team matches for ID {team_id}: {e}")
            return None

    async def get_match_odds(self, match_id: int) -> Optional[Dict[str, Any]]:
        endpoint = f"{self.base_url}/matches/{match_id}"
        try:
            response = await self.http_client.get(endpoint, headers=self._get_headers())
            
            # Check if response is successful
            if response.status != 200:
                logger.error(f"HTTP {response.status} error getting odds for match ID {match_id}")
                return None
                
            data = await response.json()
            if not data or "odds" not in data:
                logger.warning(f"No odds found for match ID {match_id}")
                return None
            
            odds_data = { "match_id": match_id, "bookmakers": [] }
            for bookmaker in data["odds"]["bookmakers"]:
                bookmaker_data = { "name": bookmaker["name"], "markets": [] }
                for market in bookmaker["markets"]:
                    market_data = { "name": market["name"], "outcomes": [] }
                    for outcome in market["outcomes"]:
                        market_data["outcomes"].append({
                            "name": outcome["name"],
                            "price": outcome["price"]
                        })
                    bookmaker_data["markets"].append(market_data)
                odds_data["bookmakers"].append(bookmaker_data)
            return odds_data
        except Exception as e:
            logger.error(f"Error getting odds for match ID {match_id}: {e}")
            return None
    
    async def get_league_standings(self, league_code: str) -> Optional[pd.DataFrame]:
        api_league_code = self.LEAGUE_CODES.get(league_code)
        if not api_league_code:
            logger.error(f"Invalid league code: {league_code}")
            return None
        
        # Use the correct endpoint format for v4 API
        endpoint = f"{self.base_url}/competitions/{api_league_code}/standings"
        
        try:
            logger.info(f"Fetching standings for {league_code} (API code: {api_league_code})")
            response = await self.http_client.get(endpoint, headers=self._get_headers())
            
            # Check if response is successful
            if response.status != 200:
                logger.error(f"HTTP {response.status} error getting standings for {league_code} from {endpoint}")
                # Try to get error details
                try:
                    error_data = await response.json()
                    logger.error(f"API Error details: {error_data}")
                except:
                    logger.error(f"Could not parse error response for {league_code}")
                return None
                
            data = await response.json()
            
            if not data or "standings" not in data:
                logger.warning(f"No standings found for {league_code}")
                return pd.DataFrame()
            
            standings_data = None
            for standing in data["standings"]:
                if standing["type"] == "TOTAL":
                    standings_data = standing
                    break
            if not standings_data and data["standings"]: # Fallback to first if TOTAL not found
                standings_data = data["standings"][0]
            
            if not standings_data: # If still no standings_data
                 logger.warning(f"No suitable standings data found for {league_code}")
                 return pd.DataFrame()

            standings = []
            for team in standings_data["table"]:
                standings.append({
                    "position": team["position"],
                    "team_id": team["team"]["id"],
                    "team_name": team["team"]["name"],
                    "played": team["playedGames"],
                    "won": team["won"],
                    "drawn": team["draw"],
                    "lost": team["lost"],
                    "points": team["points"],
                    "goals_for": team["goalsFor"],
                    "goals_against": team["goalsAgainst"],
                    "goal_difference": team["goalDifference"],
                    "form": team.get("form", "")
                })
            
            df = pd.DataFrame(standings)
            logger.info(f"Successfully fetched {len(df)} team standings for {league_code}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting standings for {league_code}: {e}")
            return None

    async def get_raw_league_standings_json(self, api_league_code: str) -> Optional[Dict[str, Any]]:
        if not api_league_code:
            logger.error("Invalid API league code provided.")
            return None
            
        endpoint = f"{self.base_url}/competitions/{api_league_code}/standings"
        
        # Add season parameter for current season
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Determine season based on current date
        if current_month >= 8:  # Season typically starts in August
            season_year = current_year
        else:
            season_year = current_year - 1
            
        # For 2025, use 2024 as the data likely isn't available yet
        if season_year >= 2025:
            season_year = 2024
            
        params = {"season": season_year}
        logger.debug(f"Fetching raw league standings from: {endpoint} with season: {season_year}")
        
        try:
            response = await self.http_client.get(endpoint, params=params, headers=self._get_headers())
            
            # Check if response is successful
            if response.status != 200:
                logger.error(f"HTTP {response.status} error getting raw standings for {api_league_code}")
                return None
                
            data = await response.json()
            
            if not data or "standings" not in data:
                logger.warning(f"No standings found for API league code {api_league_code}")
                return None
                
            logger.info(f"Successfully fetched raw standings for {api_league_code}")
            return data
            
        except aiohttp.ClientResponseError as e:
            logger.error(f"HTTP error getting raw standings for {api_league_code}: {e.status} - {e.message}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"Client error getting raw standings for {api_league_code}: {e}")
            return None
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting raw standings for {api_league_code}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting raw standings for {api_league_code}: {e}")
            return None
    
    async def get_league_data(self, league_code: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive league data from Football-Data.org API.
        
        Args:
            league_code: Code for the league (e.g., 'premier_league', 'la_liga')
            
        Returns:
            Dictionary containing league data or None if failed
        """
        api_league_code = self.LEAGUE_CODES.get(league_code)
        if not api_league_code:
            logger.error(f"Invalid league code: {league_code}")
            return None
            
        try:
            league_data = {
                'league': league_code,
                'source': self.name,
                'timestamp': datetime.now().isoformat(),
                'teams': None,
                'standings': None,
                'recent_matches': None,
                'upcoming_matches': None
            }
            
            # Get standings (includes team data)
            standings_df = await self.get_standings(league_code)
            if standings_df is not None and not standings_df.empty:
                league_data['standings'] = standings_df.to_dict('records')
                # Extract team names from standings
                league_data['teams'] = standings_df['team'].unique().tolist() if 'team' in standings_df.columns else []
                
            # Get recent matches (last 14 days)
            date_from = datetime.now() - timedelta(days=14)
            recent_matches_df = await self.get_matches(
                league_code,
                date_from=date_from,
                status="FINISHED"
            )
            if recent_matches_df is not None and not recent_matches_df.empty:
                league_data['recent_matches'] = recent_matches_df.to_dict('records')
                
            # Get upcoming matches (next 14 days)
            date_to = datetime.now() + timedelta(days=14)
            upcoming_matches_df = await self.get_matches(
                league_code,
                date_to=date_to,
                status="SCHEDULED"
            )
            if upcoming_matches_df is not None and not upcoming_matches_df.empty:
                league_data['upcoming_matches'] = upcoming_matches_df.to_dict('records')
            
            # Check if we got any data
            has_data = any([
                league_data['standings'],
                league_data['recent_matches'],
                league_data['upcoming_matches']
            ])
            
            if has_data:
                logger.info(f"Successfully retrieved Football-Data league data for {league_code}")
                return league_data
            else:
                logger.warning(f"No Football-Data league data available for {league_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting Football-Data league data for {league_code}: {e}")
            return None
    
    async def get_standings(self, league_code: str) -> Optional[pd.DataFrame]:
        """
        Get current standings for a league.
        
        Args:
            league_code: League code (e.g., 'premier_league', 'la_liga')
            
        Returns:
            DataFrame with standings data or None if failed
        """
        api_league_code = self.LEAGUE_CODES.get(league_code)
        if not api_league_code:
            logger.error(f"Invalid league code: {league_code}")
            return None
            
        endpoint = f"/competitions/{api_league_code}/standings"
        headers = self._get_headers()
        try:
            logger.info(f"Fetching standings for {league_code} ({api_league_code})")
            response = await self.http_client.get(
                f"{self.base_url}{endpoint}",
                headers=headers
            )
            
            # Check if response is successful
            if response.status != 200:
                logger.error(f"HTTP {response.status} error getting standings for {league_code}")
                return None
                
            try:
                data = await response.json()
                standings_data = data.get("standings", [])
                
                if not standings_data:
                    logger.warning(f"No standings data available for {league_code}")
                    return None
                
                # Usually there's a 'total' standings table
                total_standings = next(
                    (table for table in standings_data if table.get("type") == "TOTAL"), 
                    standings_data[0]
                )
                
                table_data = total_standings.get("table", [])
                
                # Create a DataFrame
                standings_df = pd.DataFrame(table_data)
                if not standings_df.empty:
                    # Extract team info and flatten
                    if 'team' in standings_df.columns:
                        team_df = pd.json_normalize(standings_df['team'])
                        team_df = team_df.rename(columns={
                            'id': 'team_id',
                            'name': 'team_name',
                            'crest': 'crest_url'
                        })
                        
                        # Drop the original team column and join with flattened team data
                        standings_df = standings_df.drop('team', axis=1)
                        standings_df = pd.concat([standings_df, team_df], axis=1)
                    
                    logger.info(f"Successfully fetched standings for {league_code}")
                    return standings_df
                else:
                    logger.warning(f"Empty standings data for {league_code}")
                    return None
            except Exception as e:
                logger.error(f"Error parsing standings data for {league_code}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error getting standings for {league_code}: {e}")
            return None

    async def _respect_rate_limit(self):
        """
        Respects Football-Data.org API rate limits with built-in wait time.
        Can be called before any API request to ensure rate limits are respected.
        """
        # Synchronize access with lock to prevent race conditions
        async with self.rate_limit_lock:
            try:
                from utils.asyncio_compat import loop_time
                current_time = loop_time()
            except Exception:
                try:
                    current_time = asyncio.get_running_loop().time()
                except RuntimeError:
                    current_time = time.time()
            
            # Check if we've exceeded our minute-based rate limit
            if self.request_count >= self.minute_limit:
                # Calculate time until next minute starts
                now = datetime.now()
                seconds_until_next_minute = 60 - now.second
                wait_time = max(seconds_until_next_minute, self.min_request_interval)
                logger.info(f"Rate limit capacity reached. Waiting {wait_time:.1f}s for reset.")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                
            # Enforce minimum time between requests
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_request_interval:
                wait_time = self.min_request_interval - time_since_last
                # Add some jitter to avoid thundering herd
                jitter = wait_time * 0.1 * random.random()
                total_wait = wait_time + jitter
                
                logger.debug(f"Rate limiting: Waiting {total_wait:.2f}s before next request")
                await asyncio.sleep(total_wait)
                
            # Update tracking variables
            try:
                from utils.asyncio_compat import loop_time
                self.last_request_time = loop_time()
            except Exception:
                try:
                    self.last_request_time = asyncio.get_running_loop().time()
                except RuntimeError:
                    self.last_request_time = time.time()
            self.request_count += 1

    async def get_match_details(self, match_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific match.
        
        Args:
            match_id: Match ID to get details for
            
        Returns:
            Dictionary with match details or None if failed
        """
        try:
            endpoint = f"{self.base_url}/matches/{match_id}"
            response = await self._make_request_with_backoff(endpoint)
            
            if response is None:
                logger.error(f"Failed to get details for match {match_id}")
                return None
                
            match_data = await response.json()
            if not match_data:
                logger.warning(f"No details found for match {match_id}")
                return None
                
            return match_data
            
        except Exception as e:
            logger.error(f"Error getting match details for {match_id}: {e}")
            return None


__all__ = ["FootballDataScraper"]