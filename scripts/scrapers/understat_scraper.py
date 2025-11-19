"""
Understat scraper for retrieving xG data, player statistics, and team performance metrics.
"""
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import pandas as pd

from scripts.core.scrapers.playwright_manager import PlaywrightManager
from scripts.core.scrapers.proxy_manager import ProxyManager
from scripts.scrapers.base_scraper import BaseScraper
from utils.data_validator import sanitize_dataframe
from utils.http_client import HttpClient

logger = logging.getLogger(__name__)

class UnderstatScraper(BaseScraper):
    """
    Scraper for Understat.com.
    Retrieves xG data, player statistics, and team performance metrics.
    """
    
    # League mapping
    LEAGUE_MAPPING = {
        "premier_league": "EPL",
        "la_liga": "La_liga", 
        "bundesliga": "Bundesliga",
        "serie_a": "Serie_A",
        "ligue_1": "Ligue_1",
        "eredivisie": "Eredivisie"  # Understat actually has Eredivisie data
    }
    
    def __init__(
        self,
        http_client: Optional[HttpClient] = None,
        proxy_manager: Optional[ProxyManager] = None,
        playwright_manager: Optional[PlaywrightManager] = None,
        use_proxies: bool = True,
        rate_limit_delay: Tuple[float, float] = (1.5, 3.0)
    ):
        """
        Initialize the Understat scraper.
        
        Args:
            http_client: Optional HTTP client to use
            proxy_manager: Optional proxy manager to use
            playwright_manager: Optional Playwright manager for JavaScript rendering
            use_proxies: Whether to use proxies
            rate_limit_delay: Tuple of (min_delay, max_delay) in seconds
        """
        super().__init__(
            name="Understat",
            base_url="https://understat.com",
            http_client=http_client,
            proxy_manager=proxy_manager,
            playwright_manager=playwright_manager,
            use_proxies=use_proxies,
            use_playwright=True,  # Understat requires JavaScript rendering
            rate_limit_delay=rate_limit_delay
        )
    
    async def fetch_html(self, url: str, headers: Optional[Dict[str, str]] = None, use_playwright: Optional[bool] = None) -> Optional[str]:
        """
        Fetch HTML content from a URL using aiohttp or Playwright.
        
        Args:
            url: URL to fetch
            headers: Optional headers
            use_playwright: Whether to use Playwright for JS rendering (overrides self.use_playwright if provided)
            
        Returns:
            HTML content as string or None if failed
        """
        try:
            # Determine whether to use Playwright (parameter overrides class setting)
            should_use_playwright = use_playwright if use_playwright is not None else self.use_playwright
            
            if should_use_playwright and self.playwright_manager:
                return await self.playwright_manager.get_page_content(url, headers)
            
            # Apply rate limiting before making the request
            self._apply_rate_limiting()
            
            # Get headers
            request_headers = headers or self._get_headers()
            
            # Use the http_client to make the request
            session = await self.http_client.get_session()
            async with session as client_session:
                async with client_session.get(url, headers=request_headers) as response:
                    response.raise_for_status()
                    return await response.text()
        except Exception as e:
            logger.error(f"Failed to fetch HTML from {url}: {e}")
            return None
    
    async def get_matches(
        self,
        league_code: str,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Get matches for a specific league and date range.
        
        Args:
            league_code: League code/identifier
            date_from: Start date for matches
            date_to: End date for matches
            **kwargs: Additional arguments
            
        Returns:
            DataFrame with match data or None if failed
        """
        try:
            # Get all matches for the league
            matches_df = await self.get_league_matches(league_code)
            if matches_df is None or matches_df.empty:
                return pd.DataFrame()
            
            # Filter by date range if provided
            if date_from or date_to:
                if 'match_date' in matches_df.columns:
                    if date_from:
                        matches_df = matches_df[matches_df['match_date'] >= date_from]
                    if date_to:
                        matches_df = matches_df[matches_df['match_date'] <= date_to]
            
            return matches_df
            
        except Exception as e:
            logger.error(f"Error getting matches for {league_code}: {e}")
            return pd.DataFrame()
    
    async def get_team_info(self, team_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific team.
        
        Args:
            team_id: Team identifier
            
        Returns:
            Dictionary with team information or None if failed
        """
        try:
            # For Understat, we need to get team info from league data
            # This is a simplified implementation
            team_info = {
                "id": str(team_id),
                "name": f"Team_{team_id}",
                "league": "unknown",
                "venue": "unknown",
                "founded": None,
                "website": None
            }
            
            return team_info
            
        except Exception as e:
            logger.error(f"Error getting team info for {team_id}: {e}")
            return None
    
    def _extract_json_from_script(self, html: str, variable_name: str) -> Optional[Any]:
        """
        Extract JSON data from a script tag in the HTML.
        
        Args:
            html: HTML content
            variable_name: Variable name to extract
            
        Returns:
            Extracted JSON data or None if not found
        """
        if not html:
            return None
        
        try:
            # Pattern to match the variable assignment in the script
            pattern = re.compile(f"var {variable_name} = JSON.parse\\('(.*?)'\\);", re.DOTALL)
            match = pattern.search(html)
            
            if not match:
                logger.warning(f"Variable {variable_name} not found in HTML")
                return None
            
            # Extract and decode the JSON string
            json_str = match.group(1)
            json_str = json_str.encode('utf-8').decode('unicode_escape')
            
            # Parse the JSON data
            return json.loads(json_str)
        
        except Exception as e:
            logger.error(f"Error extracting JSON for {variable_name}: {e}")
            return None
    
    async def get_league_matches(
        self,
        league_code: str,
        season: str = "2023"
    ) -> Optional[pd.DataFrame]:
        """
        Get all matches for a specific league and season.
        
        Args:
            league_code: League code (e.g., "premier_league", "la_liga")
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            DataFrame with match data or None if failed
        """
        # Convert league code to Understat format
        understat_league = self.LEAGUE_MAPPING.get(league_code)
        if not understat_league:
            logger.error(f"Invalid league code: {league_code}")
            return None

        # Handle season - 2025 data likely not available
        if season == "2025":
            logger.warning("Season 2025 data may not be available on Understat. Using 2024 season.")
            season = "2024"
        elif not season or season == "":
            # Default to 2024 for now
            season = "2024"
        
        # Build URL
        url = f"{self.base_url}/league/{understat_league}/{season}"
        
        try:
            # Fetch HTML with Playwright (JavaScript rendering)
            html = await self.fetch_html(url, use_playwright=True)
            if not html:
                logger.warning(f"Failed to fetch HTML from {url}")
                return None
            
            # Extract JSON data from the script tag
            matches_data = self._extract_json_from_script(html, "datesData")
            if not matches_data:
                logger.warning(f"No matches data found for {league_code} season {season}")
                return None
            
            # Process matches data
            all_matches = []
            for date, matches in matches_data.items():
                for match in matches:
                    match_data = {
                        "id": match["id"],
                        "home_team": match["h"]["title"],
                        "away_team": match["a"]["title"],
                        "match_date": datetime.fromtimestamp(int(match["datetime"])),
                        "league": league_code,
                        "season": season,
                        "home_goals": int(match["goals"]["h"]),
                        "away_goals": int(match["goals"]["a"]),
                        "home_xG": float(match["xG"]["h"]),
                        "away_xG": float(match["xG"]["a"]),
                        "home_team_id": match["h"]["id"],
                        "away_team_id": match["a"]["id"],
                        "forecast": {
                            "w": float(match["forecast"]["w"]),
                            "d": float(match["forecast"]["d"]),
                            "l": float(match["forecast"]["l"])
                        },
                        "result": match["result"]
                    }
                    all_matches.append(match_data)
            
            # Create DataFrame
            df = pd.DataFrame(all_matches)
            
            # Sanitize data
            numeric_columns = ["home_goals", "away_goals", "home_xG", "away_xG"]
            string_columns = ["home_team", "away_team", "league", "season", "result"]
            
            return sanitize_dataframe(
                df,
                numeric_columns=numeric_columns,
                string_columns=string_columns
            )
        
        except Exception as e:
            logger.error(f"Error getting league matches for {league_code}: {e}")
            return None
    
    async def get_team_stats(
        self,
        team_name: str,
        league_code: str,
        season: str = "2023"
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed team statistics.
        
        Args:
            team_name: Team name
            league_code: League code (e.g., "premier_league", "la_liga")
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            Dictionary with team statistics or None if failed
        """
        # Convert league code to Understat format
        understat_league = self.LEAGUE_MAPPING.get(league_code)
        if not understat_league:
            logger.error(f"Invalid league code: {league_code}")
            return None
        
        # Build URL
        url = f"{self.base_url}/team/{team_name}/{understat_league}/{season}"
        
        try:
            # Fetch HTML with Playwright (JavaScript rendering)
            html = await self.fetch_html(url, use_playwright=True)
            if not html:
                logger.warning(f"Failed to fetch HTML from {url}")
                return None
            
            # Extract JSON data from the script tags
            team_data = self._extract_json_from_script(html, "teamsData")
            matches_data = self._extract_json_from_script(html, "datesData")
            players_data = self._extract_json_from_script(html, "playersData")
            
            if not team_data:
                logger.warning(f"No team data found for {team_name}")
                return None
            
            # Process team statistics
            team_stats = {
                "team_name": team_name,
                "league": league_code,
                "season": season,
                "matches": [],
                "players": []
            }
            
            # Find team data - could be keyed by team name or team ID
            team_info = None
            if team_data:
                # First try team name as key
                if team_name in team_data:
                    team_info = team_data[team_name]
                else:
                    # Try to find by team title/name in the values
                    for key, data in team_data.items():
                        if isinstance(data, dict) and data.get("title") == team_name:
                            team_info = data
                            break
            
            # If team not found, return None
            if not team_info:
                logger.warning(f"Team '{team_name}' not found in data")
                return None
            
            # Add team summary statistics if found
            if team_info:
                team_stats.update({
                    "games": int(team_info.get("games", 0)),
                    "goals": int(team_info.get("goals", 0)),
                    "goals_against": int(team_info.get("goals_against", 0)),
                    "xG": float(team_info.get("xG", 0.0)),
                    "xGA": float(team_info.get("xGA", 0.0)),
                    "npxG": float(team_info.get("npxG", 0.0)),
                    "npxGA": float(team_info.get("npxGA", 0.0)),
                    "deep": int(team_info.get("deep", 0)),
                    "deep_allowed": int(team_info.get("deep_allowed", 0))
                })
                
                # Add scored/missed stats if available
                if "scored" in team_info:
                    team_stats["scored"] = {
                        "total": int(team_info["scored"].get("total", 0)),
                        "open_play": int(team_info["scored"].get("open_play", 0)),
                        "set_pieces": int(team_info["scored"].get("set_pieces", 0)),
                        "counter": int(team_info["scored"].get("counter", 0)),
                        "penalty": int(team_info["scored"].get("penalty", 0)),
                        "own_goals": int(team_info["scored"].get("own_goals", 0))
                    }
                    
                if "missed" in team_info:
                    team_stats["missed"] = {
                        "total": int(team_info["missed"].get("total", 0)),
                        "open_play": int(team_info["missed"].get("open_play", 0)),
                        "set_pieces": int(team_info["missed"].get("set_pieces", 0)),
                        "counter": int(team_info["missed"].get("counter", 0)),
                        "penalty": int(team_info["missed"].get("penalty", 0)),
                        "own_goals": int(team_info["missed"].get("own_goals", 0))
                    }
            
            # Add match data
            if matches_data:
                for date, matches in matches_data.items():
                    for match in matches:
                        match_data = {
                            "id": match["id"],
                            "home_team": match["h"]["title"],
                            "away_team": match["a"]["title"],
                            "match_date": datetime.fromtimestamp(int(match["datetime"])),
                            "home_goals": int(match["goals"]["h"]),
                            "away_goals": int(match["goals"]["a"]),
                            "home_xG": float(match["xG"]["h"]),
                            "away_xG": float(match["xG"]["a"]),
                            "forecast": {
                                "w": float(match["forecast"]["w"]),
                                "d": float(match["forecast"]["d"]),
                                "l": float(match["forecast"]["l"])
                            },
                            "result": match["result"]
                        }
                        team_stats["matches"].append(match_data)
            
            # Add player data
            if players_data:
                for player in players_data:
                    player_data = {
                        "id": player["id"],
                        "player_name": player["player_name"],
                        "games": int(player["games"]),
                        "time": int(player["time"]),
                        "goals": int(player["goals"]),
                        "assists": int(player["assists"]),
                        "shots": int(player["shots"]),
                        "key_passes": int(player["key_passes"]),
                        "xG": float(player["xG"]),
                        "xA": float(player["xA"]),
                        "npg": int(player["npg"]),
                        "npxG": float(player["npxG"]),
                        "xGChain": float(player["xGChain"]),
                        "xGBuildup": float(player["xGBuildup"])
                    }
                    team_stats["players"].append(player_data)
            
            return team_stats
        
        except Exception as e:
            logger.error(f"Error getting team stats for {team_name}: {e}")
            return None
    
    async def get_match_stats(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a specific match.
        
        Args:
            match_id: Match ID
            
        Returns:
            Dictionary with match statistics or None if failed
        """
        # Build URL
        url = f"{self.base_url}/match/{match_id}"
        
        try:
            # Fetch HTML with Playwright (JavaScript rendering)
            html = await self.fetch_html(url, use_playwright=True)
            if not html:
                logger.warning(f"Failed to fetch HTML from {url}")
                return None
            
            # Extract JSON data from the script tags
            match_data = self._extract_json_from_script(html, "match_info")
            shots_data = self._extract_json_from_script(html, "shotsData")
            roster_data = self._extract_json_from_script(html, "rostersData")
            
            if not match_data:
                logger.warning(f"No match data found for match ID {match_id}")
                return None
            
            # Process match statistics
            match_stats = {
                "match_id": match_id,
                "home_team": match_data["h"]["title"],
                "away_team": match_data["a"]["title"],
                "match_date": datetime.fromtimestamp(int(match_data["datetime"])),
                "league": match_data.get("league", ""),
                "season": match_data.get("season", ""),
                "home_score": int(match_data["goals"]["h"]),
                "away_score": int(match_data["goals"]["a"]),
                "home_xg": float(match_data["xG"]["h"]),
                "away_xg": float(match_data["xG"]["a"]),
                "home_team_id": match_data["h"]["id"],
                "away_team_id": match_data["a"]["id"],
                "forecast": {
                    "w": float(match_data["forecast"]["w"]),
                    "d": float(match_data["forecast"]["d"]),
                    "l": float(match_data["forecast"]["l"])
                },
                "result": match_data.get("result", ""),
                "shots": [],
                "home_players": [],
                "away_players": []
            }
            
            # Add shots data - expect list format from test
            if shots_data and isinstance(shots_data, list):
                match_stats["shots"] = shots_data
            
            # Add player data - expect dict with 'h' and 'a' keys from test
            if roster_data and isinstance(roster_data, dict):
                if "h" in roster_data:
                    match_stats["home_players"] = roster_data["h"]
                if "a" in roster_data:
                    match_stats["away_players"] = roster_data["a"]
            
            return match_stats
        
        except Exception as e:
            logger.error(f"Error getting match stats for match ID {match_id}: {e}")
            return None
    
    async def get_player_stats(
        self,
        player_id: str,
        season: str = "2023"
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a specific player.
        
        Args:
            player_id: Player ID
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            Dictionary with player statistics or None if failed
        """
        # Build URL
        url = f"{self.base_url}/player/{player_id}/{season}"
        
        try:
            # Fetch HTML with Playwright (JavaScript rendering)
            html = await self.fetch_html(url, use_playwright=True)
            if not html:
                logger.warning(f"Failed to fetch HTML from {url}")
                return None
            
            # Extract JSON data from the script tags
            player_data = self._extract_json_from_script(html, "playersData")
            matches_data = self._extract_json_from_script(html, "matchesData")
            shots_data = self._extract_json_from_script(html, "shotsData")
            
            if not player_data:
                logger.warning(f"No player data found for player ID {player_id}")
                return None
            
            # Process player statistics
            player_stats = {
                "player_id": player_id,
                "season": season,
                "matches": [],
                "shots": []  # List of shot data
            }
            
            # Add player summary statistics
            if player_data and player_id in player_data:
                player_info = player_data[player_id]
                player_stats.update({
                    "player_name": player_info["player_name"],
                    "team_name": player_info["team_title"],
                    "position": player_info["position"],
                    "games": int(player_info["games"]),
                    "time": int(player_info["time"]),
                    "goals": int(player_info["goals"]),
                    "assists": int(player_info["assists"]),
                    "shots_count": int(player_info["shots"]),  # Renamed to avoid conflict
                    "key_passes": int(player_info["key_passes"]),
                    "xG": float(player_info["xG"]),
                    "xA": float(player_info["xA"]),
                    "npg": int(player_info["npg"]),
                    "npxG": float(player_info["npxG"]),
                    "xGChain": float(player_info["xGChain"]),
                    "xGBuildup": float(player_info["xGBuildup"])
                })
            
            # Add match data
            if matches_data:
                for match in matches_data:
                    match_data = {
                        "id": match["id"],
                        "home_team": match["h_team"],
                        "away_team": match["a_team"],
                        "match_date": datetime.fromtimestamp(int(match["date"])),
                        "minutes_played": int(match["time"]),
                        "position": match["position"],
                        "goals": int(match["goals"]),
                        "assists": int(match["assists"]),
                        "shots": int(match["shots"]),
                        "key_passes": int(match["key_passes"]),
                        "xG": float(match["xG"]),
                        "xA": float(match["xA"]),
                        "npg": int(match["npg"]),
                        "npxG": float(match["npxG"]),
                        "xGChain": float(match["xGChain"]),
                        "xGBuildup": float(match["xGBuildup"])
                    }
                    player_stats["matches"].append(match_data)
            
            # Add shots data
            if shots_data:
                for shot in shots_data:
                    shot_data = {
                        "id": shot["id"],
                        "minute": int(shot["minute"]),
                        "match_id": shot["match_id"],
                        "h_team": shot["h_team"],
                        "a_team": shot["a_team"],
                        "h_goals": int(shot["h_goals"]),
                        "a_goals": int(shot["a_goals"]),
                        "date": datetime.fromtimestamp(int(shot["date"])),
                        "x": float(shot["X"]),
                        "y": float(shot["Y"]),
                        "xG": float(shot["xG"]),
                        "result": shot["result"],
                        "situation": shot["situation"],
                        "shot_type": shot["shotType"],
                        "last_action": shot["lastAction"]
                    }
                    player_stats["shots"].append(shot_data)
            
            return player_stats
        
        except Exception as e:
            logger.error(f"Error getting player stats for player ID {player_id}: {e}")
            return None
        
    async def get_league_data(self, league_code: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive league data from Understat.
        
        Args:
            league_code: Code for the league (e.g., 'premier_league', 'la_liga')
            
        Returns:
            Dictionary containing league data or None if failed
        """
        understat_league = self.LEAGUE_MAPPING.get(league_code)
        if not understat_league:
            logger.warning(f"League {league_code} not supported by Understat")
            return None
            
        try:
            league_data = {
                'league': league_code,
                'source': self.name,
                'timestamp': datetime.now().isoformat(),
                'teams': None,
                'standings': None,
                'recent_matches': None,
                'player_stats': None
            }
            
            # Get teams data
            teams_data = await self.get_teams(league_code)
            if teams_data is not None and (isinstance(teams_data, pd.DataFrame) and not teams_data.empty or isinstance(teams_data, list) and len(teams_data) > 0):
                if isinstance(teams_data, pd.DataFrame):
                    league_data['teams'] = teams_data.to_dict('records')
                else:
                    league_data['teams'] = teams_data
                
            # Get standings/league table
            standings_data = await self.get_league_table(league_code)
            if standings_data is not None and (isinstance(standings_data, pd.DataFrame) and not standings_data.empty or isinstance(standings_data, list) and len(standings_data) > 0):
                if isinstance(standings_data, pd.DataFrame):
                    league_data['standings'] = standings_data.to_dict('records')
                else:
                    league_data['standings'] = standings_data
                
            # Get recent matches with xG data
            matches_data = await self.get_matches(league_code)
            if matches_data is not None and (isinstance(matches_data, pd.DataFrame) and not matches_data.empty or isinstance(matches_data, list) and len(matches_data) > 0):
                if isinstance(matches_data, pd.DataFrame):
                    league_data['recent_matches'] = matches_data.to_dict('records')
                else:
                    league_data['recent_matches'] = matches_data
                
            # Get top player stats
            try:
                # Since get_player_stats expects a player_id, not league_code,
                # we'll skip this for now and add proper league player stats later
                logger.info(f"Player stats retrieval skipped for {league_code} - method needs league-level implementation")
                
            except Exception as e:
                logger.warning(f"Failed to get player stats from Understat: {e}")
            
            # Check if we got any data
            has_data = any([
                league_data.get('teams', []),
                league_data.get('standings', []),
                league_data.get('recent_matches', []),
                league_data.get('player_stats', [])
            ])
            
            if has_data:
                logger.info(f"Successfully retrieved Understat league data for {league_code}")
                return league_data
            else:
                logger.warning(f"No Understat league data available for {league_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting Understat league data for {league_code}: {e}")
            return None
    
    async def get_teams(
        self,
        league_id: str,
        season: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get teams for a specific league.
        
        Args:
            league_id: League ID (e.g., 'PL', 'BL1')
            season: Optional season (e.g., '2023')
            
        Returns:
            List of team data dictionaries
        """
        # Map league_id to Understat format
        # First try direct mapping
        understat_league = league_id
        
        # If not a direct match, try to map from standard codes
        if league_id not in self.LEAGUE_MAPPING.values():
            # Reverse mapping: Check if this is a standard code like 'PL'
            for std_code, understat_code in self.LEAGUE_MAPPING.items():
                if std_code == league_id or league_id.lower() == std_code.lower():
                    understat_league = understat_code
                    break
            
            # If still not found, try to map from our internal league codes
            if understat_league == league_id:
                for internal_code, understat_code in self.LEAGUE_MAPPING.items():
                    # Convert internal codes (premier_league) to standard (PL)
                    league_id_map = {
                        "premier_league": "PL",
                        "la_liga": "PD",
                        "bundesliga": "BL1",
                        "serie_a": "SA",
                        "ligue_1": "FL1",
                        "eredivisie": "DED"
                    }
                    
                    if league_id == league_id_map.get(internal_code):
                        understat_league = understat_code
                        break
        
        if not season:
            # Default to current season
            current_year = datetime.now().year
            # If after June, use the current year, otherwise use last year
            # For 2025, the season likely won't be available yet, so use 2024
            if current_year == 2025:
                season = "2024"
            else:
                season = str(current_year if datetime.now().month > 6 else current_year - 1)
        elif season == "2025":
            # 2025 season data likely not available, fallback to 2024
            logger.warning("Season 2025 data may not be available on Understat. Using 2024 season.")
            season = "2024"
        
        url = f"{self.base_url}/league/{understat_league}/{season}"
        logger.info(f"Getting teams from Understat for {league_id} (URL: {url})")
        
        try:
            # Fetch HTML with Playwright (JavaScript rendering)
            html = await self.fetch_html(url, use_playwright=True)
            if not html:
                logger.warning(f"Failed to fetch HTML from {url}")
                return []
            
            # Extract JSON data from the script tag - this contains team data
            teams_data = self._extract_json_from_script(html, "teamsData")
            if not teams_data:
                logger.warning(f"No teams data found for {league_id} season {season}")
                return []
            
            teams = []
            for team_id, team_data in teams_data.items():
                team = {
                    "id": team_id,
                    "name": team_data.get("title", "Unknown"),
                    "api_id": team_id,
                    "league_id": league_id,
                    "short_name": team_data.get("title", "")[:3].upper()  # Generate short name if needed
                }
                teams.append(team)
            
            logger.info(f"Found {len(teams)} teams for league {league_id}")
            return teams
        except Exception as e:
            logger.error(f"Error getting teams from Understat for league {league_id}: {e}")
            return []
    
    async def get_league_table(self, league_code: str) -> Optional[pd.DataFrame]:
        """
        Stub for get_league_table to prevent attribute errors. Should be implemented to fetch league table data.
        """
        logger.warning("get_league_table is not yet implemented for UnderstatScraper.")
        return pd.DataFrame()

    async def get_session(self):
        """
        Get an aiohttp ClientSession.
        
        Returns:
            aiohttp ClientSession
        """
        # Use the http_client to get a session
        return self.http_client.get_session()
