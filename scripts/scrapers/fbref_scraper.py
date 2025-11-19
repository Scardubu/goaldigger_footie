"""
FBref scraper for retrieving detailed player and team statistics.
"""
import asyncio
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

class FBrefScraper(BaseScraper):
    """
    Scraper for FBref.com.
    Retrieves detailed player and team statistics.
    """
    
    # League mapping
    LEAGUE_MAPPING = {
        "premier_league": "Premier-League",
        "la_liga": "La-Liga",
        "bundesliga": "Bundesliga",
        "serie_a": "Serie-A",
        "ligue_1": "Ligue-1",
        "eredivisie": "Eredivisie"
    }
    
    # Season mapping (FBref uses different season formats)
    SEASON_MAPPING = {
        "2023": "2023-2024",
        "2022": "2022-2023",
        "2021": "2021-2022",
        "2020": "2020-2021",
        "2019": "2019-2020"
    }
    
    def __init__(
        self,
        http_client: Optional[HttpClient] = None,
        proxy_manager: Optional[ProxyManager] = None,
        playwright_manager: Optional[PlaywrightManager] = None,
        use_proxies: bool = True,
        rate_limit_delay: Tuple[float, float] = (2.0, 4.0),
        config=None,
        **kwargs
    ):
        """
        Initialize the FBref scraper.
        
        Args:
            http_client: Optional HTTP client to use
            proxy_manager: Optional proxy manager to use
            playwright_manager: Optional playwright manager for JS rendering
            use_proxies: Whether to use proxies
            rate_limit_delay: Tuple of (min_delay, max_delay) in seconds
        """
        super().__init__(
            name="FBref",
            base_url="https://fbref.com",
            http_client=http_client,
            proxy_manager=proxy_manager,
            playwright_manager=playwright_manager,
            use_proxies=use_proxies,
            use_playwright=True,  # FBref has some JavaScript elements
            rate_limit_delay=rate_limit_delay,
            config=config
        )
    
    async def get_league_table(
        self,
        league_code: str,
        season: str = "2023"
    ) -> Optional[pd.DataFrame]:
        """
        Get league table (standings) for a specific league and season.
        
        Args:
            league_code: League code (e.g., "premier_league", "la_liga")
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            DataFrame with league table data or None if failed
        """
        # Convert league code and season to FBref format
        fbref_league = self.LEAGUE_MAPPING.get(league_code)
        fbref_season = self.SEASON_MAPPING.get(season)
        
        if not fbref_league or not fbref_season:
            logger.error(f"Invalid league code or season: {league_code}, {season}")
            return None
        
        # Build URL
        url = f"{self.base_url}/en/comps/{self._get_league_id(league_code)}/{fbref_season}/{fbref_league}-Stats"
        
        try:
            # Fetch HTML
            soup = await self.fetch_and_parse(url)
            if not soup:
                logger.warning(f"Failed to fetch or parse HTML from {url}")
                return None
            
            # Find the league table
            table = soup.find('table', {'id': 'results2023-2024' + self._get_league_id(league_code) + '_overall'})
            if not table:
                # Try alternative table ID
                table = soup.find('table', {'id': 'stats_squads_standard_for'})
            
            if not table:
                logger.warning(f"League table not found for {league_code}")
                return None
            
            # Extract table data
            rows = table.find_all('tr')
            headers = [th.text.strip() for th in rows[0].find_all('th')]
            
            # Remove rank header if present
            if headers[0] == '#' or headers[0] == 'Rk':
                headers = headers[1:]
            
            # Process table data
            data = []
            for row in rows[1:]:
                if 'class' in row.attrs and 'thead' in row.attrs['class']:
                    continue
                
                cols = row.find_all(['td', 'th'])
                if len(cols) <= 1:
                    continue
                
                # Extract team name from the first column
                team_name = cols[0].text.strip()
                
                # Extract other stats
                row_data = {'team_name': team_name}
                for i, col in enumerate(cols[1:], 1):
                    if i < len(headers):
                        row_data[headers[i]] = col.text.strip()
                
                data.append(row_data)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            numeric_cols = ['MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Rename columns to standard format
            column_mapping = {
                'MP': 'matches_played',
                'W': 'wins',
                'D': 'draws',
                'L': 'losses',
                'GF': 'goals_for',
                'GA': 'goals_against',
                'GD': 'goal_difference',
                'Pts': 'points'
            }
            df = df.rename(columns=column_mapping)
            
            # Add league and season information
            df['league'] = league_code
            df['season'] = season
            
            # Sanitize data
            string_columns = ['team_name', 'league', 'season']
            numeric_columns = ['matches_played', 'wins', 'draws', 'losses', 
                              'goals_for', 'goals_against', 'goal_difference', 'points']
            
            return sanitize_dataframe(
                df,
                string_columns=string_columns,
                numeric_columns=numeric_columns
            )
        
        except Exception as e:
            logger.error(f"Error getting league table for {league_code}: {e}")
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
        # Convert league code and season to FBref format
        fbref_league = self.LEAGUE_MAPPING.get(league_code)
        fbref_season = self.SEASON_MAPPING.get(season)
        
        if not fbref_league or not fbref_season:
            logger.error(f"Invalid league code or season: {league_code}, {season}")
            return None
        
        # Search for the team to get its URL
        team_url = await self._search_team(team_name, league_code, season)
        if not team_url:
            logger.warning(f"Team not found: {team_name}")
            return None
        
        try:
            # Fetch HTML
            soup = await self.fetch_and_parse(team_url)
            if not soup:
                logger.warning(f"Failed to fetch or parse HTML from {team_url}")
                return None
            
            # Extract team statistics
            team_stats = {
                "team_name": team_name,
                "league": league_code,
                "season": season,
                "standard_stats": {},
                "goalkeeper_stats": {},
                "shooting_stats": {},
                "passing_stats": {},
                "possession_stats": {},
                "defensive_stats": {},
                "fixtures": []
            }
            
            # Extract standard stats
            standard_table = soup.find('table', {'id': 'stats_squads_standard_for'})
            if standard_table:
                team_stats["standard_stats"] = self._extract_table_data(standard_table)
            
            # Extract goalkeeper stats
            gk_table = soup.find('table', {'id': 'stats_squads_keeper_for'})
            if gk_table:
                team_stats["goalkeeper_stats"] = self._extract_table_data(gk_table)
            
            # Extract shooting stats
            shooting_table = soup.find('table', {'id': 'stats_squads_shooting_for'})
            if shooting_table:
                team_stats["shooting_stats"] = self._extract_table_data(shooting_table)
            
            # Extract passing stats
            passing_table = soup.find('table', {'id': 'stats_squads_passing_for'})
            if passing_table:
                team_stats["passing_stats"] = self._extract_table_data(passing_table)
            
            # Extract possession stats
            possession_table = soup.find('table', {'id': 'stats_squads_possession_for'})
            if possession_table:
                team_stats["possession_stats"] = self._extract_table_data(possession_table)
            
            # Extract defensive stats
            defensive_table = soup.find('table', {'id': 'stats_squads_defense_for'})
            if defensive_table:
                team_stats["defensive_stats"] = self._extract_table_data(defensive_table)
            
            # Extract fixtures
            fixtures_table = soup.find('table', {'id': 'matchlogs_for'})
            if fixtures_table:
                fixtures = self._extract_fixtures(fixtures_table)
                team_stats["fixtures"] = fixtures
            
            return team_stats
        
        except Exception as e:
            logger.error(f"Error getting team stats for {team_name}: {e}")
            return None
    
    async def get_player_stats(
        self,
        player_name: str,
        team_name: Optional[str] = None,
        season: str = "2023"
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed player statistics.
        
        Args:
            player_name: Player name
            team_name: Optional team name to narrow search
            season: Season (e.g., "2023" for 2023/2024)
            
        Returns:
            Dictionary with player statistics or None if failed
        """
        # Search for the player to get their URL
        player_url = await self._search_player(player_name, team_name)
        if not player_url:
            logger.warning(f"Player not found: {player_name}")
            return None
        
        try:
            # Fetch HTML
            soup = await self.fetch_and_parse(player_url)
            if not soup:
                logger.warning(f"Failed to fetch or parse HTML from {player_url}")
                return None
            
            # Extract player information
            player_info = {}
            player_info_div = soup.find('div', {'itemtype': 'https://schema.org/Person'})
            if player_info_div:
                # Extract name
                name_h1 = player_info_div.find('h1')
                if name_h1:
                    player_info["name"] = name_h1.text.strip()
                
                # Extract other information
                info_items = player_info_div.find_all('p')
                for item in info_items:
                    text = item.text.strip()
                    if "Position:" in text:
                        player_info["position"] = text.replace("Position:", "").strip()
                    elif "Footed:" in text:
                        player_info["footed"] = text.replace("Footed:", "").strip()
                    elif "Born:" in text:
                        player_info["born"] = text.replace("Born:", "").strip()
                    elif "National Team:" in text:
                        player_info["national_team"] = text.replace("National Team:", "").strip()
            
            # Extract player statistics
            player_stats = {
                "player_info": player_info,
                "season": season,
                "standard_stats": {},
                "shooting_stats": {},
                "passing_stats": {},
                "possession_stats": {},
                "defensive_stats": {},
                "matches": []
            }
            
            # Extract standard stats
            standard_table = soup.find('table', {'id': 'stats_standard_dom_lg'})
            if standard_table:
                player_stats["standard_stats"] = self._extract_table_data(standard_table)
            
            # Extract shooting stats
            shooting_table = soup.find('table', {'id': 'stats_shooting_dom_lg'})
            if shooting_table:
                player_stats["shooting_stats"] = self._extract_table_data(shooting_table)
            
            # Extract passing stats
            passing_table = soup.find('table', {'id': 'stats_passing_dom_lg'})
            if passing_table:
                player_stats["passing_stats"] = self._extract_table_data(passing_table)
            
            # Extract possession stats
            possession_table = soup.find('table', {'id': 'stats_possession_dom_lg'})
            if possession_table:
                player_stats["possession_stats"] = self._extract_table_data(possession_table)
            
            # Extract defensive stats
            defensive_table = soup.find('table', {'id': 'stats_defense_dom_lg'})
            if defensive_table:
                player_stats["defensive_stats"] = self._extract_table_data(defensive_table)
            
            # Extract match logs
            matchlogs_table = soup.find('table', {'id': 'matchlogs_all'})
            if matchlogs_table:
                matches = self._extract_match_logs(matchlogs_table)
                player_stats["matches"] = matches
            
            return player_stats
        
        except Exception as e:
            logger.error(f"Error getting player stats for {player_name}: {e}")
            return None
    
    async def get_match_report(self, match_url: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed match report.
        
        Args:
            match_url: URL of the match report
            
        Returns:
            Dictionary with match report data or None if failed
        """
        try:
            # Fetch HTML
            soup = await self.fetch_and_parse(match_url)
            if not soup:
                logger.warning(f"Failed to fetch or parse HTML from {match_url}")
                return None
            
            # Extract match information
            match_info = {}
            
            # Extract teams
            scorebox = soup.find('div', {'class': 'scorebox'})
            if scorebox:
                teams = scorebox.find_all('div', {'class': 'scorebox_meta'})
                if len(teams) >= 2:
                    match_info["home_team"] = teams[0].find('strong').text.strip()
                    match_info["away_team"] = teams[1].find('strong').text.strip()
                
                # Extract score
                scores = scorebox.find_all('div', {'class': 'score'})
                if len(scores) >= 2:
                    match_info["home_score"] = int(scores[0].text.strip())
                    match_info["away_score"] = int(scores[1].text.strip())
            
            # Extract match date
            scorebox_meta = soup.find('div', {'class': 'scorebox_meta'})
            if scorebox_meta:
                date_element = scorebox_meta.find('span')
                if date_element:
                    match_info["date"] = date_element.text.strip()
            
            # Extract match stats
            match_stats = {
                "match_info": match_info,
                "team_stats": {},
                "player_stats": {
                    "home": [],
                    "away": []
                }
            }
            
            # Extract team stats
            team_stats_table = soup.find('table', {'id': 'team_stats'})
            if team_stats_table:
                match_stats["team_stats"] = self._extract_team_stats(team_stats_table)
            
            # Extract player stats
            player_stats_tables = soup.find_all('table', {'class': 'stats_table'})
            for table in player_stats_tables:
                if 'id' in table.attrs and table.attrs['id'].startswith('stats_'):
                    if 'home' in table.attrs['id']:
                        match_stats["player_stats"]["home"] = self._extract_player_match_stats(table)
                    elif 'away' in table.attrs['id']:
                        match_stats["player_stats"]["away"] = self._extract_player_match_stats(table)
            
            return match_stats
        
        except Exception as e:
            logger.error(f"Error getting match report from {match_url}: {e}")
            return None
    
    def _get_league_id(self, league_code: str) -> str:
        """Get FBref league ID."""
        league_ids = {
            "premier_league": "9",
            "la_liga": "12",
            "bundesliga": "20",
            "serie_a": "11",
            "ligue_1": "13",
            "eredivisie": "23"
        }
        return league_ids.get(league_code, "")
    
    async def _search_team(
        self,
        team_name: str,
        league_code: str,
        season: str
    ) -> Optional[str]:
        """
        Search for a team and return its URL.
        
        Args:
            team_name: Team name
            league_code: League code
            season: Season
            
        Returns:
            Team URL or None if not found
        """
        # Build search URL
        search_url = f"{self.base_url}/en/search/search.fcgi?search={team_name}"
        
        try:
            # Fetch HTML
            soup = await self.fetch_and_parse(search_url)
            if not soup:
                logger.warning(f"Failed to fetch or parse HTML from {search_url}")
                return None
            
            # Find team links
            team_links = soup.find_all('div', {'class': 'search-item-name'})
            for link in team_links:
                a_tag = link.find('a')
                if a_tag and 'href' in a_tag.attrs:
                    href = a_tag.attrs['href']
                    # Check if it's a team link
                    if '/squads/' in href:
                        # Check if it's the correct team
                        team_text = a_tag.text.strip()
                        if team_name.lower() in team_text.lower():
                            return f"{self.base_url}{href}"
            
            logger.warning(f"Team not found in search results: {team_name}")
            return None
        
        except Exception as e:
            logger.error(f"Error searching for team {team_name}: {e}")
            return None
    
    async def _search_player(
        self,
        player_name: str,
        team_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Search for a player and return their URL.
        
        Args:
            player_name: Player name
            team_name: Optional team name to narrow search
            
        Returns:
            Player URL or None if not found
        """
        # Build search URL
        search_term = player_name
        if team_name:
            search_term = f"{player_name} {team_name}"
        
        search_url = f"{self.base_url}/en/search/search.fcgi?search={search_term}"
        
        try:
            # Fetch HTML
            soup = await self.fetch_and_parse(search_url)
            if not soup:
                logger.warning(f"Failed to fetch or parse HTML from {search_url}")
                return None
            
            # Find player links
            player_links = soup.find_all('div', {'class': 'search-item-name'})
            for link in player_links:
                a_tag = link.find('a')
                if a_tag and 'href' in a_tag.attrs:
                    href = a_tag.attrs['href']
                    # Check if it's a player link
                    if '/players/' in href:
                        # Check if it's the correct player
                        player_text = a_tag.text.strip()
                        if player_name.lower() in player_text.lower():
                            # If team name is provided, check if it matches
                            if team_name:
                                if team_name.lower() in player_text.lower():
                                    return f"{self.base_url}{href}"
                            else:
                                return f"{self.base_url}{href}"
            
            logger.warning(f"Player not found in search results: {player_name}")
            return None
        
        except Exception as e:
            logger.error(f"Error searching for player {player_name}: {e}")
            return None
    
    def _extract_table_data(self, table) -> Dict[str, Any]:
        """
        Extract data from a table.
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            Dictionary with table data
        """
        data = {}
        
        try:
            # Extract headers
            headers = []
            header_row = table.find('thead').find('tr')
            for th in header_row.find_all('th'):
                header_text = th.text.strip()
                if header_text:
                    headers.append(header_text)
            
            # Extract data rows
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                # Skip header rows
                if 'class' in row.attrs and 'thead' in row.attrs['class']:
                    continue
                
                cols = row.find_all(['td', 'th'])
                if len(cols) <= 1:
                    continue
                
                # Extract row data
                row_data = {}
                for i, col in enumerate(cols):
                    if i < len(headers):
                        header = headers[i]
                        value = col.text.strip()
                        row_data[header] = value
                
                # Use the first column as the key (usually team or player name)
                key = list(row_data.keys())[0]
                data[row_data[key]] = row_data
        
        except Exception as e:
            logger.error(f"Error extracting table data: {e}")
        
        return data
    
    def _extract_fixtures(self, table) -> List[Dict[str, Any]]:
        """
        Extract fixtures data from a table.
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            List of fixtures
        """
        fixtures = []
        
        try:
            # Extract headers
            headers = []
            header_row = table.find('thead').find('tr')
            for th in header_row.find_all('th'):
                header_text = th.text.strip()
                if header_text:
                    headers.append(header_text)
            
            # Extract data rows
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                # Skip header rows
                if 'class' in row.attrs and 'thead' in row.attrs['class']:
                    continue
                
                cols = row.find_all(['td', 'th'])
                if len(cols) <= 1:
                    continue
                
                # Extract row data
                fixture = {}
                for i, col in enumerate(cols):
                    if i < len(headers):
                        header = headers[i]
                        value = col.text.strip()
                        fixture[header] = value
                        
                        # Extract match URL if available
                        if header == 'Match Report' and col.find('a'):
                            fixture['match_url'] = self.base_url + col.find('a')['href']
                
                fixtures.append(fixture)
        
        except Exception as e:
            logger.error(f"Error extracting fixtures: {e}")
        
        return fixtures
    
    def _extract_match_logs(self, table) -> List[Dict[str, Any]]:
        """
        Extract match logs data from a table.
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            List of match logs
        """
        match_logs = []
        
        try:
            # Extract headers
            headers = []
            header_row = table.find('thead').find('tr')
            for th in header_row.find_all('th'):
                header_text = th.text.strip()
                if header_text:
                    headers.append(header_text)
            
            # Extract data rows
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                # Skip header rows
                if 'class' in row.attrs and 'thead' in row.attrs['class']:
                    continue
                
                cols = row.find_all(['td', 'th'])
                if len(cols) <= 1:
                    continue
                
                # Extract row data
                match_log = {}
                for i, col in enumerate(cols):
                    if i < len(headers):
                        header = headers[i]
                        value = col.text.strip()
                        match_log[header] = value
                        
                        # Extract match URL if available
                        if header == 'Match Report' and col.find('a'):
                            match_log['match_url'] = self.base_url + col.find('a')['href']
                
                match_logs.append(match_log)
        
        except Exception as e:
            logger.error(f"Error extracting match logs: {e}")
        
        return match_logs
    
    def _extract_team_stats(self, table) -> Dict[str, Any]:
        """
        Extract team stats from a table.
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            Dictionary with team stats
        """
        team_stats = {
            "home": {},
            "away": {}
        }
        
        try:
            # Extract rows
            rows = table.find_all('tr')
            for row in rows:
                # Extract stat name and values
                cols = row.find_all(['th', 'td'])
                if len(cols) >= 3:
                    stat_name = cols[0].text.strip()
                    home_value = cols[1].text.strip()
                    away_value = cols[2].text.strip()
                    
                    team_stats["home"][stat_name] = home_value
                    team_stats["away"][stat_name] = away_value
        
        except Exception as e:
            logger.error(f"Error extracting team stats: {e}")
        
        return team_stats
    
    def _extract_player_match_stats(self, table) -> List[Dict[str, Any]]:
        """
        Extract player match stats from a table.
        
        Args:
            table: BeautifulSoup table element
            
        Returns:
            List of player match stats
        """
        player_stats = []
        
        try:
            # Extract headers
            headers = []
            header_row = table.find('thead').find('tr')
            for th in header_row.find_all('th'):
                header_text = th.text.strip()
                if header_text:
                    headers.append(header_text)
            
            # Extract data rows
            rows = table.find('tbody').find_all('tr')
            for row in rows:
                # Skip header rows
                if 'class' in row.attrs and 'thead' in row.attrs['class']:
                    continue
                
                cols = row.find_all(['td', 'th'])
                if len(cols) <= 1:
                    continue
                
                # Extract row data
                player_stat = {}
                for i, col in enumerate(cols):
                    if i < len(headers):
                        header = headers[i]
                        value = col.text.strip()
                        player_stat[header] = value
                        
                        # Extract player URL if available
                        if i == 0 and col.find('a'):
                            player_stat['player_url'] = self.base_url + col.find('a')['href']
                
                player_stats.append(player_stat)
        
        except Exception as e:
            logger.error(f"Error extracting player match stats: {e}")
        
        return player_stats
