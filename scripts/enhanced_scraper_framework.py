#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Scraper Framework
==========================

Advanced web scraping framework with stealth capabilities, multiple data sources,
and real-time data integration for improved prediction accuracy.
"""

import asyncio
import hashlib
import json
import logging
import random

# Add project root to path
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiohttp
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

sys.path.append('.')

from database.db_manager import DatabaseManager
from database.schema import League, Match, Odds, Prediction, ScrapedData, Team
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingConfig:
    """Configuration for scraping behavior"""
    user_agent_rotation: bool = True
    proxy_rotation: bool = True
    delay_range: tuple = (1.0, 3.0)
    max_retries: int = 3
    timeout: int = 30
    stealth_mode: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour

class UserAgentManager:
    """Manages user agent rotation for stealth scraping"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self.current_index = 0
    
    def get_next_user_agent(self) -> str:
        """Get next user agent in rotation"""
        user_agent = self.user_agents[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.user_agents)
        return user_agent

class ProxyManager:
    """Manages proxy rotation for stealth scraping"""
    
    def __init__(self):
        self.proxies = self.load_proxy_list()
        self.current_index = 0
    
    def load_proxy_list(self) -> List[str]:
        """Load proxy list from file or environment"""
        proxy_file = Path('config/proxies.txt')
        if proxy_file.exists():
            with open(proxy_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            # Return empty list if no proxy file
            return []
    
    def get_next_proxy(self) -> Optional[str]:
        """Get next proxy in rotation"""
        if not self.proxies:
            return None
        
        proxy = self.proxies[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.proxies)
        return proxy

class DataSource:
    """Base class for data sources"""
    
    def __init__(self, name: str, base_url: str, config: ScrapingConfig, **kwargs):
        self.name = name
        self.base_url = base_url
        self.config = config
        self.user_agent_manager = UserAgentManager()
        self.proxy_manager = ProxyManager()
        # Accept and ignore any extra kwargs for compatibility
    
    async def fetch_data(self, url: str, headers: Dict[str, str] = None) -> Optional[str]:
        """Fetch data from URL with retry logic"""
        headers = headers or {}
        
        for attempt in range(self.config.max_retries):
            try:
                # Add delay between requests
                if attempt > 0:
                    delay = random.uniform(*self.config.delay_range)
                    await asyncio.sleep(delay)
                
                # Set up headers
                if self.config.user_agent_rotation:
                    headers['User-Agent'] = self.user_agent_manager.get_next_user_agent()
                
                # Set up proxy
                proxy = None
                if self.config.proxy_rotation:
                    proxy = self.proxy_manager.get_next_proxy()
                
                # Make request
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        headers=headers,
                        proxy=proxy,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                    ) as response:
                        if response.status == 200:
                            return await response.text()
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")
                            
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"All attempts failed for {url}")
                    return None
        
        return None

    async def get_league_data(self, league_code: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive league data including teams, standings, and recent matches.
        
        Args:
            league_code: Code for the league (e.g., 'premier_league', 'la_liga')
            
        Returns:
            Dictionary containing league data or None if failed
        """
        try:
            # Default implementation attempts to gather data from available methods
            league_data = {
                'league': league_code,
                'source': self.name,
                'timestamp': datetime.now().isoformat(),
                'teams': None,
                'standings': None,
                'recent_matches': None,
                'upcoming_matches': None
            }
            
            # Try to get teams if method exists
            if hasattr(self, 'get_teams'):
                try:
                    league_data['teams'] = await self.get_teams(league_code)
                except Exception as e:
                    logger.warning(f"Failed to get teams from {self.name}: {e}")
            
            # Try to get standings if method exists
            if hasattr(self, 'get_standings'):
                try:
                    league_data['standings'] = await self.get_standings(league_code)
                except Exception as e:
                    logger.warning(f"Failed to get standings from {self.name}: {e}")
            
            # Try to get recent matches if method exists
            if hasattr(self, 'get_matches'):
                try:
                    # Get recent matches (last 7 days)
                    date_from = datetime.now() - timedelta(days=7)
                    league_data['recent_matches'] = await self.get_matches(
                        league_code, 
                        date_from=date_from,
                        status="FINISHED"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get recent matches from {self.name}: {e}")
                    
                try:
                    # Get upcoming matches (next 7 days)
                    date_to = datetime.now() + timedelta(days=7)
                    league_data['upcoming_matches'] = await self.get_matches(
                        league_code,
                        date_to=date_to,
                        status="SCHEDULED"
                    )
                except Exception as e:
                    logger.warning(f"Failed to get upcoming matches from {self.name}: {e}")
            
            # Check if we got any data
            has_data = any([
                league_data['teams'] is not None,
                league_data['standings'] is not None,
                league_data['recent_matches'] is not None,
                league_data['upcoming_matches'] is not None
            ])
            
            if has_data:
                logger.info(f"Successfully retrieved league data for {league_code} from {self.name}")
                return league_data
            else:
                logger.warning(f"No league data available for {league_code} from {self.name}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting league data for {league_code} from {self.name}: {e}")
            return None

class TransfermarktScraper(DataSource):
    """Scraper for Transfermarkt data"""
    
    def __init__(self, config: ScrapingConfig):
        super().__init__("Transfermarkt", "https://www.transfermarkt.com", config)
    
    async def get_team_form(self, team_name: str) -> Optional[Dict[str, Any]]:
        """Get team form data from Transfermarkt"""
        logger.info(f"Scraping team form for {team_name} from Transfermarkt")
        
        # Construct search URL
        search_url = f"{self.base_url}/schnellsuche/ergebnis/schnellsuche?query={team_name.replace(' ', '+')}"
        
        html = await self.fetch_data(search_url)
        if not html:
            return None
        
        # Parse team form data
        soup = BeautifulSoup(html, 'html.parser')
        return self.parse_team_form(soup, team_name)
    
    def parse_team_form(self, soup: BeautifulSoup, team_name: str) -> Optional[Dict[str, Any]]:
        """Parse team form data from HTML"""
        try:
            form_data = {
                'team_name': team_name,
                'source': self.name,
                'last_5_matches': [],
                'goals_scored': 0,
                'goals_conceded': 0,
                'points': 0,
                'form_rating': 0.0
            }
            
            # Look for form table
            form_table = soup.find('table', {'class': 'haupttabelle'})
            if form_table:
                rows = form_table.find_all('tr')[1:6]  # Last 5 matches
                
                for row in rows:
                    match_data = self.parse_match_row(row)
                    if match_data:
                        form_data['last_5_matches'].append(match_data)
            
            # Calculate form rating
            if form_data['last_5_matches']:
                wins = sum(1 for m in form_data['last_5_matches'] if m['result'] == 'W')
                draws = sum(1 for m in form_data['last_5_matches'] if m['result'] == 'D')
                form_data['form_rating'] = (wins * 3 + draws) / (len(form_data['last_5_matches']) * 3)
            
            return form_data
            
        except Exception as e:
            logger.error(f"Error parsing team form: {e}")
            return None
    
    def parse_match_row(self, row) -> Optional[Dict[str, Any]]:
        """Parse individual match row"""
        try:
            cols = row.find_all('td')
            if len(cols) < 3:
                return None
            
            # Extract match data
            date = cols[0].text.strip()
            opponent = cols[1].text.strip()
            score = cols[2].text.strip()
            
            # Determine result
            if 'W' in score or score.startswith('2') or score.startswith('3'):
                result = 'W'
            elif 'D' in score or score.endswith('-'):
                result = 'D'
            else:
                result = 'L'
            
            return {
                'date': date,
                'opponent': opponent,
                'score': score,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Error parsing match row: {e}")
            return None

class UnderstatScraper(DataSource):
    """Scraper for Understat data
    
    DEPRECATED: This class is deprecated in favor of UnderstatAPIClient
    which provides more comprehensive API access. This scraper is
    maintained only for backward compatibility with legacy code.
    """
    
    def __init__(self, config: ScrapingConfig):
        super().__init__("Understat", "https://understat.com", config)
        logger.warning(
            "UnderstatScraper is deprecated. Use api.understat_client.UnderstatAPIClient instead."
        )
    
    async def get_match_xg_data(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Get expected goals data from Understat"""
        logger.info(f"Scraping xG data for match {match_id} from Understat")
        
        url = f"{self.base_url}/match/{match_id}"
        html = await self.fetch_data(url)
        
        if not html:
            return None
        
        return self.parse_xg_data(html, match_id)
    
    def parse_xg_data(self, html: str, match_id: str) -> Optional[Dict[str, Any]]:
        """Parse xG data from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract xG data from JavaScript
            scripts = soup.find_all('script')
            for script in scripts:
                if 'xG' in script.text:
                    # Parse JavaScript data
                    data = self.extract_js_data(script.text)
                    if data:
                        return {
                            'match_id': match_id,
                            'source': self.name,
                            'home_xg': data.get('home_xg', 0),
                            'away_xg': data.get('away_xg', 0),
                            'home_goals': data.get('home_goals', 0),
                            'away_goals': data.get('away_goals', 0)
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing xG data: {e}")
            return None
    
    def extract_js_data(self, script_text: str) -> Optional[Dict[str, Any]]:
        """Extract data from JavaScript"""
        try:
            # Look for data patterns in JavaScript
            import re

            # Extract xG values
            xg_pattern = r'xG["\']?\s*:\s*\[([^\]]+)\]'
            xg_match = re.search(xg_pattern, script_text)
            
            if xg_match:
                xg_values = xg_match.group(1).split(',')
                home_xg = float(xg_values[0]) if len(xg_values) > 0 else 0
                away_xg = float(xg_values[1]) if len(xg_values) > 1 else 0
                
                return {
                    'home_xg': home_xg,
                    'away_xg': away_xg
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting JS data: {e}")
            return None

class FBrefScraper(DataSource):
    """Scraper for FBref data"""
    
    def __init__(self, config: ScrapingConfig):
        super().__init__("FBref", "https://fbref.com", config)
    
    async def get_team_stats(self, team_name: str, league: str) -> Optional[Dict[str, Any]]:
        """Get team statistics from FBref"""
        logger.info(f"Scraping team stats for {team_name} from FBref")
        
        # Construct team URL
        team_url = f"{self.base_url}/en/squads/{self.get_team_id(team_name)}"
        html = await self.fetch_data(team_url)
        
        if not html:
            return None
        
        return self.parse_team_stats(html, team_name)
    
    def get_team_id(self, team_name: str) -> str:
        """Get team ID from team name"""
        # This would need to be implemented with a mapping or search
        # For now, return a placeholder
        return "placeholder"
    
    def parse_team_stats(self, html: str, team_name: str) -> Optional[Dict[str, Any]]:
        """Parse team statistics from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            stats = {
                'team_name': team_name,
                'source': self.name,
                'goals_scored': 0,
                'goals_conceded': 0,
                'shots': 0,
                'shots_on_target': 0,
                'possession': 0.0
            }
            
            # Extract stats from tables
            stats_table = soup.find('table', {'id': 'stats_squads_standard_for'})
            if stats_table:
                # Parse standard stats
                pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Error parsing team stats: {e}")
            return None

class EnhancedScraperFramework:
    """Main scraper framework that coordinates all data sources"""
    
    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        self.db = DatabaseManager()
        
        # Initialize data sources
        self.sources = {
            'transfermarkt': TransfermarktScraper(self.config),
            'understat': UnderstatScraper(self.config),
            'fbref': FBrefScraper(self.config)
        }
        
        # Cache for storing scraped data
        self.cache = {}
    
    async def scrape_team_data(self, team_name: str) -> Dict[str, Any]:
        """Scrape comprehensive team data from all sources"""
        logger.info(f"Scraping comprehensive data for {team_name}")
        
        team_data = {
            'team_name': team_name,
            'form_data': {},
            'stats_data': {},
            'xg_data': {},
            'timestamp': datetime.now()
        }
        
        # Scrape from all sources concurrently
        tasks = []
        
        # Transfermarkt form data
        tasks.append(self.sources['transfermarkt'].get_team_form(team_name))
        
        # FBref stats data
        tasks.append(self.sources['fbref'].get_team_stats(team_name, "Premier League"))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        if results[0] and not isinstance(results[0], Exception):
            team_data['form_data'] = results[0]
        
        if results[1] and not isinstance(results[1], Exception):
            team_data['stats_data'] = results[1]
        
        # Store in database
        await self.store_scraped_data(team_data)
        
        return team_data
    
    async def store_scraped_data(self, data: Dict[str, Any]):
        """Store scraped data in database"""
        session = self.db.session_scope().__enter__()
        try:
            # Create scraped data record
            scraped_data = ScrapedData(
                source=data.get('source', 'enhanced_scraper'),
                data_type='team_data',
                external_id=data.get('team_name', ''),
                content=data,
                timestamp=datetime.now()
            )
            
            session.add(scraped_data)
            session.commit()
            
            logger.info(f"Stored scraped data for {data.get('team_name', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error storing scraped data: {e}")
            session.rollback()
        finally:
            session.close()
    
    async def enhance_predictions(self):
        """Enhance existing predictions with scraped data"""
        logger.info("Enhancing predictions with scraped data")
        
        session = self.db.session_scope().__enter__()
        try:
            # Get matches without enhanced data
            matches = session.query(Match).join(Prediction).all()
            
            for match in matches:
                # Get team names
                home_team = match.home_team.name if match.home_team else None
                away_team = match.away_team.name if match.away_team else None
                
                if home_team and away_team:
                    # Scrape data for both teams
                    home_data = await self.scrape_team_data(home_team)
                    away_data = await self.scrape_team_data(away_team)
                    
                    # Update prediction with enhanced data
                    await self.update_prediction_with_enhanced_data(match, home_data, away_data)
            
            logger.info("Prediction enhancement completed")
            
        except Exception as e:
            logger.error(f"Error enhancing predictions: {e}")
        finally:
            session.close()
    
    async def update_prediction_with_enhanced_data(self, match: Match, home_data: Dict, away_data: Dict):
        """Update prediction with enhanced data"""
        session = self.db.session_scope().__enter__()
        try:
            prediction = session.query(Prediction).filter(Prediction.match_id == match.id).first()
            
            if prediction:
                # Calculate enhanced probabilities based on form and stats
                enhanced_probs = self.calculate_enhanced_probabilities(home_data, away_data)
                
                # Update prediction
                prediction.home_win_prob = enhanced_probs['home_win']
                prediction.draw_prob = enhanced_probs['draw']
                prediction.away_win_prob = enhanced_probs['away_win']
                prediction.confidence = enhanced_probs['confidence']
                
                session.commit()
                logger.info(f"Updated prediction for match {match.id}")
            
        except Exception as e:
            logger.error(f"Error updating prediction: {e}")
            session.rollback()
        finally:
            session.close()
    
    def calculate_enhanced_probabilities(self, home_data: Dict, away_data: Dict) -> Dict[str, float]:
        """Calculate enhanced probabilities using scraped data"""
        # Base probabilities
        home_form = home_data.get('form_data', {}).get('form_rating', 0.5)
        away_form = away_data.get('form_data', {}).get('form_rating', 0.5)
        
        # Home advantage
        home_advantage = 0.1
        
        # Calculate enhanced probabilities
        home_strength = home_form + home_advantage
        away_strength = away_form
        
        # Normalize to probabilities
        total_strength = home_strength + away_strength + 0.3  # Add draw probability
        
        home_win = home_strength / total_strength
        away_win = away_strength / total_strength
        draw = 0.3 / total_strength
        
        # Ensure probabilities sum to 1
        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total
        
        # Calculate confidence based on data quality
        confidence = min(0.95, 0.6 + (max(home_win, draw, away_win) - 0.33) * 0.5)
        
        return {
            'home_win': round(home_win, 4),
            'draw': round(draw, 4),
            'away_win': round(away_win, 4),
            'confidence': round(confidence, 4)
        }

async def main():
    """Main function to run the enhanced scraper framework"""
    logger.info("Starting enhanced scraper framework")
    
    config = ScrapingConfig(
        user_agent_rotation=True,
        proxy_rotation=False,  # Set to True if you have proxies
        delay_range=(2.0, 5.0),
        max_retries=3,
        timeout=30,
        stealth_mode=True
    )
    
    framework = EnhancedScraperFramework(config)
    
    # Test scraping
    test_team = "Manchester United"
    team_data = await framework.scrape_team_data(test_team)
    
    if team_data:
        logger.info(f"Successfully scraped data for {test_team}")
        logger.info(f"Form data: {team_data.get('form_data', {})}")
        logger.info(f"Stats data: {team_data.get('stats_data', {})}")
    
    # Enhance predictions
    await framework.enhance_predictions()
    
    logger.info("Enhanced scraper framework completed")

if __name__ == "__main__":
    asyncio.run(main())