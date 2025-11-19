"""
Enhanced Injury and Suspension Data Scraper
Integrates multiple sources for comprehensive player availability tracking.
"""
import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import aiohttp
import pandas as pd
from bs4 import BeautifulSoup

from scripts.core.data_quality_monitor import DataQualityMonitor
from scripts.core.scrapers.enhanced_proxy_manager import EnhancedProxyManager
from scripts.scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

@dataclass
class InjuryRecord:
    """Represents an injury record."""
    player_id: str
    player_name: str
    team_id: str
    injury_type: str
    injury_severity: str  # minor, moderate, major, career_threatening
    injury_date: Optional[datetime]
    expected_return_date: Optional[datetime]
    recovery_progress: float  # 0.0 to 1.0
    medical_notes: str
    data_source: str

@dataclass
class SuspensionRecord:
    """Represents a suspension record."""
    player_id: str
    player_name: str
    team_id: str
    suspension_type: str  # yellow_cards, red_card, disciplinary, other
    suspension_reason: str
    suspension_date: datetime
    matches_suspended: int
    matches_served: int
    reinstatement_date: Optional[datetime]
    data_source: str

class EnhancedInjurySuspensionScraper(BaseScraper):
    """Enhanced scraper for injury and suspension data from multiple sources."""
    
    def __init__(self):
        super().__init__(
            name="enhanced_injury_suspension",
            base_url="https://www.physioroom.com",
            use_playwright=True,
            rate_limit_delay=(2.0, 4.0),
            max_retries=3
        )
        
        # Data sources configuration
        self.data_sources = {
            'physioroom': {
                'base_url': 'https://www.physioroom.com',
                'leagues': {
                    'Premier League': 'epl',
                    'La Liga': 'laliga',
                    'Serie A': 'seriea',
                    'Bundesliga': 'bundesliga',
                    'Ligue 1': 'ligue1'
                }
            },
            'transfermarkt': {
                'base_url': 'https://www.transfermarkt.com',
                'injury_path': '/injuries/detail/spieler/{player_id}'
            },
            'espn': {
                'base_url': 'https://www.espn.com',
                'injury_path': '/soccer/team/injuries/_/id/{team_id}'
            }
        }
        
        # Enhanced components
        self.proxy_manager = EnhancedProxyManager(enable_ml_patterns=True)
        self.quality_monitor = DataQualityMonitor()
        
        # Severity mapping
        self.severity_mapping = {
            'knock': 'minor',
            'strain': 'minor',
            'minor': 'minor',
            'sprain': 'moderate',
            'tear': 'moderate',
            'fracture': 'major',
            'surgery': 'major',
            'acl': 'career_threatening',
            'cruciate': 'career_threatening'
        }
    
    async def scrape_comprehensive_injury_data(self, league: str, teams: List[str]) -> List[InjuryRecord]:
        """Scrape comprehensive injury data from multiple sources."""
        all_injuries = []
        
        # Scrape from each data source
        for source_name, source_config in self.data_sources.items():
            try:
                logger.info(f"Scraping injury data from {source_name}")
                
                if source_name == 'physioroom':
                    injuries = await self._scrape_physioroom_injuries(league, teams)
                elif source_name == 'transfermarkt':
                    injuries = await self._scrape_transfermarkt_injuries(teams)
                elif source_name == 'espn':
                    injuries = await self._scrape_espn_injuries(teams)
                else:
                    continue
                
                all_injuries.extend(injuries)
                logger.info(f"Retrieved {len(injuries)} injuries from {source_name}")
                
                # Apply rate limiting
                await asyncio.sleep(self.proxy_manager.get_ml_delay())
                
            except Exception as e:
                logger.error(f"Failed to scrape {source_name}: {e}")
                continue
        
        # Deduplicate and merge records
        merged_injuries = self._merge_injury_records(all_injuries)
        
        # Quality monitoring
        await self.quality_monitor.monitor_data_quality(
            'injury_data', 
            [injury.__dict__ for injury in merged_injuries]
        )
        
        return merged_injuries
    
    async def _scrape_physioroom_injuries(self, league: str, teams: List[str]) -> List[InjuryRecord]:
        """Scrape injury data from Physioroom."""
        injuries = []
        
        league_code = self.data_sources['physioroom']['leagues'].get(league)
        if not league_code:
            logger.warning(f"League {league} not supported by Physioroom")
            return injuries
        
        for team in teams:
            try:
                # Get optimal proxy
                proxy = await self.proxy_manager.get_optimal_proxy()
                headers = self.proxy_manager.get_ml_headers()
                
                # Construct URL
                team_slug = team.lower().replace(' ', '_').replace('-', '_')
                url = f"{self.data_sources['physioroom']['base_url']}/news/{league_code}/{team_slug}_injury_news.php"
                
                # Fetch page
                # Explicit timeout to avoid hanging on site fetch
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    proxy_dict = {'http': proxy, 'https': proxy} if proxy else None
                    
                    async with session.get(url, headers=headers, proxy=proxy_dict, timeout=30) as response:
                        if response.status == 200:
                            html = await response.text()
                            team_injuries = self._parse_physioroom_injuries(html, team)
                            injuries.extend(team_injuries)
                            
                            # Record proxy performance
                            await self.proxy_manager.record_proxy_performance(proxy, True, response.headers.get('response-time', 1.0))
                        else:
                            logger.warning(f"Failed to fetch {url}: {response.status}")
                            if proxy:
                                await self.proxy_manager.record_proxy_performance(proxy, False, 0)
                
                # Rate limiting
                await asyncio.sleep(self.proxy_manager.get_ml_delay())
                
            except Exception as e:
                logger.error(f"Error scraping Physioroom for {team}: {e}")
                if proxy:
                    await self.proxy_manager.record_proxy_performance(proxy, False, 0)
                continue
        
        return injuries
    
    def _parse_physioroom_injuries(self, html: str, team: str) -> List[InjuryRecord]:
        """Parse injury data from Physioroom HTML."""
        injuries = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find injury table
            injury_table = soup.find('table', {'class': 'injury-table'}) or soup.find('table', {'id': 'injury-table'})
            
            if not injury_table:
                # Try alternative selectors
                injury_table = soup.find('div', {'class': 'injury-list'})
            
            if injury_table:
                rows = injury_table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    
                    if len(cells) >= 4:
                        player_name = cells[0].get_text(strip=True)
                        injury_type = cells[1].get_text(strip=True)
                        expected_return = cells[2].get_text(strip=True)
                        
                        # Parse expected return date
                        return_date = self._parse_return_date(expected_return)
                        
                        # Determine severity
                        severity = self._determine_injury_severity(injury_type)
                        
                        # Calculate recovery progress
                        progress = self._calculate_recovery_progress(expected_return, return_date)
                        
                        injury = InjuryRecord(
                            player_id=f"{team}_{player_name}".replace(' ', '_'),
                            player_name=player_name,
                            team_id=team,
                            injury_type=injury_type,
                            injury_severity=severity,
                            injury_date=None,  # Not available from Physioroom
                            expected_return_date=return_date,
                            recovery_progress=progress,
                            medical_notes=f"Expected return: {expected_return}",
                            data_source='physioroom'
                        )
                        
                        injuries.append(injury)
            
        except Exception as e:
            logger.error(f"Error parsing Physioroom injuries: {e}")
        
        return injuries
    
    async def _scrape_transfermarkt_injuries(self, teams: List[str]) -> List[InjuryRecord]:
        """Scrape injury data from Transfermarkt."""
        injuries = []
        
        # Implementation for Transfermarkt scraping
        # This would require team ID mapping and different parsing logic
        
        return injuries
    
    async def _scrape_espn_injuries(self, teams: List[str]) -> List[InjuryRecord]:
        """Scrape injury data from ESPN."""
        injuries = []
        
        # Implementation for ESPN scraping
        # This would require team ID mapping and different parsing logic
        
        return injuries
    
    async def scrape_suspension_data(self, league: str, teams: List[str]) -> List[SuspensionRecord]:
        """Scrape suspension data."""
        suspensions = []
        
        # Implementation for suspension data scraping
        # This would involve scraping disciplinary records from various sources
        
        return suspensions
    
    def _parse_return_date(self, return_text: str) -> Optional[datetime]:
        """Parse expected return date from text."""
        if not return_text or return_text.lower() in ['unknown', 'tbc', 'n/a']:
            return None
        
        try:
            # Handle various date formats
            if 'week' in return_text.lower():
                weeks = re.search(r'(\d+)', return_text)
                if weeks:
                    return datetime.now() + timedelta(weeks=int(weeks.group(1)))
            
            elif 'month' in return_text.lower():
                months = re.search(r'(\d+)', return_text)
                if months:
                    return datetime.now() + timedelta(days=int(months.group(1)) * 30)
            
            elif 'day' in return_text.lower():
                days = re.search(r'(\d+)', return_text)
                if days:
                    return datetime.now() + timedelta(days=int(days.group(1)))
            
            # Try to parse as date
            for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']:
                try:
                    return datetime.strptime(return_text, fmt)
                except ValueError:
                    continue
        
        except Exception as e:
            logger.debug(f"Could not parse return date '{return_text}': {e}")
        
        return None
    
    def _determine_injury_severity(self, injury_type: str) -> str:
        """Determine injury severity from injury type."""
        injury_lower = injury_type.lower()
        
        for keyword, severity in self.severity_mapping.items():
            if keyword in injury_lower:
                return severity
        
        # Default to moderate if unknown
        return 'moderate'
    
    def _calculate_recovery_progress(self, return_text: str, return_date: Optional[datetime]) -> float:
        """Calculate recovery progress based on expected return."""
        if not return_date:
            return 0.5  # Unknown progress
        
        days_until_return = (return_date - datetime.now()).days
        
        if days_until_return <= 0:
            return 1.0  # Should be recovered
        elif days_until_return <= 7:
            return 0.8  # Almost recovered
        elif days_until_return <= 30:
            return 0.6  # Moderate progress
        elif days_until_return <= 90:
            return 0.3  # Early recovery
        else:
            return 0.1  # Just started recovery
    
    def _merge_injury_records(self, injuries: List[InjuryRecord]) -> List[InjuryRecord]:
        """Merge duplicate injury records from different sources."""
        merged = {}
        
        for injury in injuries:
            key = f"{injury.player_name}_{injury.team_id}_{injury.injury_type}"
            
            if key not in merged:
                merged[key] = injury
            else:
                # Merge with existing record (prefer more detailed information)
                existing = merged[key]
                
                # Update with more recent or detailed information
                if injury.expected_return_date and not existing.expected_return_date:
                    existing.expected_return_date = injury.expected_return_date
                
                if injury.injury_date and not existing.injury_date:
                    existing.injury_date = injury.injury_date
                
                if len(injury.medical_notes) > len(existing.medical_notes):
                    existing.medical_notes = injury.medical_notes
                
                # Combine data sources
                if injury.data_source not in existing.data_source:
                    existing.data_source += f", {injury.data_source}"
        
        return list(merged.values())

    async def get_matches(self, league_code: str, date_from: Optional[datetime] = None,
                         date_to: Optional[datetime] = None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Get matches for injury/suspension analysis.

        Args:
            league_code: League identifier
            date_from: Start date for match data
            date_to: End date for match data
            **kwargs: Additional parameters

        Returns:
            DataFrame with match data or None if failed
        """
        try:
            # This scraper focuses on injury data, not match data
            # Return empty DataFrame to satisfy abstract method requirement
            logger.info(f"get_matches called for {league_code} - returning empty DataFrame")
            return pd.DataFrame(columns=['match_id', 'home_team', 'away_team', 'date'])

        except Exception as e:
            logger.error(f"Error in get_matches: {e}")
            return None

    async def get_team_info(self, team_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get team information for injury analysis.

        Args:
            team_id: Team identifier

        Returns:
            Dictionary with team information or None if failed
        """
        try:
            # Return basic team info structure for injury analysis
            logger.info(f"get_team_info called for team {team_id}")
            return {
                'team_id': team_id,
                'name': str(team_id),
                'injury_data_available': True,
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in get_team_info: {e}")
            return None

# Usage example
async def main():
    """Example usage of the enhanced injury/suspension scraper."""
    scraper = EnhancedInjurySuspensionScraper()
    
    # Scrape injury data for Premier League teams
    teams = ['Arsenal', 'Chelsea', 'Manchester United', 'Liverpool']
    injuries = await scraper.scrape_comprehensive_injury_data('Premier League', teams)
    
    print(f"Found {len(injuries)} injury records")
    for injury in injuries[:5]:  # Show first 5
        print(f"- {injury.player_name} ({injury.team_id}): {injury.injury_type} - {injury.injury_severity}")

if __name__ == "__main__":
    asyncio.run(main())
