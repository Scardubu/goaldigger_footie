"""
Scraper factory for initializing and managing scrapers.
"""
import logging
import os
from typing import Any, Dict, List, Optional, Type, Union

from scripts.core.scrapers.playwright_manager import PlaywrightManager
from scripts.core.scrapers.proxy_manager import ProxyManager
from scripts.scrapers.base_scraper import BaseScraper
from scripts.scrapers.bbc_scraper import BBCScraper
from scripts.scrapers.espn_scraper import ESPNScraper
from scripts.scrapers.fbref_scraper import FBrefScraper
from scripts.scrapers.football_data_scraper import FootballDataScraper
from scripts.scrapers.livescore_scraper import LivescoreScraper
from scripts.scrapers.odds_api_scraper import OddsAPIScraper
from scripts.scrapers.sportmonks_scraper import SportMonksScraper
from scripts.scrapers.transfermarkt_scraper import TransfermarktScraper
from scripts.scrapers.whoscored_scraper import WhoScoredScraper
from utils.http_client_async import HttpClientAsync
from utils.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

class ScraperFactory:
    """
    Factory for creating and managing scrapers.
    Provides a centralized way to initialize and access scrapers.
    """
    
    def __init__(
        self,
        http_client: Optional[HttpClientAsync] = None,
        proxy_manager: Optional[ProxyManager] = None,
        playwright_manager: Optional[PlaywrightManager] = None,
        system_monitor: Optional[SystemMonitor] = None,
        use_proxies: bool = True
    ):
        """
        Initialize the scraper factory.
        
        Args:
            http_client: Optional HTTP client to use
            proxy_manager: Optional proxy manager to use
            playwright_manager: Optional playwright manager for JS rendering
            system_monitor: Optional system monitor for tracking performance
            use_proxies: Whether to use proxies
        """
        self.http_client = http_client or HttpClientAsync()
        self.proxy_manager = proxy_manager
        self.playwright_manager = playwright_manager
        self.system_monitor = system_monitor
        self.use_proxies = use_proxies
        
        # Dictionary to store initialized scrapers
        self.scrapers: Dict[str, BaseScraper] = {}
        
        # Register available scrapers
        self.available_scrapers = {
            "football_data": FootballDataScraper,
            "fbref": FBrefScraper,
            "transfermarkt": TransfermarktScraper,
            "sportmonks": SportMonksScraper,
            "odds_api": OddsAPIScraper,
            "espn": ESPNScraper,
            "bbc": BBCScraper,
            "whoscored": WhoScoredScraper,
            "livescore": LivescoreScraper
        }
        
        logger.info(f"ScraperFactory initialized with {len(self.available_scrapers)} available scrapers")
    
    def list_available_scrapers(self) -> List[str]:
        """
        Get list of all available scraper names.
        
        Returns:
            List of scraper names
        """
        return list(self.available_scrapers.keys())
    
    def get_scraper(self, scraper_name: str) -> Optional[BaseScraper]:
        """
        Get a scraper by name. Initializes the scraper if not already initialized.
        
        Args:
            scraper_name: Name of the scraper to get
            
        Returns:
            Initialized scraper or None if not found
        """
        # Check if scraper is already initialized
        if scraper_name in self.scrapers:
            return self.scrapers[scraper_name]
        
        # Check if scraper is available
        if scraper_name not in self.available_scrapers:
            logger.error(f"Scraper not found: {scraper_name}")
            return None
        
        # Initialize scraper
        try:
            scraper_class = self.available_scrapers[scraper_name]
            # Skip abstract classes
            if hasattr(scraper_class, '__abstractmethods__') and getattr(scraper_class, '__abstractmethods__', None):
                if len(scraper_class.__abstractmethods__) > 0:
                    logger.warning(f"Skipping abstract scraper class: {scraper_name}")
                    return None
            
            # Only pass supported kwargs
            scraper = scraper_class(
                http_client=self.http_client,
                proxy_manager=self.proxy_manager,
                playwright_manager=self.playwright_manager,
                use_proxies=self.use_proxies
            )
            
            # Store initialized scraper
            self.scrapers[scraper_name] = scraper
            
            logger.info(f"Initialized scraper: {scraper_name}")
            return scraper
        
        except Exception as e:
            logger.error(f"Error initializing scraper {scraper_name}: {e}")
            return None
    
    def get_all_scrapers(self) -> Dict[str, BaseScraper]:
        """
        Initialize and get all available scrapers.
        
        Returns:
            Dictionary mapping scraper names to initialized scrapers
        """
        for scraper_name in self.available_scrapers:
            self.get_scraper(scraper_name)
        
        return self.scrapers
    
    def get_scraper_status(self) -> Dict[str, bool]:
        """
        Get the status of all scrapers.
        
        Returns:
            Dictionary mapping scraper names to their availability status
        """
        status = {}
        
        for scraper_name in self.available_scrapers:
            scraper = self.get_scraper(scraper_name)
            status[scraper_name] = scraper is not None
        
        return status
    
    def close_all(self):
        """Close all initialized scrapers and managers."""
        # Close Playwright manager if initialized
        if self.playwright_manager:
            try:
                self.playwright_manager.close()
                logger.info("Closed Playwright manager")
            except Exception as e:
                logger.error(f"Error closing Playwright manager: {e}")
        
        # Clear scrapers dictionary
        self.scrapers.clear()
        logger.info("Closed all scrapers")


# Create a singleton instance for easy access
scraper_factory = ScraperFactory()

# Convenience functions for direct use without creating a factory instance

def get_scraper(scraper_name: str) -> Optional[BaseScraper]:
    """Get a scraper by name."""
    return scraper_factory.get_scraper(scraper_name)

def get_all_scrapers() -> Dict[str, BaseScraper]:
    """Initialize and get all available scrapers."""
    return scraper_factory.get_all_scrapers()

def get_scraper_status() -> Dict[str, bool]:
    """Get the status of all scrapers."""
    return scraper_factory.get_scraper_status()

def close_all():
    """Close all initialized scrapers and managers."""
    scraper_factory.close_all()
