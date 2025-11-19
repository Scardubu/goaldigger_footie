"""
Advanced anti-scraping strategies for football data collection.
Implements multiple layers of protection to avoid detection and blocking.
"""
import logging
import random
import time
import string
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import os
import json

from fake_useragent import UserAgent
from playwright.sync_api import Page

logger = logging.getLogger(__name__)

class AntiScrapingStrategies:
    """
    Collection of advanced anti-scraping techniques for reliable data collection.
    Implements layered strategies to mimic human behavior and avoid detection.
    """
    
    def __init__(self, strategy_level: str = "medium"):
        """
        Initialize anti-scraping strategies with configurable intensity.
        
        Args:
            strategy_level: Intensity of anti-scraping measures ("low", "medium", "high", "stealth")
        """
        self.strategy_level = strategy_level.lower()
        
        # Set delay ranges based on strategy level (min, max) in seconds
        self.delay_ranges = {
            "low": (0.5, 1.5),
            "medium": (1.0, 3.0),
            "high": (2.0, 5.0),
            "stealth": (3.0, 8.0)
        }
        
        # Set behavioral parameters based on strategy level
        self.behavioral_params = {
            "low": {
                "mouse_movement_probability": 0.2,
                "scroll_probability": 0.3,
                "scroll_count_range": (1, 3),
                "viewport_variation": False,
                "inject_comments": False
            },
            "medium": {
                "mouse_movement_probability": 0.5,
                "scroll_probability": 0.7,
                "scroll_count_range": (2, 5),
                "viewport_variation": True,
                "inject_comments": True
            },
            "high": {
                "mouse_movement_probability": 0.8,
                "scroll_probability": 0.9,
                "scroll_count_range": (3, 8),
                "viewport_variation": True,
                "inject_comments": True
            },
            "stealth": {
                "mouse_movement_probability": 1.0,
                "scroll_probability": 1.0,
                "scroll_count_range": (5, 12),
                "viewport_variation": True,
                "inject_comments": True
            }
        }
        
        # Default to medium if strategy level is invalid
        if self.strategy_level not in self.delay_ranges:
            logger.warning(f"Invalid strategy level: {strategy_level}. Using 'medium' as default.")
            self.strategy_level = "medium"
            
        # Initialize user agent rotator
        try:
            self.ua = UserAgent()
            self.ua_initialized = True
        except Exception as e:
            logger.warning(f"Failed to initialize UserAgent: {e}. Using fallback list.")
            self.ua_initialized = False
            
        # Fallback user agents list
        self.fallback_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        ]
        
        # Store fingerprint data to maintain consistent identities across sessions
        self.fingerprint_store_file = os.path.join(os.path.dirname(__file__), "fingerprints.json")
        self.fingerprints = self._load_fingerprints()
        
        # Keep track of requests to domains for rate limiting
        self.domain_request_history = {}
        
        logger.info(f"AntiScrapingStrategies initialized with level: {self.strategy_level}")
        
    def _load_fingerprints(self) -> Dict[str, Dict[str, Any]]:
        """Load stored fingerprints from JSON file."""
        if os.path.exists(self.fingerprint_store_file):
            try:
                with open(self.fingerprint_store_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load fingerprints: {e}")
        
        return {}
        
    def _save_fingerprints(self):
        """Save fingerprints to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.fingerprint_store_file), exist_ok=True)
            with open(self.fingerprint_store_file, 'w') as f:
                json.dump(self.fingerprints, f)
        except Exception as e:
            logger.warning(f"Failed to save fingerprints: {e}")
    
    def get_random_user_agent(self) -> str:
        """
        Get a random user agent string.
        Returns a consistent user agent for the same domain to avoid fingerprinting.
        
        Returns:
            Random user agent string
        """
        if self.ua_initialized:
            try:
                return self.ua.random
            except Exception as e:
                logger.warning(f"Error getting random user agent: {e}")
                
        # Fallback to predefined list
        return random.choice(self.fallback_agents)
        
    def get_domain_fingerprint(self, domain: str) -> Dict[str, Any]:
        """
        Get a consistent browser fingerprint for a domain.
        
        Args:
            domain: Target domain
            
        Returns:
            Dictionary with fingerprint parameters
        """
        # Check if we already have a fingerprint for this domain
        if domain in self.fingerprints:
            return self.fingerprints[domain]
            
        # Create a new fingerprint
        fingerprint = {
            "user_agent": self.get_random_user_agent(),
            "accept_language": random.choice(["en-US,en;q=0.9", "en-GB,en;q=0.9", "en-CA,en;q=0.9"]),
            "platform": random.choice(["Windows", "Macintosh", "Linux"]),
            "hardware_concurrency": random.choice([2, 4, 8, 16]),
            "device_memory": random.choice([2, 4, 8, 16]),
            "color_depth": random.choice([24, 30, 48]),
            "resolution": random.choice([
                {"width": 1920, "height": 1080},
                {"width": 2560, "height": 1440},
                {"width": 1440, "height": 900},
                {"width": 1366, "height": 768}
            ]),
            "timezone_offset": random.choice([-480, -420, -360, -300, -240, -120, 0, 60, 120, 180, 240, 330, 480, 600]),
            "session_storage": random.choice([True, False]),
            "local_storage": random.choice([True, False]),
            "indexed_db": random.choice([True, False]),
            "cpu_class": random.choice(["Intel", "AMD", None]),
            "navigator_plugins_length": random.randint(5, 15),
            "created_at": datetime.now().isoformat(),
            "canvas_noise": random.uniform(0.1, 2.0)
        }
        
        # Store fingerprint
        self.fingerprints[domain] = fingerprint
        self._save_fingerprints()
        
        return fingerprint
    
    def apply_page_evasions(self, page: Page, url: str) -> None:
        """
        Apply anti-detection measures to a Playwright page.
        
        Args:
            page: Playwright page object
            url: Target URL
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Get consistent fingerprint for domain
        fingerprint = self.get_domain_fingerprint(domain)
        
        # Apply fingerprint evasions
        page.evaluate(f"""() => {{
            // Override property getters for fingerprinting resistance
            Object.defineProperty(navigator, 'hardwareConcurrency', {{ get: () => {fingerprint['hardware_concurrency']} }});
            Object.defineProperty(navigator, 'deviceMemory', {{ get: () => {fingerprint['device_memory']} }});
            Object.defineProperty(screen, 'colorDepth', {{ get: () => {fingerprint['color_depth']} }});
            Object.defineProperty(screen, 'width', {{ get: () => {fingerprint['resolution']['width']} }});
            Object.defineProperty(screen, 'height', {{ get: () => {fingerprint['resolution']['height']} }});
            Object.defineProperty(navigator, 'platform', {{ get: () => '{fingerprint['platform']}' }});
            
            // Fake timezone
            Object.defineProperty(Date.prototype, 'getTimezoneOffset', {{ value: () => {fingerprint['timezone_offset']} }});
            
            // Override permissions API to avoid detection
            const originalQuery = navigator.permissions.query;
            navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' 
                    ? Promise.resolve({{ state: Notification.permission }}) 
                    : originalQuery(parameters)
            );
            
            // Fake WebDriver
            Object.defineProperty(navigator, 'webdriver', {{ get: () => false }});
            
            // Add noise to canvas fingerprinting
            const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
            CanvasRenderingContext2D.prototype.getImageData = function(x, y, width, height) {{
                const imageData = originalGetImageData.call(this, x, y, width, height);
                const noise = {fingerprint['canvas_noise']};
                
                for (let i = 0; i < imageData.data.length; i += 4) {{
                    // Add slight noise to RGB values
                    imageData.data[i] = Math.max(0, Math.min(255, imageData.data[i] + Math.floor(Math.random() * noise * 2 - noise)));
                    imageData.data[i+1] = Math.max(0, Math.min(255, imageData.data[i+1] + Math.floor(Math.random() * noise * 2 - noise)));
                    imageData.data[i+2] = Math.max(0, Math.min(255, imageData.data[i+2] + Math.floor(Math.random() * noise * 2 - noise)));
                }}
                
                return imageData;
            }};
            
            // Fake plugins length
            Object.defineProperty(navigator, 'plugins', {{ 
                get: () => Array(this.navigatorPluginsLength).fill().map(() => ({{
                    name: 'Plugin ' + Math.random().toString(36).substr(2, 5)
                }}))
            }});
            
            // Chrome properties
            if (navigator.userAgent.includes('Chrome')) {{
                window.chrome = {{ runtime: {{}}, app: {{}}, loadTimes: () => {{}}, csi: () => {{}}, chrome: true }};
            }}
        }}""")
        
        # Apply stealth settings based on strategy level
        if self.strategy_level in ["high", "stealth"]:
            # Viewport variation for each session but consistent for domain
            if self.behavioral_params[self.strategy_level]["viewport_variation"]:
                width = fingerprint["resolution"]["width"]
                height = fingerprint["resolution"]["height"]
                page.set_viewport_size({"width": width, "height": height})
            
            # Set custom headers for better stealth
            page.set_extra_http_headers({
                "Accept-Language": fingerprint["accept_language"],
                "Sec-Ch-Ua": '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1"
            })
    
    def apply_human_behavior(self, page: Page, url: str) -> None:
        """
        Simulate human-like behavior on page to avoid detection.
        
        Args:
            page: Playwright page object
            url: Target URL
        """
        # Get parameters for current strategy level
        params = self.behavioral_params[self.strategy_level]
        
        # Random mouse movements
        if random.random() < params["mouse_movement_probability"]:
            # Number of movements
            movement_count = random.randint(1, 5)
            for _ in range(movement_count):
                x = random.randint(100, page.viewport_size["width"] - 200)
                y = random.randint(100, page.viewport_size["height"] - 200)
                page.mouse.move(x, y)
                
                # Occasional clicks (10% chance)
                if random.random() < 0.1:
                    page.mouse.click(x, y)
                
                time.sleep(random.uniform(0.1, 0.8))
        
        # Random scrolling
        if random.random() < params["scroll_probability"]:
            scroll_count = random.randint(*params["scroll_count_range"])
            
            for _ in range(scroll_count):
                # Random scroll distance
                scroll_distance = random.randint(100, 800)
                
                # Scroll with variable speed
                page.evaluate(f"""() => {{
                    // Smooth scroll with variable speed
                    const distance = {scroll_distance};
                    const duration = {random.uniform(500, 1500)};
                    const start = window.scrollY;
                    const startTime = Date.now();
                    
                    function scroll() {{
                        const elapsed = Date.now() - startTime;
                        const progress = Math.min(elapsed / duration, 1);
                        
                        // Ease-in-out function for more natural scrolling
                        const easeInOut = progress < 0.5 
                            ? 2 * progress * progress 
                            : 1 - Math.pow(-2 * progress + 2, 2) / 2;
                            
                        window.scrollTo(0, start + distance * easeInOut);
                        
                        if (progress < 1) {{
                            requestAnimationFrame(scroll);
                        }}
                    }}
                    
                    scroll();
                }}""")
                
                # Pause between scrolls
                time.sleep(random.uniform(0.5, 2.0))
    
    def apply_wait_strategy(self, url: str) -> None:
        """
        Apply a dynamic waiting strategy for rate limiting.
        Considers domain history to avoid hammering the same domain.
        
        Args:
            url: Target URL
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Get current time
        now = datetime.now()
        
        # Initialize domain request history if not already tracked
        if domain not in self.domain_request_history:
            self.domain_request_history[domain] = []
        
        # Clear old requests (older than 1 hour)
        self.domain_request_history[domain] = [
            time for time in self.domain_request_history[domain]
            if now - time < timedelta(hours=1)
        ]
        
        # Calculate delay based on recent requests to domain
        recent_request_count = len(self.domain_request_history[domain])
        
        # Base delay from strategy level
        min_delay, max_delay = self.delay_ranges[self.strategy_level]
        
        # Scale delay based on recent request count
        if recent_request_count > 10:
            scale_factor = min(5.0, 1.0 + (recent_request_count - 10) / 5)
            min_delay *= scale_factor
            max_delay *= scale_factor
        
        # Add randomization
        jitter = random.uniform(-0.2, 0.2)  # +/- 20% jitter
        delay = random.uniform(min_delay, max_delay) * (1 + jitter)
        
        # Apply delay
        logger.debug(f"Applying delay of {delay:.2f}s for {domain} (requests in last hour: {recent_request_count})")
        time.sleep(delay)
        
        # Record this request
        self.domain_request_history[domain].append(now)
    
    def generate_dynamic_headers(self, url: str) -> Dict[str, str]:
        """
        Generate dynamic HTTP headers for the request.
        
        Args:
            url: Target URL
            
        Returns:
            Dictionary with HTTP headers
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        
        # Get domain fingerprint
        fingerprint = self.get_domain_fingerprint(domain)
        
        # Basic headers
        headers = {
            "User-Agent": fingerprint["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": fingerprint["accept_language"],
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        
        # Add referer with 70% probability
        if random.random() < 0.7:
            # Common referring sites
            referers = [
                "https://www.google.com/",
                "https://www.bing.com/",
                "https://search.yahoo.com/",
                "https://duckduckgo.com/",
                f"https://{domain}/"  # Same domain
            ]
            headers["Referer"] = random.choice(referers)
        
        # Add Do-Not-Track header with 30% probability
        if random.random() < 0.3:
            headers["DNT"] = "1"
        
        # Add Sec- headers for Chrome based browsers
        if "Chrome" in fingerprint["user_agent"]:
            headers.update({
                "Sec-Ch-Ua": '"Chromium";v="92", " Not A;Brand";v="99", "Google Chrome";v="92"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1"
            })
        
        return headers
    
    def get_fp_canvas_value(self, domain: str) -> str:
        """
        Generate a consistent canvas fingerprint value for domain.
        
        Args:
            domain: Target domain
            
        Returns:
            Fingerprint string
        """
        # Use domain as seed for consistent fingerprinting
        fingerprint = self.get_domain_fingerprint(domain)
        
        # Create a base hash using domain
        base = domain + fingerprint["created_at"]
        hash_object = hashlib.sha256(base.encode())
        hex_dig = hash_object.hexdigest()
        
        # Apply consistent noise
        noise_level = fingerprint["canvas_noise"]
        chars = list(hex_dig)
        for i in range(len(chars)):
            if random.random() < noise_level / 5:  # Scale noise for reasonable variation
                chars[i] = random.choice(string.hexdigits.lower())
                
        return ''.join(chars)
