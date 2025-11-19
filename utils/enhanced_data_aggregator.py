#!/usr/bin/env python3
"""
Enhanced Data Aggregator - GoalDiggers Platform

Comprehensive data aggregation system that intelligently combines multiple sources
to ensure current fixture data, predictions, and analytics are up-to-date.

Features:
- Multi-source data integration
- Real-time fixture updates
- Quality-based source prioritization
- Fallback mechanism for data reliability
- Creative data enhancement using multiple APIs
"""

import asyncio
import json
import json as _json
import logging
import os
import random
from pathlib import Path

try:
    from config.settings import settings as _settings  # type: ignore
except Exception:  # Fallback defaults
    class _Tmp:  # minimal stub
        ENABLE_PERSISTENT_FIXTURE_CACHE=False; FIXTURE_CACHE_TTL=900; REDIS_URL=None
    _settings=_Tmp()

_redis_client = None
if getattr(_settings, 'REDIS_URL', None):  # lazy connect
    try:  # pragma: no cover - optional dependency
        import redis  # type: ignore
        _redis_client = redis.from_url(_settings.REDIS_URL, decode_responses=True)
    except Exception:
        _redis_client = None
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import requests

try:
    import streamlit as st  # type: ignore
except Exception:  # Non-Streamlit contexts
    st = None


def requests_with_retries(url: str, **kwargs):
    """Simple requests.get wrapper with retries and backoff."""
    max_retries = kwargs.pop('max_retries', 3)
    backoff = kwargs.pop('backoff', 1)
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, **kwargs)
            return resp
        except Exception as e:
            logger.warning(f"Request attempt {attempt} failed for {url}: {e}")
            if attempt == max_retries:
                raise
            time_to_sleep = backoff * attempt
            try:
                import time
                time.sleep(time_to_sleep)
            except Exception:
                pass

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

@dataclass
class FixtureData:
    """Standardized fixture data structure."""
    home_team: str
    away_team: str
    match_date: datetime
    league: str
    status: str = 'scheduled'
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    venue: Optional[str] = None
    match_id: Optional[str] = None
    confidence: float = 1.0
    data_source: str = 'unknown'
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

class EnhancedDataAggregator:
    """Enhanced data aggregator with multi-source integration."""
    
    def __init__(self):
        """Initialize the enhanced data aggregator."""
        self.logger = logger
        self.data_sources = self._initialize_data_sources()
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load API keys
        self.api_keys = self._load_api_keys()
        
        # Initialize HTTP client
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GoalDiggers-Platform/1.0 (Educational Purpose)'
        })
        
        logger.info("🚀 Enhanced Data Aggregator initialized with multi-source integration")
    
    def _initialize_data_sources(self) -> Dict[str, Dict]:
        """Initialize available data sources with priorities."""
        return {
            'football_data_org': {
                'name': 'Football-Data.org',
                'priority': 1,
                'base_url': 'https://api.football-data.org/v4',
                'rate_limit': 10,  # requests per minute
                'active': True,
                'last_request': 0
            },
            'api_football': {
                'name': 'API-Football',
                'priority': 2,
                'base_url': 'https://v3.football.api-sports.io',
                'rate_limit': 100,  # requests per day for free tier
                'active': False,
                'last_request': 0
            },
            'espn_scraper': {
                'name': 'ESPN Scraper',
                'priority': 3,
                'base_url': 'https://site.api.espn.com',
                'rate_limit': 60,  # requests per minute (estimated)
                'active': True,
                'last_request': 0
            },
            'fallback_generator': {
                'name': 'Intelligent Fallback',
                'priority': 10,
                'active': True
            }
        }
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment and config files."""
        api_keys = {}
        
        # Try loading from environment
        api_keys['football_data'] = os.getenv('FOOTBALL_DATA_API_KEY')
        api_keys['api_football'] = os.getenv('API_FOOTBALL_KEY')
        
        # Try loading from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            if not api_keys['football_data']:
                api_keys['football_data'] = os.getenv('FOOTBALL_DATA_API_KEY')
            if not api_keys['api_football']:
                api_keys['api_football'] = os.getenv('API_FOOTBALL_KEY')
        except ImportError:
            pass
        
        # Log available keys (safely)
        available_keys = [k for k, v in api_keys.items() if v]
        logger.info(f"📋 Available API keys: {', '.join(available_keys)}")
        
        return api_keys
    
    async def get_current_fixtures(self, days_ahead: int = 7) -> List[FixtureData]:
        """Get current fixtures from multiple sources with intelligent fallback."""
        # Cache key (daily granularity)
        cache_key = f"fixtures_{datetime.now().date()}_{days_ahead}"
        # 1. In-memory session cache
        if st and 'gd_fixtures_cache' in st.session_state:
            _sess_cached = st.session_state['gd_fixtures_cache'].get(cache_key)
            if _sess_cached:
                return _sess_cached
        # 2. Redis cache
        if _settings.ENABLE_PERSISTENT_FIXTURE_CACHE and _redis_client is not None:
            try:
                raw = _redis_client.get(cache_key)
                if raw:
                    data = _json.loads(raw)
                    return [FixtureData(**{**f, 'match_date': datetime.fromisoformat(f['match_date'])}) for f in data]
            except Exception:
                pass
        # 3. Disk cache
        disk_result = None
        if _settings.ENABLE_PERSISTENT_FIXTURE_CACHE:
            disk_result = self._load_disk_cache(cache_key, days_ahead)
            if disk_result:
                return disk_result
        all_fixtures = []
        
        # Try each data source in priority order
        for source_name, source_config in sorted(
            self.data_sources.items(),
            key=lambda x: x[1]['priority']
        ):
            if not source_config['active']:
                continue
            
            try:
                fixtures = await self._get_fixtures_from_source(source_name, days_ahead)
                if fixtures:
                    all_fixtures.extend(fixtures)
                    logger.info(f"✅ Retrieved {len(fixtures)} fixtures from {source_config['name']}")
                    
                    # If we got enough high-quality fixtures, we can stop
                    if len(all_fixtures) >= 10 and source_config['priority'] <= 2:
                        break
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to get fixtures from {source_config['name']}: {e}")
                continue
        
        # Deduplicate and rank fixtures
        unique_fixtures = self._deduplicate_fixtures(all_fixtures)
        
        # If we still don't have enough fixtures, generate intelligent fallbacks
        if len(unique_fixtures) < 5:
            fallback_fixtures = self._generate_intelligent_fallback_fixtures(days_ahead)
            unique_fixtures.extend(fallback_fixtures)
            logger.info(f"🔄 Added {len(fallback_fixtures)} intelligent fallback fixtures")
        
        # Sort by date and confidence
        unique_fixtures.sort(key=lambda x: (x.match_date, -x.confidence))
        
        logger.info(f"🎯 Total fixtures aggregated: {len(unique_fixtures)}")
        top = unique_fixtures[:20]
        # Write session cache
        if st:
            st.session_state.setdefault('gd_fixtures_cache', {})[cache_key] = top
        # Persist to Redis
        if _settings.ENABLE_PERSISTENT_FIXTURE_CACHE and _redis_client is not None:
            try:
                payload = _json.dumps([self._serialize_fixture(f) for f in top])
                _redis_client.setex(cache_key, _settings.FIXTURE_CACHE_TTL, payload)
            except Exception:
                pass
        # Persist to disk
        if _settings.ENABLE_PERSISTENT_FIXTURE_CACHE:
            try:
                self._store_disk_cache(cache_key, top)
            except Exception:
                pass
        return top
    
    async def _get_fixtures_from_source(self, source_name: str, days_ahead: int) -> List[FixtureData]:
        """Get fixtures from a specific data source."""
        if source_name == 'football_data_org':
            return await self._get_football_data_org_fixtures(days_ahead)
        elif source_name == 'api_football':
            return await self._get_api_football_fixtures(days_ahead)
        elif source_name == 'espn_scraper':
            return await self._get_espn_fixtures(days_ahead)
        else:
            return []
    
    async def _get_football_data_org_fixtures(self, days_ahead: int) -> List[FixtureData]:
        """Get fixtures from Football-Data.org API."""
        if not self.api_keys.get('football_data'):
            logger.warning("⚠️ No Football-Data API key available")
            return []
        
        fixtures = []
        
        try:
            headers = {'X-Auth-Token': self.api_keys['football_data']}
            
            # Get fixtures for major leagues
            leagues = ['PL', 'PD', 'BL1', 'SA', 'FL1']  # Premier League, La Liga, Bundesliga, Serie A, Ligue 1
            
            for league in leagues:
                try:
                    date_from = datetime.now().strftime('%Y-%m-%d')
                    date_to = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                    
                    url = f"{self.data_sources['football_data_org']['base_url']}/competitions/{league}/matches"
                    params = {
                        'dateFrom': date_from,
                        'dateTo': date_to
                    }
                    
                    # Simple manual retry with backoff + jitter
                    max_attempts = 3
                    for attempt in range(1, max_attempts + 1):
                        try:
                            response = requests.get(url, headers=headers, params=params, timeout=10)
                            break
                        except Exception as net_err:
                            if attempt == max_attempts:
                                raise net_err
                            sleep_for = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                            logger.warning(f"Network error fetching {league} attempt {attempt}: {net_err} - retrying in {sleep_for:.1f}s")
                            await asyncio.sleep(sleep_for)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for match in data.get('matches', []):
                            try:
                                fixture = FixtureData(
                                    home_team=match['homeTeam']['name'],
                                    away_team=match['awayTeam']['name'],
                                    match_date=datetime.fromisoformat(match['utcDate'].replace('Z', '')),
                                    league=self._get_league_display_name(league),
                                    status=match.get('status', 'SCHEDULED').lower(),
                                    home_score=match['score']['fullTime'].get('home'),
                                    away_score=match['score']['fullTime'].get('away'),
                                    venue=match.get('venue', ''),
                                    match_id=str(match['id']),
                                    confidence=0.95,  # High confidence for official API
                                    data_source='Football-Data.org'
                                )
                                fixtures.append(fixture)
                                
                            except Exception as match_error:
                                logger.warning(f"Error processing match: {match_error}")
                                continue
                    
                    # Rate limiting
                    await asyncio.sleep(6)  # 10 requests per minute
                    
                except Exception as league_error:
                    logger.warning(f"Error fetching {league}: {league_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in Football-Data.org fixture fetching: {e}")
        
        return fixtures

    # ----------------- Persistent Cache Helpers -----------------
    def _disk_cache_dir(self) -> Path:
        base = Path('data') / 'cache' / 'fixtures'
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _disk_cache_path(self, cache_key: str) -> Path:
        return self._disk_cache_dir() / f"{cache_key}.json"

    def _load_disk_cache(self, cache_key: str, days_ahead: int) -> Optional[List[FixtureData]]:
        path = self._disk_cache_path(cache_key)
        if not path.exists():
            return None
        try:
            stat = path.stat()
            age = datetime.now().timestamp() - stat.st_mtime
            if age > getattr(_settings,'FIXTURE_CACHE_TTL',900):
                return None
            with path.open('r', encoding='utf-8') as fh:
                data = _json.load(fh)
            fixtures = [FixtureData(**{**f, 'match_date': datetime.fromisoformat(f['match_date'])}) for f in data]
            return fixtures
        except Exception:
            return None

    def _store_disk_cache(self, cache_key: str, fixtures: List[FixtureData]) -> None:
        path = self._disk_cache_path(cache_key)
        serial = [self._serialize_fixture(f) for f in fixtures]
        with path.open('w', encoding='utf-8') as fh:
            _json.dump(serial, fh, ensure_ascii=False, indent=2, default=str)

    def _serialize_fixture(self, fixture: FixtureData) -> Dict[str, Any]:
        d = asdict(fixture)
        d['match_date'] = d['match_date'].isoformat()
        d['last_updated'] = d['last_updated'].isoformat() if isinstance(d['last_updated'], datetime) else d['last_updated']
        return d
    
    async def _get_espn_fixtures(self, days_ahead: int) -> List[FixtureData]:
        """Get fixtures from ESPN API (publicly available endpoints)."""
        fixtures = []
        
        try:
            # ESPN has publicly available sports data
            leagues = [
                {'id': 'eng.1', 'name': 'Premier League'},
                {'id': 'esp.1', 'name': 'La Liga'},
                {'id': 'ger.1', 'name': 'Bundesliga'},
                {'id': 'ita.1', 'name': 'Serie A'},
                {'id': 'fra.1', 'name': 'Ligue 1'}
            ]
            
            for league in leagues:
                try:
                    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league['id']}/scoreboard"
                    
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        for event in data.get('events', []):
                            try:
                                # Parse the event data
                                home_team = event['competitions'][0]['competitors'][0]['team']['displayName']
                                away_team = event['competitions'][0]['competitors'][1]['team']['displayName']
                                
                                # Ensure home/away assignment is correct
                                if event['competitions'][0]['competitors'][0]['homeAway'] != 'home':
                                    home_team, away_team = away_team, home_team
                                
                                match_date = datetime.fromisoformat(event['date'].replace('Z', ''))
                                
                                # Only include future matches within our date range
                                if match_date.date() >= datetime.now().date() and match_date <= datetime.now() + timedelta(days=days_ahead):
                                    fixture = FixtureData(
                                        home_team=home_team,
                                        away_team=away_team,
                                        match_date=match_date,
                                        league=league['name'],
                                        status=event.get('status', {}).get('type', {}).get('name', 'scheduled').lower(),
                                        venue=event.get('competitions', [{}])[0].get('venue', {}).get('fullName', ''),
                                        match_id=str(event.get('id', '')),
                                        confidence=0.85,  # Good confidence for ESPN data
                                        data_source='ESPN'
                                    )
                                    fixtures.append(fixture)
                                    
                            except Exception as match_error:
                                logger.warning(f"Error processing ESPN match: {match_error}")
                                continue
                    
                    # Rate limiting
                    await asyncio.sleep(1)
                    
                except Exception as league_error:
                    logger.warning(f"Error fetching ESPN {league['name']}: {league_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in ESPN fixture fetching: {e}")
        
        return fixtures
    
    async def _get_api_football_fixtures(self, days_ahead: int) -> List[FixtureData]:
        """Get fixtures from API-Football (if key available)."""
        # Placeholder for API-Football integration
        # This would require a paid subscription for meaningful data
        return []
    
    def _deduplicate_fixtures(self, fixtures: List[FixtureData]) -> List[FixtureData]:
        """Remove duplicate fixtures and keep the highest confidence version."""
        fixture_map = {}
        
        for fixture in fixtures:
            # Create a key based on teams and approximate match time
            key = (
                fixture.home_team.lower().strip(),
                fixture.away_team.lower().strip(),
                fixture.match_date.date()
            )
            
            # Keep the fixture with highest confidence
            if key not in fixture_map or fixture.confidence > fixture_map[key].confidence:
                fixture_map[key] = fixture
        
        return list(fixture_map.values())
    
    def _generate_intelligent_fallback_fixtures(self, days_ahead: int) -> List[FixtureData]:
        """Generate intelligent fallback fixtures based on real football schedules."""
        fallback_fixtures = []

        # Prefer leveraging RealDataIntegrator data if available before synthetic fixtures
        try:
            from real_data_integrator import get_real_matches

            real_matches = get_real_matches(days_ahead)
            for match in real_matches:
                if len(fallback_fixtures) >= 8:
                    break
                fixture = self._build_fixture_from_real_match(match)
                if fixture:
                    fallback_fixtures.append(fixture)

            if fallback_fixtures:
                logger.info("[RealDataFallback] Injected %d fixtures from RealDataIntegrator", len(fallback_fixtures))
                return fallback_fixtures
        except Exception as exc:
            logger.debug("[RealDataFallback] Unable to hydrate from RealDataIntegrator: %s", exc)
        
        # Realistic team matchups based on current season patterns
        realistic_matchups = [
            # Premier League
            ('Manchester City', 'Arsenal', 'Premier League', 'Etihad Stadium'),
            ('Liverpool', 'Chelsea', 'Premier League', 'Anfield'),
            ('Manchester United', 'Tottenham', 'Premier League', 'Old Trafford'),
            ('Newcastle', 'Brighton', 'Premier League', 'St. James\' Park'),
            
            # La Liga
            ('Real Madrid', 'Barcelona', 'La Liga', 'Santiago Bernabéu'),
            ('Atlético Madrid', 'Valencia', 'La Liga', 'Wanda Metropolitano'),
            ('Sevilla', 'Real Betis', 'La Liga', 'Ramón Sánchez Pizjuán'),
            
            # Bundesliga
            ('Bayern Munich', 'Borussia Dortmund', 'Bundesliga', 'Allianz Arena'),
            ('RB Leipzig', 'Bayer Leverkusen', 'Bundesliga', 'Red Bull Arena'),
            
            # Serie A
            ('Juventus', 'AC Milan', 'Serie A', 'Allianz Stadium'),
            ('Inter Milan', 'Napoli', 'Serie A', 'San Siro'),
            
            # Ligue 1
            ('Paris Saint-Germain', 'Marseille', 'Ligue 1', 'Parc des Princes'),
        ]
        
        import random
        
        for i, (home_team, away_team, league, venue) in enumerate(realistic_matchups):
            if len(fallback_fixtures) >= 8:  # Limit fallback fixtures
                break
            
            # Generate realistic future dates
            match_date = datetime.now() + timedelta(days=random.randint(1, days_ahead))
            
            # Set realistic match times based on league
            if league == 'Premier League':
                match_times = ['12:30', '15:00', '17:30']
            elif league == 'La Liga':
                match_times = ['16:15', '18:30', '21:00']
            elif league == 'Bundesliga':
                match_times = ['15:30', '18:30']
            elif league == 'Serie A':
                match_times = ['15:00', '18:00', '20:45']
            else:
                match_times = ['15:00', '18:00', '21:00']
            
            match_time = random.choice(match_times)
            match_hour, match_minute = map(int, match_time.split(':'))
            match_date = match_date.replace(hour=match_hour, minute=match_minute, second=0, microsecond=0)
            
            fixture = FixtureData(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                league=league,
                status='scheduled',
                venue=venue,
                match_id=f'fallback_{i}',
                confidence=0.7,  # Medium confidence for fallback data
                data_source='Intelligent Fallback'
            )
            fallback_fixtures.append(fixture)
        
        return fallback_fixtures

    def _build_fixture_from_real_match(self, match: Dict[str, Any]) -> Optional[FixtureData]:
        """Safely convert a RealDataIntegrator match dictionary into FixtureData."""
        try:
            home_team = match.get('home_team') or match.get('home')
            away_team = match.get('away_team') or match.get('away')
            if not home_team or not away_team:
                return None

            raw_date = match.get('match_date') or match.get('date')
            if isinstance(raw_date, datetime):
                match_date = raw_date
            elif isinstance(raw_date, str):
                try:
                    match_date = datetime.fromisoformat(raw_date.replace('Z', ''))
                except ValueError:
                    match_date = datetime.now() + timedelta(days=1)
            else:
                match_date = datetime.now() + timedelta(days=1)

            league = match.get('league') or match.get('competition') or 'Unknown League'
            match_id = str(match.get('api_id') or match.get('id') or f"real_{abs(hash(home_team + away_team + match_date.isoformat()))}")

            venue = match.get('venue')
            status = str(match.get('status') or 'scheduled').lower()
            confidence = 0.92 if not match_id.startswith('fallback_') else 0.6

            return FixtureData(
                home_team=str(home_team),
                away_team=str(away_team),
                match_date=match_date,
                league=str(league),
                status=status,
                home_score=match.get('home_score'),
                away_score=match.get('away_score'),
                venue=venue,
                match_id=match_id,
                confidence=confidence,
                data_source='RealDataIntegrator'
            )
        except Exception as exc:  # pragma: no cover - defensive conversion guard
            logger.debug("[RealDataFallback] Failed to translate real match: %s", exc)
            return None
    
    def _get_league_display_name(self, league_code: str) -> str:
        """Convert league code to display name."""
        league_names = {
            'PL': 'Premier League',
            'PD': 'La Liga',
            'BL1': 'Bundesliga',
            'SA': 'Serie A',
            'FL1': 'Ligue 1',
            'DED': 'Eredivisie'
        }
        return league_names.get(league_code, league_code)
    
    def get_fixtures_sync(self, days_ahead: int = 7) -> List[Dict]:
        """Synchronous wrapper for getting fixtures."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            fixtures = loop.run_until_complete(self.get_current_fixtures(days_ahead))

            # Convert to dictionaries for easier consumption and normalize
            dicts = [asdict(fixture) for fixture in fixtures]
            normalized = [normalize_fixture_dict(d) for d in dicts]
            return normalized
            
        except Exception as e:
            logger.error(f"Error in sync fixture retrieval: {e}")
            return self._generate_sync_fallback_fixtures(days_ahead)
        finally:
            try:
                loop.close()
            except:
                pass
    
    def _generate_sync_fallback_fixtures(self, days_ahead: int) -> List[Dict]:
        """Generate synchronous fallback fixtures."""
        fallback_fixtures = self._generate_intelligent_fallback_fixtures(days_ahead)
        return [asdict(fixture) for fixture in fallback_fixtures]


# Global instance
enhanced_data_aggregator = EnhancedDataAggregator()


def get_current_fixtures(days_ahead: int = 7) -> List[Dict]:
    """
    Get current fixtures from multiple sources.
    
    Args:
        days_ahead: Number of days ahead to fetch fixtures
        
    Returns:
        List of fixture dictionaries
    """
    return enhanced_data_aggregator.get_fixtures_sync(days_ahead)


def normalize_fixture_dict(fixture: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a fixture dictionary to a consistent schema and types.

    Ensures:
    - match_date is ISO8601 string
    - status is lowercase
    - scores are ints or None
    - confidence is float between 0 and 1
    - league is a readable name
    """
    out = {}

    # Basic string fields
    out['home_team'] = str(fixture.get('home_team') or '').strip()
    out['away_team'] = str(fixture.get('away_team') or '').strip()
    out['league'] = str(fixture.get('league') or '')

    # Status
    status = fixture.get('status') or 'scheduled'
    out['status'] = str(status).lower()

    # Match date
    md = fixture.get('match_date')
    try:
        if isinstance(md, str):
            # Accept ISO strings with or without Z
            out['match_date'] = datetime.fromisoformat(md.replace('Z', '')).isoformat()
        elif isinstance(md, dict):
            # Malformed; drop
            out['match_date'] = None
        elif hasattr(md, 'isoformat'):
            out['match_date'] = md.isoformat()
        else:
            out['match_date'] = None
    except Exception:
        out['match_date'] = None

    # Scores
    def to_int_or_none(v):
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None

    out['home_score'] = to_int_or_none(fixture.get('home_score'))
    out['away_score'] = to_int_or_none(fixture.get('away_score'))

    # Venue
    out['venue'] = fixture.get('venue') or ''

    # IDs and meta
    out['match_id'] = fixture.get('match_id') or fixture.get('id') or None
    out['data_source'] = fixture.get('data_source') or 'unknown'

    # Confidence
    try:
        conf = float(fixture.get('confidence') if fixture.get('confidence') is not None else 1.0)
        out['confidence'] = max(0.0, min(1.0, conf))
    except Exception:
        out['confidence'] = 0.5

    # last_updated
    lu = fixture.get('last_updated')
    try:
        if isinstance(lu, str):
            out['last_updated'] = datetime.fromisoformat(lu.replace('Z', '')).isoformat()
        elif hasattr(lu, 'isoformat'):
            out['last_updated'] = lu.isoformat()
        else:
            out['last_updated'] = datetime.now().isoformat()
    except Exception:
        out['last_updated'] = datetime.now().isoformat()

    return out


def get_todays_matches() -> List[Dict]:
    """Get today's matches specifically."""
    fixtures = get_current_fixtures(days_ahead=1)
    today = datetime.now().date()
    
    todays_fixtures = []
    for fixture in fixtures:
        try:
            fixture_date = fixture['match_date']
            if isinstance(fixture_date, str):
                fixture_date = datetime.fromisoformat(fixture_date)
            elif isinstance(fixture_date, dict):
                continue  # Skip malformed dates
            
            if fixture_date.date() == today:
                todays_fixtures.append(fixture)
        except:
            continue
    
    return todays_fixtures


def get_featured_matches(limit: int = 5) -> List[Dict]:
    """Return a curated subset of high-profile upcoming fixtures.

    This is a lightweight convenience API used by the UI/tests. It favors:
    - Top leagues (Premier League, La Liga, Bundesliga, Serie A, Ligue 1)
    - High confidence real API sourced fixtures first
    - Falls back to intelligently generated fixtures if real ones are scarce

    Args:
        limit: Maximum number of featured matches to return.

    Returns:
        List of normalized fixture dicts (may be empty on persistent errors).
    """
    try:
        fixtures = get_current_fixtures(days_ahead=7)
    except Exception as e:
        logger.warning(f"get_featured_matches fallback due to error: {e}")
        fixtures = []

    if not fixtures:
        # Use sync fallback generator directly by instantiating a local aggregator if needed
        try:
            local = EnhancedDataAggregator()
            fixtures = local.get_fixtures_sync(days_ahead=7)
        except Exception:
            return []

    # Rank: confidence desc, then earliest match_date
    def sort_key(fx):
        # Protect against missing fields
        md = fx.get('match_date')
        try:
            if isinstance(md, str):
                md_dt = datetime.fromisoformat(md.replace('Z',''))
            else:
                md_dt = md if hasattr(md, 'isoformat') else datetime.max
        except Exception:
            md_dt = datetime.max
        return (-float(fx.get('confidence', 0.0)), md_dt)

    # Prefer top leagues
    top_leagues = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    primary = [f for f in fixtures if f.get('league') in top_leagues]
    secondary = [f for f in fixtures if f.get('league') not in top_leagues]

    ranked = sorted(primary, key=sort_key) + sorted(secondary, key=sort_key)
    return ranked[:limit]


if __name__ == "__main__":
    # Test the enhanced data aggregator
    print("🧪 Testing Enhanced Data Aggregator")
    print("=" * 50)
    
    aggregator = EnhancedDataAggregator()
    
    # Test fixture retrieval
    fixtures = aggregator.get_fixtures_sync(days_ahead=7)
    print(f"✅ Retrieved {len(fixtures)} fixtures")
    
    if fixtures:
        print("\n📋 Sample fixtures:")
        for i, fixture in enumerate(fixtures[:3]):
            print(f"  {i+1}. {fixture['home_team']} vs {fixture['away_team']}")
            print(f"     {fixture['league']} - {fixture['match_date']}")
            print(f"     Source: {fixture['data_source']} (Confidence: {fixture['confidence']})")
            print()
    
    # Test today's matches
    todays_matches = get_todays_matches()
    print(f"🎯 Today's matches: {len(todays_matches)}")
