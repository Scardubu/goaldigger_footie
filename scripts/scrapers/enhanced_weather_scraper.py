"""
Enhanced Weather Data Scraper
Integrates multiple weather APIs for comprehensive match weather analysis.
"""
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import pandas as pd

from scripts.core.data_quality_monitor import DataQualityMonitor
from scripts.scrapers.base_scraper import BaseScraper

logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Comprehensive weather data for a match."""
    match_id: str
    venue_name: str
    latitude: float
    longitude: float
    temperature_celsius: float
    humidity_percent: int
    wind_speed_kmh: float
    wind_direction: str
    precipitation_mm: float
    visibility_km: int
    weather_condition: str
    pressure_hpa: float
    uv_index: int
    forecast_accuracy: float
    data_source: str
    recorded_at: datetime

@dataclass
class VenueCoordinates:
    """Venue location coordinates."""
    venue_name: str
    latitude: float
    longitude: float
    city: str
    country: str

class EnhancedWeatherScraper(BaseScraper):
    """Enhanced weather scraper with multiple API sources and historical data."""
    
    def __init__(self):
        super().__init__(
            name="enhanced_weather",
            base_url="https://api.openweathermap.org/data/2.5",
            rate_limit_delay=(0.5, 1.0),
            max_retries=3
        )
        
        # API configurations
        self.api_configs = {
            'openweather': {
                'base_url': 'https://api.openweathermap.org/data/2.5',
                'api_key': os.getenv('OPENWEATHER_API_KEY'),
                'endpoints': {
                    'current': '/weather',
                    'forecast': '/forecast',
                    'historical': '/onecall/timemachine'
                }
            },
            'weatherapi': {
                'base_url': 'https://api.weatherapi.com/v1',
                'api_key': os.getenv('WEATHERAPI_KEY'),
                'endpoints': {
                    'current': '/current.json',
                    'forecast': '/forecast.json',
                    'historical': '/history.json'
                }
            },
            'visualcrossing': {
                'base_url': 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline',
                'api_key': os.getenv('VISUALCROSSING_API_KEY')
            }
        }
        
        # Venue coordinates cache
        self.venue_coordinates = self._load_venue_coordinates()
        
        # Quality monitor
        self.quality_monitor = DataQualityMonitor()
        
        # Weather condition mapping
        self.condition_mapping = {
            'clear': 'clear',
            'sunny': 'clear',
            'partly cloudy': 'partly_cloudy',
            'cloudy': 'cloudy',
            'overcast': 'overcast',
            'rain': 'rain',
            'drizzle': 'light_rain',
            'heavy rain': 'heavy_rain',
            'snow': 'snow',
            'fog': 'fog',
            'mist': 'mist',
            'thunderstorm': 'thunderstorm'
        }
    
    def _load_venue_coordinates(self) -> Dict[str, VenueCoordinates]:
        """Load venue coordinates from configuration."""
        # Major football venues with coordinates
        venues = {
            'Emirates Stadium': VenueCoordinates('Emirates Stadium', 51.5549, -0.1084, 'London', 'England'),
            'Old Trafford': VenueCoordinates('Old Trafford', 53.4631, -2.2914, 'Manchester', 'England'),
            'Anfield': VenueCoordinates('Anfield', 53.4308, -2.9608, 'Liverpool', 'England'),
            'Stamford Bridge': VenueCoordinates('Stamford Bridge', 51.4816, -0.1909, 'London', 'England'),
            'Etihad Stadium': VenueCoordinates('Etihad Stadium', 53.4831, -2.2004, 'Manchester', 'England'),
            'Tottenham Hotspur Stadium': VenueCoordinates('Tottenham Hotspur Stadium', 51.6042, -0.0664, 'London', 'England'),
            'Camp Nou': VenueCoordinates('Camp Nou', 41.3809, 2.1228, 'Barcelona', 'Spain'),
            'Santiago Bernabéu': VenueCoordinates('Santiago Bernabéu', 40.4530, -3.6883, 'Madrid', 'Spain'),
            'San Siro': VenueCoordinates('San Siro', 45.4781, 9.1240, 'Milan', 'Italy'),
            'Allianz Arena': VenueCoordinates('Allianz Arena', 48.2188, 11.6242, 'Munich', 'Germany'),
            'Signal Iduna Park': VenueCoordinates('Signal Iduna Park', 51.4926, 7.4516, 'Dortmund', 'Germany'),
            'Parc des Princes': VenueCoordinates('Parc des Princes', 48.8414, 2.2530, 'Paris', 'France')
        }
        return venues
    
    async def get_comprehensive_weather_data(self, venue: str, match_date: datetime) -> Optional[WeatherData]:
        """Get comprehensive weather data from multiple sources."""
        venue_coords = self._get_venue_coordinates(venue)
        if not venue_coords:
            logger.warning(f"No coordinates found for venue: {venue}")
            return None
        
        weather_data = []
        
        # Collect data from multiple sources
        for source_name, config in self.api_configs.items():
            if not config.get('api_key'):
                logger.debug(f"No API key for {source_name}, skipping")
                continue
            
            try:
                if source_name == 'openweather':
                    data = await self._get_openweather_data(venue_coords, match_date)
                elif source_name == 'weatherapi':
                    data = await self._get_weatherapi_data(venue_coords, match_date)
                elif source_name == 'visualcrossing':
                    data = await self._get_visualcrossing_data(venue_coords, match_date)
                else:
                    continue
                
                if data:
                    weather_data.append(data)
                    logger.info(f"Retrieved weather data from {source_name}")
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to get weather from {source_name}: {e}")
                continue
        
        if not weather_data:
            logger.error(f"No weather data available for {venue}")
            return None
        
        # Merge and validate data
        merged_data = self._merge_weather_data(weather_data, venue, match_date)
        
        # Quality monitoring
        await self.quality_monitor.monitor_data_quality(
            'weather_data',
            [merged_data.__dict__] if merged_data else []
        )
        
        return merged_data
    
    def _get_venue_coordinates(self, venue: str) -> Optional[VenueCoordinates]:
        """Get coordinates for a venue."""
        # Direct lookup
        if venue in self.venue_coordinates:
            return self.venue_coordinates[venue]
        
        # Fuzzy matching for similar names
        venue_lower = venue.lower()
        for venue_name, coords in self.venue_coordinates.items():
            if venue_lower in venue_name.lower() or venue_name.lower() in venue_lower:
                return coords
        
        return None
    
    async def _get_openweather_data(self, venue_coords: VenueCoordinates, match_date: datetime) -> Optional[WeatherData]:
        """Get weather data from OpenWeatherMap."""
        config = self.api_configs['openweather']
        
        try:
            # Determine if we need current, forecast, or historical data
            now = datetime.now()
            time_diff = (match_date - now).total_seconds()
            
            if abs(time_diff) < 3600:  # Within 1 hour - current weather
                endpoint = config['endpoints']['current']
                params = {
                    'lat': venue_coords.latitude,
                    'lon': venue_coords.longitude,
                    'appid': config['api_key'],
                    'units': 'metric'
                }
            elif time_diff > 0:  # Future - forecast
                endpoint = config['endpoints']['forecast']
                params = {
                    'lat': venue_coords.latitude,
                    'lon': venue_coords.longitude,
                    'appid': config['api_key'],
                    'units': 'metric'
                }
            else:  # Past - historical (requires different API)
                # Historical data requires One Call API subscription
                return None
            
            url = f"{config['base_url']}{endpoint}"
            
            # Explicit timeout to ensure weather API calls do not hang indefinitely
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_openweather_data(data, venue_coords, match_date)
                    else:
                        logger.warning(f"OpenWeather API error: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"OpenWeather API error: {e}")
            return None
    
    def _parse_openweather_data(self, data: Dict, venue_coords: VenueCoordinates, match_date: datetime) -> WeatherData:
        """Parse OpenWeatherMap API response."""
        if 'list' in data:  # Forecast data
            # Find closest forecast to match time
            forecasts = data['list']
            closest_forecast = min(
                forecasts,
                key=lambda x: abs(datetime.fromtimestamp(x['dt']) - match_date)
            )
            weather_info = closest_forecast
        else:  # Current weather
            weather_info = data
        
        main = weather_info.get('main', {})
        weather = weather_info.get('weather', [{}])[0]
        wind = weather_info.get('wind', {})
        
        return WeatherData(
            match_id=f"{venue_coords.venue_name}_{match_date.strftime('%Y%m%d_%H%M')}",
            venue_name=venue_coords.venue_name,
            latitude=venue_coords.latitude,
            longitude=venue_coords.longitude,
            temperature_celsius=main.get('temp', 15.0),
            humidity_percent=main.get('humidity', 50),
            wind_speed_kmh=wind.get('speed', 0) * 3.6,  # Convert m/s to km/h
            wind_direction=self._degrees_to_direction(wind.get('deg', 0)),
            precipitation_mm=weather_info.get('rain', {}).get('1h', 0) + weather_info.get('snow', {}).get('1h', 0),
            visibility_km=weather_info.get('visibility', 10000) / 1000,  # Convert m to km
            weather_condition=self._normalize_weather_condition(weather.get('main', 'clear')),
            pressure_hpa=main.get('pressure', 1013),
            uv_index=0,  # Not available in basic API
            forecast_accuracy=0.85,  # OpenWeather typical accuracy
            data_source='openweather',
            recorded_at=datetime.now()
        )
    
    async def _get_weatherapi_data(self, venue_coords: VenueCoordinates, match_date: datetime) -> Optional[WeatherData]:
        """Get weather data from WeatherAPI."""
        config = self.api_configs['weatherapi']
        
        try:
            now = datetime.now()
            time_diff = (match_date - now).total_seconds()
            
            if abs(time_diff) < 3600:  # Current weather
                endpoint = config['endpoints']['current']
                params = {
                    'key': config['api_key'],
                    'q': f"{venue_coords.latitude},{venue_coords.longitude}",
                    'aqi': 'no'
                }
            elif time_diff > 0 and time_diff < 10 * 24 * 3600:  # Forecast (up to 10 days)
                endpoint = config['endpoints']['forecast']
                params = {
                    'key': config['api_key'],
                    'q': f"{venue_coords.latitude},{venue_coords.longitude}",
                    'days': min(10, int(time_diff / (24 * 3600)) + 1),
                    'aqi': 'no'
                }
            else:  # Historical data
                endpoint = config['endpoints']['historical']
                params = {
                    'key': config['api_key'],
                    'q': f"{venue_coords.latitude},{venue_coords.longitude}",
                    'dt': match_date.strftime('%Y-%m-%d')
                }
            
            url = f"{config['base_url']}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_weatherapi_data(data, venue_coords, match_date)
                    else:
                        logger.warning(f"WeatherAPI error: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"WeatherAPI error: {e}")
            return None
    
    def _parse_weatherapi_data(self, data: Dict, venue_coords: VenueCoordinates, match_date: datetime) -> WeatherData:
        """Parse WeatherAPI response."""
        if 'forecast' in data:
            # Find the right forecast day and hour
            forecast_day = None
            for day in data['forecast']['forecastday']:
                if day['date'] == match_date.strftime('%Y-%m-%d'):
                    forecast_day = day
                    break
            
            if forecast_day:
                # Find closest hour
                target_hour = match_date.hour
                closest_hour = min(
                    forecast_day['hour'],
                    key=lambda x: abs(int(x['time'].split(' ')[1].split(':')[0]) - target_hour)
                )
                weather_info = closest_hour
            else:
                weather_info = data['current']
        else:
            weather_info = data.get('current', data)
        
        return WeatherData(
            match_id=f"{venue_coords.venue_name}_{match_date.strftime('%Y%m%d_%H%M')}",
            venue_name=venue_coords.venue_name,
            latitude=venue_coords.latitude,
            longitude=venue_coords.longitude,
            temperature_celsius=weather_info.get('temp_c', 15.0),
            humidity_percent=weather_info.get('humidity', 50),
            wind_speed_kmh=weather_info.get('wind_kph', 0),
            wind_direction=weather_info.get('wind_dir', 'N'),
            precipitation_mm=weather_info.get('precip_mm', 0),
            visibility_km=weather_info.get('vis_km', 10),
            weather_condition=self._normalize_weather_condition(weather_info.get('condition', {}).get('text', 'clear')),
            pressure_hpa=weather_info.get('pressure_mb', 1013),
            uv_index=weather_info.get('uv', 0),
            forecast_accuracy=0.88,  # WeatherAPI typical accuracy
            data_source='weatherapi',
            recorded_at=datetime.now()
        )
    
    async def _get_visualcrossing_data(self, venue_coords: VenueCoordinates, match_date: datetime) -> Optional[WeatherData]:
        """Get weather data from Visual Crossing."""
        # Implementation for Visual Crossing API
        # This would be similar to the above methods
        return None
    
    def _merge_weather_data(self, weather_data: List[WeatherData], venue: str, match_date: datetime) -> WeatherData:
        """Merge weather data from multiple sources."""
        if len(weather_data) == 1:
            return weather_data[0]
        
        # Calculate weighted averages based on source reliability
        source_weights = {
            'openweather': 0.4,
            'weatherapi': 0.4,
            'visualcrossing': 0.2
        }
        
        # Weighted averages for numerical values
        total_weight = sum(source_weights.get(w.data_source, 0.1) for w in weather_data)
        
        merged = WeatherData(
            match_id=f"{venue}_{match_date.strftime('%Y%m%d_%H%M')}",
            venue_name=venue,
            latitude=weather_data[0].latitude,
            longitude=weather_data[0].longitude,
            temperature_celsius=sum(w.temperature_celsius * source_weights.get(w.data_source, 0.1) for w in weather_data) / total_weight,
            humidity_percent=int(sum(w.humidity_percent * source_weights.get(w.data_source, 0.1) for w in weather_data) / total_weight),
            wind_speed_kmh=sum(w.wind_speed_kmh * source_weights.get(w.data_source, 0.1) for w in weather_data) / total_weight,
            wind_direction=weather_data[0].wind_direction,  # Use first source
            precipitation_mm=sum(w.precipitation_mm * source_weights.get(w.data_source, 0.1) for w in weather_data) / total_weight,
            visibility_km=int(sum(w.visibility_km * source_weights.get(w.data_source, 0.1) for w in weather_data) / total_weight),
            weather_condition=weather_data[0].weather_condition,  # Use most reliable source
            pressure_hpa=sum(w.pressure_hpa * source_weights.get(w.data_source, 0.1) for w in weather_data) / total_weight,
            uv_index=int(sum(w.uv_index * source_weights.get(w.data_source, 0.1) for w in weather_data) / total_weight),
            forecast_accuracy=sum(w.forecast_accuracy * source_weights.get(w.data_source, 0.1) for w in weather_data) / total_weight,
            data_source=', '.join(w.data_source for w in weather_data),
            recorded_at=datetime.now()
        )
        
        return merged
    
    def _normalize_weather_condition(self, condition: str) -> str:
        """Normalize weather condition to standard format."""
        condition_lower = condition.lower()
        
        for key, normalized in self.condition_mapping.items():
            if key in condition_lower:
                return normalized
        
        return 'clear'  # Default
    
    def _degrees_to_direction(self, degrees: float) -> str:
        """Convert wind degrees to direction."""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        
        index = int((degrees + 11.25) / 22.5) % 16
        return directions[index]

    async def get_matches(self, league_code: str, date_from: Optional[datetime] = None,
                         date_to: Optional[datetime] = None, **kwargs) -> Optional[pd.DataFrame]:
        """
        Get matches for weather analysis.

        Args:
            league_code: League identifier
            date_from: Start date for match data
            date_to: End date for match data
            **kwargs: Additional parameters

        Returns:
            DataFrame with match data or None if failed
        """
        try:
            # This scraper focuses on weather data, not match data
            # Return empty DataFrame to satisfy abstract method requirement
            logger.info(f"get_matches called for {league_code} - returning empty DataFrame")
            return pd.DataFrame(columns=['match_id', 'home_team', 'away_team', 'date', 'venue'])

        except Exception as e:
            logger.error(f"Error in get_matches: {e}")
            return None

    async def get_team_info(self, team_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Get team information for weather analysis.

        Args:
            team_id: Team identifier

        Returns:
            Dictionary with team information or None if failed
        """
        try:
            # Return basic team info structure for weather analysis
            logger.info(f"get_team_info called for team {team_id}")
            return {
                'team_id': team_id,
                'name': str(team_id),
                'weather_data_available': True,
                'home_venue': f"{team_id} Stadium",
                'last_updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in get_team_info: {e}")
            return None

# Usage example
async def main():
    """Example usage of the enhanced weather scraper."""
    scraper = EnhancedWeatherScraper()
    
    # Get weather for a match
    match_date = datetime.now() + timedelta(hours=2)
    weather = await scraper.get_comprehensive_weather_data('Emirates Stadium', match_date)
    
    if weather:
        print(f"Weather for {weather.venue_name}:")
        print(f"Temperature: {weather.temperature_celsius}°C")
        print(f"Condition: {weather.weather_condition}")
        print(f"Wind: {weather.wind_speed_kmh} km/h {weather.wind_direction}")
        print(f"Humidity: {weather.humidity_percent}%")
        print(f"Precipitation: {weather.precipitation_mm}mm")

if __name__ == "__main__":
    asyncio.run(main())
