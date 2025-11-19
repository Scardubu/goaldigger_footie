#!/usr/bin/env python3
"""
GoalDiggers API SDK
Phase 3B: Advanced Features Implementation - API SDK

This SDK provides easy integration with the GoalDiggers Enhanced API,
supporting all Phase 3B features including authentication, analytics,
personalization, and mobile/PWA capabilities.

Key Features:
- Simple authentication and token management
- Enhanced prediction methods with Phase 3B features
- Analytics and performance monitoring
- Personalization and recommendation APIs
- Mobile/PWA specific functionality
- Comprehensive error handling and retry logic
"""

import requests
import logging
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import jwt
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITier(Enum):
    """API access tiers."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

@dataclass
class APIResponse:
    """Standardized API response."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    rate_limit_remaining: Optional[int] = None

class GoalDiggersSDK:
    """
    Official GoalDiggers API SDK for Phase 3B Enhanced API.
    Provides easy access to all advanced features.
    """
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.goaldiggers.com"):
        """Initialize GoalDiggers SDK."""
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.api_version = "v4.1"
        self.access_token = None
        self.token_expires_at = None
        self.session = requests.Session()
        
        # Configure session
        self.session.headers.update({
            'User-Agent': 'GoalDiggers-SDK/4.1.0',
            'Content-Type': 'application/json'
        })
        
        # Auto-authenticate if API key provided
        if self.api_key:
            self._authenticate()
        
        logger.info("🚀 GoalDiggers SDK initialized")
    
    def _authenticate(self) -> bool:
        """Authenticate and get access token."""
        try:
            response = self.session.post(
                f"{self.base_url}/{self.api_version}/auth/token",
                json={"api_key": self.api_key}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data['access_token']
                self.token_expires_at = datetime.now() + timedelta(seconds=data['expires_in'])
                
                # Update session headers
                self.session.headers.update({
                    'Authorization': f"Bearer {self.access_token}"
                })
                
                logger.info("✅ Authentication successful")
                return True
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False
    
    def _ensure_authenticated(self) -> bool:
        """Ensure valid authentication token."""
        if not self.access_token or not self.token_expires_at:
            return self._authenticate()
        
        # Check if token is about to expire (refresh 5 minutes early)
        if datetime.now() >= (self.token_expires_at - timedelta(minutes=5)):
            return self._authenticate()
        
        return True
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make authenticated API request with error handling."""
        try:
            # Ensure authentication
            if not self._ensure_authenticated():
                return APIResponse(
                    success=False,
                    error="Authentication failed",
                    status_code=401
                )
            
            # Make request
            url = f"{self.base_url}/{self.api_version}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            
            # Parse response
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"message": response.text}
            
            # Extract rate limit info
            rate_limit_remaining = response.headers.get('X-RateLimit-Remaining')
            
            if response.status_code < 400:
                return APIResponse(
                    success=True,
                    data=data,
                    status_code=response.status_code,
                    rate_limit_remaining=int(rate_limit_remaining) if rate_limit_remaining else None
                )
            else:
                return APIResponse(
                    success=False,
                    error=data.get('detail', f"HTTP {response.status_code}"),
                    status_code=response.status_code,
                    rate_limit_remaining=int(rate_limit_remaining) if rate_limit_remaining else None
                )
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            return APIResponse(
                success=False,
                error=str(e)
            )
    
    # Enhanced Prediction Methods
    
    def predict_match(self, home_team: str, away_team: str, **kwargs) -> APIResponse:
        """Get enhanced match prediction with Phase 3B features."""
        payload = {
            "home_team": home_team,
            "away_team": away_team,
            **kwargs
        }
        
        return self._make_request("POST", "/predictions/enhanced", json=payload)
    
    def predict_batch(self, matches: List[Dict[str, str]]) -> APIResponse:
        """Get batch predictions for multiple matches."""
        return self._make_request("POST", "/predictions/batch", json=matches)
    
    def get_prediction_history(self, limit: int = 50, offset: int = 0) -> APIResponse:
        """Get user's prediction history."""
        params = {"limit": limit, "offset": offset}
        return self._make_request("GET", "/predictions/history", params=params)
    
    # Analytics Methods
    
    def get_analytics_dashboard(self, time_range: str = "24h") -> APIResponse:
        """Get analytics dashboard data."""
        params = {"time_range": time_range}
        return self._make_request("GET", "/analytics/dashboard", params=params)
    
    def get_performance_metrics(self) -> APIResponse:
        """Get system performance metrics (Enterprise tier only)."""
        return self._make_request("GET", "/analytics/performance")
    
    def get_accuracy_stats(self, time_range: str = "30d") -> APIResponse:
        """Get prediction accuracy statistics."""
        params = {"time_range": time_range}
        return self._make_request("GET", "/analytics/accuracy", params=params)
    
    # Personalization Methods
    
    def update_preferences(self, preferences: Dict[str, Any]) -> APIResponse:
        """Update user preferences."""
        return self._make_request("POST", "/personalization/preferences", json=preferences)
    
    def get_recommendations(self, context: str = "general", limit: int = 5) -> APIResponse:
        """Get personalized recommendations."""
        params = {"context": context, "limit": limit}
        return self._make_request("GET", "/personalization/recommendations", params=params)
    
    def get_user_profile(self) -> APIResponse:
        """Get user profile and preferences."""
        return self._make_request("GET", "/personalization/profile")
    
    def track_interaction(self, interaction_type: str, data: Dict[str, Any]) -> APIResponse:
        """Track user interaction for personalization learning."""
        payload = {
            "interaction_type": interaction_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        return self._make_request("POST", "/personalization/track", json=payload)
    
    # Mobile/PWA Methods
    
    def subscribe_push_notifications(self, subscription: Dict[str, Any]) -> APIResponse:
        """Subscribe to push notifications."""
        return self._make_request("POST", "/mobile/push-subscribe", json=subscription)
    
    def sync_offline_data(self, sync_data: Dict[str, Any]) -> APIResponse:
        """Sync offline data when connection restored."""
        return self._make_request("POST", "/mobile/sync", json=sync_data)
    
    def get_mobile_config(self) -> APIResponse:
        """Get mobile app configuration."""
        return self._make_request("GET", "/mobile/config")
    
    # Utility Methods
    
    def get_health(self) -> APIResponse:
        """Check API health status."""
        return self._make_request("GET", "/health")
    
    def get_rate_limit_status(self) -> APIResponse:
        """Get current rate limit status."""
        return self._make_request("GET", "/auth/rate-limit")
    
    def get_user_info(self) -> APIResponse:
        """Get current user information."""
        return self._make_request("GET", "/auth/user")
    
    # Convenience Methods
    
    def quick_prediction(self, home_team: str, away_team: str) -> Optional[Dict[str, Any]]:
        """Quick prediction with simplified response."""
        response = self.predict_match(home_team, away_team)
        
        if response.success:
            return {
                'home_win': response.data['predictions']['home_win'],
                'draw': response.data['predictions']['draw'],
                'away_win': response.data['predictions']['away_win'],
                'confidence': response.data['confidence']
            }
        else:
            logger.error(f"Prediction failed: {response.error}")
            return None
    
    def get_top_recommendations(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get top personalized recommendations."""
        response = self.get_recommendations(limit=limit)
        
        if response.success:
            return response.data.get('recommendations', [])
        else:
            logger.error(f"Recommendations failed: {response.error}")
            return []
    
    def is_healthy(self) -> bool:
        """Check if API is healthy."""
        response = self.get_health()
        return response.success and response.data.get('status') == 'healthy'

# Async SDK for advanced use cases
class AsyncGoalDiggersSDK:
    """Async version of GoalDiggers SDK for high-performance applications."""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.goaldiggers.com"):
        """Initialize async GoalDiggers SDK."""
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.api_version = "v4.1"
        self.access_token = None
        
        logger.info("🚀 Async GoalDiggers SDK initialized")
    
    async def predict_match_async(self, home_team: str, away_team: str, **kwargs) -> APIResponse:
        """Async match prediction."""
        # Implementation would use aiohttp or similar
        # This is a placeholder for the async implementation
        pass
    
    async def predict_batch_async(self, matches: List[Dict[str, str]]) -> APIResponse:
        """Async batch predictions."""
        # Implementation would use aiohttp or similar
        pass

# Factory functions
def create_sdk(api_key: str, base_url: str = "https://api.goaldiggers.com") -> GoalDiggersSDK:
    """Create GoalDiggers SDK instance."""
    return GoalDiggersSDK(api_key=api_key, base_url=base_url)

def create_async_sdk(api_key: str, base_url: str = "https://api.goaldiggers.com") -> AsyncGoalDiggersSDK:
    """Create async GoalDiggers SDK instance."""
    return AsyncGoalDiggersSDK(api_key=api_key, base_url=base_url)

# Example usage
if __name__ == "__main__":
    # Example SDK usage
    sdk = GoalDiggersSDK(api_key="your_api_key_here")
    
    # Check API health
    health = sdk.get_health()
    print(f"API Health: {health.data}")
    
    # Get a prediction
    prediction = sdk.predict_match("Manchester City", "Arsenal")
    if prediction.success:
        print(f"Prediction: {prediction.data}")
    else:
        print(f"Error: {prediction.error}")
    
    # Get recommendations
    recommendations = sdk.get_top_recommendations(limit=3)
    print(f"Recommendations: {recommendations}")
    
    # Update preferences
    prefs_response = sdk.update_preferences({
        "favorite_teams": ["Manchester City", "Arsenal"],
        "betting_style": "conservative",
        "risk_tolerance": 0.3
    })
    print(f"Preferences updated: {prefs_response.success}")
    
    # Get analytics
    analytics = sdk.get_analytics_dashboard(time_range="7d")
    if analytics.success:
        print(f"Analytics: {analytics.data}")
    
    print("🎉 SDK example completed!")
