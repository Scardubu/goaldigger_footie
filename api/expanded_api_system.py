#!/usr/bin/env python3
"""
Expanded API System
Phase 3B: Advanced Features Implementation - API Ecosystem Expansion

This module expands the existing API endpoints with authentication, rate limiting,
and new endpoints for Phase 3B features including personalization, analytics,
and mobile/PWA capabilities. Builds upon the existing Cross-League API foundation.

Key Features:
- Authentication and authorization system
- Rate limiting and quota management
- Advanced analytics API endpoints
- Personalization API endpoints
- Mobile/PWA specific endpoints
- Third-party integration capabilities
- Comprehensive API SDK and documentation
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import jwt
import redis
from datetime import datetime, timedelta
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_VERSION = "v4.1"
API_TITLE = "GoalDiggers Enhanced API"
API_DESCRIPTION = "Phase 3B Enhanced API with Advanced Features"

class APITier(Enum):
    """API access tiers."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class RateLimitType(Enum):
    """Rate limit types."""
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"

@dataclass
class APIUser:
    """API user model."""
    user_id: str
    api_key: str
    tier: APITier
    rate_limits: Dict[str, int]
    features_enabled: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    tier: APITier
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int

# Rate limit configurations by tier
RATE_LIMITS = {
    APITier.FREE: RateLimitConfig(
        tier=APITier.FREE,
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=1000,
        burst_limit=20
    ),
    APITier.PRO: RateLimitConfig(
        tier=APITier.PRO,
        requests_per_minute=100,
        requests_per_hour=1000,
        requests_per_day=10000,
        burst_limit=200
    ),
    APITier.ENTERPRISE: RateLimitConfig(
        tier=APITier.ENTERPRISE,
        requests_per_minute=1000,
        requests_per_hour=10000,
        requests_per_day=100000,
        burst_limit=2000
    )
}

class ExpandedAPISystem:
    """
    Expanded API system with authentication, rate limiting,
    and Phase 3B feature endpoints.
    """
    
    def __init__(self):
        """Initialize expanded API system."""
        self.app = FastAPI(
            title=API_TITLE,
            description=API_DESCRIPTION,
            version=API_VERSION,
            docs_url=f"/{API_VERSION}/docs",
            redoc_url=f"/{API_VERSION}/redoc"
        )
        
        # Initialize Redis for rate limiting and caching
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_available = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
            self.redis_available = False
        
        # Security
        self.security = HTTPBearer()
        self.jwt_secret = "your-secret-key"  # Should be from environment
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("🚀 Expanded API system initialized")
    
    def _setup_middleware(self):
        """Setup API middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["api.goaldiggers.com", "localhost", "127.0.0.1"]
        )
        
        # Request logging middleware
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
    
    def _setup_routes(self):
        """Setup API routes."""
        # Authentication endpoints
        self._setup_auth_routes()
        
        # Enhanced prediction endpoints
        self._setup_prediction_routes()
        
        # Analytics endpoints
        self._setup_analytics_routes()
        
        # Personalization endpoints
        self._setup_personalization_routes()
        
        # Mobile/PWA endpoints
        self._setup_mobile_routes()
        
        # Admin endpoints
        self._setup_admin_routes()
    
    def _setup_auth_routes(self):
        """Setup authentication routes."""
        
        @self.app.post(f"/{API_VERSION}/auth/register")
        async def register_user(
            email: str,
            tier: APITier = APITier.FREE,
            features: List[str] = None
        ):
            """Register new API user."""
            try:
                # Generate API key
                api_key = self._generate_api_key()
                user_id = str(uuid.uuid4())
                
                # Create user
                user = APIUser(
                    user_id=user_id,
                    api_key=api_key,
                    tier=tier,
                    rate_limits=asdict(RATE_LIMITS[tier]),
                    features_enabled=features or [],
                    created_at=datetime.now()
                )
                
                # Store user (in production, use proper database)
                if self.redis_available:
                    self.redis_client.hset(f"user:{user_id}", mapping=asdict(user))
                
                return {
                    "user_id": user_id,
                    "api_key": api_key,
                    "tier": tier.value,
                    "rate_limits": user.rate_limits,
                    "message": "User registered successfully"
                }
                
            except Exception as e:
                logger.error(f"User registration failed: {e}")
                raise HTTPException(status_code=500, detail="Registration failed")
        
        @self.app.post(f"/{API_VERSION}/auth/token")
        async def get_access_token(api_key: str):
            """Get JWT access token."""
            try:
                user = await self._get_user_by_api_key(api_key)
                if not user:
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                # Generate JWT token
                from datetime import timezone
                payload = {
                    "user_id": user.user_id,
                    "tier": user.tier.value,
                    "exp": datetime.now(timezone.utc) + timedelta(hours=24)
                }
                
                token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
                
                return {
                    "access_token": token,
                    "token_type": "bearer",
                    "expires_in": 86400,
                    "tier": user.tier.value
                }
                
            except Exception as e:
                logger.error(f"Token generation failed: {e}")
                raise HTTPException(status_code=401, detail="Authentication failed")
    
    def _setup_prediction_routes(self):
        """Setup enhanced prediction routes."""
        
        @self.app.post(f"/{API_VERSION}/predictions/enhanced")
        async def enhanced_prediction(
            request: Dict[str, Any],
            user: APIUser = Depends(self._get_current_user)
        ):
            """Enhanced prediction with Phase 3B features."""
            try:
                # Rate limiting
                await self._check_rate_limit(user, "enhanced_prediction")
                
                # Enhanced prediction logic
                prediction_result = await self._generate_enhanced_prediction(request, user)
                
                # Track usage
                await self._track_api_usage(user, "enhanced_prediction")
                
                return prediction_result
                
            except Exception as e:
                logger.error(f"Enhanced prediction failed: {e}")
                raise HTTPException(status_code=500, detail="Prediction failed")
        
        @self.app.post(f"/{API_VERSION}/predictions/batch")
        async def batch_predictions(
            requests: List[Dict[str, Any]],
            user: APIUser = Depends(self._get_current_user)
        ):
            """Batch prediction processing."""
            try:
                # Check tier permissions
                if user.tier == APITier.FREE and len(requests) > 10:
                    raise HTTPException(
                        status_code=403, 
                        detail="Batch size limit exceeded for free tier"
                    )
                
                # Rate limiting
                await self._check_rate_limit(user, "batch_prediction", len(requests))
                
                # Process batch
                results = []
                for req in requests:
                    result = await self._generate_enhanced_prediction(req, user)
                    results.append(result)
                
                # Track usage
                await self._track_api_usage(user, "batch_prediction", len(requests))
                
                return {
                    "batch_id": str(uuid.uuid4()),
                    "total_predictions": len(results),
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Batch prediction failed: {e}")
                raise HTTPException(status_code=500, detail="Batch prediction failed")
    
    def _setup_analytics_routes(self):
        """Setup analytics API routes."""
        
        @self.app.get(f"/{API_VERSION}/analytics/dashboard")
        async def get_analytics_dashboard(
            user: APIUser = Depends(self._get_current_user),
            time_range: str = "24h"
        ):
            """Get analytics dashboard data."""
            try:
                # Check feature access
                if "analytics" not in user.features_enabled and user.tier == APITier.FREE:
                    raise HTTPException(
                        status_code=403, 
                        detail="Analytics feature not available in free tier"
                    )
                
                # Rate limiting
                await self._check_rate_limit(user, "analytics")
                
                # Generate analytics data
                analytics_data = await self._generate_analytics_data(time_range, user)
                
                return analytics_data
                
            except Exception as e:
                logger.error(f"Analytics dashboard failed: {e}")
                raise HTTPException(status_code=500, detail="Analytics unavailable")
        
        @self.app.get(f"/{API_VERSION}/analytics/performance")
        async def get_performance_metrics(
            user: APIUser = Depends(self._get_current_user)
        ):
            """Get system performance metrics."""
            try:
                # Enterprise feature
                if user.tier != APITier.ENTERPRISE:
                    raise HTTPException(
                        status_code=403, 
                        detail="Performance metrics available for enterprise tier only"
                    )
                
                performance_data = await self._get_performance_metrics()
                return performance_data
                
            except Exception as e:
                logger.error(f"Performance metrics failed: {e}")
                raise HTTPException(status_code=500, detail="Performance metrics unavailable")
    
    def _setup_personalization_routes(self):
        """Setup personalization API routes."""
        
        @self.app.post(f"/{API_VERSION}/personalization/preferences")
        async def update_preferences(
            preferences: Dict[str, Any],
            user: APIUser = Depends(self._get_current_user)
        ):
            """Update user preferences."""
            try:
                # Rate limiting
                await self._check_rate_limit(user, "personalization")
                
                # Update preferences
                result = await self._update_user_preferences(user.user_id, preferences)
                
                from datetime import timezone
                return {
                    "user_id": user.user_id,
                    "preferences_updated": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Preference update failed: {e}")
                raise HTTPException(status_code=500, detail="Preference update failed")
        
        @self.app.get(f"/{API_VERSION}/personalization/recommendations")
        async def get_recommendations(
            user: APIUser = Depends(self._get_current_user),
            context: str = "general",
            limit: int = 5
        ):
            """Get personalized recommendations."""
            try:
                # Rate limiting
                await self._check_rate_limit(user, "recommendations")
                
                # Generate recommendations
                recommendations = await self._generate_recommendations(user.user_id, context, limit)
                
                from datetime import timezone
                return {
                    "user_id": user.user_id,
                    "context": context,
                    "recommendations": recommendations,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Recommendations failed: {e}")
                raise HTTPException(status_code=500, detail="Recommendations unavailable")
    
    def _setup_mobile_routes(self):
        """Setup mobile/PWA specific routes."""
        
        @self.app.post(f"/{API_VERSION}/mobile/push-subscribe")
        async def subscribe_push_notifications(
            subscription: Dict[str, Any],
            user: APIUser = Depends(self._get_current_user)
        ):
            """Subscribe to push notifications."""
            try:
                # Store subscription
                subscription_id = await self._store_push_subscription(user.user_id, subscription)
                
                from datetime import timezone
                return {
                    "subscription_id": subscription_id,
                    "user_id": user.user_id,
                    "subscribed": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Push subscription failed: {e}")
                raise HTTPException(status_code=500, detail="Push subscription failed")
        
        @self.app.post(f"/{API_VERSION}/mobile/sync")
        async def sync_offline_data(
            sync_data: Dict[str, Any],
            user: APIUser = Depends(self._get_current_user)
        ):
            """Sync offline data when connection restored."""
            try:
                # Process offline sync
                sync_result = await self._process_offline_sync(user.user_id, sync_data)
                
                from datetime import timezone
                return {
                    "user_id": user.user_id,
                    "sync_successful": True,
                    "items_synced": sync_result.get("items_synced", 0),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"Offline sync failed: {e}")
                raise HTTPException(status_code=500, detail="Offline sync failed")
    
    def _setup_admin_routes(self):
        """Setup admin API routes."""
        
        @self.app.get(f"/{API_VERSION}/admin/users")
        async def list_users(
            user: APIUser = Depends(self._get_admin_user),
            limit: int = 100,
            offset: int = 0
        ):
            """List API users (admin only)."""
            try:
                users = await self._list_users(limit, offset)
                return {
                    "users": users,
                    "total": len(users),
                    "limit": limit,
                    "offset": offset
                }
                
            except Exception as e:
                logger.error(f"User listing failed: {e}")
                raise HTTPException(status_code=500, detail="User listing failed")
        
        @self.app.get(f"/{API_VERSION}/admin/usage")
        async def get_usage_statistics(
            user: APIUser = Depends(self._get_admin_user),
            time_range: str = "24h"
        ):
            """Get API usage statistics (admin only)."""
            try:
                usage_stats = await self._get_usage_statistics(time_range)
                return usage_stats
                
            except Exception as e:
                logger.error(f"Usage statistics failed: {e}")
                raise HTTPException(status_code=500, detail="Usage statistics unavailable")
    
    async def _get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Get current authenticated user."""
        try:
            # Decode JWT token
            payload = jwt.decode(credentials.credentials, self.jwt_secret, algorithms=["HS256"])
            user_id = payload.get("user_id")
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Get user from storage
            user = await self._get_user_by_id(user_id)
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def _get_admin_user(self, user: APIUser = Depends(lambda: self._get_current_user)):
        """Get admin user (enterprise tier required)."""
        if user.tier != APITier.ENTERPRISE or "admin" not in user.features_enabled:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user
    
    async def _check_rate_limit(self, user: APIUser, endpoint: str, count: int = 1):
        """Check rate limiting for user."""
        if not self.redis_available:
            return  # Skip rate limiting if Redis unavailable
        
        try:
            rate_limit = RATE_LIMITS[user.tier]
            current_time = int(time.time())
            
            # Check per-minute limit
            minute_key = f"rate_limit:{user.user_id}:minute:{current_time // 60}"
            minute_count = int(self.redis_client.get(minute_key) or 0)
            
            if minute_count + count > rate_limit.requests_per_minute:
                raise HTTPException(
                    status_code=429, 
                    detail=f"Rate limit exceeded: {rate_limit.requests_per_minute} requests per minute"
                )
            
            # Update counters
            pipe = self.redis_client.pipeline()
            pipe.incr(minute_key, count)
            pipe.expire(minute_key, 60)
            pipe.execute()
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Don't block requests if rate limiting fails
    
    async def _track_api_usage(self, user: APIUser, endpoint: str, count: int = 1):
        """Track API usage for analytics."""
        if not self.redis_available:
            return
        
        try:
            current_time = int(time.time())
            day_key = f"usage:{user.user_id}:day:{current_time // 86400}"
            
            pipe = self.redis_client.pipeline()
            pipe.incr(day_key, count)
            pipe.expire(day_key, 86400 * 7)  # Keep for 7 days
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Usage tracking error: {e}")
    
    def _generate_api_key(self) -> str:
        """Generate secure API key."""
        return f"gd_{hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()[:32]}"
    
    async def _get_user_by_api_key(self, api_key: str) -> Optional[APIUser]:
        """Get user by API key."""
        # In production, implement proper database lookup
        # This is a simplified implementation
        return None
    
    async def _get_user_by_id(self, user_id: str) -> Optional[APIUser]:
        """Get user by ID."""
        # In production, implement proper database lookup
        # This is a simplified implementation
        return None
    
    async def _generate_enhanced_prediction(self, request: Dict[str, Any], user: APIUser) -> Dict[str, Any]:
        """Generate enhanced prediction with Phase 3B features."""
        # Mock implementation - integrate with actual prediction engine
            from datetime import timezone
            return {
                "prediction_id": str(uuid.uuid4()),
                "home_team": request.get("home_team"),
                "away_team": request.get("away_team"),
                "predictions": {
                    "home_win": 0.52,
                    "draw": 0.28,
                    "away_win": 0.20
                },
                "confidence": 0.87,
                "phase3b_features": {
                    "personalization_applied": "personalization" in user.features_enabled,
                    "analytics_enhanced": "analytics" in user.features_enabled,
                    "mobile_optimized": True
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _generate_analytics_data(self, time_range: str, user: APIUser) -> Dict[str, Any]:
        """Generate analytics dashboard data."""
        # Mock implementation
        from datetime import timezone
        return {
            "time_range": time_range,
            "total_predictions": 1247,
            "accuracy": 0.925,
            "user_engagement": 0.78,
            "performance_metrics": {
                "avg_response_time": 0.027,
                "uptime": 0.999,
                "error_rate": 0.001
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        # Mock implementation
        from datetime import timezone
        return {
            "system_health": 0.98,
            "memory_usage": 387,
            "cpu_usage": 0.23,
            "active_users": 1247,
            "api_calls_per_minute": 342,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def _update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Update user preferences."""
        # Mock implementation
        return {"updated": True}
    
    async def _generate_recommendations(self, user_id: str, context: str, limit: int) -> List[Dict[str, Any]]:
        """Generate personalized recommendations."""
        # Mock implementation
        return [
            {
                "id": str(uuid.uuid4()),
                "title": "Manchester City vs Arsenal",
                "confidence": 0.89,
                "type": "match_prediction"
            }
        ]
    
    async def _store_push_subscription(self, user_id: str, subscription: Dict[str, Any]) -> str:
        """Store push notification subscription."""
        subscription_id = str(uuid.uuid4())
        # Store in database
        return subscription_id
    
    async def _process_offline_sync(self, user_id: str, sync_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process offline data synchronization."""
        # Mock implementation
        return {"items_synced": len(sync_data.get("items", []))}
    
    async def _list_users(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        """List API users."""
        # Mock implementation
        return []
    
    async def _get_usage_statistics(self, time_range: str) -> Dict[str, Any]:
        """Get API usage statistics."""
        # Mock implementation
        from datetime import timezone
        return {
            "time_range": time_range,
            "total_requests": 15247,
            "unique_users": 342,
            "top_endpoints": [
                {"endpoint": "/predictions/enhanced", "count": 8934},
                {"endpoint": "/analytics/dashboard", "count": 3421},
                {"endpoint": "/personalization/recommendations", "count": 2892}
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Factory function to create API app
def create_api_app() -> FastAPI:
    """Create and configure the expanded API application."""
    api_system = ExpandedAPISystem()
    return api_system.app

# Health check endpoint
@app.get(f"/{API_VERSION}/health")
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "features": {
            "authentication": True,
            "rate_limiting": True,
            "analytics": True,
            "personalization": True,
            "mobile_pwa": True
        }
    }

# Create the FastAPI app
app = create_api_app()
