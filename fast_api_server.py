#!/usr/bin/env python3
"""
Fast API Server for GoalDiggers Platform
Optimized for quick startup and production readiness
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

try:  # Optional aggregate monitoring
    from monitoring.aggregate_status import collect_aggregate_status  # type: ignore
except Exception:  # pragma: no cover
    collect_aggregate_status = None  # type: ignore

# Centralized logging configuration (skip during pytest to allow caplog control)
try:
    if 'PYTEST_CURRENT_TEST' not in os.environ:
        from utils.logging_config import configure_logging
        configure_logging(level=os.getenv('LOG_LEVEL', 'INFO'))
except Exception:
    # Fall back silently; logging will use default root configuration
    pass
logger = logging.getLogger(__name__)

# Global variables for lazy loading
db_manager = None
predictor = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    timestamp: float = Field(..., description="Current timestamp")
    version: str = Field(default="1.0.0", description="API version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    database: Optional[Dict[str, Any]] = Field(None, description="Database status")
    predictor: Optional[Dict[str, str]] = Field(None, description="Predictor status")

class PredictionRequest(BaseModel):
    """Match prediction request model"""
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    match_data: Optional[Dict[str, Any]] = Field(None, description="Additional match data")

class PredictionResponse(BaseModel):
    """Match prediction response model"""
    home_team: str
    away_team: str
    predictions: Dict[str, float]
    confidence: float
    match_insight: Optional[Dict[str, Any]] = None

def get_database():
    """Get database manager with lazy loading"""
    global db_manager
    if db_manager is None:
        try:
            from database.db_manager import DatabaseManager
            db_manager = DatabaseManager()
            logger.info("✅ Database manager initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Database components not available: {e}")
            db_manager = None
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            db_manager = None
    return db_manager

def get_predictor():
    """Get predictor with lazy loading"""
    global predictor
    if predictor is None:
        try:
            from models.enhanced_real_data_predictor import EnhancedRealDataPredictor
            predictor = EnhancedRealDataPredictor()
            logger.info("✅ EnhancedRealDataPredictor initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Predictor components not available: {e}")
            predictor = None
        except Exception as e:
            logger.error(f"❌ Predictor initialization failed: {e}")
            predictor = None
    return predictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    start_time = time.time()
    logger.info("Fast API server starting up...")
    
    # Pre-warm critical components
    try:
        get_database()
        logger.info("Database pre-warmed")
    except Exception as e:
        logger.warning(f"Database pre-warming failed: {e}")
    
    startup_time = time.time() - start_time
    logger.info(f"Fast API server startup complete in {startup_time:.2f}s")
    
    yield
    
    logger.info("Fast API server shutting down...")

# Create FastAPI app with optimized settings
app = FastAPI(
    title="GoalDiggers Fast API",
    description="High-performance API for football match predictions",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Configure CORS for development
if os.getenv("ENVIRONMENT") != "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Store startup time
startup_timestamp = time.time()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    current_time = time.time()
    
    # Check database
    db_status = {"status": "OK", "latency_ms": None}
    db = get_database()
    if db:
        try:
            start = time.time()
            with db.get_session() as session:
                session.execute(text("SELECT 1")).fetchone()
            db_status["latency_ms"] = round((time.time() - start) * 1000, 2)
        except Exception as e:
            db_status = {"status": "ERROR", "error": str(e)}
    else:
        db_status = {"status": "UNAVAILABLE", "error": "Database not initialized"}
    
    # Check predictor
    predictor_status = {"status": "OK"}
    predictor_obj = get_predictor()
    if predictor_obj is None:
        predictor_status = {"status": "UNAVAILABLE"}
    
    return HealthResponse(
        status="OK",
        timestamp=current_time,
        uptime=current_time - startup_timestamp,
        database=db_status,
        predictor=predictor_status
    )

@app.get("/simple-health")
async def simple_health_check():
    """Simple health check for load balancers"""
    return {"status": "OK", "timestamp": time.time()}

@app.get("/api/v1/health", response_model=HealthResponse)
async def api_health_check():
    """API versioned health check"""
    return await health_check()

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_match(request: PredictionRequest):
    """Predict match outcome"""
    predictor_obj = get_predictor()
    if not predictor_obj:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service unavailable"
        )
    
    try:
        # Generate prediction
        if request.match_data:
            # Use enhanced prediction
            prediction = predictor_obj.predict_match_enhanced(
                request.home_team,
                request.away_team,
                request.match_data
            )
            
            return PredictionResponse(
                home_team=request.home_team,
                away_team=request.away_team,
                predictions={
                    "home_win": prediction.home_win_probability,
                    "draw": prediction.draw_probability,
                    "away_win": prediction.away_win_probability
                },
                confidence=prediction.confidence,
                match_insight=prediction.__dict__ if hasattr(prediction, '__dict__') else None
            )
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/api/v1/status")
async def get_status():
    """Get detailed service status"""
    return {
        "service": "GoalDiggers Fast API",
        "version": "1.0.0",
        "status": "operational",
        "uptime": time.time() - startup_timestamp,
        "timestamp": time.time(),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "components": {
            "database": "available" if get_database() else "unavailable",
            "predictor": "available" if get_predictor() else "unavailable"
        }
    }

@app.get("/health/aggregate")
async def aggregate_health():  # Lightweight aggregated status
    predictor_obj = get_predictor()
    if collect_aggregate_status:
        try:
            data = collect_aggregate_status(predictor=predictor_obj, include_database=True)
            return data
        except Exception as e:  # pragma: no cover
            return {"status": "ERROR", "error": str(e)}
    return {"status": "UNAVAILABLE", "reason": "aggregate module not present"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "GoalDiggers Fast API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 5000))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting Fast API server on {host}:{port}")
    
    uvicorn.run(
        "fast_api_server:app",
        host=host,
        port=port,
        log_level="info",
        reload=False,  # Disable reload for faster startup
        workers=1  # Single worker for development
    )
