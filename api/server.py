#!/usr/bin/env python3
"""
GoalDiggers API Server

This server provides REST API endpoints for the GoalDiggers platform, including:
- Match prediction and analysis
- Historical data access
- Live data streaming
- User preferences and personalization

The server uses FastAPI for high performance and automatic documentation.
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import needed modules
try:
    from database.db_manager import DatabaseManager
    from models import MLIntegration
    from utils.config import Config
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("goaldiggers_api")

# Initialize FastAPI app
app = FastAPI(
    title="GoalDiggers API",
    description="API for football match prediction and analysis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database manager
try:
    # Use the database URI from config or environment
    db_uri = os.environ.get("DATABASE_URI")
    if not db_uri:
        config = Config()
        db_uri = config.get("database.uri")
    
    if not db_uri:
        # Fallback to default SQLite database
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(data_dir, exist_ok=True)
        db_uri = f"sqlite:///{os.path.join(data_dir, 'football.db')}"
        logger.warning(f"No database URI found, using default SQLite database: {db_uri}")
    
    db_manager = DatabaseManager(db_uri)
    # Ensure tables exist
    db_manager.create_tables()
except Exception as e:
    logger.error(f"Failed to initialize database connection: {e}")
    db_manager = None

# Initialize ML components
feature_generator = None
predictor = None
ml_integration = None
try:
    ml_integration = MLIntegration()
    # Try to create feature generator and predictor using main models
    try:
        feature_generator = ml_integration.create_feature_generator(db_manager)
    except Exception as fe:
        logger.warning(f"Main feature generator unavailable: {fe}")
        # Fallback to local API feature generator
        try:
            from api.feature_generator import \
                FeatureGenerator as LocalFeatureGenerator
            feature_generator = LocalFeatureGenerator(db_manager)
            logger.info("Using fallback API feature generator.")
        except Exception as lfe:
            logger.error(f"Fallback feature generator failed: {lfe}")
            feature_generator = None
    try:
        predictor = ml_integration.create_predictor(model_type="ensemble")
    except Exception as pe:
        logger.error(f"Predictor initialization failed: {pe}")
        predictor = None
    if feature_generator and predictor:
        logger.info("ML components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ML components: {e}")
    feature_generator = None
    predictor = None


# --- API Models ---

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current server timestamp")
    database_connected: bool = Field(..., description="Database connection status")
    ml_components_loaded: bool = Field(..., description="ML components loaded status")
    uptime_seconds: float = Field(..., description="API uptime in seconds")


class MatchPredictionRequest(BaseModel):
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    league: Optional[str] = Field(None, description="League name")
    match_date: Optional[str] = Field(None, description="Match date (ISO format)")
    context_toggles: Optional[Dict[str, bool]] = Field(None, description="Feature context toggles")


class MatchPredictionResponse(BaseModel):
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    home_win_probability: float = Field(..., description="Probability of home team winning")
    draw_probability: float = Field(..., description="Probability of a draw")
    away_win_probability: float = Field(..., description="Probability of away team winning")
    predicted_home_goals: Optional[float] = Field(None, description="Predicted goals for home team")
    predicted_away_goals: Optional[float] = Field(None, description="Predicted goals for away team")
    confidence: float = Field(..., description="Model confidence score")
    model_version: str = Field(..., description="Version of the model used")
    prediction_id: str = Field(..., description="Unique prediction identifier")


# --- Global variables ---
start_time = time.time()


# --- API Routes ---

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "GoalDiggers API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    uptime = time.time() - start_time
    # Check database connection
    db_connected = False
    db_error = None
    if db_manager:
        try:
            with db_manager.session_scope() as session:
                from sqlalchemy import text
                session.execute(text("SELECT 1"))
            db_connected = True
        except Exception as e:
            db_error = str(e)
            logger.error(f"Database health check failed: {e}")
    # Check ML components
    ml_loaded = feature_generator is not None and predictor is not None
    ml_error = None
    if not ml_loaded:
        ml_error = "Feature generator or predictor not loaded."
    return {
        "status": "healthy" if db_connected and ml_loaded else "degraded",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "database_connected": db_connected,
        "ml_components_loaded": ml_loaded,
        "uptime_seconds": uptime,
        "db_error": db_error,
        "ml_error": ml_error
    }


@app.post("/predict", response_model=MatchPredictionResponse)
async def predict(request: MatchPredictionRequest):
    """Predict match outcome."""
    if not predictor or not feature_generator:
        raise HTTPException(status_code=503, detail="ML components not initialized")
    
    try:
        # Generate features for the match
        match_info = {
            "home_team": request.home_team,
            "away_team": request.away_team,
            "match_date": datetime.now() if not request.match_date else datetime.fromisoformat(request.match_date),
            "league": request.league
        }
        
        # Generate features
        features = feature_generator.generate_features_for_match(match_info, context_toggles=request.context_toggles)
        if not features:
            raise HTTPException(status_code=422, detail="Failed to generate features")
        
        # Make prediction
        prediction = predictor.predict(features)
        if not prediction:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Format response
        prediction_id = f"pred-{int(time.time())}-{hash(request.home_team + request.away_team) % 1000}"
        
        return {
            "home_team": request.home_team,
            "away_team": request.away_team,
            "home_win_probability": prediction[0] if isinstance(prediction, list) else prediction.get("home_win", 0.33),
            "draw_probability": prediction[1] if isinstance(prediction, list) else prediction.get("draw", 0.34),
            "away_win_probability": prediction[2] if isinstance(prediction, list) else prediction.get("away_win", 0.33),
            "predicted_home_goals": None,  # Could be added if model supports it
            "predicted_away_goals": None,  # Could be added if model supports it
            "confidence": predictor.get_confidence() if hasattr(predictor, "get_confidence") else 0.75,
            "model_version": predictor.get_version() if hasattr(predictor, "get_version") else "v1.0",
            "prediction_id": prediction_id,
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/leagues", response_class=JSONResponse)
async def get_leagues():
    """Get list of available leagues."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        with db_manager.session_scope() as session:
            leagues = db_manager.get_all_leagues(session)
            return [{"id": league.id, "name": league.name, "country": league.country} for league in leagues]
    except Exception as e:
        logger.error(f"Error getting leagues: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting leagues: {str(e)}")


@app.get("/teams", response_class=JSONResponse)
async def get_teams(league_id: Optional[int] = None):
    """Get list of teams, optionally filtered by league."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        with db_manager.session_scope() as session:
            if league_id:
                teams = db_manager.get_teams_by_league(league_id, session)
            else:
                teams = db_manager.get_all_teams(session)
            
            return [{"id": team.id, "name": team.name, "league_id": team.league_id} for team in teams]
    except Exception as e:
        logger.error(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting teams: {str(e)}")


@app.get("/matches", response_class=JSONResponse)
async def get_matches(
    league_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get list of matches with optional filters."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        # Parse dates if provided
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format (YYYY-MM-DD)")
        
        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO format (YYYY-MM-DD)")
        
        with db_manager.session_scope() as session:
            matches = db_manager.get_matches(
                session=session,
                league_id=league_id,
                start_date=start_datetime,
                end_date=end_datetime,
                limit=limit,
                offset=offset
            )
            
            result = []
            for match in matches:
                # Get team names
                home_team = db_manager.get_team_by_id(match.home_team_id, session) if match.home_team_id else None
                away_team = db_manager.get_team_by_id(match.away_team_id, session) if match.away_team_id else None
                
                match_data = {
                    "id": match.id,
                    "league_id": match.league_id,
                    "home_team_id": match.home_team_id,
                    "away_team_id": match.away_team_id,
                    "home_team": home_team.name if home_team else "Unknown",
                    "away_team": away_team.name if away_team else "Unknown",
                    "match_date": match.match_date.isoformat() if match.match_date else None,
                    "status": match.status,
                    "home_score": match.home_score,
                    "away_score": match.away_score,
                }
                
                result.append(match_data)
            
            return result
    except Exception as e:
        logger.error(f"Error getting matches: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting matches: {str(e)}")


@app.get("/match/{match_id}", response_class=JSONResponse)
async def get_match(match_id: str):
    """Get details for a specific match."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        with db_manager.session_scope() as session:
            match = db_manager.get_match_by_id(match_id, session)
            
            if not match:
                raise HTTPException(status_code=404, detail=f"Match with ID {match_id} not found")
            
            # Get team names
            home_team = db_manager.get_team_by_id(match.home_team_id, session) if match.home_team_id else None
            away_team = db_manager.get_team_by_id(match.away_team_id, session) if match.away_team_id else None
            
            # Get match stats if available
            stats = None
            if hasattr(match, 'match_stats') and match.match_stats:
                stats = {
                    "possession": {
                        "home": getattr(match.match_stats, 'home_possession', 0),
                        "away": getattr(match.match_stats, 'away_possession', 0)
                    },
                    "shots": {
                        "home": getattr(match.match_stats, 'home_shots', 0),
                        "away": getattr(match.match_stats, 'away_shots', 0)
                    },
                    "shots_on_target": {
                        "home": getattr(match.match_stats, 'home_shots_on_target', 0),
                        "away": getattr(match.match_stats, 'away_shots_on_target', 0)
                    }
                }
            
            # Get latest prediction
            prediction = None
            latest_pred = db_manager.get_latest_prediction_for_match(match_id, session)
            if latest_pred:
                prediction = {
                    "home_win_prob": latest_pred.home_win_prob,
                    "draw_prob": latest_pred.draw_prob,
                    "away_win_prob": latest_pred.away_win_prob,
                    "expected_goals_home": latest_pred.expected_goals_home,
                    "expected_goals_away": latest_pred.expected_goals_away,
                    "model_version": latest_pred.model_version
                }
            
            # Get latest odds
            odds = None
            latest_odds = db_manager.get_latest_odds_for_match(match_id, session)
            if latest_odds:
                odds = {
                    "bookmaker": latest_odds.bookmaker,
                    "home_win": latest_odds.home_win,
                    "draw": latest_odds.draw,
                    "away_win": latest_odds.away_win
                }
            
            # Build response
            match_data = {
                "id": match.id,
                "league_id": match.league_id,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "home_team": home_team.name if home_team else "Unknown",
                "away_team": away_team.name if away_team else "Unknown",
                "match_date": match.match_date.isoformat() if match.match_date else None,
                "status": match.status,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "stats": stats,
                "prediction": prediction,
                "odds": odds
            }
            
            return match_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting match details: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting match details: {str(e)}")


# --- Main entrypoint ---

def start():
    """Start the API server."""
    port = int(os.environ.get("API_PORT", 5000))
    host = os.environ.get("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting GoalDiggers API server on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
