"""
Outcome Tracking API - FastAPI service for receiving match results

Provides REST endpoints to update predictions with actual match outcomes.
Runs alongside the main Streamlit dashboard.

Usage:
    uvicorn utils.outcome_api:app --port 8502 --reload
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.outcome_tracker import get_outcome_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GoalDiggers Outcome Tracking API",
    description="API for updating predictions with actual match results",
    version="1.0.0"
)

# Request models
class MatchOutcome(BaseModel):
    """Single match outcome."""
    home_team: str
    away_team: str
    match_date: datetime
    home_score: int
    away_score: int
    tolerance_hours: Optional[int] = 24

class BatchMatchOutcomes(BaseModel):
    """Batch of match outcomes."""
    matches: List[MatchOutcome]

# Response models
class OutcomeResponse(BaseModel):
    """Response for outcome update."""
    success: bool
    message: str
    prediction_id: Optional[str] = None
    predicted_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    is_correct: Optional[bool] = None

class AccuracyResponse(BaseModel):
    """Response for accuracy stats."""
    overall: dict
    by_league: dict
    by_confidence: dict
    filters: dict

# Endpoints

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "service": "GoalDiggers Outcome Tracking API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "update_outcome": "/api/outcome",
            "batch_update": "/api/outcomes/batch",
            "get_accuracy": "/api/accuracy",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        tracker = get_outcome_tracker()
        return {
            "status": "healthy",
            "service": "outcome_tracker",
            "database": str(tracker.db_path),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.post("/api/outcome", response_model=OutcomeResponse)
async def update_outcome(match: MatchOutcome):
    """
    Update a prediction with actual match outcome.
    
    Example:
        POST /api/outcome
        {
            "home_team": "Manchester United",
            "away_team": "Liverpool",
            "match_date": "2025-11-02T15:00:00",
            "home_score": 2,
            "away_score": 1
        }
    """
    try:
        logger.info(f"Updating outcome: {match.home_team} vs {match.away_team}")
        
        tracker = get_outcome_tracker()
        result = tracker.update_prediction_outcome(
            home_team=match.home_team,
            away_team=match.away_team,
            match_date=match.match_date,
            home_score=match.home_score,
            away_score=match.away_score,
            tolerance_hours=match.tolerance_hours
        )
        
        if result['success']:
            return OutcomeResponse(
                success=True,
                message=f"Updated prediction for {match.home_team} vs {match.away_team}",
                prediction_id=result.get('prediction_id'),
                predicted_outcome=result.get('predicted_outcome'),
                actual_outcome=result.get('actual_outcome'),
                is_correct=result.get('is_correct')
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=result.get('reason', 'Prediction not found')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/outcomes/batch")
async def batch_update_outcomes(batch: BatchMatchOutcomes):
    """
    Update multiple predictions in batch.
    
    Example:
        POST /api/outcomes/batch
        {
            "matches": [
                {
                    "home_team": "Arsenal",
                    "away_team": "Chelsea",
                    "match_date": "2025-11-02T15:00:00",
                    "home_score": 3,
                    "away_score": 1
                },
                ...
            ]
        }
    """
    try:
        logger.info(f"Batch updating {len(batch.matches)} outcomes")
        
        tracker = get_outcome_tracker()
        
        # Convert Pydantic models to dicts
        matches_data = [
            {
                'home_team': m.home_team,
                'away_team': m.away_team,
                'match_date': m.match_date,
                'home_score': m.home_score,
                'away_score': m.away_score,
                'tolerance_hours': m.tolerance_hours
            }
            for m in batch.matches
        ]
        
        result = tracker.batch_update_outcomes(matches_data)
        
        return {
            "success": True,
            "message": f"Processed {result['total']} matches",
            "successful": result['successful'],
            "failed": result['failed'],
            "details": result['details']
        }
        
    except Exception as e:
        logger.error(f"Error in batch update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/accuracy", response_model=AccuracyResponse)
async def get_accuracy_stats(
    days: Optional[int] = None,
    league: Optional[str] = None,
    min_confidence: Optional[str] = None
):
    """
    Get accuracy statistics.
    
    Query Parameters:
        days: Look back period in days (optional)
        league: Filter by league name (optional)
        min_confidence: Minimum confidence level - high/medium/low (optional)
    
    Example:
        GET /api/accuracy?days=30&min_confidence=high
    """
    try:
        logger.info(f"Getting accuracy stats (days={days}, league={league}, min_confidence={min_confidence})")
        
        tracker = get_outcome_tracker()
        stats = tracker.get_accuracy_stats(
            days=days,
            league=league,
            min_confidence=min_confidence
        )
        
        return AccuracyResponse(
            overall=stats['overall'],
            by_league=stats['by_league'],
            by_confidence=stats['by_confidence'],
            filters=stats['filters']
        )
        
    except Exception as e:
        logger.error(f"Error getting accuracy stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/pending")
async def get_pending_predictions():
    """Get predictions that don't have outcomes yet."""
    try:
        import sqlite3
        from pathlib import Path
        
        tracker = get_outcome_tracker()
        conn = sqlite3.connect(str(tracker.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT prediction_id, timestamp, home_team, away_team, 
                   league, predicted_outcome, confidence_level
            FROM predictions
            WHERE actual_outcome IS NULL
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = [
            {
                'prediction_id': row[0],
                'timestamp': row[1],
                'home_team': row[2],
                'away_team': row[3],
                'league': row[4],
                'predicted_outcome': row[5],
                'confidence_level': row[6]
            }
            for row in rows
        ]
        
        return {
            "success": True,
            "count": len(predictions),
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error getting pending predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 80)
    print("GoalDiggers Outcome Tracking API")
    print("=" * 80)
    print("\nStarting server on http://localhost:8502")
    print("\nEndpoints:")
    print("  - POST /api/outcome - Update single match outcome")
    print("  - POST /api/outcomes/batch - Update multiple outcomes")
    print("  - GET /api/accuracy - Get accuracy statistics")
    print("  - GET /api/predictions/pending - Get pending predictions")
    print("  - GET /health - Health check")
    print("\nAPI Docs: http://localhost:8502/docs")
    print("=" * 80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8502, log_level="info")
