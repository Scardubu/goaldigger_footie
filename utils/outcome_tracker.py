"""
Outcome Tracking System for GoalDiggers Platform

Receives match results and updates prediction history with actual outcomes.
Calculates real accuracy metrics for continuous model improvement.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class OutcomeTracker:
    """Tracks actual match outcomes and updates prediction accuracy."""
    
    def __init__(self, db_path: str = "data/prediction_history.db"):
        """Initialize outcome tracker.
        
        Args:
            db_path: Path to prediction history database
        """
        self.db_path = Path(db_path)
        logger.info(f"Outcome tracker initialized: {self.db_path}")
    
    def update_prediction_outcome(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        home_score: int,
        away_score: int,
        tolerance_hours: int = 24
    ) -> Dict[str, Any]:
        """Update prediction with actual outcome.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            match_date: Date/time of match
            home_score: Home team final score
            away_score: Away team final score
            tolerance_hours: Match time tolerance for finding prediction
            
        Returns:
            Dictionary with update status and accuracy info
        """
        try:
            # Determine actual outcome
            if home_score > away_score:
                actual_outcome = 'home_win'
            elif home_score < away_score:
                actual_outcome = 'away_win'
            else:
                actual_outcome = 'draw'
            
            # Find matching prediction
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Query for matching prediction within tolerance
            cursor.execute("""
                SELECT prediction_id, predicted_outcome, home_team, away_team, 
                       predicted_probability, confidence_level, timestamp
                FROM predictions
                WHERE home_team = ? AND away_team = ?
                  AND actual_outcome IS NULL
                  AND datetime(timestamp) BETWEEN 
                      datetime(?, '-' || ? || ' hours') AND 
                      datetime(?, '+' || ? || ' hours')
                ORDER BY ABS(julianday(timestamp) - julianday(?))
                LIMIT 1
            """, (
                home_team, away_team,
                match_date.isoformat(), tolerance_hours,
                match_date.isoformat(), tolerance_hours,
                match_date.isoformat()
            ))
            
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                logger.warning(f"No matching prediction found for {home_team} vs {away_team} "
                             f"on {match_date.date()}")
                return {
                    'success': False,
                    'reason': 'No matching prediction found',
                    'home_team': home_team,
                    'away_team': away_team,
                    'match_date': match_date.isoformat()
                }
            
            prediction_id, predicted_outcome, *_ = result
            is_correct = (predicted_outcome == actual_outcome)
            
            # Update prediction with actual outcome
            cursor.execute("""
                UPDATE predictions
                SET actual_outcome = ?,
                    is_correct = ?,
                    home_score = ?,
                    away_score = ?,
                    outcome_updated_at = CURRENT_TIMESTAMP
                WHERE prediction_id = ?
            """, (actual_outcome, is_correct, home_score, away_score, prediction_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated prediction {prediction_id}: "
                       f"{home_team} {home_score}-{away_score} {away_team} "
                       f"(predicted: {predicted_outcome}, actual: {actual_outcome}, "
                       f"correct: {is_correct})")
            
            return {
                'success': True,
                'prediction_id': prediction_id,
                'predicted_outcome': predicted_outcome,
                'actual_outcome': actual_outcome,
                'is_correct': is_correct,
                'home_score': home_score,
                'away_score': away_score,
                'home_team': home_team,
                'away_team': away_team
            }
            
        except Exception as e:
            logger.error(f"Error updating outcome: {e}")
            return {
                'success': False,
                'reason': f'Error: {str(e)}',
                'home_team': home_team,
                'away_team': away_team
            }
    
    def get_accuracy_stats(
        self,
        days: Optional[int] = None,
        league: Optional[str] = None,
        min_confidence: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate accuracy statistics for predictions with outcomes.
        
        Args:
            days: Look back period in days (None = all time)
            league: Filter by league (None = all leagues)
            min_confidence: Minimum confidence level (high/medium/low)
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Build query with filters
            query = """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) as accuracy,
                    league,
                    confidence_level
                FROM predictions
                WHERE actual_outcome IS NOT NULL
            """
            
            params = []
            if days:
                query += " AND datetime(timestamp) >= datetime('now', '-' || ? || ' days')"
                params.append(days)
            if league:
                query += " AND league = ?"
                params.append(league)
            if min_confidence:
                confidence_order = {'high': 3, 'medium': 2, 'low': 1}
                min_value = confidence_order.get(min_confidence, 1)
                query += """ AND CASE confidence_level 
                                WHEN 'high' THEN 3 
                                WHEN 'medium' THEN 2 
                                ELSE 1 END >= ?"""
                params.append(min_value)
            
            query += " GROUP BY league, confidence_level"
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            # Calculate overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM predictions
                WHERE actual_outcome IS NOT NULL
            """ + (" AND datetime(timestamp) >= datetime('now', '-' || ? || ' days')" if days else ""),
                params[:1] if days else [])
            
            overall = cursor.fetchone()
            conn.close()
            
            total_predictions = overall[0] if overall else 0
            correct_predictions = overall[1] if overall else 0
            overall_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            
            # Group results by league and confidence
            by_league = {}
            by_confidence = {}
            
            for row in results:
                total, correct, accuracy, league_name, conf_level = row
                
                if league_name not in by_league:
                    by_league[league_name] = {'total': 0, 'correct': 0}
                by_league[league_name]['total'] += total
                by_league[league_name]['correct'] += correct
                
                if conf_level not in by_confidence:
                    by_confidence[conf_level] = {'total': 0, 'correct': 0}
                by_confidence[conf_level]['total'] += total
                by_confidence[conf_level]['correct'] += correct
            
            # Calculate percentages
            for league_name in by_league:
                stats = by_league[league_name]
                stats['accuracy'] = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            
            for conf_level in by_confidence:
                stats = by_confidence[conf_level]
                stats['accuracy'] = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            
            return {
                'overall': {
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions,
                    'accuracy_percentage': round(overall_accuracy, 2)
                },
                'by_league': by_league,
                'by_confidence': by_confidence,
                'filters': {
                    'days': days,
                    'league': league,
                    'min_confidence': min_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating accuracy stats: {e}")
            return {
                'error': str(e),
                'overall': {'total_predictions': 0, 'correct_predictions': 0, 'accuracy_percentage': 0}
            }
    
    def batch_update_outcomes(self, matches: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Update multiple match outcomes in batch.
        
        Args:
            matches: List of match result dictionaries with keys:
                    home_team, away_team, match_date, home_score, away_score
                    
        Returns:
            Summary of batch update results
        """
        results = {
            'total': len(matches),
            'successful': 0,
            'failed': 0,
            'details': []
        }
        
        for match in matches:
            result = self.update_prediction_outcome(
                home_team=match['home_team'],
                away_team=match['away_team'],
                match_date=match['match_date'],
                home_score=match['home_score'],
                away_score=match['away_score'],
                tolerance_hours=match.get('tolerance_hours', 24)
            )
            
            if result['success']:
                results['successful'] += 1
            else:
                results['failed'] += 1
            
            results['details'].append(result)
        
        logger.info(f"Batch update complete: {results['successful']}/{results['total']} successful")
        return results

# Convenience functions
_tracker_instance = None

def get_outcome_tracker() -> OutcomeTracker:
    """Get singleton outcome tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = OutcomeTracker()
    return _tracker_instance

def update_match_outcome(
    home_team: str,
    away_team: str,
    match_date: datetime,
    home_score: int,
    away_score: int
) -> Dict[str, Any]:
    """Convenience function to update a match outcome."""
    tracker = get_outcome_tracker()
    return tracker.update_prediction_outcome(
        home_team, away_team, match_date, home_score, away_score
    )

def get_current_accuracy() -> Dict[str, Any]:
    """Convenience function to get current accuracy stats."""
    tracker = get_outcome_tracker()
    return tracker.get_accuracy_stats()
