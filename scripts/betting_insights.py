#!/usr/bin/env python
"""
Betting Insights Generator for GoalDiggers Platform

This script analyzes upcoming matches and generates actionable betting insights
based on historical data and trained machine learning models.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.setup_env import setup_environment

# Configure environment first
logger = setup_environment()

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    class MockMLFlow:
        @staticmethod
        def set_tracking_uri(*args, **kwargs): pass
        @staticmethod
        def set_experiment(*args, **kwargs): pass
        @staticmethod
        def start_run(*args, **kwargs): return MockMLFlow()
        @staticmethod
        def log_metric(*args, **kwargs): pass
        @staticmethod
        def log_param(*args, **kwargs): pass
        @staticmethod
        def sklearn_log_model(*args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    mlflow = MockMLFlow()
    mlflow.sklearn = MockMLFlow()
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from database.db_manager import DatabaseManager
from database.schema import (League, Match, MatchStats, Odds, Prediction, Team,
                             TeamStats)
from scripts.ingest_data import harmonize_column_names
from utils.config import Config


class BettingInsightsGenerator:
    """Generates actionable betting insights from model predictions."""
    
    def __init__(self, model_uri: str = None):
        """Initialize the insights generator."""
        self.db = DatabaseManager()
        self.model_uri = model_uri or self._get_latest_model_uri()
        self.model = self._load_model()
        
    def _get_latest_model_uri(self) -> str:
        """Get the URI of the latest trained model."""
        try:
            # Try to get the latest registered model
            latest_model = mlflow.search_registered_models(order_by=["last_updated_timestamp DESC"], max_results=1)
            if latest_model and len(latest_model) > 0:
                model_name = latest_model[0].name
                return f"models:/{model_name}/latest"
            else:
                # Fall back to latest run with a model
                latest_runs = mlflow.search_runs(
                    order_by=["start_time DESC"],
                    filter_string="attributes.status = 'FINISHED'",
                    max_results=1
                )
                if len(latest_runs) > 0:
                    run_id = latest_runs.iloc[0].run_id
                    return f"runs:/{run_id}/model"
        except Exception as e:
            logger.error(f"Error getting latest model URI: {e}")
            
        # Default to a specific model if we can't find the latest
        return "models:/random_forest_model/latest"
    
    def _load_model(self):
        """Load the model from MLflow."""
        logger.info(f"Loading model from {self.model_uri}")
        try:
            return mlflow.pyfunc.load_model(self.model_uri)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(f"Could not load model from {self.model_uri}")
    
    def get_upcoming_matches(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming matches from the database."""
        now = datetime.now()
        future_date = now + timedelta(days=days_ahead)
        
        with self.db.session_scope() as session:
            matches = session.query(Match).filter(
                Match.match_date >= now,
                Match.match_date <= future_date,
                Match.status == 'SCHEDULED'
            ).all()
            
            results = []
            for match in matches:
                home_team = session.query(Team).filter_by(id=match.home_team_id).first()
                away_team = session.query(Team).filter_by(id=match.away_team_id).first()
                league = session.query(League).filter_by(id=match.league_id).first()
                
                # Skip matches with missing team info
                if not home_team or not away_team:
                    continue
                    
                results.append({
                    'match_id': match.id,
                    'league': league.name if league else "Unknown League",
                    'league_id': match.league_id,
                    'home_team': home_team.name,
                    'away_team': away_team.name,
                    'match_date': match.match_date.isoformat(),
                    'venue': match.venue,
                    'home_team_id': match.home_team_id,
                    'away_team_id': match.away_team_id
                })
                
            return results
    
    def generate_feature_vector(self, match: Dict) -> pd.DataFrame:
        """
        Generate a feature vector for a match using historical data.
        
        This function creates features similar to what the model was trained on,
        including team performance metrics and historical data.
        """
        features = {}
        
        with self.db.session_scope() as session:
            # Get team stats
            home_stats = session.query(TeamStats).filter_by(team_id=match['home_team_id']).first()
            away_stats = session.query(TeamStats).filter_by(team_id=match['away_team_id']).first()
            
            if home_stats and away_stats:
                # Team performance metrics
                features['team_perf'] = home_stats.points / max(home_stats.matches_played, 1) if home_stats.matches_played else 0
                features['opp_perf'] = away_stats.points / max(away_stats.matches_played, 1) if away_stats.matches_played else 0
                
                # Additional features
                features['feature3'] = home_stats.goals_for / max(home_stats.matches_played, 1) if home_stats.matches_played else 0
                features['feature4'] = away_stats.goals_for / max(away_stats.matches_played, 1) if away_stats.matches_played else 0
                features['feature5'] = abs(home_stats.goals_for - away_stats.goals_for) / max(1, min(home_stats.matches_played, away_stats.matches_played))
                
                # Also include reference features that might be in the model
                features['home_score'] = home_stats.goals_for / max(home_stats.matches_played, 1) if home_stats.matches_played else 0
                features['away_score'] = away_stats.goals_for / max(away_stats.matches_played, 1) if away_stats.matches_played else 0
                
                # Include team IDs as additional context (won't be used by model)
                features['match_id'] = match['match_id']
                features['home_team'] = match['home_team']
                features['away_team'] = match['away_team']
                features['match_date'] = match['match_date']
                features['league'] = match['league']
            else:
                # If we don't have stats, use default values
                logger.warning(f"Missing team stats for {match['home_team']} or {match['away_team']}")
                features = {
                    'team_perf': 0.5,
                    'opp_perf': 0.5,
                    'feature3': 0,
                    'feature4': 0,
                    'feature5': 0,
                    'home_score': 0,
                    'away_score': 0,
                    'match_id': match['match_id'],
                    'home_team': match['home_team'],
                    'away_team': match['away_team'],
                    'match_date': match['match_date'],
                    'league': match['league']
                }
                
        return pd.DataFrame([features])
    
    def predict_match(self, match: Dict) -> Dict:
        """Generate prediction for a single match."""
        # Generate features for the match
        features_df = self.generate_feature_vector(match)
        
        # Extract the features used for prediction
        X = features_df.drop(columns=['match_id', 'home_team', 'away_team', 'match_date', 'league'])
        
        # Make prediction
        try:
            # Try to get probabilities
            probas = self.model.predict_proba(X)
            
            # Interpret the predictions
            result_proba = {}
            if hasattr(self.model, 'classes_') and len(self.model.classes_) == 3:
                # This is likely a 3-class model (home win, draw, away win)
                result_proba = {
                    'home_win': float(probas[0][0]),
                    'draw': float(probas[0][1]),
                    'away_win': float(probas[0][2])
                }
            else:
                # Binary model or other format
                prediction = int(self.model.predict(X)[0])
                result_proba = {
                    'home_win': float(1 - probas[0][1]) if prediction == 0 else 0,
                    'draw': 0,  # Not predicted by binary model
                    'away_win': float(probas[0][1]) if prediction == 1 else 0
                }
            
            # Calculate confidence
            confidence = max(result_proba.values())
            
            # Get predicted outcome
            outcomes = ['home_win', 'draw', 'away_win']
            predicted_outcome = outcomes[np.argmax([result_proba[outcome] for outcome in outcomes])]
            
            return {
                'match_id': match['match_id'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'match_date': match['match_date'],
                'league': match['league'],
                'prediction': predicted_outcome,
                'confidence': confidence,
                'probabilities': result_proba,
                'features_used': X.columns.tolist(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'match_id': match['match_id'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'match_date': match['match_date'],
                'league': match['league'],
                'error': str(e)
            }
    
    def generate_insights(self, match_prediction: Dict) -> Dict:
        """
        Generate actionable betting insights from a match prediction.
        
        This function looks at prediction confidence, historical odds,
        and other factors to generate recommendations.
        """
        insights = {
            'match_id': match_prediction['match_id'],
            'home_team': match_prediction['home_team'],
            'away_team': match_prediction['away_team'],
            'match_date': match_prediction['match_date'],
            'recommendations': []
        }
        
        if 'error' in match_prediction:
            insights['status'] = 'error'
            insights['error'] = match_prediction['error']
            return insights
            
        # Get current odds if available
        odds = self._get_current_odds(match_prediction['match_id'])
        
        # Calculate expected value and recommendations
        if odds:
            # Calculate expected value for each outcome
            ev_home = match_prediction['probabilities']['home_win'] * odds['home_win'] - 1
            ev_draw = match_prediction['probabilities']['draw'] * odds['draw'] - 1 if 'draw' in match_prediction['probabilities'] else -1
            ev_away = match_prediction['probabilities']['away_win'] * odds['away_win'] - 1
            
            # Generate recommendations based on expected value
            if ev_home > 0.1:
                insights['recommendations'].append({
                    'bet': 'home_win',
                    'team': match_prediction['home_team'],
                    'confidence': float(match_prediction['probabilities']['home_win']),
                    'odds': float(odds['home_win']),
                    'expected_value': float(ev_home),
                    'reason': f"Strong value bet on {match_prediction['home_team']} win"
                })
            
            if ev_draw > 0.1 and 'draw' in match_prediction['probabilities']:
                insights['recommendations'].append({
                    'bet': 'draw',
                    'confidence': float(match_prediction['probabilities']['draw']),
                    'odds': float(odds['draw']),
                    'expected_value': float(ev_draw),
                    'reason': "Good value on a draw result"
                })
                
            if ev_away > 0.1:
                insights['recommendations'].append({
                    'bet': 'away_win',
                    'team': match_prediction['away_team'],
                    'confidence': float(match_prediction['probabilities']['away_win']),
                    'odds': float(odds['away_win']),
                    'expected_value': float(ev_away),
                    'reason': f"Strong value bet on {match_prediction['away_team']} win"
                })
        else:
            # Without odds, recommend based on confidence only
            if match_prediction['confidence'] > 0.7:
                insights['recommendations'].append({
                    'bet': match_prediction['prediction'],
                    'team': match_prediction['home_team'] if match_prediction['prediction'] == 'home_win' else match_prediction['away_team'],
                    'confidence': float(match_prediction['confidence']),
                    'reason': f"High confidence prediction: {match_prediction['confidence']:.2f}"
                })
                
        # Add summary insight
        if insights['recommendations']:
            insights['summary'] = f"Found {len(insights['recommendations'])} value betting opportunities."
            insights['status'] = 'success'
        else:
            insights['summary'] = "No strong betting recommendations found."
            insights['status'] = 'neutral'
            
        return insights
    
    def _get_current_odds(self, match_id: str) -> Optional[Dict]:
        """Get current odds for a match from the database."""
        with self.db.session_scope() as session:
            odds_record = session.query(Odds).filter_by(match_id=match_id).order_by(Odds.timestamp.desc()).first()
            
            if odds_record:
                return {
                    'home_win': odds_record.home_win,
                    'draw': odds_record.draw,
                    'away_win': odds_record.away_win
                }
            
            # No odds in database, generate synthetic odds based on league averages
            match = session.query(Match).filter_by(id=match_id).first()
            if match and match.league_id:
                # Calculate average odds for this league
                avg_odds = session.query(
                    func.avg(Odds.home_win).label('avg_home'),
                    func.avg(Odds.draw).label('avg_draw'),
                    func.avg(Odds.away_win).label('avg_away')
                ).join(Match).filter(Match.league_id == match.league_id).first()
                
                if avg_odds and avg_odds.avg_home:
                    return {
                        'home_win': float(avg_odds.avg_home),
                        'draw': float(avg_odds.avg_draw),
                        'away_win': float(avg_odds.avg_away)
                    }
                    
            # Default odds if nothing available
            return {
                'home_win': 2.5,
                'draw': 3.2,
                'away_win': 2.9
            }
            
    def analyze_user_selected_matches(self, team_names: List[str], save_path: Optional[str] = None) -> List[Dict]:
        """
        Generate insights for matches involving specific teams selected by the user.
        
        Args:
            team_names: List of team names to find matches for
            save_path: Optional path to save results to
            
        Returns:
            List of insights for matches involving the selected teams
        """
        team_names_lower = [name.lower() for name in team_names]
        
        # Get upcoming matches
        all_matches = self.get_upcoming_matches(days_ahead=30)
        
        # Filter for matches involving the selected teams
        user_matches = [
            match for match in all_matches
            if match['home_team'].lower() in team_names_lower or 
               match['away_team'].lower() in team_names_lower
        ]
        
        if not user_matches:
            logger.warning(f"No upcoming matches found for teams: {', '.join(team_names)}")
            return []
            
        logger.info(f"Found {len(user_matches)} upcoming matches for selected teams")
        
        # Generate predictions and insights
        results = []
        for match in user_matches:
            prediction = self.predict_match(match)
            insights = self.generate_insights(prediction)
            results.append(insights)
            
            # Store predictions and insights in the database
            self._store_prediction(prediction, insights)
        
        # Save to file if requested
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
                
        return results
    
    def _store_prediction(self, prediction: Dict, insights: Dict) -> None:
        """Store prediction and insights in the database."""
        try:
            with self.db.session_scope() as session:
                # Check if prediction exists
                existing = session.query(Prediction).filter_by(match_id=prediction['match_id']).first()
                
                if existing:
                    # Update existing prediction
                    existing.home_win_prob = prediction['probabilities'].get('home_win', 0)
                    existing.draw_prob = prediction['probabilities'].get('draw', 0)
                    existing.away_win_prob = prediction['probabilities'].get('away_win', 0)
                    existing.predicted_outcome = prediction['prediction']
                    existing.confidence = prediction['confidence']
                    existing.model_version = self.model_uri
                    existing.timestamp = datetime.now()
                    
                    # Update insights JSON
                    if hasattr(existing, 'insights_json'):
                        existing.insights_json = json.dumps(insights)
                else:
                    # Create new prediction
                    pred = Prediction(
                        id=f"pred_{prediction['match_id']}",
                        match_id=prediction['match_id'],
                        home_win_prob=prediction['probabilities'].get('home_win', 0),
                        draw_prob=prediction['probabilities'].get('draw', 0),
                        away_win_prob=prediction['probabilities'].get('away_win', 0),
                        predicted_outcome=prediction['prediction'],
                        confidence=prediction['confidence'],
                        model_version=self.model_uri,
                        timestamp=datetime.now()
                    )
                    
                    # Add insights JSON if the column exists
                    if hasattr(Prediction, 'insights_json'):
                        pred.insights_json = json.dumps(insights)
                        
                    session.add(pred)
                    
                # Add value bets if found
                if insights.get('recommendations'):
                    for rec in insights['recommendations']:
                        if 'expected_value' in rec and rec['expected_value'] > 0:
                            value_bet = ValueBet(
                                id=f"vb_{prediction['match_id']}_{rec['bet']}",
                                match_id=prediction['match_id'],
                                bet_type=rec['bet'],
                                odds=rec.get('odds', 0),
                                expected_value=rec['expected_value'],
                                confidence=rec['confidence'],
                                reason=rec['reason']
                            )
                            session.add(value_bet)
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate betting insights for upcoming matches")
    parser.add_argument("--model-uri", type=str, help="URI of the model to use")
    parser.add_argument("--teams", nargs="+", help="Team names to generate insights for")
    parser.add_argument("--output", type=str, help="Path to save results to")
    parser.add_argument("--days", type=int, default=14, help="Number of days to look ahead")
    args = parser.parse_args()
    
    # Create insights generator
    insights_generator = BettingInsightsGenerator(model_uri=args.model_uri)
    
    if args.teams:
        # Generate insights for specific teams
        logger.info(f"Generating insights for teams: {', '.join(args.teams)}")
        results = insights_generator.analyze_user_selected_matches(
            team_names=args.teams,
            save_path=args.output
        )
        logger.info(f"Generated insights for {len(results)} matches")
    else:
        # Get all upcoming matches
        matches = insights_generator.get_upcoming_matches(days_ahead=args.days)
        logger.info(f"Found {len(matches)} upcoming matches")
        
        # Generate predictions and insights for each match
        results = []
        for match in matches:
            prediction = insights_generator.predict_match(match)
            insights = insights_generator.generate_insights(prediction)
            results.append(insights)
            
        # Save results if output path provided
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved insights to {args.output}")
        else:
            # Print summary of insights
            print(f"Generated insights for {len(results)} matches:")
            for insight in results[:5]:  # Show first 5
                status = insight.get('status', 'unknown')
                home = insight.get('home_team', 'Unknown')
                away = insight.get('away_team', 'Unknown')
                recs = len(insight.get('recommendations', []))
                print(f"- {home} vs {away}: {status} ({recs} recommendations)")
            
            if len(results) > 5:
                print(f"... and {len(results) - 5} more matches")
    
    return 0


if __name__ == "__main__":
    from sqlalchemy import func
    sys.exit(main())
