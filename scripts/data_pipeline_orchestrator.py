"""
Enhanced Data Pipeline Orchestrator
Coordinates all data integration, ML training, and prediction generation
"""

import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from database.db_manager import DatabaseManager
from models.enhanced_ml_pipeline import EnhancedMLPipeline
from models.ensemble_predictor import EnsemblePredictor
from models.feature_generator import FeatureGenerator
from scripts.historical_data_integrator import HistoricalDataIntegrator
from scripts.production_data_loader import ProductionDataLoader
from utils.logging_config import get_logger

logger = get_logger(__name__)

class DataPipelineOrchestrator:
    """Orchestrates the complete data pipeline from ingestion to predictions."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the pipeline orchestrator."""
        self.config = config or self._default_config()
        self.db = DatabaseManager()
        
        # Initialize components
        self.historical_integrator = HistoricalDataIntegrator(self.db)
        self.production_loader = ProductionDataLoader(self.db)
        self.feature_generator = FeatureGenerator(self.config.get('features'))
        self.ml_pipeline = EnhancedMLPipeline(self.config.get('ml_models'))
        self.ensemble_predictor = EnsemblePredictor(self.config.get('ensemble'))
        
        # Pipeline state
        self.pipeline_state = {
            'last_data_update': None,
            'last_model_training': None,
            'is_running': False,
            'components_status': {}
        }
        
        logger.info("Data Pipeline Orchestrator initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            'pipeline': {
                'auto_update_interval_hours': 24,
                'force_retrain_threshold_days': 7,
                'batch_prediction_size': 100
            },
            'data_sources': {
                'historical_depth_seasons': 3,
                'update_live_data': True,
                'validate_data_quality': True
            },
            'features': {
                'temporal_features': {'enabled': True},
                'team_strength_features': {'enabled': True},
                'statistical_features': {'enabled': True},
                'betting_features': {'enabled': True}
            },
            'ml_models': {
                'models': {
                    'random_forest': {'n_estimators': 200, 'max_depth': 15},
                    'gradient_boosting': {'n_estimators': 200, 'learning_rate': 0.05},
                    'logistic_regression': {'max_iter': 2000}
                }
            },
            'ensemble': {
                'voting_type': 'soft',
                'weights': [2, 2, 1]  # Higher weight for tree-based models
            },
            'predictions': {
                'confidence_threshold': 0.7,
                'generate_betting_insights': True,
                'include_value_analysis': True
            }
        }
    
    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete data pipeline from start to finish."""
        logger.info("🚀 Starting complete data pipeline execution...")
        
        self.pipeline_state['is_running'] = True
        start_time = time.time()
        
        pipeline_results = {
            'status': 'SUCCESS',
            'start_time': datetime.now().isoformat(),
            'phases': {},
            'summary': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Phase 1: Data Integration
            logger.info("📊 Phase 1: Historical Data Integration")
            data_phase_results = await self._run_data_integration_phase()
            pipeline_results['phases']['data_integration'] = data_phase_results
            
            if data_phase_results.get('errors'):
                pipeline_results['warnings'].extend(data_phase_results['errors'])
            
            # Phase 2: Feature Engineering
            logger.info("🔧 Phase 2: Feature Engineering")
            feature_phase_results = await self._run_feature_engineering_phase()
            pipeline_results['phases']['feature_engineering'] = feature_phase_results
            
            # Phase 3: Model Training
            logger.info("🤖 Phase 3: Model Training & Ensemble")
            model_phase_results = await self._run_model_training_phase()
            pipeline_results['phases']['model_training'] = model_phase_results
            
            # Phase 4: Prediction Generation
            logger.info("🔮 Phase 4: Prediction Generation")
            prediction_phase_results = await self._run_prediction_phase()
            pipeline_results['phases']['predictions'] = prediction_phase_results
            
            # Phase 5: Betting Insights
            logger.info("💰 Phase 5: Betting Insights Generation")
            insights_phase_results = await self._run_insights_phase()
            pipeline_results['phases']['betting_insights'] = insights_phase_results
            
            # Generate summary
            pipeline_results['summary'] = self._generate_pipeline_summary(pipeline_results)
            
        except Exception as e:
            error_msg = f"Critical pipeline error: {e}"
            logger.error(error_msg)
            pipeline_results['status'] = 'FAILED'
            pipeline_results['errors'].append(error_msg)
        
        finally:
            self.pipeline_state['is_running'] = False
            pipeline_results['duration_seconds'] = time.time() - start_time
            pipeline_results['end_time'] = datetime.now().isoformat()
        
        logger.info(f"🏁 Pipeline completed in {pipeline_results['duration_seconds']:.2f}s with status: {pipeline_results['status']}")
        
        return pipeline_results
    
    async def _run_data_integration_phase(self) -> Dict[str, Any]:
        """Run data integration phase."""
        phase_results = {
            'status': 'SUCCESS',
            'duration': 0,
            'actions': [],
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Load production data (quick setup)
            logger.info("Loading production baseline data...")
            production_results = await self.production_loader.load_essential_data()
            phase_results['actions'].append({
                'action': 'production_data_load',
                'result': production_results
            })
            
            # Historical data integration (comprehensive)
            if self.config['data_sources']['update_live_data']:
                logger.info("Integrating comprehensive historical data...")
                historical_results = await self.historical_integrator.integrate_comprehensive_data()
                phase_results['actions'].append({
                    'action': 'historical_data_integration',
                    'result': historical_results
                })
                
                if historical_results.get('errors'):
                    phase_results['errors'].extend(historical_results['errors'])
            
            # Data validation
            if self.config['data_sources']['validate_data_quality']:
                logger.info("Validating data quality...")
                validation_results = await self._validate_pipeline_data()
                phase_results['actions'].append({
                    'action': 'data_validation',
                    'result': validation_results
                })
            
            self.pipeline_state['last_data_update'] = datetime.now()
            
        except Exception as e:
            phase_results['status'] = 'FAILED'
            phase_results['errors'].append(str(e))
            logger.error(f"Data integration phase failed: {e}")
        
        finally:
            phase_results['duration'] = time.time() - start_time
        
        return phase_results
    
    async def _run_feature_engineering_phase(self) -> Dict[str, Any]:
        """Run feature engineering phase."""
        phase_results = {
            'status': 'SUCCESS',
            'duration': 0,
            'features_generated': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Get match data for feature engineering
            match_data = await self._get_training_data()
            
            if not match_data.empty:
                # Generate comprehensive features
                logger.info(f"Engineering features for {len(match_data)} matches...")
                
                enhanced_features = self.feature_generator.generate_comprehensive_features(
                    match_data, historical_data=match_data  # Use same data for now
                )
                
                phase_results['features_generated'] = len(enhanced_features.columns)
                phase_results['matches_processed'] = len(enhanced_features)
                
                # Store features for later use
                self.processed_features = enhanced_features
                
                logger.info(f"Generated {phase_results['features_generated']} features for {phase_results['matches_processed']} matches")
            else:
                phase_results['status'] = 'WARNING'
                phase_results['errors'].append("No training data available for feature engineering")
        
        except Exception as e:
            phase_results['status'] = 'FAILED'
            phase_results['errors'].append(str(e))
            logger.error(f"Feature engineering phase failed: {e}")
        
        finally:
            phase_results['duration'] = time.time() - start_time
        
        return phase_results
    
    async def _run_model_training_phase(self) -> Dict[str, Any]:
        """Run model training phase."""
        phase_results = {
            'status': 'SUCCESS',
            'duration': 0,
            'models_trained': 0,
            'best_model': None,
            'performance': {},
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            if hasattr(self, 'processed_features') and not self.processed_features.empty:
                # Train individual models
                logger.info("Training ML pipeline...")
                
                ml_results = self.ml_pipeline.train(
                    self.processed_features,
                    target_column='result',
                    test_size=0.2
                )
                
                phase_results['models_trained'] = len(ml_results)
                phase_results['performance']['individual'] = ml_results
                
                # Train ensemble
                logger.info("Training ensemble model...")
                
                ensemble_results = self.ensemble_predictor.train(
                    self.processed_features,
                    target_column='result',
                    test_size=0.2
                )
                
                phase_results['performance']['ensemble'] = ensemble_results
                phase_results['best_model'] = ensemble_results.get('best_individual', 'random_forest')
                
                # Update pipeline state
                self.pipeline_state['last_model_training'] = datetime.now()
                
                logger.info(f"Model training completed. Best model: {phase_results['best_model']}")
                
            else:
                phase_results['status'] = 'FAILED'
                phase_results['errors'].append("No processed features available for training")
        
        except Exception as e:
            phase_results['status'] = 'FAILED'
            phase_results['errors'].append(str(e))
            logger.error(f"Model training phase failed: {e}")
        
        finally:
            phase_results['duration'] = time.time() - start_time
        
        return phase_results
    
    async def _run_prediction_phase(self) -> Dict[str, Any]:
        """Run prediction generation phase."""
        phase_results = {
            'status': 'SUCCESS',
            'duration': 0,
            'predictions_generated': 0,
            'high_confidence_predictions': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Get upcoming matches for prediction
            upcoming_matches = await self._get_upcoming_matches()
            
            if not upcoming_matches.empty:
                logger.info(f"Generating predictions for {len(upcoming_matches)} upcoming matches...")
                
                # Generate ensemble predictions
                predictions = self.ensemble_predictor.predict(upcoming_matches)
                
                phase_results['predictions_generated'] = len(predictions['predictions'])
                
                # Count high confidence predictions
                if predictions.get('confidence'):
                    high_conf_count = sum(1 for conf in predictions['confidence'] 
                                        if conf >= self.config['predictions']['confidence_threshold'])
                    phase_results['high_confidence_predictions'] = high_conf_count
                
                # Store predictions for insights phase
                self.match_predictions = predictions
                
                logger.info(f"Generated {phase_results['predictions_generated']} predictions ({phase_results['high_confidence_predictions']} high confidence)")
                
            else:
                phase_results['status'] = 'WARNING'
                phase_results['errors'].append("No upcoming matches found for prediction")
        
        except Exception as e:
            phase_results['status'] = 'FAILED'
            phase_results['errors'].append(str(e))
            logger.error(f"Prediction phase failed: {e}")
        
        finally:
            phase_results['duration'] = time.time() - start_time
        
        return phase_results
    
    async def _run_insights_phase(self) -> Dict[str, Any]:
        """Run betting insights generation phase."""
        phase_results = {
            'status': 'SUCCESS',
            'duration': 0,
            'insights_generated': 0,
            'high_value_bets': 0,
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            if hasattr(self, 'processed_features') and hasattr(self, 'match_predictions'):
                logger.info("Generating betting insights...")
                
                # Generate comprehensive betting insights
                betting_insights = self.feature_generator.generate_betting_insights(
                    self.processed_features
                )
                
                phase_results['insights_generated'] = len(betting_insights.get('high_value_matches', []))
                phase_results['high_value_bets'] = len([
                    match for match in betting_insights.get('high_value_matches', [])
                    if match.get('value_score', 0) > 1.2
                ])
                
                # Store insights
                self.betting_insights = betting_insights
                
                logger.info(f"Generated insights for {phase_results['insights_generated']} matches ({phase_results['high_value_bets']} high value)")
                
            else:
                phase_results['status'] = 'WARNING'
                phase_results['errors'].append("Insufficient data for insights generation")
        
        except Exception as e:
            phase_results['status'] = 'FAILED'
            phase_results['errors'].append(str(e))
            logger.error(f"Insights phase failed: {e}")
        
        finally:
            phase_results['duration'] = time.time() - start_time
        
        return phase_results
    
    async def _get_training_data(self) -> pd.DataFrame:
        """Get training data from database."""
        try:
            with self.db.session_scope() as session:
                from database.schema import Match

                # Get completed matches for training
                matches = session.query(Match).filter(
                    Match.home_score.isnot(None),
                    Match.away_score.isnot(None)
                ).limit(1000).all()
                
                if matches:
                    # Convert to DataFrame with basic features
                    data = []
                    for match in matches:
                        # Determine result
                        if match.home_score > match.away_score:
                            result = 'home_win'
                        elif match.home_score < match.away_score:
                            result = 'away_win'
                        else:
                            result = 'draw'
                        
                        data.append({
                            'match_id': match.id,
                            'home_team_id': match.home_team_id,
                            'away_team_id': match.away_team_id,
                            'home_score': match.home_score,
                            'away_score': match.away_score,
                            'result': result,
                            'match_date': match.match_date
                        })
                    
                    return pd.DataFrame(data)
        
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
        
        return pd.DataFrame()
    
    async def _get_upcoming_matches(self) -> pd.DataFrame:
        """Get upcoming matches for prediction."""
        try:
            with self.db.session_scope() as session:
                from database.schema import Match

                # Get future matches
                upcoming = session.query(Match).filter(
                    Match.match_date > datetime.now(),
                    Match.home_score.is_(None)
                ).limit(50).all()
                
                if upcoming:
                    data = []
                    for match in upcoming:
                        data.append({
                            'match_id': match.id,
                            'home_team_id': match.home_team_id,
                            'away_team_id': match.away_team_id,
                            'match_date': match.match_date,
                            'league_id': match.league_id
                        })
                    
                    return pd.DataFrame(data)
        
        except Exception as e:
            logger.error(f"Error getting upcoming matches: {e}")
        
        return pd.DataFrame()
    
    async def _validate_pipeline_data(self) -> Dict[str, Any]:
        """Validate pipeline data quality."""
        validation_results = {
            'status': 'PASSED',
            'checks': [],
            'issues': []
        }
        
        try:
            with self.db.session_scope() as session:
                from sqlalchemy import text

                from database.schema import League, Match, Team

                # Check data completeness
                league_count = session.query(League).count()
                team_count = session.query(Team).count()
                match_count = session.query(Match).count()
                
                validation_results['checks'].extend([
                    {'check': 'league_count', 'result': league_count, 'threshold': 5, 'passed': league_count >= 5},
                    {'check': 'team_count', 'result': team_count, 'threshold': 100, 'passed': team_count >= 100},
                    {'check': 'match_count', 'result': match_count, 'threshold': 1000, 'passed': match_count >= 1000}
                ])
                
                # Check for data consistency
                orphaned_matches = session.execute(text("""
                    SELECT COUNT(*) as count FROM matches m 
                    WHERE m.home_team_id NOT IN (SELECT id FROM teams) 
                    OR m.away_team_id NOT IN (SELECT id FROM teams)
                """)).fetchone()
                
                if orphaned_matches and orphaned_matches.count > 0:
                    validation_results['issues'].append(f"Found {orphaned_matches.count} orphaned matches")
                
                # Overall status
                failed_checks = [c for c in validation_results['checks'] if not c['passed']]
                if failed_checks or validation_results['issues']:
                    validation_results['status'] = 'WARNING'
        
        except Exception as e:
            validation_results['status'] = 'FAILED'
            validation_results['issues'].append(f"Validation error: {e}")
        
        return validation_results
    
    def _generate_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pipeline execution summary."""
        summary = {
            'overall_status': results['status'],
            'phases_completed': len([p for p in results['phases'].values() if p.get('status') == 'SUCCESS']),
            'total_phases': len(results['phases']),
            'total_duration': results.get('duration_seconds', 0),
            'key_metrics': {}
        }
        
        # Extract key metrics
        if 'data_integration' in results['phases']:
            data_phase = results['phases']['data_integration']
            for action in data_phase.get('actions', []):
                if action['action'] == 'production_data_load':
                    result = action['result']
                    summary['key_metrics']['teams_loaded'] = result.get('teams_loaded', 0)
                    summary['key_metrics']['matches_created'] = result.get('sample_matches_created', 0)
        
        if 'feature_engineering' in results['phases']:
            feat_phase = results['phases']['feature_engineering']
            summary['key_metrics']['features_generated'] = feat_phase.get('features_generated', 0)
        
        if 'model_training' in results['phases']:
            model_phase = results['phases']['model_training']
            summary['key_metrics']['models_trained'] = model_phase.get('models_trained', 0)
        
        if 'predictions' in results['phases']:
            pred_phase = results['phases']['predictions']
            summary['key_metrics']['predictions_generated'] = pred_phase.get('predictions_generated', 0)
        
        return summary
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'is_running': self.pipeline_state['is_running'],
            'last_data_update': self.pipeline_state['last_data_update'].isoformat() if self.pipeline_state['last_data_update'] else None,
            'last_model_training': self.pipeline_state['last_model_training'].isoformat() if self.pipeline_state['last_model_training'] else None,
            'components_status': self.pipeline_state['components_status'],
            'config': self.config
        }


async def main():
    """Main function for testing the pipeline."""
    orchestrator = DataPipelineOrchestrator()
    results = await orchestrator.run_complete_pipeline()
    
    print("\n" + "="*60)
    print("🎯 PIPELINE EXECUTION RESULTS")
    print("="*60)
    print(f"Status: {results['status']}")
    print(f"Duration: {results.get('duration_seconds', 0):.2f}s")
    print(f"Phases: {results['summary']['phases_completed']}/{results['summary']['total_phases']}")
    
    if results.get('errors'):
        print(f"\nErrors: {len(results['errors'])}")
        for error in results['errors'][:3]:  # Show first 3 errors
            print(f"  • {error}")
    
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())
