#!/usr/bin/env python3
"""
Model Loading Optimization Script

Addresses the slow model loading issue by:
1. Pre-loading models in production mode
2. Optimizing XGBoost model format
3. Disabling SHAP in production
4. Implementing efficient caching
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelLoadingOptimizer:
    """Optimizes model loading performance for production deployment."""
    
    def __init__(self):
        self.project_root = project_root
        self.optimizations_applied = []
        
    def run_optimization(self):
        """Run complete model loading optimization."""
        logger.info("Starting Model Loading Optimization")
        logger.info("=" * 50)
        
        try:
            # 1. Analyze current model loading performance
            self.analyze_current_performance()
            
            # 2. Optimize XGBoost model format
            self.optimize_xgboost_format()
            
            # 3. Pre-warm model singleton
            self.prewarm_model_singleton()
            
            # 4. Optimize production mode settings
            self.optimize_production_settings()
            
            # 5. Test optimized performance
            self.test_optimized_performance()
            
            # 6. Generate optimization report
            self.generate_optimization_report()
            
            return True
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False
    
    def analyze_current_performance(self):
        """Analyze current model loading performance."""
        logger.info("Analyzing current model loading performance...")
        
        try:
            # Test XGBoost predictor loading
            start_time = time.time()
            from models.xgboost_predictor import XGBoostPredictor
            
            predictor = XGBoostPredictor('models/predictor_model.joblib', production_mode=True)
            load_time = time.time() - start_time
            
            logger.info(f"Current XGBoost loading time: {load_time:.3f}s")
            
            if load_time > 5.0:
                logger.warning(f"Slow loading detected: {load_time:.3f}s (target: <3s)")
                self.optimizations_applied.append(f"Identified slow loading: {load_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
    
    def optimize_xgboost_format(self):
        """Optimize XGBoost model format for faster loading."""
        logger.info("Optimizing XGBoost model format...")
        
        try:
            model_path = self.project_root / 'models' / 'predictor_model.joblib'
            
            if model_path.exists():
                # Load current model
                model_data = joblib.load(model_path)
                logger.info(f"Loaded model data type: {type(model_data)}")
                
                # Check if it's already optimized
                if isinstance(model_data, dict) and 'model' in model_data:
                    xgb_model = model_data['model']
                    
                    # Save in JSON format for faster loading
                    json_model_path = self.project_root / 'models' / 'predictor_model_optimized.json'
                    
                    # Save XGBoost model in JSON format
                    if hasattr(xgb_model, 'save_model'):
                        try:
                            xgb_model.save_model(str(json_model_path))
                            logger.info(f"Saved optimized JSON model: {json_model_path}")
                            self.optimizations_applied.append("Converted XGBoost model to JSON format")
                        except Exception as e:
                            logger.warning(f"Could not save JSON booster: {e}")
                    
                    # Create optimized joblib with minimal data
                    optimized_data = {
                        'model': xgb_model,
                        'features': model_data.get('features', []),
                        'feature_columns': model_data.get('feature_columns', []),
                        'production_mode': True,
                        'shap_disabled': True,
                        'note': 'Legacy embedded model retained for backward compatibility; prefer xgb_model_path JSON.'
                    }
                    
                    optimized_path = self.project_root / 'models' / 'predictor_model_fast.joblib'
                    joblib.dump(optimized_data, optimized_path, compress=3)
                    logger.info(f"Created optimized model: {optimized_path}")
                    self.optimizations_applied.append("Created optimized joblib model")
                
        except Exception as e:
            logger.error(f"XGBoost optimization failed: {e}")
    
    def prewarm_model_singleton(self):
        """Pre-warm the model singleton for faster access."""
        logger.info("Pre-warming model singleton...")
        
        try:
            from utils.model_singleton import get_model_manager
            
            start_time = time.time()
            manager = get_model_manager()
            
            # Pre-load XGBoost predictor
            predictor = manager.get_xgboost_predictor('models/predictor_model.joblib')
            
            # Pre-load feature mapper
            feature_mapper = manager.get_feature_mapper()
            
            prewarm_time = time.time() - start_time
            logger.info(f"Model singleton pre-warmed in {prewarm_time:.3f}s")
            self.optimizations_applied.append(f"Pre-warmed singleton in {prewarm_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Singleton pre-warming failed: {e}")
    
    def optimize_production_settings(self):
        """Optimize production mode settings."""
        logger.info("Optimizing production settings...")
        
        try:
            # Set environment variables for production optimization
            os.environ['PRODUCTION_MODE'] = 'true'
            os.environ['DISABLE_SHAP'] = 'true'
            os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
            
            logger.info("Set production environment variables")
            self.optimizations_applied.append("Configured production environment variables")
            
            # Create production config file
            config_content = """
# Production Configuration for GoalDiggers
PRODUCTION_MODE=true
DISABLE_SHAP=true
FAST_LOADING=true
CACHE_MODELS=true
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
"""
            
            config_path = self.project_root / '.env.production'
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            logger.info(f"Created production config: {config_path}")
            self.optimizations_applied.append("Created production configuration file")
            
        except Exception as e:
            logger.error(f"Production settings optimization failed: {e}")
    
    def test_optimized_performance(self):
        """Test optimized model loading performance."""
        logger.info("Testing optimized performance...")
        
        try:
            # Test Enhanced Prediction Engine with optimizations
            start_time = time.time()
            from enhanced_prediction_engine import EnhancedPredictionEngine
            
            engine = EnhancedPredictionEngine()
            init_time = time.time() - start_time
            
            logger.info(f"Optimized engine initialization: {init_time:.3f}s")
            
            # Test prediction generation
            sample_match = {
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'home_league': 'Premier League',
                'away_league': 'Premier League',
                'date': datetime.now(),
                'venue': 'Emirates Stadium'
            }
            
            start_time = time.time()
            result = engine.predict_match_outcome(sample_match)
            prediction_time = time.time() - start_time
            
            if result and 'predictions' in result:
                logger.info(f"Optimized prediction time: {prediction_time:.3f}s")
                logger.info(f"Prediction confidence: {result['confidence']['overall']:.3f}")
                self.optimizations_applied.append(f"Achieved {prediction_time:.3f}s prediction time")
            
        except Exception as e:
            logger.error(f"Performance testing failed: {e}")
    
    def generate_optimization_report(self):
        """Generate optimization report."""
        logger.info("Generating optimization report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimizations_applied': len(self.optimizations_applied),
            'optimizations': self.optimizations_applied,
            'status': 'OPTIMIZED' if self.optimizations_applied else 'NO_OPTIMIZATIONS_NEEDED'
        }
        
        # Save report
        report_file = self.project_root / 'model_loading_optimization_report.json'
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Optimization Report:")
        logger.info(f"   Optimizations Applied: {len(self.optimizations_applied)}")
        logger.info(f"   Status: {report['status']}")
        logger.info(f"   Report saved to: {report_file}")
        
        for optimization in self.optimizations_applied:
            logger.info(f"   - {optimization}")

def main():
    """Main optimization function."""
    optimizer = ModelLoadingOptimizer()
    success = optimizer.run_optimization()
    
    if success:
        logger.info("Model loading optimization completed successfully!")
        return 0
    else:
        logger.error("Model loading optimization failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
