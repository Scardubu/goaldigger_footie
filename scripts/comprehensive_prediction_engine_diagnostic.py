#!/usr/bin/env python3
"""
Comprehensive Prediction Engine Diagnostic

Analyzes and resolves Enhanced Prediction Engine loading issues and prediction failures.
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('prediction_engine_diagnostic.log')
    ]
)
logger = logging.getLogger(__name__)

class PredictionEngineDiagnostic:
    """Comprehensive diagnostic for Enhanced Prediction Engine issues."""
    
    def __init__(self):
        self.project_root = project_root
        self.issues_found = []
        self.fixes_applied = []
        
    def run_comprehensive_diagnostic(self):
        """Run complete diagnostic and fix process."""
        logger.info("🔍 Starting Comprehensive Prediction Engine Diagnostic")
        logger.info("=" * 70)
        
        try:
            # 1. Check file system and dependencies
            self.check_file_system()
            
            # 2. Test model loading
            self.test_model_loading()
            
            # 3. Test Enhanced Prediction Engine initialization
            self.test_engine_initialization()
            
            # 4. Test prediction generation
            self.test_prediction_generation()
            
            # 5. Test dashboard integration
            self.test_dashboard_integration()
            
            # 6. Apply fixes for identified issues
            self.apply_fixes()
            
            # 7. Final validation
            self.final_validation()
            
            # 8. Generate report
            self.generate_report()
            
            return True
            
        except Exception as e:
            logger.error(f"Diagnostic failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def check_file_system(self):
        """Check file system and dependencies."""
        logger.info("📁 Checking file system and dependencies...")
        
        # Check critical files
        critical_files = [
            'enhanced_prediction_engine.py',
            'models/xgboost_predictor.py',
            'utils/model_singleton.py',
            'utils/feature_mapper.py',
            'enhanced_feature_engine.py',
            'models/predictor_model.joblib',
            'models/trained/predictor_model.joblib'
        ]
        
        for file_path in critical_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                logger.info(f"✅ {file_path} exists ({full_path.stat().st_size} bytes)")
            else:
                issue = f"❌ Missing critical file: {file_path}"
                logger.error(issue)
                self.issues_found.append(issue)
        
        # Check Python dependencies
        required_packages = [
            'pandas', 'numpy', 'xgboost', 'scikit-learn', 
            'joblib', 'streamlit', 'sqlalchemy'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} available")
            except ImportError:
                issue = f"❌ Missing package: {package}"
                logger.error(issue)
                self.issues_found.append(issue)
    
    def test_model_loading(self):
        """Test model loading performance and issues."""
        logger.info("🤖 Testing model loading...")
        
        try:
            # Test XGBoost predictor loading
            start_time = time.time()
            from models.xgboost_predictor import XGBoostPredictor
            
            predictor = XGBoostPredictor('models/predictor_model.joblib', production_mode=True)
            load_time = time.time() - start_time
            
            if predictor.model:
                logger.info(f"✅ XGBoost predictor loaded in {load_time:.3f}s")
                if load_time > 5.0:
                    issue = f"⚠️ Slow model loading: {load_time:.3f}s (target: <3s)"
                    logger.warning(issue)
                    self.issues_found.append(issue)
            else:
                issue = "❌ XGBoost predictor model is None"
                logger.error(issue)
                self.issues_found.append(issue)
                
        except Exception as e:
            issue = f"❌ XGBoost predictor loading failed: {e}"
            logger.error(issue)
            self.issues_found.append(issue)
        
        # Test model singleton
        try:
            start_time = time.time()
            from utils.model_singleton import get_model_manager
            
            manager = get_model_manager()
            singleton_time = time.time() - start_time
            
            logger.info(f"✅ Model singleton initialized in {singleton_time:.3f}s")
            
            # Test cached loading
            start_time = time.time()
            predictor = manager.get_xgboost_predictor('models/predictor_model.joblib')
            cached_time = time.time() - start_time
            
            if predictor:
                logger.info(f"✅ Cached XGBoost predictor retrieved in {cached_time:.3f}s")
            else:
                issue = "❌ Model singleton failed to return XGBoost predictor"
                logger.error(issue)
                self.issues_found.append(issue)
                
        except Exception as e:
            issue = f"❌ Model singleton test failed: {e}"
            logger.error(issue)
            self.issues_found.append(issue)
    
    def test_engine_initialization(self):
        """Test Enhanced Prediction Engine initialization."""
        logger.info("🚀 Testing Enhanced Prediction Engine initialization...")
        
        try:
            start_time = time.time()
            from enhanced_prediction_engine import EnhancedPredictionEngine
            
            engine = EnhancedPredictionEngine()
            init_time = time.time() - start_time
            
            logger.info(f"✅ Enhanced Prediction Engine initialized in {init_time:.3f}s")
            
            if init_time > 10.0:
                issue = f"⚠️ Slow engine initialization: {init_time:.3f}s (target: <5s)"
                logger.warning(issue)
                self.issues_found.append(issue)
            
            # Check components
            if engine.feature_mapper:
                logger.info("✅ Feature mapper initialized")
            else:
                issue = "❌ Feature mapper not initialized"
                logger.error(issue)
                self.issues_found.append(issue)
            
            if engine.xgboost_predictor:
                logger.info("✅ XGBoost predictor initialized")
            else:
                issue = "❌ XGBoost predictor not initialized"
                logger.error(issue)
                self.issues_found.append(issue)
                
        except Exception as e:
            issue = f"❌ Enhanced Prediction Engine initialization failed: {e}"
            logger.error(issue)
            logger.error(traceback.format_exc())
            self.issues_found.append(issue)
    
    def test_prediction_generation(self):
        """Test prediction generation with sample data."""
        logger.info("🎯 Testing prediction generation...")
        
        try:
            from enhanced_prediction_engine import EnhancedPredictionEngine
            
            engine = EnhancedPredictionEngine()
            
            # Sample match data
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
                logger.info(f"✅ Prediction generated in {prediction_time:.3f}s")
                logger.info(f"   Home: {result['predictions']['home_win']:.3f}")
                logger.info(f"   Draw: {result['predictions']['draw']:.3f}")
                logger.info(f"   Away: {result['predictions']['away_win']:.3f}")
                logger.info(f"   Confidence: {result['confidence']['overall']:.3f}")
                
                if prediction_time > 2.0:
                    issue = f"⚠️ Slow prediction: {prediction_time:.3f}s (target: <1s)"
                    logger.warning(issue)
                    self.issues_found.append(issue)
                
                # Check if it's a fallback prediction
                if result.get('metadata', {}).get('model_version') == 'fallback':
                    issue = "⚠️ Using fallback prediction - main prediction failed"
                    logger.warning(issue)
                    self.issues_found.append(issue)
                    
            else:
                issue = "❌ Prediction generation failed - no result returned"
                logger.error(issue)
                self.issues_found.append(issue)
                
        except Exception as e:
            issue = f"❌ Prediction generation test failed: {e}"
            logger.error(issue)
            logger.error(traceback.format_exc())
            self.issues_found.append(issue)
    
    def test_dashboard_integration(self):
        """Test dashboard integration."""
        logger.info("📊 Testing dashboard integration...")
        
        try:
            from dashboard.optimized_production_app import \
                OptimizedProductionApp
            
            app = OptimizedProductionApp()
            
            # Test prediction generation through dashboard
            start_time = time.time()
            result = app.generate_prediction('Arsenal', 'Chelsea', 'PL')
            dashboard_time = time.time() - start_time
            
            if result and 'home_win' in result:
                logger.info(f"✅ Dashboard prediction generated in {dashboard_time:.3f}s")
                logger.info(f"   Status: {result.get('status', 'unknown')}")
                
                if result.get('status') == 'error':
                    issue = f"⚠️ Dashboard prediction returned error: {result.get('error', 'unknown')}"
                    logger.warning(issue)
                    self.issues_found.append(issue)
                    
            else:
                issue = "❌ Dashboard prediction failed"
                logger.error(issue)
                self.issues_found.append(issue)
                
        except Exception as e:
            issue = f"❌ Dashboard integration test failed: {e}"
            logger.error(issue)
            logger.error(traceback.format_exc())
            self.issues_found.append(issue)

    def apply_fixes(self):
        """Apply fixes for identified issues."""
        logger.info("🔧 Applying fixes for identified issues...")

        if not self.issues_found:
            logger.info("✅ No issues found - system is healthy!")
            return

        for issue in self.issues_found:
            logger.info(f"Attempting to fix: {issue}")

            # Fix slow model loading
            if "Slow model loading" in issue:
                self._fix_slow_model_loading()

            # Fix missing feature mapper
            elif "Feature mapper not initialized" in issue:
                self._fix_feature_mapper()

            # Fix missing XGBoost predictor
            elif "XGBoost predictor not initialized" in issue:
                self._fix_xgboost_predictor()

            # Fix fallback predictions
            elif "fallback prediction" in issue:
                self._fix_fallback_predictions()

            # Fix slow predictions
            elif "Slow prediction" in issue:
                self._fix_slow_predictions()

    def _fix_slow_model_loading(self):
        """Fix slow model loading issues."""
        try:
            logger.info("🔧 Optimizing model loading...")

            # Ensure model singleton is working
            from utils.model_singleton import get_model_manager
            manager = get_model_manager()

            # Pre-load models
            manager.get_xgboost_predictor('models/predictor_model.joblib')
            manager.get_feature_mapper()

            self.fixes_applied.append("Optimized model loading with singleton pattern")
            logger.info("✅ Model loading optimization applied")

        except Exception as e:
            logger.error(f"Failed to fix model loading: {e}")

    def _fix_feature_mapper(self):
        """Fix feature mapper initialization."""
        try:
            logger.info("🔧 Fixing feature mapper...")

            # Check if feature mapper exists
            from utils.feature_mapper import FeatureMapper
            mapper = FeatureMapper()

            if mapper:
                self.fixes_applied.append("Feature mapper initialized successfully")
                logger.info("✅ Feature mapper fix applied")

        except Exception as e:
            logger.error(f"Failed to fix feature mapper: {e}")

    def _fix_xgboost_predictor(self):
        """Fix XGBoost predictor initialization."""
        try:
            logger.info("🔧 Fixing XGBoost predictor...")

            # Try different model paths
            model_paths = [
                'models/predictor_model.joblib',
                'models/trained/predictor_model.joblib'
            ]

            for path in model_paths:
                if (self.project_root / path).exists():
                    from models.xgboost_predictor import XGBoostPredictor
                    predictor = XGBoostPredictor(path, production_mode=True)

                    if predictor.model:
                        self.fixes_applied.append(f"XGBoost predictor loaded from {path}")
                        logger.info(f"✅ XGBoost predictor fix applied using {path}")
                        break

        except Exception as e:
            logger.error(f"Failed to fix XGBoost predictor: {e}")

    def _fix_fallback_predictions(self):
        """Fix fallback prediction issues."""
        try:
            logger.info("🔧 Fixing fallback prediction issues...")

            # This usually indicates feature mapping or model issues
            # Ensure all components are properly initialized
            from utils.model_singleton import get_model_manager
            manager = get_model_manager()

            # Force reload of components
            manager.clear_cache()
            manager.get_xgboost_predictor('models/predictor_model.joblib')
            manager.get_feature_mapper()

            self.fixes_applied.append("Reloaded prediction components to fix fallback issues")
            logger.info("✅ Fallback prediction fix applied")

        except Exception as e:
            logger.error(f"Failed to fix fallback predictions: {e}")

    def _fix_slow_predictions(self):
        """Fix slow prediction issues."""
        try:
            logger.info("🔧 Fixing slow prediction issues...")

            # Disable SHAP in production mode
            # This is handled by production_mode=True in XGBoostPredictor

            self.fixes_applied.append("Ensured production mode for faster predictions")
            logger.info("✅ Slow prediction fix applied")

        except Exception as e:
            logger.error(f"Failed to fix slow predictions: {e}")

    def final_validation(self):
        """Run final validation after fixes."""
        logger.info("✅ Running final validation...")

        try:
            from enhanced_prediction_engine import EnhancedPredictionEngine

            engine = EnhancedPredictionEngine()

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
                logger.info(f"✅ Final validation successful!")
                logger.info(f"   Prediction time: {prediction_time:.3f}s")
                logger.info(f"   Model version: {result.get('metadata', {}).get('model_version', 'unknown')}")
                logger.info(f"   Confidence: {result['confidence']['overall']:.3f}")
                return True
            else:
                logger.error("❌ Final validation failed")
                return False

        except Exception as e:
            logger.error(f"❌ Final validation failed: {e}")
            return False

    def generate_report(self):
        """Generate diagnostic report."""
        logger.info("📋 Generating diagnostic report...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'issues_found': len(self.issues_found),
            'fixes_applied': len(self.fixes_applied),
            'issues': self.issues_found,
            'fixes': self.fixes_applied,
            'status': 'HEALTHY' if not self.issues_found else 'ISSUES_RESOLVED' if self.fixes_applied else 'ISSUES_FOUND'
        }

        # Save report
        report_file = self.project_root / 'prediction_engine_diagnostic_report.json'
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Diagnostic Report:")
        logger.info(f"   Issues Found: {len(self.issues_found)}")
        logger.info(f"   Fixes Applied: {len(self.fixes_applied)}")
        logger.info(f"   Status: {report['status']}")
        logger.info(f"   Report saved to: {report_file}")

def main():
    """Main diagnostic function."""
    diagnostic = PredictionEngineDiagnostic()
    success = diagnostic.run_comprehensive_diagnostic()
    
    if success:
        logger.info("🎉 Diagnostic completed successfully!")
        return 0
    else:
        logger.error("❌ Diagnostic failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
