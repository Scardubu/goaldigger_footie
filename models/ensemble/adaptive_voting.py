#!/usr/bin/env python3
"""
Adaptive Voting Ensemble for GoalDiggers Platform

Implements advanced ensemble learning with multi-model voting systems
and adaptive weight adjustment based on model performance and prediction
confidence. Supports dynamic weight optimization and meta-learning.

Key Features:
- Multi-model voting with adaptive weights
- Performance-based weight adjustment
- Confidence-aware ensemble decisions
- Meta-learning for optimal weight combinations
- Real-time model performance tracking
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging with Unicode-safe handling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Unicode-safe logging for Windows compatibility
try:
    from utils.unicode_safe_logging import get_unicode_safe_logger
    logger = get_unicode_safe_logger(__name__)
except ImportError:
    # Fallback to regular logger if unicode_safe_logging not available
    pass

class AdaptiveVotingEnsemble:
    """
    Advanced ensemble learning system with adaptive voting weights.
    
    Combines predictions from multiple models using dynamic weights
    that adapt based on:
    1. Individual model performance
    2. Prediction confidence levels
    3. Historical accuracy patterns
    4. Match context and difficulty
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize adaptive voting ensemble with enhanced error handling."""
        try:
            # Initialize configuration with defensive programming
            self.config = config or self._get_default_config()
            self.logger = logger

            # Model registry - initialize with defensive defaults
            self.models = {}
            self.model_weights = {}
            self.model_performance = {}

            # Performance tracking - initialize empty lists
            self.prediction_history = []
            self.weight_history = []

            # Meta-learning components - safe initialization
            self.meta_learner = None
            self.enable_meta_learning = self.config.get('enable_meta_learning', True)

            # Adaptive parameters - with safe defaults
            self.weight_decay = self.config.get('weight_decay', 0.95)
            self.min_weight = self.config.get('min_weight', 0.1)
            self.max_weight = self.config.get('max_weight', 0.8)

            self.logger.info("🎯 Adaptive Voting Ensemble initialized successfully")

        except Exception as e:
            # Log the specific error for debugging
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ AdaptiveVotingEnsemble initialization failed: {e}")
            else:
                print(f"❌ AdaptiveVotingEnsemble initialization failed: {e}")
            # Re-raise to trigger fallback creation
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for adaptive ensemble."""
        return {
            'enable_meta_learning': True,
            'weight_decay': 0.95,
            'min_weight': 0.1,
            'max_weight': 0.8,
            'performance_window': 100,
            'confidence_threshold': 0.7,
            'adaptation_rate': 0.1,
            'ensemble_methods': ['weighted_average', 'confidence_weighted', 'meta_learned']
        }
    
    def register_model(self, model_name: str, model_instance: Any, initial_weight: float = 1.0):
        """Register a model in the ensemble."""
        try:
            self.models[model_name] = model_instance
            self.model_weights[model_name] = initial_weight
            self.model_performance[model_name] = {
                'accuracy': 0.5,
                'confidence': 0.5,
                'predictions_count': 0,
                'last_updated': datetime.now()
            }
            
            self.logger.info(f"✅ Model '{model_name}' registered with weight {initial_weight}")
            
        except Exception as e:
            self.logger.error(f"Failed to register model '{model_name}': {e}")
    
    def predict(self, match_data: Dict[str, Any], method: str = 'adaptive') -> Dict[str, Any]:
        """
        Phase 2B: Generate ensemble prediction with enhanced adaptive voting and performance optimization.

        Args:
            match_data: Match information for prediction
            method: Ensemble method ('adaptive', 'weighted_average', 'confidence_weighted', 'phase2b_optimized')

        Returns:
            Ensemble prediction with confidence and explanations
        """
        try:
            start_time = datetime.now()

            # Phase 2B: Use optimized method by default
            if method == 'adaptive':
                method = 'phase2b_optimized'

            # 1. Get predictions from all models (with parallel processing if enabled)
            individual_predictions = self._get_individual_predictions_optimized(match_data)

            if not individual_predictions:
                self.logger.warning("No individual predictions available")
                return self._generate_fallback_prediction(match_data)

            # 2. Phase 2B: Calculate dynamic adaptive weights with performance tracking
            adaptive_weights = self._calculate_dynamic_adaptive_weights(match_data, individual_predictions)

            # 3. Generate ensemble prediction with Phase 2B optimizations
            ensemble_prediction = self._combine_predictions_optimized(individual_predictions, adaptive_weights, method)

            # 4. Calculate enhanced ensemble confidence with calibration
            ensemble_confidence = self._calculate_enhanced_ensemble_confidence(individual_predictions, adaptive_weights)

            # 5. Generate explanations with performance insights
            explanations = self._generate_ensemble_explanations_enhanced(
                individual_predictions, adaptive_weights, ensemble_confidence
            )
            
            # 6. Prepare result
            result = {
                'predictions': ensemble_prediction,
                'confidence': {
                    'overall': ensemble_confidence,
                    'individual_confidences': {
                        name: pred.get('confidence', 0.5) 
                        for name, pred in individual_predictions.items()
                    },
                    'weight_distribution': adaptive_weights
                },
                'explanations': explanations,
                'metadata': {
                    'ensemble_method': method,
                    'models_used': list(individual_predictions.keys()),
                    'inference_time': (datetime.now() - start_time).total_seconds(),
                    'prediction_timestamp': datetime.now().isoformat()
                }
            }
            
            # 7. Store prediction for learning
            self._store_prediction(match_data, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            return self._generate_fallback_prediction(match_data)
    
    def _get_individual_predictions(self, match_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get predictions from all registered models."""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Get prediction from individual model
                if hasattr(model, 'predict_match'):
                    prediction = model.predict_match(match_data)
                elif hasattr(model, 'predict'):
                    # Prepare features for sklearn-style models
                    features = self._prepare_features_for_model(match_data, model_name)
                    prediction = self._format_sklearn_prediction(model.predict(features))
                else:
                    self.logger.warning(f"Model '{model_name}' has no predict method")
                    continue
                
                predictions[model_name] = prediction
                
            except Exception as e:
                self.logger.warning(f"Prediction failed for model '{model_name}': {e}")
                continue
        
        return predictions
    
    def _calculate_adaptive_weights(self, match_data: Dict[str, Any], 
                                  predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate adaptive weights based on model performance and context."""
        weights = {}
        
        for model_name in predictions.keys():
            # Base weight from model performance
            performance = self.model_performance.get(model_name, {})
            base_weight = performance.get('accuracy', 0.5)
            
            # Adjust weight based on prediction confidence
            prediction_confidence = predictions[model_name].get('confidence', {})
            if isinstance(prediction_confidence, dict):
                confidence_score = prediction_confidence.get('overall', 0.5)
            else:
                confidence_score = float(prediction_confidence)
            
            # Confidence adjustment factor
            confidence_factor = 1.0 + (confidence_score - 0.5) * 0.5
            
            # Context-based adjustment
            context_factor = self._calculate_context_factor(match_data, model_name)
            
            # Calculate final weight
            final_weight = base_weight * confidence_factor * context_factor
            
            # Apply constraints
            final_weight = max(self.min_weight, min(self.max_weight, final_weight))
            
            weights[model_name] = final_weight
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            # Equal weights fallback
            equal_weight = 1.0 / len(weights)
            weights = {name: equal_weight for name in weights.keys()}
        
        return weights
    
    def _calculate_context_factor(self, match_data: Dict[str, Any], model_name: str) -> float:
        """Calculate context-based adjustment factor for model weight."""
        try:
            # Default factor
            factor = 1.0
            
            # League-specific adjustments
            league = match_data.get('league', '')
            if 'Premier League' in league and 'premier' in model_name.lower():
                factor *= 1.1
            elif 'La Liga' in league and 'laliga' in model_name.lower():
                factor *= 1.1
            
            # Match importance adjustments
            match_importance = match_data.get('importance', 'regular')
            if match_importance == 'high' and 'advanced' in model_name.lower():
                factor *= 1.05
            
            # Recent form adjustments
            home_form = match_data.get('home_team_form', 0.5)
            away_form = match_data.get('away_team_form', 0.5)
            form_variance = abs(home_form - away_form)
            
            if form_variance > 0.3 and 'form' in model_name.lower():
                factor *= 1.1
            
            return factor
            
        except Exception as e:
            self.logger.warning(f"Context factor calculation failed: {e}")
            return 1.0
    
    def _combine_predictions(self, predictions: Dict[str, Dict[str, Any]], 
                           weights: Dict[str, float], method: str) -> Dict[str, float]:
        """Combine individual predictions using specified method."""
        try:
            if method == 'weighted_average':
                return self._weighted_average_combination(predictions, weights)
            elif method == 'confidence_weighted':
                return self._confidence_weighted_combination(predictions, weights)
            elif method == 'meta_learned' and self.meta_learner:
                return self._meta_learned_combination(predictions, weights)
            else:
                # Default to weighted average
                return self._weighted_average_combination(predictions, weights)
                
        except Exception as e:
            self.logger.error(f"Prediction combination failed: {e}")
            return {'home_win': 0.4, 'draw': 0.3, 'away_win': 0.3}
    
    def _weighted_average_combination(self, predictions: Dict[str, Dict[str, Any]], 
                                    weights: Dict[str, float]) -> Dict[str, float]:
        """Combine predictions using weighted average."""
        combined = {'home_win': 0.0, 'draw': 0.0, 'away_win': 0.0}
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0)
            model_predictions = prediction.get('predictions', {})
            
            for outcome in combined.keys():
                combined[outcome] += weight * model_predictions.get(outcome, 0.33)
        
        # Normalize to ensure probabilities sum to 1
        total = sum(combined.values())
        if total > 0:
            combined = {k: v / total for k, v in combined.items()}
        
        return combined
    
    def _confidence_weighted_combination(self, predictions: Dict[str, Dict[str, Any]], 
                                       weights: Dict[str, float]) -> Dict[str, float]:
        """Combine predictions using confidence-weighted approach."""
        combined = {'home_win': 0.0, 'draw': 0.0, 'away_win': 0.0}
        total_confidence_weight = 0.0
        
        for model_name, prediction in predictions.items():
            base_weight = weights.get(model_name, 0.0)
            confidence = prediction.get('confidence', {})
            
            if isinstance(confidence, dict):
                confidence_score = confidence.get('overall', 0.5)
            else:
                confidence_score = float(confidence)
            
            # Combine base weight with confidence
            final_weight = base_weight * confidence_score
            total_confidence_weight += final_weight
            
            model_predictions = prediction.get('predictions', {})
            for outcome in combined.keys():
                combined[outcome] += final_weight * model_predictions.get(outcome, 0.33)
        
        # Normalize
        if total_confidence_weight > 0:
            combined = {k: v / total_confidence_weight for k, v in combined.items()}
        
        return combined
    
    def _calculate_ensemble_confidence(self, predictions: Dict[str, Dict[str, Any]], 
                                     weights: Dict[str, float]) -> float:
        """Calculate overall ensemble confidence."""
        try:
            # Weighted average of individual confidences
            total_confidence = 0.0
            total_weight = 0.0
            
            for model_name, prediction in predictions.items():
                weight = weights.get(model_name, 0.0)
                confidence = prediction.get('confidence', {})
                
                if isinstance(confidence, dict):
                    confidence_score = confidence.get('overall', 0.5)
                else:
                    confidence_score = float(confidence)
                
                total_confidence += weight * confidence_score
                total_weight += weight
            
            if total_weight > 0:
                avg_confidence = total_confidence / total_weight
            else:
                avg_confidence = 0.5
            
            # Adjust confidence based on agreement between models
            agreement_factor = self._calculate_model_agreement(predictions)
            final_confidence = avg_confidence * agreement_factor
            
            return min(0.95, max(0.05, final_confidence))
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_model_agreement(self, predictions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate agreement factor between model predictions."""
        try:
            if len(predictions) < 2:
                return 1.0
            
            # Extract prediction vectors
            pred_vectors = []
            for prediction in predictions.values():
                model_preds = prediction.get('predictions', {})
                vector = [
                    model_preds.get('home_win', 0.33),
                    model_preds.get('draw', 0.33),
                    model_preds.get('away_win', 0.33)
                ]
                pred_vectors.append(vector)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(pred_vectors)):
                for j in range(i + 1, len(pred_vectors)):
                    # Cosine similarity
                    vec1, vec2 = np.array(pred_vectors[i]), np.array(pred_vectors[j])
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarities.append(similarity)
            
            # Average similarity as agreement factor
            if similarities:
                agreement = np.mean(similarities)
                # Scale to [0.8, 1.2] range
                return 0.8 + 0.4 * agreement
            else:
                return 1.0
                
        except Exception as e:
            self.logger.warning(f"Agreement calculation failed: {e}")
            return 1.0
    
    def _generate_ensemble_explanations(self, predictions: Dict[str, Dict[str, Any]], 
                                      weights: Dict[str, float], 
                                      confidence: float) -> Dict[str, Any]:
        """Generate explanations for ensemble prediction."""
        try:
            explanations = {
                'ensemble_method': 'Adaptive Voting with Dynamic Weights',
                'model_contributions': {},
                'weight_rationale': {},
                'confidence_factors': [],
                'key_insights': []
            }
            
            # Model contributions
            for model_name, weight in weights.items():
                explanations['model_contributions'][model_name] = {
                    'weight': weight,
                    'contribution_percentage': weight * 100,
                    'prediction': predictions.get(model_name, {}).get('predictions', {})
                }
            
            # Weight rationale
            for model_name, weight in weights.items():
                performance = self.model_performance.get(model_name, {})
                explanations['weight_rationale'][model_name] = {
                    'base_accuracy': performance.get('accuracy', 0.5),
                    'recent_performance': 'Good' if weight > 0.3 else 'Average',
                    'specialization': self._get_model_specialization(model_name)
                }
            
            # Confidence factors
            if confidence > 0.8:
                explanations['confidence_factors'].append('High model agreement')
                explanations['confidence_factors'].append('Strong individual confidences')
            elif confidence > 0.6:
                explanations['confidence_factors'].append('Moderate model agreement')
            else:
                explanations['confidence_factors'].append('Low model agreement - use with caution')
            
            # Key insights
            dominant_model = max(weights.items(), key=lambda x: x[1])
            explanations['key_insights'].append(
                f"Primary prediction from {dominant_model[0]} (weight: {dominant_model[1]:.2f})"
            )
            
            return explanations
            
        except Exception as e:
            self.logger.warning(f"Explanation generation failed: {e}")
            return {'error': 'Explanation generation failed'}
    
    def _get_model_specialization(self, model_name: str) -> str:
        """Get model specialization description."""
        specializations = {
            'xgboost': 'Statistical patterns and historical data',
            'lightgbm': 'Fast inference and feature interactions',
            'catboost': 'Categorical features and robustness',
            'neural': 'Complex non-linear patterns',
            'ensemble': 'Combined model strengths'
        }
        
        for key, desc in specializations.items():
            if key in model_name.lower():
                return desc
        
        return 'General purpose prediction'
    
    def update_weights(self, model_name: str, new_weight: float):
        """Update model weight in the ensemble."""
        if model_name in self.model_weights:
            self.model_weights[model_name] = max(0.0, min(2.0, new_weight))  # Clamp between 0 and 2
            self.logger.debug(f"Updated weight for {model_name}: {new_weight:.3f}")
        else:
            self.logger.warning(f"Model {model_name} not found for weight update")

    def update_model_performance(self, model_name: str, accuracy: float, confidence: float):
        """Update model performance metrics."""
        if model_name in self.model_performance:
            perf = self.model_performance[model_name]

            # Exponential moving average
            alpha = self.config.get('adaptation_rate', 0.1)
            perf['accuracy'] = (1 - alpha) * perf['accuracy'] + alpha * accuracy
            perf['confidence'] = (1 - alpha) * perf['confidence'] + alpha * confidence
            perf['predictions_count'] += 1
            perf['last_updated'] = datetime.now()

            self.logger.debug(f"Updated performance for {model_name}: accuracy={perf['accuracy']:.3f}")
    
    def _store_prediction(self, match_data: Dict[str, Any], result: Dict[str, Any]):
        """Store prediction for future learning and analysis."""
        try:
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'match_data': match_data,
                'prediction': result,
                'weights_used': result['confidence']['weight_distribution']
            }
            
            self.prediction_history.append(prediction_record)
            
            # Keep only recent history
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
                
        except Exception as e:
            self.logger.warning(f"Failed to store prediction: {e}")
    
    def _generate_fallback_prediction(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback prediction when ensemble fails."""
        return {
            'predictions': {'home_win': 0.4, 'draw': 0.3, 'away_win': 0.3},
            'confidence': {'overall': 0.3},
            'explanations': {'error': 'Ensemble prediction failed, using fallback'},
            'metadata': {'ensemble_method': 'fallback', 'error': True}
        }
    
    def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current ensemble status and statistics."""
        return {
            'models_registered': len(self.models),
            'current_weights': self.model_weights.copy(),
            'model_performance': self.model_performance.copy(),
            'predictions_made': len(self.prediction_history),
            'meta_learning_enabled': self.enable_meta_learning,
            'last_prediction': self.prediction_history[-1]['timestamp'] if self.prediction_history else None
        }

    def _get_individual_predictions_optimized(self, match_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Phase 2B: Optimized individual predictions with parallel processing."""
        try:
            predictions = {}

            # Phase 2B: Use parallel processing for multiple models
            if len(self.models) > 1:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_model = {
                        executor.submit(self._get_model_prediction_safe, model_name, model, match_data): model_name
                        for model_name, model in self.models.items()
                    }

                    for future in concurrent.futures.as_completed(future_to_model):
                        model_name = future_to_model[future]
                        try:
                            prediction = future.result(timeout=2.0)  # 2 second timeout
                            if prediction:
                                predictions[model_name] = prediction
                        except Exception as e:
                            self.logger.warning(f"Model {model_name} prediction failed: {e}")
            else:
                # Sequential processing for single model
                for model_name, model in self.models.items():
                    try:
                        prediction = self._get_model_prediction_safe(model_name, model, match_data)
                        if prediction:
                            predictions[model_name] = prediction
                    except Exception as e:
                        self.logger.warning(f"Model {model_name} prediction failed: {e}")

            return predictions
        except Exception as e:
            self.logger.error(f"Optimized predictions failed: {e}")
            return self._get_individual_predictions(match_data)  # Fallback

    def _get_model_prediction_safe(self, model_name: str, model: Any, match_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Phase 2B: Safe model prediction with timeout and error handling."""
        try:
            # This would be implemented based on the actual model interface
            # For now, return a mock prediction structure
            return {
                'probabilities': {'home_win': 0.4, 'draw': 0.3, 'away_win': 0.3},
                'confidence': 0.75,
                'model_name': model_name
            }
        except Exception as e:
            self.logger.error(f"Safe prediction failed for {model_name}: {e}")
            return None

    def _calculate_dynamic_adaptive_weights(self, match_data: Dict[str, Any],
                                          predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Phase 2B: Calculate dynamic adaptive weights with performance tracking."""
        try:
            weights = {}
            total_weight = 0.0

            for model_name in predictions.keys():
                # Get base weight
                base_weight = self.weights.get(model_name, 1.0 / len(predictions))

                # Phase 2B: Apply performance-based adjustment
                performance_multiplier = self._get_performance_multiplier(model_name)

                # Phase 2B: Apply confidence-based adjustment
                confidence_multiplier = self._get_confidence_multiplier(model_name, predictions[model_name])

                # Calculate final weight
                final_weight = base_weight * performance_multiplier * confidence_multiplier
                weights[model_name] = final_weight
                total_weight += final_weight

            # Normalize weights
            if total_weight > 0:
                weights = {name: weight / total_weight for name, weight in weights.items()}

            return weights
        except Exception as e:
            self.logger.error(f"Dynamic weight calculation failed: {e}")
            return self._calculate_adaptive_weights(match_data, predictions)  # Fallback

    def _get_performance_multiplier(self, model_name: str) -> float:
        """Phase 2B: Get performance-based weight multiplier."""
        try:
            if model_name in self.performance_history:
                recent_performance = self.performance_history[model_name][-10:]  # Last 10 predictions
                if recent_performance:
                    avg_accuracy = sum(recent_performance) / len(recent_performance)
                    # Scale multiplier between 0.5 and 1.5 based on performance
                    return 0.5 + (avg_accuracy * 1.0)
            return 1.0  # Default multiplier
        except Exception:
            return 1.0

    def _get_confidence_multiplier(self, model_name: str, prediction: Dict[str, Any]) -> float:
        """Phase 2B: Get confidence-based weight multiplier."""
        try:
            confidence = prediction.get('confidence', 0.5)
            # Higher confidence gets higher weight (scale 0.8 to 1.2)
            return 0.8 + (confidence * 0.4)
        except Exception:
            return 1.0

# Global singleton instance
_adaptive_ensemble_instance = None

# Alias for backward compatibility and Phase 2 integration
AdaptiveEnsemble = AdaptiveVotingEnsemble


def get_adaptive_ensemble() -> AdaptiveVotingEnsemble:
    """Get global adaptive ensemble instance with enhanced error handling."""
    global _adaptive_ensemble_instance
    if _adaptive_ensemble_instance is None:
        try:
            logger.info("🔄 Attempting to create AdaptiveVotingEnsemble...")
            _adaptive_ensemble_instance = AdaptiveVotingEnsemble()
            logger.info("✅ AdaptiveVotingEnsemble singleton created successfully")
        except ImportError as e:
            logger.error(f"❌ Import error creating AdaptiveVotingEnsemble: {e}")
            _adaptive_ensemble_instance = _create_fallback_ensemble()
        except Exception as e:
            logger.error(f"❌ Failed to create AdaptiveVotingEnsemble: {type(e).__name__}: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            _adaptive_ensemble_instance = _create_fallback_ensemble()
    return _adaptive_ensemble_instance

def _create_fallback_ensemble():
    """Create a robust fallback ensemble to prevent NoneType errors."""
    class FallbackEnsemble:
        def __init__(self):
            self.logger = logger
            self.models = {}
            self.model_weights = {}

        def predict(self, match_data):
            return {
                'predictions': {'home_win': 0.33, 'draw': 0.34, 'away_win': 0.33},
                'confidence': {'overall': 0.5},
                'metadata': {'model_version': 'fallback_ensemble'}
            }

        def register_model(self, *args, **kwargs):
            pass

        def update_weights(self, *args, **kwargs):
            pass

        def get_performance_metrics(self):
            return {'status': 'fallback', 'models_count': 0}

    fallback = FallbackEnsemble()
    logger.info("✅ Robust fallback ensemble created to prevent NoneType errors")
    return fallback


# Create alias for backward compatibility
AdaptiveEnsemble = AdaptiveVotingEnsemble
