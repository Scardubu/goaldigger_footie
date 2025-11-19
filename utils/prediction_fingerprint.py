#!/usr/bin/env python3
"""
Prediction Fingerprinting Utility
Generate unique fingerprints for predictions to track consistency

Usage:
    from utils.prediction_fingerprint import get_prediction_fingerprint
    
    fingerprint = get_prediction_fingerprint(features, probabilities)
    logger.info(f"Prediction fingerprint: {fingerprint}")
"""
import hashlib
import json
from typing import Any, Dict

def get_prediction_fingerprint(
    features: Dict[str, Any],
    probabilities: Dict[str, float] = None,
    precision: int = 4
) -> str:
    """
    Generate deterministic fingerprint for prediction.
    
    Args:
        features: Input features used for prediction
        probabilities: Output probabilities (optional)
        precision: Decimal precision for rounding
    
    Returns:
        8-character hexadecimal fingerprint
    """
    # Select key features that should determine prediction
    key_features = {
        'home_strength': round(features.get('home_strength_adj', 0), precision),
        'away_strength': round(features.get('away_strength_adj', 0), precision),
        'form_diff': round(features.get('form_momentum', 0), precision),
        'h2h': round(features.get('head_to_head', 0), precision),
        'venue': round(features.get('venue_advantage', 0), precision),
        'data_quality': round(features.get('real_data_quality', 0), precision),
        'xg_diff': round(features.get('xg_differential', 0), precision),
    }
    
    # Add probabilities if provided
    if probabilities:
        key_features['home_prob'] = round(probabilities.get('home_win_prob', 0), precision)
        key_features['draw_prob'] = round(probabilities.get('draw_prob', 0), precision)
        key_features['away_prob'] = round(probabilities.get('away_win_prob', 0), precision)
    
    # Create deterministic JSON representation
    fingerprint_data = json.dumps(key_features, sort_keys=True)
    
    # Generate hash
    hash_obj = hashlib.md5(fingerprint_data.encode('utf-8'))
    return hash_obj.hexdigest()[:8]

def compare_predictions(
    pred1: Dict[str, Any],
    pred2: Dict[str, Any],
    tolerance: float = 0.001
) -> Dict[str, Any]:
    """
    Compare two predictions and identify differences.
    
    Args:
        pred1: First prediction
        pred2: Second prediction  
        tolerance: Acceptable difference threshold
    
    Returns:
        Comparison results with differences highlighted
    """
    differences = {}
    
    # Compare probabilities
    for key in ['home_win_prob', 'draw_prob', 'away_win_prob']:
        if key in pred1 and key in pred2:
            diff = abs(pred1[key] - pred2[key])
            if diff > tolerance:
                differences[key] = {
                    'pred1': pred1[key],
                    'pred2': pred2[key],
                    'diff': diff
                }
    
    # Compare confidence
    if 'confidence' in pred1 and 'confidence' in pred2:
        diff = abs(pred1['confidence'] - pred2['confidence'])
        if diff > tolerance:
            differences['confidence'] = {
                'pred1': pred1['confidence'],
                'pred2': pred2['confidence'],
                'diff': diff
            }
    
    return {
        'is_consistent': len(differences) == 0,
        'differences': differences,
        'max_diff': max([d['diff'] for d in differences.values()]) if differences else 0.0
    }

def validate_prediction_consistency(
    home_team: str,
    away_team: str,
    league: str,
    num_trials: int = 5,
    predictor = None
) -> Dict[str, Any]:
    """
    Validate that predictions are consistent across multiple calls.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        league: League code
        num_trials: Number of times to generate prediction
        predictor: Predictor instance (optional)
    
    Returns:
        Validation results
    """
    if predictor is None:
        from models.enhanced_real_data_predictor import EnhancedRealDataPredictor
        predictor = EnhancedRealDataPredictor()
    
    predictions = []
    fingerprints = []
    
    for i in range(num_trials):
        pred = predictor.predict_match_enhanced(
            home_team, away_team, 
            {'league': league}
        )
        
        pred_dict = {
            'home_win_prob': pred.home_win_probability,
            'draw_prob': pred.draw_probability,
            'away_win_prob': pred.away_win_probability,
            'confidence': pred.confidence
        }
        
        predictions.append(pred_dict)
        fingerprints.append(get_prediction_fingerprint({}, pred_dict))
    
    # Check consistency
    unique_fingerprints = len(set(fingerprints))
    is_consistent = unique_fingerprints == 1
    
    # Calculate variance
    home_probs = [p['home_win_prob'] for p in predictions]
    variance = max(home_probs) - min(home_probs)
    
    return {
        'is_consistent': is_consistent,
        'unique_fingerprints': unique_fingerprints,
        'total_trials': num_trials,
        'fingerprints': fingerprints,
        'max_variance': variance,
        'predictions': predictions
    }

if __name__ == "__main__":
    # Test consistency
    result = validate_prediction_consistency(
        "Arsenal", "Chelsea", "Premier League",
        num_trials=5
    )
    
    print(f"\nConsistency Test Results:")
    print(f"Consistent: {result['is_consistent']}")
    print(f"Unique fingerprints: {result['unique_fingerprints']}/{result['total_trials']}")
    print(f"Max variance: {result['max_variance']:.4f}")
    print(f"Fingerprints: {result['fingerprints']}")
