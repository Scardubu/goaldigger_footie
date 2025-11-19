#!/usr/bin/env python3
"""
Run a quick validator integration check without pytest.
Exits with code 0 on success, 1 on failure.
"""
import os
import sys

# Add parent to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from batched_prediction_engine import BatchedPredictionEngine, PredictionRequest
from utils.real_data_validator import get_real_data_validator


def main():
    try:
        engine = BatchedPredictionEngine(max_workers=1)

        # Monkeypatch feature generator to return features indicating real data present
        def fake_generate_vectorized_features(self, match_data):
            return {
                'real_data_used': 1.0,
                'real_data_historic_form': 1.0,
                'real_data_head_to_head': 1.0,
                'real_data_league_table': 1.0,
                'expected_goals_home': 1.2,
                'real_data_timestamp_epoch':  ( __import__('time').time() ),
            }

        engine.feature_generator.generate_vectorized_features = fake_generate_vectorized_features.__get__(engine.feature_generator, engine.feature_generator.__class__)

        validator = get_real_data_validator()
        validator.min_quality_score = 0.1

        req = PredictionRequest(home_team='TestHome', away_team='TestAway', league='TestLeague', request_id='r1')
        results = engine.predict_batch([req])
        if not results or len(results) != 1:
            print('FAILED: no results')
            return 1
        res = results[0]
        if not hasattr(res, 'validation_report') or res.validation_report is None:
            print('FAILED: validation_report missing')
            return 1
        vr = res.validation_report
        if not (vr.get('valid') or vr.get('recommendation') in ('PUBLISH_ALLOWED','PUBLISH_WITH_WARNING')):
            print('FAILED: validation not passing; report:', vr)
            return 1
        print('OK: validation_report present and acceptable')
        return 0
    except Exception as e:
        print('ERROR during run:', e)
        return 2

if __name__ == '__main__':
    code = main()
    sys.exit(code)
