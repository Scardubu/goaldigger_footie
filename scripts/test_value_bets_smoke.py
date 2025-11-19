# Smoke test for value_bets plotting functions
import json

try:
    from dashboard.visualizations import value_bets as vb
    sample = [
        {"bet_type": "home_win", "odds": 2.5, "predicted_prob": 0.45, "kelly_stake": 0.02, "confidence": "High"},
        {"bet_type": "draw", "odds": 3.2, "predicted_prob": 0.2, "kelly_stake": 0.01, "confidence": "Low"},
    ]
    fig1 = vb.plot_value_bet_edge(sample)
    fig2 = vb.plot_kelly_stakes(sample)
    fig3 = vb.plot_probability_comparison(sample)
    print('SMOKE_OK', type(fig1).__name__, type(fig2).__name__, type(fig3).__name__)
except Exception as e:
    print('SMOKE_ERROR', repr(e))
    raise
