"""
Advanced betting insights generator combining real-time and historical data.
"""
import logging
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AdvancedBettingInsightsGenerator:
    """Generate advanced betting insights using ML predictions and market data."""
    
    def __init__(self):
        self.logger = logger
        
    def generate_insights(self, 
                        home_team: str, 
                        away_team: str, 
                        match_data: Dict,
                        odds_data: Dict,
                        risk_tolerance: str = 'medium') -> Dict:
        """Generate comprehensive betting insights."""
        
        # Get ML predictions
        predictions = self._get_match_predictions(home_team, away_team, match_data)
        
        # Calculate expected values
        ev_analysis = self._calculate_expected_values(predictions, odds_data)
        
        # Generate betting opportunities
        opportunities = self._identify_betting_opportunities(ev_analysis, risk_tolerance)
        
        # Risk assessment
        risk_assessment = self._assess_risk(predictions, odds_data, risk_tolerance)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(opportunities, risk_assessment)
        
        return {
            'predictions': predictions,
            'expected_values': ev_analysis,
            'betting_opportunities': opportunities,
            'risk_assessment': risk_assessment,
            'recommendations': recommendations
        }
        
    def _get_match_predictions(self, home_team: str, away_team: str, match_data: Dict) -> Dict:
        """Get ML predictions for match outcome."""
        # Mock predictions - would integrate with actual predictor
        return {
            'home_win_prob': 0.45,
            'draw_prob': 0.30,
            'away_win_prob': 0.25,
            'confidence': 0.78,
            'key_factors': ['Home advantage', 'Recent form', 'Head-to-head']
        }
        
    def _calculate_expected_values(self, predictions: Dict, odds_data: Dict) -> Dict:
        """Calculate expected values for different betting markets."""
        ev_data = {}
        
        # Match result markets
        home_odds = odds_data.get('home_win', 2.0)
        draw_odds = odds_data.get('draw', 3.2)
        away_odds = odds_data.get('away_win', 3.5)
        
        ev_data['home_win'] = {
            'odds': home_odds,
            'probability': predictions['home_win_prob'],
            'expected_value': (predictions['home_win_prob'] * home_odds) - 1,
            'market': 'Match Result - Home Win'
        }
        
        ev_data['draw'] = {
            'odds': draw_odds,
            'probability': predictions['draw_prob'],
            'expected_value': (predictions['draw_prob'] * draw_odds) - 1,
            'market': 'Match Result - Draw'
        }
        
        ev_data['away_win'] = {
            'odds': away_odds,
            'probability': predictions['away_win_prob'],
            'expected_value': (predictions['away_win_prob'] * away_odds) - 1,
            'market': 'Match Result - Away Win'
        }
        
        return ev_data
        
    def _identify_betting_opportunities(self, ev_data: Dict, risk_tolerance: str) -> List[Dict]:
        """Identify positive expected value betting opportunities."""
        opportunities = []
        
        for market, data in ev_data.items():
            ev = data['expected_value']
            
            # Only consider positive EV bets
            if ev > 0.05:  # 5% minimum EV threshold
                risk_level = self._determine_risk_level(ev, data['probability'])
                
                # Filter by risk tolerance
                if self._matches_risk_tolerance(risk_level, risk_tolerance):
                    opportunities.append({
                        'market': data['market'],
                        'odds': data['odds'],
                        'probability': data['probability'],
                        'expected_value': ev,
                        'confidence': min(0.95, data['probability'] + 0.1),
                        'risk_level': risk_level,
                        'stake_suggestion': self._suggest_stake(ev, risk_level)
                    })
                    
        # Sort by expected value
        opportunities.sort(key=lambda x: x['expected_value'], reverse=True)
        
        return opportunities[:5]  # Return top 5 opportunities
        
    def _determine_risk_level(self, ev: float, probability: float) -> str:
        """Determine risk level based on EV and probability."""
        if ev > 0.15 and probability > 0.4:
            return 'Low'
        elif ev > 0.08 and probability > 0.25:
            return 'Medium'
        else:
            return 'High'
            
    def _matches_risk_tolerance(self, risk_level: str, tolerance: str) -> bool:
        """Check if risk level matches user tolerance."""
        if tolerance == 'low':
            return risk_level == 'Low'
        elif tolerance == 'medium':
            return risk_level in ['Low', 'Medium']
        else:  # high tolerance
            return True
            
    def _suggest_stake(self, ev: float, risk_level: str) -> str:
        """Suggest stake size based on EV and risk."""
        if risk_level == 'Low':
            return f"{min(5, max(1, int(ev * 20)))} units"
        elif risk_level == 'Medium':
            return f"{min(3, max(1, int(ev * 15)))} units"
        else:
            return f"{min(2, max(1, int(ev * 10)))} units"
            
    def _assess_risk(self, predictions: Dict, odds_data: Dict, risk_tolerance: str) -> Dict:
        """Assess overall risk for the betting session."""
        confidence = predictions['confidence']
        
        # Determine risk category
        if confidence > 0.8:
            risk_category = 'Low Risk'
        elif confidence > 0.6:
            risk_category = 'Medium Risk'
        else:
            risk_category = 'High Risk'
            
        # Calculate recommended max stake
        base_stake = 10 if risk_tolerance == 'high' else 5 if risk_tolerance == 'medium' else 2
        max_stake = base_stake * confidence
        
        return {
            'risk_category': risk_category,
            'confidence_level': confidence,
            'recommended_max_stake': max_stake,
            'risk_factors': self._identify_risk_factors(predictions, odds_data)
        }
        
    def _identify_risk_factors(self, predictions: Dict, odds_data: Dict) -> List[str]:
        """Identify key risk factors."""
        factors = []
        
        if predictions['confidence'] < 0.7:
            factors.append('Low prediction confidence')
            
        if abs(predictions['home_win_prob'] - predictions['away_win_prob']) < 0.1:
            factors.append('Evenly matched teams')
            
        if any(odds < 1.5 for odds in odds_data.values()):
            factors.append('Very low odds indicate strong favorites')
            
        return factors
        
    def _generate_recommendations(self, opportunities: List[Dict], risk_assessment: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if opportunities:
            best_opp = opportunities[0]
            recommendations.append(f"🎯 Best opportunity: {best_opp['market']} at {best_opp['odds']:.2f} odds")
            recommendations.append(f"💰 Expected value: +{best_opp['expected_value']:.1%}")
            
        if risk_assessment['confidence_level'] > 0.8:
            recommendations.append("🔥 High confidence match - consider larger stakes")
        elif risk_assessment['confidence_level'] < 0.6:
            recommendations.append("⚠️ Low confidence - consider smaller stakes or skip")
            
        if len(opportunities) > 3:
            recommendations.append("💡 Multiple opportunities available - diversify your bets")
            
        recommendations.append(f"📊 Max recommended stake: {risk_assessment['recommended_max_stake']:.1f} units")
        
        return recommendations
