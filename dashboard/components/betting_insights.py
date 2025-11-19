"""
Betting insights component for the GoalDiggers dashboard.
Provides UI elements for displaying actionable betting insights.
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.ui_elements import badge, card
from dashboard.visualizations.betting_insights import \
    plot_match_outcome_probabilities
from scripts.betting_insights import BettingInsightsGenerator

logger = logging.getLogger(__name__)


def render_betting_insights(match_id: Optional[str] = None, team_names: Optional[List[str]] = None):
    """
    Render betting insights for a specific match or teams.
    
    Args:
        match_id: ID of the match to analyze
        team_names: List of team names to find matches for
    """
    st.header("🎯 Betting Insights")
    
    with st.spinner("Generating betting insights..."):
        try:
            # Create insights generator
            insights_generator = BettingInsightsGenerator()
            
            if match_id:
                # Get match details
                match = None
                with insights_generator.db.session_scope() as session:
                    from database.schema import Match, Team
                    match_obj = session.query(Match).filter_by(id=match_id).first()
                    if match_obj:
                        home_team = session.query(Team).filter_by(id=match_obj.home_team_id).first()
                        away_team = session.query(Team).filter_by(id=match_obj.away_team_id).first()
                        
                        if home_team and away_team:
                            match = {
                                'match_id': match_id,
                                'home_team': home_team.name,
                                'away_team': away_team.name,
                                'home_team_id': match_obj.home_team_id,
                                'away_team_id': match_obj.away_team_id,
                                'match_date': match_obj.match_date.isoformat() if match_obj.match_date else datetime.now().isoformat(),
                                'league': match_obj.league_id
                            }
                
                if not match:
                    st.error(f"Match with ID {match_id} not found")
                    return
                    
                # Generate prediction and insights
                with st.spinner("Running model and generating insights..."):
                    prediction = insights_generator.predict_match(match)
                    insights = insights_generator.generate_insights(prediction)
                
                # Display insights
                _display_single_match_insights(insights, prediction)
                
            elif team_names:
                # Generate insights for selected teams
                results = insights_generator.analyze_user_selected_matches(team_names)
                
                if not results:
                    st.info("No upcoming matches found for the selected teams.")
                    return
                    
                # Display insights for multiple matches
                _display_multiple_matches_insights(results)
                
            else:
                st.info("Select a match or teams to view betting insights.")
                
        except Exception as e:
            logger.error(f"Error rendering betting insights: {e}", exc_info=True)
            st.error("An error occurred while generating betting insights. Please try again later.")


def _display_single_match_insights(insights: Dict, prediction: Dict):
    """Display insights for a single match."""
    home_team = insights.get('home_team', 'Home')
    away_team = insights.get('away_team', 'Away')
    
    header_cols = st.columns([2, 1, 2])
    with header_cols[0]:
        st.markdown(f"<h3 style='text-align: right;'>{home_team}</h3>", unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown(f"<h3 style='text-align: center;'>vs</h3>", unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown(f"<h3 style='text-align: left;'>{away_team}</h3>", unsafe_allow_html=True)
        
    st.markdown(f"<p style='text-align: center; font-size: 1.1em;'><strong>Match Date:</strong> {datetime.fromisoformat(insights['match_date']).strftime('%A, %B %d, %Y')}</p>", unsafe_allow_html=True)
    
        from dashboard.components.unified_design_system import get_unified_design_system
    st.markdown("---")

    # Prediction visualization
    if 'probabilities' in prediction:
        fig = plot_match_outcome_probabilities(prediction)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("Betting Recommendations")
    
    if insights.get('recommendations'):
        for i, rec in enumerate(insights['recommendations']):
            with card(key=f"rec_{i}"):
                st.markdown(f"**{rec['bet'].replace('_', ' ').title()}**")
                st.markdown(f"> {rec['reason']}")
                
                rec_cols = st.columns(3)
                if 'odds' in rec:
                    with rec_cols[0]:
                        st.markdown(badge("Market Odds", tooltip=f"{rec['odds']:.2f}"), unsafe_allow_html=True)
            uds = get_unified_design_system()
            uds.inject_unified_css(dashboard_type="premium")
            uds.create_unified_header("Betting Insights", subtitle="Actionable AI-powered recommendations for every match")
                        st.markdown(badge("Model Probability", tooltip=f"{rec.get('probability', 0)*100:.1f}%"), unsafe_allow_html=True)
                    with rec_cols[2]:
                        st.markdown(badge("Expected Value", tooltip=f"{rec.get('expected_value', 0)*100:.1f}%"), unsafe_allow_html=True)
                else:
                    with rec_cols[0]:
                        st.markdown(badge("Bet Type", tooltip=rec['bet'].replace('_', ' ').title()), unsafe_allow_html=True)
                    with rec_cols[1]:
                        st.markdown(badge("Confidence", tooltip=f"{rec['confidence']*100:.1f}%"), unsafe_allow_html=True)
    else:
        st.info("No strong betting recommendations found for this match.")
    
    # Feature importance if available
    if 'feature_importance' in prediction and prediction['feature_importance']:
        with st.expander("View Model Feature Importance"):
            from dashboard.visualizations.plots import plot_feature_importance

            # Ensure it's a dict
            if isinstance(prediction['feature_importance'], dict):
                importance_df = pd.DataFrame(list(prediction['feature_importance'].items()), columns=['Feature', 'Importance'])
                fig = plot_feature_importance(importance_df)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Feature importance data is not in the expected format.")

def _display_multiple_matches_insights(results: List[Dict]):
    """Display insights for multiple matches."""
    st.write(f"Found betting insights for {len(results)} upcoming matches")
    
    # Convert to DataFrame for easier filtering
    matches_data = []
    for insight in results:
        if insight.get('recommendations'):
            status = "Value Bet Found"
            top_rec = insight['recommendations'][0]
            rec_type = top_rec['bet'].replace('_', ' ').title()
            if 'expected_value' in top_rec:
                value = f"{top_rec['expected_value']*100:.1f}%"
            else:
                value = f"{top_rec['confidence']*100:.1f}%"
        else:
            status = "No Value"
            rec_type = "None"
            value = "N/A"
            
        matches_data.append({
            'Match ID': insight['match_id'],
            'Home Team': insight['home_team'],
            'Away Team': insight['away_team'],
            'Match Date': datetime.fromisoformat(insight['match_date']).date(),
            'Status': status,
            'Top Recommendation': rec_type,
            'Value': value,
            'Recommendations': len(insight.get('recommendations', []))
        })
    
    if not matches_data:
        st.info("No insights available for the selected matches.")
        return
        
    # Create DataFrame
    df = pd.DataFrame(matches_data)
    
            uds = get_unified_design_system()
            def match_header():
                cols = st.columns([2, 1, 2])
                with cols[0]:
                    st.markdown(f"<h3 style='text-align: right;'>{home_team}</h3>", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"<h3 style='text-align: center;'>vs</h3>", unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(f"<h3 style='text-align: left;'>{away_team}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 1.1em;'><strong>Match Date:</strong> {datetime.fromisoformat(insights['match_date']).strftime('%A, %B %d, %Y')}</p>", unsafe_allow_html=True)
            uds.create_unified_card(match_header)
        
    # Apply sorting
    if date_sort == "Earliest First":
        filtered_df = filtered_df.sort_values('Match Date')
    else:
        filtered_df = filtered_df.sort_values('Match Date', ascending=False)
    
    # Display the table
    st.dataframe(
        filtered_df,
        column_config={
                    def rec_card(rec):
                        st.markdown(f"**{rec['bet'].replace('_', ' ').title()}**")
                        st.markdown(f"> {rec['reason']}")
                        rec_cols = st.columns(3)
                        if 'odds' in rec:
                            with rec_cols[0]:
                                st.markdown(badge("Market Odds", tooltip=f"{rec['odds']:.2f}"), unsafe_allow_html=True)
                            with rec_cols[1]:
                                st.markdown(badge("Model Probability", tooltip=f"{rec.get('probability', 0)*100:.1f}%"), unsafe_allow_html=True)
                            with rec_cols[2]:
                                st.markdown(badge("Expected Value", tooltip=f"{rec.get('expected_value', 0)*100:.1f}%"), unsafe_allow_html=True)
                        else:
                            with rec_cols[0]:
                                st.markdown(badge("Bet Type", tooltip=rec['bet'].replace('_', ' ').title()), unsafe_allow_html=True)
                            with rec_cols[1]:
                                st.markdown(badge("Confidence", tooltip=f"{rec['confidence']*100:.1f}%"), unsafe_allow_html=True)
        for insight in results:
                    uds.create_unified_card(lambda rec=rec: rec_card(rec))
            if insight['match_id'] == selected_match_id:
                insights_generator = BettingInsightsGenerator()
                match = None
                with insights_generator.db.session_scope() as session:
                    from database.schema import Match, Team
                    match_obj = session.query(Match).filter_by(id=selected_match_id).first()
                    if match_obj:
                        home_team = session.query(Team).filter_by(id=match_obj.home_team_id).first()
                        away_team = session.query(Team).filter_by(id=match_obj.away_team_id).first()
                        
                        if home_team and away_team:
                            match = {
                                'match_id': selected_match_id,
                                'home_team': home_team.name,
                                'away_team': away_team.name,
                                'home_team_id': match_obj.home_team_id,
                                'away_team_id': match_obj.away_team_id,
                                'match_date': match_obj.match_date.isoformat() if match_obj.match_date else datetime.now().isoformat(),
                                'league': match_obj.league_id
                            }
                
                if match:
                    prediction = insights_generator.predict_match(match)
                    _display_single_match_insights(insight, prediction)
                break
