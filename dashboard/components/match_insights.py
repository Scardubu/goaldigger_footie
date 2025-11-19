"""
Match insights component for displaying AI-driven betting predictions and analysis.
Provides an interactive UI for viewing XGBoost model predictions and explanations.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.ui_elements import card, header, spinner_wrapper
from dashboard.visualizations.plots import interactive_feature_importance
from dashboard.visualizations.xgboost_plots import create_prediction_gauge
from utils.ai_insights import MatchAnalyzer, create_enhanced_analysis
from utils.prediction_handler import PredictionHandler

logger = logging.getLogger(__name__)

# Initialize singleton instances
prediction_handler = None
match_analyzer = None


def get_prediction_handler():
    """Get or initialize the prediction handler singleton."""
    global prediction_handler
    if prediction_handler is None:
        prediction_handler = PredictionHandler()
    return prediction_handler


def get_match_analyzer():
    """Get or initialize the match analyzer singleton."""
    global match_analyzer
    if match_analyzer is None:
        match_analyzer = MatchAnalyzer()
    return match_analyzer


def render_match_insights(match_data=None, match_features=None, match_id=None, data_loader=None):
    """
    Render the match insights component with predictions and AI analysis.
    
    Args:
        match_data: Dictionary containing match information
        match_features: Optional dictionary of features for XGBoost prediction
        match_id: Optional match ID to load details from the data_loader
        data_loader: Optional data loader instance to fetch match details
    """
    # If match_id and data_loader are provided but match_data is not, load match data
    if match_id and data_loader and not match_data:
        try:
            logger.info(f"Loading match details for match_id: {match_id}")
            match_details = data_loader.load_match_details(match_id)
            
            # Get fixture details - this is the basic match information
            match_data = match_details.get('fixture_details', {})
            
            # Get features directly or from nested location in stats
            match_features = match_details.get('features', {})
            if not match_features and match_details.get('stats', {}).get('features'):
                match_features = match_details.get('stats', {}).get('features', {})
                
            # For complete visibility in templates, add the entire match_details
            # to match_data so components can access all data if needed
            match_data['_full_details'] = match_details
            
            # Log what we found
            logger.debug(f"Loaded match data: {match_data}")
            logger.debug(f"Found features: {list(match_features.keys()) if match_features else 'No features found'}")
        except Exception as e:
            logger.error(f"Failed to load match details: {e}", exc_info=True)
            st.error(f"Failed to load match details: {e}")
            match_data = {}
            match_features = {}
    
    # Ensure we have match_data dictionary
    if not match_data:
        match_data = {}
    
    # Enhanced team name resolution with detailed logging for diagnosis
    logger.info(f"Match data keys available: {list(match_data.keys())}")
    
    # First check if we have _full_details with fixture_details
    if '_full_details' in match_data and match_data['_full_details'] and 'fixture_details' in match_data['_full_details']:
        # Use fixture_details as the most reliable source
        fixture_details = match_data['_full_details']['fixture_details']
        home_team = fixture_details.get('home_team', None)
        away_team = fixture_details.get('away_team', None)
        logger.info(f"Got team names from fixture_details: {home_team} vs {away_team}")
    else:
        # Fallback to direct fields in match_data
        home_team = match_data.get('home_team', None)
        away_team = match_data.get('away_team', None)
        
        # If not found, try alternative field names in order of preference
        if not home_team:
            home_team = match_data.get('home_team_name', match_data.get('home_team_id', 'Home Team'))
        if not away_team:
            away_team = match_data.get('away_team_name', match_data.get('away_team_id', 'Away Team'))
    
    # Log what we found initially
    logger.info(f"Initial team names from match data: {home_team} vs {away_team}")
    
    # Resolve team names if they're IDs or require resolution
    if data_loader and hasattr(data_loader, 'resolve_team_name'):
        if not isinstance(home_team, bool) and home_team:  # Exclude boolean values, include all other non-None values
            try:
                resolved_home = data_loader.resolve_team_name(home_team)
                if resolved_home:
                    logger.info(f"Resolved home team from '{home_team}' to '{resolved_home}'")
                    home_team = resolved_home
                else:
                    logger.warning(f"Home team resolution returned None for '{home_team}'")
            except Exception as e:
                logger.warning(f"Could not resolve home team name '{home_team}': {e}")
        
        if not isinstance(away_team, bool) and away_team:  # Exclude boolean values, include all other non-None values
            try:
                resolved_away = data_loader.resolve_team_name(away_team)
                if resolved_away:
                    logger.info(f"Resolved away team from '{away_team}' to '{resolved_away}'")
                    away_team = resolved_away
                else:
                    logger.warning(f"Away team resolution returned None for '{away_team}'")
            except Exception as e:
                logger.warning(f"Could not resolve away team name '{away_team}': {e}")
    
    # Check if we still have placeholder team names (either 'Home Team' or starts with 'Unknown Team')
    if home_team == 'Home Team' or (isinstance(home_team, str) and home_team.startswith('Unknown Team')):
        # Try harder to get a team name from other sources
        if '_full_details' in match_data and isinstance(match_data['_full_details'], dict):
            full_details = match_data['_full_details']
            if 'fixture_details' in full_details and isinstance(full_details['fixture_details'], dict):
                fixture_details = full_details['fixture_details']
                potential_home = fixture_details.get('home_team', fixture_details.get('home_team_name', None))
                if potential_home and data_loader and hasattr(data_loader, 'resolve_team_name'):
                    home_team = data_loader.resolve_team_name(potential_home)
                    logger.info(f"Found home team '{home_team}' in nested fixture details")
                    
    if away_team == 'Away Team' or (isinstance(away_team, str) and away_team.startswith('Unknown Team')):
        # Try harder to get a team name from other sources
        if '_full_details' in match_data and isinstance(match_data['_full_details'], dict):
            full_details = match_data['_full_details']
            if 'fixture_details' in full_details and isinstance(full_details['fixture_details'], dict):
                fixture_details = full_details['fixture_details']
                potential_away = fixture_details.get('away_team', fixture_details.get('away_team_name', None))
                if potential_away and data_loader and hasattr(data_loader, 'resolve_team_name'):
                    away_team = data_loader.resolve_team_name(potential_away)
                    logger.info(f"Found away team '{away_team}' in nested fixture details")
    
    # Ensure we have reasonable team names for display if resolution failed
    if home_team == 'Home Team' or (isinstance(home_team, str) and home_team.startswith('Unknown Team [Home')):
        home_team = "Home Team"
        logger.warning("Failed to resolve home team name properly")
    
    if away_team == 'Away Team' or (isinstance(away_team, str) and away_team.startswith('Unknown Team [Away')):
        away_team = "Away Team"  
        logger.warning("Failed to resolve away team name properly")
    
    # Log final result
    logger.info(f"Final resolved team names: {home_team} vs {away_team}")
    
    # Try to use the UnifiedDesignSystem for consistent styling; fall back to legacy header
    try:
        from dashboard.components.unified_design_system import \
            get_unified_design_system
        uds = get_unified_design_system()
        uds.inject_unified_css('integrated')
        uds.create_unified_header(f"Match Insights: {home_team} vs {away_team}")
    except Exception:
        uds = None
        st.markdown(header(f"Match Insights: {home_team} vs {away_team}", level=2))
    
    # Create tabs for different insights
    tab1, tab2, tab3 = st.tabs(["Prediction", "AI Analysis", "Key Factors"])
    
    with tab1:
        render_prediction_tab(match_data, match_features)
    
    with tab2:
        render_analysis_tab(match_data, match_features)
    
    with tab3:
        render_factors_tab(match_data, match_features)


@spinner_wrapper("Loading prediction...")
def render_prediction_tab(match_data: Dict[str, Any], match_features: Optional[Dict[str, Any]] = None):
    """Render the prediction tab with XGBoost model results."""
    # Use the same robust team name resolution logic as the parent function
    # First try the primary fields we expect from the loader
    home_team = match_data.get('home_team', None)
    away_team = match_data.get('away_team', None)
    
    # If not found, try alternative field names in order of preference
    if not home_team:
        home_team = match_data.get('home_team_name', match_data.get('home_team_id', 'Home Team'))
    if not away_team:
        away_team = match_data.get('away_team_name', match_data.get('away_team_id', 'Away Team'))
    
    # Check if either team name appears to be a placeholder or unresolved ID
    if home_team == 'Home Team' or (isinstance(home_team, str) and home_team.startswith('Unknown Team')):
        # Fall back to a generic name if we couldn't resolve the team name
        home_team = "Home Team"
        logger.warning("Using generic home team name in prediction tab")
                
    if away_team == 'Away Team' or (isinstance(away_team, str) and away_team.startswith('Unknown Team')):
        # Fall back to a generic name if we couldn't resolve the team name
        away_team = "Away Team"
        logger.warning("Using generic away team name in prediction tab")
    
    # Log the available data
    logger.debug(f"Rendering prediction tab with match data keys: {list(match_data.keys())}")
    logger.debug(f"Team names for prediction: {home_team} vs {away_team}")
    logger.debug(f"Feature data available: {True if match_features and match_features != {} else False}")
    if match_features:
        logger.debug(f"Feature keys: {list(match_features.keys())}")
    
    # Attempt to access design system for consistent cards/metrics
    try:
        from dashboard.components.unified_design_system import \
            get_unified_design_system
        uds = get_unified_design_system()
    except Exception:
        uds = None

    with st.container():
        # Get prediction
        prediction = None
        
        # Check if features might be nested in stats
        if not match_features and isinstance(match_data.get('stats'), dict):
            nested_features = match_data.get('stats', {}).get('features')
            if nested_features and isinstance(nested_features, dict):
                logger.info("Found features nested in match_data.stats.features")
                match_features = nested_features
        
        if match_features and match_features != {}:
            try:
                handler = get_prediction_handler()
                
                # Convert features to DataFrame if needed
                if isinstance(match_features, dict):
                    # Make sure we have valid feature values (non-empty dict)
                    if any(v is not None for v in match_features.values()):
                        features_df = pd.DataFrame([match_features])
                    else:
                        st.info("Feature data is available but values are incomplete. Please refresh match statistics.")
                        return
                else:
                    features_df = pd.DataFrame(match_features)
                
                # Generate match ID
                match_id = f"{home_team}_{away_team}".replace(" ", "_").lower()
                
                # Get prediction
                prediction = handler.get_match_prediction(features_df, match_id)
                
                # Display prediction
                # Use unified card when available for consistent UI
                card_html = f"### Match Outcome Prediction\n\n**{home_team}** vs **{away_team}**"
                if uds:
                    uds.create_unified_card(lambda: st.markdown(card_html, unsafe_allow_html=True))
                else:
                    st.markdown(card(card_html), unsafe_allow_html=True)
                
                # Create probability gauge chart using the new centralized function
                fig = create_prediction_gauge(
                    prediction=prediction,
                    home_team=home_team,
                    away_team=away_team
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Format for display
                formatted = handler.format_prediction_for_display(prediction)

                # Display confidence
                confidence = formatted.get('confidence', 0)
                # If unified design system offers metric row, use it; otherwise use progress
                try:
                    if uds:
                        uds.create_unified_metric_row({"Model Confidence": {"value": f"{confidence}%"}})
                    else:
                        st.progress(confidence/100, text=f"Model Confidence: {confidence}%")
                except Exception:
                    st.progress(confidence/100, text=f"Model Confidence: {confidence}%")
                
                # Display status
                status = formatted.get('status', 'unknown')
                status_color = "green" if status == "success" else "orange" if status == "fallback" else "red"
                st.markdown(f"<span style='color:{status_color}'>Status: {status.upper()}</span>", unsafe_allow_html=True)
                
                # Display value betting tips
                display_value_betting_tips(match_data, prediction)
                
            except Exception as e:
                logger.error(f"Error rendering prediction: {e}")
                st.warning(f"⚠️ Unable to generate prediction: {str(e)}")
        else:
            st.info("No feature data available for prediction. Please load match statistics first.")


@spinner_wrapper("Generating analysis...")
def render_analysis_tab(match_data: Dict[str, Any], match_features: Optional[Dict[str, Any]] = None):
    """Render the AI analysis tab with detailed betting insights."""
    home_team = match_data.get('home_team', 'Home Team')
    away_team = match_data.get('away_team', 'Away Team')
    
    try:
        from dashboard.components.unified_design_system import \
            get_unified_design_system
        uds = get_unified_design_system()
    except Exception:
        uds = None

    with st.container():
        # Generate analysis
        try:
            # Use the enhanced analysis function
            analysis_result = create_enhanced_analysis(
                home_team=home_team,
                away_team=away_team,
                match_data=match_data,
                match_features=match_features
            )
            
            if analysis_result['success']:
                # Show processing time
                st.caption(f"Analysis generated in {analysis_result['processing_time']:.2f} seconds")
                
                analysis_html = f"### AI Betting Analysis\n\n{analysis_result['analysis_text']}"
                if uds:
                    uds.create_unified_card(lambda: st.markdown(analysis_html, unsafe_allow_html=True))
                else:
                    st.markdown(card(analysis_html), unsafe_allow_html=True)
                
                # Show disclaimer
                st.info("⚠️ **Disclaimer**: This analysis is AI-generated and should be used as supplementary information only. Always do your own research before placing bets.")
            else:
                st.error(f"⚠️ Failed to generate analysis: {analysis_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error rendering analysis: {e}")
            st.warning(f"⚠️ Unable to generate analysis: {str(e)}")


@spinner_wrapper("Analyzing factors...")
def render_factors_tab(match_data: Dict[str, Any], match_features: Optional[Dict[str, Any]] = None):
    """Render the key factors tab with XGBoost model explanations."""
    # Check if features might be nested in stats
    if not match_features and isinstance(match_data.get('stats'), dict):
        nested_features = match_data.get('stats', {}).get('features')
        if nested_features and isinstance(nested_features, dict):
            logger.info("Found features nested in match_data.stats.features")
            match_features = nested_features
    
    if not match_features or match_features == {}:
        st.info("No feature data available for factor analysis. Please load match statistics first.")
        return
    
    try:
        handler = get_prediction_handler()
        
        # Log the feature data for debugging
        logger.debug(f"Factor analysis with feature data: {bool(match_features)}")
        if match_features:
            logger.debug(f"Available features: {list(match_features.keys())}")
        
        # Convert features to DataFrame if needed
        if isinstance(match_features, dict):
            # Verify we have valid feature data
            if not any(v is not None for v in match_features.values()):
                st.info("Feature data is available but values are incomplete. Please refresh match statistics.")
                return
            features_df = pd.DataFrame([match_features])
        else:
            features_df = pd.DataFrame(match_features)
        
        # Generate match ID
        home_team = match_data.get('home_team', 'Home Team')
        away_team = match_data.get('away_team', 'Away Team')
        match_id = f"{home_team}_{away_team}".replace(" ", "_").lower()
        
        # Get prediction
        prediction = handler.get_match_prediction(features_df, match_id)
        
        # Check if we have explanations
        try:
            from dashboard.components.unified_design_system import \
                get_unified_design_system
            uds = get_unified_design_system()
        except Exception:
            uds = None

        if 'explanations' in prediction and 'top_features' in prediction['explanations']:
            title_html = "### Key Influential Factors"
            if uds:
                uds.create_unified_card(lambda: st.markdown(title_html, unsafe_allow_html=True))
            else:
                st.markdown(card(title_html), unsafe_allow_html=True)
            
            # Get top features
            top_features = prediction['explanations']['top_features']
            
            # Create feature importance bar chart using the new centralized function
            fig = interactive_feature_importance(top_features)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display feature details in expandable section
            with st.expander("Feature Details"):
                feature_items = list(top_features.items())
                feature_data = []
                for feature, importance in feature_items[:10]:
                    # Format feature name
                    formatted_name = feature.replace('_', ' ').title()
                    formatted_name = formatted_name.replace('Xg', 'xG').replace('Ht', 'HT').replace('Ft', 'FT')
                    
                    # Get feature value if available
                    value = match_features.get(feature, "N/A")
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                    else:
                        formatted_value = str(value)
                    
                    feature_data.append({
                        "Feature": formatted_name,
                        "Importance": f"{importance:.4f}",
                        "Value": formatted_value
                    })
                
                st.table(pd.DataFrame(feature_data))
        else:
            st.info("No feature importance data available for this match.")
            
    except Exception as e:
        logger.error(f"Error rendering factors: {e}")
        st.warning(f"⚠️ Unable to generate factor analysis: {str(e)}")




def display_value_betting_tips(match_data: Dict[str, Any], prediction: Dict[str, Any]):
    """
    Display value betting tips based on comparing model predictions to odds.
    
    Args:
        match_data: Match data including odds
        prediction: Model prediction
    """
    if not prediction or 'home_win' not in prediction:
        return
    
    # Get model probabilities
    home_prob = prediction.get('home_win', 0)
    draw_prob = prediction.get('draw', 0)
    away_prob = prediction.get('away_win', 0)
    
    # Get odds if available
    odds = match_data.get('odds', {})
    if not odds:
        return
    
    home_odds = odds.get('home_win')
    draw_odds = odds.get('draw')
    away_odds = odds.get('away_win')
    
    # Check if we have valid odds
    if not all(isinstance(x, (int, float)) for x in [home_odds, draw_odds, away_odds]):
        return
    
    # Convert odds to implied probabilities
    home_implied = 1 / home_odds if home_odds else 0
    draw_implied = 1 / draw_odds if draw_odds else 0
    away_implied = 1 / away_odds if away_odds else 0
    
    # Calculate value (model probability - implied probability)
    home_value = home_prob - home_implied
    draw_value = draw_prob - draw_implied
    away_value = away_prob - away_implied
    
    # Display value bets (difference > 5%)
    st.markdown(card("### Value Betting Opportunities"), unsafe_allow_html=True)
    
    value_found = False
    
    # Check for significant value (>5%)
    if home_value > 0.05:
        st.markdown(f"✅ **Home Win** - Model suggests **{home_prob*100:.1f}%** probability vs bookmaker implied **{home_implied*100:.1f}%** (Odds: {home_odds})")
        value_found = True
    
    if draw_value > 0.05:
        st.markdown(f"✅ **Draw** - Model suggests **{draw_prob*100:.1f}%** probability vs bookmaker implied **{draw_implied*100:.1f}%** (Odds: {draw_odds})")
        value_found = True
    
    if away_value > 0.05:
        st.markdown(f"✅ **Away Win** - Model suggests **{away_prob*100:.1f}%** probability vs bookmaker implied **{away_implied*100:.1f}%** (Odds: {away_odds})")
        value_found = True
    
    if not value_found:
        st.info("No significant value betting opportunities identified for this match.")
