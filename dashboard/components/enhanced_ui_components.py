"""
Enhanced UI/UX components for professional team and match display.

Provides modern, responsive components for displaying team information,
match cards, and interactive elements with proper styling and animations.
"""
import streamlit as st
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedTeamDisplay:
    """Enhanced team display components with modern UI/UX."""
    
    def __init__(self):
        self.setup_custom_css()
    
    def setup_custom_css(self):
        """Setup custom CSS for enhanced team display."""
        st.markdown("""
        <style>
        .team-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }
        
        .team-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.2);
        }
        
        .team-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        
        .team-flag {
            font-size: 2.5em;
            margin-right: 15px;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
        }
        
        .team-info h3 {
            color: white;
            margin: 0;
            font-size: 1.4em;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .team-info p {
            color: rgba(255,255,255,0.8);
            margin: 5px 0 0 0;
            font-size: 0.9em;
        }
        
        .country-info {
            display: flex;
            align-items: center;
            color: rgba(255,255,255,0.9);
            font-size: 0.8em;
        }
        
        .match-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            border: 1px solid #e0e0e0;
            transition: all 0.3s ease;
        }
        
        .match-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        }
        
        .match-header {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .match-teams {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0;
        }
        
        .team-side {
            text-align: center;
            flex: 1;
        }
        
        .team-side.home {
            margin-right: 20px;
        }
        
        .team-side.away {
            margin-left: 20px;
        }
        
        .vs-divider {
            font-size: 1.5em;
            font-weight: bold;
            color: #666;
            margin: 0 20px;
        }
        
        .prediction-badge {
            background: linear-gradient(45deg, #ff6b6b, #ee5a24);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            display: inline-block;
            margin: 10px 0;
        }
        
        .confidence-bar {
            background: #f0f0f0;
            border-radius: 10px;
            height: 8px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ecdc4, #44a08d);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        .league-badge {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.7em;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 10px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            backdrop-filter: blur(5px);
        }
        
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            color: white;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.8em;
            color: rgba(255,255,255,0.8);
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """, unsafe_allow_html=True)
    
    def display_team_card(self, team_data: Dict[str, str], show_stats: bool = False):
        """Display an enhanced team card."""
        team_color = team_data.get('color', '#667eea')
        
        st.markdown(f"""
        <div class="team-card fade-in" style="background: linear-gradient(135deg, {team_color} 0%, {team_color}dd 100%);">
            <div class="team-header">
                <div class="team-flag">{team_data.get('flag', '⚽')}</div>
                <div class="team-info">
                    <h3>{team_data.get('display_name', team_data.get('name', 'Unknown Team'))}</h3>
                    <p>{team_data.get('full_name', team_data.get('name', 'Unknown Team'))}</p>
                    <div class="country-info">
                        <span>{team_data.get('country_flag', '🏳️')}</span>
                        <span style="margin-left: 5px;">{team_data.get('country', 'Unknown')}</span>
                    </div>
                </div>
            </div>
            {self._get_team_stats_html(team_data) if show_stats else ''}
        </div>
        """, unsafe_allow_html=True)
    
    def _get_team_stats_html(self, team_data: Dict[str, str]) -> str:
        """Get team stats HTML."""
        venue = team_data.get('venue', 'Unknown Stadium')
        capacity = team_data.get('capacity', 'Unknown')
        league = team_data.get('league', 'Unknown League')
        
        return f"""
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">🏟️</div>
                <div class="stat-label">{venue}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{capacity:,}</div>
                <div class="stat-label">Capacity</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{league}</div>
                <div class="stat-label">League</div>
            </div>
        </div>
        """
    
    def display_match_card(self, match_data: Dict[str, any], prediction_data: Optional[Dict] = None):
        """Display an enhanced match card."""
        home_team = match_data.get('home_team', 'Home Team')
        away_team = match_data.get('away_team', 'Away Team')
        league = match_data.get('league', 'Unknown League')
        match_date = match_data.get('match_date', 'TBD')
        
        # Get team data
        try:
            from utils.team_data_enhancer import get_enhanced_team_data
            home_team_data = get_enhanced_team_data(home_team)
            away_team_data = get_enhanced_team_data(away_team)
        except ImportError:
            home_team_data = {'flag': '🏠', 'display_name': home_team, 'color': '#667eea'}
            away_team_data = {'flag': '✈️', 'display_name': away_team, 'color': '#667eea'}
        
        prediction_html = ""
        if prediction_data:
            confidence = prediction_data.get('confidence', 0.5)
            predicted_outcome = prediction_data.get('predicted_outcome', 'Unknown')
            
            prediction_html = f"""
            <div class="prediction-badge">
                Predicted: {predicted_outcome.replace('_', ' ').title()}
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
            </div>
            <p style="text-align: center; font-size: 0.8em; color: #666;">
                Confidence: {confidence:.1%}
            </p>
            """
        
        st.markdown(f"""
        <div class="match-card fade-in">
            <div class="match-header">
                <div class="league-badge">{league}</div>
                <h4 style="margin: 10px 0; color: #333;">{match_date}</h4>
            </div>
            
            <div class="match-teams">
                <div class="team-side home">
                    <div style="font-size: 2em; margin-bottom: 10px;">{home_team_data.get('flag', '🏠')}</div>
                    <h5 style="margin: 0; color: #333;">{home_team_data.get('display_name', home_team)}</h5>
                    <p style="margin: 5px 0 0 0; font-size: 0.8em; color: #666;">{home_team_data.get('full_name', home_team)}</p>
                </div>
                
                <div class="vs-divider">VS</div>
                
                <div class="team-side away">
                    <div style="font-size: 2em; margin-bottom: 10px;">{away_team_data.get('flag', '✈️')}</div>
                    <h5 style="margin: 0; color: #333;">{away_team_data.get('display_name', away_team)}</h5>
                    <p style="margin: 5px 0 0 0; font-size: 0.8em; color: #666;">{away_team_data.get('full_name', away_team)}</p>
                </div>
            </div>
            
            {prediction_html}
        </div>
        """, unsafe_allow_html=True)
    
    def display_loading_state(self, message: str = "Loading..."):
        """Display a loading state."""
        st.markdown(f"""
        <div style="text-align: center; padding: 40px;">
            <div class="loading-spinner"></div>
            <p style="margin-top: 15px; color: #666;">{message}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_error_state(self, message: str = "Something went wrong"):
        """Display an error state."""
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; color: #e74c3c;">
            <div style="font-size: 3em; margin-bottom: 15px;">⚠️</div>
            <h4 style="color: #e74c3c;">{message}</h4>
            <p style="color: #666;">Please try again or contact support if the problem persists.</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_empty_state(self, message: str = "No data available", icon: str = "📭"):
        """Display an empty state."""
        st.markdown(f"""
        <div style="text-align: center; padding: 40px; color: #666;">
            <div style="font-size: 3em; margin-bottom: 15px;">{icon}</div>
            <h4 style="color: #333;">{message}</h4>
            <p style="color: #666;">Check back later for updates.</p>
        </div>
        """, unsafe_allow_html=True)


# Global instance
enhanced_display = EnhancedTeamDisplay()


def display_team_card(team_data: Dict[str, str], show_stats: bool = False):
    """Convenience function to display team card."""
    enhanced_display.display_team_card(team_data, show_stats)


def display_match_card(match_data: Dict[str, any], prediction_data: Optional[Dict] = None):
    """Convenience function to display match card."""
    enhanced_display.display_match_card(match_data, prediction_data)


def display_loading_state(message: str = "Loading..."):
    """Convenience function to display loading state."""
    enhanced_display.display_loading_state(message)


def display_error_state(message: str = "Something went wrong"):
    """Convenience function to display error state."""
    enhanced_display.display_error_state(message)


def display_empty_state(message: str = "No data available", icon: str = "📭"):
    """Convenience function to display empty state."""
    enhanced_display.display_empty_state(message, icon)
