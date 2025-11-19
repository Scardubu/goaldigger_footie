"""
HTML Sanitization Utility for GoalDiggers Dashboard
Provides secure HTML rendering functions to prevent XSS vulnerabilities.
"""

import html
import re
from typing import Any, Dict, Optional, Union


class HTMLSanitizer:
    """Secure HTML sanitization for dashboard components."""
    
    # Allowed HTML tags for basic formatting
    ALLOWED_TAGS = {
        'div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'strong', 'em', 'br', 'hr', 'ul', 'ol', 'li'
    }
    
    # Allowed CSS properties for styling
    ALLOWED_CSS_PROPERTIES = {
        'color', 'background', 'background-color', 'font-size', 'font-weight',
        'text-align', 'margin', 'padding', 'border', 'border-radius',
        'display', 'flex', 'grid', 'width', 'height', 'max-width', 'max-height',
        'opacity', 'transform', 'transition', 'animation'
    }
    
    @staticmethod
    def escape_html(text: Any) -> str:
        """Escape HTML characters in user input."""
        if text is None:
            return ""
        return html.escape(str(text), quote=True)
    
    @staticmethod
    def sanitize_team_name(team_name: str) -> str:
        """Sanitize team names for safe HTML rendering."""
        if not team_name:
            return "Unknown Team"
        
        # Remove any HTML tags and escape special characters
        sanitized = re.sub(r'<[^>]*>', '', str(team_name))
        sanitized = html.escape(sanitized, quote=True)
        
        # Limit length to prevent layout issues
        if len(sanitized) > 50:
            sanitized = sanitized[:47] + "..."
        
        return sanitized
    
    @staticmethod
    def sanitize_percentage(value: Union[int, float]) -> str:
        """Sanitize percentage values for display."""
        try:
            num_value = float(value)
            # Clamp between 0 and 100
            num_value = max(0, min(100, num_value))
            return f"{num_value:.1f}"
        except (ValueError, TypeError):
            return "0.0"
    
    @staticmethod
    def sanitize_metric_value(value: Any, decimal_places: int = 2) -> str:
        """Sanitize numeric metric values."""
        try:
            if isinstance(value, str):
                # Remove any non-numeric characters except decimal point
                value = re.sub(r'[^\d.-]', '', value)
            
            num_value = float(value)
            return f"{num_value:.{decimal_places}f}"
        except (ValueError, TypeError):
            return "0.00"
    
    @staticmethod
    def create_safe_status_html(
        health_percentage: float,
        healthy_components: int,
        total_components: int,
        ml_integration_rate: float,
        ml_loaded: int,
        ml_total: int,
        load_time: float,
        user_interactions: int
    ) -> str:
        """Create sanitized status HTML with safe values."""
        
        # Sanitize all inputs
        health_pct = HTMLSanitizer.sanitize_percentage(health_percentage)
        healthy_comp = max(0, int(healthy_components))
        total_comp = max(1, int(total_components))
        ml_rate = HTMLSanitizer.sanitize_percentage(ml_integration_rate)
        ml_loaded_safe = max(0, int(ml_loaded))
        ml_total_safe = max(1, int(ml_total))
        load_time_safe = HTMLSanitizer.sanitize_metric_value(load_time, 2)
        interactions_safe = max(0, int(user_interactions))
        
        # Determine status colors based on thresholds
        health_color = "#10b981" if float(health_pct) >= 80 else "#f59e0b" if float(health_pct) >= 60 else "#ef4444"
        ml_color = "#10b981" if float(ml_rate) >= 80 else "#f59e0b" if float(ml_rate) >= 50 else "#ef4444"
        load_color = "#10b981" if float(load_time_safe) < 3 else "#f59e0b" if float(load_time_safe) < 6 else "#ef4444"
        
        return f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                   gap: 1rem; margin-bottom: 2rem;">
            
            <div style="background: white; border: 1px solid #e5e7eb; 
                       border-radius: 1rem; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #6b7280; font-size: 0.875rem; 
                              text-transform: uppercase; letter-spacing: 0.05em;">System Health</h4>
                    <div style="width: 8px; height: 8px; border-radius: 50%; 
                               background: {health_color}; animation: pulse 2s infinite;"></div>
                </div>
                <div style="font-size: 2rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">
                    {health_pct}%
                </div>
                <div style="color: #6b7280; font-size: 0.875rem;">
                    {healthy_comp}/{total_comp} systems active
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #e5e7eb; 
                       border-radius: 1rem; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #6b7280; font-size: 0.875rem; 
                              text-transform: uppercase; letter-spacing: 0.05em;">ML Integration</h4>
                    <div style="width: 8px; height: 8px; border-radius: 50%; 
                               background: {ml_color}; animation: pulse 2s infinite;"></div>
                </div>
                <div style="font-size: 2rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">
                    {ml_rate}%
                </div>
                <div style="color: #6b7280; font-size: 0.875rem;">
                    {ml_loaded_safe}/{ml_total_safe} ML components
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #e5e7eb; 
                       border-radius: 1rem; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #6b7280; font-size: 0.875rem; 
                              text-transform: uppercase; letter-spacing: 0.05em;">Performance</h4>
                    <div style="width: 8px; height: 8px; border-radius: 50%; 
                               background: {load_color}; animation: pulse 2s infinite;"></div>
                </div>
                <div style="font-size: 2rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">
                    {load_time_safe}s
                </div>
                <div style="color: #6b7280; font-size: 0.875rem;">
                    Dashboard load time
                </div>
            </div>
            
            <div style="background: white; border: 1px solid #e5e7eb; 
                       border-radius: 1rem; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h4 style="margin: 0; color: #6b7280; font-size: 0.875rem; 
                              text-transform: uppercase; letter-spacing: 0.05em;">User Activity</h4>
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: #10b981;
                               animation: pulse 2s infinite;"></div>
                </div>
                <div style="font-size: 2rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem;">
                    {interactions_safe}
                </div>
                <div style="color: #6b7280; font-size: 0.875rem;">
                    Active interactions
                </div>
            </div>
        </div>
        
        <style>
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        </style>
        """
    
    @staticmethod
    def create_safe_team_display_html(
        home_team: str,
        away_team: str,
        home_league: str = "Premier League",
        away_league: str = "Premier League",
        is_cross_league: bool = False
    ) -> str:
        """Create sanitized team display HTML."""
        
        # Sanitize all inputs
        home_team_safe = HTMLSanitizer.sanitize_team_name(home_team)
        away_team_safe = HTMLSanitizer.sanitize_team_name(away_team)
        home_league_safe = HTMLSanitizer.escape_html(home_league)
        away_league_safe = HTMLSanitizer.escape_html(away_league)
        
        cross_league_indicator = ""
        if is_cross_league and home_league != away_league:
            cross_league_indicator = '<div style="font-size: 0.75rem; color: #f59e0b; font-weight: 600; margin-top: 0.25rem;">CROSS-LEAGUE</div>'
        
        return f"""
        <div style="display: flex; align-items: center; justify-content: space-between; 
                   background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                   border-radius: 1rem; padding: 2rem; margin: 1rem 0; border: 1px solid #cbd5e1;">
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏠</div>
                <div style="font-weight: 700; color: #1e293b; font-size: 1.1rem;">{home_team_safe}</div>
                <div style="font-size: 0.875rem; color: #64748b;">
                    {home_league_safe}
                </div>
            </div>
            <div style="text-align: center; padding: 0 2rem;">
                <div style="font-size: 2rem; font-weight: 700; color: #475569;">VS</div>
                {cross_league_indicator}
            </div>
            <div style="text-align: center; flex: 1;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">✈️</div>
                <div style="font-weight: 700; color: #1e293b; font-size: 1.1rem;">{away_team_safe}</div>
                <div style="font-size: 0.875rem; color: #64748b;">
                    {away_league_safe}
                </div>
            </div>
        </div>
        """


def sanitize_for_html(value: Any) -> str:
    """Quick sanitization function for general use."""
    return HTMLSanitizer.escape_html(value)


def create_safe_metric_html(title: str, value: Any, description: str = "", color: str = "#10b981") -> str:
    """Create a safe metric display HTML."""
    title_safe = HTMLSanitizer.escape_html(title)
    value_safe = HTMLSanitizer.escape_html(value)
    desc_safe = HTMLSanitizer.escape_html(description)
    
    # Validate color is a hex color
    if not re.match(r'^#[0-9a-fA-F]{6}$', color):
        color = "#10b981"  # Default safe color
    
    return f"""
    <div style="background: white; border: 1px solid #e5e7eb; border-radius: 0.5rem; 
               padding: 1rem; text-align: center;">
        <div style="font-size: 1.5rem; font-weight: 700; color: {color}; margin-bottom: 0.5rem;">
            {value_safe}
        </div>
        <div style="font-weight: 600; color: #374151; margin-bottom: 0.25rem;">
            {title_safe}
        </div>
        <div style="font-size: 0.875rem; color: #6b7280;">
            {desc_safe}
        </div>
    </div>
    """
