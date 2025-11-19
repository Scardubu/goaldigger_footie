"""
UI Elements for the GoalDiggers Football Betting Platform.
Contains reusable UI components with consistent styling.
"""

import logging
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import streamlit as st

# Import HTML sanitizer for secure rendering
try:
    from utils.html_sanitizer import (HTMLSanitizer, create_safe_metric_html,
                                      sanitize_for_html)
except ImportError:
    # Fallback if sanitizer not available
    class HTMLSanitizer:
        @staticmethod
        def escape_html(text):
            import html
            return html.escape(str(text), quote=True) if text else ""

    def sanitize_for_html(value):
        return HTMLSanitizer.escape_html(value)

def render_html_safely(html_content, key=None):
    """
    Safely render HTML content with Streamlit using proper sanitization.
    Args:
        html_content: HTML string to render (will be sanitized)
        key: Optional key for the component
    """
    # Sanitize the HTML content before rendering
    if html_content:
        # Use Streamlit's built-in components for safer rendering
        st.markdown(sanitize_for_html(html_content), unsafe_allow_html=True)
    return True

# Data validation utilities for dashboard components
def validate_match_data(match_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validates and sanitizes match data to ensure all required fields exist.
    
    Args:
        match_data: Raw match data dictionary that might be missing fields
        
    Returns:
        Sanitized match data with all required fields populated with defaults if missing
    """
    if not match_data:
        match_data = {}
        
    defaults = {
        'home_team': 'Unknown Team',
        'away_team': 'Unknown Team',
        'home_score': 0,
        'away_score': 0,
        'match_time': 'TBD',
        'league': 'Unknown League',
        'stadium': 'Unknown Stadium',
        'home_win_probability': 0.33,
        'draw_probability': 0.33,
        'away_win_probability': 0.33,
    }
    
    # Apply defaults for missing fields
    for key, value in defaults.items():
        if key not in match_data or match_data[key] is None:
            match_data[key] = value
    
    return match_data

@contextmanager
def card(key=None, padding="20px", margin="10px 0", border_radius="12px", card_type="default", hover_effect=True, animate=True):
    """
    Create a styled card component as a context manager for Streamlit.
    
    Args:
        key: Optional key for the container
        padding: CSS padding value
        margin: CSS margin value
        border_radius: CSS border-radius value
        card_type: Type of card (default, primary, success, warning, danger, info, gradient)
        hover_effect: Whether to apply hover effects
        animate: Whether to animate the card on hover
    """
    # Get background and border colors based on card type
    bg_color = "var(--color-card)"
    border_color = "var(--color-border)"
    shadow_opacity = "0.05"
    text_color = "var(--color-text)"
    
    card_styles = {
        "primary": ("var(--color-primary)", "var(--color-primary)", "white"),
        "success": ("var(--color-success)", "var(--color-success)", "white"),
        "warning": ("var(--color-warning)", "var(--color-warning)", "#333"),
        "danger": ("var(--color-danger)", "var(--color-danger)", "white"),
        "info": ("var(--color-info)", "var(--color-info)", "white"),
        "gradient": ("linear-gradient(45deg, var(--color-primary), var(--color-secondary))", "transparent", "white")
    }

    if card_type in card_styles:
        bg_color, border_color, text_color = card_styles[card_type]
    
    # Hover and animation effects
    hover_style = ""
    if hover_effect:
        hover_scale = "scale(1.02)" if animate else "none"
        hover_shadow = "0 6px 15px rgba(0,0,0,0.1)"
        hover_style = f"""
        .card-custom:hover {{  
            transform: {hover_scale};
            box-shadow: {hover_shadow};
            border-color: var(--color-primary);
        }}
        """
    
    # Use Streamlit's native container with safe styling
    # Note: Custom CSS styling removed for security - using Streamlit's built-in components
    with st.container():
        # Create a visual card effect using Streamlit's native styling
        if card_type == "primary":
            st.info("", icon="🔵")
        elif card_type == "success":
            st.success("", icon="✅")
        elif card_type == "warning":
            st.warning("", icon="⚠️")
        elif card_type == "danger":
            st.error("", icon="❌")
        else:
            # Use expander for card-like appearance
            with st.expander("", expanded=True):
                yield
                return

        yield


def styled_card(content, padding="20px", margin="10px 0", border_radius="12px", card_type="default", hover_effect=True, animate=True):
    """
    Create a styled card component with custom content and multiple design options.
    
    Args:
        content: HTML or markdown content for the card
        padding: CSS padding value
        margin: CSS margin value
        border_radius: CSS border-radius value
        card_type: Type of card (default, primary, success, warning, danger, info, gradient)
        hover_effect: Whether to apply hover effects
        animate: Whether to animate the card on hover
        
    Returns:
        HTML string for the card
    """
    # Get background and border colors based on card type
    bg_color = "var(--color-card)"
    border_color = "var(--color-border)"
    text_color = "var(--color-text)"
    
    card_styles = {
        "primary": ("var(--color-primary)", "var(--color-primary)", "white"),
        "success": ("var(--color-success)", "var(--color-success)", "white"),
        "warning": ("var(--color-warning)", "var(--color-warning)", "#333"),
        "danger": ("var(--color-danger)", "var(--color-danger)", "white"),
        "info": ("var(--color-info)", "var(--color-info)", "white"),
        "gradient": ("linear-gradient(45deg, var(--color-primary), var(--color-secondary))", "transparent", "white")
    }

    if card_type in card_styles:
        bg_color, border_color, text_color = card_styles[card_type]
    
    # Hover and animation effects
    hover_style = ""
    if hover_effect:
        hover_scale = "scale(1.02)" if animate else "none"
        hover_shadow = "0 6px 15px rgba(0,0,0,0.1)"
        hover_style = ".card-custom:hover { transform: " + hover_scale + "; box-shadow: " + hover_shadow + "; border-color: var(--color-primary); }"
    
    # Generate a unique ID for this card
    card_id = "card_" + uuid.uuid4().hex[:8]
    
    # Use string concatenation instead of f-strings to avoid any parsing issues
    style_content = "#" + card_id + " { background: " + bg_color + "; color: " + text_color + "; padding: " + padding + "; margin: " + margin + "; border-radius: " + border_radius + "; box-shadow: 0 3px 10px rgba(0,0,0,0.05); border: 1px solid " + border_color + "; transition: all var(--transition-speed) ease-in-out; }"
    
    html_content = "<style>" + style_content + " " + hover_style + "</style><div id='" + card_id + "' class='card-custom'>" + content + "</div>"
    
    return html_content


def header(text, level=1, color=None, align="left", margin="20px 0 10px 0", icon=None, 
           subtitle=None, divider=True, accent_bar=True, animation='slide'):
    """
    Create a modern styled header component with multiple design options.
    
    Args:
        text: Header text
        level: Header level (1-6)
        color: Text color
        align: Text alignment
        margin: CSS margin value
        icon: Optional icon to display before text
        subtitle: Optional subtitle text
        divider: Whether to add a divider after the header
        accent_bar: Whether to add a colored accent bar
        animation: Animation effect (None, 'fade', 'slide')
    """
    header_id = "header_" + uuid.uuid4().hex[:8]
    
    if not color:
        color = "var(--color-heading)"
    
    icon_html = '<span style="margin-right: 12px; font-size: 1.2em;">' + (icon or "") + '</span>' if icon else ""
    subtitle_html = '<div style="color:var(--color-text-secondary); font-size:0.9em; margin-top:5px;">' + (subtitle or "") + '</div>' if subtitle else ""
    
    # Accent bar styles
    accent_bar_html = ""
    if accent_bar:
        accent_bar_html = '<div style="width: 80px; height: 4px; background: linear-gradient(90deg, var(--color-primary), var(--color-secondary)); border-radius: 2px; margin-top: 8px;"></div>'

    # Animation
    animation_css = ""
    if animation:
        if animation == 'fade':
            animation_css = "animation: fadeIn 0.8s ease-out;"
        elif animation == 'slide':
            animation_css = "animation: slideIn 0.5s ease-out;"
    
    # Combine all parts using string concatenation
    html_parts = []
    html_parts.append("<style>")
    html_parts.append("@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }")
    html_parts.append("@keyframes slideIn { from { transform: translateY(-10px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }")
    html_parts.append("#" + header_id + " {")
    html_parts.append("    margin: " + margin + ";")
    html_parts.append("    color: " + color + ";")
    html_parts.append("    text-align: " + align + ";")
    html_parts.append("    " + animation_css)
    html_parts.append("}")
    html_parts.append("</style>")
    
    # Create the header with the correct level
    html_parts.append('<div id="' + header_id + '">')
    html_parts.append('<h' + str(level) + '>' + icon_html + text + '</h' + str(level) + '>')
    html_parts.append(subtitle_html)
    html_parts.append(accent_bar_html)
    
    if divider:
        html_parts.append('<hr style="margin-top: 15px; opacity: 0.2;">')
    
    html_parts.append('</div>')
    
    return "\n".join(html_parts)


def badge(text, type="default", size="medium", icon=None, rounded=True, pulse=False, tooltip=None):
    """
    Create a modern styled badge component with multiple design options.
    
    Args:
        text: Badge text
        type: Badge type (default, primary, success, warning, danger, info)
        size: Badge size (small, medium, large)
        icon: Optional icon to display before text
        rounded: Whether to use rounded corners
        pulse: Whether to add pulse animation
        tooltip: Optional tooltip text
        
    Returns:
        HTML string for the badge
    """
    import uuid
    badge_id = "badge_" + uuid.uuid4().hex[:8]
    
    # Determine colors based on type
    if type == "primary":
        bg_color = "var(--color-primary)"
        text_color = "white"
    elif type == "success":
        bg_color = "var(--color-success)"
        text_color = "white"
    elif type == "warning":
        bg_color = "var(--color-warning)"
        text_color = "#333"
    elif type == "danger":
        bg_color = "var(--color-danger)"
        text_color = "white"
    elif type == "info":
        bg_color = "var(--color-info)"
        text_color = "white"
    else:  # default
        bg_color = "var(--color-text-secondary)"
        text_color = "white"
    
    # Size properties
    if size == "small":
        padding = "0.2em 0.6em"
        font_size = "0.75em"
    elif size == "large":
        padding = "0.5em 1em"
        font_size = "1em"
    else:  # medium
        padding = "0.35em 0.8em"
        font_size = "0.85em"
    
    # Border radius
    border_radius = "30px" if rounded else "4px"
    
    # Icon
    icon_html = (icon + " ") if icon else ""
    
    # Pulse animation
    if pulse:
        pulse_css = """
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    """
        pulse_css += "#" + badge_id + " {\n"
        pulse_css += "        animation: pulse 2s infinite;\n"
        pulse_css += "    }"
    else:
        pulse_css = ""
    
    # Tooltip
    tooltip_attr = "title=\"" + str(tooltip) + "\" data-toggle=\"tooltip\"" if tooltip else ""
    
    # Build HTML using string concatenation
    style_content = "#" + badge_id + " { display: inline-block; padding: " + padding + "; font-size: " + font_size + "; font-weight: 600; line-height: 1; text-align: center; white-space: nowrap; vertical-align: baseline; border-radius: " + border_radius + "; background-color: " + bg_color + "; color: " + text_color + "; transition: all var(--transition-speed) ease-in-out; }"
    hover_style = "#" + badge_id + ":hover { opacity: 0.9; }"
    
    html_content = "<style>" + style_content + " " + hover_style + " " + pulse_css + "</style><span id='" + badge_id + "' " + tooltip_attr + ">" + icon_html + str(text) + "</span>"
    
    return html_content

def spinner_wrapper(message="Loading..."):
    """
    Decorator that adds a spinner while a function executes.
    
    Args:
        message: Message to display during loading
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with st.spinner(message):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                logger.debug(f"Function {func.__name__} completed in {end_time - start_time:.2f}s")
            return result
        return wrapper
    return decorator

def load_custom_css():
    """
    Load custom CSS styles - DEPRECATED: Use UnifiedGoalDiggersDesignSystem instead.
    This function now delegates to the unified design system for consistency.
    """
    # Import and use the unified design system
    try:
        from dashboard.components.consistent_styling import \
            get_unified_design_system
        design_system = get_unified_design_system()
        design_system.apply_unified_styling(dashboard_tier='integrated', enable_animations=True)
        return
    except ImportError:
        pass

    # Fallback minimal CSS for backward compatibility
    css_content = """
        /* Minimal fallback CSS - Use UnifiedGoalDiggersDesignSystem for full features */
        .stApp header { display: none !important; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .animated-fade { animation: fadeIn 0.8s ease-in-out; }
    """

    # Apply the minimal CSS
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    logger.info("✅ Fallback CSS applied - Recommend using UnifiedGoalDiggersDesignSystem")

def info_tooltip(content, icon="ℹ️", placement="top", style="icon", max_width="300px", color=None):
    """
    Create an enhanced tooltip component with multiple design options.
    
    Args:
        content: Tooltip content
        icon: Icon to display
        placement: Tooltip placement (top, bottom, left, right)
        style: Tooltip style (icon, text, dotted, question)
        max_width: Maximum width of the tooltip
        color: Color for the tooltip indicator
        
    Returns:
        HTML string for the tooltip
    """
    tooltip_id = "tooltip_" + uuid.uuid4().hex[:8]
    
    if not color:
        color = "var(--color-primary)"
    
    # Determine tooltip style
    if style == "text":
        trigger = "<span style='color:" + color + "; text-decoration:underline; cursor:help;'>" + icon + "</span>"
    elif style == "dotted":
        trigger = "<span style='border-bottom:1px dotted " + color + "; cursor:help;'>" + icon + "</span>"
    elif style == "question":
        trigger = "<span style='display:inline-block; width:16px; height:16px; line-height:16px; text-align:center; background-color:" + color + "; color:white; border-radius:50%; font-size:10px; cursor:help;'>?</span>"
    else:  # icon
        trigger = "<span style='color:" + color + "; cursor:help;'>" + icon + "</span>"
    
    # Use string concatenation for CSS
    tooltip_css = """
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        opacity: 0;
        transition: opacity 0.3s;
        white-space: normal;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        max-width: """ + max_width + """;
        line-height: 1.4;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    """
    
    # Adjust positioning based on placement
    if placement == "top":
        position_css = ".tooltip .tooltiptext { bottom: 125%; left: 50%; margin-left: -100px; }"
    elif placement == "bottom":
        position_css = ".tooltip .tooltiptext { top: 125%; left: 50%; margin-left: -100px; }"
    elif placement == "left":
        position_css = ".tooltip .tooltiptext { top: -5px; right: 105%; }"
    else:  # right
        position_css = ".tooltip .tooltiptext { top: -5px; left: 105%; }"
    
    # Build the HTML
    html_parts = []
    html_parts.append("<style>")
    html_parts.append(tooltip_css)
    html_parts.append(position_css)
    html_parts.append("#" + tooltip_id + " { display: inline; }")
    html_parts.append("</style>")
    
    html_parts.append("<div id='" + tooltip_id + "' class='tooltip'>")
    html_parts.append(trigger)
    html_parts.append("<span class='tooltiptext'>" + str(content) + "</span>")
    html_parts.append("</div>")
    
    return "".join(html_parts)

def progress_indicator(value, max_value=100, color=None, style="bar", label=None, size="medium", show_percentage=True):
    """
    Create an enhanced progress indicator with multiple design options.
    
    Args:
        value: Current value
        max_value: Maximum value
        color: Color of the progress indicator
        style: Style of the indicator (bar, circle, pill)
        label: Label text
        size: Size of the indicator (small, medium, large)
        show_percentage: Whether to show the percentage value
        
    Returns:
        HTML string for the progress indicator
    """
    # Generate a unique ID for this progress indicator
    progress_id = "progress_" + uuid.uuid4().hex[:8]
    
    # Calculate percentage
    percent = min(100, max(0, int((value / max_value) * 100))) if max_value > 0 else 0
    
    if not color:
        # Color based on percentage
        if percent < 30:
            color = "var(--color-danger)"
        elif percent < 70:
            color = "var(--color-warning)"
        else:
            color = "var(--color-success)"
    
    # Size properties
    if size == "small":
        height = "6px"
        font_size = "12px"
    elif size == "large":
        height = "15px"
        font_size = "16px"
    else:  # medium
        height = "10px"
        font_size = "14px"
    
    # Label and percentage display
    label_html = "<div style='margin-bottom: 5px; font-weight: 500;'>" + str(label) + "</div>" if label else ""
    percent_html = "<div style='margin-left: 10px; font-size: " + font_size + ";'>" + str(percent) + "%</div>" if show_percentage else ""
    
    html_parts = []
    html_parts.append("<style>")
    
    if style == "circle":
        # Circle progress indicator using CSS conic-gradient
        html_parts.append("#" + progress_id + " .circle-progress {")
        html_parts.append("  width: 60px;")
        html_parts.append("  height: 60px;")
        html_parts.append("  border-radius: 50%;")
        html_parts.append("  background: conic-gradient(" + color + " " + str(percent) + "%, #e0e0e0 0);")
        html_parts.append("  display: flex;")
        html_parts.append("  align-items: center;")
        html_parts.append("  justify-content: center;")
        html_parts.append("}")
        html_parts.append("#" + progress_id + " .circle-inset {")
        html_parts.append("  width: 45px;")
        html_parts.append("  height: 45px;")
        html_parts.append("  border-radius: 50%;")
        html_parts.append("  background: white;")
        html_parts.append("  display: flex;")
        html_parts.append("  align-items: center;")
        html_parts.append("  justify-content: center;")
        html_parts.append("  font-weight: bold;")
        html_parts.append("}")
        html_parts.append("</style>")
        
        html_parts.append("<div id='" + progress_id + "'>")
        html_parts.append(label_html)
        html_parts.append("<div style='display:flex; align-items:center;'>")
        html_parts.append("  <div class='circle-progress'>")
        html_parts.append("    <div class='circle-inset'>" + str(percent) + "%</div>")
        html_parts.append("  </div>")
        html_parts.append("</div>")
        html_parts.append("</div>")
    
    elif style == "pill":
        # Pill style progress bar with rounded corners
        html_parts.append("#" + progress_id + " .pill-progress {")
        html_parts.append("  width: 100%;")
        html_parts.append("  height: " + height + ";")
        html_parts.append("  background-color: #e0e0e0;")
        html_parts.append("  border-radius: 20px;")
        html_parts.append("  overflow: hidden;")
        html_parts.append("}")
        html_parts.append("#" + progress_id + " .pill-bar {")
        html_parts.append("  height: 100%;")
        html_parts.append("  width: " + str(percent) + "%;")
        html_parts.append("  background-color: " + color + ";")
        html_parts.append("  border-radius: 20px;")
        html_parts.append("  transition: width 0.5s ease-in-out;")
        html_parts.append("}")
        html_parts.append("</style>")
        
        html_parts.append("<div id='" + progress_id + "'>")
        html_parts.append(label_html)
        html_parts.append("<div style='display:flex; align-items:center;'>")
        html_parts.append("  <div class='pill-progress' style='flex-grow:1;'>")
        html_parts.append("    <div class='pill-bar'></div>")
        html_parts.append("  </div>")
        html_parts.append(percent_html)
        html_parts.append("</div>")
        html_parts.append("</div>")
    
    else:  # Standard bar
        # Standard progress bar
        html_parts.append("#" + progress_id + " .progress-bar {")
        html_parts.append("  width: 100%;")
        html_parts.append("  height: " + height + ";")
        html_parts.append("  background-color: #e0e0e0;")
        html_parts.append("  border-radius: 3px;")
        html_parts.append("  overflow: hidden;")
        html_parts.append("}")
        html_parts.append("#" + progress_id + " .bar {")
        html_parts.append("  height: 100%;")
        html_parts.append("  width: " + str(percent) + "%;")
        html_parts.append("  background-color: " + color + ";")
        html_parts.append("  transition: width 0.5s ease-in-out;")
        html_parts.append("}")
        html_parts.append("</style>")
        
        html_parts.append("<div id='" + progress_id + "'>")
        html_parts.append(label_html)
        html_parts.append("<div style='display:flex; align-items:center;'>")
        html_parts.append("  <div class='progress-bar' style='flex-grow:1;'>")
        html_parts.append("    <div class='bar'></div>")
        html_parts.append("  </div>")
        html_parts.append(percent_html)
        html_parts.append("</div>")
        html_parts.append("</div>")
    
    return "".join(html_parts)

def collapsible_section(title, content, expanded=False, key=None, icon=None, style="default", color=None):
    """
    Create a collapsible section component with custom content and design options.
    
    Args:
        title: Section title
        content: Section content (HTML or markdown)
        expanded: Whether the section is expanded by default
        key: Unique key for the section
        icon: Optional icon for the section header
        style: Style of the section (default, outline, subtle, card)
        color: Accent color for the section
        
    Returns:
        HTML string for the collapsible section
    """
    section_id = "section_" + uuid.uuid4().hex[:8]
    content_id = "content_" + section_id
    
    if not color:
        color = "var(--color-primary)"
    
    # Icon
    icon_html = (icon + " ") if icon else ""
    
    # Determine styling based on style parameter
    if style == "outline":
        border = "1px solid #ddd"
        background = "transparent"
        header_bg = "transparent"
    elif style == "subtle":
        border = "none"
        background = "transparent"
        header_bg = "#f7f7f7"
    elif style == "card":
        border = "1px solid #ddd"
        background = "white"
        header_bg = "white"
    else:  # default
        border = "1px solid #ddd"
        background = "white"
        header_bg = "#f7f7f7"
    
    # Create the HTML for the collapsible section
    html_parts = []
    
    # CSS for the collapsible section
    html_parts.append("<style>")
    html_parts.append("#" + section_id + " {")
    html_parts.append("  border: " + border + ";")
    html_parts.append("  border-radius: 4px;")
    html_parts.append("  margin-bottom: 10px;")
    html_parts.append("  background: " + background + ";")
    html_parts.append("}")
    html_parts.append("#" + section_id + " .section-header {")
    html_parts.append("  padding: 10px 15px;")
    html_parts.append("  background: " + header_bg + ";")
    html_parts.append("  cursor: pointer;")
    html_parts.append("  display: flex;")
    html_parts.append("  justify-content: space-between;")
    html_parts.append("  align-items: center;")
    html_parts.append("  transition: background-color 0.2s;")
    html_parts.append("}")
    html_parts.append("#" + section_id + " .section-header:hover {")
    html_parts.append("  background-color: rgba(0,0,0,0.05);")
    html_parts.append("}")
    html_parts.append("#" + section_id + " .section-title {")
    html_parts.append("  font-weight: 500;")
    html_parts.append("  color: " + color + ";")
    html_parts.append("}")
    html_parts.append("#" + section_id + " .section-content {")
    html_parts.append("  padding: 15px;")
    html_parts.append("  display: " + ("block" if expanded else "none") + ";")
    html_parts.append("}")
    html_parts.append("#" + section_id + " .arrow {")
    html_parts.append("  transition: transform 0.3s;")
    html_parts.append("  transform: rotate(" + ("90" if expanded else "0") + "deg);")
    html_parts.append("}")
    html_parts.append("</style>")
    
    # JavaScript for the collapsible section
    html_parts.append("<script>")
    html_parts.append("function toggleSection(sectionId, contentId) {")
    html_parts.append("  const content = document.getElementById(contentId);")
    html_parts.append("  const arrow = document.querySelector('#' + sectionId + ' .arrow');")
    html_parts.append("  if (content.style.display === 'none' || content.style.display === '') {")
    html_parts.append("    content.style.display = 'block';")
    html_parts.append("    arrow.style.transform = 'rotate(90deg)';")
    html_parts.append("  } else {")
    html_parts.append("    content.style.display = 'none';")
    html_parts.append("    arrow.style.transform = 'rotate(0deg)';")
    html_parts.append("  }")
    html_parts.append("}")
    html_parts.append("</script>")
    
    # HTML for the collapsible section
    html_parts.append("<div id='" + section_id + "'>")
    html_parts.append("  <div class='section-header' onclick=\"toggleSection('" + section_id + "', '" + content_id + "')\">")
    html_parts.append("    <div class='section-title'>" + icon_html + sanitize_for_html(title) + "</div>")
    html_parts.append("    <div class='arrow'>▶</div>")
    html_parts.append("  </div>")
    html_parts.append("  <div id='" + content_id + "' class='section-content'>")
    html_parts.append(content)
    html_parts.append("  </div>")
    html_parts.append("</div>")
    
    return "".join(html_parts)

def create_themed_card(title, content, sub_header=None, border_color=None, bg_color=None, icon=None, hover_effect=True):
    """
    Create a themed card with title and content.
    
    Args:
        title: Card title
        content: Card content
        sub_header: Optional subtitle
        border_color: Color for the card border
        bg_color: Background color
        icon: Optional icon
        hover_effect: Whether to apply hover effect
        
    Returns:
        HTML string for the card
    """
    if not border_color:
        border_color = "var(--color-primary)"
    if not bg_color:
        bg_color = "white"
    
    # Create subtitle if provided
    sub_header_html = ""
    if sub_header:
        sub_header_html = "<div style='font-size: 0.85em; color: var(--color-text-secondary); margin-bottom: 10px;'>" + str(sub_header) + "</div>"
    
    # Create icon if provided
    icon_html = ""
    if icon:
        icon_html = "<div style='float: right; font-size: 1.5em;'>" + str(icon) + "</div>"
    
    # Combine title, subtitle, and content
    card_content = "<div style='margin-bottom: 15px;'>"
    card_content += icon_html
    card_content += "<h3 style='margin: 0; padding: 0; font-size: 1.2em; font-weight: 500;'>" + str(title) + "</h3>"
    card_content += sub_header_html
    card_content += "</div>"
    card_content += content
    
    # Create the card using the styled_card function
    return styled_card(
        card_content,
        padding="20px", 
        border_radius="8px",
        hover_effect=hover_effect
    )

def create_metric_card(title, value, description=None, delta=None, delta_color="normal", tooltip=None, icon=None, trend=None, help_text=None):
    """
    Create a metric card for displaying KPIs and important metrics.
    
    Args:
        title: Metric title
        value: Metric value
        description: Additional description
        delta: Change value (can be percentage or absolute)
        delta_color: Color scheme for delta (normal, inverted)
        tooltip: Optional tooltip for the metric
        icon: Optional icon to display
        trend: Trend direction (up, down, neutral)
        help_text: Help text for the metric
        
    Returns:
        HTML string for the metric card
    """
    # Generate a unique ID for this card
    card_id = "metric_" + uuid.uuid4().hex[:8]
    
    # Process the delta value and determine color
    delta_html = ""
    if delta is not None:
        # Determine if the delta is a percentage or absolute value
        if isinstance(delta, str) and "%" in delta:
            delta_value = delta
        else:
            delta_value = str(delta)
            if not delta_value.startswith("+") and not delta_value.startswith("-"):
                delta_value = "+" + delta_value
        
        # Determine color based on delta value and color scheme
        delta_sign = delta_value[0]
        if delta_color == "inverted":
            # For metrics where decrease is positive (e.g., cost reduction)
            delta_text_color = "var(--color-success)" if delta_sign == "-" else "var(--color-danger)"
        else:
            # For metrics where increase is positive (e.g., revenue)
            delta_text_color = "var(--color-success)" if delta_sign == "+" else "var(--color-danger)"
        
        # Create delta HTML
        delta_html = "<div style='color: " + delta_text_color + "; display: flex; align-items: center; margin-top: 5px;'>"
        
        # Add trend arrow if specified
        if trend == "up":
            delta_html += "<span style='margin-right: 5px;'>↑</span>"
        elif trend == "down":
            delta_html += "<span style='margin-right: 5px;'>↓</span>"
        
        delta_html += delta_value + "</div>"
    
    # Help text tooltip
    help_html = ""
    if help_text:
        help_html = info_tooltip(help_text, icon="ℹ️", style="icon", placement="top")
    
    # Description
    description_html = ""
    if description:
        description_html = "<div style='color: var(--color-text-secondary); font-size: 0.9em; margin-top: 5px;'>" + str(description) + "</div>"
    
    # Icon
    icon_html = ""
    if icon:
        icon_html = "<div style='margin-bottom: 10px; font-size: 1.5em;'>" + str(icon) + "</div>"
    
    # Build the metric card content
    card_content = ""
    card_content += "<div id='" + card_id + "' class='metric-card'>"
    card_content += icon_html
    card_content += "<div style='display: flex; align-items: center; justify-content: space-between;'>"
    card_content += "<div style='font-size: 0.9em; color: var(--color-text-secondary);'>" + str(title)
    if help_html:
        card_content += " <span style='margin-left: 5px;'>" + help_html + "</span>"
    card_content += "</div>"
    card_content += "</div>"
    card_content += "<div style='font-size: 1.8em; font-weight: 500; margin-top: 10px;'>" + str(value) + "</div>"
    card_content += delta_html
    card_content += description_html
    card_content += "</div>"
    
    # Create the card using the styled_card function
    return styled_card(card_content, padding="20px", hover_effect=True)

def create_info_card(title, content, icon=None, color=None, border=True):
    """
    Create an information card with a title and content.
    
    Args:
        title: Card title
        content: Card content
        icon: Optional icon
        color: Color theme for the card
        border: Whether to add a border
        
    Returns:
        HTML string for the info card
    """
    # Set default color if not provided
    if not color:
        color = "var(--color-info)"
    
    # Icon HTML
    icon_html = icon if icon else ""
    
    # Build the card content
    card_content = "<div>"
    if title:
        card_content += "<div style='display: flex; align-items: center; margin-bottom: 10px;'>"
        if icon_html:
            card_content += "<div style='margin-right: 10px; color: " + color + ";'>" + icon_html + "</div>"
        card_content += "<h3 style='margin: 0; padding: 0; font-size: 1.2em; color: " + color + ";'>" + str(title) + "</h3>"
        card_content += "</div>"
    
    card_content += "<div>" + str(content) + "</div>"
    card_content += "</div>"
    
    # Create the card using styled_card
    return styled_card(
        card_content, 
        card_type="default" if border else "gradient",
        border_radius="8px",
        hover_effect=False
    )

# Additional UI components can be added as needed
