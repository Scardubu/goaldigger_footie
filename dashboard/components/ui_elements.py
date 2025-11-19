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
def card(title=None, key=None):
    """
    Create a styled card component as a context manager for Streamlit.
    This is simplified to use st.container with a border.
    """
    
    # The use of st.markdown for styling is a common workaround.
    # A container with a border is a requested feature in Streamlit.
    st.markdown(
        f"""
        <style>
            .custom-card-container {{
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # We use a container to group the elements, and apply the class via markdown.
    with st.container():
        st.markdown('<div class="custom-card-container">', unsafe_allow_html=True)
        
        if title:
            st.subheader(title)
        
        yield
        
        st.markdown('</div>', unsafe_allow_html=True)


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
    Create a modern styled header component using Streamlit native components.
    
    Args:
        text: Header text
        level: Header level (1-6) - used to determine header size
        color: Text color (note: direct color application is limited in markdown)
        align: Text alignment
        margin: CSS margin value (note: will be applied via markdown)
        icon: Optional icon to display before text
        subtitle: Optional subtitle text
        divider: Whether to add a divider after the header
        accent_bar: Whether to add a colored accent bar
        animation: Animation effect (note: animations are not supported with this simplified version)
    """
    
    # Map header level to Streamlit's header functions
    if level == 1:
        header_func = st.header
    elif level == 2:
        header_func = st.subheader
    else:
        header_func = st.write # Fallback for smaller headers

    # Icon and text
    display_text = f"{icon} {text}" if icon else text
    
    header_func(display_text)

    if subtitle:
        st.markdown(f"<p style='color:var(--color-text-secondary); margin-top:-10px;'>{subtitle}</p>", unsafe_allow_html=True)

    if accent_bar:
        st.markdown(
            """
            <div style="width: 80px; height: 4px; background: linear-gradient(90deg, var(--color-primary), var(--color-secondary)); border-radius: 2px; margin-top: 8px;"></div>
            """,
            unsafe_allow_html=True
        )

    if divider:
        st.divider()


def badge(text, badge_type="default", tooltip=None):
    """
    Creates a simple styled badge using Streamlit markdown.

    Args:
        text (str): The text to display in the badge.
        badge_type (str, optional): The type of badge ('default', 'primary', 'success', 'warning', 'danger', 'info'). 
                                    Defaults to "default".
        tooltip (str, optional): Tooltip text to show on hover. Defaults to None.
    """
    
    color_map = {
        "primary": "blue",
        "success": "green",
        "warning": "orange",
        "danger": "red",
        "info": "cyan",
        "default": "gray"
    }
    color = color_map.get(badge_type, "gray")

    st.markdown(f'<span style="background-color:{color};color:white;padding:2px 8px;border-radius:12px;" title="{tooltip or ""}">{text}</span>', unsafe_allow_html=True)

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

def collapsible_section(title, content_callback, expanded=False, key=None, icon=None, style="default", color=None):
    """
    Create a collapsible section component using st.expander.
    
    Args:
        title: Section title
        content_callback: A function that renders the content when the section is expanded.
        expanded: Whether the section is expanded by default
        key: Unique key for the section
        icon: Optional icon for the section header
        style: Style of the section (note: styling options are limited with st.expander)
        color: Accent color for the section (note: not directly supported)
        
    Returns:
        None
    """
    display_title = f"{icon} {title}" if icon else title
    
    with st.expander(display_title, expanded=expanded):
        content_callback()

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
    Create a metric card for displaying KPIs and important metrics using st.metric.
    
    Args:
        title: Metric title
        value: Metric value
        description: Additional description (not directly supported by st.metric, will be added as markdown)
        delta: Change value (can be percentage or absolute)
        delta_color: Color scheme for delta (normal, inverse)
        tooltip: Optional tooltip for the metric (used as help text)
        icon: Optional icon to display (will be prepended to title)
        trend: Trend direction (up, down, neutral) - st.metric handles this automatically
        help_text: Help text for the metric
        
    Returns:
        None
    """
    
    # Combine icon and title
    display_title = f"{icon} {title}" if icon else title

    # Use tooltip as help text if help_text is not provided
    help_content = help_text or tooltip

    # st.metric handles delta formatting automatically.
    # The 'delta' parameter in st.metric expects a string or number.
    # The 'delta_color' can be "normal", "inverse", or "off".
    st.metric(
        label=display_title,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_content
    )

    if description:
        st.markdown(f"<p style='color:var(--color-text-secondary);'>{description}</p>", unsafe_allow_html=True)

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

def render_banner(width="100%", container=None):
    """
    Renders the GoalDiggers banner at the top of the page.
    
    Args:
        width: Width of the banner (default: 100%)
        container: Optional container to render the banner in
    
    Returns:
        None
    """
    banner_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "images", "GoalDiggers_banner.png")
    
    if not os.path.exists(banner_path):
        # Create a simple text banner as fallback
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1e3c72, #2a5298); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <h1 style="margin: 0; padding: 0; font-size: 2.5rem;">⚽ GoalDiggers</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2rem;">Football Betting Insights</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    try:
        # Ensure width is properly handled for Streamlit
        if width == "100%" or width.lower() == "full":
            # Use None for full width
            if container:
                container.image(banner_path)
            else:
                st.image(banner_path)
        else:
            # Try to convert to integer if it's a numeric string
            try:
                numeric_width = int(width.rstrip("px%"))
                if container:
                    container.image(banner_path, width=numeric_width)
                else:
                    st.image(banner_path, width=numeric_width)
            except (ValueError, AttributeError):
                # Fallback to None (full width) if conversion fails
                if container:
                    container.image(banner_path)
                else:
                    st.image(banner_path)
    except Exception as e:
        # Fallback to text banner in case of error
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #1e3c72, #2a5298); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <h1 style="margin: 0; padding: 0; font-size: 2.5rem;">⚽ GoalDiggers</h1>
            <p style="margin: 10px 0 0 0; font-size: 1.2rem;">Football Betting Insights</p>
        </div>
        """, unsafe_allow_html=True)

# Additional UI components can be added as needed


class UIElements:
    """
    UI Elements class that provides a unified interface to access UI components.
    This class wraps the standalone UI element functions for backward compatibility.
    """
    
    @staticmethod
    def render_banner(width="100%", container=None):
        """ 
        Renders the GoalDiggers banner at the top of the page.
        Wrapper for the standalone render_banner function.
        """
        return render_banner(width=width, container=container)
    
    @staticmethod
    def create_metric_card(title, value, description=None, delta=None, delta_color="normal", tooltip=None, icon=None, trend=None, help_text=None):
        """
        Create a metric card for displaying KPIs and important metrics.
        Wrapper for the standalone create_metric_card function.
        """
        return create_metric_card(
            title=title, value=value, description=description, delta=delta,
            delta_color=delta_color, tooltip=tooltip, icon=icon,
            trend=trend, help_text=help_text
        )
    
    @staticmethod
    def create_info_card(title, content, icon=None, color=None, border=True):
        """
        Create an information card with a title and content.
        Wrapper for the standalone create_info_card function.
        """
        return create_info_card(title=title, content=content, icon=icon, color=color, border=border)
    
    @staticmethod
    def styled_card(content, padding="20px", margin="10px 0", border_radius="12px", card_type="default", hover_effect=True, animate=True):
        """
        Create a styled card component with custom content and multiple design options.
        Wrapper for the standalone styled_card function.
        """
        return styled_card(
            content=content, padding=padding, margin=margin, 
            border_radius=border_radius, card_type=card_type, 
            hover_effect=hover_effect, animate=animate
        )
    
    @staticmethod
    def badge(text, badge_type="default", tooltip=None):
        """
        Creates a simple styled badge using Streamlit markdown.
        Wrapper for the standalone badge function.
        """
        return badge(text=text, badge_type=badge_type, tooltip=tooltip)
    
    @staticmethod
    def header(text, level=1, color=None, align="left", margin="20px 0 10px 0", icon=None,
               subtitle=None, divider=True, accent_bar=True, animation='slide'):
        """
        Create a modern styled header component.
        Wrapper for the standalone header function.
        """
        return header(
            text=text, level=level, color=color, align=align, margin=margin,
            icon=icon, subtitle=subtitle, divider=divider, accent_bar=accent_bar,
            animation=animation
        )
    
    @staticmethod
    def progress_indicator(value, max_value=100, color=None, style="bar", label=None, size="medium", show_percentage=True):
        """
        Create an enhanced progress indicator with multiple design options.
        Wrapper for the standalone progress_indicator function.
        """
        return progress_indicator(
            value=value, max_value=max_value, color=color, style=style,
            label=label, size=size, show_percentage=show_percentage
        )
    
    @staticmethod
    def info_tooltip(content, icon="ℹ️", placement="top", style="icon", max_width="300px", color=None):
        """
        Create an enhanced tooltip component with multiple design options.
        Wrapper for the standalone info_tooltip function.
        """
        return info_tooltip(
            content=content, icon=icon, placement=placement, 
            style=style, max_width=max_width, color=color
        )