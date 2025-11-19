#!/usr/bin/env python3
"""
Loading States and Skeleton Components
Provides professional loading indicators and skeleton loaders
"""

from typing import Optional

import streamlit as st


def render_skeleton_card(height: str = "200px", animation: bool = True) -> None:
    """Render a skeleton loading card with optional pulse animation."""
    animation_css = """
        @keyframes skeleton-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
    """ if animation else ""
    
    st.markdown(f"""
    <style>
        {animation_css}
        .skeleton-card {{
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: skeleton-pulse 1.5s ease-in-out infinite;
            border-radius: 8px;
            height: {height};
            margin: 10px 0;
        }}
    </style>
    <div class="skeleton-card"></div>
    """, unsafe_allow_html=True)


def render_loading_spinner(message: str = "Loading data...", size: str = "medium") -> None:
    """Render a branded loading spinner with custom message."""
    size_map = {
        "small": "30px",
        "medium": "50px",
        "large": "70px"
    }
    spinner_size = size_map.get(size, "50px")
    
    st.markdown(f"""
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .loading-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
        }}
        .loading-spinner {{
            border: 4px solid #f3f3f3;
            border-top: 4px solid #1f4e79;
            border-radius: 50%;
            width: {spinner_size};
            height: {spinner_size};
            animation: spin 1s linear infinite;
        }}
        .loading-message {{
            margin-top: 20px;
            color: #1f4e79;
            font-size: 16px;
            font-weight: 500;
        }}
    </style>
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-message">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_progress_bar(progress: float, message: Optional[str] = None) -> None:
    """Render a styled progress bar with optional message.
    
    Args:
        progress: Progress value between 0.0 and 1.0
        message: Optional progress message
    """
    progress_percent = int(progress * 100)
    
    message_html = f'<div class="progress-message">{message}</div>' if message else ''
    
    st.markdown(f"""
    <style>
        .progress-container {{
            width: 100%;
            padding: 20px 0;
        }}
        .progress-bar-bg {{
            width: 100%;
            height: 24px;
            background-color: #e0e0e0;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }}
        .progress-bar-fill {{
            height: 100%;
            width: {progress_percent}%;
            background: linear-gradient(90deg, #1f4e79 0%, #2a5298 100%);
            border-radius: 12px;
            transition: width 0.3s ease;
            position: relative;
        }}
        .progress-bar-fill::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }}
        @keyframes shimmer {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(100%); }}
        }}
        .progress-text {{
            text-align: center;
            margin-top: 10px;
            color: #1f4e79;
            font-weight: 600;
        }}
        .progress-message {{
            text-align: center;
            margin-bottom: 10px;
            color: #6c757d;
            font-size: 14px;
        }}
    </style>
    <div class="progress-container">
        {message_html}
        <div class="progress-bar-bg">
            <div class="progress-bar-fill"></div>
        </div>
        <div class="progress-text">{progress_percent}%</div>
    </div>
    """, unsafe_allow_html=True)


def render_skeleton_match_card() -> None:
    """Render a skeleton loader for match prediction cards."""
    st.markdown("""
    <style>
        @keyframes skeleton-shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        .skeleton-match-card {
            background: #f6f7f8;
            background-image: linear-gradient(
                90deg,
                #f6f7f8 0px,
                #edeef1 40px,
                #f6f7f8 80px
            );
            background-size: 1000px 100%;
            animation: skeleton-shimmer 2s infinite;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .skeleton-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 12px 0;
        }
        .skeleton-team {
            width: 35%;
            height: 24px;
            background: #e0e0e0;
            border-radius: 4px;
        }
        .skeleton-vs {
            width: 10%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 4px;
        }
        .skeleton-metric {
            width: 60px;
            height: 40px;
            background: #e0e0e0;
            border-radius: 6px;
            margin: 0 5px;
        }
        .skeleton-metrics-row {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
    </style>
    <div class="skeleton-match-card">
        <div class="skeleton-row">
            <div class="skeleton-team"></div>
            <div class="skeleton-vs"></div>
            <div class="skeleton-team"></div>
        </div>
        <div class="skeleton-metrics-row">
            <div class="skeleton-metric"></div>
            <div class="skeleton-metric"></div>
            <div class="skeleton-metric"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_data_fetch_progress(stage: str, progress: float) -> None:
    """Show progress during async data fetching stages.
    
    Args:
        stage: Current stage name (e.g., "Fetching fixtures", "Loading teams")
        progress: Progress value between 0.0 and 1.0
    """
    stages = {
        "init": ("🚀 Initializing data fetch...", 0.1),
        "fixtures": ("⚽ Fetching fixtures from API...", 0.3),
        "teams": ("👥 Loading team data...", 0.6),
        "stats": ("📊 Gathering match statistics...", 0.85),
        "complete": ("✅ Data loaded successfully!", 1.0)
    }
    
    message, base_progress = stages.get(stage, (stage, progress))
    render_progress_bar(base_progress, message)


def render_error_state(
    message: str = "Something went wrong",
    details: Optional[str] = None,
    show_retry: bool = True
) -> bool:
    """Render an error state with optional retry button.
    
    Args:
        message: Main error message
        details: Optional error details
        show_retry: Whether to show a retry button
    
    Returns:
        True if retry button was clicked, False otherwise
    """
    details_html = f'<div class="error-details">{details}</div>' if details else ''
    
    st.markdown(f"""
    <style>
        .error-container {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
        }}
        .error-icon {{
            font-size: 48px;
            margin-bottom: 15px;
        }}
        .error-message {{
            color: #856404;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        .error-details {{
            color: #6c757d;
            font-size: 14px;
            margin-top: 10px;
        }}
    </style>
    <div class="error-container">
        <div class="error-icon">⚠️</div>
        <div class="error-message">{message}</div>
        {details_html}
    </div>
    """, unsafe_allow_html=True)
    
    if show_retry:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            return st.button("🔄 Retry", key="error_retry_button", type="primary")
    
    return False


def render_empty_state(
    icon: str = "📭",
    message: str = "No data available",
    suggestion: Optional[str] = None
) -> None:
    """Render an empty state placeholder.
    
    Args:
        icon: Emoji icon to display
        message: Main empty state message
        suggestion: Optional suggestion text
    """
    suggestion_html = f'<div class="empty-suggestion">{suggestion}</div>' if suggestion else ''
    
    st.markdown(f"""
    <style>
        .empty-container {{
            text-align: center;
            padding: 60px 20px;
            color: #6c757d;
        }}
        .empty-icon {{
            font-size: 64px;
            margin-bottom: 20px;
        }}
        .empty-message {{
            font-size: 20px;
            font-weight: 500;
            margin-bottom: 10px;
        }}
        .empty-suggestion {{
            font-size: 14px;
            margin-top: 10px;
        }}
    </style>
    <div class="empty-container">
        <div class="empty-icon">{icon}</div>
        <div class="empty-message">{message}</div>
        {suggestion_html}
    </div>
    """, unsafe_allow_html=True)
