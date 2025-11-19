"""
Enhanced loading animations for the GoalDiggers platform.

This module provides high-quality, branded loading animations that 
can be used across the platform for consistent user experience.
"""

import streamlit as st


def render_branded_loader(message: str = "Loading...", size: str = "medium"):
    """
    Render a branded GoalDiggers loading animation.
    
    Args:
        message: Loading message to display
        size: Size of the loader (small, medium, large)
    """
    # Size configurations
    sizes = {
        "small": {"ball_size": "20px", "font_size": "0.9rem", "padding": "0.5rem"},
        "medium": {"ball_size": "30px", "font_size": "1.1rem", "padding": "1rem"},
        "large": {"ball_size": "40px", "font_size": "1.3rem", "padding": "1.5rem"}
    }
    
    size_config = sizes.get(size, sizes["medium"])
    
    loader_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: {size_config['padding']};">
        <div class="goaldiggers-loader">
            <div class="football"></div>
            <div class="shadow"></div>
        </div>
        <div style="margin-top: 1.5rem; font-size: {size_config['font_size']}; color: #1e3c72;">
            {message}
        </div>
        
        <style>
        .goaldiggers-loader {{
            position: relative;
            width: {size_config['ball_size']};
            height: {size_config['ball_size']};
            perspective: 1000px;
        }}
        
        .football {{
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 65% 15%, white 1px, #1e3c72 3%, #2a5298 60%, #1e3c72 100%);
            border-radius: 50%;
            position: relative;
            animation: bounce 1.5s ease-in-out infinite;
            background-size: 100% 100%;
            background-repeat: no-repeat;
            box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.4);
        }}
        
        .football::before,
        .football::after {{
            content: "";
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
        }}
        
        .football::before {{
            background: 
                radial-gradient(circle at 50% 50%, transparent 66%, white 66.5%, white 71%, transparent 71.5%) no-repeat,
                radial-gradient(circle at 50% 50%, transparent 60%, white 60.5%, white 67%, transparent 67.5%) no-repeat,
                radial-gradient(circle at 50% 50%, transparent 54%, white 54.5%, white 62%, transparent 62.5%) no-repeat;
            background-size: 100% 100%;
            animation: rotate 3s linear infinite;
        }}
        
        .shadow {{
            position: absolute;
            bottom: -20%;
            left: 0;
            right: 0;
            height: 5px;
            border-radius: 50%;
            background: rgba(0, 0, 0, 0.15);
            transform: scaleX(0.7);
            animation: shadow 1.5s ease-in-out infinite;
            filter: blur(2px);
        }}
        
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-70%); }}
        }}
        
        @keyframes rotate {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        @keyframes shadow {{
            0%, 100% {{ transform: scaleX(0.7); opacity: 0.3; }}
            50% {{ transform: scaleX(0.5); opacity: 0.15; }}
        }}
        </style>
    </div>
    """
    
    st.markdown(loader_html, unsafe_allow_html=True)

def render_pulse_loader(message: str = "Processing...", color: str = "#1e3c72"):
    """
    Render a pulsing loader animation for lightweight loading states.
    
    Args:
        message: Message to display
        color: Primary color for the animation
    """
    pulse_html = f"""
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <div class="pulse-loader">
            <div class="pulse-dot"></div>
            <div class="pulse-dot"></div>
            <div class="pulse-dot"></div>
        </div>
        <span style="margin-left: 0.75rem;">{message}</span>
        
        <style>
        .pulse-loader {{
            display: flex;
            align-items: center;
        }}
        
        .pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: {color};
            margin: 0 3px;
            animation: pulse 1.5s ease-in-out infinite;
        }}
        
        .pulse-dot:nth-child(2) {{
            animation-delay: 0.2s;
        }}
        
        .pulse-dot:nth-child(3) {{
            animation-delay: 0.4s;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(0.5); opacity: 0.3; }}
        }}
        </style>
    </div>
    """
    
    st.markdown(pulse_html, unsafe_allow_html=True)

def render_progress_loader(progress: float, message: str = "Loading...", show_percentage: bool = True):
    """
    Render a branded progress loader.
    
    Args:
        progress: Progress value between 0 and 1
        message: Message to display
        show_percentage: Whether to show percentage text
    """
    # Ensure progress is between 0 and 1
    progress = max(0, min(1, progress))
    percentage = int(progress * 100)
    
    progress_html = f"""
    <div style="margin: 1rem 0;">
        <div style="margin-bottom: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
            <span>{message}</span>
            {f'<span>{percentage}%</span>' if show_percentage else ''}
        </div>
        <div style="
            height: 10px;
            background-color: #e6e6e6;
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                width: {percentage}%;
                background: linear-gradient(90deg, #1e3c72, #2a5298);
                border-radius: 5px;
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)

def render_data_loading_placeholder():
    """Render a placeholder while data is loading."""
    placeholder_html = """
    <div class="loading-placeholder">
        <div class="placeholder-item title-placeholder"></div>
        <div class="placeholder-item subtitle-placeholder"></div>
        
        <div class="placeholder-grid">
            <div class="placeholder-item card-placeholder"></div>
            <div class="placeholder-item card-placeholder"></div>
            <div class="placeholder-item card-placeholder"></div>
        </div>
        
        <div class="placeholder-item bar-placeholder"></div>
        <div class="placeholder-item bar-placeholder short"></div>
        
        <style>
        .loading-placeholder {
            padding: 1rem 0;
        }
        
        .placeholder-item {
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 4px;
            margin-bottom: 0.75rem;
        }
        
        .title-placeholder {
            height: 32px;
            width: 70%;
        }
        
        .subtitle-placeholder {
            height: 20px;
            width: 50%;
        }
        
        .placeholder-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .card-placeholder {
            height: 120px;
        }
        
        .bar-placeholder {
            height: 16px;
            width: 100%;
        }
        
        .short {
            width: 60%;
        }
        
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        @media (max-width: 768px) {
            .placeholder-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
    </div>
    """
    
    st.markdown(placeholder_html, unsafe_allow_html=True)

def render_spinner_with_message(message: str):
    """Wrapper for st.spinner with better styling."""
    with st.spinner(message):
        st.markdown("""
        <style>
        /* Enhanced spinner styling */
        div[data-testid="stSpinner"] > div {
            border-color: #1e3c72 #e6e6e6 #e6e6e6 !important;
            width: 24px !important;
            height: 24px !important;
        }
        
        div[data-testid="stSpinner"] > div ~ div {
            font-size: 1rem !important;
            color: #1e3c72 !important;
            margin-left: 0.75rem !important;
        }
        </style>
        """, unsafe_allow_html=True)
        return st.empty()
