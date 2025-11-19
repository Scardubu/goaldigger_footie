"""
🎨 UI Showcase
Showcase of enhanced UI components and design system features
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main():
    """Main UI Showcase dashboard."""
    st.set_page_config(
        page_title="🎨 UI Showcase - GoalDiggers",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced header with professional styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="margin: 0; font-weight: 600; font-size: 2.5rem;">🎨 UI Showcase</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem;">
            Enhanced UI components and design system features
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load the UI showcase
    try:
        # Import the original UI showcase content
        exec(open('dashboard/pages/ui_showcase.py').read())
    except Exception as e:
        st.error(f"Error loading UI Showcase: {e}")
        st.info("Please ensure all dependencies are installed and try again.")

if __name__ == "__main__":
    main()
