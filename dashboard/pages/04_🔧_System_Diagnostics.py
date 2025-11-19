"""
🔧 System Diagnostics
Advanced diagnostic tools and recovery options for system issues
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main():
    """Main System Diagnostics dashboard."""
    st.set_page_config(
        page_title="🔧 System Diagnostics - GoalDiggers",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced header with professional styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(240, 147, 251, 0.3);
    ">
        <h1 style="margin: 0; font-weight: 600; font-size: 2.5rem;">🔧 System Diagnostics</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem;">
            Advanced diagnostic tools and recovery options
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load the error recovery dashboard
    try:
        # Import the original error recovery content
        exec(open('dashboard/pages/error_recovery_dashboard.py').read())
    except Exception as e:
        st.error(f"Error loading System Diagnostics: {e}")
        st.info("Please ensure all dependencies are installed and try again.")

if __name__ == "__main__":
    main()
