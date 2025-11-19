"""
⚡ Performance Monitor
Real-time system performance monitoring and optimization tools
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main():
    """Main Performance Monitor dashboard."""
    st.set_page_config(
        page_title="⚡ Performance Monitor - GoalDiggers",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced header with professional styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
    ">
        <h1 style="margin: 0; font-weight: 600; font-size: 2.5rem;">⚡ Performance Monitor</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem;">
            Real-time system performance monitoring and optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load the performance monitor
    try:
        # Import the original performance monitor content
        exec(open('dashboard/pages/performance_monitor.py').read())
    except Exception as e:
        st.error(f"Error loading Performance Monitor: {e}")
        st.info("Please ensure all dependencies are installed and try again.")

if __name__ == "__main__":
    main()
