"""
🎯 AI Betting Intelligence
Main dashboard for AI-powered football betting insights and predictions
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main():
    """Main AI Betting Intelligence dashboard."""
    st.set_page_config(
        page_title="🎯 AI Betting Intelligence - GoalDiggers",
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
        <h1 style="margin: 0; font-weight: 600; font-size: 2.5rem;">🎯 AI Betting Intelligence</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem;">
            Advanced AI-powered football betting insights and predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load the premium dashboard
    try:
        from dashboard.enhanced_production_homepage import ProductionDashboardHomepage
        dashboard = ProductionDashboardProductionDashboardHomepage()
        dashboard.run()
    except Exception as e:
        st.error(f"Error loading AI Betting Intelligence: {e}")
        st.info("Please ensure all dependencies are installed and try again.")

if __name__ == "__main__":
    main()
