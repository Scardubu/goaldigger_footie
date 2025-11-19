"""
🌍 Cross-League Analysis
Interactive cross-league analysis and "What-if" scenarios between Europe's top leagues
"""

import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def main():
    """Main Cross-League Analysis dashboard."""
    st.set_page_config(
        page_title="🌍 Cross-League Analysis - GoalDiggers",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced header with professional styling
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
    ">
        <h1 style="margin: 0; font-weight: 600; font-size: 2.5rem;">🌍 Cross-League Analysis</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.2rem;">
            Explore exciting "What-if" scenarios between Europe's top leagues
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load the interactive cross-league dashboard
    try:
        from dashboard.enhanced_production_homepage import ProductionDashboardHomepage
        dashboard = ProductionDashboardHomepage()
        dashboard.render_dashboard()
    except Exception as e:
        st.error(f"Error loading Cross-League Analysis: {e}")
        st.info("Please ensure all dependencies are installed and try again.")

if __name__ == "__main__":
    main()
