"""
UI Component Showcase for GoalDiggers.

This page demonstrates all the enhanced UI components available in the GoalDiggers platform,
serving as both a visual showcase and a reference for developers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import random

# Import our custom components
from dashboard.components.ui_elements import (
    card,
    header,
    badge,
    info_tooltip,
    progress_indicator,
    collapsible_section
)
from dashboard.components.theme_switcher import render_theme_switcher
from dashboard.components.ui_enhancements import load_custom_css, animate_on_hover
from dashboard.components.data_integrity_visualizer import get_data_integrity_visualizer
from dashboard.components.user_interaction_tracker import (
    get_interaction_tracker, 
    track_function_usage
)

# Set page configuration
st.set_page_config(
    page_title="GoalDiggers UI Showcase",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize theme if not in session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Get our trackers and visualizers
tracker = get_interaction_tracker()
data_visualizer = get_data_integrity_visualizer()

# Track page view
tracker.track_page_view(page_name="UI Showcase", path="/ui_showcase")

# Load custom CSS
load_custom_css()

# Render theme switcher in sidebar
with st.sidebar:
    st.markdown(header("⚙️ Settings", level=3))
    render_theme_switcher()
    
    st.markdown(header("📚 Navigation", level=3))
    if st.button("🏠 Dashboard"):
        tracker.track_ui_interaction("button", "click", "Dashboard", "navigation")
    if st.button("📊 Data Analysis"):
        tracker.track_ui_interaction("button", "click", "Data Analysis", "navigation")
    if st.button("🔮 Predictions"):
        tracker.track_ui_interaction("button", "click", "Predictions", "navigation")
    if st.button("⚙️ Settings"):
        tracker.track_ui_interaction("button", "click", "Settings", "navigation")
    
    # User stats
    st.markdown(header("📈 User Activity", level=3))
    st.markdown(f"Session duration: **{tracker.get_session_duration() / 60:.1f}** minutes")
    st.markdown(f"Interactions: **{len(tracker.get_interactions_dataframe())}**")

# Main content
st.markdown(
    header(
        "GoalDiggers UI Component Showcase", 
        icon="⚽", 
        subtitle="A demonstration of modern UI components for the betting insights platform",
        divider=True,
        animation="slide-in"
    ),
    unsafe_allow_html=True
)

st.markdown("""
This page showcases all the enhanced UI components available in the GoalDiggers platform. 
Use this as a reference when building new features or pages. All components respect the 
selected theme and provide consistent user experience.
""")

# Create tabs for different component categories
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Basic Components", 
    "Layout & Structure", 
    "Interactive Elements", 
    "Data Visualization",
    "Analytics Dashboard"
])

# Tab 1: Basic Components
with tab1:
    st.markdown(header("Basic UI Components", level=2))
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Headers
        st.markdown(header("Header Component", level=3, accent_bar=True))
        st.markdown("""Headers provide structure to your content and can include icons, 
                    subtitles, and visual elements like dividers or accent bars.""")
        
        # Examples of headers
        st.markdown(header("Default Header", level=4))
        st.markdown(header("Header with Icon", level=4, icon="🔍"))
        st.markdown(header("Header with Subtitle", level=4, subtitle="This is a subtitle example"))
        st.markdown(header("Accent Bar Header", level=4, accent_bar=True))
        
        # Badges
        st.markdown(header("Badge Component", level=3, accent_bar=True))
        st.markdown("""Badges highlight important information like status, counts, or categories.""")
        
        st.markdown("### Badge Types")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(badge("Default", type="default"), unsafe_allow_html=True)
        with col_b:
            st.markdown(badge("Success", type="success"), unsafe_allow_html=True)
        with col_c:
            st.markdown(badge("Warning", type="warning"), unsafe_allow_html=True)
        with col_d:
            st.markdown(badge("Danger", type="danger"), unsafe_allow_html=True)
            
        st.markdown("### Badge Sizes")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(badge("Small", size="small"), unsafe_allow_html=True)
        with col_b:
            st.markdown(badge("Medium", size="medium"), unsafe_allow_html=True)
        with col_c:
            st.markdown(badge("Large", size="large"), unsafe_allow_html=True)
            
        st.markdown("### Badges with Icons")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(badge("New", type="success", icon="✨"), unsafe_allow_html=True)
        with col_b:
            st.markdown(badge("Warning", type="warning", icon="⚠️"), unsafe_allow_html=True)
        with col_c:
            st.markdown(badge("Error", type="danger", icon="❌", pulse=True), unsafe_allow_html=True)
    
    with col2:
        # Info Tooltips
        st.markdown(header("Info Tooltip Component", level=3, accent_bar=True))
        st.markdown("""Tooltips provide additional context or explanations without cluttering the interface.""")
        
        # Examples of tooltips
        st.markdown("### Default Tooltip")
        st.markdown(
            f"Hover over this {info_tooltip('This is a tooltip with additional information.')} for more information.",
            unsafe_allow_html=True
        )
        
        st.markdown("### Tooltip Styles")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"Icon style {info_tooltip('Information in an icon-style tooltip.', style='icon')}",
                unsafe_allow_html=True
            )
        with col_b:
            st.markdown(
                f"Text style {info_tooltip('Information in a text-style tooltip.', style='text', icon='(?)')}",
                unsafe_allow_html=True
            )
            
        st.markdown("### Tooltip Placements")
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(
                f"{info_tooltip('Tooltip on top', placement='top')} Top",
                unsafe_allow_html=True
            )
        with col_b:
            st.markdown(
                f"{info_tooltip('Tooltip on right', placement='right')} Right",
                unsafe_allow_html=True
            )
        with col_c:
            st.markdown(
                f"{info_tooltip('Tooltip on bottom', placement='bottom')} Bottom",
                unsafe_allow_html=True
            )
        with col_d:
            st.markdown(
                f"{info_tooltip('Tooltip on left', placement='left')} Left",
                unsafe_allow_html=True
            )
        
        # Progress Indicators
        st.markdown(header("Progress Indicator Component", level=3, accent_bar=True))
        st.markdown("""Progress indicators visualize completion status, steps, or metrics.""")
        
        # Examples of progress indicators
        st.markdown("### Bar Style Progress (Default)")
        st.markdown(progress_indicator(65, 100, style="bar", label="Progress", animated=True), unsafe_allow_html=True)
        
        st.markdown("### Circle Style Progress")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(progress_indicator(25, 100, style="circle", size="small"), unsafe_allow_html=True)
        with col_b:
            st.markdown(progress_indicator(50, 100, style="circle"), unsafe_allow_html=True)
        with col_c:
            st.markdown(progress_indicator(75, 100, style="circle", size="large"), unsafe_allow_html=True)
            
        st.markdown("### Step Progress (Multi-step processes)")
        st.markdown(progress_indicator(2, 5, style="step", label="Registration Process"), unsafe_allow_html=True)

# Tab 2: Layout & Structure
with tab2:
    st.markdown(header("Layout & Structure Components", level=2))
    
    # Cards
    st.markdown(header("Card Component", level=3, accent_bar=True))
    st.markdown("""Cards group related content into visually distinct sections.""")
    
    # Examples of cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            card(
                """
                <h4>Default Card</h4>
                <p>This is a basic card with default styling.</p>
                """, 
                hover_effect=True
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            card(
                """
                <h4>Primary Card</h4>
                <p>This card has primary styling.</p>
                <button style="background-color: var(--color-primary); color: white; border: none; padding: 5px 10px; border-radius: 4px;">Action</button>
                """, 
                card_type="primary", 
                hover_effect=True
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            card(
                """
                <h4>Success Card</h4>
                <p>This card indicates a successful operation.</p>
                <div style="background-color: rgba(0,255,0,0.1); padding: 10px; border-radius: 4px;">
                    ✅ Operation completed successfully!
                </div>
                """, 
                card_type="success", 
                hover_effect=True
            ),
            unsafe_allow_html=True
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            card(
                """
                <h4>Warning Card</h4>
                <p>This card indicates a warning or caution.</p>
                <div style="background-color: rgba(255,255,0,0.1); padding: 10px; border-radius: 4px;">
                    ⚠️ Please review the information before proceeding.
                </div>
                """, 
                card_type="warning", 
                hover_effect=True
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            card(
                """
                <h4>Danger Card</h4>
                <p>This card indicates an error or critical issue.</p>
                <div style="background-color: rgba(255,0,0,0.1); padding: 10px; border-radius: 4px;">
                    ❌ Error encountered. Please try again.
                </div>
                """, 
                card_type="danger", 
                hover_effect=True
            ),
            unsafe_allow_html=True
        )
    
    # Collapsible Sections
    st.markdown(header("Collapsible Section Component", level=3, accent_bar=True))
    st.markdown("""Collapsible sections help manage content density and allow users to focus on specific information.""")
    
    # Examples of collapsible sections
    st.markdown("### Default Style (Streamlit Native)")
    collapsible_section(
        "Basic Collapsible Section", 
        lambda: st.markdown("This is the content of a basic collapsible section using Streamlit's native expander."),
        key="basic_section"
    )
    
    st.markdown("### Card Style")
    collapsible_section(
        "Card Style Collapsible Section", 
        lambda: st.markdown("This collapsible section uses the card style for a more visually distinct appearance."),
        style="card",
        icon="📋",
        key="card_section"
    )
    
    st.markdown("### Minimal Style")
    collapsible_section(
        "Minimal Style Collapsible Section", 
        lambda: st.markdown("This collapsible section uses the minimal style for a cleaner look."),
        style="minimal",
        icon="🔍",
        key="minimal_section"
    )
    
    st.markdown("### Outline Style")
    collapsible_section(
        "Outline Style Collapsible Section", 
        lambda: st.markdown("This collapsible section uses the outline style for a subtle border."),
        style="outline",
        icon="📝",
        key="outline_section"
    )

# Tab 3: Interactive Elements
with tab3:
    st.markdown(header("Interactive Elements", level=2))
    
    # Interactive Card Example
    def interact_with_card():
        if "card_clicks" not in st.session_state:
            st.session_state.card_clicks = 0
        st.session_state.card_clicks += 1
        tracker.track_ui_interaction("card", "click", st.session_state.card_clicks, "interactive demo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(header("Interactive Cards", level=3, accent_bar=True))
        st.markdown("""Cards can include interactive elements that respond to user actions.""")
        
        # Example of an interactive card
        card_content = f"""
        <h4>Interactive Card</h4>
        <p>Click the button below to interact:</p>
        <div id="click-counter">Clicks: {st.session_state.get('card_clicks', 0)}</div>
        """
        
        st.markdown(
            card(card_content, card_type="primary", hover_effect=True, animate=True),
            unsafe_allow_html=True
        )
        
        if st.button("Click Me!", key="card_button", on_click=interact_with_card):
            pass
    
    with col2:
        st.markdown(header("Animated Elements", level=3, accent_bar=True))
        st.markdown("""UI elements can include animations to provide visual feedback and improve user engagement.""")
        
        # Example of animated elements
        st.markdown(animate_on_hover(
            """
            <div style="padding: 20px; background-color: var(--color-card); border-radius: 8px; text-align: center;">
                <h4>Hover over me!</h4>
                <p>This element animates on hover.</p>
            </div>
            """,
            animation="pulse"
        ), unsafe_allow_html=True)
        
        st.markdown(animate_on_hover(
            """
            <div style="padding: 20px; background-color: var(--color-card); border-radius: 8px; text-align: center;">
                <h4>Scale Effect</h4>
                <p>This element scales on hover.</p>
            </div>
            """,
            animation="scale"
        ), unsafe_allow_html=True)
    
    # Form with validation
    st.markdown(header("Form with Validation", level=3, accent_bar=True))
    st.markdown("""Forms can include validation to ensure data quality and provide user feedback.""")
    
    with st.form("demo_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name")
            email = st.text_input("Email")
        
        with col2:
            age = st.number_input("Age", min_value=0, max_value=120)
            favorite_team = st.selectbox("Favorite Team", ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", "Other"])
        
        prediction_frequency = st.slider("Prediction Frequency (per week)", 0, 10, 3)
        
        submit = st.form_submit_button("Submit")
        
        if submit:
            # Validate form data
            errors = []
            
            if not name:
                errors.append("Name is required")
            
            if not email:
                errors.append("Email is required")
            elif "@" not in email or "." not in email:
                errors.append("Email is invalid")
            
            if age <= 0:
                errors.append("Age must be greater than 0")
            
            # Display validation results
            if errors:
                for error in errors:
                    st.markdown(badge(error, type="danger", icon="❌"), unsafe_allow_html=True)
                
                # Track error
                tracker.track_error("validation_error", str(errors), "demo_form")
            else:
                st.success("Form submitted successfully!")
                st.markdown(badge("Success", type="success", icon="✅"), unsafe_allow_html=True)
                
                # Track success
                tracker.track_feature_usage(
                    "form_submission", 
                    "submit", 
                    "success", 
                    {
                        "favorite_team": favorite_team,
                        "prediction_frequency": prediction_frequency
                    }
                )

# Tab 4: Data Visualization
with tab4:
    st.markdown(header("Data Visualization Components", level=2))
    
    # Generate sample data for demonstration
    @track_function_usage("generate_live_data")
    def generate_live_data(rows=100):
        np.random.seed(42)
        
        # Create date range
        start_date = datetime.now() - timedelta(days=365)
        dates = [start_date + timedelta(days=i) for i in range(rows)]
        
        # Create teams
        teams = ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United", 
                "Tottenham", "Leicester", "Everton", "West Ham", "Newcastle"]
        
        # Create sample data
        data = {
            "match_date": dates,
            "team_id": np.random.choice(range(1, 11), size=rows),
            "team_name": np.random.choice(teams, size=rows),
            "match_result": np.random.choice(["W", "D", "L"], size=rows, p=[0.45, 0.25, 0.3]),
            "goals_scored": np.random.choice(range(0, 6), size=rows, p=[0.15, 0.35, 0.25, 0.15, 0.07, 0.03]),
            "goals_conceded": np.random.choice(range(0, 6), size=rows, p=[0.2, 0.4, 0.2, 0.1, 0.07, 0.03]),
            "shots": np.random.randint(5, 25, size=rows),
            "shots_on_target": np.random.randint(1, 15, size=rows),
            "possession": np.random.randint(30, 71, size=rows),
            "passes": np.random.randint(300, 800, size=rows),
            "pass_accuracy": np.random.randint(70, 96, size=rows),
            "fouls": np.random.randint(5, 20, size=rows),
            "yellow_cards": np.random.choice(range(0, 6), size=rows, p=[0.2, 0.4, 0.2, 0.1, 0.07, 0.03]),
            "red_cards": np.random.choice(range(0, 2), size=rows, p=[0.95, 0.05]),
            "expected_goals": np.random.uniform(0.5, 3.0, size=rows),
        }
        
        # Add some missing data to demonstrate visualization capabilities
        for field in ["shots", "shots_on_target", "possession", "passes", "expected_goals"]:
            missing_indices = np.random.choice(rows, size=int(rows * 0.2), replace=False)
            data[field] = pd.Series(data[field])
            data[field].iloc[missing_indices] = np.nan
        
        # Add weather data with more missing values
        data["temperature"] = np.random.uniform(5.0, 30.0, size=rows)
        data["weather_condition"] = np.random.choice(["Sunny", "Cloudy", "Rainy", "Snowy"], size=rows)
        
        # Make 40% of weather data missing
        missing_indices = np.random.choice(rows, size=int(rows * 0.4), replace=False)
        data["temperature"] = pd.Series(data["temperature"])
        data["temperature"].iloc[missing_indices] = np.nan
        data["weather_condition"] = pd.Series(data["weather_condition"])
        data["weather_condition"].iloc[missing_indices] = np.nan
        
        return pd.DataFrame(data)
    
    # Generate sample data
    if "live_data" not in st.session_state:
        st.session_state.sample_data = generate_live_data(200)
    
    sample_data = st.session_state.live_data
    
    # Data Integrity Visualizer demonstration
    st.markdown(header("Data Integrity Visualizer", level=3, accent_bar=True))
    st.markdown("""The Data Integrity Visualizer helps identify missing or incomplete data, 
                which is essential for ensuring the quality of your machine learning models and analytics.""")
    
    # Define fields of interest for the visualizer
    fields_of_interest = {
        "Match Info": ["match_date", "team_id", "team_name", "match_result"],
        "Performance Metrics": ["goals_scored", "goals_conceded", "shots", "shots_on_target", "possession", "passes", "pass_accuracy"],
        "Event Data": ["fouls", "yellow_cards", "red_cards"],
        "Advanced Metrics": ["expected_goals"],
        "Environmental Data": ["temperature", "weather_condition"]
    }
    
    # Render the data integrity dashboard
    data_visualizer.render_missing_data_dashboard(
        live_data,
        fields_of_interest,
        date_field="match_date"
    )

# Tab 5: Analytics Dashboard
with tab5:
    st.markdown(header("User Interaction Analytics", level=2))
    
    st.markdown("""The User Interaction Tracker monitors how users engage with the dashboard,
                providing valuable insights for UI/UX improvements and understanding user behavior.""")
    
    # Generate some mock interactions if there are very few
    if len(tracker.get_interactions_dataframe()) < 10:
        # Add some mock interactions for demonstration
        components = ["page", "button", "card", "form", "chart"]
        actions = ["view", "click", "submit", "filter", "hover"]
        
        for _ in range(20):
            component = random.choice(components)
            action = random.choice(actions)
            
            tracker.track_event(
                event_type="demo",
                component=component,
                action=action,
                details={
                    "demo_data": True,
                    "timestamp": time.time() - random.randint(0, 3600)
                }
            )
    
    # Render the analytics dashboard
    tracker.render_analytics_dashboard()

# Footer
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### GoalDiggers UI Component Showcase")
    st.markdown("This showcase demonstrates the modern UI components available for the GoalDiggers platform.")

with col2:
    st.markdown("### Next Steps")
    st.markdown("""
    - Use these components in your production pages
    - Contribute new component ideas
    - Report any issues or suggestions
    """)

# Track when showcase is fully loaded
tracker.track_event("lifecycle", "page", "fully_loaded", {"load_time": time.time() - st.session_state.get("page_start_time", time.time())})
