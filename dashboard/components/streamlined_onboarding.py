#!/usr/bin/env python3
"""
Streamlined Onboarding System for GoalDiggers Platform
Quick, engaging, and informative 3-step onboarding process.
"""

import logging
import random
from enum import Enum
from typing import Any, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

class OnboardingStep(Enum):
    """Streamlined onboarding steps."""
    WELCOME = "welcome"
    QUICK_SETUP = "quick_setup" 
    READY = "ready"

class StreamlinedOnboarding:
    """Quick 3-step onboarding that's fun and informative."""
    
    def __init__(self):
        """Initialize the streamlined onboarding system."""
        self.session_key = "streamlined_onboarding"
        
    def is_onboarding_complete(self) -> bool:
        """Check if onboarding is complete."""
        # Check direct key first (used by production integration)
        direct_complete = st.session_state.get("streamlined_onboarding_complete", None)
        if direct_complete is not None:
            return direct_complete
        
        # Fall back to the dynamic key format
        return st.session_state.get(f"{self.session_key}_complete", False)
    
    def render_onboarding_flow(self) -> bool:
        """Render the complete onboarding flow. Returns True if complete."""
        if self.is_onboarding_complete():
            return True
        
        # Initialize step if not set
        if f"{self.session_key}_step" not in st.session_state:
            st.session_state[f"{self.session_key}_step"] = OnboardingStep.WELCOME.value
            # Add direct access key for compatibility with production_ready_integration.py
            st.session_state["streamlined_onboarding_step"] = OnboardingStep.WELCOME.value
        
        current_step = st.session_state[f"{self.session_key}_step"]
        # Keep direct access key in sync
        st.session_state["streamlined_onboarding_step"] = current_step
        
        # Progress bar
        steps = ["Welcome", "Quick Setup", "Ready!"]
        step_index = list(OnboardingStep).index(OnboardingStep(current_step))
        progress = (step_index + 1) / len(steps)
        
        st.progress(progress, f"Step {step_index + 1} of {len(steps)}: {steps[step_index]}")
        
        # Render current step
        if current_step == OnboardingStep.WELCOME.value:
            return self._render_welcome_step()
        elif current_step == OnboardingStep.QUICK_SETUP.value:
            return self._render_quick_setup_step()
        elif current_step == OnboardingStep.READY.value:
            return self._render_ready_step()
        
        return False
    
    def _render_welcome_step(self) -> bool:
        """Render welcome step with platform introduction."""
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; color: white; margin: 2rem 0;'>
            <h1 style='margin: 0; font-size: 2.5rem;'>⚽ Welcome to GoalDiggers!</h1>
            <p style='font-size: 1.2rem; margin: 1rem 0;'>AI-Powered Football Betting Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎯 Smart Predictions
            Advanced AI analyzes team performance, 
            form, and historical data to give you 
            accurate match predictions.
            """)
        
        with col2:
            st.markdown("""
            ### 💰 Value Betting
            Our algorithms identify betting 
            opportunities with positive expected 
            value for smarter wagering.
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Real-time Insights
            Live match data, team statistics, 
            and betting market analysis at 
            your fingertips.
            """)
        
        st.markdown("---")
        
        # Fun facts to keep users engaged
        fun_facts = [
            "🔢 Our AI analyzes over 50 match statistics",
            "🏆 We cover 6 major European leagues",
            "⚡ Predictions update in real-time",
            "🎲 Expected value calculations help maximize profits",
            "📈 Track your prediction accuracy over time"
        ]
        
        st.markdown("### ⚡ Quick Facts:")
        for fact in random.sample(fun_facts, 3):
            st.markdown(f"• {fact}")
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        with col2:
            if st.button("Let's Get Started! 🚀", type="primary", use_container_width=True):
                st.session_state[f"{self.session_key}_step"] = OnboardingStep.QUICK_SETUP.value
                # Update direct access key for compatibility with production_ready_integration.py
                st.session_state["streamlined_onboarding_step"] = OnboardingStep.QUICK_SETUP.value
                st.rerun()
        
        return False
    
    def _render_quick_setup_step(self) -> bool:
        """Render quick setup step with smart defaults."""
        st.markdown("### 🛠️ Quick Setup")
        st.markdown("*We'll set up some smart defaults to get you started quickly!*")
        
        # Quick preferences with smart defaults
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚽ Favorite League")
            favorite_league = st.selectbox(
                "",
                ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "All Leagues"],
                index=0,
                help="Choose your primary league of interest"
            )
            
            st.markdown("#### 🎨 Display Style")
            display_style = st.radio(
                "",
                ["💎 Premium (Detailed)", "⚡ Quick (Essential)", "📊 Data-Rich"],
                index=0,
                help="How much information would you like to see?"
            )
        
        with col2:
            st.markdown("#### 🎯 Betting Focus")
            betting_focus = st.selectbox(
                "",
                ["🏆 Match Winners", "⚽ Goals (Over/Under)", "💰 Value Bets", "📊 All Markets"],
                index=2,
                help="What type of bets interest you most?"
            )
            
            st.markdown("#### ⚠️ Risk Level")
            risk_level = st.select_slider(
                "",
                options=["🟢 Conservative", "🟡 Balanced", "🔴 Aggressive"],
                value="🟡 Balanced",
                help="Your preferred risk level for betting insights"
            )
        
        # Preview what they'll get
        st.markdown("---")
        st.markdown("### 📋 Your Setup Preview:")
        
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            st.info(f"""
            **🎯 Primary League:** {favorite_league}  
            **🎨 Display Style:** {display_style.split(' ')[1]}
            """)
        
        with preview_col2:
            st.info(f"""
            **💰 Betting Focus:** {betting_focus.split(' ', 1)[1]}  
            **⚠️ Risk Level:** {risk_level.split(' ', 1)[1]}
            """)
        
        st.markdown("---")
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state[f"{self.session_key}_step"] = OnboardingStep.WELCOME.value
                # Update direct access key for compatibility
                st.session_state["streamlined_onboarding_step"] = OnboardingStep.WELCOME.value
                st.rerun()
        
        with col3:
            if st.button("Complete Setup! 🎉", type="primary", use_container_width=True):
                # Save preferences
                preferences = {
                    'favorite_league': favorite_league,
                    'display_style': display_style.split(' ')[0],
                    'betting_focus': betting_focus,
                    'risk_level': risk_level.split(' ')[0]
                }
                
                for key, value in preferences.items():
                    st.session_state[f"{self.session_key}_{key}"] = value
                
                st.session_state[f"{self.session_key}_step"] = OnboardingStep.READY.value
                # Update direct access key for compatibility
                st.session_state["streamlined_onboarding_step"] = OnboardingStep.READY.value
                st.rerun()
        
        return False
    
    def _render_ready_step(self) -> bool:
        """Render completion step with celebration."""
        # Celebration
        st.balloons()
        
        st.markdown("""
        <div style='text-align: center; padding: 3rem; background: linear-gradient(135deg, #10b981 0%, #047857 100%); 
                    border-radius: 15px; color: white; margin: 2rem 0;'>
            <h1 style='margin: 0; font-size: 3rem;'>🎉</h1>
            <h2 style='margin: 0.5rem 0; font-size: 2rem;'>You're All Set!</h2>
            <p style='font-size: 1.1rem; margin: 1rem 0;'>Welcome to the future of football betting intelligence!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show what they can do now
        st.markdown("### 🚀 What you can do now:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 🔮 Get Predictions
            View AI-powered match predictions 
            with confidence scores and 
            expected value calculations.
            """)
        
        with col2:
            st.markdown("""
            #### 📊 Analyze Matches
            Dive deep into team statistics, 
            form analysis, and head-to-head 
            performance data.
            """)
        
        with col3:
            st.markdown("""
            #### 💎 Find Value Bets
            Discover betting opportunities 
            with positive expected value 
            based on AI analysis.
            """)
        
        # Quick tips
        st.markdown("### 💡 Quick Tips:")
        tips = [
            "🎯 Look for matches with 70%+ confidence for safer bets",
            "💰 Value bets are marked with a special indicator",
            "📈 Your prediction accuracy will be tracked over time",
            "⚙️ You can change preferences anytime in settings"
        ]
        
        for tip in tips:
            st.markdown(f"• {tip}")
        
        st.markdown("---")
        
        # Complete onboarding
        col1, col2 = st.columns([1, 1])
        with col2:
            if st.button("Start Exploring! 🎯", type="primary", use_container_width=True):
                st.session_state[f"{self.session_key}_complete"] = True
                # Update direct access key for compatibility
                st.session_state["streamlined_onboarding_complete"] = True
                st.rerun()
        
        return False
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from onboarding."""
        if not self.is_onboarding_complete():
            return {}
        
        return {
            'favorite_league': st.session_state.get(f"{self.session_key}_favorite_league", "All Leagues"),
            'display_style': st.session_state.get(f"{self.session_key}_display_style", "💎"),
            'betting_focus': st.session_state.get(f"{self.session_key}_betting_focus", "💰 Value Bets"),
            'risk_level': st.session_state.get(f"{self.session_key}_risk_level", "🟡")
        }
    
    def reset_onboarding(self) -> None:
        """Reset onboarding state (for testing)."""
        keys_to_remove = [key for key in st.session_state.keys() if key.startswith(self.session_key)]
        for key in keys_to_remove:
            del st.session_state[key]

# Global instance
streamlined_onboarding = StreamlinedOnboarding()

def show_onboarding() -> bool:
    """Show onboarding flow. Returns True if complete."""
    return streamlined_onboarding.render_onboarding_flow()

def is_onboarding_complete() -> bool:
    """Check if onboarding is complete."""
    # Ensure both keys are properly initialized if either is missing
    session_key = "streamlined_onboarding"
    
    # Initialize direct access key if missing
    if "streamlined_onboarding_complete" not in st.session_state and f"{session_key}_complete" in st.session_state:
        st.session_state["streamlined_onboarding_complete"] = st.session_state[f"{session_key}_complete"]
    
    # Initialize dynamic key if missing
    if f"{session_key}_complete" not in st.session_state and "streamlined_onboarding_complete" in st.session_state:
        st.session_state[f"{session_key}_complete"] = st.session_state["streamlined_onboarding_complete"]
    
    # Default both keys if both are missing
    if "streamlined_onboarding_complete" not in st.session_state and f"{session_key}_complete" not in st.session_state:
        st.session_state["streamlined_onboarding_complete"] = False
        st.session_state[f"{session_key}_complete"] = False
    
    # Check direct access pattern first (used by production code)
    if "streamlined_onboarding_complete" in st.session_state:
        return st.session_state["streamlined_onboarding_complete"]
        
    # Fall back to the class method which itself checks both formats
    return streamlined_onboarding.is_onboarding_complete()

def get_onboarding_preferences() -> Dict[str, Any]:
    """Get user preferences from onboarding."""
    return streamlined_onboarding.get_user_preferences()
