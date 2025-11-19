#!/usr/bin/env python3
"""
Streamlit Compatibility Layer for GoalDiggers Platform
Handles deprecated function warnings and provides forward compatibility
"""
import streamlit as st
import warnings
from typing import Any, Dict, Optional


class StreamlitCompatibility:
    """
    Compatibility layer for Streamlit deprecated functions.
    Provides seamless migration from deprecated to current APIs.
    """
    
    def __init__(self):
        """Initialize compatibility layer."""
        self._setup_warning_filters()
    
    def _setup_warning_filters(self):
        """Setup warning filters to suppress known deprecation warnings."""
        # Filter out specific Streamlit deprecation warnings that we can't control
        warnings.filterwarnings(
            "ignore",
            message=".*st.user.*",
            category=FutureWarning
        )
        
        warnings.filterwarnings(
            "ignore", 
            message=".*experimental_user.*",
            category=DeprecationWarning
        )
    
    @staticmethod
    def get_user_info() -> Dict[str, Any]:
        """
        Get user information using the current Streamlit API.
        Replaces deprecated st.user functionality.
        """
        try:
            # Try to use the new st.user if available
            if hasattr(st, 'user') and st.user is not None:
                return dict(st.user)
            
            # Fallback to session-based user info
            if 'user_info' not in st.session_state:
                st.session_state.user_info = {
                    'email': 'anonymous@goaldiggers.com',
                    'name': 'GoalDiggers User',
                    'id': f"user_{hash(str(st.session_state)) % 10000:04d}"
                }
            
            return st.session_state.user_info
            
        except Exception:
            # Ultimate fallback
            return {
                'email': 'anonymous@goaldiggers.com',
                'name': 'GoalDiggers User', 
                'id': 'anonymous_user'
            }
    
    @staticmethod
    def set_user_preference(key: str, value: Any):
        """Set user preference in session state."""
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        st.session_state.user_preferences[key] = value
    
    @staticmethod
    def get_user_preference(key: str, default: Any = None) -> Any:
        """Get user preference from session state."""
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        return st.session_state.user_preferences.get(key, default)
    
    @staticmethod
    def initialize_user_session():
        """Initialize user session with modern Streamlit practices."""
        # Initialize user info if not present
        if 'user_info' not in st.session_state:
            st.session_state.user_info = StreamlitCompatibility.get_user_info()
        
        # Initialize user preferences if not present
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
            
        return True


# Global compatibility instance
compatibility = StreamlitCompatibility()


def get_user_info() -> Dict[str, Any]:
    """Convenience function to get user info."""
    return compatibility.get_user_info()


def initialize_session():
    """Convenience function to initialize user session."""
    return compatibility.initialize_user_session()
