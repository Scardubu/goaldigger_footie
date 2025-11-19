#!/usr/bin/env python3
"""
Streamlit compatibility imports for deprecated functions.
This module provides seamless migration from deprecated Streamlit APIs.
"""

import streamlit as st
import warnings

# Suppress deprecation warnings for functions we're handling
warnings.filterwarnings("ignore", message=".*st.user.*")
warnings.filterwarnings("ignore", message=".*experimental_user.*")

# Compatibility mappings for deprecated functions
def experimental_user_compatibility():
    """Compatibility function for st.user."""
    try:
        if hasattr(st, 'user') and st.user is not None:
            return st.user
        return {"email": "anonymous@goaldiggers.com", "name": "GoalDiggers User"}
    except:
        return {"email": "anonymous@goaldiggers.com", "name": "GoalDiggers User"}

# Install compatibility layer
if not hasattr(st, '_compatibility_installed'):
    st._compatibility_installed = True
    
    # Override deprecated functions if they exist
    if hasattr(st, 'experimental_user'):
        st._original_experimental_user = st.user
        st.user = experimental_user_compatibility()
