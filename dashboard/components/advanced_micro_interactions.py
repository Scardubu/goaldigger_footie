#!/usr/bin/env python3
"""
Advanced Micro-Interactions Component
Provides smooth animations and micro-interactions for enhanced UX
"""

import time
from typing import Any, Dict, List, Optional

import streamlit as st


class AdvancedMicroInteractions:
    """Advanced micro-interactions and animations for dashboard components."""
    
    def __init__(self):
        """Initialize micro-interactions system."""
        self.animation_cache = {}
        self.interaction_state = {}
        
    def inject_advanced_css(self):
        """Inject advanced CSS for micro-interactions and animations."""
        css = """
        <style>
        /* Advanced Micro-Interactions CSS */
        
        /* Smooth transitions for all interactive elements */
        .stButton > button, .stSelectbox > div, .stMetric, .stAlert {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        /* Button hover effects with elevation */
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15) !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
        }
        
        /* Pulse animation for loading states */
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.05); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        .pulse-animation {
            animation: pulse 2s infinite;
        }
        
        /* Slide-in animation for content */
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .slide-in-up {
            animation: slideInUp 0.6s ease-out;
        }
        
        /* Fade-in animation */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .fade-in {
            animation: fadeIn 0.8s ease-in;
        }
        
        /* Bounce animation for success states */
        @keyframes bounce {
            0%, 20%, 53%, 80%, 100% {
                transform: translate3d(0,0,0);
            }
            40%, 43% {
                transform: translate3d(0, -30px, 0);
            }
            70% {
                transform: translate3d(0, -15px, 0);
            }
            90% {
                transform: translate3d(0, -4px, 0);
            }
        }
        
        .bounce-animation {
            animation: bounce 1s ease-in-out;
        }
        
        /* Shake animation for errors */
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
            20%, 40%, 60%, 80% { transform: translateX(10px); }
        }
        
        .shake-animation {
            animation: shake 0.6s ease-in-out;
        }
        
        /* Glow effect for important elements */
        .glow-effect {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.6) !important;
            border: 2px solid rgba(102, 126, 234, 0.8) !important;
        }
        
        /* Smooth card hover effects */
        .metric-card {
            transition: all 0.3s ease !important;
            border-radius: 12px !important;
            padding: 20px !important;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%) !important;
        }
        
        .metric-card:hover {
            transform: translateY(-5px) !important;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1) !important;
        }
        
        /* Progress bar animations */
        @keyframes progressFill {
            from { width: 0%; }
            to { width: var(--progress-width); }
        }
        
        .animated-progress {
            animation: progressFill 1.5s ease-out;
        }
        
        /* Typing animation for text */
        @keyframes typing {
            from { width: 0; }
            to { width: 100%; }
        }
        
        .typing-animation {
            overflow: hidden;
            white-space: nowrap;
            animation: typing 2s steps(40, end);
        }
        
        /* Floating animation for decorative elements */
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
        }
        
        .float-animation {
            animation: float 3s ease-in-out infinite;
        }
        
        /* Ripple effect for buttons */
        .ripple-effect {
            position: relative;
            overflow: hidden;
        }
        
        .ripple-effect::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255,255,255,0.5);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .ripple-effect:active::before {
            width: 300px;
            height: 300px;
        }
        
        /* Gradient text animation */
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .gradient-text {
            background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #f5576c);
            background-size: 400% 400%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 3s ease infinite;
        }
        
        /* Smooth scrolling */
        html {
            scroll-behavior: smooth;
        }
        
        /* Enhanced focus states */
        .stButton > button:focus,
        .stSelectbox > div:focus-within,
        .stTextInput > div > div > input:focus {
            outline: 2px solid #667eea !important;
            outline-offset: 2px !important;
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    def animated_metric(self, label: str, value: str, delta: Optional[str] = None, 
                       animation_type: str = "slide-in-up"):
        """Display an animated metric card."""
        animation_class = f"{animation_type} metric-card"
        
        html = f"""
        <div class="{animation_class}">
            <h3 style="margin: 0; color: #333; font-size: 1.1em;">{label}</h3>
            <h1 style="margin: 10px 0; color: #667eea; font-size: 2.5em;">{value}</h1>
            {f'<p style="margin: 0; color: #666; font-size: 0.9em;">{delta}</p>' if delta else ''}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    def animated_progress_bar(self, progress: float, label: str = "", 
                            color: str = "#667eea"):
        """Display an animated progress bar."""
        html = f"""
        <div style="margin: 20px 0;">
            {f'<p style="margin-bottom: 10px; color: #333;">{label}</p>' if label else ''}
            <div style="background: #f0f0f0; border-radius: 10px; height: 20px; overflow: hidden;">
                <div class="animated-progress" style="
                    --progress-width: {progress}%;
                    height: 100%;
                    background: linear-gradient(90deg, {color}, {color}aa);
                    border-radius: 10px;
                    transition: width 1.5s ease-out;
                "></div>
            </div>
            <p style="text-align: right; margin-top: 5px; color: #666; font-size: 0.9em;">
                {progress:.1f}%
            </p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    def typing_text(self, text: str, speed: float = 2.0):
        """Display text with typing animation."""
        html = f"""
        <div class="typing-animation" style="
            animation-duration: {speed}s;
            font-size: 1.2em;
            color: #333;
            border-right: 2px solid #667eea;
            padding-right: 5px;
        ">
            {text}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    def floating_element(self, content: str, delay: float = 0):
        """Create a floating animated element."""
        html = f"""
        <div class="float-animation" style="
            animation-delay: {delay}s;
            display: inline-block;
            margin: 10px;
        ">
            {content}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    def gradient_title(self, title: str, size: str = "2em"):
        """Display a title with animated gradient text."""
        html = f"""
        <h1 class="gradient-text" style="
            font-size: {size};
            text-align: center;
            margin: 20px 0;
            font-weight: bold;
        ">
            {title}
        </h1>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    def success_animation(self, message: str):
        """Display success message with bounce animation."""
        html = f"""
        <div class="bounce-animation" style="
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        ">
            ✅ {message}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    def error_animation(self, message: str):
        """Display error message with shake animation."""
        html = f"""
        <div class="shake-animation" style="
            background: linear-gradient(135deg, #f44336, #d32f2f);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        ">
            ❌ {message}
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    def loading_spinner(self, message: str = "Loading..."):
        """Display an animated loading spinner."""
        html = f"""
        <div style="text-align: center; margin: 40px 0;">
            <div class="pulse-animation" style="
                width: 60px;
                height: 60px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                margin: 0 auto 20px;
                animation: spin 1s linear infinite;
            "></div>
            <p style="color: #666; font-size: 1.1em;">{message}</p>
        </div>
        
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
        st.markdown(html, unsafe_allow_html=True)

# Global instance
_micro_interactions = None

def get_advanced_micro_interactions():
    """Get the global AdvancedMicroInteractions instance."""
    global _micro_interactions
    if _micro_interactions is None:
        _micro_interactions = AdvancedMicroInteractions()
    return _micro_interactions
