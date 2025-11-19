"""
Mobile device detection utilities for GoalDiggers platform.

Provides reliable mobile device detection using user agent analysis
and responsive design considerations.
"""

import logging
import re
from typing import Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)


class MobileDetector:
    """Enhanced mobile device detection with caching."""
    
    def __init__(self):
        self.mobile_patterns = [
            r'Mobile|Android|iPhone|iPad|iPod|webOS|BlackBerry|Windows Phone',
            r'Opera Mini|Opera Mobi|Mobile Safari',
            r'Mobile.*Firefox|Mobile.*Chrome',
            r'Edge Mobile|SamsungBrowser'
        ]
        self.tablet_patterns = [
            r'iPad|Android.*Tablet|Surface|Kindle Fire',
            r'KFAPWI|KFOT|KFJWI|KFTHWI|KFTT'
        ]
        self._compiled_patterns = {
            'mobile': [re.compile(pattern, re.IGNORECASE) for pattern in self.mobile_patterns],
            'tablet': [re.compile(pattern, re.IGNORECASE) for pattern in self.tablet_patterns]
        }
    
    def detect_from_user_agent(self, user_agent: str) -> Dict[str, bool]:
        """Detect device type from user agent string."""
        if not user_agent:
            return {'is_mobile': False, 'is_tablet': False, 'is_desktop': True}
        
        is_tablet = any(pattern.search(user_agent) for pattern in self._compiled_patterns['tablet'])
        is_mobile = any(pattern.search(user_agent) for pattern in self._compiled_patterns['mobile'])
        
        # iPad detection enhancement
        if 'iPad' in user_agent or ('Macintosh' in user_agent and 'Safari' in user_agent and 'Touch' in user_agent):
            is_tablet = True
            is_mobile = False
        
        is_desktop = not (is_mobile or is_tablet)
        
        return {
            'is_mobile': is_mobile and not is_tablet,
            'is_tablet': is_tablet,
            'is_desktop': is_desktop
        }
    
    def detect_from_viewport(self) -> Dict[str, bool]:
        """Detect device type from viewport size using JavaScript."""
        detection_script = """
        <script>
        function detectDeviceFromViewport() {
            const width = window.innerWidth || document.documentElement.clientWidth;
            const height = window.innerHeight || document.documentElement.clientHeight;
            const aspectRatio = width / height;
            
            // Device detection based on viewport
            const isMobile = width < 768;
            const isTablet = width >= 768 && width < 1024;
            const isDesktop = width >= 1024;
            
            // Touch capability detection
            const hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
            
            // Orientation detection
            const orientation = width > height ? 'landscape' : 'portrait';
            
            // Store results
            sessionStorage.setItem('device_detection', JSON.stringify({
                is_mobile: isMobile,
                is_tablet: isTablet,
                is_desktop: isDesktop,
                has_touch: hasTouch,
                orientation: orientation,
                viewport: { width: width, height: height },
                aspect_ratio: aspectRatio,
                timestamp: Date.now()
            }));
        }
        
        // Run detection
        detectDeviceFromViewport();
        
        // Re-run on resize
        window.addEventListener('resize', detectDeviceFromViewport);
        window.addEventListener('orientationchange', detectDeviceFromViewport);
        </script>
        """
        
        st.markdown(detection_script, unsafe_allow_html=True)
        
        # Default fallback
        return {'is_mobile': False, 'is_tablet': False, 'is_desktop': True}


# Global detector instance
_detector = MobileDetector()


def detect_mobile(user_agent: Optional[str] = None) -> bool:
    """
    Detect if the current device is mobile.
    
    Args:
        user_agent: Optional user agent string. If not provided, will attempt
                   to detect from Streamlit context.
    
    Returns:
        bool: True if mobile device detected, False otherwise.
    """
    try:
        # Use cached result if available
        if 'is_mobile_device' in st.session_state:
            return st.session_state.is_mobile_device
        
        # Try to get user agent from various sources
        if not user_agent:
            # Try to get from Streamlit headers (if available)
            try:
                headers = st.context.headers if hasattr(st.context, 'headers') else {}
                user_agent = headers.get('user-agent', '')
            except Exception:
                # Fallback for older Streamlit versions
                user_agent = ''
        
        # Perform detection
        detection_result = _detector.detect_from_user_agent(user_agent)
        is_mobile = detection_result['is_mobile']
        
        # Cache result
        st.session_state.is_mobile_device = is_mobile
        st.session_state.is_tablet_device = detection_result['is_tablet']
        st.session_state.is_desktop_device = detection_result['is_desktop']
        
        # Also run viewport detection for enhanced accuracy
        _detector.detect_from_viewport()
        
        logger.debug(f"Mobile detection result: {detection_result}")
        return is_mobile
        
    except Exception as e:
        logger.warning(f"Mobile detection failed: {e}")
        # Conservative fallback - assume desktop
        return False


def is_tablet() -> bool:
    """Detect if the current device is a tablet."""
    try:
        if 'is_tablet_device' in st.session_state:
            return st.session_state.is_tablet_device
        
        # Trigger mobile detection which also sets tablet info
        detect_mobile()
        return st.session_state.get('is_tablet_device', False)
        
    except Exception as e:
        logger.warning(f"Tablet detection failed: {e}")
        return False


def is_desktop() -> bool:
    """Detect if the current device is desktop."""
    try:
        if 'is_desktop_device' in st.session_state:
            return st.session_state.is_desktop_device
        
        # Trigger mobile detection which also sets desktop info
        detect_mobile()
        return st.session_state.get('is_desktop_device', True)
        
    except Exception as e:
        logger.warning(f"Desktop detection failed: {e}")
        return True


def get_device_info() -> Dict[str, any]:
    """Get comprehensive device information."""
    try:
        # Ensure detection has been run
        detect_mobile()
        
        return {
            'is_mobile': st.session_state.get('is_mobile_device', False),
            'is_tablet': st.session_state.get('is_tablet_device', False), 
            'is_desktop': st.session_state.get('is_desktop_device', True),
            'detection_timestamp': st.session_state.get('device_detection_timestamp'),
            'user_agent_available': bool(st.session_state.get('user_agent_string')),
        }
        
    except Exception as e:
        logger.error(f"Failed to get device info: {e}")
        return {
            'is_mobile': False,
            'is_tablet': False,
            'is_desktop': True,
            'detection_timestamp': None,
            'user_agent_available': False,
        }


def reset_device_detection():
    """Reset cached device detection results."""
    try:
        keys_to_remove = [
            'is_mobile_device',
            'is_tablet_device', 
            'is_desktop_device',
            'device_detection_timestamp',
            'user_agent_string'
        ]
        
        for key in keys_to_remove:
            if key in st.session_state:
                del st.session_state[key]
                
        logger.debug("Device detection cache cleared")
        
    except Exception as e:
        logger.warning(f"Failed to reset device detection: {e}")


# Enhanced viewport detection with JavaScript integration
def inject_enhanced_mobile_detection():
    """Inject enhanced JavaScript-based mobile detection."""
    detection_js = """
    <script>
    (function() {
        'use strict';
        
        function performEnhancedDetection() {
            const results = {
                // Screen metrics
                screen_width: screen.width,
                screen_height: screen.height,
                viewport_width: window.innerWidth,
                viewport_height: window.innerHeight,
                pixel_ratio: window.devicePixelRatio || 1,
                
                // Touch capabilities
                has_touch: 'ontouchstart' in window || navigator.maxTouchPoints > 0,
                max_touch_points: navigator.maxTouchPoints || 0,
                
                // Device orientation
                orientation: screen.orientation ? screen.orientation.type : 'unknown',
                orientation_angle: screen.orientation ? screen.orientation.angle : 0,
                
                // Connection info (if available)
                connection_type: navigator.connection ? navigator.connection.effectiveType : 'unknown',
                
                // Device memory (if available)
                device_memory: navigator.deviceMemory || 'unknown',
                
                // Hardware concurrency
                hardware_concurrency: navigator.hardwareConcurrency || 'unknown',
                
                // User agent
                user_agent: navigator.userAgent,
                
                // Platform
                platform: navigator.platform,
                
                // Timestamp
                timestamp: Date.now()
            };
            
            // Store in sessionStorage for Streamlit access
            sessionStorage.setItem('enhanced_device_info', JSON.stringify(results));
            
            // Dispatch custom event
            window.dispatchEvent(new CustomEvent('deviceDetectionComplete', { detail: results }));
        }
        
        // Run detection
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', performEnhancedDetection);
        } else {
            performEnhancedDetection();
        }
        
        // Re-run on significant events
        window.addEventListener('resize', performEnhancedDetection);
        window.addEventListener('orientationchange', performEnhancedDetection);
        
        // Debug logging
        window.addEventListener('deviceDetectionComplete', function(e) {
            console.log('GoalDiggers: Enhanced device detection completed', e.detail);
        });
        
    })();
    </script>
    """
    
    st.markdown(detection_js, unsafe_allow_html=True)


# Export main function for compatibility
__all__ = [
    'detect_mobile',
    'is_tablet', 
    'is_desktop',
    'get_device_info',
    'reset_device_detection',
    'inject_enhanced_mobile_detection',
    'MobileDetector'
]
