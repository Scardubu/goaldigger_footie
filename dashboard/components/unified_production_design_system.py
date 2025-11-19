"""
Unified Design System - Production CSS Consolidation

Mobile-first responsive design system consolidating all CSS from:
- unified_design_system.css
- unified-production-style.css
- enhanced_dashboard_layout.css
- inline styles from dashboard components

Features:
- Mobile-first responsive breakpoints (320px, 768px, 1024px, 1440px)
- Performance-optimized animations (GPU-accelerated)
- Accessibility compliance (WCAG 2.1 AA)
- Dark mode support
- Professional visual hierarchy
"""

import logging
from typing import Literal, Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Responsive breakpoints (mobile-first)
BREAKPOINTS = {
    'mobile': '320px',      # Small phones
    'tablet': '768px',      # Tablets and large phones
    'desktop': '1024px',    # Small desktops and laptops
    'wide': '1440px'        # Large desktops
}


class UnifiedProductionDesignSystem:
    """
    Consolidated design system for GoalDiggers production dashboard.
    Replaces all previous CSS systems with single unified approach.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._css_injected = False
    
    def inject_unified_production_css(self, theme: Literal['light', 'dark'] = 'light'):
        """
        Inject consolidated production CSS with mobile-first responsive design.
        
        Args:
            theme: Color theme ('light' or 'dark')
        """
        if self._css_injected:
            return
        
        css = self._build_unified_css(theme)
        st.markdown(css, unsafe_allow_html=True)
        self._css_injected = True
        self.logger.info(f"✅ Unified production CSS injected (theme: {theme})")
    
    def _build_unified_css(self, theme: str) -> str:
        """Build complete unified CSS with all features."""
        return f"""
        <style>
        /* ========================================
           GOALDIGGERS UNIFIED PRODUCTION CSS v4.0
           Mobile-First Responsive Design System
           ======================================== */
        
        /* === DESIGN TOKENS === */
        :root {{
            /* Brand Colors - Trust & Authority */
            --gd-primary: #1e3c72;
            --gd-primary-light: #3b82f6;
            --gd-primary-dark: #1d4ed8;
            --gd-secondary: #2a5298;
            --gd-accent: #10b981;
            --gd-accent-cyan: #00d2ff;
            
            /* Premium Sports Broadcast Palette */
            --gd-brand-navy-900: #0e1b41;
            --gd-brand-navy-700: #162b59;
            --gd-brand-black: #0b0f14;
            --gd-brand-gold-500: #f5c518;
            --gd-brand-gold-600: #e0b715;
            --gd-brand-emerald-500: #10b981;
            --gd-brand-emerald-600: #059669;
            
            /* Semantic Colors */
            --gd-success: #10b981;
            --gd-success-light: #34d399;
            --gd-warning: #f59e0b;
            --gd-warning-light: #fbbf24;
            --gd-error: #ef4444;
            --gd-error-light: #f87171;
            --gd-info: #3b82f6;
            
            /* Neutral Colors */
            --gd-white: #ffffff;
            --gd-gray-50: #f9fafb;
            --gd-gray-100: #f3f4f6;
            --gd-gray-200: #e5e7eb;
            --gd-gray-300: #d1d5db;
            --gd-gray-400: #9ca3af;
            --gd-gray-500: #6b7280;
            --gd-gray-600: #4b5563;
            --gd-gray-700: #374151;
            --gd-gray-800: #1f2937;
            --gd-gray-900: #111827;
            
            /* Typography */
            --gd-font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            --gd-font-mono: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
            
            /* Font Sizes - Mobile First */
            --gd-text-xs: clamp(0.75rem, 2vw, 0.75rem);    /* 12px */
            --gd-text-sm: clamp(0.875rem, 2.5vw, 0.875rem);/* 14px */
            --gd-text-base: clamp(0.875rem, 3vw, 1rem);    /* 14-16px fluid */
            --gd-text-lg: clamp(1rem, 3.5vw, 1.125rem);    /* 16-18px fluid */
            --gd-text-xl: clamp(1.125rem, 4vw, 1.25rem);   /* 18-20px fluid */
            --gd-text-2xl: clamp(1.25rem, 5vw, 1.5rem);    /* 20-24px fluid */
            --gd-text-3xl: clamp(1.5rem, 6vw, 1.875rem);   /* 24-30px fluid */
            --gd-text-4xl: clamp(1.75rem, 7vw, 2.25rem);   /* 28-36px fluid */
            
            /* Spacing Scale - Mobile First */
            --gd-space-1: 0.25rem;    /* 4px */
            --gd-space-2: 0.5rem;     /* 8px */
            --gd-space-3: 0.75rem;    /* 12px */
            --gd-space-4: 1rem;       /* 16px */
            --gd-space-5: 1.25rem;    /* 20px */
            --gd-space-6: 1.5rem;     /* 24px */
            --gd-space-8: 2rem;       /* 32px */
            --gd-space-10: 2.5rem;    /* 40px */
            --gd-space-12: 3rem;      /* 48px */
            
            /* Border Radius */
            --gd-radius-sm: 4px;
            --gd-radius-md: 8px;
            --gd-radius-lg: 12px;
            --gd-radius-xl: 16px;
            --gd-radius-full: 9999px;
            
            /* Shadows - Performance Optimized */
            --gd-shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --gd-shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --gd-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --gd-shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            
            /* Transitions - GPU Accelerated */
            --gd-transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --gd-transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
            --gd-transition-slow: 300ms cubic-bezier(0.4, 0, 0.2, 1);
            
            /* Z-index Scale */
            --gd-z-dropdown: 1000;
            --gd-z-sticky: 1020;
            --gd-z-fixed: 1030;
            --gd-z-modal-backdrop: 1040;
            --gd-z-modal: 1050;
            --gd-z-popover: 1060;
            --gd-z-tooltip: 1070;
        }}
        
        /* === BASE STYLES === */
        
        /* Mobile-first base styles */
        .stApp {{
            font-family: var(--gd-font-family);
            color: var(--gd-gray-900);
            background-color: var(--gd-gray-50);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        /* Hide Streamlit branding on mobile */
        #MainMenu, footer, header {{
            display: none !important;
        }}
        
        /* === TYPOGRAPHY === */
        
        h1, h2, h3, h4, h5, h6 {{
            font-family: var(--gd-font-family);
            font-weight: 700;
            color: var(--gd-gray-900);
            line-height: 1.2;
            margin-top: var(--gd-space-6);
            margin-bottom: var(--gd-space-4);
        }}
        
        h1 {{ font-size: var(--gd-text-4xl); }}
        h2 {{ font-size: var(--gd-text-3xl); }}
        h3 {{ font-size: var(--gd-text-2xl); }}
        h4 {{ font-size: var(--gd-text-xl); }}
        h5 {{ font-size: var(--gd-text-lg); }}
        h6 {{ font-size: var(--gd-text-base); }}
        
        p {{
            margin-bottom: var(--gd-space-4);
            color: var(--gd-gray-700);
        }}
        
        /* === LAYOUT COMPONENTS === */
        
        /* Container - Mobile First */
        .gd-container {{
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            padding: var(--gd-space-4);
        }}
        
        /* Tablet */
        @media (min-width: {BREAKPOINTS['tablet']}) {{
            .gd-container {{
                max-width: 720px;
                padding: var(--gd-space-6);
            }}
        }}
        
        /* Desktop */
        @media (min-width: {BREAKPOINTS['desktop']}) {{
            .gd-container {{
                max-width: 960px;
                padding: var(--gd-space-8);
            }}
        }}
        
        /* Wide Desktop */
        @media (min-width: {BREAKPOINTS['wide']}) {{
            .gd-container {{
                max-width: 1200px;
            }}
        }}
        
        /* === CARD COMPONENTS === */
        
        /* Base card - Mobile First */
        .gd-card {{
            background-color: var(--gd-white);
            border-radius: var(--gd-radius-lg);
            padding: var(--gd-space-4);
            margin-bottom: var(--gd-space-4);
            box-shadow: var(--gd-shadow-sm);
            border: 1px solid var(--gd-gray-200);
            transition: all var(--gd-transition-base);
            will-change: transform, box-shadow;
        }}
        
        .gd-card:hover {{
            box-shadow: var(--gd-shadow-md);
            transform: translateY(-2px);
        }}
        
        /* Tablet and up - larger padding */
        @media (min-width: {BREAKPOINTS['tablet']}) {{
            .gd-card {{
                padding: var(--gd-space-6);
            }}
        }}
        
        /* Featured card with gradient */
        .gd-card--featured {{
            background: linear-gradient(135deg, var(--gd-primary) 0%, var(--gd-secondary) 100%);
            color: var(--gd-white);
            border: none;
        }}
        
        .gd-card--featured h3,
        .gd-card--featured p {{
            color: var(--gd-white);
        }}
        
        /* === BUTTON COMPONENTS === */
        
        /* Base button - Mobile First */
        .gd-button {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: var(--gd-space-3) var(--gd-space-5);
            font-size: var(--gd-text-sm);
            font-weight: 600;
            line-height: 1;
            border-radius: var(--gd-radius-md);
            border: none;
            cursor: pointer;
            transition: all var(--gd-transition-fast);
            text-decoration: none;
            white-space: nowrap;
            will-change: transform, background-color;
        }}
        
        /* Tablet and up - larger buttons */
        @media (min-width: {BREAKPOINTS['tablet']}) {{
            .gd-button {{
                padding: var(--gd-space-3) var(--gd-space-6);
                font-size: var(--gd-text-base);
            }}
        }}
        
        /* Button variants */
        .gd-button--primary {{
            background-color: var(--gd-primary);
            color: var(--gd-white);
        }}
        
        .gd-button--primary:hover {{
            background-color: var(--gd-primary-dark);
            transform: translateY(-1px);
            box-shadow: var(--gd-shadow-md);
        }}
        
        .gd-button--success {{
            background-color: var(--gd-success);
            color: var(--gd-white);
        }}
        
        .gd-button--success:hover {{
            background-color: var(--gd-brand-emerald-600);
        }}
        
        /* === STATUS INDICATORS === */
        
        .gd-badge {{
            display: inline-flex;
            align-items: center;
            padding: var(--gd-space-1) var(--gd-space-3);
            font-size: var(--gd-text-xs);
            font-weight: 600;
            line-height: 1;
            border-radius: var(--gd-radius-full);
            white-space: nowrap;
        }}
        
        .gd-badge--success {{
            background-color: rgba(16, 185, 129, 0.1);
            color: var(--gd-brand-emerald-600);
        }}
        
        .gd-badge--warning {{
            background-color: rgba(245, 158, 11, 0.1);
            color: #92400e;
        }}
        
        .gd-badge--error {{
            background-color: rgba(239, 68, 68, 0.1);
            color: #991b1b;
        }}
        
        /* === METRICS === */
        
        /* Metric card - Mobile First */
        .gd-metric {{
            background-color: var(--gd-white);
            border-radius: var(--gd-radius-lg);
            padding: var(--gd-space-4);
            text-align: center;
            border: 1px solid var(--gd-gray-200);
            transition: all var(--gd-transition-base);
        }}
        
        @media (min-width: {BREAKPOINTS['tablet']}) {{
            .gd-metric {{
                padding: var(--gd-space-6);
            }}
        }}
        
        .gd-metric__label {{
            font-size: var(--gd-text-sm);
            color: var(--gd-gray-600);
            margin-bottom: var(--gd-space-2);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .gd-metric__value {{
            font-size: var(--gd-text-3xl);
            font-weight: 700;
            color: var(--gd-gray-900);
            margin-bottom: var(--gd-space-1);
        }}
        
        .gd-metric__change {{
            font-size: var(--gd-text-sm);
            font-weight: 600;
        }}
        
        .gd-metric__change--positive {{
            color: var(--gd-success);
        }}
        
        .gd-metric__change--negative {{
            color: var(--gd-error);
        }}
        
        /* === GRID SYSTEM === */
        
        /* Mobile-first grid */
        .gd-grid {{
            display: grid;
            gap: var(--gd-space-4);
            grid-template-columns: 1fr;
        }}
        
        /* Tablet - 2 columns */
        @media (min-width: {BREAKPOINTS['tablet']}) {{
            .gd-grid {{
                grid-template-columns: repeat(2, 1fr);
                gap: var(--gd-space-6);
            }}
            
            .gd-grid--3 {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}
        
        /* Desktop - preserve columns or expand */
        @media (min-width: {BREAKPOINTS['desktop']}) {{
            .gd-grid {{
                gap: var(--gd-space-8);
            }}
            
            .gd-grid--4 {{
                grid-template-columns: repeat(4, 1fr);
            }}
        }}
        
        /* === STREAMLIT OVERRIDES === */
        
        /* Fix Streamlit metrics on mobile */
        [data-testid="stMetric"] {{
            background-color: var(--gd-white);
            border-radius: var(--gd-radius-lg);
            padding: var(--gd-space-4) !important;
            border: 1px solid var(--gd-gray-200);
        }}
        
        @media (min-width: {BREAKPOINTS['tablet']}) {{
            [data-testid="stMetric"] {{
                padding: var(--gd-space-6) !important;
            }}
        }}
        
        [data-testid="stMetricLabel"] {{
            font-size: var(--gd-text-sm) !important;
            color: var(--gd-gray-600) !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
        }}
        
        [data-testid="stMetricValue"] {{
            font-size: var(--gd-text-3xl) !important;
            font-weight: 700 !important;
            color: var(--gd-gray-900) !important;
        }}
        
        /* Streamlit buttons */
        .stButton > button {{
            width: 100%;
            padding: var(--gd-space-3) var(--gd-space-5);
            font-size: var(--gd-text-sm);
            font-weight: 600;
            border-radius: var(--gd-radius-md);
            background: linear-gradient(135deg, var(--gd-primary) 0%, var(--gd-secondary) 100%);
            color: var(--gd-white);
            border: none;
            transition: all var(--gd-transition-base);
        }}
        
        @media (min-width: {BREAKPOINTS['tablet']}) {{
            .stButton > button {{
                width: auto;
                padding: var(--gd-space-3) var(--gd-space-6);
                font-size: var(--gd-text-base);
            }}
        }}
        
        .stButton > button:hover {{
            background: linear-gradient(135deg, var(--gd-primary-dark) 0%, var(--gd-primary) 100%);
            transform: translateY(-2px);
            box-shadow: var(--gd-shadow-lg);
        }}
        
        /* === ANIMATIONS === */
        
        /* Fade in animation - GPU accelerated */
        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        .gd-fade-in {{
            animation: fadeIn var(--gd-transition-slow) ease-out;
        }}
        
        /* Pulse animation */
        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.7;
            }}
        }}
        
        .gd-pulse {{
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }}
        
        /* === ACCESSIBILITY === */
        
        /* Focus visible styles */
        *:focus-visible {{
            outline: 2px solid var(--gd-primary);
            outline-offset: 2px;
        }}
        
        /* Reduce motion for users who prefer it */
        @media (prefers-reduced-motion: reduce) {{
            * {{
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }}
        }}
        
        /* === DARK MODE SUPPORT === */
        
        @media (prefers-color-scheme: dark) {{
            :root {{
                --gd-gray-50: #111827;
                --gd-gray-100: #1f2937;
                --gd-gray-900: #f9fafb;
                --gd-white: #1f2937;
            }}
            
            .stApp {{
                background-color: var(--gd-gray-50);
                color: var(--gd-gray-900);
            }}
            
            .gd-card {{
                background-color: var(--gd-gray-100);
                border-color: var(--gd-gray-700);
            }}
        }}
        
        /* === PRINT STYLES === */
        
        @media print {{
            .gd-card {{
                box-shadow: none;
                border: 1px solid var(--gd-gray-300);
                page-break-inside: avoid;
            }}
            
            .stButton {{
                display: none;
            }}
        }}
        
        /* === UTILITY CLASSES === */
        
        .gd-text-center {{ text-align: center; }}
        .gd-text-right {{ text-align: right; }}
        .gd-text-left {{ text-align: left; }}
        
        .gd-mb-0 {{ margin-bottom: 0 !important; }}
        .gd-mb-4 {{ margin-bottom: var(--gd-space-4); }}
        .gd-mb-6 {{ margin-bottom: var(--gd-space-6); }}
        .gd-mb-8 {{ margin-bottom: var(--gd-space-8); }}
        
        .gd-mt-0 {{ margin-top: 0 !important; }}
        .gd-mt-4 {{ margin-top: var(--gd-space-4); }}
        .gd-mt-6 {{ margin-top: var(--gd-space-6); }}
        .gd-mt-8 {{ margin-top: var(--gd-space-8); }}
        
        .gd-hidden-mobile {{
            display: none;
        }}
        
        @media (min-width: {BREAKPOINTS['tablet']}) {{
            .gd-hidden-mobile {{
                display: block;
            }}
            
            .gd-hidden-tablet {{
                display: none;
            }}
        }}
        
        @media (min-width: {BREAKPOINTS['desktop']}) {{
            .gd-hidden-desktop {{
                display: none;
            }}
        }}
        
        /* === PERFORMANCE OPTIMIZATIONS === */
        
        /* Enable GPU acceleration for animations */
        .gd-card,
        .gd-button,
        .gd-metric {{
            transform: translateZ(0);
            backface-visibility: hidden;
            perspective: 1000px;
        }}
        
        /* Optimize repaints */
        .gd-card:hover,
        .gd-button:hover {{
            will-change: transform;
        }}
        
        </style>
        """
    
    def get_responsive_container_class(self) -> str:
        """Get responsive container class name."""
        return "gd-container"
    
    def get_card_class(self, variant: Optional[str] = None) -> str:
        """
        Get card class name with optional variant.
        
        Args:
            variant: Card variant ('featured', None)
        """
        base_class = "gd-card"
        if variant:
            return f"{base_class} {base_class}--{variant}"
        return base_class
    
    def get_button_class(self, variant: str = 'primary') -> str:
        """
        Get button class name with variant.
        
        Args:
            variant: Button variant ('primary', 'success', etc.)
        """
        return f"gd-button gd-button--{variant}"
    
    def get_badge_class(self, status: str) -> str:
        """
        Get badge class name for status.
        
        Args:
            status: Status type ('success', 'warning', 'error')
        """
        return f"gd-badge gd-badge--{status}"


# Global singleton instance
_unified_design_system = None


def get_unified_production_design_system() -> UnifiedProductionDesignSystem:
    """Get global unified production design system instance."""
    global _unified_design_system
    if _unified_design_system is None:
        _unified_design_system = UnifiedProductionDesignSystem()
    return _unified_design_system


# Convenience function
def inject_production_css(theme: Literal['light', 'dark'] = 'light'):
    """Inject unified production CSS."""
    get_unified_production_design_system().inject_unified_production_css(theme)


if __name__ == "__main__":
    # Test design system
    design_system = UnifiedProductionDesignSystem()
    print("✅ Unified Production Design System initialized")
    print(f"📱 Responsive breakpoints: {BREAKPOINTS}")
    print(f"🎨 Container class: {design_system.get_responsive_container_class()}")
    print(f"🃏 Card class: {design_system.get_card_class()}")
    print(f"🔘 Button class: {design_system.get_button_class('primary')}")
    print(f"🏷️ Badge class: {design_system.get_badge_class('success')}")
