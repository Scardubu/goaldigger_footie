"""
Enhanced Progressive Web App (PWA) implementation for the GoalDiggers Football Betting Platform.

This module provides comprehensive PWA functionality including:
- Service worker registration and management
- Offline functionality with intelligent caching
- Push notifications with enhanced UX
- Mobile-optimized interface components
- Performance monitoring and analytics
- Installation prompts and app management
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Conditional imports for mobile components
try:
    from dashboard.components.consistent_styling import ConsistentStyling
    from dashboard.mobile.mobile_detection import detect_mobile
    MOBILE_COMPONENTS_AVAILABLE = True
except ImportError:
    MOBILE_COMPONENTS_AVAILABLE = False
    logging.warning("Mobile components not available, PWA will run in limited mode")

# Try to import psutil for memory monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PWAFeature(Enum):
    """Enumeration of available PWA features."""
    OFFLINE_SUPPORT = "offline_support"
    PUSH_NOTIFICATIONS = "push_notifications"
    APP_INSTALL = "app_install"
    BACKGROUND_SYNC = "background_sync"
    PERFORMANCE_MONITORING = "performance_monitoring"


class CacheStrategy(Enum):
    """Enumeration of caching strategies for PWA."""
    CACHE_FIRST = "cache-first"
    NETWORK_FIRST = "network-first"
    STALE_WHILE_REVALIDATE = "stale-while-revalidate"


@dataclass
class PWAConfig:
    """Configuration class for PWA implementation."""
    # Basic PWA settings
    enable_pwa: bool = True
    app_name: str = "GoalDiggers"
    app_short_name: str = "GoalDiggers"
    app_description: str = "Advanced Football Betting Platform with AI Predictions"
    theme_color: str = "#667eea"
    background_color: str = "#ffffff"
    
    # Feature toggles
    enable_offline: bool = True
    enable_push_notifications: bool = True
    enable_app_install: bool = True
    enable_background_sync: bool = True
    enable_performance_monitoring: bool = True
    
    # Performance settings
    cache_strategy: CacheStrategy = CacheStrategy.STALE_WHILE_REVALIDATE
    cache_max_age: int = 86400  # 24 hours
    cache_max_entries: int = 50
    lazy_load_images: bool = True
    preload_critical_resources: bool = True
    
    # Advanced settings
    sw_update_check_interval: int = 3600  # 1 hour
    notification_badge_color: str = "#ff4444"
    install_prompt_delay: int = 3000  # 3 seconds
    
    # Security settings
    csp_enabled: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


class PWAPerformanceMonitor:
    """Performance monitoring for PWA components."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def measure(self, operation_name: str):
        """Context manager for measuring operation performance."""
        return self._PerformanceMeasure(self, operation_name)
    
    class _PerformanceMeasure:
        def __init__(self, monitor, operation_name):
            self.monitor = monitor
            self.operation_name = operation_name
            
        def __enter__(self):
            self.monitor.start_times[self.operation_name] = time.time()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.monitor.start_times[self.operation_name]
            
            if self.operation_name not in self.monitor.metrics:
                self.monitor.metrics[self.operation_name] = {
                    'count': 0,
                    'total_time': 0,
                    'average': 0,
                    'min': float('inf'),
                    'max': 0
                }
            
            metric = self.monitor.metrics[self.operation_name]
            metric['count'] += 1
            metric['total_time'] += duration
            metric['average'] = metric['total_time'] / metric['count']
            metric['min'] = min(metric['min'], duration)
            metric['max'] = max(metric['max'], duration)
    
    def get_performance_report(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive performance report."""
        return self.metrics.copy()


class ServiceWorkerOptimizer:
    """Optimizes service worker code generation and management."""
    
    def __init__(self, config: PWAConfig):
        self.config = config
        
    def generate_service_worker(self, dashboard_variant: str = "premium_ui") -> str:
        """Generate optimized service worker code for specific dashboard variant."""
        cache_strategy = self._get_variant_cache_strategy(dashboard_variant)
        
        return f"""
// Enhanced Service Worker for GoalDiggers PWA - {dashboard_variant}
const CACHE_NAME = 'goaldiggers-v1.0.0-{dashboard_variant}';
const CACHE_STRATEGY = '{cache_strategy}';
const CACHE_MAX_AGE = {self.config.cache_max_age};
const CACHE_MAX_ENTRIES = {self.config.cache_max_entries};

// Essential resources to cache
const ESSENTIAL_RESOURCES = [
    '/',
    '/static/css/main.css',
    '/static/js/main.js',
    '/manifest.json'
];

// Install event - cache essential resources
self.addEventListener('install', event => {{
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(ESSENTIAL_RESOURCES))
            .then(() => self.skipWaiting())
    );
}});

// Activate event - clean old caches
self.addEventListener('activate', event => {{
    event.waitUntil(
        caches.keys().then(cacheNames => {{
            return Promise.all(
                cacheNames.map(cacheName => {{
                    if (cacheName !== CACHE_NAME) {{
                        return caches.delete(cacheName);
                    }}
                }})
            );
        }}).then(() => self.clients.claim())
    );
}});

// Fetch event - implement caching strategy
self.addEventListener('fetch', event => {{
    if (event.request.method !== 'GET') return;
    
    event.respondWith(
        handleFetchWithStrategy(event.request, CACHE_STRATEGY)
    );
}});

// Cache strategy implementation
async function handleFetchWithStrategy(request, strategy) {{
    const cache = await caches.open(CACHE_NAME);
    
    switch (strategy) {{
        case 'cache-first':
            return handleCacheFirst(request, cache);
        case 'network-first':
            return handleNetworkFirst(request, cache);
        case 'stale-while-revalidate':
        default:
            return handleStaleWhileRevalidate(request, cache);
    }}
}}

async function handleCacheFirst(request, cache) {{
    const cachedResponse = await cache.match(request);
    if (cachedResponse) return cachedResponse;
    
    try {{
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {{
            cache.put(request, networkResponse.clone());
        }}
        return networkResponse;
    }} catch (error) {{
        return new Response('Offline content unavailable', {{ status: 503 }});
    }}
}}

async function handleNetworkFirst(request, cache) {{
    try {{
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {{
            cache.put(request, networkResponse.clone());
        }}
        return networkResponse;
    }} catch (error) {{
        const cachedResponse = await cache.match(request);
        return cachedResponse || new Response('Offline content unavailable', {{ status: 503 }});
    }}
}}

async function handleStaleWhileRevalidate(request, cache) {{
    const cachedResponse = await cache.match(request);
    
    const fetchPromise = fetch(request).then(networkResponse => {{
        if (networkResponse.ok) {{
            cache.put(request, networkResponse.clone());
        }}
        return networkResponse;
    }}).catch(() => cachedResponse);
    
    return cachedResponse || fetchPromise;
}}

// Background sync for offline actions
self.addEventListener('sync', event => {{
    if (event.tag === 'background-sync') {{
        event.waitUntil(handleBackgroundSync());
    }}
}});

async function handleBackgroundSync() {{
    const syncData = await getSyncData();
    for (const data of syncData) {{
        try {{
            await processSyncData(data);
            await removeSyncData(data.id);
        }} catch (error) {{
            console.error('Sync failed:', error);
        }}
    }}
}}

async function getSyncData() {{
    return [];
}}

async function processSyncData(data) {{
    return fetch('/api/sync', {{
        method: 'POST',
        body: JSON.stringify(data),
        headers: {{ 'Content-Type': 'application/json' }}
    }});
}}

async function removeSyncData(id) {{
    console.log('SW: Removed sync data:', id);
}}
        """
        
    def _get_variant_cache_strategy(self, dashboard_variant: str) -> str:
        """Get optimized caching strategy for specific dashboard variant."""
        strategies = {
            'ultra_fast_premium': 'cache-first',
            'optimized_dashboard': 'cache-first', 
            'fast_production': 'cache-first',
            'premium_ui': 'stale-while-revalidate',
            'integrated_production': 'network-first',
            'interactive_cross_league': 'stale-while-revalidate'
        }
        return strategies.get(dashboard_variant, 'stale-while-revalidate')


class PWAImplementation:
    """
    Enhanced Progressive Web App implementation with optimized performance
    and improved code quality while maintaining full backward compatibility.
    """
    
    def __init__(self, config: PWAConfig = None):
        """Initialize enhanced PWA implementation."""
        self.config = config or PWAConfig()
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PWAPerformanceMonitor()
        self.service_worker_optimizer = ServiceWorkerOptimizer(self.config)
        
        # Initialize mobile detection with caching
        self._is_mobile = None
        
        # Initialize styling component with lazy loading
        self._styling = None
        
        # PWA state management with better defaults
        self.pwa_state = {
            'installed': False,
            'offline_mode': False,
            'notifications_enabled': False,
            'performance_mode': 'balanced'  # balanced, fast, battery-saver
        }
        
        self.logger.info("🚀 Enhanced PWA implementation initialized")
        
    def configure_page(self):
        """Configure PWA page settings."""
        try:
            # Set environment variable to indicate PWA is pre-configured
            os.environ["PWA_PRE_CONFIGURED"] = "true"
            
            # Initialize basic PWA interface
            self.render_pwa_interface()
            
            self.logger.debug("PWA page configuration completed")
        except Exception as e:
            self.logger.error(f"PWA page configuration failed: {e}")
    
    @property
    def is_mobile(self) -> bool:
        """Cached mobile detection."""
        if self._is_mobile is None:
            self._is_mobile = detect_mobile() if MOBILE_COMPONENTS_AVAILABLE else False
        return self._is_mobile
    
    @property
    def styling(self):
        """Lazy-loaded styling component."""
        if self._styling is None and MOBILE_COMPONENTS_AVAILABLE:
            self._styling = ConsistentStyling()
        return self._styling
    
    def render_pwa_interface(self, dashboard_variant: str = "premium_ui"):
        """Enhanced PWA interface rendering with performance monitoring."""
        with self.performance_monitor.measure("pwa_interface_render"):
            try:
                if not self.config.enable_pwa:
                    return

                # Inject optimized PWA components
                self._inject_pwa_manifest()
                self._inject_optimized_service_worker(dashboard_variant)

                # Apply variant-specific mobile optimizations
                self._apply_variant_optimizations(dashboard_variant)

                # Render mobile PWA controls if needed
                if self.is_mobile:
                    self._render_mobile_interface(dashboard_variant)

                # Setup enhanced offline functionality
                if self.config.enable_offline:
                    self._setup_offline_functionality(dashboard_variant)

                # Initialize push notifications
                if self.config.enable_push_notifications:
                    self._initialize_push_notifications()

                # Add installation prompt
                self._render_installation_interface(dashboard_variant)

                # Track usage analytics
                self._track_pwa_analytics(dashboard_variant)

            except Exception as e:
                self.logger.error(f"PWA interface rendering error: {e}")
                # Graceful degradation - continue without PWA features
                st.warning("⚠️ Some PWA features may be unavailable")
    
    @lru_cache(maxsize=1)
    def _generate_optimized_manifest(self) -> Dict[str, Any]:
        """Generate optimized PWA manifest with caching."""
        return {
            "name": self.config.app_name,
            "short_name": self.config.app_short_name,
            "description": self.config.app_description,
            "start_url": "/",
            "display": "standalone",
            "theme_color": self.config.theme_color,
            "background_color": self.config.background_color,
            "orientation": "any",
            "scope": "/",
            "lang": "en",
            "dir": "ltr",
            "categories": ["sports", "entertainment", "lifestyle"],
            "shortcuts": self._generate_shortcuts(),
            "screenshots": self._generate_screenshots(),
            "icons": [
                {
                    "src": "/static/icons/icon-72x72.png",
                    "sizes": "72x72",
                    "type": "image/png",
                    "purpose": "maskable any"
                },
                {
                    "src": "/static/icons/icon-96x96.png",
                    "sizes": "96x96",
                    "type": "image/png",
                    "purpose": "maskable any"
                },
                {
                    "src": "/static/icons/icon-128x128.png",
                    "sizes": "128x128",
                    "type": "image/png",
                    "purpose": "maskable any"
                },
                {
                    "src": "/static/icons/icon-144x144.png",
                    "sizes": "144x144",
                    "type": "image/png",
                    "purpose": "maskable any"
                },
                {
                    "src": "/static/icons/icon-152x152.png",
                    "sizes": "152x152",
                    "type": "image/png",
                    "purpose": "maskable any"
                },
                {
                    "src": "/static/icons/icon-192x192.png",
                    "sizes": "192x192",
                    "type": "image/png",
                    "purpose": "maskable any"
                },
                {
                    "src": "/static/icons/icon-384x384.png",
                    "sizes": "384x384",
                    "type": "image/png",
                    "purpose": "maskable any"
                },
                {
                    "src": "/static/icons/icon-512x512.png",
                    "sizes": "512x512",
                    "type": "image/png",
                    "purpose": "maskable any"
                }
            ]
        }
    
    def _generate_shortcuts(self) -> List[Dict[str, Any]]:
        """Generate app shortcuts for PWA manifest."""
        return [
            {
                "name": "Quick Predict",
                "short_name": "Predict",
                "description": "Make quick match predictions",
                "url": "/#predict",
                "icons": [{"src": "/static/icons/shortcut-predict.png", "sizes": "96x96"}]
            },
            {
                "name": "Live Matches",
                "short_name": "Live",
                "description": "View live match data",
                "url": "/#live",
                "icons": [{"src": "/static/icons/shortcut-live.png", "sizes": "96x96"}]
            },
            {
                "name": "Analytics",
                "short_name": "Analytics",
                "description": "View detailed analytics",
                "url": "/#analytics",
                "icons": [{"src": "/static/icons/shortcut-analytics.png", "sizes": "96x96"}]
            }
        ]
    
    def _generate_screenshots(self) -> List[Dict[str, Any]]:
        """Generate screenshot metadata."""
        return [
            {
                "src": "/static/screenshots/desktop-wide.png",
                "sizes": "1920x1080",
                "type": "image/png",
                "form_factor": "wide",
                "label": "Desktop view of GoalDiggers dashboard"
            },
            {
                "src": "/static/screenshots/mobile-narrow.png",
                "sizes": "375x812",
                "type": "image/png",
                "form_factor": "narrow",
                "label": "Mobile view of GoalDiggers dashboard"
            }
        ]
    
    def _inject_pwa_manifest(self):
        """Inject optimized PWA manifest."""
        with self.performance_monitor.measure("manifest_injection"):
            manifest = self._generate_optimized_manifest()
            manifest_json = json.dumps(manifest, separators=(',', ':'))  # Minimal JSON
            manifest_b64 = base64.b64encode(manifest_json.encode()).decode()
            
            # Check if page config is already set (from environment variable)
            if not os.environ.get("PWA_PRE_CONFIGURED"):
                try:
                    # Use Streamlit's set_page_config first for basic page configuration
                    st.set_page_config(
                        page_title=self.config.app_name,
                        page_icon="⚽",
                        layout="wide",
                        initial_sidebar_state="expanded"
                    )
                except Exception as e:
                    # If page config is already set, just log a warning
                    self.logger.warning(f"Page config already set: {e}")
            
            # Generate CSP nonce for security
            import secrets
            nonce = secrets.token_urlsafe(16)
            
            # Create a JavaScript function to inject meta tags into the head
            # This is the proper way to handle CSP and other head elements in Streamlit
            head_injection_js = f"""
            <script>
                function injectHeadElements() {{
                    // Manifest
                    let manifestLink = document.createElement('link');
                    manifestLink.rel = 'manifest';
                    manifestLink.href = 'data:application/json;base64,{manifest_b64}';
                    document.head.appendChild(manifestLink);
                    
                    // Theme and mobile settings
                    let metaTags = [
                        {{"name": "theme-color", "content": "{self.config.theme_color}"}},
                        {{"name": "apple-mobile-web-app-capable", "content": "yes"}},
                        {{"name": "apple-mobile-web-app-status-bar-style", "content": "default"}},
                        {{"name": "apple-mobile-web-app-title", "content": "{self.config.app_short_name}"}},
                        {{"name": "mobile-web-app-capable", "content": "yes"}},
                        {{"name": "application-name", "content": "{self.config.app_name}"}},
                        {{"name": "msapplication-TileColor", "content": "{self.config.theme_color}"}},
                        {{"name": "msapplication-config", "content": "/browserconfig.xml"}}
                    ];
                    
                    metaTags.forEach(tagInfo => {{
                        let tag = document.createElement('meta');
                        Object.keys(tagInfo).forEach(attr => {{
                            tag.setAttribute(attr, tagInfo[attr]);
                        }});
                        document.head.appendChild(tag);
                    }});
                    
                    // Icons
                    let iconLinks = [
                        {{"rel": "apple-touch-icon", "href": "/static/icons/apple-touch-icon.png"}},
                        {{"rel": "icon", "type": "image/png", "sizes": "32x32", "href": "/static/icons/favicon-32x32.png"}},
                        {{"rel": "icon", "type": "image/png", "sizes": "16x16", "href": "/static/icons/favicon-16x16.png"}}
                    ];
                    
                    iconLinks.forEach(linkInfo => {{
                        let link = document.createElement('link');
                        Object.keys(linkInfo).forEach(attr => {{
                            link.setAttribute(attr, linkInfo[attr]);
                        }});
                        document.head.appendChild(link);
                    }});
                    
                    // Add CSP if enabled (this won't have any effect since CSP must be in head before any content loads)
                    // But we'll keep it here for completeness
                    if ({str(self.config.csp_enabled).lower()}) {{
                        let cspMeta = document.createElement('meta');
                        cspMeta.httpEquiv = "Content-Security-Policy";
                        cspMeta.content = "default-src 'self'; script-src 'self' 'nonce-{nonce}' 'unsafe-inline'; style-src 'self' 'unsafe-inline';";
                        document.head.appendChild(cspMeta);
                    }}
                }}
                
                // Execute as soon as possible
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', injectHeadElements);
                }} else {{
                    injectHeadElements();
                }}
            </script>
            """
            
            # Inject the JavaScript 
            st.markdown(head_injection_js, unsafe_allow_html=True)
    
    def _inject_optimized_service_worker(self, dashboard_variant: str):
        """Inject optimized service worker with better error handling."""
        with self.performance_monitor.measure("service_worker_injection"):
            sw_code = self.service_worker_optimizer.generate_service_worker(dashboard_variant)
            sw_b64 = base64.b64encode(sw_code.encode()).decode()
            
            registration_script = f"""
            <script>
            (function() {{
                'use strict';
                
                // Enhanced service worker registration with better error handling
                if ('serviceWorker' in navigator && (window.location.protocol === 'https:' || window.location.hostname === 'localhost')) {{
                    let registrationPromise;
                    
                    function registerServiceWorker() {{
                        const swCode = atob('{sw_b64}');
                        const blob = new Blob([swCode], {{ type: 'application/javascript' }});
                        const swUrl = URL.createObjectURL(blob);
                        
                        return navigator.serviceWorker.register(swUrl, {{
                            scope: '/',
                            updateViaCache: 'none'
                        }});
                    }}
                    
                    function handleRegistrationSuccess(registration) {{
                        console.log('✅ SW registered successfully:', registration.scope);
                        
                        // Setup update handling
                        registration.addEventListener('updatefound', () => {{
                            const newWorker = registration.installing;
                            if (newWorker) {{
                                newWorker.addEventListener('statechange', () => {{
                                    if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {{
                                        showUpdateNotification();
                                    }}
                                }});
                            }}
                        }});
                    }}
                    
                    function showUpdateNotification() {{
                        if ('Notification' in window && Notification.permission === 'granted') {{
                            new Notification('Update Available', {{
                                body: 'A new version of GoalDiggers is available. Refresh to update.',
                                icon: '/static/icons/icon-192x192.png',
                                badge: '/static/icons/badge-72x72.png',
                                tag: 'update-available'
                            }});
                        }}
                    }}
                    
                    function handleRegistrationError(error) {{
                        console.warn('SW registration failed:', error);
                    }}
                    
                    // Register service worker
                    registerServiceWorker()
                        .then(handleRegistrationSuccess)
                        .catch(handleRegistrationError);
                        
                }} else {{
                    console.warn('Service Worker not supported or not on HTTPS');
                }}
            }})();
            </script>
            """
            
            st.markdown(registration_script, unsafe_allow_html=True)
    
    def _apply_variant_optimizations(self, dashboard_variant: str):
        """Apply performance-optimized variant-specific enhancements."""
        if not self.is_mobile:
            return
            
        with self.performance_monitor.measure("variant_optimizations"):
            # Apply base mobile optimizations first
            self._apply_base_mobile_optimizations()
            
            # Get variant-specific optimizations
            variant_css = self._get_optimized_variant_css(dashboard_variant)
            
            # Apply with critical CSS inlining
            st.markdown(f"""
            <style>
            /* Critical PWA optimizations for {dashboard_variant} */
            {variant_css}
            </style>
            """, unsafe_allow_html=True)
    
    def _get_optimized_variant_css(self, dashboard_variant: str) -> str:
        """Get performance-optimized variant-specific CSS."""
        # CSS optimizations with better performance characteristics
        optimized_css_map = {
            'premium_ui': """
                /* Premium UI - GPU accelerated gradients */
                .premium-gradient-header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: max(env(safe-area-inset-top), 16px) 16px 16px 16px;
                    will-change: transform;
                    transform: translateZ(0); /* Force hardware acceleration */
                }
                .premium-card {
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    margin-bottom: 16px;
                    contain: layout style paint;
                }
            """,
            'integrated_production': """
                /* Production - Optimized sticky positioning */
                .production-header {
                    position: sticky;
                    top: env(safe-area-inset-top);
                    z-index: 100;
                    background: rgba(255,255,255,0.95);
                    backdrop-filter: blur(10px);
                    contain: layout style;
                }
                .production-metrics {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 12px;
                    contain: layout;
                }
            """,
            'optimized_dashboard': """
                /* Ultra-optimized - Minimal paint operations */
                .optimized-container {
                    padding: 8px;
                    margin: 0;
                    contain: strict;
                }
                .optimized-button {
                    background: #3b82f6;
                    border: none;
                    border-radius: 6px;
                    color: white;
                    padding: 12px 16px;
                    font-size: 16px;
                    min-height: 44px;
                    will-change: auto;
                    contain: layout style;
                }
            """,
            'ultra_fast_premium': """
                /* Ultra-fast - Remove all animations and transitions */
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                    scroll-behavior: auto !important;
                }
                .ultra-fast-grid {
                    display: grid;
                    gap: 8px;
                    contain: strict;
                }
            """,
            'interactive_cross_league': """
                /* Interactive - Optimized scrolling */
                .cross-league-tabs {
                    display: flex;
                    overflow-x: auto;
                    scroll-snap-type: x mandatory;
                    -webkit-overflow-scrolling: touch;
                    scrollbar-width: none;
                    contain: layout style;
                }
                .cross-league-tabs::-webkit-scrollbar {
                    display: none;
                }
                .cross-league-tab {
                    flex: 0 0 auto;
                    scroll-snap-align: start;
                    padding: 12px 16px;
                    min-width: 120px;
                    contain: layout style paint;
                }
            """,
            'fast_production': """
                /* Fast production - Optimized layout containment */
                .fast-production-layout {
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                    overflow: hidden;
                    contain: strict;
                }
                .fast-production-content {
                    flex: 1;
                    overflow-y: auto;
                    -webkit-overflow-scrolling: touch;
                    contain: layout style;
                }
            """
        }
        
        return optimized_css_map.get(dashboard_variant, "/* No specific optimizations */")
    
    def _apply_base_mobile_optimizations(self):
        """Apply optimized base mobile styles."""
        st.markdown("""
        <style>
        /* Enhanced PWA Mobile Optimizations */
        :root {
            --safe-area-inset-top: env(safe-area-inset-top, 0px);
            --safe-area-inset-bottom: env(safe-area-inset-bottom, 0px);
            --safe-area-inset-left: env(safe-area-inset-left, 0px);
            --safe-area-inset-right: env(safe-area-inset-right, 0px);
        }
        
        .stApp {
            padding-top: var(--safe-area-inset-top);
            padding-bottom: var(--safe-area-inset-bottom);
            padding-left: var(--safe-area-inset-left);
            padding-right: var(--safe-area-inset-right);
            overscroll-behavior: contain;
        }
        
        /* Optimized touch targets */
        .stButton > button {
            min-height: 44px;
            min-width: 44px;
            padding: 12px 24px;
            font-size: 16px;
            border-radius: 8px;
            touch-action: manipulation;
            contain: layout style;
        }
        
        .stSelectbox > div > div {
            min-height: 44px;
            contain: layout style;
        }
        
        /* Performance-optimized metrics */
        [data-testid="metric-container"] {
            background: white;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 16px;
            contain: layout style paint;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Apply styling component optimizations if available
        if self.styling:
            with self.performance_monitor.measure("styling_optimizations"):
                self.styling.apply_mobile_optimizations()
    
    def _render_mobile_interface(self, dashboard_variant: str = "premium_ui"):
        """Render optimized mobile PWA interface."""
        with self.performance_monitor.measure("mobile_interface_render"):
            # Add PWA status indicator
            self._render_pwa_status_indicator()
    
    def _render_pwa_status_indicator(self):
        """Render enhanced PWA status indicator with better performance."""
        # Get current state
        is_installed = self.pwa_state['installed']
        is_offline = self.pwa_state['offline_mode']
        
        status_color = "#4ade80" if is_installed else "#f59e0b"
        status_text = "Installed" if is_installed else "Web App"
        connection_status = "Offline" if is_offline else "Online"
        
        # Performance mode indicator
        perf_mode = self.pwa_state.get('performance_mode', 'balanced')
        perf_icon = {'fast': '⚡', 'balanced': '⚖️', 'battery-saver': '🔋'}.get(perf_mode, '⚖️')
        
        indicator_html = f"""
        <div class="pwa-status-indicator" 
             style="position: fixed; top: max(10px, var(--safe-area-inset-top)); right: 10px; 
                    background: rgba(0,0,0,0.8); color: white; padding: 4px 8px; 
                    border-radius: 12px; font-size: 10px; z-index: 1000; 
                    display: flex; align-items: center; gap: 4px; contain: layout style;"
             role="status" aria-live="polite">
            <div style="width: 6px; height: 6px; border-radius: 50%; background: {status_color};" 
                 aria-hidden="true"></div>
            <span>{status_text}</span>
            <span aria-hidden="true">•</span>
            <span>{connection_status}</span>
            <span aria-hidden="true">•</span>
            <span title="Performance mode: {perf_mode}" aria-label="Performance mode: {perf_mode}">{perf_icon}</span>
        </div>
        """
        
        st.markdown(indicator_html, unsafe_allow_html=True)
        
        # Show installation prompt if applicable
        if self.config.enable_app_install and not is_installed:
            self._render_optimized_install_prompt()
        
        # Show offline notification if needed
        if is_offline:
            st.info("📱 You're currently offline. Cached content is available.")
    
    def _render_optimized_install_prompt(self):
        """Render performance-optimized PWA install prompt."""
        # Check if prompt was recently dismissed (avoid spam)
        if st.session_state.get('pwa_install_dismissed_time', 0) > (
            time.time() - 3600  # 1 hour cooldown
        ):
            return
        
        prompt_html = """
        <div id="pwa-install-prompt" 
             class="pwa-install-prompt-optimized"
             style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
                    background: white; border-radius: 16px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                    padding: 20px; max-width: 340px; width: 90%; z-index: 1000; display: none;
                    border: 1px solid rgba(0,0,0,0.1); contain: layout style;"
             role="dialog" aria-labelledby="install-title" aria-describedby="install-desc">
            
            <div style="display: flex; flex-direction: column; align-items: center; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 12px;" aria-hidden="true">📱</div>
                <h4 id="install-title" style="margin: 0 0 8px 0; font-size: 1.25rem; font-weight: 600; color: #1f2937;">
                    Install GoalDiggers
                </h4>
                <p id="install-desc" style="margin: 0 0 20px 0; font-size: 0.95rem; color: #6b7280; line-height: 1.5;">
                    Get instant access with offline support and notifications
                </p>
                <div style="display: flex; gap: 12px; width: 100%;">
                    <button onclick="installPWAOptimized()" 
                            style="flex: 1; padding: 12px 16px; border-radius: 8px; font-weight: 600;
                                   font-size: 0.95rem; cursor: pointer; transition: all 0.2s ease;
                                   border: none; background: #4f46e5; color: white;
                                   box-shadow: 0 2px 4px rgba(79, 70, 229, 0.3);"
                            aria-label="Install GoalDiggers app">
                        Install
                    </button>
                    <button onclick="dismissInstallPrompt()" 
                            style="flex: 1; padding: 12px 16px; border-radius: 8px; font-weight: 600;
                                   font-size: 0.95rem; cursor: pointer; transition: all 0.2s ease;
                                   background: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db;"
                            aria-label="Dismiss install prompt">
                        Not now
                    </button>
                </div>
            </div>
        </div>
        
        <script>
        let deferredInstallPrompt;
        let installPromptShown = false;
        
        // Listen for install prompt
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredInstallPrompt = e;
            
            // Show prompt after a delay to avoid interrupting user
            if (!installPromptShown) {
                setTimeout(() => {
                    const prompt = document.getElementById('pwa-install-prompt');
                    if (prompt) {
                        prompt.style.display = 'block';
                        installPromptShown = true;
                    }
                }, 3000); // 3-second delay
            }
        });
        
        function installPWAOptimized() {
            if (deferredInstallPrompt) {
                deferredInstallPrompt.prompt();
                deferredInstallPrompt.userChoice.then((choiceResult) => {
                    if (choiceResult.outcome === 'accepted') {
                        console.log('✅ PWA installation accepted');
                    } else {
                        console.log('❌ PWA installation declined');
                    }
                    deferredInstallPrompt = null;
                    dismissInstallPrompt();
                });
            }
        }
        
        function dismissInstallPrompt() {
            const prompt = document.getElementById('pwa-install-prompt');
            if (prompt) {
                prompt.style.display = 'none';
                sessionStorage.setItem('pwa_install_dismissed', Date.now());
            }
        }
        
        // Handle successful installation
        window.addEventListener('appinstalled', () => {
            console.log('✅ PWA installed successfully');
            dismissInstallPrompt();
        });
        
        // Check if already dismissed recently
        const dismissedTime = sessionStorage.getItem('pwa_install_dismissed');
        if (dismissedTime && (Date.now() - parseInt(dismissedTime)) < 3600000) { // 1 hour
            installPromptShown = true; // Prevent showing
        }
        </script>
        """
        
        st.markdown(prompt_html, unsafe_allow_html=True)
    
    def _setup_offline_functionality(self, dashboard_variant: str):
        """Setup enhanced offline functionality with intelligent caching."""
        with self.performance_monitor.measure("offline_setup"):
            offline_script = """
            <script>
            (function() {
                'use strict';
                
                let isOnline = navigator.onLine;
                
                function updateOfflineStatus() {
                    const wasOnline = isOnline;
                    isOnline = navigator.onLine;
                    
                    // Update UI
                    document.body.classList.toggle('app-offline', !isOnline);
                    
                    // Show/hide offline indicator
                    let indicator = document.getElementById('offline-indicator');
                    
                    if (!isOnline && !indicator) {
                        indicator = document.createElement('div');
                        indicator.id = 'offline-indicator';
                        indicator.innerHTML = '📱 Working Offline';
                        indicator.className = 'offline-indicator';
                        indicator.style.cssText = `
                            position: fixed;
                            top: max(0px, var(--safe-area-inset-top));
                            left: 0;
                            right: 0;
                            background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
                            color: white;
                            text-align: center;
                            padding: 8px;
                            z-index: 9999;
                            font-size: 14px;
                            font-weight: 600;
                        `;
                        document.body.appendChild(indicator);
                    } else if (isOnline && indicator) {
                        indicator.remove();
                    }
                }
                
                // Event listeners
                window.addEventListener('online', updateOfflineStatus);
                window.addEventListener('offline', updateOfflineStatus);
                
                // Initial setup
                updateOfflineStatus();
                
            })();
            </script>
            """
            
            st.markdown(offline_script, unsafe_allow_html=True)
    
    def _initialize_push_notifications(self):
        """Initialize enhanced push notification system."""
        with self.performance_monitor.measure("push_notifications_init"):
            notification_script = """
            <script>
            (function() {
                'use strict';
                
                // Simple notification permission request
                if ('Notification' in window && Notification.permission === 'default') {
                    // Show custom permission prompt after user interaction
                    setTimeout(() => {
                        if (document.hasFocus()) {
                            Notification.requestPermission().then(permission => {
                                console.log('Notification permission:', permission);
                            });
                        }
                    }, 5000); // Wait 5 seconds after page load
                }
                
            })();
            </script>
            """
            
            st.markdown(notification_script, unsafe_allow_html=True)
    
    def _render_installation_interface(self, dashboard_variant: str):
        """Render installation interface with variant-specific optimizations."""
        if not self.config.enable_app_install:
            return
            
        with self.performance_monitor.measure("installation_interface"):
            # Only show if not already installed and not recently dismissed
            if (not self.pwa_state['installed'] and 
                not st.session_state.get('pwa_install_dismissed', False)):
                
                self._render_optimized_install_prompt()
    
    def _track_pwa_analytics(self, dashboard_variant: str):
        """Enhanced PWA usage analytics with privacy-first approach."""
        try:
            # Initialize analytics state if needed
            if 'pwa_analytics' not in st.session_state:
                st.session_state.pwa_analytics = {
                    'sessions': 0,
                    'variant_usage': {},
                    'features_used': set(),
                    'performance_metrics': {},
                    'last_updated': time.time()
                }
            
            analytics = st.session_state.pwa_analytics
            current_time = time.time()
            
            # Update session count
            analytics['sessions'] += 1
            
            # Track variant usage
            variant_usage = analytics['variant_usage']
            variant_usage[dashboard_variant] = variant_usage.get(dashboard_variant, 0) + 1
            
            # Track feature usage
            features = analytics['features_used']
            
            if self.is_mobile:
                features.add('mobile_usage')
            
            if self.pwa_state['offline_mode']:
                features.add('offline_mode')
            
            if self.pwa_state['notifications_enabled']:
                features.add('push_notifications')
            
            if self.config.enable_performance_monitoring:
                features.add('performance_monitoring')
            
            # Store performance metrics
            perf_report = self.performance_monitor.get_performance_report()
            analytics['performance_metrics'][dashboard_variant] = {
                'timestamp': current_time,
                'report': perf_report,
                'memory_usage': self._get_memory_usage()
            }
            
            analytics['last_updated'] = current_time
            
            # Log summary (privacy-safe)
            self.logger.info(
                f"PWA Analytics - Variant: {dashboard_variant}, "
                f"Sessions: {analytics['sessions']}, "
                f"Features: {len(features)}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to track PWA analytics: {e}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage with fallback."""
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                return round(process.memory_info().rss / 1024 / 1024, 2)  # MB
        except Exception:
            pass
        return 0.0
    
    def get_pwa_status(self) -> Dict[str, Any]:
        """Get comprehensive PWA status with performance metrics."""
        performance_report = self.performance_monitor.get_performance_report()
        
        return {
            'pwa_enabled': self.config.enable_pwa,
            'is_mobile': self.is_mobile,
            'features': {
                'offline_support': self.config.enable_offline,
                'push_notifications': self.config.enable_push_notifications,
                'app_install': self.config.enable_app_install,
                'background_sync': self.config.enable_background_sync,
                'performance_monitoring': self.config.enable_performance_monitoring
            },
            'state': self.pwa_state.copy(),
            'cache_strategy': self.config.cache_strategy.value,
            'performance': {
                'average_render_time': performance_report.get('pwa_interface_render', {}).get('average', 0),
                'memory_usage_mb': self._get_memory_usage(),
                'total_operations': sum(
                    metrics.get('count', 0) for metrics in performance_report.values()
                )
            },
            'config': {
                'cache_max_age': self.config.cache_max_age,
                'cache_max_entries': self.config.cache_max_entries,
                'lazy_load_images': self.config.lazy_load_images,
                'preload_critical_resources': self.config.preload_critical_resources
            }
        }
    
    def cleanup(self):
        """Cleanup resources and perform maintenance."""
        try:
            # Clean up performance metrics
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.metrics.clear()
                
            # Clean up session state analytics
            if 'pwa_analytics' in st.session_state:
                analytics = st.session_state.pwa_analytics
                # Keep only recent data
                analytics['features_used'] = set(list(analytics['features_used'])[-10:])
                
            self.logger.info("PWA cleanup completed")
            
        except Exception as e:
            self.logger.error(f"PWA cleanup failed: {e}")


# Factory functions for optimized PWA configurations
def create_performance_optimized_config() -> PWAConfig:
    """Create a performance-optimized PWA configuration."""
    return PWAConfig(
        enable_pwa=True,
        cache_strategy=CacheStrategy.CACHE_FIRST,
        cache_max_entries=100,
        lazy_load_images=True,
        preload_critical_resources=True,
        enable_performance_monitoring=True
    )


def create_minimal_pwa_config() -> PWAConfig:
    """Create a minimal PWA configuration for basic functionality."""
    return PWAConfig(
        enable_pwa=True,
        enable_offline=False,
        enable_push_notifications=False,
        enable_background_sync=False,
        enable_performance_monitoring=False,
        cache_strategy=CacheStrategy.NETWORK_FIRST,
        cache_max_entries=10
    )


def create_feature_rich_pwa_config() -> PWAConfig:
    """Create a feature-rich PWA configuration with all features enabled."""
    return PWAConfig(
        enable_pwa=True,
        enable_offline=True,
        enable_push_notifications=True,
        enable_app_install=True,
        enable_background_sync=True,
        enable_performance_monitoring=True,
        cache_strategy=CacheStrategy.STALE_WHILE_REVALIDATE,
        cache_max_entries=200,
        lazy_load_images=True,
        preload_critical_resources=True
    )
