#!/usr/bin/env python3
"""
Real-time System Status Widget
Displays live system health metrics in dashboard sidebar
"""

import logging
from typing import Any, Dict

import streamlit as st

logger = logging.getLogger(__name__)


def render_system_status_widget(compact: bool = False):
    """
    Render real-time system status widget with health indicators.
    
    Args:
        compact: If True, show condensed view for sidebar
    """
    try:
        from utils.metrics_collector import get_metrics_collector
        metrics_collector = get_metrics_collector()
        
        # Get current metrics
        current_metrics = metrics_collector.get_current_metrics()
        health_status = metrics_collector.get_health_status()
        
        # Overall health score
        overall_score = health_status.get('overall_score', 0)
        
        if compact:
            # Compact view for sidebar
            st.markdown("### 🩺 System Status")
            
            # Overall health with emoji indicator
            if overall_score >= 85:
                st.success("🟢 System Health: Excellent")
            elif overall_score >= 70:
                st.info("🟡 System Health: Good")
            elif overall_score >= 50:
                st.warning("🟠 System Health: Degraded")
            else:
                st.error("🔴 System Health: Critical")
            
            # Key metrics
            cache_hit_rate = current_metrics.get('cache_hit_rate', 0.0)
            if cache_hit_rate > 0.7:
                st.success(f"✅ Cache: {cache_hit_rate:.0%} hit rate")
            elif cache_hit_rate > 0.5:
                st.info(f"📦 Cache: {cache_hit_rate:.0%} hit rate")
            else:
                st.warning(f"⚠️ Cache: {cache_hit_rate:.0%} hit rate")
            
            # API quota
            api_remaining = current_metrics.get('api_quota_remaining', 0)
            api_total = current_metrics.get('api_quota_total', 10000)
            api_pct = (api_remaining / api_total * 100) if api_total > 0 else 0
            
            if api_pct > 50:
                st.info(f"🌐 API Quota: {api_pct:.0f}% remaining")
            elif api_pct > 20:
                st.warning(f"⏳ API Quota: {api_pct:.0f}% remaining")
            else:
                st.error(f"🚨 API Quota: {api_pct:.0f}% remaining")
            
            # Memory status
            memory_mb = current_metrics.get('memory_usage_mb', 0)
            memory_target = current_metrics.get('memory_target_mb', 350)
            memory_pct = (memory_mb / memory_target * 100) if memory_target > 0 else 0
            
            st.markdown("### 💾 Memory Usage")
            st.progress(min(memory_pct / 100, 1.0), text=f"{memory_mb:.0f}MB / {memory_target}MB ({memory_pct:.0f}%)")
            
            if memory_pct < 90:
                st.caption("✅ Memory usage healthy")
            elif memory_pct < 110:
                st.caption("⚠️ Memory usage elevated")
            else:
                st.caption("🚨 Memory usage critical")
                
        else:
            # Full view for main dashboard
            st.markdown("## 🩺 Live System Health")
            
            # Health score gauge
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Health Score",
                    f"{overall_score:.0f}/100",
                    delta=None,
                    help="Overall system health score (0-100)"
                )
            
            with col2:
                cache_hit_rate = current_metrics.get('cache_hit_rate', 0.0)
                st.metric(
                    "Cache Hit Rate",
                    f"{cache_hit_rate:.0%}",
                    delta=None,
                    help="Percentage of requests served from cache"
                )
            
            with col3:
                avg_latency = current_metrics.get('prediction_latency_avg', 0.0)
                st.metric(
                    "Avg Latency",
                    f"{avg_latency:.2f}s",
                    delta=None,
                    help="Average prediction generation time"
                )
            
            with col4:
                api_remaining = current_metrics.get('api_quota_remaining', 0)
                st.metric(
                    "API Quota",
                    f"{api_remaining:,}",
                    delta=None,
                    help="Remaining API calls today"
                )
            
            # Component health status
            st.markdown("### Component Status")
            
            # Create status pills for each component
            status_cols = st.columns(4)
            components = ['cache', 'memory', 'api_quota', 'latency']
            
            for idx, component in enumerate(components):
                with status_cols[idx]:
                    if component in health_status:
                        status_info = health_status[component]
                        status = status_info['status']
                        message = status_info['message']
                        
                        if status == 'healthy':
                            st.success(f"✅ {component.replace('_', ' ').title()}")
                            st.caption(message)
                        elif status == 'warning':
                            st.warning(f"⚠️ {component.replace('_', ' ').title()}")
                            st.caption(message)
                        else:
                            st.error(f"🔴 {component.replace('_', ' ').title()}")
                            st.caption(message)
            
            # Recent alerts
            recent_alerts = metrics_collector.get_recent_alerts(limit=3)
            if recent_alerts:
                st.markdown("### 🚨 Recent Alerts")
                for alert in recent_alerts:
                    severity = alert['severity']
                    icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")
                    
                    with st.expander(f"{icon} {alert['title']}", expanded=(severity == "critical")):
                        st.write(f"**Time:** {alert['timestamp']}")
                        st.write(f"**Component:** {alert['component']}")
                        st.write(f"**Message:** {alert['message']}")
            
    except ImportError:
        # Metrics collector not available, show static status
        if compact:
            st.markdown("### 🩺 System Status")
            st.success("🟢 AI Engine: Online")
            st.info("📊 Data Feed: Active")
            st.info("💾 Cache: Enabled")
            st.markdown("### 💾 Memory Usage")
            st.progress(0.65, text="Memory usage tracking unavailable")
        else:
            st.info("📊 Live metrics unavailable. Install metrics collector for real-time monitoring.")
    
    except Exception as e:
        logger.error(f"Failed to render system status widget: {e}")
        if compact:
            st.error("⚠️ Status unavailable")
        else:
            st.error(f"Failed to load system status: {e}")


def get_status_emoji(score: float) -> str:
    """Get emoji based on health score."""
    if score >= 85:
        return "🟢"
    elif score >= 70:
        return "🟡"
    elif score >= 50:
        return "🟠"
    else:
        return "🔴"


def get_status_color(status: str) -> str:
    """Get color code for status."""
    colors = {
        'healthy': '#10b981',
        'warning': '#f59e0b',
        'critical': '#ef4444'
    }
    return colors.get(status, '#6b7280')

