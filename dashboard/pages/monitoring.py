#!/usr/bin/env python3
"""
Production Monitoring Dashboard
Real-time metrics and system health monitoring for GoalDiggers Platform
"""

import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.metrics_collector import SystemMetricsCollector

st.set_page_config(
    page_title="Production Monitoring | GoalDiggers",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize metrics collector
@st.cache_resource
def get_metrics_collector():
    return SystemMetricsCollector()

metrics_collector = get_metrics_collector()

# Custom CSS for monitoring dashboard
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1em;
        opacity: 0.9;
    }
    .metric-delta {
        font-size: 0.9em;
        margin-top: 5px;
    }
    .status-healthy { color: #10b981; }
    .status-warning { color: #f59e0b; }
    .status-critical { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 Production Monitoring Dashboard")
st.markdown("Real-time system health and performance metrics")

# Auto-refresh
refresh_rate = st.sidebar.selectbox(
    "Auto-refresh interval",
    options=[5, 10, 30, 60, 300],
    index=1,
    format_func=lambda x: f"{x}s" if x < 60 else f"{x//60}min"
)

# Add refresh button
if st.sidebar.button("🔄 Refresh Now"):
    st.rerun()

# Time range selector
time_range = st.sidebar.selectbox(
    "Time Range",
    options=["1h", "6h", "24h", "7d", "30d"],
    index=2
)

# Get current metrics
current_metrics = metrics_collector.get_current_metrics()
historical_data = metrics_collector.get_historical_metrics(time_range)

# === KEY METRICS ROW ===
st.subheader("🎯 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    cache_hit_rate = current_metrics.get('cache_hit_rate', 0.0)
    cache_delta = current_metrics.get('cache_hit_rate_delta', 0.0)
    st.metric(
        "Cache Hit Rate",
        f"{cache_hit_rate:.1%}",
        delta=f"{cache_delta:+.1%}",
        help="Percentage of requests served from cache"
    )

with col2:
    avg_latency = current_metrics.get('prediction_latency_avg', 0.0)
    latency_delta = current_metrics.get('prediction_latency_delta', 0.0)
    st.metric(
        "Avg Prediction Time",
        f"{avg_latency:.2f}s",
        delta=f"{latency_delta:+.2f}s" if latency_delta != 0 else None,
        delta_color="inverse",
        help="Average time to generate a prediction"
    )

with col3:
    api_remaining = current_metrics.get('api_quota_remaining', 0)
    api_total = current_metrics.get('api_quota_total', 10000)
    api_pct = (api_remaining / api_total * 100) if api_total > 0 else 0
    st.metric(
        "API Quota Left",
        f"{api_remaining:,}",
        delta=f"{api_pct:.0f}%",
        help=f"Remaining API calls (out of {api_total:,})"
    )

with col4:
    memory_mb = current_metrics.get('memory_usage_mb', 0)
    memory_target = current_metrics.get('memory_target_mb', 350)
    memory_pct = (memory_mb / memory_target * 100) if memory_target > 0 else 0
    memory_status = "normal" if memory_pct < 100 else "inverse"
    st.metric(
        "Memory Usage",
        f"{memory_mb:.0f} MB",
        delta=f"{memory_pct:.0f}% of target",
        delta_color=memory_status,
        help=f"Current memory usage (target: {memory_target} MB)"
    )

# === SYSTEM HEALTH STATUS ===
st.subheader("🏥 System Health Status")

health_status = metrics_collector.get_health_status()

col1, col2 = st.columns([2, 1])

with col1:
    # Health indicators
    health_items = []
    for component, status in health_status.items():
        if status['status'] == 'healthy':
            icon = "✅"
            color = "status-healthy"
        elif status['status'] == 'warning':
            icon = "⚠️"
            color = "status-warning"
        else:
            icon = "❌"
            color = "status-critical"
        
        health_items.append(f"{icon} <span class='{color}'>{component.title()}: {status['message']}</span>")
    
    st.markdown("<br>".join(health_items), unsafe_allow_html=True)

with col2:
    # Overall health score
    health_score = health_status.get('overall_score', 0)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)

# === PERFORMANCE CHARTS ===
st.subheader("📈 Performance Trends")

tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction Latency",
    "Cache Performance", 
    "API Usage",
    "Memory & Resources"
])

with tab1:
    # Prediction latency over time
    if historical_data.get('latency_history'):
        df_latency = pd.DataFrame(historical_data['latency_history'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_latency['timestamp'],
            y=df_latency['p50'],
            name='P50 (Median)',
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_latency['timestamp'],
            y=df_latency['p95'],
            name='P95',
            mode='lines',
            line=dict(color='orange', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_latency['timestamp'],
            y=df_latency['p99'],
            name='P99',
            mode='lines',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Prediction Latency Percentiles",
            xaxis_title="Time",
            yaxis_title="Latency (seconds)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Latency distribution
        col1, col2 = st.columns(2)
        with col1:
            st.metric("P50 (Median)", f"{df_latency['p50'].iloc[-1]:.3f}s")
            st.metric("P95", f"{df_latency['p95'].iloc[-1]:.3f}s")
        with col2:
            st.metric("P99", f"{df_latency['p99'].iloc[-1]:.3f}s")
            st.metric("Max", f"{df_latency['max'].iloc[-1]:.3f}s")
    else:
        st.info("No latency data available yet. Generate some predictions to see metrics.")

with tab2:
    # Cache performance
    if historical_data.get('cache_stats'):
        df_cache = pd.DataFrame(historical_data['cache_stats'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cache hit rate over time
            fig = px.line(
                df_cache,
                x='timestamp',
                y=['l1_hit_rate', 'l2_hit_rate', 'l3_hit_rate'],
                title="Cache Hit Rates by Layer",
                labels={'value': 'Hit Rate (%)', 'variable': 'Cache Layer'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cache operations
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_cache['timestamp'],
                y=df_cache['hits'],
                name='Hits',
                marker_color='green'
            ))
            fig.add_trace(go.Bar(
                x=df_cache['timestamp'],
                y=df_cache['misses'],
                name='Misses',
                marker_color='red'
            ))
            fig.update_layout(
                title="Cache Hits vs Misses",
                xaxis_title="Time",
                yaxis_title="Count",
                barmode='group',
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cache statistics summary
        latest_cache = df_cache.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Hits", f"{latest_cache['hits']:,.0f}")
        col2.metric("Total Misses", f"{latest_cache['misses']:,.0f}")
        col3.metric("Hit Rate", f"{latest_cache['hit_rate']:.1%}")
        col4.metric("Cache Size", f"{latest_cache['size_mb']:.1f} MB")
    else:
        st.info("No cache data available yet.")

with tab3:
    # API usage tracking
    if historical_data.get('api_usage'):
        df_api = pd.DataFrame(historical_data['api_usage'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # API calls over time
            fig = px.area(
                df_api,
                x='timestamp',
                y='calls_per_minute',
                title="API Calls Per Minute",
                labels={'calls_per_minute': 'Calls/min'}
            )
            fig.update_traces(fill='tozeroy')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # API quota remaining
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_api['timestamp'],
                y=df_api['quota_remaining'],
                mode='lines+markers',
                name='Quota Remaining',
                line=dict(color='blue', width=3),
                fill='tozeroy'
            ))
            fig.update_layout(
                title="API Quota Remaining",
                xaxis_title="Time",
                yaxis_title="Calls Remaining",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # API statistics
        latest_api = df_api.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Calls Today", f"{latest_api['calls_today']:,.0f}")
        col2.metric("Rate Limit Hits", f"{latest_api['rate_limit_hits']:.0f}")
        col3.metric("Avg Response Time", f"{latest_api['avg_response_time']:.2f}s")
        col4.metric("Error Rate", f"{latest_api['error_rate']:.2%}")
    else:
        st.info("No API usage data available yet.")

with tab4:
    # Memory and resource usage
    if historical_data.get('resource_usage'):
        df_resources = pd.DataFrame(historical_data['resource_usage'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Memory usage over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_resources['timestamp'],
                y=df_resources['memory_mb'],
                mode='lines',
                name='Memory Usage',
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ))
            fig.add_hline(
                y=350,
                line_dash="dash",
                line_color="red",
                annotation_text="Target: 350 MB"
            )
            fig.update_layout(
                title="Memory Usage Over Time",
                xaxis_title="Time",
                yaxis_title="Memory (MB)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Active sessions and throughput
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_resources['timestamp'],
                y=df_resources['active_sessions'],
                mode='lines+markers',
                name='Active Sessions',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                title="Active User Sessions",
                xaxis_title="Time",
                yaxis_title="Sessions",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Resource statistics
        latest_resources = df_resources.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CPU Usage", f"{latest_resources['cpu_percent']:.1f}%")
        col2.metric("Active Sessions", f"{latest_resources['active_sessions']:.0f}")
        col3.metric("Requests/min", f"{latest_resources['requests_per_minute']:.0f}")
        col4.metric("Avg Load Time", f"{latest_resources['avg_load_time']:.2f}s")
    else:
        st.info("No resource usage data available yet.")

# === DATA QUALITY MONITORING ===
st.subheader("🎯 Prediction Quality Metrics")

quality_metrics = metrics_collector.get_quality_metrics()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Avg Data Quality Score",
        f"{quality_metrics.get('avg_quality_score', 0):.2f}",
        help="Average quality score of prediction data (0.0-1.0)"
    )
    
    # Quality distribution
    if quality_metrics.get('quality_distribution'):
        fig = go.Figure(data=[go.Pie(
            labels=['Excellent (>0.85)', 'Good (0.70-0.85)', 'Fair (<0.70)'],
            values=[
                quality_metrics['quality_distribution'].get('excellent', 0),
                quality_metrics['quality_distribution'].get('good', 0),
                quality_metrics['quality_distribution'].get('fair', 0)
            ],
            marker=dict(colors=['#10b981', '#f59e0b', '#ef4444'])
        )])
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.metric(
        "Feature Completeness",
        f"{quality_metrics.get('feature_completeness', 0):.1%}",
        help="Percentage of features successfully populated"
    )
    
    # Feature availability
    if quality_metrics.get('feature_availability'):
        df_features = pd.DataFrame([
            {'Feature': k, 'Available': v}
            for k, v in quality_metrics['feature_availability'].items()
        ])
        fig = px.bar(
            df_features,
            x='Feature',
            y='Available',
            title="Feature Availability",
            labels={'Available': 'Availability (%)'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.metric(
        "Prediction Confidence",
        f"{quality_metrics.get('avg_confidence', 0):.1%}",
        help="Average confidence level of predictions"
    )
    
    # Historical accuracy (if available)
    if quality_metrics.get('historical_accuracy'):
        accuracy = quality_metrics['historical_accuracy']
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=accuracy * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Accuracy"},
            delta={'reference': 65},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# === ALERTS & NOTIFICATIONS ===
st.subheader("🚨 Recent Alerts")

alerts = metrics_collector.get_recent_alerts(limit=10)

if alerts:
    for alert in alerts:
        severity = alert['severity']
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")
        
        with st.expander(f"{icon} {alert['title']} - {alert['timestamp']}", expanded=(severity == "critical")):
            st.write(f"**Severity:** {severity.upper()}")
            st.write(f"**Component:** {alert['component']}")
            st.write(f"**Message:** {alert['message']}")
            if alert.get('details'):
                st.json(alert['details'])
else:
    st.success("✅ No alerts - System running smoothly!")

# === FOOTER ===
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col2:
    st.caption(f"Monitoring window: {time_range}")
with col3:
    st.caption("GoalDiggers v1.1.0")

# Auto-refresh
if refresh_rate:
    st.markdown(f"""
    <script>
        setTimeout(function(){{
            window.location.reload();
        }}, {refresh_rate * 1000});
    </script>
    """, unsafe_allow_html=True)
