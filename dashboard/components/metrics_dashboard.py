#!/usr/bin/env python3
"""
Metrics Dashboard Component for GoalDiggers Platform.
Shows system metrics, user activity, and performance data in a structured dashboard.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from config.settings import settings
from metrics.recorder import record_counter, record_gauge, timing


def get_latest_metrics(days_back: int = 0) -> Optional[Dict]:
    """
    Get the metrics data for a specific day.
    
    Args:
        days_back: Number of days back from today (0=today, 1=yesterday, etc.)
        
    Returns:
        Dict containing metrics data or None if not available
    """
    target_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
    metrics_dir = os.path.join(settings.METRICS_DIR, "summaries")
    summary_file = os.path.join(metrics_dir, f"summary_{target_date}.json")
    
    try:
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
    
    return None


def show_metrics_dashboard():
    """Show comprehensive metrics dashboard with interactive elements."""
    with timing("ui.render.metrics_dashboard_ms"):
        st.markdown("""
        <div class='goaldiggers-header gd-fade-in'>
            <h1>📊 System Metrics Dashboard</h1>
            <p>Performance metrics and monitoring for the GoalDiggers Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Date selection for metrics
        col1, col2 = st.columns([3, 1])
        with col1:
            days_back_options = {
                "Today": 0,
                "Yesterday": 1,
                "2 days ago": 2,
                "3 days ago": 3
            }
            days_label = st.selectbox("Select date", list(days_back_options.keys()), index=1)
            days_back = days_back_options[days_label]
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Get metrics for selected date
        metrics_data = get_latest_metrics(days_back)
        
        if metrics_data:
            # Record dashboard view
            record_counter("metrics.dashboard.view", 1, {"days_back": days_back})
            
            # Tabs for different metric categories
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 System Performance", 
                "👥 User Activity",
                "⏱️ Response Times",
                "📉 Raw Data"
            ])
            
            with tab1:
                st.subheader("System Resource Metrics")
                
                # System metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "CPU Usage", 
                        f"{metrics_data['gauges_latest'].get('system.cpu.usage_percent', 0):.1f}%",
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Memory Usage", 
                        f"{metrics_data['gauges_latest'].get('system.memory.usage_mb', 0):.1f} MB",
                        delta=None
                    )
                with col3:
                    st.metric(
                        "Ingestion Success", 
                        f"{metrics_data['gauges_latest'].get('ingestion.run.success_ratio', 0)*100:.1f}%",
                        delta=None
                    )
                
                st.subheader("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Prediction Freshness", 
                        f"{metrics_data['gauges_latest'].get('prediction.freshness_minutes', 0):.1f} min",
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Model Calibration Error", 
                        f"{metrics_data['gauges_latest'].get('model.calibration.error', 0):.4f}",
                        delta=None
                    )
            
            with tab2:
                st.subheader("User Activity Metrics")
                
                # User activity metrics in cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "User Logins", 
                        f"{int(metrics_data['counters'].get('user.login.count', 0))}",
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Predictions Submitted", 
                        f"{int(metrics_data['counters'].get('user.prediction.submitted', 0))}",
                        delta=None
                    )
                with col3:
                    st.metric(
                        "API Requests", 
                        f"{int(metrics_data['counters'].get('api.requests.total', 0))}",
                        delta=None
                    )
                    
                # Ingestion metrics
                st.subheader("Data Processing")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Successful Ingestion Steps", 
                        f"{int(metrics_data['counters'].get('ingestion.steps.success', 0))}",
                        delta=None
                    )
                with col2:
                    st.metric(
                        "Failed Ingestion Steps", 
                        f"{int(metrics_data['counters'].get('ingestion.steps.failure', 0))}",
                        delta=None
                    )
                    
                # Prediction metrics
                st.subheader("Match Predictions")
                st.metric(
                    "Total Match Predictions", 
                    f"{int(metrics_data['counters'].get('prediction.match.count', 0))}",
                    delta=None
                )
                
            with tab3:
                st.subheader("Performance Timings")
                
                # Performance timings
                timing_data = metrics_data['timings']
                
                for metric_name, metric_values in timing_data.items():
                    friendly_names = {
                        "api.response_time_ms": "API Response Time",
                        "prediction.latency_ms": "Prediction Latency",
                        "database.query_time_ms": "Database Query Time",
                        "ingestion.step_duration_ms": "Ingestion Step Duration"
                    }
                    
                    friendly_name = friendly_names.get(metric_name, metric_name)
                    
                    st.markdown(f"##### {friendly_name}")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Average", f"{metric_values['avg']:.1f} ms")
                    with col2:
                        st.metric("95th Percentile", f"{metric_values['p95']:.1f} ms")
                    with col3:
                        st.metric("99th Percentile", f"{metric_values['p99']:.1f} ms")
                    with col4:
                        st.metric("Count", f"{metric_values['count']}")
                    
                    st.markdown("---")
            
            with tab4:
                st.subheader("Raw Metrics Data")
                
                # Display meta information
                st.info(f"Metrics generated at: {metrics_data.get('generated_at')}")
                
                # Raw metrics data in expandable sections
                with st.expander("Counters"):
                    st.json(metrics_data.get('counters', {}))
                
                with st.expander("Gauges (Latest Values)"):
                    st.json(metrics_data.get('gauges_latest', {}))
                
                with st.expander("Timings"):
                    st.json(metrics_data.get('timings', {}))
                
                # Download button for raw data
                st.download_button(
                    "Download Raw Metrics JSON",
                    data=json.dumps(metrics_data, indent=2),
                    file_name=f"metrics_export_{metrics_data.get('day')}.json",
                    mime="application/json"
                )
        else:
            # Show message if no data found
            target_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            st.warning(f"⚠️ No metrics data found for {target_date}.")
            
            # Generate sample metrics option
            with st.expander("Generate sample metrics"):
                st.write("Generate test metrics data for the selected date")
                num_records = st.slider("Number of records", 100, 1000, 500)
                
                if st.button("Generate Test Metrics"):
                    with st.spinner("Generating sample metrics..."):
                        from generate_test_metrics import generate_test_metrics
                        generate_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
                        generate_test_metrics(generate_date, num_records)
                        
                        # Run aggregation
                        from metrics.aggregate_metrics import aggregate
                        aggregate(generate_date)
                        
                        st.success(f"✅ Sample metrics generated and aggregated for {target_date}!")
                        st.info("Click 'Refresh' to view the generated metrics.")


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="GoalDiggers Metrics", layout="wide")
    show_metrics_dashboard()