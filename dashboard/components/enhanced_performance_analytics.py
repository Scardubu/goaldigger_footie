#!/usr/bin/env python3
"""
Enhanced Performance Analytics Component

Comprehensive performance monitoring dashboard integrating:
- Prediction accuracy tracking with 7-day and 30-day rolling metrics
- Model performance comparison (XGBoost, Ensemble, Real-time)
- Cache analytics and hit rate visualization
- API latency and throughput monitoring
- Real-time system health indicators

Phase 5B Integration: Advanced analytics for production monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def render_enhanced_performance_analytics():
    """
    Render comprehensive performance analytics dashboard.
    
    Integrates multiple data sources:
    - Performance tracker for model metrics
    - Prediction history for accuracy tracking
    - Cache monitor for efficiency metrics
    - Performance instrumentation for latency data
    """
    st.markdown("### 📊 Performance Analytics")
    
    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Accuracy", 
        "🤖 Models", 
        "⚡ Performance", 
        "💾 Cache"
    ])
    
    with tab1:
        _render_accuracy_analytics()
    
    with tab2:
        _render_model_comparison()
    
    with tab3:
        _render_performance_metrics()
    
    with tab4:
        _render_cache_analytics()


def _render_accuracy_analytics():
    """Render prediction accuracy tracking with rolling metrics."""
    try:
        from dashboard.components.prediction_history import prediction_history

        # Get accuracy statistics
        stats = prediction_history.get_accuracy_stats()
        
        if not stats or stats.get('total_predictions', 0) == 0:
            st.info("📈 Accuracy metrics will appear after predictions are made.")
            return
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = stats.get('overall_accuracy', 0.0)
            st.metric(
                "Overall Accuracy",
                f"{accuracy*100:.1f}%",
                delta="🎯 Target: 85%"
            )
        
        with col2:
            completed = stats.get('completed_predictions', 0)
            total = stats.get('total_predictions', 0)
            st.metric(
                "Completed Predictions",
                f"{completed}",
                delta=f"{total-completed} pending"
            )
        
        with col3:
            # Calculate 7-day accuracy
            trend = stats.get('accuracy_trend', [])
            if trend and len(trend) >= 7:
                recent_7d = sum(t.get('mean', 0) for t in trend[-7:]) / 7
                st.metric(
                    "7-Day Accuracy",
                    f"{recent_7d*100:.1f}%",
                    delta=f"{(recent_7d - accuracy)*100:+.1f}%"
                )
            else:
                st.metric("7-Day Accuracy", "N/A", delta="Need 7+ days")
        
        with col4:
            # Calculate 30-day accuracy
            if trend and len(trend) >= 30:
                recent_30d = sum(t.get('mean', 0) for t in trend[-30:]) / 30
                st.metric(
                    "30-Day Accuracy",
                    f"{recent_30d*100:.1f}%",
                    delta=f"{(recent_30d - accuracy)*100:+.1f}%"
                )
            else:
                st.metric("30-Day Accuracy", "N/A", delta="Need 30+ days")
        
        st.markdown("---")
        
        # Accuracy trend chart
        if trend:
            st.markdown("#### 📈 Accuracy Trend (Rolling Average)")
            trend_df = pd.DataFrame(trend)
            
            fig = px.line(
                trend_df,
                x='week',
                y='mean',
                title="Weekly Prediction Accuracy",
                labels={'mean': 'Accuracy', 'week': 'Week'}
            )
            
            fig.add_hline(
                y=0.85, 
                line_dash="dash", 
                line_color="green",
                annotation_text="Target: 85%"
            )
            
            fig.update_traces(line_color='#1e40af', line_width=3)
            fig.update_layout(
                yaxis_tickformat='.0%',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # League-specific accuracy
        league_stats = stats.get('league_accuracy', {})
        if league_stats:
            st.markdown("#### ⚽ Accuracy by League")
            
            league_df = pd.DataFrame([
                {'League': league, 'Accuracy': acc}
                for league, acc in league_stats.items()
            ]).sort_values('Accuracy', ascending=False)
            
            fig = px.bar(
                league_df,
                x='League',
                y='Accuracy',
                title="Prediction Accuracy by League",
                color='Accuracy',
                color_continuous_scale='RdYlGn'
            )
            
            fig.update_layout(
                yaxis_tickformat='.0%',
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        st.warning("⚠️ Prediction history component not available")
    except Exception as e:
        logger.error(f"Error rendering accuracy analytics: {e}")
        st.error(f"Failed to load accuracy analytics: {e}")


def _render_model_comparison():
    """Render model performance comparison dashboard."""
    try:
        from models.performance_tracker import performance_tracker

        # Get available models
        models = performance_tracker._get_available_models()
        
        if not models:
            st.info("📊 Model performance data will appear after predictions are made.")
            return
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=models,
            index=0
        )
        
        # Get model summary
        summary = performance_tracker.get_model_summary(selected_model)
        
        if not summary:
            st.warning(f"No performance data for {selected_model}")
            return
        
        # Model metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Predictions",
                summary.get('total_predictions', 0)
            )
        
        with col2:
            accuracy = summary.get('accuracy', {})
            mean_acc = accuracy.get('mean', 0.0)
            st.metric(
                "Mean Accuracy",
                f"{mean_acc:.1%}",
                delta=f"σ={accuracy.get('std', 0.0):.2%}"
            )
        
        with col3:
            latest_acc = accuracy.get('latest', 0.0)
            st.metric(
                "Latest Accuracy",
                f"{latest_acc:.1%}",
                delta=f"{(latest_acc - mean_acc):+.1%}"
            )
        
        with col4:
            pred_time = summary.get('prediction_time', {})
            st.metric(
                "Avg Pred Time",
                f"{pred_time.get('mean', 0.0):.3f}s"
            )
        
        st.markdown("---")
        
        # Performance trends
        st.markdown("#### 📈 Model Performance Over Time")
        
        history = performance_tracker.get_performance_history(
            model_id=selected_model,
            start_date=datetime.now() - timedelta(days=30)
        )
        
        if not history.empty:
            # Create subplots for different metrics
            metrics_to_plot = ['accuracy', 'log_loss', 'f1_score']
            available_metrics = [m for m in metrics_to_plot if m in history['metric_name'].values]
            
            if available_metrics:
                fig = make_subplots(
                    rows=len(available_metrics),
                    cols=1,
                    subplot_titles=[m.replace('_', ' ').title() for m in available_metrics],
                    vertical_spacing=0.1
                )
                
                for i, metric in enumerate(available_metrics, 1):
                    metric_df = history[history['metric_name'] == metric]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=metric_df['timestamp'],
                            y=metric_df['metric_value'],
                            mode='lines+markers',
                            name=metric.title(),
                            line=dict(width=2)
                        ),
                        row=i, col=1
                    )
                
                fig.update_layout(
                    height=200 * len(available_metrics),
                    showlegend=False,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical performance data available yet.")
        
        # Drift detection
        st.markdown("#### 🔍 Performance Drift Detection")
        
        drift = performance_tracker.detect_performance_drift(
            model_id=selected_model,
            metric_name='accuracy'
        )
        
        if drift and drift.get('drift_detected'):
            st.warning(
                f"⚠️ Performance drift detected! "
                f"Current: {drift.get('recent_mean', 0):.1%} | "
                f"Baseline: {drift.get('baseline_mean', 0):.1%} | "
                f"Change: {drift.get('drift_magnitude', 0):+.1%}"
            )
        else:
            st.success("✅ Model performance is stable")
        
    except ImportError:
        st.warning("⚠️ Performance tracker not available")
    except Exception as e:
        logger.error(f"Error rendering model comparison: {e}")
        st.error(f"Failed to load model comparison: {e}")


def _render_performance_metrics():
    """Render system performance metrics and latency monitoring."""
    try:
        from dashboard.components.performance_panel import (
            _INGESTION_LATENCY_SAMPLES,
            _LATENCY_SAMPLES,
            _REAL_DATA_PREDICTIONS,
            _TOTAL_PREDICTIONS,
        )
        
        st.markdown("#### ⚡ System Performance")
        
        # Real-time metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if _LATENCY_SAMPLES:
                samples = list(_LATENCY_SAMPLES)
                latest = samples[-1]
                avg = sum(samples) / len(samples)
                st.metric(
                    "Prediction Latency",
                    f"{latest:.3f}s",
                    delta=f"Avg: {avg:.3f}s"
                )
            else:
                st.metric("Prediction Latency", "N/A")
        
        with col2:
            if _INGESTION_LATENCY_SAMPLES:
                ing_samples = list(_INGESTION_LATENCY_SAMPLES)
                ing_latest = ing_samples[-1]
                ing_avg = sum(ing_samples) / len(ing_samples)
                st.metric(
                    "Data Ingestion",
                    f"{ing_latest:.3f}s",
                    delta=f"Avg: {ing_avg:.3f}s"
                )
            else:
                st.metric("Data Ingestion", "N/A")
        
        with col3:
            if _TOTAL_PREDICTIONS > 0:
                real_ratio = (_REAL_DATA_PREDICTIONS / _TOTAL_PREDICTIONS) * 100
                st.metric(
                    "Real Data Usage",
                    f"{real_ratio:.1f}%",
                    delta=f"{_REAL_DATA_PREDICTIONS}/{_TOTAL_PREDICTIONS}"
                )
            else:
                st.metric("Real Data Usage", "N/A")
        
        st.markdown("---")
        
        # Latency trend visualization
        if _LATENCY_SAMPLES:
            st.markdown("#### 📊 Latency Trends")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Prediction Latency (s)**")
                latency_df = pd.DataFrame({
                    'Sample': range(len(_LATENCY_SAMPLES)),
                    'Latency': list(_LATENCY_SAMPLES)
                })
                st.line_chart(latency_df.set_index('Sample'))
            
            with col_b:
                if _INGESTION_LATENCY_SAMPLES:
                    st.markdown("**Ingestion Latency (s)**")
                    ing_df = pd.DataFrame({
                        'Sample': range(len(_INGESTION_LATENCY_SAMPLES)),
                        'Latency': list(_INGESTION_LATENCY_SAMPLES)
                    })
                    st.line_chart(ing_df.set_index('Sample'))
        
        # Performance instrumentation
        try:
            from utils import performance_instrumentation as pi
            
            st.markdown("#### 🔧 Performance Instrumentation")
            
            snap = pi.snapshot()
            perf_data = snap.get('performance', {})
            
            if perf_data:
                # Convert to DataFrame
                perf_records = []
                for name, data in perf_data.items():
                    perf_records.append({
                        'Component': name.split('.')[-1],
                        'Avg (ms)': data.get('avg_ms', 0),
                        'Min (ms)': data.get('min_ms', 0),
                        'Max (ms)': data.get('max_ms', 0),
                        'Calls': data.get('count', 0)
                    })
                
                perf_df = pd.DataFrame(perf_records)
                st.dataframe(perf_df, use_container_width=True)
            else:
                st.info("No instrumentation data available yet")
        
        except ImportError:
            pass
        
    except ImportError:
        st.warning("⚠️ Performance panel metrics not available")
    except Exception as e:
        logger.error(f"Error rendering performance metrics: {e}")
        st.error(f"Failed to load performance metrics: {e}")


def _render_cache_analytics():
    """Render cache analytics and efficiency metrics."""
    try:
        from cached_data_utilities import cache_monitor
        
        if not cache_monitor:
            st.info("💾 Cache monitoring not available")
            return
        
        st.markdown("#### 💾 Cache Performance")
        
        # Get cache statistics with error handling
        try:
            stats = cache_monitor.get_stats_summary()
        except Exception as stats_error:
            logger.debug(f"Cache stats not available: {stats_error}")
            st.info("💾 Cache statistics initializing...")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Entries",
                stats.get('total_entries', 0)
            )
        
        with col2:
            hit_rate = stats.get('hit_rate', 0.0)
            target_hit_rate = 0.80
            delta_color = "normal" if hit_rate >= target_hit_rate else "inverse"
            st.metric(
                "Hit Rate",
                f"{hit_rate*100:.1f}%",
                delta=f"Target: {target_hit_rate*100:.0f}%"
            )
        
        with col3:
            cache_size = stats.get('approx_cache_size_mb', 0)
            st.metric(
                "Cache Size",
                f"{cache_size:.1f} MB"
            )
        
        with col4:
            evictions = stats.get('evictions', 0)
            st.metric(
                "Evictions",
                evictions
            )
        
        st.markdown("---")
        
        # Cache breakdown by type
        st.markdown("#### 📦 Cache Breakdown")
        
        cache_types = stats.get('cache_breakdown', {})
        if cache_types:
            breakdown_df = pd.DataFrame([
                {
                    'Cache Type': cache_type,
                    'Entries': data.get('entries', 0),
                    'Hit Rate': data.get('hit_rate', 0.0),
                    'Size (MB)': data.get('size_mb', 0.0)
                }
                for cache_type, data in cache_types.items()
            ])
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Entries by cache type
                fig = px.pie(
                    breakdown_df,
                    values='Entries',
                    names='Cache Type',
                    title='Cache Entries Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_b:
                # Hit rates by cache type
                fig = px.bar(
                    breakdown_df,
                    x='Cache Type',
                    y='Hit Rate',
                    title='Hit Rate by Cache Type',
                    color='Hit Rate',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(yaxis_tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.dataframe(breakdown_df, use_container_width=True)
        
        # Cache recommendations
        st.markdown("#### 💡 Optimization Recommendations")
        
        if hit_rate < 0.60:
            st.warning(
                "⚠️ Cache hit rate is below 60%. Consider:\n"
                "- Increasing cache size\n"
                "- Adjusting TTL values\n"
                "- Reviewing cache key strategies"
            )
        elif hit_rate < target_hit_rate:
            st.info(
                "ℹ️ Cache hit rate is good but can be improved. Consider:\n"
                "- Fine-tuning TTL values\n"
                "- Preloading frequently accessed data"
            )
        else:
            st.success("✅ Cache performance is optimal!")
        
    except ImportError:
        st.warning("⚠️ Cache monitor not available")
    except Exception as e:
        logger.error(f"Error rendering cache analytics: {e}")
        st.error(f"Failed to load cache analytics: {e}")


def get_performance_summary() -> Dict[str, Any]:
    """
    Get aggregated performance summary for quick dashboard display.
    
    Returns:
        Dictionary with key performance indicators
    """
    summary = {
        'accuracy': {
            'overall': 0.0,
            '7_day': 0.0,
            '30_day': 0.0,
            'trend': 'stable'
        },
        'performance': {
            'avg_latency_ms': 0.0,
            'real_data_ratio': 0.0,
            'predictions_today': 0
        },
        'cache': {
            'hit_rate': 0.0,
            'entries': 0,
            'size_mb': 0.0
        },
        'models': {
            'total': 0,
            'best_accuracy': 0.0,
            'drift_detected': False
        }
    }
    
    try:
        # Aggregate accuracy metrics
        from dashboard.components.prediction_history import prediction_history
        stats = prediction_history.get_accuracy_stats()
        if stats:
            summary['accuracy']['overall'] = stats.get('overall_accuracy', 0.0)
            trend = stats.get('accuracy_trend', [])
            if trend and len(trend) >= 7:
                summary['accuracy']['7_day'] = sum(t.get('mean', 0) for t in trend[-7:]) / 7
            if trend and len(trend) >= 30:
                summary['accuracy']['30_day'] = sum(t.get('mean', 0) for t in trend[-30:]) / 30
    except Exception as e:
        logger.debug(f"Could not load accuracy metrics: {e}")
    
    try:
        # Aggregate cache metrics
        from cached_data_utilities import cache_monitor
        if cache_monitor:
            cache_stats = cache_monitor.get_stats_summary()
            summary['cache']['hit_rate'] = cache_stats.get('hit_rate', 0.0)
            summary['cache']['entries'] = cache_stats.get('total_entries', 0)
            summary['cache']['size_mb'] = cache_stats.get('approx_cache_size_mb', 0.0)
    except Exception as e:
        logger.debug(f"Could not load cache metrics: {e}")
    
    try:
        # Aggregate model metrics
        from models.performance_tracker import performance_tracker
        models = performance_tracker._get_available_models()
        summary['models']['total'] = len(models)
        
        # Find best performing model
        best_acc = 0.0
        for model_id in models:
            model_summary = performance_tracker.get_model_summary(model_id)
            if model_summary:
                acc = model_summary.get('accuracy', {}).get('mean', 0.0)
                if acc > best_acc:
                    best_acc = acc
        summary['models']['best_accuracy'] = best_acc
    except Exception as e:
        logger.debug(f"Could not load model metrics: {e}")
    
    return summary
