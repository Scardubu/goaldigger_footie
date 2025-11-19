"""
Dashboard Performance Monitoring Module

This module provides tools for tracking and visualizing performance metrics
for the GoalDiggers dashboard, including:

1. Query execution times
2. Cache hit rates
3. Memory usage
4. Component rendering times
5. Data processing benchmarks
"""

import functools
import logging
import statistics
import time
import tracemalloc
from contextlib import contextmanager  # Added for PerformanceTracker
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.error_log import ErrorLog
from dashboard.optimizations.caching import get_cache_stats

logger = logging.getLogger(__name__) # Module-level logger for existing global functions
# error_log = ErrorLog(component_name="performance_monitor") # Module-level error_log for existing global functions

class PerformanceTracker:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PerformanceTracker")
        self.error_log = ErrorLog(component_name="PerformanceTracker_class_instance") # Instance-specific error log
        self.metrics = {
            "query_times": [],
            "render_times": {},
            "memory_snapshots": [],
            "api_response_times": [],
            "component_render_times": {},
            "data_processing_times": {}
        }
        self._tracemalloc_started = False

    @contextmanager
    def track_component(self, component_name: str):
        """
        Context manager to signify tracking for a component's execution.
        Currently, this is mainly a structural element for IntegrationManager's usage pattern.
        Actual timing and recording are handled by record_component_render.
        """
        # self.logger.debug(f"Entering tracking context for component: {component_name}")
        try:
            yield
        finally:
            # self.logger.debug(f"Exiting tracking context for component: {component_name}")
            pass

    def record_component_render(self, component_name: str, duration_ms: float, func_qualname: str = "N/A"):
        """
        Records the rendering time for a specific component.
        Args:
            component_name: Name of the component.
            duration_ms: Duration of the render in milliseconds.
            func_qualname: The qualified name of the function/method being tracked.
        """
        execution_time_seconds = duration_ms / 1000.0

        if component_name not in self.metrics["component_render_times"]:
            self.metrics["component_render_times"][component_name] = []
        
        self.metrics["component_render_times"][component_name].append({
            "function": func_qualname,
            "execution_time": execution_time_seconds,
            "timestamp": datetime.now().isoformat()
        })
        
        # Log slow component renders (e.g., > 200ms)
        if execution_time_seconds > 0.2:
            self.logger.warning(
                f"Slow component render: {component_name} (invoking {func_qualname}) "
                f"took {execution_time_seconds:.2f}s ({duration_ms:.2f}ms)"
            )

    # Placeholder for other methods that might be moved into this class later
    def start_memory_tracking(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self.logger.info("PerformanceTracker started memory tracking")
            self._tracemalloc_started = True
        else:
            self.logger.info("Memory tracking already started")
            self._tracemalloc_started = True

    def get_memory_snapshot(self):
        if not self._tracemalloc_started or not tracemalloc.is_tracing():
            self.start_memory_tracking()
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        memory_data = {
            "timestamp": datetime.now().isoformat(),
            "total_allocated": sum(stat.size for stat in top_stats),
            "top_allocations": [
                {
                    "filename": stat.traceback[0].filename,
                    "lineno": stat.traceback[0].lineno,
                    "size": stat.size,
                    "count": stat.count
                }
                for stat in top_stats[:10]
            ]
        }
        
        self.metrics["memory_snapshots"].append(memory_data)
        return memory_data

# Global performance tracker instance (consolidated approach)
_global_performance_tracker = None

def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance."""
    global _global_performance_tracker
    if _global_performance_tracker is None:
        _global_performance_tracker = PerformanceTracker()
    return _global_performance_tracker

# Legacy functions for backward compatibility - delegate to PerformanceTracker
def start_memory_tracking():
    """Start tracking memory usage - DEPRECATED: Use PerformanceTracker."""
    get_performance_tracker().start_memory_tracking()

def get_memory_snapshot():
    """Get current memory usage - DEPRECATED: Use PerformanceTracker."""
    return get_performance_tracker().get_memory_snapshot()

# Consolidated timing decorators using PerformanceTracker
def track_execution_time(category: str):
    """
    Decorator to track execution time of functions.
    CONSOLIDATED: Now uses global PerformanceTracker instance.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            # Use PerformanceTracker for consistent tracking
            tracker.record_component_render(category, execution_time * 1000, func.__qualname__)

            return result
        return wrapper
    return decorator

# Track database query time specifically
def track_query_time(func):
    """Decorator to track database query execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # First argument is typically 'self', second is query
        query = args[1] if len(args) > 1 else "Unknown query"
        # Truncate long queries for logging
        query_summary = query[:100] + "..." if len(query) > 100 else query
        
        _performance_metrics["query_times"].append({
            "query": query_summary,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        })
        
        # Log slow queries
        if execution_time > 0.5:  # Log queries taking more than 0.5 seconds
            logger.warning(f"Slow query: {query_summary} took {execution_time:.2f}s")
        
        return result
    return wrapper

# Track component render time
def track_component_render(component_name: str):
    """
    Decorator to track rendering time of dashboard components.
    
    Args:
        component_name: Name of the component being rendered
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            render_time = time.time() - start_time
            
            # Initialize component render times if needed
            if "component_render_times" not in _performance_metrics:
                _performance_metrics["component_render_times"] = {}
            
            if component_name not in _performance_metrics["component_render_times"]:
                _performance_metrics["component_render_times"][component_name] = []
            
            _performance_metrics["component_render_times"][component_name].append({
                "render_time": render_time,
                "timestamp": datetime.now().isoformat()
            })
            
            # Log slow renders
            if render_time > 0.5:  # Log components taking more than 0.5 seconds
                logger.warning(f"Slow component render: {component_name} took {render_time:.2f}s")
            
            return result
        return wrapper
    return decorator

# Consolidated performance statistics
def get_performance_stats():
    """Get performance statistics across all tracked metrics - CONSOLIDATED."""
    tracker = get_performance_tracker()

    # Use consolidated tracker data
    stats = {
        "query_performance": _analyze_query_performance_consolidated(tracker),
        "component_performance": _analyze_component_performance_consolidated(tracker),
        "memory_usage": _analyze_memory_usage_consolidated(tracker),
        "cache_stats": get_cache_stats()
    }
    return stats

# Consolidated analysis functions
def _analyze_query_performance_consolidated(tracker: PerformanceTracker):
    """Analyze query performance using consolidated tracker."""
    query_times = tracker.metrics.get("query_times", [])
    if not query_times:
        return {"status": "No query data available"}

    times = [q["execution_time"] for q in query_times]
    return {
        "total_queries": len(query_times),
        "avg_query_time": statistics.mean(times),
        "slowest_queries": sorted(query_times, key=lambda x: x["execution_time"], reverse=True)[:5]
    }

def _analyze_component_performance_consolidated(tracker: PerformanceTracker):
    """Analyze component performance using consolidated tracker."""
    component_times = tracker.metrics.get("component_render_times", {})
    if not component_times:
        return {"status": "No component data available"}

    analysis = {}
    for component, times in component_times.items():
        if times:
            execution_times = [t["execution_time"] for t in times]
            analysis[component] = {
                "total_renders": len(times),
                "avg_render_time": statistics.mean(execution_times),
                "max_render_time": max(execution_times),
                "recent_renders": times[-5:]  # Last 5 renders
            }

    return analysis

def _analyze_memory_usage_consolidated(tracker: PerformanceTracker):
    """Analyze memory usage using consolidated tracker."""
    snapshots = tracker.metrics.get("memory_snapshots", [])
    if not snapshots:
        return {"status": "No memory data available"}

    latest = snapshots[-1] if snapshots else None
    return {
        "total_snapshots": len(snapshots),
        "latest_memory_mb": latest["total_allocated"] / (1024 * 1024) if latest else 0,
        "memory_trend": "stable"  # Simplified for now
    }

def _analyze_query_performance():
    """Analyze database query performance."""
    if not _performance_metrics["query_times"]:
        return {"status": "No query data available"}
    
    # Calculate average query time
    avg_query_time = statistics.mean(q["execution_time"] for q in _performance_metrics["query_times"])
    
    # Find the slowest queries
    slowest_queries = sorted(
        _performance_metrics["query_times"],
        key=lambda q: q["execution_time"],
        reverse=True
    )[:5]  # Top 5 slowest
    
    # Group by time ranges to see performance over time
    now = datetime.now()
    last_hour = []
    last_day = []
    
    for query in _performance_metrics["query_times"]:
        query_time = datetime.fromisoformat(query["timestamp"])
        if now - query_time < timedelta(hours=1):
            last_hour.append(query["execution_time"])
        if now - query_time < timedelta(days=1):
            last_day.append(query["execution_time"])
    
    return {
        "total_queries": len(_performance_metrics["query_times"]),
        "avg_query_time": avg_query_time,
        "slowest_queries": slowest_queries,
        "last_hour_avg": statistics.mean(last_hour) if last_hour else 0,
        "last_day_avg": statistics.mean(last_day) if last_day else 0
    }

def _analyze_component_performance():
    """Analyze component rendering performance."""
    if not _performance_metrics.get("component_render_times"):
        return {"status": "No component render data available"}
    
    component_stats = {}
    for component, timings in _performance_metrics["component_render_times"].items():
        render_times = [t["render_time"] for t in timings]
        component_stats[component] = {
            "avg_render_time": statistics.mean(render_times),
            "min_render_time": min(render_times),
            "max_render_time": max(render_times),
            "std_dev": statistics.stdev(render_times) if len(render_times) > 1 else 0,
            "total_renders": len(render_times)
        }
    
    return component_stats

def _analyze_memory_usage():
    """Analyze memory usage trends."""
    if not _performance_metrics["memory_snapshots"]:
        return {"status": "No memory data available"}
    
    # Extract total allocated memory over time
    allocated_memory = [
        {
            "timestamp": datetime.fromisoformat(snapshot["timestamp"]),
            "allocated_mb": snapshot["total_allocated"] / (1024 * 1024)  # Convert to MB
        }
        for snapshot in _performance_metrics["memory_snapshots"]
    ]
    
    # Sort by timestamp
    allocated_memory.sort(key=lambda x: x["timestamp"])
    
    # Calculate memory growth rate
    if len(allocated_memory) >= 2:
        first = allocated_memory[0]
        last = allocated_memory[-1]
        time_diff = (last["timestamp"] - first["timestamp"]).total_seconds()
        memory_diff = last["allocated_mb"] - first["allocated_mb"]
        
        if time_diff > 0:
            growth_rate = memory_diff / time_diff  # MB per second
        else:
            growth_rate = 0
    else:
        growth_rate = 0
    
    # Find top memory consumers from the most recent snapshot
    if _performance_metrics["memory_snapshots"]:
        latest_snapshot = max(_performance_metrics["memory_snapshots"], 
                             key=lambda x: datetime.fromisoformat(x["timestamp"]))
        top_consumers = latest_snapshot["top_allocations"]
    else:
        top_consumers = []
    
    return {
        "current_memory_mb": allocated_memory[-1]["allocated_mb"] if allocated_memory else 0,
        "memory_growth_rate_mb_per_sec": growth_rate,
        "top_memory_consumers": top_consumers,
        "memory_over_time": allocated_memory
    }

# Visualization functions for the performance dashboard
def render_performance_dashboard():
    """Render the dashboard performance monitoring UI."""
    st.title("Dashboard Performance Monitor")
    
    # Get the latest performance stats
    stats = get_performance_stats()
    cache_stats = get_cache_stats()
    
    # Create tabs for different performance aspects
    tab1, tab2, tab3, tab4 = st.tabs([
        "Query Performance", 
        "Component Rendering", 
        "Memory Usage",
        "Cache Efficiency"
    ])
    
    with tab1:
        _render_query_performance_tab(stats["query_performance"])
    
    with tab2:
        _render_component_performance_tab(stats["component_performance"])
    
    with tab3:
        _render_memory_usage_tab(stats["memory_usage"])
    
    with tab4:
        _render_cache_efficiency_tab(cache_stats)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Take Memory Snapshot"):
            snapshot = get_memory_snapshot()
            st.success(f"Memory snapshot taken: {snapshot['total_allocated'] / (1024*1024):.2f} MB")
    
    with col2:
        if st.button("Clear Performance Data"):
            _clear_performance_data()
            st.success("Performance data cleared")
    
    with col3:
        if st.button("Generate Performance Report"):
            report = _generate_performance_report()
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def _render_query_performance_tab(query_stats):
    """Render the query performance tab."""
    st.header("Database Query Performance")
    
    if query_stats.get("status") == "No query data available":
        st.info("No query performance data available yet. Start using the dashboard to collect data.")
        return
    
    # Query metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Queries", query_stats["total_queries"])
    with col2:
        st.metric("Avg Query Time", f"{query_stats['avg_query_time']*1000:.2f} ms")
    with col3:
        if query_stats.get("last_hour_avg"):
            st.metric("Last Hour Avg", f"{query_stats['last_hour_avg']*1000:.2f} ms")
    
    # Slowest queries
    st.subheader("Slowest Queries")
    if query_stats.get("slowest_queries"):
        slow_query_df = pd.DataFrame([
            {
                "Query": q["query"],
                "Execution Time (ms)": q["execution_time"] * 1000,
                "Timestamp": q["timestamp"]
            }
            for q in query_stats["slowest_queries"]
        ])
        st.dataframe(slow_query_df)
    else:
        st.info("No slow queries recorded yet.")

def _render_component_performance_tab(component_stats):
    """Render the component performance tab."""
    st.header("Component Rendering Performance")
    
    if component_stats.get("status") == "No component render data available":
        st.info("No component rendering data available yet. Start using the dashboard to collect data.")
        return
    
    # Convert component stats to DataFrame for visualization
    components_df = pd.DataFrame([
        {
            "Component": component,
            "Avg Render Time (ms)": stats["avg_render_time"] * 1000,
            "Min Time (ms)": stats["min_render_time"] * 1000,
            "Max Time (ms)": stats["max_render_time"] * 1000,
            "StdDev (ms)": stats["std_dev"] * 1000,
            "Total Renders": stats["total_renders"]
        }
        for component, stats in component_stats.items()
    ])
    
    # Sort by average render time
    components_df = components_df.sort_values("Avg Render Time (ms)", ascending=False)
    
    # Display component render times
    st.dataframe(components_df)
    
    # Render time chart
    if not components_df.empty:
        fig = px.bar(
            components_df, 
            x="Component", 
            y="Avg Render Time (ms)",
            error_y="StdDev (ms)",
            title="Average Component Render Times",
            color="Avg Render Time (ms)",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

def _render_memory_usage_tab(memory_stats):
    """Render the memory usage tab."""
    st.header("Memory Usage")
    
    if memory_stats.get("status") == "No memory data available":
        st.info("No memory data available yet. Take a snapshot to start tracking.")
        return
    
    # Memory metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Memory Usage", f"{memory_stats['current_memory_mb']:.2f} MB")
    with col2:
        growth_rate = memory_stats["memory_growth_rate_mb_per_sec"]
        st.metric(
            "Memory Growth Rate", 
            f"{growth_rate*3600:.2f} MB/hour",
            delta=f"{growth_rate*60:.2f} MB/min"
        )
    
    # Memory usage over time
    if memory_stats.get("memory_over_time"):
        memory_df = pd.DataFrame([
            {"Timestamp": m["timestamp"], "Memory (MB)": m["allocated_mb"]}
            for m in memory_stats["memory_over_time"]
        ])
        
        fig = px.line(
            memory_df, 
            x="Timestamp", 
            y="Memory (MB)",
            title="Memory Usage Over Time",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top memory consumers
    st.subheader("Top Memory Consumers")
    if memory_stats.get("top_memory_consumers"):
        top_mem_df = pd.DataFrame([
            {
                "File": c["filename"].split("/")[-1],  # Just the filename, not full path
                "Line": c["lineno"],
                "Size (KB)": c["size"] / 1024,
                "Count": c["count"]
            }
            for c in memory_stats["top_memory_consumers"]
        ])
        st.dataframe(top_mem_df)
    else:
        st.info("No memory consumer data available.")

def _render_cache_efficiency_tab(cache_stats):
    """Render the cache efficiency tab."""
    st.header("Cache Efficiency")
    
    # Overall cache metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cache Hits", cache_stats["total_hits"])
    with col2:
        st.metric("Total Cache Misses", cache_stats["total_misses"])
    with col3:
        hit_rate = cache_stats["hit_rate"] * 100
        st.metric("Overall Hit Rate", f"{hit_rate:.1f}%")
    
    # Individual cache stats
    st.subheader("Individual Cache Performance")
    if cache_stats.get("caches"):
        cache_df = pd.DataFrame([
            {
                "Cache": name,
                "Type": info["type"],
                "Function": info["function"],
                "TTL (seconds)": info["ttl"],
                "Hits": info["hits"],
                "Misses": info["misses"],
                "Hit Rate": f"{info['hit_rate']*100:.1f}%"
            }
            for name, info in cache_stats["caches"].items()
        ])
        
        st.dataframe(cache_df)
        
        # Cache hit rate chart
        if not cache_df.empty:
            # Convert hit rate string to float
            cache_df["Hit Rate (%)"] = cache_df["Hit Rate"].str.rstrip("%").astype(float)
            
            fig = px.bar(
                cache_df,
                x="Cache",
                y="Hit Rate (%)",
                color="Type",
                title="Cache Hit Rates",
                hover_data=["Function", "TTL (seconds)", "Hits", "Misses"]
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No individual cache data available.")

def _clear_performance_data():
    """Clear all performance tracking data."""
    global _performance_metrics
    _performance_metrics = {
        "query_times": [],
        "render_times": {},
        "memory_snapshots": [],
        "api_response_times": [],
        "component_render_times": {},
        "data_processing_times": {}
    }
    logger.info("Performance data cleared")

def _generate_performance_report():
    """Generate a comprehensive performance report."""
    stats = get_performance_stats()
    cache_stats = get_cache_stats()
    
    report = [
        f"GoalDiggers Dashboard Performance Report",
        f"Generated: {datetime.now().isoformat()}",
        f"\n{'='*80}\n",
        
        f"QUERY PERFORMANCE",
        f"{'-'*80}",
    ]
    
    # Query performance
    query_stats = stats["query_performance"]
    if query_stats.get("status") != "No query data available":
        report.extend([
            f"Total Queries: {query_stats['total_queries']}",
            f"Average Query Time: {query_stats['avg_query_time']*1000:.2f} ms",
            f"Last Hour Average: {query_stats.get('last_hour_avg', 0)*1000:.2f} ms",
            f"Last Day Average: {query_stats.get('last_day_avg', 0)*1000:.2f} ms",
            f"\nSlowest Queries:",
        ])
        
        for i, query in enumerate(query_stats.get("slowest_queries", []), 1):
            report.append(
                f"{i}. {query['query']} - {query['execution_time']*1000:.2f} ms at {query['timestamp']}"
            )
    else:
        report.append("No query data available")
    
    # Component performance
    report.extend([
        f"\n{'='*80}\n",
        f"COMPONENT RENDERING PERFORMANCE",
        f"{'-'*80}",
    ])
    
    component_stats = stats["component_performance"]
    if component_stats.get("status") != "No component render data available":
        for component, cstats in component_stats.items():
            report.extend([
                f"Component: {component}",
                f"  Average Render Time: {cstats['avg_render_time']*1000:.2f} ms",
                f"  Min/Max Time: {cstats['min_render_time']*1000:.2f} / {cstats['max_render_time']*1000:.2f} ms",
                f"  Standard Deviation: {cstats['std_dev']*1000:.2f} ms",
                f"  Total Renders: {cstats['total_renders']}",
                f""
            ])
    else:
        report.append("No component rendering data available")
    
    # Memory usage
    report.extend([
        f"\n{'='*80}\n",
        f"MEMORY USAGE",
        f"{'-'*80}",
    ])
    
    memory_stats = stats["memory_usage"]
    if memory_stats.get("status") != "No memory data available":
        report.extend([
            f"Current Memory Usage: {memory_stats['current_memory_mb']:.2f} MB",
            f"Memory Growth Rate: {memory_stats['memory_growth_rate_mb_per_sec']*3600:.2f} MB/hour",
            f"\nTop Memory Consumers:",
        ])
        
        for i, consumer in enumerate(memory_stats.get("top_memory_consumers", []), 1):
            report.append(
                f"{i}. {consumer['filename']}:{consumer['lineno']} - {consumer['size']/1024:.2f} KB ({consumer['count']} objects)"
            )
    else:
        report.append("No memory data available")
    
    # Cache efficiency
    report.extend([
        f"\n{'='*80}\n",
        f"CACHE EFFICIENCY",
        f"{'-'*80}",
        f"Total Cache Hits: {cache_stats['total_hits']}",
        f"Total Cache Misses: {cache_stats['total_misses']}",
        f"Overall Hit Rate: {cache_stats['hit_rate']*100:.1f}%",
        f"Last Cleared: {cache_stats['last_cleared']}",
        f"\nIndividual Cache Performance:",
    ])
    
    for name, info in cache_stats.get("caches", {}).items():
        report.extend([
            f"Cache: {name}",
            f"  Type: {info['type']}",
            f"  Function: {info['function']}",
            f"  TTL: {info['ttl']} seconds",
            f"  Hits/Misses: {info['hits']}/{info['misses']}",
            f"  Hit Rate: {info['hit_rate']*100:.1f}%",
            f""
        ])
    
    # Final recommendations
    report.extend([
        f"\n{'='*80}\n",
        f"PERFORMANCE RECOMMENDATIONS",
        f"{'-'*80}",
    ])
    
    # Add automatic recommendations based on collected metrics
    recommendations = _generate_recommendations(stats, cache_stats)
    report.extend(recommendations)
    
    return "\n".join(report)

def _generate_recommendations(stats, cache_stats):
    """Generate performance improvement recommendations based on metrics."""
    recommendations = []
    
    # Query recommendations
    query_stats = stats["query_performance"]
    if query_stats.get("status") != "No query data available":
        avg_query_time = query_stats.get("avg_query_time", 0)
        if avg_query_time > 0.1:  # More than 100ms average
            recommendations.append(
                "- Consider optimizing database queries: The average query time is high."
            )
            recommendations.append(
                "  Add indexes for frequently queried columns or optimize query patterns."
            )
    
    # Cache recommendations
    if cache_stats.get("hit_rate", 0) < 0.7:  # Less than 70% hit rate
        recommendations.append(
            "- Improve cache utilization: The overall cache hit rate is below optimal levels."
        )
        recommendations.append(
            "  Review TTL settings and ensure proper cache invalidation strategies."
        )
    
    # Memory recommendations
    memory_stats = stats["memory_usage"]
    if memory_stats.get("status") != "No memory data available":
        growth_rate = memory_stats.get("memory_growth_rate_mb_per_sec", 0)
        if growth_rate * 3600 > 100:  # More than 100MB/hour growth
            recommendations.append(
                "- Address memory growth: Memory usage is increasing rapidly."
            )
            recommendations.append(
                "  Check for memory leaks, large object retention, or inefficient data structures."
            )
    
    # Component recommendations
    component_stats = stats["component_performance"]
    if component_stats.get("status") != "No component render data available":
        slow_components = [
            component for component, cstats in component_stats.items() 
            if cstats["avg_render_time"] > 0.5  # More than 500ms average render time
        ]
        
        if slow_components:
            recommendations.append(
                f"- Optimize slow components: {', '.join(slow_components)}"
            )
            recommendations.append(
                "  Consider lazy loading, reducing re-renders, or simplifying these components."
            )
    
    # Add general recommendations if we don't have enough data
    if not recommendations:
        recommendations.extend([
            "- Not enough performance data collected for specific recommendations.",
            "- Consider gathering more usage data or triggering manual memory snapshots.",
            "- For general performance improvements:",
            "  * Implement lazy loading for visualizations",
            "  * Use pagination for large datasets",
            "  * Optimize database queries with proper indexing",
            "  * Reduce component re-renders in Streamlit"
        ])
    
    return recommendations
