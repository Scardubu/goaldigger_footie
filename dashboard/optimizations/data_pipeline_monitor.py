"""
Data Pipeline Monitoring Module

This module tracks data processing efficiency, resource usage, and performance metrics
for the GoalDiggers betting insights platform's data pipelines.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

import pandas as pd
import numpy as np

from dashboard.error_log import ErrorLog, log_exceptions_decorator

logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="data_pipeline_monitor")

class DataPipelineMonitor:
    """Monitor data processing efficiency and resource usage."""
    
    def __init__(self):
        """Initialize the data pipeline monitor."""
        self.processing_times = {}
        self.memory_usage = {}
        self.throughput = {}
        self.error_counts = {}
        self.last_reset_time = datetime.now()
    
    @log_exceptions_decorator
    def start_tracking(self, pipeline_name: str) -> str:
        """
        Start tracking a data pipeline operation.
        
        Args:
            pipeline_name: Name of the pipeline to track
            
        Returns:
            Tracking key for the operation
        """
        key = f"{pipeline_name}_{time.time()}"
        self.processing_times[key] = {"start": time.time()}
        return key
    
    @log_exceptions_decorator
    def end_tracking(self, tracking_key: str, items_processed: int, 
                     success_count: Optional[int] = None, 
                     error_count: Optional[int] = None,
                     data_volume_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        End tracking and record metrics.
        
        Args:
            tracking_key: Tracking key returned by start_tracking
            items_processed: Number of items processed
            success_count: Number of successfully processed items
            error_count: Number of items that failed processing
            data_volume_mb: Volume of data processed in MB
            
        Returns:
            Dictionary with performance metrics
        """
        if tracking_key not in self.processing_times:
            logger.warning(f"Unknown tracking key: {tracking_key}")
            return {}
            
        start_time = self.processing_times[tracking_key]["start"]
        duration = time.time() - start_time
        
        pipeline_name = tracking_key.split("_")[0]
        if pipeline_name not in self.throughput:
            self.throughput[pipeline_name] = []
            
        # If success/error counts not provided, assume all items were successful
        if success_count is None:
            success_count = items_processed
        if error_count is None:
            error_count = 0
            
        # Calculate metrics
        throughput_per_second = items_processed / duration if duration > 0 else 0
        success_rate = success_count / items_processed if items_processed > 0 else 0
        
        # Record metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "items_processed": items_processed,
            "success_count": success_count,
            "error_count": error_count,
            "throughput_per_second": throughput_per_second,
            "success_rate": success_rate,
            "data_volume_mb": data_volume_mb
        }
        
        self.throughput[pipeline_name].append(metrics)
        
        # Log slow operations
        if duration > 5.0:  # Configurable threshold
            logger.warning(
                f"Slow data pipeline: {pipeline_name} processed {items_processed} items "
                f"in {duration:.2f}s ({throughput_per_second:.2f} items/s)"
            )
            
        # Keep only the most recent 100 entries per pipeline to manage memory usage
        if len(self.throughput[pipeline_name]) > 100:
            self.throughput[pipeline_name] = self.throughput[pipeline_name][-100:]
            
        return metrics
    
    @log_exceptions_decorator
    def record_error(self, pipeline_name: str, error_type: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Record an error in the pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            error_type: Type of error
            details: Optional error details
        """
        if pipeline_name not in self.error_counts:
            self.error_counts[pipeline_name] = {}
            
        if error_type not in self.error_counts[pipeline_name]:
            self.error_counts[pipeline_name][error_type] = []
            
        self.error_counts[pipeline_name][error_type].append({
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        })
        
        # Keep only the most recent 50 errors per type to manage memory usage
        if len(self.error_counts[pipeline_name][error_type]) > 50:
            self.error_counts[pipeline_name][error_type] = self.error_counts[pipeline_name][error_type][-50:]
    
    @log_exceptions_decorator
    def get_pipeline_stats(self, pipeline_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for one or all pipelines.
        
        Args:
            pipeline_name: Optional name of pipeline to get stats for (None for all)
            
        Returns:
            Dictionary with pipeline statistics
        """
        if pipeline_name:
            return self._calculate_pipeline_stats(pipeline_name)
        
        # Calculate stats for all pipelines
        all_stats = {}
        for name in self.throughput.keys():
            all_stats[name] = self._calculate_pipeline_stats(name)
            
        return all_stats
    
    def _calculate_pipeline_stats(self, pipeline_name: str) -> Dict[str, Any]:
        """Calculate statistics for a specific pipeline."""
        if pipeline_name not in self.throughput or not self.throughput[pipeline_name]:
            return {
                "status": "No data available",
                "total_items_processed": 0,
                "total_duration": 0,
                "avg_throughput": 0,
                "success_rate": 0,
                "error_count": 0
            }
            
        pipeline_data = self.throughput[pipeline_name]
        
        # Calculate summary statistics
        total_items = sum(entry["items_processed"] for entry in pipeline_data)
        total_duration = sum(entry["duration"] for entry in pipeline_data)
        total_errors = sum(entry["error_count"] for entry in pipeline_data)
        avg_throughput = total_items / total_duration if total_duration > 0 else 0
        success_rate = (total_items - total_errors) / total_items if total_items > 0 else 0
        
        # Calculate recent performance (last 10 entries)
        recent_data = pipeline_data[-10:] if len(pipeline_data) >= 10 else pipeline_data
        recent_items = sum(entry["items_processed"] for entry in recent_data)
        recent_duration = sum(entry["duration"] for entry in recent_data)
        recent_throughput = recent_items / recent_duration if recent_duration > 0 else 0
        
        # Get error details if available
        error_details = {}
        if pipeline_name in self.error_counts:
            error_details = {
                "total_error_types": len(self.error_counts[pipeline_name]),
                "error_types": list(self.error_counts[pipeline_name].keys()),
                "most_common_error": max(
                    self.error_counts[pipeline_name].items(), 
                    key=lambda x: len(x[1])
                )[0] if self.error_counts[pipeline_name] else None
            }
        
        return {
            "status": "Active" if recent_duration > 0 else "Inactive",
            "total_items_processed": total_items,
            "total_duration": total_duration,
            "avg_throughput": avg_throughput,
            "recent_throughput": recent_throughput,
            "success_rate": success_rate,
            "error_count": total_errors,
            "error_details": error_details,
            "operations_count": len(pipeline_data)
        }
    
    @log_exceptions_decorator
    def reset_stats(self, pipeline_name: Optional[str] = None) -> None:
        """
        Reset statistics for one or all pipelines.
        
        Args:
            pipeline_name: Optional name of pipeline to reset (None for all)
        """
        if pipeline_name:
            if pipeline_name in self.throughput:
                self.throughput[pipeline_name] = []
            if pipeline_name in self.error_counts:
                self.error_counts[pipeline_name] = {}
        else:
            # Reset all pipelines
            self.throughput = {}
            self.error_counts = {}
            self.processing_times = {}
            
        self.last_reset_time = datetime.now()
    
    @log_exceptions_decorator
    def generate_report(self, pipeline_name: Optional[str] = None) -> str:
        """
        Generate a comprehensive report of pipeline performance.
        
        Args:
            pipeline_name: Optional name of pipeline to report on (None for all)
            
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 80,
            f"DATA PIPELINE PERFORMANCE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            f"Period: {self.last_reset_time.strftime('%Y-%m-%d %H:%M:%S')} to {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        stats = self.get_pipeline_stats(pipeline_name)
        
        if pipeline_name:
            # Single pipeline report
            if pipeline_name not in self.throughput:
                report_lines.append(f"No data available for pipeline: {pipeline_name}")
                return "\n".join(report_lines)
                
            pipeline_stats = stats
            report_lines.extend(self._format_pipeline_stats(pipeline_name, pipeline_stats))
        else:
            # All pipelines report
            if not stats:
                report_lines.append("No pipeline data available")
                return "\n".join(report_lines)
                
            for name, pipeline_stats in stats.items():
                report_lines.extend(self._format_pipeline_stats(name, pipeline_stats))
                report_lines.append("-" * 80)
        
        return "\n".join(report_lines)
    
    def _format_pipeline_stats(self, name: str, stats: Dict[str, Any]) -> List[str]:
        """Format statistics for a single pipeline."""
        lines = [
            f"PIPELINE: {name}",
            f"Status: {stats['status']}",
            f"Total Items Processed: {stats['total_items_processed']}",
            f"Total Duration: {stats['total_duration']:.2f} seconds",
            f"Average Throughput: {stats['avg_throughput']:.2f} items/second",
            f"Recent Throughput: {stats['recent_throughput']:.2f} items/second",
            f"Success Rate: {stats['success_rate']*100:.2f}%",
            f"Error Count: {stats['error_count']}",
            ""
        ]
        
        # Add error details if available
        if stats.get('error_details') and stats['error_details'].get('total_error_types', 0) > 0:
            lines.append("ERROR BREAKDOWN:")
            lines.append(f"Total Error Types: {stats['error_details']['total_error_types']}")
            lines.append(f"Most Common Error: {stats['error_details']['most_common_error']}")
            lines.append("Error Types: " + ", ".join(stats['error_details']['error_types']))
            lines.append("")
        
        return lines

# Create a global instance for easy import
data_pipeline_monitor = DataPipelineMonitor()
