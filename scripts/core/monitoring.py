import logging
import os
import time  # Added for runtime calculation
from typing import Any, Dict, Optional  # Added for typing AND Optional

import plotly.graph_objects as go
import sentry_sdk
from dotenv import load_dotenv
# Added for PerformanceMonitor dashboard
from plotly.subplots import make_subplots
from sentry_sdk.integrations.logging import LoggingIntegration

from dashboard.error_log import log_error  # Import log_error

# Load environment variables (optional, if running standalone or need SENTRY_DSN early)
load_dotenv()

logger = logging.getLogger(__name__)

def init_monitoring():
    """Initializes Sentry monitoring if SENTRY_DSN is configured."""
    sentry_dsn = os.getenv("SENTRY_DSN")
    if sentry_dsn:
        try:
            sentry_sdk.init(
                dsn=sentry_dsn,
                # Enable logging integration to capture log messages as breadcrumbs
                integrations=[
                    LoggingIntegration(
                        level=logging.INFO,        # Capture info and above as breadcrumbs
                        event_level=logging.ERROR  # Send errors as events
                    )
                ],
                # Set traces_sample_rate to 1.0 to capture 100%
                # of transactions for performance monitoring.
                # Adjust sampling rate in production as needed.
                traces_sample_rate=1.0,
                # Consider setting environment and release for better filtering in Sentry
                # environment=os.getenv("APP_ENV", "development"),
                # release="goaldiggers-analytics@1.0.0" # Example release version
            )
            logger.info("Sentry SDK initialized successfully.")
        except Exception as e:
            log_error("High-level operation failed", e)
    else:
        logger.warning("SENTRY_DSN environment variable not set. Sentry monitoring is disabled.")

class PredictionMonitor:
    """A simple in-memory monitor for tracking prediction statistics."""
    def __init__(self):
        self.stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0
        }
        logger.info("PredictionMonitor initialized.")

    def track_prediction(self, success: bool = True):
        """Tracks a single prediction attempt."""
        self.stats["total_predictions"] += 1
        if success:
            self.stats["successful_predictions"] += 1
        else:
            self.stats["failed_predictions"] += 1
        # logger.debug(f"Prediction tracked. Success: {success}. Current stats: {self.stats}")

    def get_metrics(self) -> dict:
        """Calculates and returns current monitoring metrics."""
        total = self.stats["total_predictions"]
        successful = self.stats["successful_predictions"]
        failed = self.stats["failed_predictions"]

        # Calculate success rate, handle division by zero
        success_rate = (successful / total) if total > 0 else 0.0

        metrics = {
            "total_predictions": total,
            "successful_predictions": successful,
            "failed_predictions": failed,
            "success_rate": round(success_rate, 4) # Round for cleaner output
        }
        logger.debug(f"Returning metrics: {metrics}")
        return metrics

    def reset(self):
        """Resets the statistics."""
        self.stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0
        }
        logger.info("PredictionMonitor stats reset.")

# --- Added Performance Monitoring System ---

class PerformanceMonitor:
    """Monitors performance across different system components."""
    def __init__(self):
        # Initialize metrics structure for key components
        self.metrics: Dict[str, Dict[str, Any]] = {
            'scraping': {'success': 0, 'failures': 0, 'total_time': 0.0, 'count': 0},
            'proxy': {'success': 0, 'failures': 0, 'total_time': 0.0, 'count': 0}, # Added proxy component
            'model': {'predictions': 0, 'total_runtime': 0.0, 'count': 0},
            'validation': {'passed': 0, 'failed': 0, 'count': 0},
            'feature_eng': {'success': 0, 'failures': 0, 'total_time': 0.0, 'count': 0} # Added feature_eng
            # Add more components as needed
        }
        self.component_keys = list(self.metrics.keys()) # Store component names
        logger.info("PerformanceMonitor initialized.")

    def update(self, component: str, success: bool, runtime: float):
        """
        Updates the metrics for a specific component.

        Args:
            component (str): The name of the component (e.g., 'scraping', 'model', 'validation').
            success (bool): Whether the operation was successful.
            runtime (float): The duration of the operation in seconds.
        """
        if component not in self.metrics:
            logger.warning(f"Attempted to update metrics for unknown component: {component}")
            # Optionally initialize the component metrics here if desired
            # self.metrics[component] = {'success': 0, 'failures': 0, 'total_time': 0.0, 'count': 0}
            return

        self.metrics[component]['count'] += 1

        if component == 'scraping':
            self.metrics[component]['success' if success else 'failures'] += 1
            self.metrics[component]['total_time'] += runtime
        elif component == 'model':
            # Assuming 'success' indicates a prediction was generated (even if low confidence)
            if success:
                self.metrics[component]['predictions'] += 1 # Count successful predictions
            # Always track runtime for model component
            self.metrics[component]['total_runtime'] += runtime
        elif component == 'validation':
            self.metrics[component]['passed' if success else 'failed'] += 1
        else:
             # Handle generic or other components if added later
             pass # Or add specific logic

        # logger.debug(f"Updated performance for '{component}': Success={success}, Runtime={runtime:.4f}s. Current: {self.metrics[component]}")


    def get_metrics(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns calculated metrics for a specific component or all components.

        Args:
            component (Optional[str]): The specific component to get metrics for.
                                       If None, returns metrics for all components.

        Returns:
            Dict[str, Any]: A dictionary containing calculated metrics.
        """
        if component:
            if component not in self.metrics:
                return {"error": f"Component '{component}' not found."}
            return self._calculate_component_metrics(component)
        else:
            all_metrics = {}
            for comp in self.metrics.keys():
                all_metrics[comp] = self._calculate_component_metrics(comp)
            return all_metrics

    def _calculate_component_metrics(self, component: str) -> Dict[str, Any]:
        """Helper to calculate derived metrics for a single component."""
        data = self.metrics[component]
        count = data.get('count', 0)
        calculated = {'raw_counts': data.copy()} # Include raw counts

        if component == 'scraping':
            total_time = data.get('total_time', 0.0)
            success = data.get('success', 0)
            failures = data.get('failures', 0)
            calculated['avg_time'] = (total_time / count) if count > 0 else 0.0
            calculated['success_rate'] = (success / count) if count > 0 else 0.0
            calculated['failure_rate'] = (failures / count) if count > 0 else 0.0
        elif component == 'model':
            total_runtime = data.get('total_runtime', 0.0)
            predictions = data.get('predictions', 0) # Successful predictions
            calculated['avg_runtime'] = (total_runtime / count) if count > 0 else 0.0
            # 'predictions' already represents successful count here
        elif component == 'validation':
            passed = data.get('passed', 0)
            failed = data.get('failed', 0)
            calculated['pass_rate'] = (passed / count) if count > 0 else 0.0
            calculated['fail_rate'] = (failed / count) if count > 0 else 0.0
        elif component == 'proxy': # Added proxy calculations
            total_time = data.get('total_time', 0.0)
            success = data.get('success', 0)
            failures = data.get('failures', 0)
            calculated['avg_time'] = (total_time / count) if count > 0 else 0.0
            calculated['success_rate'] = (success / count) if count > 0 else 0.0
            calculated['failure_rate'] = (failures / count) if count > 0 else 0.0
        elif component == 'feature_eng': # Added feature_eng calculations
            # Need to ensure 'success' and 'failures' keys exist from update() calls
            total_time = data.get('total_time', 0.0)
            success = data.get('success', 0) # Assuming FeatureGenerator reports success/failure
            # Calculate failures based on count and success if not explicitly tracked
            failures = data.get('failures', count - success)
            calculated['avg_time'] = (total_time / count) if count > 0 else 0.0
            calculated['success_rate'] = (success / count) if count > 0 else 0.0
            calculated['failure_rate'] = (failures / count) if count > 0 else 0.0
        # Add calculations for other components if needed

        # Round floats for display
        for key, value in calculated.items():
            if isinstance(value, float):
                calculated[key] = round(value, 4)

        return calculated

    def get_dashboard(self) -> go.Figure:
        """Generates a Plotly figure summarizing the performance metrics."""
        metrics = self.get_metrics() # Get calculated metrics for all components

        # Determine grid size: One row per component, two columns (Pie, Bar)
        num_components = len(self.metrics)
        rows = num_components # Ensure one row per component
        cols = 2

        # Create specs and titles dynamically based on the number of components
        specs = [[{'type': 'domain'}, {'type': 'xy'}] for _ in range(rows)]
        titles = []
        for comp in self.metrics.keys():
             # One title for the pie chart column, one for the bar chart column
             titles.append(f"{comp.capitalize()} Status")
             titles.append(f"{comp.capitalize()} Performance Metrics")

        # Ensure titles list matches the number of subplots (rows * cols)
        # If num_components is 0, titles should be empty
        expected_titles = rows * cols
        if len(titles) != expected_titles:
             # Handle potential edge case if num_components was 0 or logic changes
             logger.warning(f"Mismatch in subplot titles. Expected {expected_titles}, got {len(titles)}. Adjusting.")
             # Fallback or adjust titles as needed. For now, let's use the generated ones if possible.
             # If expected_titles is 0, titles should be None or empty list for make_subplots
             if expected_titles == 0:
                 titles = None
             else:
                 # If mismatch for non-zero components, something is wrong. Log and potentially truncate/pad.
                 # For safety, let's truncate if too long, though this indicates a logic error.
                 titles = titles[:expected_titles]


        fig = make_subplots(
            rows=rows, cols=cols,
            specs=specs,
            subplot_titles=titles
        )

        plot_row, plot_col = 1, 1
        for comp, comp_metrics in metrics.items():
            # --- Pie Chart for Success/Failure/Pass/Fail ---
            labels = []
            values = []
            colors = []
            title_suffix = "Status"
            if comp == 'scraping':
                labels = ['Success', 'Failures']
                values = [comp_metrics['raw_counts']['success'], comp_metrics['raw_counts']['failures']]
                colors = ['#1f77b4', '#d62728'] # Blue, Red
                title_suffix = "Success Rate"
            elif comp == 'validation':
                labels = ['Passed', 'Failed']
                values = [comp_metrics['raw_counts']['passed'], comp_metrics['raw_counts']['failed']]
                colors = ['#2ca02c', '#ff7f0e'] # Green, Orange
                title_suffix = "Pass Rate"
            elif comp == 'model':
                 # Model might just have total count vs successful predictions
                 labels = ['Successful Preds', 'Other Runs (e.g. errors)']
                 values = [comp_metrics['raw_counts']['predictions'], comp_metrics['raw_counts']['count'] - comp_metrics['raw_counts']['predictions']]
                 colors = ['#9467bd', '#8c564b'] # Purple, Brown
                 title_suffix = "Prediction Runs"
            elif comp == 'proxy': # Added proxy pie chart
                 labels = ['Success', 'Failures']
                 # Ensure raw_counts exists before accessing keys
                 raw_counts = comp_metrics.get('raw_counts', {})
                 values = [raw_counts.get('success', 0), raw_counts.get('failures', 0)]
                 colors = ['#17becf', '#e377c2'] # Cyan, Pink
                 title_suffix = "Proxy Usage"
            elif comp == 'feature_eng': # Added feature_eng pie chart
                 labels = ['Success', 'Failures']
                 # Ensure raw_counts exists before accessing keys
                 raw_counts = comp_metrics.get('raw_counts', {})
                 values = [raw_counts.get('success', 0), raw_counts.get('failures', 0)]
                 colors = ['#bcbd22', '#7f7f7f'] # Olive, Grey
                 title_suffix = "Feature Eng Runs"


            if labels and sum(values) > 0: # Only plot if there's data
                fig.add_trace(go.Pie(labels=labels, values=values, name=comp.capitalize(),
                                     marker_colors=colors, hole=.3),
                              row=plot_row, col=plot_col)

            # --- Bar Chart for Avg Time / Rates ---
            plot_col += 1 # Move to next column for the bar chart
            bar_x = []
            bar_y = []
            bar_title = ""
            if comp == 'scraping':
                 bar_x = ['Avg Time (s)', 'Success Rate']
                 bar_y = [comp_metrics.get('avg_time', 0), comp_metrics.get('success_rate', 0)]
                 bar_title = "Scraping Performance"
            elif comp == 'model':
                 bar_x = ['Avg Runtime (s)']
                 bar_y = [comp_metrics.get('avg_runtime', 0)]
                 bar_title = "Model Performance"
            elif comp == 'validation':
                 bar_x = ['Pass Rate', 'Fail Rate']
                 bar_y = [comp_metrics.get('pass_rate', 0), comp_metrics.get('fail_rate', 0)]
                 bar_title = "Validation Performance"
            elif comp == 'proxy': # Added proxy bar chart
                 bar_x = ['Avg Time (s)', 'Success Rate']
                 bar_y = [comp_metrics.get('avg_time', 0), comp_metrics.get('success_rate', 0)]
                 bar_title = "Proxy Performance"
            elif comp == 'feature_eng': # Added feature_eng bar chart
                 bar_x = ['Avg Time (s)', 'Success Rate']
                 bar_y = [comp_metrics.get('avg_time', 0), comp_metrics.get('success_rate', 0)]
                 bar_title = "Feature Eng Performance"


            if bar_x:
                 fig.add_trace(go.Bar(x=bar_x, y=bar_y, name=bar_title, showlegend=False),
                               row=plot_row, col=plot_col)
                 # Add text labels to bars
                 fig.add_trace(go.Scatter(x=bar_x, y=[y + 0.01 * max(bar_y + [1]) for y in bar_y], # Position above
                                          mode='text', text=[f'{v:.3f}' for v in bar_y],
                                          textposition='top center', showlegend=False),
                               row=plot_row, col=plot_col)


            # Move to next row if needed
            if plot_col == cols:
                plot_row += 1
                plot_col = 1
            else:
                 # If staying on same row, ensure next plot starts in correct column
                 pass # plot_col is already incremented


        fig.update_layout(
            title_text="System Performance Overview",
            height=400 * rows, # Adjust height based on rows
            showlegend=False # Pie charts have labels, bars have titles
        )
        # Update y-axis ranges for rate bars if needed
        # fig.update_yaxes(range=[0, 1.1], row=1, col=2) # Example for rates

        return fig

    def reset(self):
        """Resets all performance metrics."""
        for component in self.metrics:
             self.metrics[component] = {key: 0 if isinstance(val, int) else 0.0 for key, val in self.metrics[component].items()}
             self.metrics[component]['count'] = 0 # Ensure count is reset
        logger.info("PerformanceMonitor metrics reset.")

    def get_metrics_json(self, component: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns all performance metrics as a JSON-serializable dict for export.
        Args:
            component (Optional[str]): If provided, only export metrics for this component.
        Returns:
            Dict[str, Any]: JSON-serializable metrics.
        """
        return self.get_metrics(component)


# Example usage (optional, for testing)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG)
#     init_monitoring() # Initialize Sentry if DSN is set
#
#     monitor = PredictionMonitor()
#     monitor.track_prediction(success=True)
#     monitor.track_prediction(success=True)
#     monitor.track_prediction(success=False)
#
#     print("Current Metrics:", monitor.get_metrics())
#
#     # Example of capturing an exception with Sentry (if initialized)
#     try:
#         result = 1 / 0
#     except ZeroDivisionError as e:
#         logger.error("A handled error occurred.", exc_info=True) # Log the error
#         sentry_sdk.capture_exception(e) # Explicitly capture exception in Sentry
#         print("Captured ZeroDivisionError with Sentry (if enabled).")
#
#     monitor.reset()
#     print("Reset Metrics:", monitor.get_metrics())
