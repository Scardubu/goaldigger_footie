"""
Data Integrity Visualization components for the GoalDiggers dashboard.

This module provides specialized visualization components to help identify
and diagnose data completeness and integrity issues in the dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union, Tuple
import datetime
import logging
from functools import wraps

from dashboard.error_log import log_exceptions_decorator, ErrorLog
from dashboard.components.ui_elements import (
    card, 
    badge, 
    progress_indicator,
    info_tooltip
)

# Set up logging
logger = logging.getLogger(__name__)
error_log = ErrorLog(component_name="data_integrity_visualizer")

class DataIntegrityVisualizer:
    """Class for visualizing data integrity and completeness metrics."""
    
    def __init__(self, theme: str = "light"):
        """
        Initialize the data integrity visualizer.
        
        Args:
            theme: The current theme (light, dark, high_contrast)
        """
        self.theme = theme
        self.color_map = self._get_color_map(theme)
    
    def _get_color_map(self, theme: str) -> Dict[str, str]:
        """Get color map based on theme."""
        if theme == "dark":
            return {
                "complete": "#4CAF50",
                "partial": "#FFC107",
                "missing": "#F44336",
                "background": "#1E1E1E",
                "text": "#E0E0E0",
                "grid": "#333333"
            }
        elif theme == "high_contrast":
            return {
                "complete": "#00FF00",
                "partial": "#FFFF00",
                "missing": "#FF0000",
                "background": "#000000",
                "text": "#FFFFFF",
                "grid": "#444444"
            }
        else:  # light theme
            return {
                "complete": "#4CAF50",
                "partial": "#FFC107",
                "missing": "#F44336",
                "background": "#FFFFFF",
                "text": "#333333",
                "grid": "#E0E0E0"
            }
    
    def update_theme(self, theme: str) -> None:
        """Update the visualizer theme."""
        self.theme = theme
        self.color_map = self._get_color_map(theme)
    
    @log_exceptions_decorator
    def render_data_completeness_heatmap(self, 
                                         data: pd.DataFrame, 
                                         fields_to_check: List[str],
                                         title: str = "Data Completeness Heatmap",
                                         height: int = 500,
                                         show_percentages: bool = True) -> None:
        """
        Render a heatmap showing the completeness of specified fields across the dataset.
        
        Args:
            data: DataFrame containing the data to check
            fields_to_check: List of field names to check for completeness
            title: Title for the visualization
            height: Height of the visualization in pixels
            show_percentages: Whether to show completion percentages
        """
        try:
            # Calculate completeness for each field
            completeness_data = []
            
            for field in fields_to_check:
                if field in data.columns:
                    # Calculate percentage of non-null values
                    non_null_count = data[field].notna().sum()
                    total_count = len(data)
                    percentage = (non_null_count / total_count) * 100 if total_count > 0 else 0
                    
                    completeness_data.append({
                        "Field": field,
                        "Complete (%)": round(percentage, 1),
                        "Complete Count": non_null_count,
                        "Missing Count": total_count - non_null_count,
                        "Total Count": total_count
                    })
                else:
                    # Field doesn't exist in the dataframe
                    completeness_data.append({
                        "Field": field,
                        "Complete (%)": 0,
                        "Complete Count": 0,
                        "Missing Count": len(data),
                        "Total Count": len(data)
                    })
            
            # Create DataFrame from completeness data
            completeness_df = pd.DataFrame(completeness_data)
            
            # Sort by completeness percentage
            completeness_df = completeness_df.sort_values("Complete (%)", ascending=False)
            
            # Create the heatmap
            fig = px.bar(
                completeness_df,
                x="Field",
                y="Complete (%)",
                color="Complete (%)",
                color_continuous_scale=["red", "yellow", "green"],
                labels={"Complete (%)": "Completeness (%)", "Field": "Field Name"},
                height=height,
                text="Complete (%)" if show_percentages else None
            )
            
            # Update layout for better appearance
            fig.update_layout(
                title=title,
                paper_bgcolor=self.color_map["background"],
                plot_bgcolor=self.color_map["background"],
                font=dict(
                    color=self.color_map["text"]
                ),
                xaxis=dict(
                    gridcolor=self.color_map["grid"],
                    tickangle=-45
                ),
                yaxis=dict(
                    gridcolor=self.color_map["grid"],
                    range=[0, 100]
                ),
                coloraxis_colorbar=dict(
                    title="Completeness (%)"
                )
            )
            
            # Format text labels
            if show_percentages:
                fig.update_traces(
                    texttemplate='%{y:.1f}%',
                    textposition='outside'
                )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Show tabular data with more details
            with st.expander("View Detailed Completeness Data"):
                st.dataframe(completeness_df)
                
        except Exception as e:
            st.error(f"Failed to render data completeness heatmap: {str(e)}")
            error_log.error(
                "Failed to render data completeness heatmap",
                exception=e,
                err_type="visualization_error",
                details={"fields_checked": fields_to_check}
            )
    
    @log_exceptions_decorator
    def render_data_integrity_matrix(self,
                                     data: pd.DataFrame,
                                     related_fields: Dict[str, List[str]],
                                     title: str = "Data Integrity Matrix",
                                     height: int = 600) -> None:
        """
        Render a matrix visualization showing relationships between data fields.
        
        Args:
            data: DataFrame containing the data to analyze
            related_fields: Dictionary mapping field groups to lists of related fields
            title: Title for the visualization
            height: Height of the visualization in pixels
        """
        try:
            # Prepare data for the matrix
            matrix_data = []
            
            for group_name, fields in related_fields.items():
                # Check if all fields in this group exist in the dataframe
                existing_fields = [f for f in fields if f in data.columns]
                
                if existing_fields:
                    # Calculate how many rows have all fields complete
                    all_complete = data[existing_fields].notna().all(axis=1).sum()
                    # Calculate how many rows have some fields complete
                    some_complete = data[existing_fields].notna().any(axis=1).sum() - all_complete
                    # Calculate how many rows have no fields complete
                    none_complete = len(data) - all_complete - some_complete
                    
                    # Calculate percentages
                    total = len(data)
                    all_pct = (all_complete / total) * 100 if total > 0 else 0
                    some_pct = (some_complete / total) * 100 if total > 0 else 0
                    none_pct = (none_complete / total) * 100 if total > 0 else 0
                    
                    matrix_data.append({
                        "Group": group_name,
                        "Complete": all_complete,
                        "Complete (%)": round(all_pct, 1),
                        "Partial": some_complete,
                        "Partial (%)": round(some_pct, 1),
                        "Missing": none_complete,
                        "Missing (%)": round(none_pct, 1),
                        "Total": total,
                        "Fields": ", ".join(existing_fields)
                    })
            
            # Create DataFrame from matrix data
            matrix_df = pd.DataFrame(matrix_data)
            
            if matrix_df.empty:
                st.warning("No data available for integrity matrix visualization.")
                return
            
            # Sort by completeness percentage
            matrix_df = matrix_df.sort_values("Complete (%)", ascending=False)
            
            # Create a stacked bar chart
            fig = go.Figure()
            
            # Add complete data
            fig.add_trace(go.Bar(
                y=matrix_df["Group"],
                x=matrix_df["Complete (%)"],
                name="Complete",
                orientation="h",
                marker=dict(color=self.color_map["complete"]),
                text=matrix_df["Complete (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto"
            ))
            
            # Add partial data
            fig.add_trace(go.Bar(
                y=matrix_df["Group"],
                x=matrix_df["Partial (%)"],
                name="Partial",
                orientation="h",
                marker=dict(color=self.color_map["partial"]),
                text=matrix_df["Partial (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto"
            ))
            
            # Add missing data
            fig.add_trace(go.Bar(
                y=matrix_df["Group"],
                x=matrix_df["Missing (%)"],
                name="Missing",
                orientation="h",
                marker=dict(color=self.color_map["missing"]),
                text=matrix_df["Missing (%)"].apply(lambda x: f"{x:.1f}%"),
                textposition="auto"
            ))
            
            # Update layout
            fig.update_layout(
                title=title,
                barmode="stack",
                height=height,
                paper_bgcolor=self.color_map["background"],
                plot_bgcolor=self.color_map["background"],
                font=dict(color=self.color_map["text"]),
                xaxis=dict(
                    title="Percentage (%)",
                    gridcolor=self.color_map["grid"],
                    range=[0, 100]
                ),
                yaxis=dict(
                    title="Field Group",
                    gridcolor=self.color_map["grid"]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Show tabular data with more details
            with st.expander("View Detailed Integrity Data"):
                st.dataframe(matrix_df)
                
        except Exception as e:
            st.error(f"Failed to render data integrity matrix: {str(e)}")
            error_log.error(
                "Failed to render data integrity matrix",
                exception=e,
                err_type="visualization_error",
                details={"field_groups": list(related_fields.keys())}
            )
    
    @log_exceptions_decorator
    def render_time_series_completeness(self,
                                        data: pd.DataFrame,
                                        date_field: str,
                                        fields_to_check: List[str],
                                        time_interval: str = "W",
                                        title: str = "Data Completeness Over Time",
                                        height: int = 500) -> None:
        """
        Render a time series visualization showing data completeness over time.
        
        Args:
            data: DataFrame containing the data to analyze
            date_field: Name of the field containing date values
            fields_to_check: List of field names to check for completeness
            time_interval: Time interval for grouping (D=daily, W=weekly, M=monthly)
            title: Title for the visualization
            height: Height of the visualization in pixels
        """
        try:
            # Check if date field exists
            if date_field not in data.columns:
                st.warning(f"Date field '{date_field}' not found in data.")
                return
            
            # Convert date field to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data[date_field]):
                try:
                    data = data.copy()
                    data[date_field] = pd.to_datetime(data[date_field])
                except Exception as e:
                    st.error(f"Failed to convert date field to datetime: {str(e)}")
                    return
            
            # Filter out invalid dates
            data = data[data[date_field].notna()]
            
            if data.empty:
                st.warning("No valid date data available for time series visualization.")
                return
            
            # Group by time interval
            data['time_group'] = data[date_field].dt.to_period(time_interval)
            
            # Check for fields existence
            existing_fields = [f for f in fields_to_check if f in data.columns]
            
            if not existing_fields:
                st.warning("None of the specified fields exist in the data.")
                return
            
            # Calculate completeness for each time group and field
            completeness_data = []
            
            for time_group, group_data in data.groupby('time_group'):
                for field in existing_fields:
                    # Calculate percentage of non-null values
                    non_null_count = group_data[field].notna().sum()
                    total_count = len(group_data)
                    percentage = (non_null_count / total_count) * 100 if total_count > 0 else 0
                    
                    completeness_data.append({
                        "Time Period": time_group.to_timestamp(),
                        "Field": field,
                        "Complete (%)": round(percentage, 1),
                        "Complete Count": non_null_count,
                        "Total Count": total_count
                    })
            
            # Create DataFrame from completeness data
            completeness_df = pd.DataFrame(completeness_data)
            
            # Create the line chart
            fig = px.line(
                completeness_df,
                x="Time Period",
                y="Complete (%)",
                color="Field",
                markers=True,
                labels={"Complete (%)": "Completeness (%)", "Time Period": "Time Period"},
                title=title,
                height=height
            )
            
            # Update layout for better appearance
            fig.update_layout(
                paper_bgcolor=self.color_map["background"],
                plot_bgcolor=self.color_map["background"],
                font=dict(
                    color=self.color_map["text"]
                ),
                xaxis=dict(
                    gridcolor=self.color_map["grid"]
                ),
                yaxis=dict(
                    gridcolor=self.color_map["grid"],
                    range=[0, 100]
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
            
            # Show tabular data with more details
            with st.expander("View Detailed Time Series Data"):
                # Sort by time period and field
                display_df = completeness_df.sort_values(["Time Period", "Field"])
                st.dataframe(display_df)
                
        except Exception as e:
            st.error(f"Failed to render time series completeness: {str(e)}")
            error_log.error(
                "Failed to render time series completeness",
                exception=e,
                err_type="visualization_error",
                details={"date_field": date_field, "fields_checked": fields_to_check}
            )
    
    @log_exceptions_decorator
    def render_missing_data_dashboard(self,
                                      data: pd.DataFrame,
                                      fields_of_interest: Dict[str, List[str]],
                                      date_field: Optional[str] = None) -> None:
        """
        Render a comprehensive dashboard for analyzing missing data.
        
        Args:
            data: DataFrame containing the data to analyze
            fields_of_interest: Dictionary mapping category names to lists of related fields
            date_field: Optional name of the date field for time-series analysis
        """
        try:
            st.markdown("## 📊 Data Integrity Dashboard")
            st.markdown("""This dashboard helps identify missing or incomplete data across the dataset,
                        which may impact feature generation and model performance.""")
            
            # Calculate overall completeness
            all_fields = [field for fields in fields_of_interest.values() for field in fields]
            existing_fields = [f for f in all_fields if f in data.columns]
            
            if not existing_fields:
                st.warning("None of the specified fields exist in the data.")
                return
            
            # Overall stats
            total_fields = len(existing_fields)
            total_data_points = len(data) * total_fields
            
            # Calculate missing data points
            missing_counts = data[existing_fields].isna().sum().sum()
            missing_percentage = (missing_counts / total_data_points) * 100 if total_data_points > 0 else 0
            
            # Display overview metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                completeness = 100 - missing_percentage
                st.markdown(
                    f"""<div style='text-align:center;'>
                        <h3>Overall Completeness</h3>
                        {progress_indicator(completeness, 100, style="circle", animated=True, size="large")}
                    </div>""",
                    unsafe_allow_html=True
                )
            
            with col2:
                st.metric(
                    "Total Records",
                    f"{len(data):,}",
                    help="Total number of records in the dataset"
                )
                st.metric(
                    "Fields Analyzed",
                    f"{total_fields}",
                    help="Number of fields being analyzed"
                )
            
            with col3:
                st.metric(
                    "Missing Data Points",
                    f"{missing_counts:,}",
                    f"{missing_percentage:.1f}% of total",
                    delta_color="inverse"
                )
                
                # Add warning if missing data is high
                if missing_percentage > 20:
                    st.markdown(
                        f"""{badge("High Missing Data", type="danger", icon="⚠️", pulse=True, 
                              tooltip="Missing data may significantly impact model performance")}""",
                        unsafe_allow_html=True
                    )
            
            # Add tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Field Completeness", "Data Group Integrity", "Time Series Analysis"])
            
            with tab1:
                # Flatten the list of fields
                self.render_data_completeness_heatmap(
                    data,
                    existing_fields,
                    title="Field Completeness Analysis",
                    height=max(400, len(existing_fields) * 15)  # Adjust height based on number of fields
                )
            
            with tab2:
                # Filter the dictionary to only include fields that exist in the data
                filtered_fields = {
                    group: [f for f in fields if f in data.columns]
                    for group, fields in fields_of_interest.items()
                }
                
                # Only include groups with at least one existing field
                filtered_fields = {
                    group: fields for group, fields in filtered_fields.items() if fields
                }
                
                self.render_data_integrity_matrix(
                    data,
                    filtered_fields,
                    title="Data Group Integrity Analysis",
                    height=max(400, len(filtered_fields) * 50)  # Adjust height based on number of groups
                )
            
            with tab3:
                if date_field and date_field in data.columns:
                    # Allow user to select time interval
                    time_interval = st.selectbox(
                        "Time Interval",
                        options=[
                            ("Daily", "D"),
                            ("Weekly", "W"),
                            ("Monthly", "M"),
                            ("Quarterly", "Q")
                        ],
                        format_func=lambda x: x[0],
                        index=1  # Default to weekly
                    )[1]
                    
                    # Allow user to select fields to view
                    selected_fields = st.multiselect(
                        "Select Fields to View",
                        options=existing_fields,
                        default=existing_fields[:5]  # Default to first 5 fields
                    )
                    
                    if selected_fields:
                        self.render_time_series_completeness(
                            data,
                            date_field,
                            selected_fields,
                            time_interval=time_interval,
                            title=f"Data Completeness Over Time ({time_interval})"
                        )
                    else:
                        st.info("Please select at least one field to view time series data.")
                else:
                    st.info("No valid date field provided for time series analysis.")
            
            # Add recommendations section
            st.markdown("## 🔍 Data Quality Recommendations")
            
            # Generate recommendations based on missing data
            recommendations = []
            
            # Get fields with high missing data
            field_missing_pcts = {
                field: (data[field].isna().sum() / len(data)) * 100
                for field in existing_fields
            }
            
            high_missing_fields = {
                field: pct for field, pct in field_missing_pcts.items() if pct > 30
            }
            
            if high_missing_fields:
                field_list = ", ".join([f"`{field}` ({pct:.1f}%)" for field, pct in high_missing_fields.items()])
                recommendations.append(
                    f"🔴 **Critical:** The following fields have >30% missing data: {field_list}. "
                    "Consider implementing fallback data sources or imputation strategies."
                )
            
            # Check for time-based patterns
            if date_field and date_field in data.columns:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(data[date_field]):
                    try:
                        date_series = pd.to_datetime(data[date_field])
                    except:
                        date_series = None
                else:
                    date_series = data[date_field]
                
                if date_series is not None:
                    # Check for recent data quality issues
                    recent_cutoff = date_series.max() - pd.Timedelta(days=30)
                    recent_data = data[date_series >= recent_cutoff]
                    
                    if not recent_data.empty:
                        recent_missing_pct = recent_data[existing_fields].isna().mean().mean() * 100
                        overall_missing_pct = data[existing_fields].isna().mean().mean() * 100
                        
                        if recent_missing_pct > overall_missing_pct * 1.5:
                            recommendations.append(
                                f"🟠 **Warning:** Recent data (last 30 days) has {recent_missing_pct:.1f}% missing values, "
                                f"which is significantly higher than the overall average of {overall_missing_pct:.1f}%. "
                                "Check for recent data source failures."
                            )
            
            # Check for specific data groups with issues
            for group, fields in filtered_fields.items():
                if fields:  # Only process groups with existing fields
                    all_complete = data[fields].notna().all(axis=1).sum()
                    total = len(data)
                    complete_pct = (all_complete / total) * 100 if total > 0 else 0
                    
                    if complete_pct < 50:
                        recommendations.append(
                            f"🟡 **Issue:** The '{group}' data group has only {complete_pct:.1f}% of records with complete data. "
                            "This may impact features and models that rely on this data group."
                        )
            
            # Add specific recommendation for weather data if it exists
            weather_fields = [f for f in existing_fields if "weather" in f.lower() or "temperature" in f.lower()]
            
            if weather_fields and any(field_missing_pcts.get(field, 0) > 20 for field in weather_fields):
                recommendations.append(
                    f"🔵 **Insight:** Weather-related fields ({', '.join(weather_fields)}) show significant missing data. "
                    "This may be due to missing coordinates in match_info. Consider implementing coordinate fallbacks "
                    "or alternative weather data sources."
                )
            
            # Display recommendations
            if recommendations:
                for i, recommendation in enumerate(recommendations):
                    st.markdown(recommendation)
            else:
                st.success("✅ No significant data quality issues detected. Data completeness is good.")
            
            # Add action items section
            with st.expander("📋 Recommended Action Items"):
                st.markdown("""
                1. **Implement data source fallbacks** for critical fields with high missing rates
                2. **Create data quality monitoring** alerts for sudden drops in data completeness
                3. **Develop imputation strategies** for fields where missing data can't be avoided
                4. **Document data dependencies** between field groups to better understand impacts
                5. **Review data pipeline** for potential points of failure in data collection
                """)
                
        except Exception as e:
            st.error(f"Failed to render missing data dashboard: {str(e)}")
            error_log.error(
                "Failed to render missing data dashboard",
                exception=e,
                err_type="dashboard_error"
            )


# Helper function to create the visualizer with current theme
def get_data_integrity_visualizer() -> DataIntegrityVisualizer:
    """Get a data integrity visualizer instance with the current theme."""
    theme = st.session_state.get("theme", "light")
    return DataIntegrityVisualizer(theme=theme)
