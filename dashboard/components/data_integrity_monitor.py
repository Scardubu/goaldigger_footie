"""
Data integrity monitoring component for the dashboard.
Provides visualization and tracking of data completeness, scraping success, and database health.
"""
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from database.db_manager import DatabaseManager
from dashboard.error_log import log_error
from utils.config import Config

logger = logging.getLogger(__name__)

def get_scraped_data_stats(db_manager: DatabaseManager, days_lookback: int = 30) -> Dict[str, Any]:
    """
    Get statistics on scraped data completeness and quality.
    
    Args:
        db_manager: Database manager instance
        days_lookback: Number of days to look back for statistics
        
    Returns:
        Dictionary with statistics on scraped data
    """
    # Initialize default values for fallback
    default_stats = {
        "total_matches": 0,
        "scraped_matches": 0,
        "completion_rate": 0,
        "column_stats": {},
        "daily_stats": [],
        "error": None,
        "status": "error"  # Default status
    }
    
    max_retries = 3
    retry_count = 0
    retry_delay = 1  # Initial delay in seconds
    
    while retry_count < max_retries:
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_lookback)
            
            # Use database manager session instead of raw SQL
            with db_manager.session_scope() as session:
                from database.schema import Match
                from sqlalchemy import func
                
                # Get match counts
                total_matches = session.query(func.count(Match.id)).filter(
                    Match.match_date.between(start_date, end_date)
                ).scalar() or 0
                
                # Get scraped data counts (simplified for now)
                scraped_matches = 0  # Will be implemented when scraped_data table is properly set up
                
                # Calculate completion rate
                completion_rate = (scraped_matches / total_matches) * 100 if total_matches > 0 else 0
                
                # Simplified column stats for now
                column_stats = {
                    "fixture_details": {"count": 0, "rate": 0},
                    "odds": {"count": 0, "rate": 0},
                    "stats": {"count": 0, "rate": 0},
                    "lineups": {"count": 0, "rate": 0},
                    "form": {"count": 0, "rate": 0},
                    "h2h": {"count": 0, "rate": 0},
                    "injuries": {"count": 0, "rate": 0}
                }
                
                # Simplified daily stats
                daily_stats = []
                
                # Successful result
                stats = {
                    "total_matches": total_matches,
                    "scraped_matches": scraped_matches,
                    "completion_rate": completion_rate,
                    "column_stats": column_stats,
                    "daily_stats": daily_stats,
                    "error": None,
                    "status": "success"
                }
                
                # Log successful completion
                critical_threshold = 20
                warning_threshold = 5
                
                # Check for critical data issues
                has_critical_issues = False
                missing_rate = 100 - completion_rate
                
                if missing_rate > critical_threshold:
                    logger.warning(f"CRITICAL: High missing data rate detected: {missing_rate:.1f}% of matches missing scraped data")
                    has_critical_issues = True
                    stats["alert"] = {
                        "level": "critical",
                        "message": f"High missing data rate detected: {missing_rate:.1f}% of matches missing scraped data"
                    }
                elif missing_rate > warning_threshold:
                    logger.info(f"WARNING: Elevated missing data rate: {missing_rate:.1f}% of matches missing scraped data")
                    stats["alert"] = {
                        "level": "warning",
                        "message": f"Elevated missing data rate: {missing_rate:.1f}% of matches missing scraped data"
                    }
                else:
                    logger.info(f"Successfully retrieved scraped data stats: {total_matches} matches, {completion_rate:.1f}% completion rate")
                
                return stats
                
        except Exception as e:
            error_msg = f"Error during DB query attempt {retry_count+1}/{max_retries}: {str(e)}"
            logger.warning(error_msg)
            retry_count += 1
            
            if retry_count < max_retries:
                # Implement exponential backoff
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Final failure after all retries
                logger.error(f"Failed to get scraped data stats after {max_retries} attempts: {str(e)}")
                # Add error to the response to display in UI
                default_stats["error"] = str(e)
                # Log detailed error with stack trace
                log_error("Failed to retrieve scraped data stats", e)
                break
                
    return default_stats

def get_weather_data_stats(db_manager: DatabaseManager, days_lookback: int = 30) -> Dict[str, Any]:
    """
    Get statistics on weather data completeness and quality.
    
    Args:
        db_manager: Database manager instance
        days_lookback: Number of days to look back for statistics
        
    Returns:
        Dictionary with statistics on weather data
    """
    # Initialize default values for fallback
    default_stats = {
        "total_matches": 0,
        "matches_with_coords": 0,
        "matches_with_weather": 0,
        "completion_rate": 0,
        "weather_quality": {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0
        },
        "daily_stats": [],
        "error": None,
        "status": "error"  # Default status
    }
    
    max_retries = 3
    retry_count = 0
    retry_delay = 1
    last_exception = None  # Initialize exception variable
    
    while retry_count < max_retries:
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_lookback)
            
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get total match counts - Fix SQL parameter issue
                cursor.execute(
                    """
                    SELECT COUNT(*) AS total_matches
                    FROM matches
                    WHERE match_date BETWEEN ? AND ?
                    """,
                    ([start_date.isoformat(), end_date.isoformat()])
                )
                result = cursor.fetchone()
                total_matches = result[0] if result else 0
                
                # Get match counts with coordinates - Fix SQL parameter issue
                cursor.execute(
                    """
                    SELECT COUNT(*) AS matches_with_coords
                    FROM matches
                    WHERE match_date BETWEEN ? AND ?
                    AND (latitude IS NOT NULL AND longitude IS NOT NULL)
                    """,
                    ([start_date.isoformat(), end_date.isoformat()])
                )
                result = cursor.fetchone()
                matches_with_coords = result[0] if result else 0
                
                # Get weather data counts - Fix SQL parameter issue
                cursor.execute(
                    """
                    SELECT COUNT(*) AS weather_matches
                    FROM weather_data
                    INNER JOIN matches ON weather_data.match_id = matches.id
                    WHERE matches.match_date BETWEEN ? AND ?
                    """,
                    ([start_date.isoformat(), end_date.isoformat()])
                )
                result = cursor.fetchone()
                matches_with_weather = result[0] if result else 0
                
                # Calculate completion rate
                completion_rate = (matches_with_weather / matches_with_coords) * 100 if matches_with_coords > 0 else 0
                coords_rate = (matches_with_coords / total_matches) * 100 if total_matches > 0 else 0
                
                # Get weather quality metrics - Fix SQL parameter issue
                # Assuming there's some quality measure in the weather_data table
                # This is a placeholder - adapt to your actual schema
                cursor.execute(
                    """
                    SELECT 
                        SUM(CASE WHEN temperature IS NOT NULL AND humidity IS NOT NULL AND wind_speed IS NOT NULL THEN 1 ELSE 0 END) AS excellent,
                        SUM(CASE WHEN temperature IS NOT NULL AND (humidity IS NULL OR wind_speed IS NULL) THEN 1 ELSE 0 END) AS good,
                        SUM(CASE WHEN temperature IS NULL AND (humidity IS NOT NULL OR wind_speed IS NOT NULL) THEN 1 ELSE 0 END) AS fair,
                        SUM(CASE WHEN data_quality_score < 50 OR (temperature IS NULL AND humidity IS NULL AND wind_speed IS NULL) THEN 1 ELSE 0 END) AS poor
                    FROM weather_data
                    INNER JOIN matches ON weather_data.match_id = matches.id
                    WHERE matches.match_date BETWEEN ? AND ?
                    """,
                    ([start_date.isoformat(), end_date.isoformat()])
                )
                
                result = cursor.fetchone()
                if result:
                    excellent, good, fair, poor = result
                else:
                    excellent, good, fair, poor = 0, 0, 0, 0
                
                # Get daily weather data statistics - Fix SQL parameter issue
                cursor.execute(
                    """
                    SELECT 
                        DATE(matches.match_date) AS date,
                        COUNT(weather_data.id) AS weather_count,
                        COUNT(matches.id) AS match_count
                    FROM matches
                    LEFT JOIN weather_data ON matches.id = weather_data.match_id
                    WHERE matches.match_date BETWEEN ? AND ?
                    AND (matches.latitude IS NOT NULL AND matches.longitude IS NOT NULL)
                    GROUP BY DATE(matches.match_date)
                    ORDER BY DATE(matches.match_date)
                    """,
                    ([start_date.isoformat(), end_date.isoformat()])
                )
                
                daily_stats = []
                for row in cursor.fetchall():
                    date, weather_count, match_count = row
                    day_completion_rate = (weather_count / match_count) * 100 if match_count > 0 else 0
                    day_stats = {
                        "date": date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date),
                        "weather_count": weather_count,
                        "match_count": match_count,
                        "completion_rate": day_completion_rate
                    }
                    daily_stats.append(day_stats)
                
                # Prepare successful result
                stats = {
                    "total_matches": total_matches,
                    "matches_with_coords": matches_with_coords,
                    "matches_with_weather": matches_with_weather,
                    "coords_rate": coords_rate,
                    "completion_rate": completion_rate,
                    "weather_quality": {
                        "excellent": excellent if excellent else 0,
                        "good": good if good else 0,
                        "fair": fair if fair else 0,
                        "poor": poor if poor else 0
                    },
                    "daily_stats": daily_stats,
                    "error": None,
                    "status": "success"
                }
                
                # Check for critical issues
                critical_threshold = 20
                warning_threshold = 5
                
                missing_coords_rate = 100 - coords_rate
                missing_weather_rate = 100 - completion_rate
                
                if missing_coords_rate > critical_threshold:
                    logger.warning(f"CRITICAL: High missing coordinates rate: {missing_coords_rate:.1f}% of matches missing coordinates")
                    stats["alert"] = {
                        "level": "critical",
                        "message": f"High missing coordinates rate: {missing_coords_rate:.1f}% of matches missing coordinates"
                    }
                elif missing_weather_rate > critical_threshold:
                    logger.warning(f"CRITICAL: High missing weather data rate: {missing_weather_rate:.1f}% of matches missing weather data")
                    stats["alert"] = {
                        "level": "critical",
                        "message": f"High missing weather data rate: {missing_weather_rate:.1f}% of matches missing weather data"
                    }
                elif missing_coords_rate > warning_threshold or missing_weather_rate > warning_threshold:
                    logger.info(f"WARNING: Elevated missing data rates - Coords: {missing_coords_rate:.1f}%, Weather: {missing_weather_rate:.1f}%")
                    stats["alert"] = {
                        "level": "warning",
                        "message": f"Elevated missing data rates - Coords: {missing_coords_rate:.1f}%, Weather: {missing_weather_rate:.1f}%"
                    }
                else:
                    logger.info(f"Successfully retrieved weather data stats: {matches_with_weather}/{matches_with_coords} matches have weather data ({completion_rate:.1f}%)")
                
                return stats
                
        except Exception as e:
            last_exception = e  # Store the exception
            logger.warning(f"Error during weather stats DB query attempt {retry_count + 1}/{max_retries}: {e}")
            retry_count += 1
            if retry_count < max_retries:
                logger.info(f"Retrying weather stats in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
    
    # Use the stored exception or a default message
    error_message = str(last_exception) if last_exception else "Unknown error after all retries"
    logger.error(f"Failed to get weather data stats after {max_retries} attempts: {error_message}")
    default_stats["error"] = error_message
    return default_stats

def analyze_data_gaps(db_manager: DatabaseManager, days_lookback: int = 30) -> Dict[str, Any]:
    """
    Analyze gaps in the scraped data to identify patterns.
    
    Args:
        db_manager: Database manager instance
        days_lookback: Number of days to look back for analysis
        
    Returns:
        Dictionary with gap analysis results
    """
    # Initialize default values for fallback
    default_analysis = {
        "league_stats": [],
        "weekday_stats": [],
        "time_of_day_stats": [],
        "summary": {
            "total_matches": 0,
            "scraped_matches": 0,
            "missing_rate": 0,
            "worst_leagues": [],
            "worst_days": []
        },
        "error": None,
        "status": "error"  # Default status
    }
    
    # Set up retry mechanism
    max_retries = 3
    retry_count = 0
    retry_delay = 1  # Initial delay in seconds
    
    while retry_count < max_retries:
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_lookback)
            
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get overall summary stats first
                cursor.execute(
                    """
                    SELECT 
                        COUNT(matches.id) AS total_matches,
                        COUNT(scraped_data.match_id) AS scraped_matches
                    FROM matches
                    LEFT JOIN scraped_data ON matches.id = scraped_data.match_id
                    WHERE matches.match_date BETWEEN ? AND ?
                    """,
                    ([start_date.isoformat(), end_date.isoformat()])
                )
                
                result = cursor.fetchone()
                if result:
                    total_matches, scraped_matches = result
                    missing_matches = total_matches - scraped_matches
                    overall_missing_rate = (missing_matches / total_matches) * 100 if total_matches > 0 else 0
                else:
                    total_matches, scraped_matches = 0, 0
                    missing_matches = 0
                    overall_missing_rate = 0
                
                # Analyze gaps by league
                cursor.execute(
                    """
                    SELECT 
                        matches.competition AS league,
                        COUNT(matches.id) AS total_matches,
                        COUNT(scraped_data.match_id) AS scraped_matches
                    FROM matches
                    LEFT JOIN scraped_data ON matches.id = scraped_data.match_id
                    WHERE matches.match_date BETWEEN ? AND ?
                    GROUP BY matches.competition
                    ORDER BY (COUNT(matches.id) - COUNT(scraped_data.match_id)) DESC
                    """,
                    ([start_date.isoformat(), end_date.isoformat()])
                )
                
                league_stats = []
                worst_leagues = []
                critical_threshold = 20  # 20% missing data is critical
                
                for row in cursor.fetchall():
                    league, total, scraped = row
                    missing = total - scraped
                    missing_rate = (missing / total) * 100 if total > 0 else 0
                    
                    league_stat = {
                        "league": league,
                        "total_matches": total,
                        "scraped_matches": scraped,
                        "missing_matches": missing,
                        "missing_rate": missing_rate
                    }
                    
                    league_stats.append(league_stat)
                    
                    # Track leagues with critical missing data
                    if missing_rate > critical_threshold and total >= 5:  # Only consider leagues with at least 5 matches
                        worst_leagues.append({
                            "league": league,
                            "missing_rate": missing_rate,
                            "missing_matches": missing
                        })
                
                # Sort worst leagues by missing rate
                worst_leagues = sorted(worst_leagues, key=lambda x: x["missing_rate"], reverse=True)[:5]  # Top 5 worst
                
                # Analyze gaps by day of week
                cursor.execute(
                    """
                    SELECT 
                        strftime('%w', matches.match_date) AS weekday,
                        COUNT(matches.id) AS total_matches,
                        COUNT(scraped_data.match_id) AS scraped_matches
                    FROM matches
                    LEFT JOIN scraped_data ON matches.id = scraped_data.match_id
                    WHERE matches.match_date BETWEEN ? AND ?
                    GROUP BY weekday
                    ORDER BY weekday
                    """,
                    ([start_date.isoformat(), end_date.isoformat()])
                )
                
                weekday_stats = []
                worst_days = []
                
                for row in cursor.fetchall():
                    weekday_num, total, scraped = row
                    try:
                        weekday_num = int(weekday_num)
                    except (ValueError, TypeError):
                        weekday_num = 0  # Default to Sunday if conversion fails
                    
                    missing = total - scraped
                    missing_rate = (missing / total) * 100 if total > 0 else 0
                    
                    # Convert weekday number to name
                    weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                    weekday_name = weekday_names[weekday_num % 7]  # Ensure index is within range
                    
                    weekday_stat = {
                        "weekday_num": weekday_num,
                        "weekday_name": weekday_name,
                        "total_matches": total,
                        "scraped_matches": scraped,
                        "missing_matches": missing,
                        "missing_rate": missing_rate
                    }
                    
                    weekday_stats.append(weekday_stat)
                    
                    # Track days with critical missing data
                    if missing_rate > critical_threshold and total >= 5:  # Only consider days with at least 5 matches
                        worst_days.append({
                            "day": weekday_name,
                            "missing_rate": missing_rate,
                            "missing_matches": missing
                        })
                
                # Sort worst days by missing rate
                worst_days = sorted(worst_days, key=lambda x: x["missing_rate"], reverse=True)[:3]  # Top 3 worst
                
                # Analyze gaps by time of day (morning, afternoon, evening, night)
                cursor.execute(
                    """
                    SELECT 
                        CASE 
                            WHEN strftime('%H', matches.match_date) BETWEEN '06' AND '11' THEN 'Morning'
                            WHEN strftime('%H', matches.match_date) BETWEEN '12' AND '16' THEN 'Afternoon'
                            WHEN strftime('%H', matches.match_date) BETWEEN '17' AND '20' THEN 'Evening'
                            ELSE 'Night'
                        END AS time_of_day,
                        COUNT(matches.id) AS total_matches,
                        COUNT(scraped_data.match_id) AS scraped_matches
                    FROM matches
                    LEFT JOIN scraped_data ON matches.id = scraped_data.match_id
                    WHERE matches.match_date BETWEEN ? AND ?
                    GROUP BY time_of_day
                    ORDER BY time_of_day
                    """,
                    ([start_date.isoformat(), end_date.isoformat()])
                )
                
                time_of_day_stats = []
                
                for row in cursor.fetchall():
                    time_of_day, total, scraped = row
                    missing = total - scraped
                    missing_rate = (missing / total) * 100 if total > 0 else 0
                    
                    time_of_day_stats.append({
                        "time_of_day": time_of_day,
                        "total_matches": total,
                        "scraped_matches": scraped,
                        "missing_matches": missing,
                        "missing_rate": missing_rate
                    })
                
                # Complete result with analysis summary
                result = {
                    "league_stats": league_stats,
                    "weekday_stats": weekday_stats,
                    "time_of_day_stats": time_of_day_stats,
                    "summary": {
                        "total_matches": total_matches,
                        "scraped_matches": scraped_matches,
                        "missing_matches": missing_matches,
                        "missing_rate": overall_missing_rate,
                        "worst_leagues": worst_leagues,
                        "worst_days": worst_days,
                        "analyzed_at": datetime.now().isoformat(),
                        "period": f"{days_lookback} days"
                    },
                    "error": None,
                    "status": "success"
                }
                
                # Log insights from analysis
                if worst_leagues:
                    worst_league = worst_leagues[0]["league"]
                    worst_rate = worst_leagues[0]["missing_rate"]
                    logger.warning(f"Data gap analysis: {worst_league} has the highest missing data rate at {worst_rate:.1f}%")
                
                if worst_days:
                    worst_day = worst_days[0]["day"]
                    worst_day_rate = worst_days[0]["missing_rate"]
                    logger.warning(f"Data gap analysis: {worst_day} has the highest missing data rate at {worst_day_rate:.1f}%")
                
                logger.info(f"Data gap analysis complete: {missing_matches} of {total_matches} matches missing data ({overall_missing_rate:.1f}%)")
                
                return result
                
        except Exception as e:
            error_msg = f"Error during data gap analysis attempt {retry_count+1}/{max_retries}: {str(e)}"
            logger.warning(error_msg)
            retry_count += 1
            
            if retry_count < max_retries:
                # Implement exponential backoff
                logger.info(f"Retrying data gap analysis in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # Final failure after all retries
                logger.error(f"Failed to analyze data gaps after {max_retries} attempts: {str(e)}")
                # Add error to the response
                default_analysis["error"] = str(e)
                # Log detailed error with stack trace
                log_error("Failed to analyze data gaps", e)
                break
    
    return default_analysis


def render_data_integrity_dashboard(days_lookback: int = 30) -> None:
    """
    Render the data integrity dashboard component.
    
    Args:
        days_lookback: Number of days to look back for analysis
    """
    try:
        st.header("Data Integrity Monitor")
        
        # Add lookback period selector
        col1, col2 = st.columns([3, 1])
        with col2:
            days_lookback = st.selectbox(
                "Analysis Period",
                options=[7, 14, 30, 60, 90],
                index=2,  # Default to 30 days
                format_func=lambda x: f"{x} days"
            )
        
        with col1:
            st.write(f"Analyzing data integrity for the last {days_lookback} days (from {(datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')})")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Get statistics
        scraped_stats = get_scraped_data_stats(db_manager, days_lookback)
        weather_stats = get_weather_data_stats(db_manager, days_lookback)
        gap_analysis = analyze_data_gaps(db_manager, days_lookback)
        
        # Create tabs for different sections
        tabs = st.tabs(["Overview", "Data Completeness", "Weather Data", "Gap Analysis"])
        
        # --- Tab 1: Overview ---
        with tabs[0]:
            st.subheader("Data Completeness Overview")
            
            # Create metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                completion_color = "green" if scraped_stats["completion_rate"] >= 95 else "orange" if scraped_stats["completion_rate"] >= 80 else "red"
                st.metric(
                    "Overall Data Completeness",
                    f"{scraped_stats['completion_rate']:.1f}%",
                    f"{scraped_stats['scraped_matches']} of {scraped_stats['total_matches']} matches"
                )
                st.markdown(f"<div style='text-align:center;color:{completion_color};font-size:24px;'>{'●' if scraped_stats['completion_rate'] >= 95 else '▲' if scraped_stats['completion_rate'] >= 80 else '■'}</div>", unsafe_allow_html=True)
                
            with col2:
                weather_color = "green" if weather_stats["completion_rate"] >= 90 else "orange" if weather_stats["completion_rate"] >= 70 else "red"
                st.metric(
                    "Weather Data Completeness",
                    f"{weather_stats['completion_rate']:.1f}%",
                    f"{weather_stats['matches_with_weather']} of {weather_stats['matches_with_coords']} matches"
                )
                st.markdown(f"<div style='text-align:center;color:{weather_color};font-size:24px;'>{'●' if weather_stats['completion_rate'] >= 90 else '▲' if weather_stats['completion_rate'] >= 70 else '■'}</div>", unsafe_allow_html=True)
                
            with col3:
                # Most problematic league
                if gap_analysis.get("league_stats"):
                    problem_leagues = sorted(gap_analysis["league_stats"], key=lambda x: x["missing_rate"], reverse=True)
                    if problem_leagues:
                        worst_league = problem_leagues[0]
                        league_color = "green" if worst_league["missing_rate"] <= 5 else "orange" if worst_league["missing_rate"] <= 20 else "red"
                        st.metric(
                            "Most Problematic League",
                            worst_league["league"],
                            f"Missing {worst_league['missing_rate']:.1f}% of matches"
                        )
                        st.markdown(f"<div style='text-align:center;color:{league_color};font-size:24px;'>{'●' if worst_league['missing_rate'] <= 5 else '▲' if worst_league['missing_rate'] <= 20 else '■'}</div>", unsafe_allow_html=True)
                    else:
                        st.metric("Most Problematic League", "N/A", "No data available")
                else:
                    st.metric("Most Problematic League", "N/A", "No league data available")
            
            # Completeness trend over time
            if scraped_stats["daily_stats"]:
                st.subheader("Data Completeness Trend")
                
                # Convert to DataFrame for plotting
                daily_df = pd.DataFrame(scraped_stats["daily_stats"])
                
                # Create the trend chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=daily_df["date"],
                    y=daily_df["completion_rate"],
                    mode="lines+markers",
                    name="Completion Rate (%)",
                    line=dict(width=3, color="#1f77b4"),
                    marker=dict(size=8)
                ))
                
                # Add reference line at 95%
                fig.add_shape(
                    type="line",
                    x0=daily_df["date"].min(),
                    y0=95,
                    x1=daily_df["date"].max(),
                    y1=95,
                    line=dict(color="green", width=2, dash="dash")
                )
                
                # Add annotation for the reference line
                fig.add_annotation(
                    x=daily_df["date"].max(),
                    y=95,
                    text="Target (95%)",
                    showarrow=False,
                    yshift=10,
                    font=dict(color="green")
                )
                
                fig.update_layout(
                    title="Data Completeness Trend",
                    xaxis_title="Date",
                    yaxis_title="Completion Rate (%)",
                    yaxis=dict(range=[0, 105]),
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        # --- Tab 2: Data Completeness by Column ---
        with tabs[1]:
            st.subheader("Data Completeness by Data Type")
            
            if scraped_stats["column_stats"]:
                # Convert column stats to DataFrame
                columns_df = pd.DataFrame([
                    {
                        "column": column,
                        "count": stats["count"],
                        "rate": stats["rate"]
                    }
                    for column, stats in scraped_stats["column_stats"].items()
                ])
                
                # Sort by completion rate
                columns_df = columns_df.sort_values("rate", ascending=False)
                
                # Create the bar chart
                fig = go.Figure()
                
                # Add bars for completion rate
                fig.add_trace(go.Bar(
                    x=columns_df["column"],
                    y=columns_df["rate"],
                    text=[f"{rate:.1f}%" for rate in columns_df["rate"]],
                    textposition="auto",
                    marker_color=[
                        "green" if rate >= 95 else "orange" if rate >= 80 else "red"
                        for rate in columns_df["rate"]
                    ],
                    name="Completion Rate (%)"
                ))
                
                fig.update_layout(
                    title="Data Completeness by Type",
                    xaxis_title="Data Type",
                    yaxis_title="Completion Rate (%)",
                    yaxis=dict(range=[0, 105]),
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table with detailed stats
                st.subheader("Detailed Data Type Statistics")
                
                # Add match count context
                columns_df["available_matches"] = columns_df["count"]
                columns_df["total_matches"] = scraped_stats["scraped_matches"]
                columns_df["missing_matches"] = columns_df["total_matches"] - columns_df["available_matches"]
                
                # Reorder and rename columns for display
                display_df = columns_df[[
                    "column", "available_matches", "missing_matches", 
                    "total_matches", "rate"
                ]].rename(columns={
                    "column": "Data Type",
                    "available_matches": "Available",
                    "missing_matches": "Missing",
                    "total_matches": "Total",
                    "rate": "Completion Rate (%)"
                })
                
                # Format the completion rate
                display_df["Completion Rate (%)"] = display_df["Completion Rate (%)"].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(display_df, use_container_width=True)
            
            else:
                st.info("No column statistics available.")
                
        # --- Tab 3: Weather Data ---
        with tabs[2]:
            st.subheader("Weather Data Analysis")
            
            # Create metrics
            col1, col2 = st.columns(2)
            
            with col1:
                weather_color = "green" if weather_stats["completion_rate"] >= 90 else "orange" if weather_stats["completion_rate"] >= 70 else "red"
                st.metric(
                    "Weather Data Completeness",
                    f"{weather_stats['completion_rate']:.1f}%",
                    f"{weather_stats['matches_with_weather']} of {weather_stats['matches_with_coords']} matches"
                )
                st.markdown(f"<div style='text-align:center;color:{weather_color};font-size:24px;'>{'●' if weather_stats['completion_rate'] >= 90 else '▲' if weather_stats['completion_rate'] >= 70 else '■'}</div>", unsafe_allow_html=True)
                
            with col2:
                missing_weather = weather_stats["matches_with_coords"] - weather_stats["matches_with_weather"]
                st.metric(
                    "Matches Missing Weather Data",
                    missing_weather,
                    f"{100 - weather_stats['completion_rate']:.1f}% of matches"
                )
                
            # Info box about validation settings
            with st.expander("Weather Data Validation Settings", expanded=True):
                strict_validation = Config.get("preprocessing.validation.strict_weather_validation", False)
                allow_missing = Config.get("preprocessing.validation.allow_missing_weather", True)
                warning_threshold = Config.get("preprocessing.validation.weather_warning_threshold", 0.3)
                
                status_icon = "✅" if not strict_validation and allow_missing else "⚠️"
                
                st.markdown(f"""
                **Current Weather Validation Configuration** {status_icon}
                
                - **Strict Validation**: {'Enabled' if strict_validation else 'Disabled'}
                - **Allow Missing Weather**: {'Yes' if allow_missing else 'No'}
                - **Warning Threshold**: {warning_threshold * 100}% missing data
                
                *These settings can be adjusted in the configuration file.*
                """)
                
                st.markdown("""
                **Impact on Feature Generation:**
                
                With relaxed validation (current setting), the feature generator can proceed
                even with missing weather data, providing more complete prediction coverage.
                Weather features will default to `NaN` when data is unavailable.
                """)
        
        # --- Tab 4: Gap Analysis ---
        with tabs[3]:
            st.subheader("Data Completeness Gap Analysis")
            
            col1, col2 = st.columns(2)
            
            # League-specific gap analysis
            with col1:
                st.subheader("By League/Competition")
                
                if gap_analysis["league_stats"]:
                    # Convert to DataFrame for plotting
                    leagues_df = pd.DataFrame(gap_analysis["league_stats"])
                    
                    # Sort by missing rate descending
                    leagues_df = leagues_df.sort_values("missing_rate", ascending=False)
                    
                    # Take top 10 for visualization
                    top_leagues_df = leagues_df.head(10)
                    
                    # Create the horizontal bar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        y=top_leagues_df["league"],
                        x=top_leagues_df["missing_rate"],
                        orientation="h",
                        text=[f"{rate:.1f}%" for rate in top_leagues_df["missing_rate"]],
                        textposition="auto",
                        marker_color=[
                            "red" if rate > 20 else "orange" if rate > 5 else "green"
                            for rate in top_leagues_df["missing_rate"]
                        ],
                        name="Missing Rate (%)"
                    ))
                    
                    fig.update_layout(
                        title="Missing Data by League",
                        xaxis_title="Missing Rate (%)",
                        yaxis_title="League",
                        height=400,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No league statistics available.")
            
            # Day of week gap analysis
            with col2:
                st.subheader("By Day of Week")
                
                if gap_analysis["weekday_stats"]:
                    # Convert to DataFrame for plotting
                    weekday_df = pd.DataFrame(gap_analysis["weekday_stats"])
                    
                    # Sort by weekday number for chronological order
                    weekday_df = weekday_df.sort_values("weekday_num")
                    
                    # Create the bar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=weekday_df["weekday_name"],
                        y=weekday_df["missing_rate"],
                        text=[f"{rate:.1f}%" for rate in weekday_df["missing_rate"]],
                        textposition="auto",
                        marker_color=[
                            "red" if rate > 20 else "orange" if rate > 5 else "green"
                            for rate in weekday_df["missing_rate"]
                        ],
                        name="Missing Rate (%)"
                    ))
                    
                    fig.update_layout(
                        title="Missing Data by Day of Week",
                        xaxis_title="Day of Week",
                        yaxis_title="Missing Rate (%)",
                        height=400,
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No weekday statistics available.")
            
            # Action buttons for data backfill
            st.subheader("Data Recovery Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Identify Missing Data", use_container_width=True):
                    st.info("Executing data identification process...")
                    # In a real implementation, this would call a background process to identify missing data
                    # For now, we'll provide a message directing to use the script
                    st.success("Use the following command to identify missing data:\n\n```\npython scripts/data_backfill.py --mode identify\n```")
            
            with col2:
                if st.button("Start Data Backfill", use_container_width=True):
                    st.info("Preparing data backfill process...")
                    # In a real implementation, this would call a background process to start the backfill
                    # For now, we'll provide a message directing to use the script
                    st.success("Use the following commands to backfill missing data:\n\n```\npython scripts/data_backfill.py --mode backfill\n```\n\nThen follow the instructions to execute MCP requests and process the results.")
                    
    except Exception as e:
        log_error("Error rendering data integrity dashboard", e)
        st.error(f"An error occurred while rendering the data integrity dashboard: {str(e)}")
