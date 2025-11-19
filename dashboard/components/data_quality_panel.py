#!/usr/bin/env python3
"""
Data Quality Panel - Display real-time data quality metrics
Shows validation status, freshness indicators, and source health
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)


def render_data_quality_panel(
    validation_report: Optional[Dict[str, Any]] = None,
    show_details: bool = False
):
    """
    Render data quality status panel.
    
    Args:
        validation_report: Validation report from RealDataValidator
        show_details: Whether to show detailed metrics
    """
    if not validation_report:
        st.info("📊 Data quality metrics will appear here once predictions are generated")
        return
    
    quality_score = validation_report.get('quality_score', 0.0)
    recommendation = validation_report.get('recommendation', 'UNKNOWN')
    is_valid = validation_report.get('valid', False)
    
    # Header with quality score
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 📊 Data Quality Status")
    
    with col2:
        # Quality badge
        if quality_score >= 0.8:
            st.success(f"**{quality_score:.0%}** Quality")
        elif quality_score >= 0.5:
            st.warning(f"**{quality_score:.0%}** Quality")
        else:
            st.error(f"**{quality_score:.0%}** Quality")
    
    with col3:
        # Validation status
        if is_valid:
            st.success("✅ **Valid**")
        else:
            st.error("🚫 **Invalid**")
    
    # Recommendation banner
    if recommendation == 'PUBLISH_ALLOWED':
        st.success("✅ **High quality prediction** - Publication allowed")
    elif recommendation == 'PUBLISH_WITH_WARNING':
        st.warning("⚠️ **Acceptable quality** - Publish with caution")
    else:
        st.error("🚫 **Insufficient quality** - Publication blocked")
    
    # Show details if requested
    if show_details:
        with st.expander("🔍 Detailed Quality Metrics", expanded=False):
            
            # Feature coverage
            st.markdown("#### Feature Coverage")
            feature_coverage = validation_report.get('checks', {}).get('feature_coverage', {})
            
            if feature_coverage:
                cov_cols = st.columns(4)
                
                with cov_cols[0]:
                    if feature_coverage.get('historic_form'):
                        st.success("✅ Form Data")
                    else:
                        st.error("❌ Form Data")
                
                with cov_cols[1]:
                    if feature_coverage.get('head_to_head'):
                        st.success("✅ H2H Data")
                    else:
                        st.error("❌ H2H Data")
                
                with cov_cols[2]:
                    if feature_coverage.get('league_table'):
                        st.success("✅ Standings")
                    else:
                        st.error("❌ Standings")
                
                with cov_cols[3]:
                    if feature_coverage.get('xg'):
                        st.success("✅ xG Data")
                    else:
                        st.error("❌ xG Data")
            
            # Freshness check
            st.markdown("#### Data Freshness")
            checks = validation_report.get('checks', {})
            
            if checks.get('fixture_fresh') is not None:
                fixture_age = checks.get('fixture_age_hours', 0)
                if checks.get('fixture_fresh'):
                    st.success(f"✅ Fixture data is fresh ({fixture_age:.1f}h old)")
                else:
                    st.warning(f"⚠️ Fixture data is stale ({fixture_age:.1f}h old)")
            
            if checks.get('real_data_used'):
                st.success("✅ Using real match data")
            else:
                st.error("❌ Using fallback/synthetic data")
            
            # Source health
            st.markdown("#### Data Source Health")
            source_health = checks.get('source_health', {})
            
            if source_health:
                src_cols = st.columns(3)
                
                with src_cols[0]:
                    if source_health.get('api_available'):
                        st.success("✅ API Connected")
                    else:
                        st.error("❌ API Offline")
                
                with src_cols[1]:
                    if source_health.get('db_available'):
                        st.success("✅ Database OK")
                    else:
                        st.error("❌ Database Issue")
                
                with src_cols[2]:
                    if source_health.get('scraper_available'):
                        st.success("✅ Scraper Active")
                    else:
                        st.warning("⚠️ Scraper Limited")
            
            # Warnings and errors
            warnings = validation_report.get('warnings', [])
            errors = validation_report.get('errors', [])
            blocking_issues = validation_report.get('blocking_issues', [])
            
            if warnings:
                st.markdown("#### ⚠️ Warnings")
                for warning in warnings:
                    st.warning(warning)
            
            if errors:
                st.markdown("#### ❌ Errors")
                for error in errors:
                    st.error(error)
            
            if blocking_issues:
                st.markdown("#### 🚫 Blocking Issues")
                for issue in blocking_issues:
                    st.error(f"**{issue.replace('_', ' ').title()}**")


def render_compact_quality_indicator(quality_score: float, is_valid: bool):
    """
    Render a compact quality indicator for prediction cards.
    
    Args:
        quality_score: Quality score (0-1)
        is_valid: Whether validation passed
    """
    if quality_score >= 0.8:
        badge_color = "#28a745"
        icon = "✅"
        text = "High Quality"
    elif quality_score >= 0.5:
        badge_color = "#ffc107"
        icon = "⚠️"
        text = "Acceptable"
    else:
        badge_color = "#dc3545"
        icon = "🚫"
        text = "Low Quality"
    
    st.markdown(
        f"""
        <div style="
            background: {badge_color}22;
            border-left: 4px solid {badge_color};
            padding: 8px 12px;
            border-radius: 4px;
            margin: 8px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            <span style="font-size: 1.2em;">{icon}</span>
            <div style="flex: 1;">
                <strong>{text}</strong> ({quality_score:.0%})
                <br/>
                <small style="opacity: 0.8;">
                    {"Publication allowed" if is_valid else "Publication blocked"}
                </small>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_quality_summary_metrics(validator_stats: Dict[str, Any]):
    """
    Render summary metrics for overall validation performance.
    
    Args:
        validator_stats: Statistics from RealDataValidator.get_validation_stats()
    """
    st.markdown("### 📈 Validation Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Validations",
            validator_stats.get('total_validations', 0)
        )
    
    with col2:
        st.metric(
            "Passed",
            validator_stats.get('passed', 0),
            delta=None
        )
    
    with col3:
        st.metric(
            "Failed",
            validator_stats.get('failed', 0),
            delta=None
        )
    
    with col4:
        pass_rate = validator_stats.get('pass_rate_pct', 0)
        st.metric(
            "Pass Rate",
            f"{pass_rate:.1f}%",
            delta=f"{pass_rate - 70:.1f}%" if pass_rate != 0 else None
        )


if __name__ == "__main__":
    # Demo
    st.set_page_config(page_title="Data Quality Panel Demo", layout="wide")
    
    st.title("Data Quality Panel Demo")
    
    # Mock validation report
    mock_report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'valid': True,
        'quality_score': 0.85,
        'checks': {
            'real_data_used': True,
            'fixture_fresh': True,
            'fixture_age_hours': 2.5,
            'feature_coverage': {
                'historic_form': True,
                'head_to_head': True,
                'league_table': True,
                'xg': False
            },
            'source_health': {
                'api_available': True,
                'db_available': True,
                'scraper_available': False
            }
        },
        'warnings': [
            "xG data unavailable - using historical average"
        ],
        'errors': [],
        'blocking_issues': [],
        'recommendation': 'PUBLISH_ALLOWED'
    }
    
    # Full panel
    render_data_quality_panel(mock_report, show_details=True)
    
    st.markdown("---")
    
    # Compact indicator
    st.markdown("### Compact Indicator Example")
    render_compact_quality_indicator(0.85, True)
    
    st.markdown("---")
    
    # Summary metrics
    mock_stats = {
        'total_validations': 150,
        'passed': 120,
        'failed': 30,
        'pass_rate_pct': 80.0
    }
    render_quality_summary_metrics(mock_stats)
