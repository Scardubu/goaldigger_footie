#!/usr/bin/env python3
"""
Universal Workflow Manager for GoalDiggers Platform

Modular step-based workflow system that can be integrated across all dashboard variants
to provide consistent guided user journeys and progressive disclosure of features.

Features:
- Flexible step-based navigation with customizable workflows
- Progress indicators and visual feedback
- Step validation and transition management
- Optional workflow toggle (can be disabled for full-view mode)
- Extensible architecture for adding new workflow types
- Performance-optimized with minimal memory footprint
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

class UniversalWorkflowManager:
    """
    Universal workflow manager for GoalDiggers dashboards.
    
    Provides consistent step-based navigation, progress tracking,
    and guided user journeys across all dashboard variants.
    """
    
    def __init__(self, dashboard_type: str = "universal", workflow_enabled: bool = True):
        """
        Initialize the universal workflow manager.
        
        Args:
            dashboard_type: Type of dashboard for customized workflows
            workflow_enabled: Whether to enable step-based workflow (False for full-view mode)
        """
        self.dashboard_type = dashboard_type
        self.workflow_enabled = workflow_enabled
        self.workflows = self._initialize_workflows()
        self._initialize_session_state()
        
        logger.debug(f"✅ Universal Workflow Manager initialized for {dashboard_type}")
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state for workflow tracking."""
        # Core workflow state
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
        if 'workflow_enabled' not in st.session_state:
            st.session_state.workflow_enabled = self.workflow_enabled
        if 'selected_workflow' not in st.session_state:
            st.session_state.selected_workflow = self._get_default_workflow()
        if 'workflow_data' not in st.session_state:
            st.session_state.workflow_data = {}
        if 'step_completion_times' not in st.session_state:
            st.session_state.step_completion_times = {}
        if 'workflow_start_time' not in st.session_state:
            st.session_state.workflow_start_time = time.time()
    
    def _initialize_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Initialize workflow definitions."""
        base_workflows = {
            'prediction_analysis': {
                'name': 'Prediction Analysis',
                'description': 'Step-by-step prediction generation and analysis',
                'steps': [
                    {
                        'id': 'team_selection',
                        'title': 'Select Teams',
                        'icon': '🎯',
                        'description': 'Choose teams for analysis',
                        'validation': self._validate_team_selection,
                        'estimated_time': '1-2 min'
                    },
                    {
                        'id': 'ai_analysis',
                        'title': 'AI Analysis',
                        'icon': '🤖',
                        'description': 'AI processes data',
                        'validation': self._validate_ai_analysis,
                        'estimated_time': '30 sec'
                    },
                    {
                        'id': 'results',
                        'title': 'View Results',
                        'icon': '📊',
                        'description': 'Review predictions',
                        'validation': self._validate_results,
                        'estimated_time': '2-3 min'
                    },
                    {
                        'id': 'insights',
                        'title': 'Betting Insights',
                        'icon': '💰',
                        'description': 'Get actionable tips',
                        'validation': self._validate_insights,
                        'estimated_time': '1-2 min'
                    }
                ]
            },
            'cross_league_analysis': {
                'name': 'Cross-League Analysis',
                'description': 'Compare teams across different leagues',
                'steps': [
                    {
                        'id': 'league_selection',
                        'title': 'Select Leagues',
                        'icon': '🌍',
                        'description': 'Choose leagues to compare',
                        'validation': self._validate_league_selection,
                        'estimated_time': '1 min'
                    },
                    {
                        'id': 'team_selection',
                        'title': 'Select Teams',
                        'icon': '⚽',
                        'description': 'Pick teams from each league',
                        'validation': self._validate_cross_league_teams,
                        'estimated_time': '2 min'
                    },
                    {
                        'id': 'scenario_setup',
                        'title': 'Setup Scenario',
                        'icon': '🎮',
                        'description': 'Configure match scenario',
                        'validation': self._validate_scenario_setup,
                        'estimated_time': '1 min'
                    },
                    {
                        'id': 'analysis',
                        'title': 'Cross-League Analysis',
                        'icon': '🔍',
                        'description': 'Generate predictions',
                        'validation': self._validate_cross_league_analysis,
                        'estimated_time': '45 sec'
                    },
                    {
                        'id': 'comparison',
                        'title': 'League Comparison',
                        'icon': '📈',
                        'description': 'Compare league strengths',
                        'validation': self._validate_comparison,
                        'estimated_time': '2 min'
                    }
                ]
            },
            'premium_analysis': {
                'name': 'Premium Analysis',
                'description': 'Advanced analysis with premium features',
                'steps': [
                    {
                        'id': 'data_selection',
                        'title': 'Data Sources',
                        'icon': '📡',
                        'description': 'Select data sources',
                        'validation': self._validate_data_selection,
                        'estimated_time': '1 min'
                    },
                    {
                        'id': 'team_selection',
                        'title': 'Team Selection',
                        'icon': '🎯',
                        'description': 'Choose teams to analyze',
                        'validation': self._validate_team_selection,
                        'estimated_time': '1-2 min'
                    },
                    {
                        'id': 'premium_analysis',
                        'title': 'Premium Analysis',
                        'icon': '💎',
                        'description': 'Advanced AI processing',
                        'validation': self._validate_premium_analysis,
                        'estimated_time': '1 min'
                    },
                    {
                        'id': 'value_betting',
                        'title': 'Value Betting',
                        'icon': '💰',
                        'description': 'Find value opportunities',
                        'validation': self._validate_value_betting,
                        'estimated_time': '1-2 min'
                    },
                    {
                        'id': 'risk_analysis',
                        'title': 'Risk Analysis',
                        'icon': '⚠️',
                        'description': 'Assess betting risks',
                        'validation': self._validate_risk_analysis,
                        'estimated_time': '1 min'
                    }
                ]
            }
        }
        
        # Add dashboard-specific workflows
        if self.dashboard_type == "optimized_premium":
            # Optimized workflow is already defined above as 'prediction_analysis'
            pass
        elif self.dashboard_type == "interactive_cross_league":
            # Cross-league workflow is already defined above
            pass
        elif self.dashboard_type == "premium":
            # Premium workflow is already defined above
            pass
        
        return base_workflows
    
    def _get_default_workflow(self) -> str:
        """Get default workflow for the dashboard type."""
        workflow_mapping = {
            'optimized_premium': 'prediction_analysis',
            'interactive_cross_league': 'cross_league_analysis',
            'premium': 'premium_analysis',
            'integrated_production': 'prediction_analysis'
        }
        return workflow_mapping.get(self.dashboard_type, 'prediction_analysis')
    
    def toggle_workflow_mode(self):
        """Toggle between workflow mode and full-view mode."""
        st.session_state.workflow_enabled = not st.session_state.workflow_enabled
        self.workflow_enabled = st.session_state.workflow_enabled
        
        # Reset workflow state when toggling
        if self.workflow_enabled:
            st.session_state.current_step = 1
            st.session_state.workflow_start_time = time.time()
        
        logger.info(f"Workflow mode {'enabled' if self.workflow_enabled else 'disabled'}")
    
    def render_workflow_toggle(self):
        """Render workflow mode toggle control."""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 🎯 Analysis Mode")
        
        with col2:
            workflow_enabled = st.toggle(
                "Step-by-step",
                value=st.session_state.workflow_enabled,
                help="Toggle between guided workflow and full dashboard view"
            )
            
            if workflow_enabled != st.session_state.workflow_enabled:
                self.toggle_workflow_mode()
                st.rerun()
    
    def render_progress_indicator(self, workflow_id: Optional[str] = None):
        """
        Render enhanced step progress indicator with smooth transitions.
        
        Args:
            workflow_id: ID of the workflow to display (uses selected workflow if None)
        """
        if not self.workflow_enabled:
            return
        
        workflow_id = workflow_id or st.session_state.selected_workflow
        if workflow_id not in self.workflows:
            return
        
        workflow = self.workflows[workflow_id]
        steps = workflow['steps']
        current_step = st.session_state.current_step
        
        # Progress bar container
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0 2rem 0;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <h4 style="margin: 0 0 1rem 0; color: #495057;">
                🚀 {workflow['name']} Progress
            </h4>
        """, unsafe_allow_html=True)
        
        # Calculate progress percentage
        progress_percentage = ((current_step - 1) / len(steps)) * 100
        
        # Overall progress bar
        st.progress(progress_percentage / 100, text=f"Step {current_step} of {len(steps)}")
        
        # Step indicators
        cols = st.columns(len(steps))
        for i, (col, step) in enumerate(zip(cols, steps)):
            with col:
                step_num = i + 1
                is_current = step_num == current_step
                is_completed = step_num < current_step
                is_upcoming = step_num > current_step
                
                # Step styling based on state
                if is_completed:
                    bg_color = "#28a745"
                    text_color = "white"
                    border_color = "#28a745"
                    icon = "✅"
                    opacity = "1.0"
                elif is_current:
                    bg_color = "#007bff"
                    text_color = "white"
                    border_color = "#007bff"
                    icon = step["icon"]
                    opacity = "1.0"
                else:
                    bg_color = "#f8f9fa"
                    text_color = "#6c757d"
                    border_color = "#dee2e6"
                    icon = "⏳"
                    opacity = "0.6"
                
                # Render step indicator
                st.markdown(f"""
                <div style="
                    text-align: center;
                    padding: 1rem 0.5rem;
                    opacity: {opacity};
                ">
                    <div style="
                        background: {bg_color};
                        color: {text_color};
                        border: 2px solid {border_color};
                        border-radius: 50%;
                        width: 50px;
                        height: 50px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 0 auto 0.5rem auto;
                        font-size: 1.2rem;
                        font-weight: bold;
                        transition: all 0.3s ease;
                    ">
                        {icon}
                    </div>
                    <div style="
                        font-size: 0.8rem;
                        font-weight: 600;
                        color: {text_color if not is_upcoming else '#6c757d'};
                        margin-bottom: 0.25rem;
                    ">
                        {step['title']}
                    </div>
                    <div style="
                        font-size: 0.7rem;
                        color: #6c757d;
                        line-height: 1.2;
                    ">
                        {step['description']}
                    </div>
                    <div style="
                        font-size: 0.6rem;
                        color: #868e96;
                        margin-top: 0.25rem;
                        font-style: italic;
                    ">
                        {step.get('estimated_time', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    def render_workflow_selector(self):
        """Render workflow selection interface."""
        if not self.workflow_enabled:
            return
        
        st.markdown("### 🎯 Choose Your Analysis Journey")
        
        workflow_options = {
            workflow_id: f"{workflow['name']} - {workflow['description']}"
            for workflow_id, workflow in self.workflows.items()
        }
        
        selected_workflow = st.selectbox(
            "Select analysis type:",
            options=list(workflow_options.keys()),
            format_func=lambda x: workflow_options[x],
            index=list(workflow_options.keys()).index(st.session_state.selected_workflow),
            help="Choose the type of analysis you want to perform"
        )
        
        if selected_workflow != st.session_state.selected_workflow:
            st.session_state.selected_workflow = selected_workflow
            st.session_state.current_step = 1
            st.session_state.workflow_data = {}
            st.session_state.workflow_start_time = time.time()
            st.rerun()
    
    def get_current_step_info(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about the current step.
        
        Args:
            workflow_id: ID of the workflow (uses selected workflow if None)
            
        Returns:
            Dictionary containing current step information
        """
        if not self.workflow_enabled:
            return {}
        
        workflow_id = workflow_id or st.session_state.selected_workflow
        if workflow_id not in self.workflows:
            return {}
        
        workflow = self.workflows[workflow_id]
        steps = workflow['steps']
        current_step = st.session_state.current_step
        
        if current_step <= len(steps):
            step_info = steps[current_step - 1].copy()
            step_info['step_number'] = current_step
            step_info['total_steps'] = len(steps)
            step_info['workflow_name'] = workflow['name']
            return step_info
        
        return {}
    
    def advance_step(self, workflow_id: Optional[str] = None) -> bool:
        """
        Advance to the next step if current step is valid.
        
        Args:
            workflow_id: ID of the workflow (uses selected workflow if None)
            
        Returns:
            True if step was advanced, False otherwise
        """
        if not self.workflow_enabled:
            return False
        
        workflow_id = workflow_id or st.session_state.selected_workflow
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        steps = workflow['steps']
        current_step = st.session_state.current_step
        
        # Validate current step
        if current_step <= len(steps):
            step = steps[current_step - 1]
            validation_func = step.get('validation')
            
            if validation_func and not validation_func():
                return False
        
        # Record completion time
        step_key = f"step_{current_step}"
        st.session_state.step_completion_times[step_key] = time.time()
        
        # Advance step
        if current_step < len(steps):
            st.session_state.current_step += 1
            logger.info(f"Advanced to step {st.session_state.current_step} in workflow {workflow_id}")
            return True
        
        return False
    
    def go_to_step(self, step_number: int, workflow_id: Optional[str] = None) -> bool:
        """
        Navigate directly to a specific step.
        
        Args:
            step_number: Step number to navigate to (1-based)
            workflow_id: ID of the workflow (uses selected workflow if None)
            
        Returns:
            True if navigation was successful, False otherwise
        """
        if not self.workflow_enabled:
            return False
        
        workflow_id = workflow_id or st.session_state.selected_workflow
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        steps = workflow['steps']
        
        if 1 <= step_number <= len(steps):
            st.session_state.current_step = step_number
            logger.info(f"Navigated to step {step_number} in workflow {workflow_id}")
            return True
        
        return False
    
    def reset_workflow(self, workflow_id: Optional[str] = None):
        """
        Reset workflow to the beginning.
        
        Args:
            workflow_id: ID of the workflow (uses selected workflow if None)
        """
        workflow_id = workflow_id or st.session_state.selected_workflow
        
        st.session_state.current_step = 1
        st.session_state.workflow_data = {}
        st.session_state.step_completion_times = {}
        st.session_state.workflow_start_time = time.time()
        
        logger.info(f"Reset workflow {workflow_id}")
    
    def get_workflow_progress(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive workflow progress information.
        
        Args:
            workflow_id: ID of the workflow (uses selected workflow if None)
            
        Returns:
            Dictionary containing progress information
        """
        workflow_id = workflow_id or st.session_state.selected_workflow
        if workflow_id not in self.workflows:
            return {}
        
        workflow = self.workflows[workflow_id]
        steps = workflow['steps']
        current_step = st.session_state.current_step
        
        total_time = time.time() - st.session_state.workflow_start_time
        completed_steps = current_step - 1
        progress_percentage = (completed_steps / len(steps)) * 100
        
        return {
            'workflow_id': workflow_id,
            'workflow_name': workflow['name'],
            'current_step': current_step,
            'total_steps': len(steps),
            'completed_steps': completed_steps,
            'progress_percentage': progress_percentage,
            'total_time_seconds': total_time,
            'step_completion_times': st.session_state.step_completion_times.copy(),
            'is_complete': current_step > len(steps)
        }

    # Validation Methods
    def _validate_team_selection(self) -> bool:
        """Validate team selection step."""
        return (
            hasattr(st.session_state, 'selected_teams') and
            len(getattr(st.session_state, 'selected_teams', [])) >= 2
        )

    def _validate_ai_analysis(self) -> bool:
        """Validate AI analysis step."""
        return (
            hasattr(st.session_state, 'analysis_complete') and
            st.session_state.analysis_complete
        )

    def _validate_results(self) -> bool:
        """Validate results viewing step."""
        return (
            hasattr(st.session_state, 'results_viewed') and
            st.session_state.results_viewed
        )

    def _validate_insights(self) -> bool:
        """Validate insights step."""
        return True  # Always valid as final step

    def _validate_league_selection(self) -> bool:
        """Validate league selection for cross-league analysis."""
        return (
            hasattr(st.session_state, 'selected_leagues') and
            len(getattr(st.session_state, 'selected_leagues', [])) >= 2
        )

    def _validate_cross_league_teams(self) -> bool:
        """Validate cross-league team selection."""
        return (
            hasattr(st.session_state, 'cross_league_teams') and
            len(getattr(st.session_state, 'cross_league_teams', [])) >= 2
        )

    def _validate_scenario_setup(self) -> bool:
        """Validate scenario setup."""
        return (
            hasattr(st.session_state, 'scenario_configured') and
            st.session_state.scenario_configured
        )

    def _validate_cross_league_analysis(self) -> bool:
        """Validate cross-league analysis completion."""
        return (
            hasattr(st.session_state, 'cross_league_analysis_complete') and
            st.session_state.cross_league_analysis_complete
        )

    def _validate_comparison(self) -> bool:
        """Validate league comparison step."""
        return True  # Always valid as final step

    def _validate_data_selection(self) -> bool:
        """Validate data source selection."""
        return (
            hasattr(st.session_state, 'selected_data_sources') and
            len(getattr(st.session_state, 'selected_data_sources', [])) >= 1
        )

    def _validate_premium_analysis(self) -> bool:
        """Validate premium analysis completion."""
        return (
            hasattr(st.session_state, 'premium_analysis_complete') and
            st.session_state.premium_analysis_complete
        )

    def _validate_value_betting(self) -> bool:
        """Validate value betting analysis."""
        return (
            hasattr(st.session_state, 'value_betting_complete') and
            st.session_state.value_betting_complete
        )

    def _validate_risk_analysis(self) -> bool:
        """Validate risk analysis completion."""
        return True  # Always valid as final step

    def render_step_navigation(self, workflow_id: Optional[str] = None):
        """
        Render step navigation controls.

        Args:
            workflow_id: ID of the workflow (uses selected workflow if None)
        """
        if not self.workflow_enabled:
            return

        workflow_id = workflow_id or st.session_state.selected_workflow
        if workflow_id not in self.workflows:
            return

        workflow = self.workflows[workflow_id]
        steps = workflow['steps']
        current_step = st.session_state.current_step

        col1, col2, col3 = st.columns([1, 2, 1])

        with col1:
            if current_step > 1:
                if st.button("⬅️ Previous", type="secondary", use_container_width=True):
                    st.session_state.current_step -= 1
                    st.rerun()

        with col2:
            # Step indicator
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem;">
                <strong>Step {current_step} of {len(steps)}</strong><br>
                <span style="color: #6c757d;">{steps[current_step-1]['title']}</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            if current_step < len(steps):
                # Check if current step is valid before allowing advance
                step = steps[current_step - 1]
                validation_func = step.get('validation')
                can_advance = not validation_func or validation_func()

                if st.button(
                    "Next ➡️",
                    type="primary" if can_advance else "secondary",
                    disabled=not can_advance,
                    use_container_width=True
                ):
                    if self.advance_step(workflow_id):
                        st.rerun()
            else:
                if st.button("🔄 Start Over", type="secondary", use_container_width=True):
                    self.reset_workflow(workflow_id)
                    st.rerun()

    def render_workflow_summary(self, workflow_id: Optional[str] = None):
        """
        Render workflow completion summary.

        Args:
            workflow_id: ID of the workflow (uses selected workflow if None)
        """
        if not self.workflow_enabled:
            return

        progress = self.get_workflow_progress(workflow_id)

        if not progress.get('is_complete'):
            return

        st.success("🎉 Workflow Complete!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Time", f"{progress['total_time_seconds']:.1f}s")

        with col2:
            st.metric("Steps Completed", f"{progress['completed_steps']}/{progress['total_steps']}")

        with col3:
            avg_step_time = progress['total_time_seconds'] / progress['total_steps']
            st.metric("Avg Step Time", f"{avg_step_time:.1f}s")

        # Detailed step times
        if progress['step_completion_times']:
            st.markdown("### ⏱️ Step Completion Times")

            step_times = []
            prev_time = st.session_state.workflow_start_time

            for step_key, completion_time in progress['step_completion_times'].items():
                step_duration = completion_time - prev_time
                step_times.append(step_duration)
                prev_time = completion_time

            workflow = self.workflows[progress['workflow_id']]
            for i, (step, duration) in enumerate(zip(workflow['steps'], step_times)):
                st.write(f"**{step['title']}**: {duration:.1f}s")

    def get_workflow_analytics(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get workflow analytics and performance metrics.

        Args:
            workflow_id: ID of the workflow (uses selected workflow if None)

        Returns:
            Dictionary containing analytics data
        """
        progress = self.get_workflow_progress(workflow_id)

        if not progress:
            return {}

        # Calculate step efficiency
        step_times = []
        if progress['step_completion_times']:
            prev_time = st.session_state.workflow_start_time
            for completion_time in progress['step_completion_times'].values():
                step_duration = completion_time - prev_time
                step_times.append(step_duration)
                prev_time = completion_time

        analytics = {
            'workflow_efficiency': {
                'total_time': progress['total_time_seconds'],
                'average_step_time': sum(step_times) / len(step_times) if step_times else 0,
                'fastest_step': min(step_times) if step_times else 0,
                'slowest_step': max(step_times) if step_times else 0
            },
            'completion_rate': progress['progress_percentage'],
            'user_engagement': {
                'steps_completed': progress['completed_steps'],
                'workflow_abandoned': not progress['is_complete'] and progress['completed_steps'] > 0
            },
            'workflow_metadata': {
                'workflow_id': progress['workflow_id'],
                'workflow_name': progress['workflow_name'],
                'total_steps': progress['total_steps']
            }
        }

        return analytics

    def export_workflow_data(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export workflow data for backup or analysis.

        Args:
            workflow_id: ID of the workflow (uses selected workflow if None)

        Returns:
            Dictionary containing exportable workflow data
        """
        progress = self.get_workflow_progress(workflow_id)
        analytics = self.get_workflow_analytics(workflow_id)

        return {
            'workflow_progress': progress,
            'workflow_analytics': analytics,
            'workflow_data': st.session_state.workflow_data.copy(),
            'export_timestamp': datetime.now().isoformat(),
            'dashboard_type': self.dashboard_type
        }
