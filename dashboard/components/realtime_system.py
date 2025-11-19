#!/usr/bin/env python3
"""
Real-time Update System for GoalDiggers Platform

Implements real-time data refresh capabilities with intelligent update
scheduling and efficient data synchronization.
"""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

logger = logging.getLogger(__name__)

@dataclass
class UpdateConfig:
    """Configuration for real-time updates."""
    interval_seconds: int
    priority: str  # 'high', 'medium', 'low'
    enabled: bool = True
    last_update: Optional[datetime] = None
    update_count: int = 0

class RealTimeUpdateSystem:
    """
    Advanced real-time update system with intelligent scheduling.
    """
    
    def __init__(self):
        """Initialize the real-time update system."""
        self.update_configs = {}
        self.update_callbacks = {}
        self.update_data = {}
        self.update_locks = {}
        
        # System state
        self.is_active = False
        self.update_thread = None
        self.last_activity = time.time()
        
        # Performance tracking
        self.update_stats = {
            'total_updates': 0,
            'successful_updates': 0,
            'failed_updates': 0,
            'average_update_time': 0
        }
        
        logger.info("Real-time update system initialized")
    
    def register_update_source(self, 
                              source_id: str,
                              update_callback: Callable,
                              interval_seconds: int = 30,
                              priority: str = 'medium') -> None:
        """
        Register a data source for real-time updates.
        
        Args:
            source_id: Unique identifier for the data source
            update_callback: Function to call for updates
            interval_seconds: Update interval in seconds
            priority: Update priority ('high', 'medium', 'low')
        """
        self.update_configs[source_id] = UpdateConfig(
            interval_seconds=interval_seconds,
            priority=priority,
            enabled=True
        )
        
        self.update_callbacks[source_id] = update_callback
        self.update_locks[source_id] = threading.Lock()
        
        logger.info(f"Registered update source: {source_id} (interval: {interval_seconds}s, priority: {priority})")
    
    def update_source_config(self, 
                           source_id: str,
                           interval_seconds: Optional[int] = None,
                           priority: Optional[str] = None,
                           enabled: Optional[bool] = None) -> None:
        """
        Update configuration for a data source.
        
        Args:
            source_id: Data source identifier
            interval_seconds: New update interval
            priority: New priority level
            enabled: Enable/disable updates
        """
        if source_id not in self.update_configs:
            logger.error(f"Unknown update source: {source_id}")
            return
        
        config = self.update_configs[source_id]
        
        if interval_seconds is not None:
            config.interval_seconds = interval_seconds
        if priority is not None:
            config.priority = priority
        if enabled is not None:
            config.enabled = enabled
        
        logger.info(f"Updated config for {source_id}: interval={config.interval_seconds}s, priority={config.priority}, enabled={config.enabled}")
    
    def _should_update(self, source_id: str) -> bool:
        """Check if a source should be updated."""
        config = self.update_configs[source_id]
        
        if not config.enabled:
            return False
        
        if config.last_update is None:
            return True
        
        time_since_update = datetime.now() - config.last_update
        return time_since_update.total_seconds() >= config.interval_seconds
    
    def _get_update_priority_order(self) -> List[str]:
        """Get sources ordered by update priority."""
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        
        return sorted(
            self.update_configs.keys(),
            key=lambda x: (
                priority_order.get(self.update_configs[x].priority, 3),
                self.update_configs[x].last_update or datetime.min
            )
        )
    
    def _update_source(self, source_id: str) -> bool:
        """Update a single data source."""
        if source_id not in self.update_callbacks:
            return False
        
        config = self.update_configs[source_id]
        
        with self.update_locks[source_id]:
            try:
                start_time = time.time()
                
                # Call the update callback
                new_data = self.update_callbacks[source_id]()
                
                # Store the updated data
                self.update_data[source_id] = {
                    'data': new_data,
                    'timestamp': datetime.now(),
                    'update_time': time.time() - start_time
                }
                
                # Update configuration
                config.last_update = datetime.now()
                config.update_count += 1
                
                # Update statistics
                self.update_stats['successful_updates'] += 1
                self.update_stats['total_updates'] += 1
                
                update_time = time.time() - start_time
                self.update_stats['average_update_time'] = (
                    (self.update_stats['average_update_time'] * (self.update_stats['successful_updates'] - 1) + update_time) /
                    self.update_stats['successful_updates']
                )
                
                logger.debug(f"Updated {source_id} in {update_time:.3f}s")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update {source_id}: {e}")
                self.update_stats['failed_updates'] += 1
                self.update_stats['total_updates'] += 1
                return False
    
    def _update_loop(self):
        """Main update loop running in background thread."""
        while self.is_active:
            try:
                # Check if user is still active (no updates if inactive for 5 minutes)
                if time.time() - self.last_activity > 300:
                    time.sleep(10)
                    continue
                
                # Get sources that need updating, ordered by priority
                sources_to_update = [
                    source_id for source_id in self._get_update_priority_order()
                    if self._should_update(source_id)
                ]
                
                # Update sources
                for source_id in sources_to_update:
                    if not self.is_active:
                        break
                    
                    self._update_source(source_id)
                    
                    # Small delay between updates to prevent overwhelming
                    time.sleep(0.1)
                
                # Sleep before next cycle
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                time.sleep(10)
    
    def start_updates(self):
        """Start the real-time update system."""
        if self.is_active:
            return
        
        self.is_active = True
        self.last_activity = time.time()
        
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info("Real-time updates started")
    
    def stop_updates(self):
        """Stop the real-time update system."""
        self.is_active = False
        
        if self.update_thread:
            self.update_thread.join(timeout=5)
        
        logger.info("Real-time updates stopped")
    
    def mark_user_activity(self):
        """Mark user activity to keep updates active."""
        self.last_activity = time.time()
    
    def get_latest_data(self, source_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest data for a source.
        
        Args:
            source_id: Data source identifier
            
        Returns:
            Latest data with metadata
        """
        self.mark_user_activity()
        return self.update_data.get(source_id)
    
    def force_update(self, source_id: str) -> bool:
        """
        Force an immediate update for a specific source.
        
        Args:
            source_id: Data source identifier
            
        Returns:
            True if update was successful
        """
        if source_id not in self.update_callbacks:
            logger.error(f"Unknown update source: {source_id}")
            return False
        
        logger.info(f"Forcing update for {source_id}")
        return self._update_source(source_id)
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get comprehensive update system status."""
        source_status = {}
        
        for source_id, config in self.update_configs.items():
            latest_data = self.update_data.get(source_id)
            
            source_status[source_id] = {
                'enabled': config.enabled,
                'priority': config.priority,
                'interval_seconds': config.interval_seconds,
                'update_count': config.update_count,
                'last_update': config.last_update.isoformat() if config.last_update else None,
                'has_data': latest_data is not None,
                'data_age_seconds': (
                    (datetime.now() - latest_data['timestamp']).total_seconds()
                    if latest_data else None
                ),
                'next_update_in_seconds': (
                    max(0, config.interval_seconds - (datetime.now() - config.last_update).total_seconds())
                    if config.last_update else 0
                )
            }
        
        return {
            'system_active': self.is_active,
            'total_sources': len(self.update_configs),
            'active_sources': sum(1 for c in self.update_configs.values() if c.enabled),
            'last_activity_ago_seconds': time.time() - self.last_activity,
            'statistics': self.update_stats,
            'sources': source_status
        }
    
    def render_update_status_widget(self):
        """Render a status widget showing real-time update information."""
        status = self.get_update_status()
        
        # System status indicator
        if status['system_active']:
            status_color = "#10b981"  # Green
            status_text = "🟢 Active"
        else:
            status_color = "#ef4444"  # Red
            status_text = "🔴 Inactive"
        
        # Success rate
        total_updates = status['statistics']['total_updates']
        success_rate = (
            (status['statistics']['successful_updates'] / total_updates * 100)
            if total_updates > 0 else 100
        )
        
        widget_html = f"""
        <div style="
            background: white;
            border-radius: 8px;
            padding: 1rem;
            border: 1px solid #e5e7eb;
            margin: 1rem 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h4 style="margin: 0; color: #374151;">Real-time Updates</h4>
                <span style="color: {status_color}; font-weight: 600;">{status_text}</span>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem;">
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e40af;">{status['active_sources']}</div>
                    <div style="font-size: 0.875rem; color: #6b7280;">Active Sources</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #059669;">{success_rate:.1f}%</div>
                    <div style="font-size: 0.875rem; color: #6b7280;">Success Rate</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: 700; color: #f59e0b;">{status['statistics']['average_update_time']:.2f}s</div>
                    <div style="font-size: 0.875rem; color: #6b7280;">Avg Update Time</div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(widget_html, unsafe_allow_html=True)
        
        # Show individual source status
        if st.expander("Source Details", expanded=False):
            for source_id, source_info in status['sources'].items():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{source_id}**")
                
                with col2:
                    enabled_icon = "✅" if source_info['enabled'] else "❌"
                    st.write(f"{enabled_icon} {source_info['priority'].title()}")
                
                with col3:
                    if source_info['has_data']:
                        age = source_info['data_age_seconds']
                        if age < 60:
                            st.write(f"🟢 {age:.0f}s ago")
                        elif age < 300:
                            st.write(f"🟡 {age/60:.1f}m ago")
                        else:
                            st.write(f"🔴 {age/60:.1f}m ago")
                    else:
                        st.write("⚪ No data")
                
                with col4:
                    next_update = source_info['next_update_in_seconds']
                    if next_update <= 0:
                        st.write("🔄 Updating...")
                    else:
                        st.write(f"⏱️ {next_update:.0f}s")

# Global instance for easy access
realtime_system = RealTimeUpdateSystem()


class AdvancedFilterSystem:
    """
    Advanced filtering system with intelligent filter combinations and UX.
    """

    def __init__(self):
        """Initialize the advanced filter system."""
        self.filter_configs = {}
        self.active_filters = {}
        self.filter_history = []

        logger.info("Advanced filter system initialized")

    def register_filter(self,
                       filter_id: str,
                       filter_type: str,
                       options: List[Any],
                       default_value: Any = None,
                       label: str = None,
                       help_text: str = None) -> None:
        """
        Register a new filter.

        Args:
            filter_id: Unique filter identifier
            filter_type: Type of filter ('select', 'multiselect', 'slider', 'date_range', 'text')
            options: Available options for the filter
            default_value: Default filter value
            label: Display label for the filter
            help_text: Help text for the filter
        """
        self.filter_configs[filter_id] = {
            'type': filter_type,
            'options': options,
            'default_value': default_value,
            'label': label or filter_id.replace('_', ' ').title(),
            'help_text': help_text,
            'enabled': True
        }

        # Set default value
        if default_value is not None:
            self.active_filters[filter_id] = default_value

        logger.info(f"Registered filter: {filter_id} ({filter_type})")

    def render_filter_panel(self,
                           columns: int = 3,
                           show_reset: bool = True,
                           show_presets: bool = True) -> Dict[str, Any]:
        """
        Render an advanced filter panel with multiple filter types.

        Args:
            columns: Number of columns for filter layout
            show_reset: Whether to show reset button
            show_presets: Whether to show filter presets

        Returns:
            Dictionary of active filter values
        """
        st.markdown("### 🔍 Advanced Filters")

        # Filter presets
        if show_presets:
            preset_col1, preset_col2, preset_col3 = st.columns(3)

            with preset_col1:
                if st.button("⚡ Quick Filters", help="Apply commonly used filters"):
                    self._apply_preset("quick")

            with preset_col2:
                if st.button("🎯 Detailed Analysis", help="Apply filters for detailed analysis"):
                    self._apply_preset("detailed")

            with preset_col3:
                if st.button("📊 Overview", help="Apply filters for overview"):
                    self._apply_preset("overview")

        # Create filter columns
        filter_cols = st.columns(columns)
        filter_items = list(self.filter_configs.items())

        for i, (filter_id, config) in enumerate(filter_items):
            if not config['enabled']:
                continue

            col_index = i % columns

            with filter_cols[col_index]:
                self._render_single_filter(filter_id, config)

        # Control buttons
        if show_reset or show_presets:
            st.markdown("---")
            control_cols = st.columns([1, 1, 1, 2])

            with control_cols[0]:
                if st.button("🔄 Reset All"):
                    self._reset_all_filters()
                    st.rerun()

            with control_cols[1]:
                if st.button("💾 Save Preset"):
                    self._save_current_as_preset()

            with control_cols[2]:
                if st.button("📋 Copy Filters"):
                    self._copy_filters_to_clipboard()

        return self.active_filters.copy()

    def _render_single_filter(self, filter_id: str, config: Dict[str, Any]):
        """Render a single filter widget."""
        filter_type = config['type']
        label = config['label']
        help_text = config['help_text']
        options = config['options']
        default_value = config.get('default_value')

        current_value = self.active_filters.get(filter_id, default_value)

        if filter_type == 'select':
            new_value = st.selectbox(
                label,
                options=options,
                index=options.index(current_value) if current_value in options else 0,
                help=help_text,
                key=f"filter_{filter_id}"
            )

        elif filter_type == 'multiselect':
            new_value = st.multiselect(
                label,
                options=options,
                default=current_value if isinstance(current_value, list) else [],
                help=help_text,
                key=f"filter_{filter_id}"
            )

        elif filter_type == 'slider':
            min_val, max_val = options[0], options[1]
            new_value = st.slider(
                label,
                min_value=min_val,
                max_value=max_val,
                value=current_value if current_value is not None else min_val,
                help=help_text,
                key=f"filter_{filter_id}"
            )

        elif filter_type == 'date_range':
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    f"{label} (Start)",
                    value=current_value[0] if current_value else options[0],
                    help=help_text,
                    key=f"filter_{filter_id}_start"
                )
            with col2:
                end_date = st.date_input(
                    f"{label} (End)",
                    value=current_value[1] if current_value else options[1],
                    help=help_text,
                    key=f"filter_{filter_id}_end"
                )
            new_value = (start_date, end_date)

        elif filter_type == 'text':
            new_value = st.text_input(
                label,
                value=current_value or "",
                help=help_text,
                key=f"filter_{filter_id}"
            )

        else:
            st.error(f"Unknown filter type: {filter_type}")
            return

        # Update active filters
        if new_value != current_value:
            self.active_filters[filter_id] = new_value
            self._add_to_history(filter_id, current_value, new_value)

    def _apply_preset(self, preset_name: str):
        """Apply a filter preset."""
        presets = {
            'quick': {
                'league': 'Premier League',
                'confidence': 0.7,
                'match_type': 'All'
            },
            'detailed': {
                'league': ['Premier League', 'La Liga'],
                'confidence': 0.8,
                'date_range': 'Last 30 days'
            },
            'overview': {
                'league': 'All',
                'confidence': 0.6,
                'match_type': 'All'
            }
        }

        if preset_name in presets:
            preset_filters = presets[preset_name]
            for filter_id, value in preset_filters.items():
                if filter_id in self.filter_configs:
                    self.active_filters[filter_id] = value

            st.success(f"Applied {preset_name} preset")

    def _reset_all_filters(self):
        """Reset all filters to default values."""
        for filter_id, config in self.filter_configs.items():
            default_value = config.get('default_value')
            if default_value is not None:
                self.active_filters[filter_id] = default_value
            elif filter_id in self.active_filters:
                del self.active_filters[filter_id]

        st.success("All filters reset to defaults")

    def _save_current_as_preset(self):
        """Save current filter state as a preset."""
        # In a real implementation, this would save to user preferences
        st.info("Filter preset saved (feature coming soon)")

    def _copy_filters_to_clipboard(self):
        """Copy current filters to clipboard."""
        import json
        filter_json = json.dumps(self.active_filters, indent=2, default=str)
        st.code(filter_json, language='json')
        st.info("Filter configuration displayed above")

    def _add_to_history(self, filter_id: str, old_value: Any, new_value: Any):
        """Add filter change to history."""
        self.filter_history.append({
            'timestamp': datetime.now(),
            'filter_id': filter_id,
            'old_value': old_value,
            'new_value': new_value
        })

        # Keep only last 50 changes
        if len(self.filter_history) > 50:
            self.filter_history = self.filter_history[-50:]

    def get_filter_summary(self) -> str:
        """Get a human-readable summary of active filters."""
        if not self.active_filters:
            return "No filters applied"

        summary_parts = []
        for filter_id, value in self.active_filters.items():
            config = self.filter_configs.get(filter_id, {})
            label = config.get('label', filter_id)

            if isinstance(value, list):
                if len(value) == 1:
                    summary_parts.append(f"{label}: {value[0]}")
                elif len(value) > 1:
                    summary_parts.append(f"{label}: {len(value)} selected")
            elif isinstance(value, tuple):
                summary_parts.append(f"{label}: {value[0]} to {value[1]}")
            else:
                summary_parts.append(f"{label}: {value}")

        return " | ".join(summary_parts)

# Global instance for easy access
advanced_filter_system = AdvancedFilterSystem()
