#!/usr/bin/env python3
"""
Universal Achievement System for GoalDiggers Platform

Modular achievement engine that can be integrated across all dashboard variants
to provide consistent gamification and user engagement features.

Features:
- Flexible achievement definitions with multiple types
- Progress tracking and persistence via Streamlit session state
- Visual achievement display components
- Performance-optimized with minimal memory footprint
- Extensible architecture for adding new achievement types
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

class UniversalAchievementSystem:
    """
    Universal achievement system for GoalDiggers dashboards.
    
    Provides consistent achievement tracking, progress monitoring,
    and visual display across all dashboard variants.
    """
    
    def __init__(self, dashboard_type: str = "universal"):
        """
        Initialize the universal achievement system.
        
        Args:
            dashboard_type: Type of dashboard for customized achievements
        """
        self.dashboard_type = dashboard_type
        self.achievements = self._initialize_achievements()
        self._initialize_session_state()
        
        logger.debug(f"✅ Universal Achievement System initialized for {dashboard_type}")
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state for achievement tracking."""
        # Core user statistics
        if 'user_level' not in st.session_state:
            st.session_state.user_level = 1
        if 'total_predictions' not in st.session_state:
            st.session_state.total_predictions = 0
        if 'correct_predictions' not in st.session_state:
            st.session_state.correct_predictions = 0
        if 'achievements_unlocked' not in st.session_state:
            st.session_state.achievements_unlocked = []
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        # Extended tracking for advanced achievements
        if 'cross_league_predictions' not in st.session_state:
            st.session_state.cross_league_predictions = 0
        if 'scenarios_explored' not in st.session_state:
            st.session_state.scenarios_explored = 0
        if 'features_used' not in st.session_state:
            st.session_state.features_used = set()
        if 'session_start_time' not in st.session_state:
            st.session_state.session_start_time = time.time()
        if 'total_session_time' not in st.session_state:
            st.session_state.total_session_time = 0
    
    def _initialize_achievements(self) -> Dict[str, Dict[str, Any]]:
        """Initialize achievement definitions."""
        base_achievements = {
            # Beginner achievements
            'first_prediction': {
                'name': '🎯 First Shot',
                'description': 'Make your first prediction',
                'requirement': 1,
                'type': 'predictions',
                'category': 'beginner',
                'points': 10
            },
            'early_adopter': {
                'name': '🚀 Early Adopter',
                'description': 'Use 3 different features',
                'requirement': 3,
                'type': 'features',
                'category': 'beginner',
                'points': 15
            },
            
            # Accuracy achievements
            'accuracy_novice': {
                'name': '🎯 Accuracy Novice',
                'description': 'Achieve 50% prediction accuracy',
                'requirement': 0.5,
                'type': 'accuracy',
                'category': 'accuracy',
                'points': 20
            },
            'accuracy_master': {
                'name': '🏆 Accuracy Master',
                'description': 'Achieve 70% prediction accuracy',
                'requirement': 0.7,
                'type': 'accuracy',
                'category': 'accuracy',
                'points': 50
            },
            'accuracy_legend': {
                'name': '👑 Accuracy Legend',
                'description': 'Achieve 85% prediction accuracy',
                'requirement': 0.85,
                'type': 'accuracy',
                'category': 'accuracy',
                'points': 100
            },
            
            # Volume achievements
            'prediction_enthusiast': {
                'name': '📈 Prediction Enthusiast',
                'description': 'Make 25 predictions',
                'requirement': 25,
                'type': 'predictions',
                'category': 'volume',
                'points': 30
            },
            'prediction_expert': {
                'name': '🔥 Prediction Expert',
                'description': 'Make 100 predictions',
                'requirement': 100,
                'type': 'predictions',
                'category': 'volume',
                'points': 75
            },
            
            # Cross-league achievements
            'cross_league_explorer': {
                'name': '🌍 Cross-League Explorer',
                'description': 'Make 5 cross-league predictions',
                'requirement': 5,
                'type': 'cross_league',
                'category': 'exploration',
                'points': 25
            },
            'cross_league_expert': {
                'name': '🌟 Cross-League Expert',
                'description': 'Make 20 cross-league predictions',
                'requirement': 20,
                'type': 'cross_league',
                'category': 'exploration',
                'points': 60
            },
            
            # Scenario achievements
            'scenario_explorer': {
                'name': '🔍 Scenario Explorer',
                'description': 'Try 5 different scenarios',
                'requirement': 5,
                'type': 'scenarios',
                'category': 'exploration',
                'points': 20
            },
            'scenario_master': {
                'name': '🎭 Scenario Master',
                'description': 'Try 15 different scenarios',
                'requirement': 15,
                'type': 'scenarios',
                'category': 'exploration',
                'points': 45
            },
            
            # Engagement achievements
            'dedicated_user': {
                'name': '⏰ Dedicated User',
                'description': 'Spend 30 minutes using the platform',
                'requirement': 1800,  # 30 minutes in seconds
                'type': 'session_time',
                'category': 'engagement',
                'points': 25
            },
            'power_user': {
                'name': '💪 Power User',
                'description': 'Use all available features',
                'requirement': 8,  # Adjust based on total features
                'type': 'features',
                'category': 'engagement',
                'points': 80
            }
        }
        
        # Add dashboard-specific achievements
        if self.dashboard_type == "premium":
            base_achievements.update({
                'value_betting_expert': {
                    'name': '💰 Value Betting Expert',
                    'description': 'Use value betting analysis 10 times',
                    'requirement': 10,
                    'type': 'value_betting',
                    'category': 'premium',
                    'points': 40
                }
            })
        elif self.dashboard_type == "interactive_cross_league":
            base_achievements.update({
                'entertainment_seeker': {
                    'name': '🎪 Entertainment Seeker',
                    'description': 'View entertaining commentary 10 times',
                    'requirement': 10,
                    'type': 'entertainment',
                    'category': 'interactive',
                    'points': 30
                }
            })
        
        return base_achievements
    
    def track_prediction(self, is_correct: Optional[bool] = None, is_cross_league: bool = False):
        """
        Track a prediction and update relevant statistics.
        
        Args:
            is_correct: Whether the prediction was correct (None if unknown)
            is_cross_league: Whether this was a cross-league prediction
        """
        st.session_state.total_predictions += 1
        
        if is_correct is not None and is_correct:
            st.session_state.correct_predictions += 1
        
        if is_cross_league:
            st.session_state.cross_league_predictions += 1
        
        # Add to prediction history
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'is_correct': is_correct,
            'is_cross_league': is_cross_league
        }
        st.session_state.prediction_history.append(prediction_record)
        
        # Keep only last 100 predictions for memory efficiency
        if len(st.session_state.prediction_history) > 100:
            st.session_state.prediction_history = st.session_state.prediction_history[-100:]
        
        # Check for new achievements
        self.check_achievements()
        
        logger.debug(f"Prediction tracked: total={st.session_state.total_predictions}, correct={st.session_state.correct_predictions}")
    
    def track_feature_usage(self, feature_name: str):
        """
        Track usage of a specific feature.
        
        Args:
            feature_name: Name of the feature being used
        """
        st.session_state.features_used.add(feature_name)
        self.check_achievements()
        
        logger.debug(f"Feature usage tracked: {feature_name}")
    
    def track_scenario_exploration(self):
        """Track exploration of a new scenario."""
        st.session_state.scenarios_explored += 1
        self.check_achievements()
        
        logger.debug(f"Scenario exploration tracked: total={st.session_state.scenarios_explored}")
    
    def update_session_time(self):
        """Update total session time."""
        current_time = time.time()
        session_duration = current_time - st.session_state.session_start_time
        st.session_state.total_session_time += session_duration
        st.session_state.session_start_time = current_time
        
        self.check_achievements()
    
    def get_achievement_progress(self, achievement_id: str) -> float:
        """
        Get progress towards a specific achievement.
        
        Args:
            achievement_id: ID of the achievement
            
        Returns:
            Progress as a float between 0.0 and 1.0
        """
        if achievement_id not in self.achievements:
            return 0.0
        
        achievement = self.achievements[achievement_id]
        achievement_type = achievement['type']
        requirement = achievement['requirement']
        
        if achievement_type == 'predictions':
            current = st.session_state.total_predictions
        elif achievement_type == 'accuracy':
            if st.session_state.total_predictions == 0:
                return 0.0
            current = st.session_state.correct_predictions / st.session_state.total_predictions
        elif achievement_type == 'cross_league':
            current = st.session_state.cross_league_predictions
        elif achievement_type == 'scenarios':
            current = st.session_state.scenarios_explored
        elif achievement_type == 'features':
            current = len(st.session_state.features_used)
        elif achievement_type == 'session_time':
            self.update_session_time()
            current = st.session_state.total_session_time
        else:
            # Handle custom achievement types
            current = 0
        
        return min(1.0, current / requirement)
    
    def check_achievements(self):
        """Check for newly unlocked achievements."""
        newly_unlocked = []
        
        for achievement_id, achievement in self.achievements.items():
            if achievement_id not in st.session_state.achievements_unlocked:
                if self.get_achievement_progress(achievement_id) >= 1.0:
                    st.session_state.achievements_unlocked.append(achievement_id)
                    newly_unlocked.append(achievement)
                    
                    # Update user level based on points
                    points = achievement.get('points', 0)
                    self._update_user_level(points)
        
        # Display enhanced gradient notifications for newly unlocked achievements
        for achievement in newly_unlocked:
            self._render_achievement_notification(achievement)
            logger.info(f"Achievement unlocked: {achievement['name']}")

    def _render_achievement_notification(self, achievement: Dict[str, Any]):
        """Render a professional gradient notification for newly unlocked achievements."""
        achievement_name = achievement.get('name', 'Unknown Achievement')
        achievement_desc = achievement.get('description', '')
        points = achievement.get('points', 0)

        # Enhanced gradient notification card
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            color: #1f2937;
            padding: 1.5rem;
            border-radius: 12px;
            margin: 1rem 0;
            border: 2px solid #f59e0b;
            box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
            animation: pulse 2s infinite;
            text-align: center;
        ">
            <h3 style="margin: 0 0 0.5rem 0; font-weight: 700; color: #1f2937;">
                🎉 Achievement Unlocked!
            </h3>
            <h4 style="margin: 0 0 0.5rem 0; font-weight: 600; color: #1f2937;">
                {achievement_name}
            </h4>
            <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #374151;">
                {achievement_desc}
            </p>
            <div style="
                background: rgba(31, 41, 55, 0.1);
                padding: 0.5rem;
                border-radius: 8px;
                margin-top: 1rem;
            ">
                <span style="font-weight: 600; color: #1f2937;">
                    +{points} Points Earned! 🏆
                </span>
            </div>
        </div>
        <style>
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.02); }}
            100% {{ transform: scale(1); }}
        }}
        </style>
        """, unsafe_allow_html=True)

    def _update_user_level(self, points: int):
        """Update user level based on points earned."""
        # Simple leveling system: 100 points per level
        total_points = sum(
            self.achievements[aid].get('points', 0) 
            for aid in st.session_state.achievements_unlocked
        )
        new_level = max(1, total_points // 100 + 1)
        
        if new_level > st.session_state.user_level:
            st.session_state.user_level = new_level
            st.success(f"🆙 Level Up! You've reached Level {new_level}!")
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        accuracy = 0.0
        if st.session_state.total_predictions > 0:
            accuracy = st.session_state.correct_predictions / st.session_state.total_predictions
        
        total_points = sum(
            self.achievements[aid].get('points', 0) 
            for aid in st.session_state.achievements_unlocked
        )
        
        return {
            'level': st.session_state.user_level,
            'total_predictions': st.session_state.total_predictions,
            'correct_predictions': st.session_state.correct_predictions,
            'accuracy': accuracy,
            'achievements_unlocked': len(st.session_state.achievements_unlocked),
            'total_achievements': len(self.achievements),
            'total_points': total_points,
            'cross_league_predictions': st.session_state.cross_league_predictions,
            'scenarios_explored': st.session_state.scenarios_explored,
            'features_used': len(st.session_state.features_used)
        }

    def render_achievements_section(self, show_locked: bool = True):
        """
        Render the achievements section with unlocked and locked achievements.

        Args:
            show_locked: Whether to show locked achievements
        """
        st.markdown("## 🏆 Achievements")

        # Achievement categories
        categories = {
            'beginner': '🌱 Getting Started',
            'accuracy': '🎯 Accuracy Masters',
            'volume': '📈 Volume Champions',
            'exploration': '🌍 Explorers',
            'engagement': '💪 Power Users',
            'premium': '💎 Premium Features',
            'interactive': '🎪 Interactive Features'
        }

        # Create tabs for different categories
        category_tabs = st.tabs(list(categories.values()))

        for i, (category_key, category_name) in enumerate(categories.items()):
            with category_tabs[i]:
                self._render_category_achievements(category_key, show_locked)

    def _render_category_achievements(self, category: str, show_locked: bool):
        """Render achievements for a specific category."""
        category_achievements = {
            aid: achievement for aid, achievement in self.achievements.items()
            if achievement.get('category') == category
        }

        if not category_achievements:
            st.info(f"No achievements in this category yet.")
            return

        # Split into unlocked and locked
        unlocked = []
        locked = []

        for aid, achievement in category_achievements.items():
            if aid in st.session_state.achievements_unlocked:
                unlocked.append((aid, achievement))
            else:
                locked.append((aid, achievement))

        # Display unlocked achievements
        if unlocked:
            st.markdown("### ✅ Unlocked")
            cols = st.columns(min(3, len(unlocked)))
            for i, (aid, achievement) in enumerate(unlocked):
                with cols[i % len(cols)]:
                    self._render_achievement_card(aid, achievement, unlocked=True)

        # Display locked achievements
        if locked and show_locked:
            st.markdown("### 🔒 Locked")
            cols = st.columns(min(3, len(locked)))
            for i, (aid, achievement) in enumerate(locked):
                with cols[i % len(cols)]:
                    self._render_achievement_card(aid, achievement, unlocked=False)

    def _render_achievement_card(self, achievement_id: str, achievement: Dict[str, Any], unlocked: bool):
        """Render an individual achievement card."""
        progress = self.get_achievement_progress(achievement_id)
        points = achievement.get('points', 0)

        # Enhanced card styling based on unlock status with premium gradients
        if unlocked:
            card_style = """
            <div style="
                background: linear-gradient(135deg, #1f4e79 0%, #2a5298 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                box-shadow: 0 8px 25px rgba(31, 78, 121, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            ">
            """
        else:
            card_style = """
            <div style="
                background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1rem;
                box-shadow: 0 4px 15px rgba(107, 114, 128, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.1);
                opacity: 0.8;
                transition: opacity 0.3s ease;
            ">
            """

        st.markdown(f"""
        {card_style}
            <h4 style="margin: 0 0 0.5rem 0;">{achievement['name']}</h4>
            <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem;">{achievement['description']}</p>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.8rem;">Points: {points}</span>
                <span style="font-size: 0.8rem;">{progress:.0%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Progress bar for locked achievements
        if not unlocked:
            st.progress(progress)

    def render_sidebar_stats(self):
        """Render achievement stats in sidebar with enhanced gradient header."""
        # Enhanced gradient header for sidebar stats
        st.sidebar.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            color: white;
            text-align: center;
            box-shadow: 0 3px 12px rgba(102, 126, 234, 0.3);
        ">
            <h4 style="margin: 0; font-weight: 600;">🎮 Player Stats</h4>
        </div>
        """, unsafe_allow_html=True)

        stats = self.get_user_stats()

        # Core metrics
        st.sidebar.metric("Level", stats['level'])
        st.sidebar.metric("Total Predictions", stats['total_predictions'])
        st.sidebar.metric("Accuracy", f"{stats['accuracy']:.1%}")
        st.sidebar.metric("Achievements", f"{stats['achievements_unlocked']}/{stats['total_achievements']}")

        # Progress to next level
        current_points = stats['total_points']
        next_level_points = stats['level'] * 100
        level_progress = (current_points % 100) / 100

        st.sidebar.markdown("#### 🆙 Level Progress")
        st.sidebar.progress(level_progress)
        st.sidebar.caption(f"{current_points % 100}/100 points to next level")

        # Recent achievements
        if st.session_state.achievements_unlocked:
            st.sidebar.markdown("### 🏆 Recent Achievements")
            recent_achievements = st.session_state.achievements_unlocked[-3:]
            for achievement_id in recent_achievements:
                if achievement_id in self.achievements:
                    achievement = self.achievements[achievement_id]
                    st.sidebar.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                        color: white;
                        padding: 0.5rem;
                        border-radius: 0.25rem;
                        margin-bottom: 0.5rem;
                        font-size: 0.8rem;
                    ">
                        {achievement['name']}
                    </div>
                    """, unsafe_allow_html=True)

    def render_compact_stats(self):
        """Render compact achievement stats for dashboard headers."""
        stats = self.get_user_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Level", stats['level'], delta=f"{stats['total_points'] % 100}/100 XP")

        with col2:
            st.metric("Predictions", stats['total_predictions'])

        with col3:
            st.metric("Accuracy", f"{stats['accuracy']:.1%}")

        with col4:
            st.metric("Achievements", f"{stats['achievements_unlocked']}/{stats['total_achievements']}")

    def get_achievement_categories(self) -> Dict[str, List[str]]:
        """Get achievements organized by category."""
        categories = {}
        for aid, achievement in self.achievements.items():
            category = achievement.get('category', 'other')
            if category not in categories:
                categories[category] = []
            categories[category].append(aid)
        return categories

    def export_user_progress(self) -> Dict[str, Any]:
        """Export user progress for backup or analysis."""
        return {
            'user_stats': self.get_user_stats(),
            'achievements_unlocked': st.session_state.achievements_unlocked,
            'prediction_history': st.session_state.prediction_history[-10:],  # Last 10 for privacy
            'export_timestamp': datetime.now().isoformat()
        }
