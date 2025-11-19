#!/usr/bin/env python3
"""
Gamification Integration for GoalDiggers Platform
Phase 5.3: Missing Component Implementation

Provides gamification features including points, levels, streaks, and user engagement mechanics.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Import Streamlit safely
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit not available for gamification integration")

@dataclass
class UserLevel:
    """User level data structure."""
    level: int
    name: str
    min_points: int
    max_points: int
    icon: str
    benefits: List[str]

@dataclass
class Streak:
    """Streak data structure."""
    type: str
    current: int
    best: int
    last_activity: Optional[datetime]

class GamificationIntegration:
    """
    Gamification Integration for GoalDiggers Platform
    
    Manages user levels, points, streaks, and engagement mechanics.
    """
    
    def __init__(self):
        """Initialize gamification integration."""
        self.logger = logger
        self.levels = self._initialize_levels()
        self.user_data = self._initialize_user_data()
        self.streaks = self._initialize_streaks()
        self.logger.info("🎮 Gamification Integration initialized")
    
    def _initialize_levels(self) -> List[UserLevel]:
        """Initialize user level system."""
        return [
            UserLevel(1, "🌱 Rookie Predictor", 0, 99, "🌱", ["Basic predictions", "Team analysis"]),
            UserLevel(2, "⚽ Football Fan", 100, 249, "⚽", ["Cross-league analysis", "Value betting basics"]),
            UserLevel(3, "📊 Data Analyst", 250, 499, "📊", ["Advanced statistics", "Trend analysis"]),
            UserLevel(4, "🎯 Prediction Pro", 500, 999, "🎯", ["Premium insights", "Custom alerts"]),
            UserLevel(5, "🔮 Oracle", 1000, 1999, "🔮", ["AI model access", "Prediction confidence"]),
            UserLevel(6, "💎 Diamond Expert", 2000, 3999, "💎", ["Exclusive features", "Priority support"]),
            UserLevel(7, "🏆 Champion", 4000, 7999, "🏆", ["Beta features", "Community recognition"]),
            UserLevel(8, "👑 Legend", 8000, 15999, "👑", ["All features", "Influence platform development"]),
            UserLevel(9, "🌟 Master", 16000, 31999, "🌟", ["Master tier benefits", "Special recognition"]),
            UserLevel(10, "🚀 GoalDiggers Elite", 32000, float('inf'), "🚀", ["Elite status", "Lifetime benefits"])
        ]
    
    def _initialize_user_data(self) -> Dict[str, Any]:
        """Initialize user gamification data."""
        return {
            'total_points': 0,
            'current_level': 1,
            'predictions_today': 0,
            'last_activity': None,
            'total_predictions': 0,
            'successful_predictions': 0,
            'value_bets_found': 0,
            'leagues_explored': set(),
            'features_used': set(),
            'daily_goals_completed': 0,
            'weekly_goals_completed': 0,
            'monthly_goals_completed': 0
        }
    
    def _initialize_streaks(self) -> Dict[str, Streak]:
        """Initialize streak tracking."""
        return {
            'daily_login': Streak('daily_login', 0, 0, None),
            'prediction_accuracy': Streak('prediction_accuracy', 0, 0, None),
            'value_betting': Streak('value_betting', 0, 0, None),
            'cross_league': Streak('cross_league', 0, 0, None)
        }
    
    def award_points(self, action: str, amount: int = None, multiplier: float = 1.0):
        """Award points for user actions."""
        try:
            # Default point values for different actions
            point_values = {
                'prediction': 10,
                'accurate_prediction': 25,
                'value_bet_found': 15,
                'daily_login': 5,
                'cross_league_analysis': 20,
                'feature_usage': 8,
                'streak_bonus': 50
            }
            
            points = amount if amount is not None else point_values.get(action, 5)
            final_points = int(points * multiplier)
            
            self.user_data['total_points'] += final_points
            self.user_data['last_activity'] = datetime.now()
            
            # Check for level up
            level_up = self._check_level_up()
            
            self.logger.info(f"� Awarded {final_points} points for {action} (total: {self.user_data['total_points']})")
            
            return {
                'points_awarded': final_points,
                'total_points': self.user_data['total_points'],
                'level_up': level_up
            }
            
        except Exception as e:
            self.logger.error(f"Error awarding points: {e}")
            return {'points_awarded': 0, 'total_points': self.user_data['total_points'], 'level_up': False}
    
    def _check_level_up(self):
        """Check if user should level up."""
        current_points = self.user_data['total_points']
        current_level = self.user_data['current_level']
        
        for level in self.levels:
            if level.level > current_level and current_points >= level.min_points:
                self.user_data['current_level'] = level.level
                self.logger.info(f"🎉 Level up! Now level {level.level}: {level.name}")
                return True
        return False
    
    def update_streak(self, streak_type: str, success: bool = True):
        """Update streak tracking."""
        try:
            if streak_type not in self.streaks:
                return False
            
            streak = self.streaks[streak_type]
            today = datetime.now().date()
            
            if success:
                # Check if this continues the streak
                if streak.last_activity and (today - streak.last_activity.date()).days == 1:
                    streak.current += 1
                elif not streak.last_activity or (today - streak.last_activity.date()).days > 1:
                    streak.current = 1
                
                streak.last_activity = datetime.now()
                
                # Update best streak
                if streak.current > streak.best:
                    streak.best = streak.current
                    self.award_points('streak_bonus', 25)
                    
            else:
                # Streak broken
                streak.current = 0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating streak: {e}")
            return False
    
    def get_current_level(self) -> UserLevel:
        """Get current user level."""
        current_level_num = self.user_data['current_level']
        return next((level for level in self.levels if level.level == current_level_num), self.levels[0])
    
    def get_next_level(self) -> Optional[UserLevel]:
        """Get next level to achieve."""
        current_level_num = self.user_data['current_level']
        return next((level for level in self.levels if level.level == current_level_num + 1), None)
    
    def get_level_progress(self) -> float:
        """Get progress to next level as percentage."""
        current_level = self.get_current_level()
        next_level = self.get_next_level()
        
        if not next_level:
            return 100.0  # Max level reached
        
        current_points = self.user_data['total_points']
        level_start = current_level.min_points
        level_end = next_level.min_points
        
        if level_end == level_start:
            return 100.0
        
        progress = (current_points - level_start) / (level_end - level_start)
        return min(100.0, max(0.0, progress * 100.0))
    
    def get_available_achievements(self) -> List[Dict[str, Any]]:
        """Get list of available achievements."""
        return [
            {
                'id': 'first_prediction',
                'name': 'First Steps',
                'description': 'Make your first prediction',
                'icon': '🎯',
                'points': 25,
                'completed': self.user_data['total_predictions'] > 0
            },
            {
                'id': 'accurate_streak',
                'name': 'Sharp Shooter',
                'description': 'Get 5 predictions right in a row',
                'icon': '🔥',
                'points': 100,
                'completed': self.streaks['prediction_accuracy'].best >= 5
            },
            {
                'id': 'cross_league_master',
                'name': 'League Explorer',
                'description': 'Make predictions across 3 different leagues',
                'icon': '🌍',
                'points': 150,
                'completed': len(self.user_data['leagues_explored']) >= 3
            },
            {
                'id': 'value_hunter',
                'name': 'Value Hunter',
                'description': 'Find 10 value betting opportunities',
                'icon': '💎',
                'points': 200,
                'completed': self.user_data['value_bets_found'] >= 10
            },
            {
                'id': 'daily_warrior',
                'name': 'Daily Warrior',
                'description': 'Login for 7 consecutive days',
                'icon': '⚔️',
                'points': 175,
                'completed': self.streaks['daily_login'].best >= 7
            },
            {
                'id': 'prediction_master',
                'name': 'Prediction Master',
                'description': 'Make 100 predictions',
                'icon': '👑',
                'points': 300,
                'completed': self.user_data['total_predictions'] >= 100
            }
        ]
    
    def get_daily_goals(self) -> List[Dict[str, Any]]:
        """Get daily goals for user engagement."""
        return [
            {
                'id': 'daily_predictions',
                'name': 'Make 3 Predictions',
                'description': 'Analyze and predict 3 matches today',
                'target': 3,
                'current': self.user_data['predictions_today'],
                'reward': 50,
                'icon': '🎯'
            },
            {
                'id': 'cross_league',
                'name': 'Cross-League Analysis',
                'description': 'Compare teams from different leagues',
                'target': 1,
                'current': len([s for s in self.streaks.values() if s.type == 'cross_league' and s.current > 0]),
                'reward': 75,
                'icon': '🌍'
            },
            {
                'id': 'value_betting',
                'name': 'Find Value Bet',
                'description': 'Identify a value betting opportunity',
                'target': 1,
                'current': min(1, self.user_data['value_bets_found']),
                'reward': 100,
                'icon': '💰'
            }
        ]
    
    def render_sidebar_features(self):
        """Render gamification features in sidebar."""
        if not STREAMLIT_AVAILABLE:
            return
        
        try:
            # Current level display
            current_level = self.get_current_level()
            next_level = self.get_next_level()
            progress = self.get_level_progress()
            
            st.sidebar.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                text-align: center;
                color: white;
            ">
                <h4 style="margin: 0; color: white;">🎮 Level</h4>
                <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                    {current_level.icon} {current_level.name}
                </p>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">
                    {self.user_data['total_points']} Points
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress to next level
            if next_level:
                st.sidebar.markdown("**📈 Next Level:**")
                st.sidebar.markdown(f"{next_level.icon} {next_level.name}")
                st.sidebar.progress(progress / 100.0)
                st.sidebar.markdown(f"Progress: {progress:.1f}%")
            
            # Active streaks
            active_streaks = [s for s in self.streaks.values() if s.current > 0]
            if active_streaks:
                st.sidebar.markdown("**🔥 Active Streaks:**")
                for streak in active_streaks[:3]:  # Show top 3
                    streak_name = streak.type.replace('_', ' ').title()
                    st.sidebar.markdown(f"• {streak_name}: {streak.current} days")
            
            # Daily goals
            goals = self.get_daily_goals()
            incomplete_goals = [g for g in goals if g['current'] < g['target']]
            if incomplete_goals:
                st.sidebar.markdown("**🎯 Today's Goals:**")
                for goal in incomplete_goals[:2]:  # Show top 2
                    progress_pct = (goal['current'] / goal['target']) * 100
                    st.sidebar.markdown(f"{goal['icon']} {goal['name']}")
                    st.sidebar.progress(progress_pct / 100.0)
            
        except Exception as e:
            self.logger.error(f"Error rendering sidebar features: {e}")
    
    def render_gamification_page(self):
        """Render full gamification page."""
        if not STREAMLIT_AVAILABLE:
            return
        
        try:
            st.markdown("# 🎮 Gamification")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("## 📊 Your Progress")
                current_level = self.get_current_level()
                
                # Level info
                st.markdown(f"### {current_level.icon} {current_level.name}")
                st.markdown(f"**Points:** {self.user_data['total_points']}")
                st.markdown(f"**Level:** {current_level.level}/10")
                
                # Benefits
                st.markdown("**Benefits:**")
                for benefit in current_level.benefits:
                    st.markdown(f"• {benefit}")
                
                # Progress to next level
                next_level = self.get_next_level()
                if next_level:
                    progress = self.get_level_progress()
                    st.markdown(f"### 📈 Progress to {next_level.name}")
                    st.progress(progress / 100.0)
                    st.markdown(f"{progress:.1f}% complete")
            
            with col2:
                st.markdown("## 🔥 Streaks")
                for streak_type, streak in self.streaks.items():
                    streak_name = streak_type.replace('_', ' ').title()
                    st.markdown(f"**{streak_name}**")
                    st.markdown(f"Current: {streak.current} days")
                    st.markdown(f"Best: {streak.best} days")
                    st.markdown("---")
            
            # Daily goals
            st.markdown("## 🎯 Daily Goals")
            goals = self.get_daily_goals()
            
            for goal in goals:
                progress_pct = min(100, (goal['current'] / goal['target']) * 100)
                completed = goal['current'] >= goal['target']
                
                st.markdown(f"""
                <div style="
                    border: 2px solid {'#4CAF50' if completed else '#FFC107'};
                    border-radius: 8px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                    background: {'#f8f9fa' if completed else '#ffffff'};
                ">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{goal['icon']}</span>
                        <div>
                            <h4 style="margin: 0; color: #333;">{goal['name']}</h4>
                            <p style="margin: 0; color: #666; font-size: 0.9rem;">{goal['description']}</p>
                        </div>
                    </div>
                    <p style="margin: 0; color: {'#4CAF50' if completed else '#FFC107'}; font-weight: bold;">
                        {goal['current']}/{goal['target']} - {goal['reward']} points
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if not completed:
                    st.progress(progress_pct / 100.0)
            
        except Exception as e:
            self.logger.error(f"Error rendering gamification page: {e}")
    
    def track_user_action(self, action: str, data: Any = None):
        """Track user action for gamification."""
        try:
            if action == 'prediction_made':
                self.user_data['predictions_today'] += 1
                self.user_data['total_predictions'] += 1
                self.award_points('prediction_made')
                
                if data and data.get('is_cross_league'):
                    self.award_points('cross_league_prediction')
                    self.update_streak('cross_league', True)
            
            elif action == 'accurate_prediction':
                self.user_data['successful_predictions'] += 1
                self.award_points('accurate_prediction')
                self.update_streak('prediction_accuracy', True)
            
            elif action == 'value_bet_found':
                self.user_data['value_bets_found'] += 1
                self.award_points('value_bet_found')
                self.update_streak('value_betting', True)
            
            elif action == 'daily_login':
                self.award_points('daily_login')
                self.update_streak('daily_login', True)
            
            elif action == 'feature_used':
                if data:
                    self.user_data['features_used'].add(data)
                    self.award_points('feature_exploration')
            
            self.user_data['last_activity'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error tracking user action: {e}")

# Factory function for easy import
def get_gamification_integration():
    """Get gamification integration instance."""
    return GamificationIntegration()
