#!/usr/bin/env python3
"""
Achievement System for GoalDiggers Platform
Phase 5.3: Missing Component Implementation

Provides gamification features including badges, progress tracking, and user engagement.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Import Streamlit safely
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit not available for achievement system")

@dataclass
class Achievement:
    """Achievement data structure."""
    id: str
    name: str
    description: str
    icon: str
    category: str
    requirement: int
    requirement_type: str  # 'predictions', 'leagues', 'accuracy', etc.
    unlocked: bool = False
    progress: int = 0
    unlock_date: Optional[datetime] = None

class AchievementSystem:
    """
    Achievement System for GoalDiggers Platform
    
    Manages user achievements, badges, and gamification features.
    """
    
    def __init__(self):
        """Initialize achievement system."""
        self.logger = logger
        self.achievements = self._initialize_achievements()
        self.user_stats = self._initialize_user_stats()
        self.logger.info("🏆 Achievement System initialized")
    
    def _initialize_achievements(self) -> Dict[str, Achievement]:
        """Initialize the 16 achievement categories."""
        achievements = {
            # Prediction Achievements
            'first_prediction': Achievement(
                id='first_prediction',
                name='🎯 First Prediction',
                description='Make your first match prediction',
                icon='🎯',
                category='Predictions',
                requirement=1,
                requirement_type='predictions'
            ),
            'prediction_master': Achievement(
                id='prediction_master',
                name='🔮 Prediction Master',
                description='Make 100 successful predictions',
                icon='🔮',
                category='Predictions',
                requirement=100,
                requirement_type='predictions'
            ),
            
            # League Achievements
            'league_explorer': Achievement(
                id='league_explorer',
                name='🌍 League Explorer',
                description='Analyze teams from all 6 supported leagues',
                icon='🌍',
                category='Leagues',
                requirement=6,
                requirement_type='leagues'
            ),
            'cross_league_expert': Achievement(
                id='cross_league_expert',
                name='⚔️ Cross-League Expert',
                description='Make 50 cross-league predictions',
                icon='⚔️',
                category='Cross-League',
                requirement=50,
                requirement_type='cross_league_predictions'
            ),
            'cross_league_selector': Achievement(
                id='cross_league_selector',
                name='🌐 Cross-League Selector',
                description='Select teams from different leagues 10 times',
                icon='🌐',
                category='Team Selection',
                requirement=10,
                requirement_type='cross_league_selections'
            ),
            
            # Accuracy Achievements
            'accuracy_ace': Achievement(
                id='accuracy_ace',
                name='🎪 Accuracy Ace',
                description='Achieve 80%+ prediction accuracy',
                icon='🎪',
                category='Accuracy',
                requirement=80,
                requirement_type='accuracy_percentage'
            ),
            'perfect_week': Achievement(
                id='perfect_week',
                name='💎 Perfect Week',
                description='Get 100% accuracy for a week',
                icon='💎',
                category='Accuracy',
                requirement=100,
                requirement_type='weekly_accuracy'
            ),
            
            # Engagement Achievements
            'daily_user': Achievement(
                id='daily_user',
                name='📅 Daily User',
                description='Use platform for 7 consecutive days',
                icon='📅',
                category='Engagement',
                requirement=7,
                requirement_type='consecutive_days'
            ),
            'power_user': Achievement(
                id='power_user',
                name='⚡ Power User',
                description='Use platform for 30 days total',
                icon='⚡',
                category='Engagement',
                requirement=30,
                requirement_type='total_days'
            ),
            
            # Value Betting Achievements
            'value_hunter': Achievement(
                id='value_hunter',
                name='💰 Value Hunter',
                description='Find 25 value betting opportunities',
                icon='💰',
                category='Value Betting',
                requirement=25,
                requirement_type='value_bets'
            ),

            # Team Selection Achievements
            'team_selector': Achievement(
                id='team_selector',
                name='⚽ Team Selector',
                description='Select teams 25 times',
                icon='⚽',
                category='Team Selection',
                requirement=25,
                requirement_type='team_selections'
            ),
            'league_combination_master': Achievement(
                id='league_combination_master',
                name='🏆 League Combination Master',
                description='Try all possible league combinations',
                icon='🏆',
                category='Team Selection',
                requirement=15,  # 6 leagues = 15 unique combinations
                requirement_type='league_combinations'
            ),
            'profit_master': Achievement(
                id='profit_master',
                name='📈 Profit Master',
                description='Achieve positive ROI over 100 bets',
                icon='📈',
                category='Value Betting',
                requirement=100,
                requirement_type='profitable_bets'
            ),
            
            # Team Analysis Achievements
            'team_specialist': Achievement(
                id='team_specialist',
                name='🏆 Team Specialist',
                description='Analyze the same team 20 times',
                icon='🏆',
                category='Analysis',
                requirement=20,
                requirement_type='team_analyses'
            ),
            'data_analyst': Achievement(
                id='data_analyst',
                name='📊 Data Analyst',
                description='View detailed statistics 50 times',
                icon='📊',
                category='Analysis',
                requirement=50,
                requirement_type='detailed_views'
            ),
            
            # Special Achievements
            'early_adopter': Achievement(
                id='early_adopter',
                name='🚀 Early Adopter',
                description='Join GoalDiggers in its early phase',
                icon='🚀',
                category='Special',
                requirement=1,
                requirement_type='early_access'
            ),
            'feature_explorer': Achievement(
                id='feature_explorer',
                name='🔍 Feature Explorer',
                description='Use all premium features',
                icon='🔍',
                category='Special',
                requirement=10,
                requirement_type='features_used'
            ),
            
            # Social Achievements
            'community_member': Achievement(
                id='community_member',
                name='👥 Community Member',
                description='Share insights with the community',
                icon='👥',
                category='Social',
                requirement=5,
                requirement_type='shared_insights'
            ),
            'mentor': Achievement(
                id='mentor',
                name='🎓 Mentor',
                description='Help other users with predictions',
                icon='🎓',
                category='Social',
                requirement=10,
                requirement_type='helped_users'
            )
        }
        
        return achievements
    
    def _initialize_user_stats(self) -> Dict[str, Any]:
        """Initialize user statistics for achievement tracking."""
        return {
            'predictions_made': 0,
            'leagues_analyzed': set(),
            'cross_league_predictions': 0,
            'cross_league_correct': 0,
            'cross_league_accuracy': 0.0,
            'accuracy_percentage': 0.0,
            'current_streak': 0,
            'best_streak': 0,
            'league_combinations': set(),
            'consecutive_days': 0,
            'total_days': 0,
            'value_bets_found': 0,
            'profitable_bets': 0,
            'team_analyses': {},
            'detailed_views': 0,
            'features_used': set(),
            'shared_insights': 0,
            'helped_users': 0,
            'last_activity': None
        }
    
    def update_user_progress(self, action: str, data: Any = None):
        """Update user progress for achievement tracking."""
        try:
            if action == 'prediction_made':
                self.user_stats['predictions_made'] += 1
                if data and 'league' in data:
                    self.user_stats['leagues_analyzed'].add(data['league'])
                if data and data.get('is_cross_league'):
                    self.user_stats['cross_league_predictions'] += 1
                    # Track league combinations for cross-league achievements
                    if 'home_league' in data and 'away_league' in data:
                        combination = tuple(sorted([data['home_league'], data['away_league']]))
                        self.user_stats['league_combinations'].add(combination)

                # Handle prediction result for streak tracking
                if data and 'is_correct' in data:
                    if data['is_correct']:
                        self.user_stats['current_streak'] += 1
                        self.user_stats['best_streak'] = max(
                            self.user_stats['best_streak'],
                            self.user_stats['current_streak']
                        )
                        # Track cross-league accuracy separately
                        if data.get('is_cross_league'):
                            self.user_stats['cross_league_correct'] += 1
                    else:
                        self.user_stats['current_streak'] = 0  # Reset streak on incorrect prediction

                # Update cross-league accuracy
                if self.user_stats['cross_league_predictions'] > 0:
                    self.user_stats['cross_league_accuracy'] = (
                        self.user_stats['cross_league_correct'] /
                        self.user_stats['cross_league_predictions']
                    ) * 100
            
            elif action == 'accuracy_updated':
                if data:
                    self.user_stats['accuracy_percentage'] = data
            
            elif action == 'daily_activity':
                self.user_stats['total_days'] += 1
                # Update consecutive days logic would go here
            
            elif action == 'value_bet_found':
                self.user_stats['value_bets_found'] += 1
            
            elif action == 'feature_used':
                if data:
                    self.user_stats['features_used'].add(data)

            elif action == 'league_explored':
                # Track league exploration from team selection
                if data and 'league' in data:
                    self.user_stats['leagues_analyzed'].add(data['league'])
                    self.logger.debug(f"League explored: {data['league']}")

            elif action == 'cross_league_selection':
                # Track cross-league team selection
                if data and 'home_league' in data and 'away_league' in data:
                    home_league = data['home_league']
                    away_league = data['away_league']
                    if home_league != away_league and home_league != "Unknown League" and away_league != "Unknown League":
                        # Track as cross-league activity
                        self.user_stats['cross_league_selections'] = self.user_stats.get('cross_league_selections', 0) + 1
                        # Track league combination
                        combination = tuple(sorted([home_league, away_league]))
                        self.user_stats['league_combinations'].add(combination)
                        self.logger.debug(f"Cross-league selection: {home_league} vs {away_league}")

            elif action == 'team_selection':
                # Track general team selection activity
                if data and 'leagues' in data:
                    for league in data['leagues']:
                        if league != "Unknown League":
                            self.user_stats['leagues_analyzed'].add(league)
                    self.user_stats['team_selections'] = self.user_stats.get('team_selections', 0) + 1

            # Check for newly unlocked achievements
            self._check_achievements()

        except Exception as e:
            self.logger.error(f"Error updating user progress: {e}")
    
    def _check_achievements(self):
        """Check if any achievements should be unlocked."""
        for achievement_id, achievement in self.achievements.items():
            if not achievement.unlocked:
                if self._is_achievement_unlocked(achievement):
                    achievement.unlocked = True
                    achievement.unlock_date = datetime.now()
                    self.logger.info(f"🎉 Achievement unlocked: {achievement.name}")
    
    def _is_achievement_unlocked(self, achievement: Achievement) -> bool:
        """Check if a specific achievement should be unlocked."""
        req_type = achievement.requirement_type
        requirement = achievement.requirement
        
        if req_type == 'predictions':
            return self.user_stats['predictions_made'] >= requirement
        elif req_type == 'leagues':
            return len(self.user_stats['leagues_analyzed']) >= requirement
        elif req_type == 'cross_league_predictions':
            return self.user_stats['cross_league_predictions'] >= requirement
        elif req_type == 'accuracy_percentage':
            return self.user_stats['accuracy_percentage'] >= requirement
        elif req_type == 'total_days':
            return self.user_stats['total_days'] >= requirement
        elif req_type == 'value_bets':
            return self.user_stats['value_bets_found'] >= requirement
        elif req_type == 'features_used':
            return len(self.user_stats['features_used']) >= requirement
        elif req_type == 'early_access':
            return True  # Always unlocked for early users
        
        return False

    def get_prediction_context(self, prediction_data: Dict[str, Any] = None,
                             match_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get achievement context relevant to current prediction for display alongside predictions.

        Args:
            prediction_data: Current prediction data (optional)
            match_context: Match context including teams and leagues (optional)

        Returns:
            Dictionary containing relevant achievements, progress, and personalized insights
        """
        try:
            context = {
                'relevant_achievements': [],
                'progress_indicators': {},
                'milestone_alerts': [],
                'personalized_insights': {},
                'user_level_info': {},
                'streak_status': {},
                'next_milestones': []
            }

            # Get current user stats
            current_stats = self.user_stats.copy()

            # Determine relevant achievements based on prediction context
            relevant_achievement_ids = self._get_relevant_achievement_ids(prediction_data, match_context)

            for achievement_id in relevant_achievement_ids:
                if achievement_id in self.achievements:
                    achievement = self.achievements[achievement_id]
                    progress = self.get_achievement_progress(achievement_id)

                    context['relevant_achievements'].append({
                        'id': achievement_id,
                        'name': achievement.name,
                        'description': achievement.description,
                        'category': achievement.category,
                        'progress': progress,
                        'is_unlocked': achievement.unlocked,
                        'icon': achievement.icon
                    })

                    # Add progress indicator
                    context['progress_indicators'][achievement_id] = {
                        'current': progress,
                        'target': 100.0,
                        'percentage': min(progress, 100),
                        'is_close': progress >= 80  # Close to completion
                    }

            # Generate milestone alerts for achievements close to completion
            context['milestone_alerts'] = self._generate_milestone_alerts(current_stats)

            # Add personalized insights based on user level and achievements
            context['personalized_insights'] = self._generate_personalized_insights(
                current_stats, prediction_data, match_context
            )

            # Add user level information
            context['user_level_info'] = self._get_user_level_info(current_stats)

            # Add streak status
            context['streak_status'] = self._get_streak_status(current_stats)

            # Add next milestones
            context['next_milestones'] = self._get_next_milestones(current_stats)

            return context

        except Exception as e:
            self.logger.error(f"Error generating prediction context: {e}")
            return {
                'relevant_achievements': [],
                'progress_indicators': {},
                'milestone_alerts': [],
                'personalized_insights': {'error': 'Context generation temporarily unavailable'},
                'user_level_info': {},
                'streak_status': {},
                'next_milestones': []
            }

    def _get_relevant_achievement_ids(self, prediction_data: Dict[str, Any] = None,
                                    match_context: Dict[str, Any] = None) -> List[str]:
        """Get achievement IDs relevant to current prediction context."""
        try:
            relevant_ids = []

            # Always include prediction-related achievements
            prediction_achievements = [
                'first_prediction', 'prediction_master', 'accuracy_expert',
                'streak_champion'
            ]
            relevant_ids.extend([aid for aid in prediction_achievements if aid in self.achievements])

            # Add cross-league achievements if applicable
            if match_context and match_context.get('is_cross_league'):
                cross_league_achievements = ['cross_league_pioneer']
                relevant_ids.extend([aid for aid in cross_league_achievements if aid in self.achievements])

            # Add league-specific achievements
            if match_context and 'league' in match_context:
                league_achievements = ['league_explorer']
                relevant_ids.extend([aid for aid in league_achievements if aid in self.achievements])

            return list(set(relevant_ids))  # Remove duplicates

        except Exception as e:
            self.logger.error(f"Error getting relevant achievement IDs: {e}")
            return []

    def get_achievement_progress(self, achievement_id: str) -> float:
        """Get progress percentage for a specific achievement."""
        if achievement_id not in self.achievements:
            return 0.0
        
        achievement = self.achievements[achievement_id]
        if achievement.unlocked:
            return 100.0
        
        req_type = achievement.requirement_type
        requirement = achievement.requirement
        
        if req_type == 'predictions':
            progress = self.user_stats['predictions_made']
        elif req_type == 'leagues':
            progress = len(self.user_stats['leagues_analyzed'])
        elif req_type == 'cross_league_predictions':
            progress = self.user_stats['cross_league_predictions']
        elif req_type == 'accuracy_percentage':
            progress = self.user_stats['accuracy_percentage']
        elif req_type == 'total_days':
            progress = self.user_stats['total_days']
        elif req_type == 'value_bets':
            progress = self.user_stats['value_bets_found']
        elif req_type == 'features_used':
            progress = len(self.user_stats['features_used'])
        elif req_type == 'cross_league_selections':
            progress = self.user_stats.get('cross_league_selections', 0)
        elif req_type == 'team_selections':
            progress = self.user_stats.get('team_selections', 0)
        elif req_type == 'league_combinations':
            progress = len(self.user_stats.get('league_combinations', set()))
        else:
            progress = 0
        
        return min(100.0, (progress / requirement) * 100.0)

    def _generate_milestone_alerts(self, current_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate milestone alerts for achievements close to completion."""
        try:
            alerts = []

            for achievement_id, achievement in self.achievements.items():
                if not achievement.unlocked:
                    progress = self.get_achievement_progress(achievement_id)

                    # Alert if achievement is 80% or more complete
                    if progress >= 80:
                        alerts.append({
                            'achievement_id': achievement_id,
                            'name': achievement.name,
                            'progress': progress,
                            'remaining': 100 - progress,
                            'message': f"You're {progress:.0f}% of the way to {achievement.name}!"
                        })

            return alerts[:3]  # Return top 3 alerts

        except Exception as e:
            self.logger.error(f"Error generating milestone alerts: {e}")
            return []

    def _generate_personalized_insights(self, current_stats: Dict[str, Any],
                                      prediction_data: Dict[str, Any] = None,
                                      match_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate personalized insights based on user achievements and stats."""
        try:
            insights = {
                'experience_level': 'Beginner',
                'strength_areas': [],
                'improvement_suggestions': [],
                'achievement_recommendations': []
            }

            # Determine experience level
            predictions_made = current_stats.get('predictions_made', 0)
            accuracy = current_stats.get('accuracy_percentage', 0)

            if predictions_made >= 100 and accuracy >= 80:
                insights['experience_level'] = 'Expert'
            elif predictions_made >= 50 and accuracy >= 70:
                insights['experience_level'] = 'Advanced'
            elif predictions_made >= 20 and accuracy >= 60:
                insights['experience_level'] = 'Intermediate'

            # Identify strength areas
            if accuracy >= 75:
                insights['strength_areas'].append('High prediction accuracy')
            if current_stats.get('cross_league_predictions', 0) >= 10:
                insights['strength_areas'].append('Cross-league analysis')
            if len(current_stats.get('leagues_analyzed', set())) >= 4:
                insights['strength_areas'].append('Multi-league expertise')

            # Generate improvement suggestions
            if accuracy < 60:
                insights['improvement_suggestions'].append('Focus on detailed match analysis to improve accuracy')
            if current_stats.get('cross_league_predictions', 0) < 5:
                insights['improvement_suggestions'].append('Try analyzing cross-league matches for diverse insights')

            return insights

        except Exception as e:
            self.logger.error(f"Error generating personalized insights: {e}")
            return {'experience_level': 'Beginner', 'strength_areas': [], 'improvement_suggestions': []}

    def _get_user_level_info(self, current_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get user level information based on achievements and stats."""
        try:
            unlocked_count = len([a for a in self.achievements.values() if a.unlocked])
            total_count = len(self.achievements)

            # Calculate level based on achievements and predictions
            predictions_made = current_stats.get('predictions_made', 0)
            level = min(10, max(1, (predictions_made // 10) + (unlocked_count // 2)))

            return {
                'current_level': level,
                'achievements_unlocked': unlocked_count,
                'total_achievements': total_count,
                'completion_percentage': (unlocked_count / total_count) * 100 if total_count > 0 else 0,
                'next_level_requirement': f"{((level + 1) * 10) - predictions_made} more predictions"
            }

        except Exception as e:
            self.logger.error(f"Error getting user level info: {e}")
            return {'current_level': 1, 'achievements_unlocked': 0, 'total_achievements': 0}

    def _get_streak_status(self, current_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Get current streak status and information."""
        try:
            # For now, use a simple streak calculation
            # In a real implementation, this would track actual prediction streaks
            current_streak = current_stats.get('current_streak', 0)
            best_streak = current_stats.get('best_streak', 0)

            return {
                'current_streak': current_streak,
                'best_streak': best_streak,
                'is_active': current_streak > 0,
                'next_milestone': 5 if current_streak < 5 else 10 if current_streak < 10 else current_streak + 5,
                'streak_message': f"🔥 {current_streak} correct in a row!" if current_streak > 0 else "Start a new streak!"
            }

        except Exception as e:
            self.logger.error(f"Error getting streak status: {e}")
            return {'current_streak': 0, 'best_streak': 0, 'is_active': False}

    def _get_next_milestones(self, current_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get next achievement milestones to work towards."""
        try:
            milestones = []

            # Find achievements that are not unlocked and calculate progress
            for achievement_id, achievement in self.achievements.items():
                if not achievement.unlocked:
                    progress = self.get_achievement_progress(achievement_id)
                    milestones.append({
                        'achievement_id': achievement_id,
                        'name': achievement.name,
                        'description': achievement.description,
                        'progress': progress,
                        'category': achievement.category,
                        'icon': achievement.icon
                    })

            # Sort by progress (closest to completion first)
            milestones.sort(key=lambda x: x['progress'], reverse=True)

            return milestones[:5]  # Return top 5 next milestones

        except Exception as e:
            self.logger.error(f"Error getting next milestones: {e}")
            return []

    def render_sidebar_stats(self):
        """Render achievement statistics in sidebar."""
        if not STREAMLIT_AVAILABLE:
            return
        
        try:
            # Achievement summary
            unlocked_count = sum(1 for a in self.achievements.values() if a.unlocked)
            total_count = len(self.achievements)
            
            st.sidebar.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                text-align: center;
                color: white;
            ">
                <h4 style="margin: 0; color: white;">🏆 Achievements</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; font-weight: bold;">
                    {unlocked_count}/{total_count}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recent achievements
            recent_achievements = [a for a in self.achievements.values() if a.unlocked][-3:]
            if recent_achievements:
                st.sidebar.markdown("**🎉 Recent Achievements:**")
                for achievement in recent_achievements:
                    st.sidebar.markdown(f"• {achievement.icon} {achievement.name}")
            
            # Progress on next achievement
            next_achievement = self._get_next_achievement()
            if next_achievement:
                progress = self.get_achievement_progress(next_achievement.id)
                st.sidebar.markdown("**🎯 Next Goal:**")
                st.sidebar.markdown(f"{next_achievement.icon} {next_achievement.name}")
                st.sidebar.progress(progress / 100.0)
                st.sidebar.markdown(f"Progress: {progress:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error rendering sidebar stats: {e}")
    
    def get_achievements(self) -> Dict[str, Achievement]:
        """Get all achievements."""
        return self.achievements

    def get_unlocked_achievements(self) -> List[Achievement]:
        """Get all unlocked achievements."""
        return [achievement for achievement in self.achievements.values() if achievement.unlocked]

    def get_locked_achievements(self) -> List[Achievement]:
        """Get all locked achievements."""
        return [achievement for achievement in self.achievements.values() if not achievement.unlocked]

    def _get_next_achievement(self) -> Optional[Achievement]:
        """Get the next achievement to unlock."""
        unlocked_achievements = [a for a in self.achievements.values() if not a.unlocked]
        if not unlocked_achievements:
            return None

        # Return achievement with highest progress
        return max(unlocked_achievements, key=lambda a: self.get_achievement_progress(a.id))
    
    def render_achievements_page(self):
        """Render full achievements page."""
        if not STREAMLIT_AVAILABLE:
            return
        
        try:
            st.markdown("# 🏆 Achievements")
            
            # Achievement categories
            categories = {}
            for achievement in self.achievements.values():
                if achievement.category not in categories:
                    categories[achievement.category] = []
                categories[achievement.category].append(achievement)
            
            # Render by category
            for category, achievements in categories.items():
                st.markdown(f"## {category}")
                
                cols = st.columns(2)
                for i, achievement in enumerate(achievements):
                    with cols[i % 2]:
                        self._render_achievement_card(achievement)
            
        except Exception as e:
            self.logger.error(f"Error rendering achievements page: {e}")
    
    def _render_achievement_card(self, achievement: Achievement):
        """Render individual achievement card."""
        if not STREAMLIT_AVAILABLE:
            return
        
        try:
            progress = self.get_achievement_progress(achievement.id)
            status_color = "#4CAF50" if achievement.unlocked else "#FFC107"
            status_text = "Unlocked" if achievement.unlocked else f"{progress:.1f}%"
            
            st.markdown(f"""
            <div style="
                border: 2px solid {status_color};
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                background: {'#f8f9fa' if achievement.unlocked else '#ffffff'};
            ">
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 2rem; margin-right: 0.5rem;">{achievement.icon}</span>
                    <div>
                        <h4 style="margin: 0; color: #333;">{achievement.name}</h4>
                        <p style="margin: 0; color: {status_color}; font-weight: bold;">{status_text}</p>
                    </div>
                </div>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">{achievement.description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if not achievement.unlocked and progress > 0:
                st.progress(progress / 100.0)
            
        except Exception as e:
            self.logger.error(f"Error rendering achievement card: {e}")

# Factory function for easy import
def get_achievement_system():
    """Get achievement system instance."""
    return AchievementSystem()
