"""
User preference engine for GoalDiggers application.
Handles storing, retrieving, and applying user preferences across the application.

This module provides the core functionality for the personalization feature.
"""

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

@dataclass
class UserBehavior:
    """Model for tracking user behavior."""
    session_id: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class UserPreference:
    """Model for user preferences."""
    user_id: str
    betting_style: str = "balanced"
    risk_tolerance: float = 0.5
    interface_preferences: Dict[str, Any] = field(default_factory=dict)
    favorite_teams: List[str] = field(default_factory=list)
    favorite_leagues: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class PersonalizedRecommendation:
    """Model for personalized recommendations."""
    title: str
    description: str
    confidence: float
    recommendation_type: str
    user_id: Optional[str] = None
    context: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class PreferenceEngine:
    """
    Engine for managing user preferences and personalized recommendations.
    Provides functionality for tracking user behavior, storing preferences,
    and generating personalized recommendations.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the preference engine.
        
        Args:
            storage_path: Path to store preference data. If None, uses a default path.
        """
        self.storage_path = storage_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "preferences")
        os.makedirs(self.storage_path, exist_ok=True)
        
        # In-memory cache for faster access
        self.preferences_cache: Dict[str, UserPreference] = {}
        self.behavior_cache: Dict[str, List[UserBehavior]] = {}
        
        # Load existing preferences from disk
        self._load_preferences()
        
        logger.info("🧠 PreferenceEngine initialized")
    
    def get_user_preferences(self, user_id: str) -> Optional[UserPreference]:
        """
        Get preferences for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserPreference object or None if not found
        """
        # Check cache first
        if user_id in self.preferences_cache:
            return self.preferences_cache[user_id]
        
        # Load from disk if not in cache
        prefs_path = os.path.join(self.storage_path, f"user_{user_id}.json")
        if os.path.exists(prefs_path):
            try:
                with open(prefs_path, 'r') as f:
                    data = json.load(f)
                preference = UserPreference(**data)
                self.preferences_cache[user_id] = preference
                return preference
            except Exception as e:
                logger.error(f"Error loading preferences for user {user_id}: {e}")
                return None
        
        # Not found
        return None
    
    def update_user_preferences(self, user_id: str, **kwargs) -> bool:
        """
        Update preferences for a specific user.
        
        Args:
            user_id: User identifier
            **kwargs: Preference fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing preferences or create new ones
            preference = self.get_user_preferences(user_id)
            if not preference:
                preference = UserPreference(user_id=user_id)
            
            # Update fields
            for field, value in kwargs.items():
                if hasattr(preference, field):
                    setattr(preference, field, value)
            
            # Update timestamp
            preference.last_updated = datetime.now().isoformat()
            
            # Save to cache
            self.preferences_cache[user_id] = preference
            
            # Save to disk
            prefs_path = os.path.join(self.storage_path, f"user_{user_id}.json")
            with open(prefs_path, 'w') as f:
                json.dump(asdict(preference), f)
            
            return True
        except Exception as e:
            logger.error(f"Error updating preferences for user {user_id}: {e}")
            return False
    
    def track_user_behavior(self, session_id: str, action: str, context: Dict[str, Any]) -> bool:
        """
        Track user behavior for preference learning.
        
        Args:
            session_id: User session identifier
            action: Action performed
            context: Context of the action
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create behavior object
            behavior = UserBehavior(
                session_id=session_id,
                action=action,
                context=context
            )
            
            # Add to cache
            if session_id not in self.behavior_cache:
                self.behavior_cache[session_id] = []
            self.behavior_cache[session_id].append(behavior)
            
            # Save to disk periodically (only if cache reaches a certain size)
            if len(self.behavior_cache[session_id]) >= 10:
                self._save_behavior(session_id)
            
            return True
        except Exception as e:
            logger.error(f"Error tracking behavior for session {session_id}: {e}")
            return False
    
    def get_personalized_recommendations(
        self,
        user_id: str,
        context: str,
        limit: int = 5
    ) -> List[PersonalizedRecommendation]:
        """
        Get personalized recommendations for a user.
        
        Args:
            user_id: User identifier
            context: Context for recommendations (e.g., "dashboard", "match_details")
            limit: Maximum number of recommendations
            
        Returns:
            List of PersonalizedRecommendation objects
        """
        try:
            # Get user preferences
            preference = self.get_user_preferences(user_id)
            if not preference:
                return self._get_default_recommendations(context, limit)
            
            # Generate recommendations based on preferences
            recommendations = []
            
            if context == "dashboard" or context == "sidebar":
                # Add team-based recommendations
                for team in preference.favorite_teams[:2]:
                    recommendations.append(
                        PersonalizedRecommendation(
                            title=f"Check {team} Analysis",
                            description=f"View detailed analysis for {team}",
                            confidence=0.9,
                            recommendation_type="team",
                            user_id=user_id,
                            context=context
                        )
                    )
                
                # Add betting style recommendations
                if preference.betting_style == "conservative":
                    recommendations.append(
                        PersonalizedRecommendation(
                            title="Low-Risk Opportunities",
                            description="View matches with high prediction confidence",
                            confidence=0.85,
                            recommendation_type="betting_style",
                            user_id=user_id,
                            context=context
                        )
                    )
                elif preference.betting_style == "aggressive":
                    recommendations.append(
                        PersonalizedRecommendation(
                            title="High-Value Opportunities",
                            description="View matches with high-value bets",
                            confidence=0.82,
                            recommendation_type="betting_style",
                            user_id=user_id,
                            context=context
                        )
                    )
            
            elif context == "match_details":
                # Add match-specific recommendations
                recommendations.append(
                    PersonalizedRecommendation(
                        title="Personalized Analysis",
                        description="View analysis based on your preferences",
                        confidence=0.88,
                        recommendation_type="analysis",
                        user_id=user_id,
                        context=context
                    )
                )
                
                # Add betting recommendations based on risk tolerance
                if preference.risk_tolerance < 0.3:
                    recommendations.append(
                        PersonalizedRecommendation(
                            title="Conservative Betting Options",
                            description="View safer betting options for this match",
                            confidence=0.9,
                            recommendation_type="betting",
                            user_id=user_id,
                            context=context
                        )
                    )
                elif preference.risk_tolerance > 0.7:
                    recommendations.append(
                        PersonalizedRecommendation(
                            title="High-Value Betting Options",
                            description="View high-risk, high-reward options",
                            confidence=0.75,
                            recommendation_type="betting",
                            user_id=user_id,
                            context=context
                        )
                    )
            
            # Add general recommendations
            recommendations.append(
                PersonalizedRecommendation(
                    title="Update Preferences",
                    description="Refine your preferences for better recommendations",
                    confidence=0.7,
                    recommendation_type="general",
                    user_id=user_id,
                    context=context
                )
            )
            
            # Return limited number of recommendations
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return self._get_default_recommendations(context, limit)
    
    def get_adaptive_interface_config(self, user_id: str) -> Dict[str, Any]:
        """
        Get adaptive interface configuration based on user preferences.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with interface configuration
        """
        try:
            # Get user preferences
            preference = self.get_user_preferences(user_id)
            if not preference:
                return self._get_default_interface_config()
            
            # Get interface preferences
            interface_prefs = preference.interface_preferences
            
            # Build configuration
            config = {
                "theme": interface_prefs.get("theme", "professional"),
                "layout": interface_prefs.get("layout", "detailed"),
                "info_density": interface_prefs.get("info_density", 3),
                "default_teams": preference.favorite_teams[:3],
                "default_leagues": preference.favorite_leagues[:3],
                "betting_style": preference.betting_style,
                "risk_tolerance": preference.risk_tolerance,
                "show_recommendations": True,
                "show_achievements": True,
                "adaptive_settings": {
                    "hide_complex_stats": preference.betting_style == "conservative",
                    "highlight_value_bets": preference.betting_style == "aggressive",
                    "show_risk_warnings": preference.risk_tolerance < 0.4,
                    "show_advanced_options": interface_prefs.get("info_density", 3) > 3
                }
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Error generating interface config for user {user_id}: {e}")
            return self._get_default_interface_config()
    
    def _get_default_recommendations(self, context: str, limit: int) -> List[PersonalizedRecommendation]:
        """
        Get default recommendations for users without preferences.
        
        Args:
            context: Context for recommendations
            limit: Maximum number of recommendations
            
        Returns:
            List of PersonalizedRecommendation objects
        """
        default_recs = [
            PersonalizedRecommendation(
                title="Setup Your Preferences",
                description="Customize your experience for better recommendations",
                confidence=0.95,
                recommendation_type="general",
                context=context
            ),
            PersonalizedRecommendation(
                title="Premier League Analysis",
                description="View the latest Premier League insights",
                confidence=0.8,
                recommendation_type="league",
                context=context
            ),
            PersonalizedRecommendation(
                title="Top Value Bets Today",
                description="Check today's best betting opportunities",
                confidence=0.75,
                recommendation_type="betting",
                context=context
            )
        ]
        
        return default_recs[:limit]
    
    def _get_default_interface_config(self) -> Dict[str, Any]:
        """
        Get default interface configuration.
        
        Returns:
            Dictionary with default interface configuration
        """
        return {
            "theme": "professional",
            "layout": "detailed",
            "info_density": 3,
            "default_teams": [],
            "default_leagues": ["Premier League", "La Liga", "Bundesliga"],
            "betting_style": "balanced",
            "risk_tolerance": 0.5,
            "show_recommendations": True,
            "show_achievements": True,
            "adaptive_settings": {
                "hide_complex_stats": False,
                "highlight_value_bets": True,
                "show_risk_warnings": True,
                "show_advanced_options": False
            }
        }
    
    def _load_preferences(self) -> None:
        """Load all user preferences from disk into cache."""
        try:
            for filename in os.listdir(self.storage_path):
                if filename.startswith("user_") and filename.endswith(".json"):
                    user_id = filename[5:-5]  # Extract user_id from filename
                    try:
                        with open(os.path.join(self.storage_path, filename), 'r') as f:
                            data = json.load(f)
                        self.preferences_cache[user_id] = UserPreference(**data)
                    except Exception as e:
                        logger.warning(f"Error loading preferences for {filename}: {e}")
        except Exception as e:
            logger.error(f"Error loading preferences: {e}")
    
    def _save_behavior(self, session_id: str) -> None:
        """
        Save behavior data for a session to disk.
        
        Args:
            session_id: Session identifier
        """
        try:
            if session_id not in self.behavior_cache:
                return
            
            behaviors = self.behavior_cache[session_id]
            if not behaviors:
                return
            
            # Create behavior directory if it doesn't exist
            behavior_dir = os.path.join(self.storage_path, "behavior")
            os.makedirs(behavior_dir, exist_ok=True)
            
            # Save behaviors to disk
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            behavior_path = os.path.join(behavior_dir, f"behavior_{session_id}_{timestamp}.json")
            
            with open(behavior_path, 'w') as f:
                json.dump([asdict(b) for b in behaviors], f)
            
            # Clear cache
            self.behavior_cache[session_id] = []
            
        except Exception as e:
            logger.error(f"Error saving behavior for session {session_id}: {e}")
