#!/usr/bin/env python3
"""
Coping Strategies Library and Recommendation Engine
Contains 20+ evidence-based strategies for sensory overload
"""

import json
import random
from datetime import datetime
from pathlib import Path

class RecommendationEngine:
    """Handles coping strategies and personalized recommendations"""
    
    def __init__(self):
        self.strategies = self._load_strategies()
        self.user_profiles = {}
        self.feedback_file = Path("data/strategy_feedback.json")
        self.feedback_file.parent.mkdir(exist_ok=True)
        self._load_feedback()
        self._load_user_profiles()
    
    def _load_strategies(self):
        """Load the library of coping strategies"""
        return {
            # Auditory strategies (for noise overload)
            "auditory": [
                {
                    "id": "noise_cancelling_headphones",
                    "name": "Use noise-cancelling headphones",
                    "description": "Block out overwhelming sounds with specialized headphones",
                    "age_min": 4,
                    "age_max": 16,
                    "effectiveness": 0.9,
                    "ease_of_use": 0.8,
                    "equipment_required": True,
                    "emoji": "🎧"
                },
                {
                    "id": "white_noise",
                    "name": "Play white noise or nature sounds",
                    "description": "Soothing background sounds to mask overwhelming noise",
                    "age_min": 2,
                    "age_max": 16,
                    "effectiveness": 0.7,
                    "ease_of_use": 0.9,
                    "equipment_required": False,
                    "emoji": "🌊"
                },
                {
                    "id": "calming_music",
                    "name": "Listen to calming music",
                    "description": "Soft, instrumental music to reduce auditory stress",
                    "age_min": 3,
                    "age_max": 16,
                    "effectiveness": 0.75,
                    "ease_of_use": 0.95,
                    "equipment_required": False,
                    "emoji": "🎵"
                },
                {
                    "id": "ear_plugs",
                    "name": "Use soft ear plugs",
                    "description": "Reduce noise levels while still hearing important sounds",
                    "age_min": 6,
                    "age_max": 16,
                    "effectiveness": 0.65,
                    "ease_of_use": 0.7,
                    "equipment_required": True,
                    "emoji": "🔇"
                },
                {
                    "id": "quiet_space",
                    "name": "Move to a quiet space",
                    "description": "Find a calm, quiet environment to reduce sensory input",
                    "age_min": 4,
                    "age_max": 16,
                    "effectiveness": 0.85,
                    "ease_of_use": 0.6,
                    "equipment_required": False,
                    "emoji": "🏠"
                }
            ],
            
            # Visual strategies (for light overload)
            "visual": [
                {
                    "id": "dim_lights",
                    "name": "Dim the lights",
                    "description": "Reduce brightness to comfortable levels",
                    "age_min": 3,
                    "age_max": 16,
                    "effectiveness": 0.8,
                    "ease_of_use": 0.9,
                    "equipment_required": False,
                    "emoji": "💡"
                },
                {
                    "id": "blue_light_filter",
                    "name": "Use blue light filter",
                    "description": "Reduce eye strain with color temperature adjustment",
                    "age_min": 4,
                    "age_max": 16,
                    "effectiveness": 0.7,
                    "ease_of_use": 0.85,
                    "equipment_required": False,
                    "emoji": "👓"
                },
                {
                    "id": "sunglasses_indoors",
                    "name": "Wear sunglasses indoors",
                    "description": "Reduce light sensitivity with tinted lenses",
                    "age_min": 4,
                    "age_max": 16,
                    "effectiveness": 0.75,
                    "ease_of_use": 0.8,
                    "equipment_required": True,
                    "emoji": "🕶️"
                },
                {
                    "id": "visual_schedule",
                    "name": "Use visual schedule",
                    "description": "Provide predictability with visual cues",
                    "age_min": 3,
                    "age_max": 12,
                    "effectiveness": 0.65,
                    "ease_of_use": 0.7,
                    "equipment_required": False,
                    "emoji": "📅"
                },
                {
                    "id": "reduce_screen_time",
                    "name": "Reduce screen time",
                    "description": "Take a break from bright screens and devices",
                    "age_min": 4,
                    "age_max": 16,
                    "effectiveness": 0.8,
                    "ease_of_use": 0.6,
                    "equipment_required": False,
                    "emoji": "📵"
                }
            ],
            
            # Motion/Tactile strategies
            "motion": [
                {
                    "id": "deep_pressure",
                    "name": "Apply deep pressure",
                    "description": "Calming through weighted blankets or hugs",
                    "age_min": 3,
                    "age_max": 16,
                    "effectiveness": 0.85,
                    "ease_of_use": 0.75,
                    "equipment_required": False,
                    "emoji": "🛌"
                },
                {
                    "id": "fidget_tools",
                    "name": "Use fidget tools",
                    "description": "Provide tactile stimulation with safe objects",
                    "age_min": 4,
                    "age_max": 16,
                    "effectiveness": 0.7,
                    "ease_of_use": 0.9,
                    "equipment_required": True,
                    "emoji": "🎮"
                },
                {
                    "id": "movement_break",
                    "name": "Take movement break",
                    "description": "Release energy with safe, controlled movement",
                    "age_min": 4,
                    "age_max": 16,
                    "effectiveness": 0.75,
                    "ease_of_use": 0.8,
                    "equipment_required": False,
                    "emoji": "🚶"
                },
                {
                    "id": "rocking_chair",
                    "name": "Use rocking chair",
                    "description": "Soothing rhythmic motion for self-regulation",
                    "age_min": 3,
                    "age_max": 16,
                    "effectiveness": 0.8,
                    "ease_of_use": 0.7,
                    "equipment_required": True,
                    "emoji": "🪑"
                },
                {
                    "id": "stretching",
                    "name": "Gentle stretching",
                    "description": "Release tension through simple stretches",
                    "age_min": 5,
                    "age_max": 16,
                    "effectiveness": 0.65,
                    "ease_of_use": 0.85,
                    "equipment_required": False,
                    "emoji": "🧘"
                }
            ],
            
            # Breathing/Regulation strategies
            "regulatory": [
                {
                    "id": "deep_breathing",
                    "name": "Deep breathing exercises",
                    "description": "Calm the nervous system with controlled breathing",
                    "age_min": 5,
                    "age_max": 16,
                    "effectiveness": 0.8,
                    "ease_of_use": 0.9,
                    "equipment_required": False,
                    "emoji": "🌬️"
                },
                {
                    "id": "counting_exercise",
                    "name": "Counting exercise",
                    "description": "Focus mind by counting objects or breaths",
                    "age_min": 4,
                    "age_max": 16,
                    "effectiveness": 0.7,
                    "ease_of_use": 0.95,
                    "equipment_required": False,
                    "emoji": "🔢"
                },
                {
                    "id": "guided_imagery",
                    "name": "Guided imagery",
                    "description": "Visualize calming scenes or stories",
                    "age_min": 6,
                    "age_max": 16,
                    "effectiveness": 0.65,
                    "ease_of_use": 0.7,
                    "equipment_required": False,
                    "emoji": "🌈"
                },
                {
                    "id": "progressive_relaxation",
                    "name": "Progressive muscle relaxation",
                    "description": "Systematically tense and relax muscle groups",
                    "age_min": 8,
                    "age_max": 16,
                    "effectiveness": 0.75,
                    "ease_of_use": 0.6,
                    "equipment_required": False,
                    "emoji": "💪"
                },
                {
                    "id": "mindful_coloring",
                    "name": "Mindful coloring",
                    "description": "Focus attention on coloring activities",
                    "age_min": 4,
                    "age_max": 16,
                    "effectiveness": 0.7,
                    "ease_of_use": 0.8,
                    "equipment_required": True,
                    "emoji": "🎨"
                }
            ]
        }
    
    def _load_feedback(self):
        """Load strategy feedback data"""
        try:
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    self.feedback_data = json.load(f)
            else:
                self.feedback_data = {}
        except:
            self.feedback_data = {}
    
    def _save_feedback(self):
        """Save strategy feedback data"""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def _load_user_profiles(self):
        """Load user profiles from file"""
        profiles_file = Path("data/user_profiles.json")
        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    self.user_profiles = json.load(f)
            except:
                self.user_profiles = {}
        else:
            self.user_profiles = {}
    
    def _save_user_profiles(self):
        """Save user profiles to file"""
        profiles_file = Path("data/user_profiles.json")
        profiles_file.parent.mkdir(exist_ok=True)
        with open(profiles_file, 'w') as f:
            json.dump(self.user_profiles, f, indent=2)
    
    def get_recommendations(self, overload_type, user_id="default", count=3):
        """
        Get personalized recommendations based on overload type and user profile
        """
        if overload_type not in self.strategies:
            return []
        
        # Get user profile
        user_profile = self.user_profiles.get(user_id, {})
        
        # Get strategies for this overload type
        strategies = self.strategies[overload_type]
        
        # Filter by age if user profile available
        if 'age' in user_profile:
            age = user_profile['age']
            strategies = [s for s in strategies if s['age_min'] <= age <= s['age_max']]
        
        # Filter by preferences if available
        if 'preferences' in user_profile:
            preferences = user_profile['preferences']
            if not preferences.get('equipment_required', True):
                strategies = [s for s in strategies if not s['equipment_required']]
        
        # Add feedback scores
        for strategy in strategies:
            strategy['feedback_score'] = self.feedback_data.get(strategy['id'], {}).get('success_rate', 0.5)
        
        # Sort by effectiveness and user feedback
        strategies.sort(key=lambda x: (
            x['feedback_score'] * 0.7 +
            x['effectiveness'] * 0.3
        ), reverse=True)
        
        # Return top recommendations
        return strategies[:count]
    
    def record_feedback(self, strategy_id, was_helpful, user_id="default"):
        """
        Record user feedback about a strategy
        """
        if strategy_id not in self.feedback_data:
            self.feedback_data[strategy_id] = {
                "helpful_count": 0,
                "total_uses": 0,
                "success_rate": 0.5
            }
        
        self.feedback_data[strategy_id]["total_uses"] += 1
        if was_helpful:
            self.feedback_data[strategy_id]["helpful_count"] += 1
        
        # Calculate success rate
        helpful = self.feedback_data[strategy_id]["helpful_count"]
        total = self.feedback_data[strategy_id]["total_uses"]
        self.feedback_data[strategy_id]["success_rate"] = helpful / total if total > 0 else 0.5
        
        self._save_feedback()
        return self.feedback_data[strategy_id]["success_rate"]
    
    def create_user_profile(self, user_id, age, preferences=None):
        """Create or update a user profile"""
        self.user_profiles[user_id] = {
            "age": age,
            "preferences": preferences or {},
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        self._save_user_profiles()
        return self.user_profiles[user_id]
    
    def get_strategy_by_id(self, strategy_id):
        """Get a strategy by its ID"""
        for category in self.strategies.values():
            for strategy in category:
                if strategy['id'] == strategy_id:
                    return strategy
        return None
    
    def get_all_strategies(self):
        """Get all strategies organized by category"""
        return self.strategies
    
    def get_user_profile(self, user_id):
        """Get a user profile by ID"""
        return self.user_profiles.get(user_id, None)
    
    def update_user_preferences(self, user_id, preferences):
        """Update user preferences"""
        if user_id in self.user_profiles:
            self.user_profiles[user_id]["preferences"] = preferences
            self.user_profiles[user_id]["last_updated"] = datetime.now().isoformat()
            self._save_user_profiles()
            return True
        return False
    
    def get_strategy_effectiveness(self, strategy_id):
        """Get effectiveness data for a specific strategy"""
        if strategy_id in self.feedback_data:
            return self.feedback_data[strategy_id]
        return None
    
    def reset_feedback(self, strategy_id=None):
        """Reset feedback data for a strategy or all strategies"""
        if strategy_id:
            if strategy_id in self.feedback_data:
                del self.feedback_data[strategy_id]
        else:
            self.feedback_data = {}
        self._save_feedback()
        return True

# Global instance
recommendation_engine = RecommendationEngine()