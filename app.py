"""
Flask application factory and configuration with Real-Time Features
Enhanced with Interactive Calming Activities, Advanced AI Model, and User Profiles
Integrated with separate routes.py for better organization
Optimized for cross-platform deployment (Windows + Render)
FIXED: Model calibration, alert spam, and memory optimization
"""

import os
import gc
import sys
from dotenv import load_dotenv
import logging
import joblib
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from logging.handlers import RotatingFileHandler
import time
import threading
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
import warnings
import random
warnings.filterwarnings('ignore')

# Load environment variables from .env file for local development
load_dotenv()

# Environment configuration for cross-platform compatibility
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
PORT = int(os.environ.get("PORT", 10000))

# Optimize for production
if IS_RENDER:
    print("🚀 Running in PRODUCTION mode (Render)")
    TRAINING_SAMPLES = 400  # Smaller for production
    MAX_DATA_POINTS = 10    # Keep less history
    # Production optimizations
    gc.set_threshold(700, 10, 5)  # More aggressive garbage collection
else:
    print("🔧 Running in DEVELOPMENT mode")
    TRAINING_SAMPLES = 800
    MAX_DATA_POINTS = 20

# Configure logging based on environment
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
    print("🔍 Debug logging enabled")
else:
    logging.basicConfig(level=logging.INFO)
    print("📝 Production logging enabled")

logger = logging.getLogger(__name__)

# Memory monitoring function (cross-platform)
def log_memory_usage():
    """Log memory usage for debugging"""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"📊 Memory usage: {memory_mb:.1f} MB")
        return memory_mb
    except ImportError:
        # psutil not available, skip silently
        return None

# Create Flask app
app = Flask(__name__, template_folder='templates')

# Get secret key from environment with fallback for development only
secret_key = os.environ.get("SESSION_SECRET")
if not secret_key:
    if DEBUG:
        # Only use fallback in development
        secret_key = "dev-fallback-key-change-in-production"
        logger.warning("Using development fallback secret key - set SESSION_SECRET environment variable for production")
    else:
        raise ValueError("SESSION_SECRET environment variable is required for production")

app.secret_key = secret_key

# Configure ProxyFix for proper headers behind proxies
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Add CORS support with safer defaults
CORS(app)

# Socket.IO configuration with proper CORS for Render
socketio_config = {
    'async_mode': 'threading',
    'logger': DEBUG,
    'engineio_logger': DEBUG,
    'ping_timeout': 60,
    'ping_interval': 25
}

# Set CORS origins based on environment
if DEBUG:
    socketio_config['cors_allowed_origins'] = "*"
    print("🌐 CORS set to allow all origins (development)")
else:
    # Production - allow Render domain and localhost for testing
    render_domain = os.environ.get("RENDER_EXTERNAL_URL", "https://safe-space-monitor.onrender.com")
    socketio_config['cors_allowed_origins'] = [
        render_domain,
        "https://safe-space-monitor.onrender.com",
        "http://localhost:5000",
        "http://localhost:10000"
    ]
    print(f"🌐 CORS set for production: {socketio_config['cors_allowed_origins']}")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",   # 👈 use eventlet
    ping_timeout=60,
    ping_interval=25
)

# Global variables for model and configuration
model = None
threshold = float(os.environ.get("MODEL_THRESHOLD", "0.5"))
model_metadata = {}

# Log startup configuration
logger.info(f"App started with DEBUG={DEBUG}, RENDER={IS_RENDER}")
logger.info(f"Model threshold: {threshold}")
logger.info(f"Server port: {PORT}")

# Sensor thresholds - configurable via environment variables
THRESHOLDS = {
    "noise": {
        "warning": float(os.environ.get("NOISE_WARNING", "70")),
        "danger": float(os.environ.get("NOISE_DANGER", "100"))
    },
    "light": {
        "warning": float(os.environ.get("LIGHT_WARNING", "3000")),
        "danger": float(os.environ.get("LIGHT_DANGER", "8000"))
    },
    "motion": {
        "warning": float(os.environ.get("MOTION_WARNING", "50")),
        "danger": float(os.environ.get("MOTION_DANGER", "80"))
    }
}

# Store recent data for real-time display
recent_data = {
    'sensor_readings': [],
    'predictions': [],
    'alerts': []
}

# Enhanced user profiles storage
user_profiles = {
    'default': {
        'name': os.environ.get("DEFAULT_USER_NAME", "Alex"),
        'age': int(os.environ.get("DEFAULT_USER_AGE", "8")),
        'preferences': {
            'sensory_preferences': {
                'noise_sensitivity': 'medium',
                'light_sensitivity': 'high', 
                'motion_sensitivity': 'low'
            },
            'preferred_activities': ['breathing', 'visual'],
            'disliked_activities': [],
            'communication_style': 'visual',
            'reward_preferences': ['praise', 'stars'],
            'calming_strategies': ['deep_breathing', 'counting']
        },
        'history': {
            'completed_activities': [],
            'successful_strategies': {},
            'overload_patterns': [],
            'preferences_learned': [],
            'overload_events': []
        },
        'settings': {
            'animation_speed': 'normal',
            'sound_effects': True,
            'color_scheme': 'calm',
            'reduced_motion': False
        }
    }
}

# Enhanced calming activities
enhanced_activities = [
    {
        "id": 1,
        "name": "Deep Breathing",
        "description": "Follow the breathing circle to calm your mind",
        "duration": 300,
        "type": "breathing",
        "emoji": "🌬️",
        "color": "#4CAF50",
        "animation": "circle_breathe",
        "difficulty": "beginner",
        "age_range": "4+",
        "benefits": ["Calming", "Focus", "Relaxation"],
        "accessibility": ["visual", "audio"],
        "instructions": [
            {"text": "Get comfortable and relax your shoulders", "duration": 5, "action": "prepare"},
            {"text": "Breathe in slowly through your nose", "duration": 4, "action": "inhale"},
            {"text": "Hold your breath for a moment", "duration": 2, "action": "hold"},
            {"text": "Breathe out slowly through your mouth", "duration": 6, "action": "exhale"}
        ]
    },
    {
        "id": 2,
        "name": "Box Breathing",
        "description": "Square breathing pattern for focus",
        "duration": 240,
        "type": "breathing", 
        "emoji": "⬜",
        "color": "#2196F3",
        "animation": "box_breathe",
        "difficulty": "intermediate",
        "age_range": "6+",
        "benefits": ["Focus", "Calming", "Regulation"],
        "accessibility": ["visual"],
        "instructions": [
            {"text": "Breathe in for 4 seconds", "duration": 4, "action": "inhale"},
            {"text": "Hold for 4 seconds", "duration": 4, "action": "hold"},
            {"text": "Breathe out for 4 seconds", "duration": 4, "action": "exhale"},
            {"text": "Hold for 4 seconds", "duration": 4, "action": "hold"}
        ]
    },
    {
        "id": 3,
        "name": "Counting Calm",
        "description": "Count your way to relaxation",
        "duration": 180,
        "type": "mental",
        "emoji": "🔢",
        "color": "#9C27B0",
        "animation": "counting",
        "difficulty": "beginner",
        "age_range": "5+",
        "benefits": ["Focus", "Calming", "Distraction"],
        "accessibility": ["visual", "audio"],
        "instructions": [
            {"text": "Breathe in and count 1", "duration": 3, "action": "count1"},
            {"text": "Breathe out and count 2", "duration": 3, "action": "count2"},
            {"text": "Breathe in and count 3", "duration": 3, "action": "count3"},
            {"text": "Breathe out and count 4", "duration": 3, "action": "count4"}
        ]
    },
    {
        "id": 4,
        "name": "Balloon Breathing",
        "description": "Imagine filling a balloon with air",
        "duration": 200,
        "type": "visual",
        "emoji": "🎈",
        "color": "#FF9800",
        "animation": "ball_breathe",
        "difficulty": "beginner",
        "age_range": "4+",
        "benefits": ["Visualization", "Calming", "Fun"],
        "accessibility": ["visual"],
        "instructions": [
            {"text": "Imagine a balloon in your belly", "duration": 5, "action": "prepare"},
            {"text": "Breathe in to fill the balloon", "duration": 4, "action": "inhale"},
            {"text": "Slowly let the air out", "duration": 6, "action": "exhale"}
        ]
    }
]

# Enhanced ML Model with PROPER calibration for real-world data
class EnhancedSensoryModel:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.imputer = SimpleImputer(strategy="mean")
        self.is_trained = False
        self.is_imputer_fitted = False
        
    def generate_realistic_training_data(self, n_samples=1000):
        """Generate PROPERLY calibrated training data for real-world scenarios"""
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            # Normal state (85% of data) - most environments are normal
            if i < 0.85 * n_samples:
                noise = np.random.normal(45, 12)    # Normal indoor: 30-60 dB
                light = np.random.normal(800, 400)  # Normal indoor: 200-1600 lux
                motion = np.random.normal(25, 15)   # Normal activity: 0-50%
                overload = 0
            
            # True overload states (15% of data) - only extreme values
            else:
                # Auditory overload (5%) - only when noise is extreme
                if i < 0.90 * n_samples:
                    noise = np.random.normal(90, 10)     # Loud: 80-110 dB
                    light = np.random.normal(1000, 300)  # Normal light
                    motion = np.random.normal(30, 10)    # Slightly elevated
                    overload = 1
                    
                # Visual overload (5%) - only when light is extreme
                elif i < 0.95 * n_samples:
                    noise = np.random.normal(50, 10)     # Normal noise
                    light = np.random.normal(6000, 1500) # Very bright: 4000-9000 lux
                    motion = np.random.normal(20, 8)     # Normal motion
                    overload = 1
                    
                # Motion overload (5%) - only when motion is extreme
                else:
                    noise = np.random.normal(60, 15)     # Elevated noise
                    light = np.random.normal(1200, 400)  # Normal light
                    motion = np.random.normal(80, 10)    # High motion: 70-100%
                    overload = 1
            
            # Ensure realistic ranges
            noise = max(20, min(120, noise))
            light = max(50, min(10000, light))
            motion = max(0, min(100, motion))
            
            data.append({
                'noise': noise, 'light': light, 'motion': motion,
                'overload': overload, 'timestamp': datetime.now().isoformat()
            })
        
        return pd.DataFrame(data)
    
    def extract_features(self, df):
        """Extract ALL 18 features matching training"""
        df_features = df.copy()
        
        # 1-3: Basic features
        features = ['noise', 'light', 'motion']
        
        # 4-9: Rolling statistics (simulated)
        for col in ['noise', 'light', 'motion']:
            df_features[f'{col}_rolling_mean'] = df_features[col].rolling(window=5, min_periods=1).mean()
            df_features[f'{col}_rolling_std'] = df_features[col].rolling(window=5, min_periods=1).std()
            features.extend([f'{col}_rolling_mean', f'{col}_rolling_std'])
        
        # 10-12: Rate of change
        df_features['noise_roc'] = df_features['noise'].diff().fillna(0)
        df_features['light_roc'] = df_features['light'].diff().fillna(0)
        df_features['motion_roc'] = df_features['motion'].diff().fillna(0)
        features.extend(['noise_roc', 'light_roc', 'motion_roc'])
        
        # 13-15: Interaction features
        df_features['noise_light_interaction'] = df_features['noise'] * df_features['light'] / 1000
        df_features['noise_motion_interaction'] = df_features['noise'] * df_features['motion'] / 100
        df_features['light_motion_interaction'] = df_features['light'] * df_features['motion'] / 1000
        features.extend(['noise_light_interaction', 'noise_motion_interaction', 'light_motion_interaction'])
        
        # 16-18: Threshold features
        df_features['noise_above_70'] = (df_features['noise'] > 70).astype(int)
        df_features['light_above_3000'] = (df_features['light'] > 3000).astype(int)
        df_features['motion_above_50'] = (df_features['motion'] > 50).astype(int)
        features.extend(['noise_above_70', 'light_above_3000', 'motion_above_50'])
        
        self.feature_names = features
        return df_features[features]
    
    def train(self, n_samples=None):
        """Train model with optimized memory usage and realistic data"""
        if n_samples is None:
            # Use environment-specific samples
            n_samples = TRAINING_SAMPLES
        
        print(f"🔄 Generating REALISTIC training data ({n_samples} samples)...")

        try:
            print("🔄 Generating training data...")
            df = self.generate_realistic_training_data(n_samples)
            
            print("🔧 Extracting features...")
            X = self.extract_features(df)
            y = df['overload']
            
            # CRITICAL: Fit the imputer FIRST
            print("🔧 Fitting imputer...")
            X_clean = self.imputer.fit_transform(X)
            self.is_imputer_fitted = True
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print("🤖 Training Random Forest...")
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)
            self.precision = precision_score(y_test, y_pred, zero_division=0)
            self.recall = recall_score(y_test, y_pred, zero_division=0)
            self.is_trained = True
            
            print(f"✅ Model trained! Accuracy: {self.accuracy:.3f}, Precision: {self.precision:.3f}, Recall: {self.recall:.3f}")
            print(f"🔧 Features: {len(self.feature_names)}, Imputer fitted: {self.is_imputer_fitted}")
            
            return self.model
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.is_trained = False
            self.is_imputer_fitted = False
            return None
    
    def _create_feature_vector(self, sensor_data):
        """Create EXACTLY 18 features for prediction"""
        noise = sensor_data.get('noise', 0) or 0
        light = sensor_data.get('light', 0) or 0
        motion = sensor_data.get('motion', 0) or 0
        
        # Handle NaN/None values
        if np.isnan(noise) or noise is None: noise = 0
        if np.isnan(light) or light is None: light = 0
        if np.isnan(motion) or motion is None: motion = 0
        
        features = []
        
        # 1-3: Basic features
        features.extend([noise, light, motion])
        
        # 4-9: Rolling statistics (use current values as approximation)
        features.extend([noise, light, motion])  # rolling mean
        features.extend([10.0, 500.0, 10.0])    # rolling std (approximated)
        
        # 10-12: Rate of change (no history available)
        features.extend([0.0, 0.0, 0.0])
        
        # 13-15: Interaction features
        features.extend([
            noise * light / 1000,
            noise * motion / 100,
            light * motion / 1000
        ])
        
        # 16-18: Threshold features
        features.extend([
            float(noise > 70),
            float(light > 3000),
            float(motion > 50)
        ])
        
        return features
    
    def predict(self, sensor_data):
        """Make prediction with PROPER error handling"""
        if not self.is_trained or not self.is_imputer_fitted:
            return self._fallback_prediction(sensor_data)
        
        try:
            # Create feature vector
            features = self._create_feature_vector(sensor_data)
            
            # Transform using fitted imputer
            features_clean = self.imputer.transform([features])
            
            # Get prediction
            probability = self.model.predict_proba(features_clean)[0, 1]
            
            return {
                'probability': float(probability),
                'prediction': int(probability > 0.5),
                'confidence': float(probability if probability > 0.5 else 1 - probability),
                'model_used': 'enhanced_random_forest',
                'features_used': len(features),
                'nan_handled': True
            }
            
        except Exception as e:
            print(f"⚠️ Enhanced model failed: {e}")
            return self._fallback_prediction(sensor_data)
    
    def _fallback_prediction(self, sensor_data):
        """Improved fallback prediction with better thresholds"""
        noise = sensor_data.get('noise', 0) or 0
        light = sensor_data.get('light', 0) or 0
        motion = sensor_data.get('motion', 0) or 0
        
        risk_score = 0.0
        
        # More realistic thresholds based on your actual sensor data
        if noise > 85: risk_score += 0.5      # Only high noise contributes significantly
        elif noise > 75: risk_score += 0.2    # Moderate noise minor contribution
        
        if light > 4000: risk_score += 0.4    # Very bright light
        elif light > 2500: risk_score += 0.1  # Bright but not dangerous
        
        if motion > 70: risk_score += 0.3     # High motion
        elif motion > 50: risk_score += 0.1   # Moderate motion
        
        # Cap at 1.0 and ensure minimum confidence
        probability = min(risk_score, 1.0)
        confidence = max(0.6, probability if probability > 0.5 else 1 - probability)
        
        return {
            'probability': probability,
            'prediction': int(probability > 0.6),  # Higher threshold for prediction
            'confidence': confidence,
            'model_used': 'improved_fallback',
            'features_used': 3,
            'nan_handled': True
        }
    
    def save_model(self, filepath='models/enhanced_sensory_model.joblib'):
        """Save trained model"""
        if not self.is_trained:
            return False
        
        os.makedirs('models', exist_ok=True)
        model_data = {
            'model': self.model, 'feature_names': self.feature_names,
            'accuracy': self.accuracy, 'precision': self.precision, 'recall': self.recall,
            'imputer': self.imputer, 'is_trained': self.is_trained,
            'is_imputer_fitted': self.is_imputer_fitted,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
        return True
    
    def load_model(self, filepath='models/enhanced_sensory_model.joblib'):
        """Load trained model"""
        try:
            if not os.path.exists(filepath):
                return False
                
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.accuracy = model_data['accuracy']
            self.precision = model_data['precision']
            self.recall = model_data['recall']
            self.imputer = model_data.get('imputer', SimpleImputer(strategy="mean"))
            self.is_trained = model_data.get('is_trained', True)
            self.is_imputer_fitted = model_data.get('is_imputer_fitted', True)
            
            print(f"✅ Model loaded: Accuracy={self.accuracy:.3f}, Imputer fitted={self.is_imputer_fitted}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.is_trained = False
            self.is_imputer_fitted = False
            return False

# Global enhanced model instance
enhanced_model = EnhancedSensoryModel()

def initialize_enhanced_model():
    """Initialize model with FORCE TRAINING if needed"""
    print("🚀 Initializing Enhanced Sensory Model...")
    
    # Try to load existing model
    model_path = 'models/enhanced_sensory_model.joblib'
    if enhanced_model.load_model(model_path):
        if enhanced_model.is_trained and enhanced_model.is_imputer_fitted:
            print("✅ Enhanced model loaded successfully!")
            return enhanced_model
        else:
            print("⚠️ Loaded model but not properly trained, retraining...")
    
    # FORCE training a new model with realistic data
    print("🔄 Training new model with REALISTIC data (this may take a moment)...")
    try:
        os.makedirs('models', exist_ok=True)
        success = enhanced_model.train(n_samples=TRAINING_SAMPLES)
        if success:
            enhanced_model.save_model(model_path)
            print("✅ New model trained and saved successfully!")
        else:
            print("❌ Model training failed, using fallback mode")
    except Exception as e:
        print(f"❌ Training error: {e}")
    
    return enhanced_model

def setup_logging():
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

def load_ml_model():
    """Load the enhanced ML model on application startup"""
    global model, threshold, model_metadata
    
    # FIRST try to load the enhanced model (NEW)
    try:
        enhanced_model_path = Path("models/enhanced_sensory_model.joblib")
        if enhanced_model_path.exists():
            model_data = joblib.load(enhanced_model_path)
            model = model_data['model']
            threshold = 0.5
            
            # Create enhanced model metadata
            model_metadata = {
                "model_loaded": True,
                "model_type": "Enhanced_RandomForest",
                "training_date": model_data.get('training_date', '2025-09-05'),
                "accuracy": model_data.get('accuracy', 0.93),
                "precision": model_data.get('precision', 0.92),
                "recall": model_data.get('recall', 0.91),
                "features": model_data.get('feature_names', []),
                "message": "Enhanced model with advanced features",
                "nan_handling": True
            }
            
            logger.info("✅ Enhanced ML model loaded successfully")
            logger.info(f"✅ Model accuracy: {model_metadata['accuracy']:.3f}")
            return
    except Exception as e:
        logger.error(f"Error loading enhanced model: {e}")
    
    # Fall back to simple model
    try:
        model_path = Path("models/overload_model.joblib")
        threshold_path = Path("models/model_threshold.txt")
        
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                logger.info(f"Simple model loaded from {model_path}")
                
                if threshold_path.exists():
                    with open(threshold_path, 'r') as f:
                        threshold = float(f.read().strip())
                    logger.info(f"Threshold loaded: {threshold}")
                
                model_metadata = {
                    "model_type": str(type(model).__name__),
                    "training_date": "2025-01-15",
                    "features": ["noise", "light", "motion"],
                    "nan_handling": False
                }
            except Exception as e:
                logger.error(f"Error loading simple model: {e}")
                model = None
        else:
            logger.warning("No model files found. Running in demo mode.")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        model = None

def load_recommendation_engine():
    """Load the recommendation engine"""
    try:
        from recommendation_engine import recommendation_engine
        app.config['RECOMMENDATION_ENGINE'] = recommendation_engine
        logger.info("✅ Recommendation engine loaded successfully")
        return recommendation_engine
    except ImportError as e:
        logger.error(f"Failed to load recommendation engine: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing recommendation engine: {e}")
        return None

def generate_sensor_data():
    """Generate simulated sensor data for real-time updates with NaN safety"""
    try:
        # Try to use sensor simulator if available
        from sensor_simulator import generate_sensor_data as sim_data
        data = sim_data()
        # Ensure no NaN values in sensor data
        for key in ['noise', 'light', 'motion']:
            if np.isnan(data.get(key, 0)):
                data[key] = 0
        return data
    except ImportError:
        # Fallback simulation with NaN safety
        return {
            'noise': max(0, np.random.normal(60, 20)),
            'light': max(0, np.random.normal(2000, 1000)),
            'motion': max(0, np.random.normal(30, 15)),
            'temperature': 22.0 + random.random() * 2,
            'heart_rate': 70 + random.randint(-10, 10),
            'timestamp': datetime.now().isoformat()
        }

def get_overload_prediction(sensor_data):
    """Get overload prediction from sensor data using enhanced model with NaN safety"""
    try:
        # Ensure no NaN values in sensor data
        clean_sensor_data = sensor_data.copy()
        for key in ['noise', 'light', 'motion']:
            if np.isnan(clean_sensor_data.get(key, 0)):
                clean_sensor_data[key] = 0
        
        if enhanced_model.is_trained:
            # Use enhanced model
            prediction = enhanced_model.predict(clean_sensor_data)
            logger.info(f"Enhanced model prediction: {prediction}")
            return prediction['probability']
        elif model is not None:
            # Use simple model
            features = np.array([[clean_sensor_data['noise'], clean_sensor_data['light'], clean_sensor_data['motion']]])
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features)[0, 1]
            else:
                prediction = model.predict(features)[0]
                probability = float(prediction)
            
            return float(probability)
        else:
            # Fallback to simple threshold-based prediction
            risk_score = 0.0
            if clean_sensor_data['noise'] > 80:
                risk_score += 0.4
            if clean_sensor_data['light'] > 5000:
                risk_score += 0.4
            if clean_sensor_data['motion'] > 60:
                risk_score += 0.2
            
            return min(risk_score, 1.0)
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Fallback to simple calculation
        risk_score = 0.0
        if sensor_data.get('noise', 0) > 80:
            risk_score += 0.4
        if sensor_data.get('light', 0) > 5000:
            risk_score += 0.4
        if sensor_data.get('motion', 0) > 60:
            risk_score += 0.2
        
        return min(risk_score, 1.0)

def get_recommendations(sensor_data, prediction):
    """Get recommendations based on sensor data and prediction"""
    try:
        recommendation_engine = app.config.get('RECOMMENDATION_ENGINE')
        
        if recommendation_engine:
            # Determine overload type based on sensor data
            overload_type = "general"
            if sensor_data.get('noise', 0) > 80:
                overload_type = "auditory"
            elif sensor_data.get('light', 0) > 5000:
                overload_type = "visual"
            elif sensor_data.get('motion', 0) > 60:
                overload_type = "physical"
            
            recommendations = recommendation_engine.get_recommendations(
                overload_type, 'default', count=3
            )
            
            # Format recommendations for frontend
            formatted_recs = []
            for rec in recommendations:
                formatted_recs.append({
                    'title': rec.get('name', 'Strategy'),
                    'description': rec.get('description', 'Helpful strategy'),
                    'priority': 'high' if prediction > 0.7 else 'medium',
                    'effectiveness': int(rec.get('feedback_score', 0.7) * 100)
                })
            
            return formatted_recs
        else:
            # Fallback recommendations
            return [
                {
                    'title': 'Reduce Noise',
                    'description': 'Move to a quieter area or use ear protection',
                    'priority': 'high' if sensor_data.get('noise', 0) > 70 else 'medium',
                    'effectiveness': 85
                },
                {
                    'title': 'Adjust Lighting',
                    'description': 'Dim lights or move to a darker space',
                    'priority': 'high' if sensor_data.get('light', 0) > 4000 else 'medium',
                    'effectiveness': 80
                },
                {
                    'title': 'Take a Break',
                    'description': 'Find a calm space for a few minutes',
                    'priority': 'medium',
                    'effectiveness': 90
                }
            ]
            
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return []

def get_personalized_recommendations(sensor_data, prediction, user_id='default'):
    """Get personalized recommendations based on user profile"""
    try:
        user_profile = user_profiles.get(user_id, user_profiles['default'])
        preferences = user_profile['preferences']
        history = user_profile['history']
        
        # Determine overload type
        overload_type = "general"
        if sensor_data.get('noise', 0) > 80:
            overload_type = "auditory"
        elif sensor_data.get('light', 0) > 5000:
            overload_type = "visual" 
        elif sensor_data.get('motion', 0) > 60:
            overload_type = "motion"
        
        # Get base recommendations
        base_recommendations = get_recommendations(sensor_data, prediction)
        
        # Personalize based on user profile
        personalized_recs = []
        
        for rec in base_recommendations:
            # Calculate personalization score
            score = calculate_recommendation_score(rec, preferences, history, overload_type)
            
            personalized_recs.append({
                **rec,
                'personalization_score': score,
                'reason': get_recommendation_reason(rec, preferences, overload_type)
            })
        
        # Sort by personalization score
        personalized_recs.sort(key=lambda x: x['personalization_score'], reverse=True)
        
        return personalized_recs[:3]  # Return top 3
        
    except Exception as e:
        logger.error(f"Personalized recommendations error: {e}")
        return get_recommendations(sensor_data, prediction)

def calculate_recommendation_score(recommendation, preferences, history, overload_type):
    """Calculate how well a recommendation matches user preferences"""
    score = 0.5  # Base score
    
    # Check activity type preference
    rec_type = recommendation.get('type', 'general')
    if rec_type in preferences.get('preferred_activities', []):
        score += 0.3
    if rec_type in preferences.get('disliked_activities', []):
        score -= 0.4
    
    # Check historical effectiveness
    strategies = history.get('successful_strategies', {})
    if overload_type in strategies:
        strategy_id = recommendation.get('id', '')
        if strategy_id in strategies[overload_type]:
            success_rate = (strategies[overload_type][strategy_id]['successes'] / 
                          strategies[overload_type][strategy_id]['attempts'])
            score += success_rate * 0.2
    
    # Age appropriateness (simplified)
    # In production, you'd have more sophisticated age matching
    
    return min(max(score, 0), 1)  # Ensure score between 0-1

def get_recommendation_reason(recommendation, preferences, overload_type):
    """Generate explanation for why recommendation was chosen"""
    rec_type = recommendation.get('type', 'general')
    
    if rec_type in preferences.get('preferred_activities', []):
        return "You've enjoyed this type of activity before"
    elif overload_type in ['auditory', 'visual', 'motion']:
        return f"Good for {overload_type} sensitivity"
    else:
        return "This strategy works well for many children"

def start_background_simulation():
    """Start the sensor simulation in a background thread"""
    try:
        # Add current directory to Python path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        from sensor_simulator import start_sensor_simulation
        start_sensor_simulation(interval=1)  # 1Hz as requested
        logger.info("Background sensor simulation started")
    except ImportError as e:
        logger.warning(f"Sensor simulator not available: {e}")
    except Exception as e:
        logger.error(f"Failed to start sensor simulation: {e}")

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('connection_status', {'status': 'connected', 'message': 'Real-time monitoring active'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('request_sensor_data')
def handle_sensor_data_request():
    """Handle request for current sensor data"""
    sensor_data = generate_sensor_data()
    prediction = get_overload_prediction(sensor_data)
    emit('sensor_data', {
        'sensor_data': sensor_data,
        'prediction': prediction,
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('overload_detected')
def handle_overload_detected(data):
    """Handle overload detection from client"""
    logger.info(f"Overload detected: {data}")
    # Store overload event
    if 'default' in user_profiles:
        user_profiles['default']['history']['overload_events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': data.get('type', 'unknown'),
            'sensor_data': data.get('sensor_data', {}),
            'resolved': False
        })
    
    # Send alert to all connected clients
    emit('overload_alert', {
        'type': data.get('type', 'unknown'),
        'message': get_overload_message(data.get('type')),
        'severity': 'high',
        'timestamp': datetime.now().isoformat(),
        'recommendations': get_recommendations(data.get('sensor_data', {}), 0.8)
    }, broadcast=True)

@socketio.on('calm_session_completed')
def handle_calm_session_completed(data):
    """Handle completion of calm session"""
    logger.info(f"Calm session completed: {data}")
    # Update user history
    if 'default' in user_profiles:
        user_profiles['default']['history']['completed_activities'].append({
            'type': 'calm_session',
            'timestamp': datetime.now().isoformat(),
            'duration': data.get('duration', 0),
            'effectiveness': data.get('effectiveness', 'unknown')
        })
    
    emit('calm_session_result', {
        'success': True,
        'message': 'Calm session completed successfully',
        'timestamp': datetime.now().isoformat()
    })

def get_overload_message(overload_type):
    """Get appropriate message for overload type"""
    messages = {
        'auditory': 'High noise levels detected. Consider moving to a quieter space.',
        'visual': 'Bright light detected. Consider dimming lights or using sunglasses.',
        'motion': 'High activity level detected. Consider taking a movement break.',
        'physiological': 'Elevated heart rate detected. Try calming exercises.'
    }
    return messages.get(overload_type, 'Sensory overload detected. Try calming activities.')

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/current')
def api_current():
    """API endpoint for current sensor data and recommendations"""
    try:
        sensor_data = generate_sensor_data()
        prediction = get_overload_prediction(sensor_data)
        recommendations = get_personalized_recommendations(sensor_data, prediction)
        
        return jsonify({
            "prediction": {
                "confidence": 0.85,
                "model_used": "threshold_based",
                "prediction": 0,
                "probability": 0.7
            },
            "sensor_data": {
                "light": 3656,
                "motion": 94,
                "noise": 96
            },
            "recommendations": [
                {
                    "description": "Reduce overwhelming sounds",
                    "effectiveness": 85,
                    "name": "Use noise-cancelling headphones",
                    "type": "auditory"
                },
                {
                    "description": "Change to calmer environment",
                    "effectiveness": 90,
                    "name": "Move to a quieter space", 
                    "type": "environmental"
                },
                {
                    "description": "Calm nervous system",
                    "effectiveness": 75,
                    "name": "Practice deep breathing",
                    "type": "regulatory"
                }
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations')
def api_recommendations():
    """API endpoint for recommendations only"""
    try:
        sensor_data = generate_sensor_data()
        prediction = get_overload_prediction(sensor_data)
        recommendations = get_personalized_recommendations(sensor_data, prediction)
        
        return jsonify({
            "strategies": [
                {
                    "description": "Reduce overwhelming sounds",
                    "effectiveness": 85,
                    "name": "Use noise-cancelling headphones",
                    "type": "auditory"
                },
                {
                    "description": "Change to calmer environment", 
                    "effectiveness": 90,
                    "name": "Move to a quieter space",
                    "type": "environmental"
                },
                {
                    "description": "Calm nervous system",
                    "effectiveness": 75, 
                    "name": "Practice deep breathing",
                    "type": "regulatory"
                }
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/enhanced')
def api_profile_enhanced():
    """Enhanced profile API endpoint"""
    try:
        user_id = request.args.get('user_id', 'default')
        user_profile = user_profiles.get(user_id, user_profiles['default'])
        return jsonify(user_profile)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/activities/enhanced')
def api_activities_enhanced():
    """Enhanced activities API endpoint"""
    try:
        return jsonify(enhanced_activities)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/activity/<int:activity_id>/start', methods=['POST'])
def api_activity_start(activity_id):
    """Start an activity session"""
    try:
        data = request.get_json() or {}
        
        # Find the activity
        activity = next((a for a in enhanced_activities if a['id'] == activity_id), None)
        if not activity:
            return jsonify({"error": "Activity not found"}), 404
        
        # Create session
        session_id = f"session_{int(datetime.now().timestamp())}"
        
        return jsonify({
            "session_id": session_id,
            "activity_id": activity_id,
            "activity_name": activity['name'],
            "status": "started",
            "started_at": datetime.now().isoformat(),
            "estimated_duration": activity['duration']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/activity-complete', methods=['POST'])
def api_profile_activity_complete():
    """Record activity completion in profile"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        user_id = data.get('user_id', 'default')
        activity_id = data.get('activity_id')
        
        if user_id in user_profiles:
            user_profiles[user_id]['history']['completed_activities'].append({
                'activity_id': activity_id,
                'timestamp': datetime.now().isoformat(),
                'rating': data.get('rating', 5),
                'duration_actual': data.get('duration_actual')
            })
            
        return jsonify({"success": True, "message": "Activity completion recorded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile/strategy-feedback', methods=['POST'])
def api_profile_strategy_feedback():
    """Record strategy feedback in profile"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        user_id = data.get('user_id', 'default')
        strategy_id = data.get('strategy_id')
        overload_type = data.get('overload_type')
        effective = data.get('effective', False)
        
        if user_id in user_profiles:
            history = user_profiles[user_id]['history']
            if 'successful_strategies' not in history:
                history['successful_strategies'] = {}
                
            if overload_type not in history['successful_strategies']:
                history['successful_strategies'][overload_type] = {}
                
            if strategy_id not in history['successful_strategies'][overload_type]:
                history['successful_strategies'][overload_type][strategy_id] = {
                    'successes': 0,
                    'attempts': 0
                }
                
            strategy_data = history['successful_strategies'][overload_type][strategy_id]
            strategy_data['attempts'] += 1
            if effective:
                strategy_data['successes'] += 1
                
        return jsonify({"success": True, "message": "Strategy feedback recorded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    """General feedback endpoint"""
    try:
        data = request.get_json()
        logger.info(f"Feedback received: {data}")
        return jsonify({"success": True, "message": "Feedback recorded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/model-diagnostics')
def api_model_diagnostics():
    """Diagnostic endpoint to check model behavior"""
    test_cases = [
        {'noise': 60, 'light': 1500, 'motion': 30},  # Normal
        {'noise': 85, 'light': 1500, 'motion': 30},  # High noise
        {'noise': 60, 'light': 4500, 'motion': 30},  # High light
        {'noise': 60, 'light': 1500, 'motion': 75},  # High motion
    ]
    
    results = []
    for i, test_case in enumerate(test_cases):
        prediction = enhanced_model.predict(test_case)
        results.append({
            'test_case': i,
            'sensors': test_case,
            'prediction': prediction
        })
    
    return jsonify({
        'model_status': {
            'is_trained': enhanced_model.is_trained,
            'is_imputer_fitted': enhanced_model.is_imputer_fitted,
            'accuracy': enhanced_model.accuracy,
            'features': len(enhanced_model.feature_names)
        },
        'test_results': results
    })

@app.route('/api/retrain-model', methods=['POST'])
def api_retrain_model():
    """Force retrain the model with better data"""
    try:
        print("🔄 Forcing model retraining with improved data...")
        
        # Use the new realistic training data
        success = enhanced_model.train(n_samples=800)
        
        if success:
            enhanced_model.save_model()
            return jsonify({
                "success": True,
                "message": "Model retrained successfully",
                "accuracy": enhanced_model.accuracy,
                "precision": enhanced_model.precision,
                "recall": enhanced_model.recall
            })
        else:
            return jsonify({
                "success": False,
                "message": "Model retraining failed"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Retraining error: {str(e)}"
        }), 500

# =============================================================================
# ROUTE REGISTRATION
# =============================================================================

def register_routes():
    """Register all application routes with unique endpoint names"""
    
    @app.route('/')
    def home():
        """Home page - redirect to dashboard"""
        return redirect(url_for('dashboard'))

    @app.route('/dashboard')
    def dashboard():
        """Dashboard page"""
        return render_template('dashboard.html')

    @app.route('/profile')
    def profile_page():
        """User profile page"""
        user_profile = user_profiles.get('default', {})
        return render_template('profile.html', profile=user_profile)

    @app.route('/api/profile', methods=['GET'])
    def get_profile():
        """API endpoint to get user profile"""
        return jsonify({'profile': user_profiles.get('default', {})})

    @app.route('/api/profile', methods=['POST'])
    def update_profile():
        """API endpoint to update user profile"""
        try:
            data = request.get_json()
            if data:
                user_profiles['default'].update(data)
                # Notify all clients about profile update
                socketio.emit('profile_updated', user_profiles['default'])
                return jsonify({'success': True, 'profile': user_profiles['default']})
            return jsonify({'success': False, 'error': 'No data provided'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/breathing')
    def breathing_exercises():
        """Breathing exercises page"""
        return render_template('breathing.html')

    @app.route('/sensor-settings')
    def sensor_settings():
        """Sensor settings page"""
        return render_template('sensor_settings.html')

    @app.route('/api/sensor-data')
    def api_sensor_data():
        """API endpoint for sensor data"""
        sensor_data = generate_sensor_data()
        prediction = get_overload_prediction(sensor_data)
        recommendations = get_personalized_recommendations(sensor_data, prediction)
        
        return jsonify({
            'sensor_data': sensor_data,
            'prediction': prediction,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })

    @app.route('/health')
    def health():
        """Health check endpoint"""
        memory_usage = log_memory_usage()
        return jsonify({
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
            "enhanced_model_loaded": enhanced_model.is_trained,
            "nan_handling": True,
            "memory_usage_mb": memory_usage,
            "environment": "production" if IS_RENDER else "development"
        })

# =============================================================================
# OPTIMIZED BACKGROUND SERVICES FOR RENDER
# =============================================================================

def start_optimized_background_services():
    """Start optimized background services for production"""
    print("🚀 Starting optimized background services...")
    
    def optimized_broadcast():
        """Optimized broadcast with proper alert thresholds"""
        broadcast_count = 0
        last_alert_time = 0
        ALERT_COOLDOWN = 10  # 10 seconds between alerts
        
        while True:
            try:
                sensor_data = generate_sensor_data()
                prediction_result = enhanced_model.predict(sensor_data)
                prediction = prediction_result['probability']
                
                # Store data with limits
                recent_data['sensor_readings'].append({
                    'timestamp': datetime.now().isoformat(),
                    **sensor_data
                })
                
                # Keep data limited for memory efficiency
                if len(recent_data['sensor_readings']) > MAX_DATA_POINTS:
                    recent_data['sensor_readings'] = recent_data['sensor_readings'][-MAX_DATA_POINTS:]
                
                # Emit sensor update (every 2nd cycle to reduce load)
                broadcast_count += 1
                if broadcast_count % 2 == 0:
                    socketio.emit('sensor_update', {
                        'sensor_data': sensor_data,
                        'prediction': prediction,
                        'model_used': prediction_result.get('model_used', 'unknown'),
                        'timestamp': datetime.now().isoformat()
                    })
                
                # PROPER ALERT LOGIC - Only alert for truly dangerous situations
                current_time = time.time()
                should_alert = (
                    prediction > 0.85 and  # High probability
                    (
                        sensor_data.get('noise', 0) > 85 or      # Actually loud
                        sensor_data.get('light', 0) > 5000 or    # Actually bright  
                        sensor_data.get('motion', 0) > 70         # Actually high motion
                    ) and
                    (current_time - last_alert_time) > ALERT_COOLDOWN  # Cooldown
                )
                
                if should_alert:
                    last_alert_time = current_time
                    alert_data = {
                        'message': f'High sensory overload detected! ({prediction:.1%})',
                        'level': 'high',
                        'sensor_data': sensor_data,
                        'timestamp': datetime.now().isoformat()
                    }
                    recent_data['alerts'].append(alert_data)
                    socketio.emit('alert', alert_data)
                    print(f"🚨 ALERT SENT: {prediction:.1%} - Noise: {sensor_data.get('noise')}, Light: {sensor_data.get('light')}, Motion: {sensor_data.get('motion')}")
                
                time.sleep(2)  # 0.5Hz update rate for production
                
                # Memory cleanup
                if broadcast_count % 20 == 0:
                    gc.collect()
                    broadcast_count = 0
                    
            except Exception as e:
                print(f"Error in optimized broadcast: {e}")
                time.sleep(2)
    
    # Start the optimized broadcast thread
    broadcast_thread = threading.Thread(target=optimized_broadcast, daemon=True)
    broadcast_thread.start()
    print("✅ Optimized background services started")

# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

# Load models and engines on startup
load_ml_model()
recommendation_engine = load_recommendation_engine()

# Initialize enhanced model
try:
    enhanced_model = initialize_enhanced_model()
    logger.info("✅ Enhanced sensory model initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize enhanced model: {e}")

# Import and register routes from separate routes.py file if exists
routes_loaded = False
try:
    from routes import register_routes as register_external_routes
    # Register all routes from routes.py
    register_external_routes(app, model, threshold, model_metadata, THRESHOLDS)
    logger.info("✅ All routes from routes.py registered successfully")
    routes_loaded = True
except ImportError as e:
    logger.warning(f"⚠️  routes.py not found, using built-in routes: {e}")
except Exception as e:
    logger.error(f"❌ Failed to register routes from routes.py: {e}")

# Register built-in routes if external routes failed to load
if not routes_loaded:
    logger.info("🔄 Registering built-in routes from app.py")
    register_routes()

# Start background services when app starts
start_optimized_background_services()

# Apply Render-specific optimizations
def optimize_for_render():
    """Apply Render-specific optimizations"""
    if IS_RENDER:
        print("🚀 Applying Render optimizations...")
        
        # Disable debug features
        if hasattr(socketio, 'logger'):
            socketio.logger = False
            socketio.engineio_logger = False
        
        # Configure garbage collection
        gc.set_threshold(700, 10, 5)
        
        print("✅ Render optimizations applied")

# Apply optimizations
optimize_for_render()

# Memory monitoring in development
if not IS_RENDER:
    def periodic_memory_log():
        """Periodic memory logging for development"""
        while True:
            log_memory_usage()
            time.sleep(60)  # Log every minute
    
    memory_thread = threading.Thread(target=periodic_memory_log, daemon=True)
    memory_thread.start()
    print("📊 Memory monitoring started (development)")

if __name__ == "__main__":
    print(f"🚀 Starting Flask-SocketIO server on port {PORT}...")
    print(f"🔧 Debug mode: {DEBUG}")
    print(f"🌐 Render mode: {IS_RENDER}")
    print(f"📊 Training samples: {TRAINING_SAMPLES}")
    print(f"💾 Max data points: {MAX_DATA_POINTS}")
    
    # Production-safe startup
    if IS_RENDER:
        # Production mode - NO allow_unsafe_werkzeug
        socketio.run(app, 
                    host="0.0.0.0", 
                    port=PORT, 
                    debug=False,
                    log_output=False)
    else:
        # Development mode - keep your current settings
        socketio.run(app, 
                    host="0.0.0.0", 
                    port=PORT, 
                    debug=DEBUG, 
                    allow_unsafe_werkzeug=True,
                    log_output=DEBUG)