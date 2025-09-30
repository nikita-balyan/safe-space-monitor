"""
Flask application factory and configuration with Real-Time Features
Enhanced with Interactive Calming Activities, Advanced AI Model, and User Profiles
Integrated with separate routes.py for better organization
"""

import os
import logging
import joblib
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.middleware.proxy_fix import ProxyFix
import sys
from logging.handlers import RotatingFileHandler
import time
import threading
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
import random
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Socket.IO for real-time features
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables for model and configuration
model = None
threshold = 0.5
model_metadata = {}

# Sensor thresholds
THRESHOLDS = {
    "noise": {"warning": 70, "danger": 100},
    "light": {"warning": 3000, "danger": 8000}, 
    "motion": {"warning": 50, "danger": 80}
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
        'name': 'Alex',
        'age': 8,
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
            'preferences_learned': []
        },
        'settings': {
            'animation_speed': 'normal',
            'sound_effects': True,
            'color_scheme': 'calm',
            'reduced_motion': False
        }
    }
}

# Enhanced ML Model
class EnhancedSensoryModel:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        
    def generate_training_data(self, n_samples=1000):
        """Generate realistic synthetic training data"""
        np.random.seed(42)
        
        # Base patterns for different overload types
        data = []
        
        for i in range(n_samples):
            # Normal state (60% of data)
            if i < 0.6 * n_samples:
                noise = np.random.normal(50, 15)
                light = np.random.normal(2000, 800)
                motion = np.random.normal(30, 12)
                overload = 0
                
            # Auditory overload (15% of data)
            elif i < 0.75 * n_samples:
                noise = np.random.normal(90, 20)
                light = np.random.normal(1800, 600)
                motion = np.random.normal(25, 10)
                overload = 1
                
            # Visual overload (15% of data)
            elif i < 0.9 * n_samples:
                noise = np.random.normal(55, 12)
                light = np.random.normal(6000, 1500)
                motion = np.random.normal(20, 8)
                overload = 1
                
            # Motion overload (10% of data)
            else:
                noise = np.random.normal(60, 10)
                light = np.random.normal(2200, 700)
                motion = np.random.normal(70, 15)
                overload = 1
            
            # Ensure realistic ranges
            noise = max(20, min(120, noise))
            light = max(100, min(10000, light))
            motion = max(0, min(100, motion))
            
            data.append({
                'noise': noise,
                'light': light,
                'motion': motion,
                'overload': overload,
                'timestamp': datetime.now().isoformat()
            })
        
        return pd.DataFrame(data)
    
    def extract_features(self, df):
        """Extract advanced features from sensor data"""
        features = []
        
        # Basic features
        features.extend(['noise', 'light', 'motion'])
        
        # Statistical features (simulating rolling windows)
        df['noise_rolling_mean'] = df['noise'].rolling(window=5, min_periods=1).mean()
        df['light_rolling_mean'] = df['light'].rolling(window=5, min_periods=1).mean()
        df['motion_rolling_mean'] = df['motion'].rolling(window=5, min_periods=1).mean()
        
        df['noise_rolling_std'] = df['noise'].rolling(window=5, min_periods=1).std()
        df['light_rolling_std'] = df['light'].rolling(window=5, min_periods=1).std()
        df['motion_rolling_std'] = df['motion'].rolling(window=5, min_periods=1).std()
        
        features.extend([
            'noise_rolling_mean', 'light_rolling_mean', 'motion_rolling_mean',
            'noise_rolling_std', 'light_rolling_std', 'motion_rolling_std'
        ])
        
        # Rate of change features
        df['noise_roc'] = df['noise'].diff().fillna(0)
        df['light_roc'] = df['light'].diff().fillna(0)
        df['motion_roc'] = df['motion'].diff().fillna(0)
        
        features.extend(['noise_roc', 'light_roc', 'motion_roc'])
        
        # Interaction features
        df['noise_light_interaction'] = df['noise'] * df['light'] / 1000
        df['noise_motion_interaction'] = df['noise'] * df['motion'] / 100
        df['light_motion_interaction'] = df['light'] * df['motion'] / 1000
        
        features.extend([
            'noise_light_interaction', 
            'noise_motion_interaction', 
            'light_motion_interaction'
        ])
        
        # Threshold crossing features
        df['noise_above_70'] = (df['noise'] > 70).astype(int)
        df['light_above_3000'] = (df['light'] > 3000).astype(int)
        df['motion_above_50'] = (df['motion'] > 50).astype(int)
        
        features.extend(['noise_above_70', 'light_above_3000', 'motion_above_50'])
        
        self.feature_names = features
        return df[features]
    
    def train(self, n_samples=1000):
        """Train the enhanced model"""
        print("🔄 Generating training data...")
        df = self.generate_training_data(n_samples)
        
        print("🔧 Extracting features...")
        X = self.extract_features(df)
        y = df['overload']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("🤖 Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, zero_division=0)
        self.recall = recall_score(y_test, y_pred, zero_division=0)
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Accuracy: {self.accuracy:.3f}")
        print(f"🎯 Precision: {self.precision:.3f}")
        print(f"📈 Recall: {self.recall:.3f}")
        
        return self.model
    
    def predict(self, sensor_data):
        """Make prediction on new sensor data"""
        if self.model is None:
            # Fallback to simple threshold-based prediction
            return self._fallback_prediction(sensor_data)
        
        try:
            # Create feature vector
            features = self._create_feature_vector(sensor_data)
            
            # Get prediction probability
            probability = self.model.predict_proba([features])[0, 1]
            
            return {
                'probability': float(probability),
                'prediction': int(probability > 0.5),
                'confidence': float(probability if probability > 0.5 else 1 - probability),
                'model_used': 'enhanced_random_forest',
                'features_used': len(features)
            }
            
        except Exception as e:
            print(f"⚠️ Enhanced model prediction failed: {e}")
            return self._fallback_prediction(sensor_data)
    
    def _create_feature_vector(self, sensor_data):
        """Create feature vector from sensor data"""
        noise = sensor_data['noise']
        light = sensor_data['light']
        motion = sensor_data['motion']
        
        # Basic features
        features = [noise, light, motion]
        
        # Statistical features (simplified)
        features.extend([noise, light, motion])  # Using current values as mean
        features.extend([5, 300, 5])  # Simplified std dev
        
        # Rate of change (simplified)
        features.extend([0, 0, 0])  # Assuming no previous data
        
        # Interaction features
        features.extend([
            noise * light / 1000,
            noise * motion / 100,
            light * motion / 1000
        ])
        
        # Threshold features
        features.extend([
            int(noise > 70),
            int(light > 3000),
            int(motion > 50)
        ])
        
        return features
    
    def get_safe_prediction(noise, light, motion):
        """Provide fallback predictions when ML model fails"""
        try:
            # Simple threshold-based fallback
            risk_score = 0.0
            if noise > 80 or light > 5000 or motion > 60:
                risk_score = 0.7
            elif noise > 60 or light > 3000 or motion > 40:
                risk_score = 0.4
            elif noise > 40 or light > 1500 or motion > 20:
                risk_score = 0.2
                
            return {
                'probability': risk_score,
                'prediction': 1 if risk_score > 0.5 else 0,
                'confidence': 0.8,
                'model_used': 'fallback_threshold',
                'features_used': 3
            }
        except Exception as e:
            return {
                'probability': 0.1,
                'prediction': 0,
                'confidence': 0.5,
                'model_used': 'emergency_fallback',
                'error': str(e)
            }

    def _fallback_prediction(self, sensor_data):
        """Fallback prediction using simple thresholds"""
        noise = sensor_data['noise']
        light = sensor_data['light']
        motion = sensor_data['motion']
        
        # Simple weighted threshold model
        risk_score = 0.0
        
        if noise > 80:
            risk_score += 0.4
        elif noise > 70:
            risk_score += 0.2
            
        if light > 5000:
            risk_score += 0.4
        elif light > 3000:
            risk_score += 0.2
            
        if motion > 60:
            risk_score += 0.2
        elif motion > 50:
            risk_score += 0.1
            
        probability = min(risk_score, 1.0)
        
        return {
            'probability': probability,
            'prediction': int(probability > 0.5),
            'confidence': 0.7,
            'model_used': 'fallback_threshold',
            'features_used': 3
        }
    
    def save_model(self, filepath='models/enhanced_sensory_model.joblib'):
        """Save trained model to file"""
        if self.model is None:
            print("❌ No model to save. Train the model first.")
            return False
        
        import os
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
        return True
    
    def load_model(self, filepath='models/enhanced_sensory_model.joblib'):
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.accuracy = model_data['accuracy']
            self.precision = model_data['precision']
            self.recall = model_data['recall']
            
            print(f"✅ Model loaded from {filepath}")
            print(f"📊 Previous performance - Accuracy: {self.accuracy:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

# Global enhanced model instance
enhanced_model = EnhancedSensoryModel()

def initialize_enhanced_model():
    """Initialize and train the enhanced model"""
    print("🚀 Initializing Enhanced Sensory Model...")
    
    # Try to load existing model first
    if enhanced_model.load_model():
        return enhanced_model
    
    # Train new model
    enhanced_model.train(n_samples=1000)
    enhanced_model.save_model()
    
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
                "message": "Enhanced model with advanced features"
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
                    "features": ["noise", "light", "motion"]
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
    """Generate simulated sensor data for real-time updates"""
    try:
        # Try to use sensor simulator if available
        from sensor_simulator import generate_sensor_data as sim_data
        return sim_data()
    except ImportError:
        # Fallback simulation
        return {
            'noise': np.random.normal(60, 20),
            'light': np.random.normal(2000, 1000),
            'motion': np.random.normal(30, 15),
            'temperature': 22.0 + random.random() * 2,
            'heart_rate': 70 + random.randint(-10, 10),
            'timestamp': datetime.now().isoformat()
        }

def get_overload_prediction(sensor_data):
    """Get overload prediction from sensor data using enhanced model"""
    try:
        if enhanced_model.model is not None:
            # Use enhanced model
            prediction = enhanced_model.predict(sensor_data)
            logger.info(f"Enhanced model prediction: {prediction}")
            return prediction['probability']
        elif model is not None:
            # Use simple model
            features = np.array([[sensor_data['noise'], sensor_data['light'], sensor_data['motion']]])
            
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features)[0, 1]
            else:
                prediction = model.predict(features)[0]
                probability = float(prediction)
            
            return float(probability)
        else:
            # Fallback to simple threshold-based prediction
            risk_score = 0.0
            if sensor_data['noise'] > 80:
                risk_score += 0.4
            if sensor_data['light'] > 5000:
                risk_score += 0.4
            if sensor_data['motion'] > 60:
                risk_score += 0.2
            
            return min(risk_score, 1.0)
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Fallback to simple calculation
        risk_score = 0.0
        if sensor_data['noise'] > 80:
            risk_score += 0.4
        if sensor_data['light'] > 5000:
            risk_score += 0.4
        if sensor_data['motion'] > 60:
            risk_score += 0.2
        
        return min(risk_score, 1.0)

def get_recommendations(sensor_data, prediction):
    """Get recommendations based on sensor data and prediction"""
    try:
        recommendation_engine = app.config.get('RECOMMENDATION_ENGINE')
        
        if recommendation_engine:
            # Determine overload type based on sensor data
            overload_type = "general"
            if sensor_data['noise'] > 80:
                overload_type = "auditory"
            elif sensor_data['light'] > 5000:
                overload_type = "visual"
            elif sensor_data['motion'] > 60:
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
                    'priority': 'high' if sensor_data['noise'] > 70 else 'medium',
                    'effectiveness': 85
                },
                {
                    'title': 'Adjust Lighting',
                    'description': 'Dim lights or move to a darker space',
                    'priority': 'high' if sensor_data['light'] > 4000 else 'medium',
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
        if sensor_data['noise'] > 80:
            overload_type = "auditory"
        elif sensor_data['light'] > 5000:
            overload_type = "visual" 
        elif sensor_data['motion'] > 60:
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

# Real-time data broadcasting
def broadcast_sensor_data():
    """Broadcast real-time sensor data via WebSocket"""
    while True:
        try:
            sensor_data = generate_sensor_data()
            prediction = get_overload_prediction(sensor_data)
            
            # Store for real-time updates
            recent_data['sensor_readings'].append({
                'timestamp': datetime.now().isoformat(),
                **sensor_data
            })
            recent_data['predictions'].append({
                'timestamp': datetime.now().isoformat(),
                'probability': prediction
            })
            
            # Keep only last 60 points (1 minute of data)
            for key in ['sensor_readings', 'predictions']:
                if len(recent_data[key]) > 60:
                    recent_data[key] = recent_data[key][-60:]
            
            # Emit real-time update
            socketio.emit('sensor_update', {
                'sensor_data': sensor_data,
                'prediction': prediction,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check for alerts
            if prediction > 0.7:  # High overload probability
                alert_data = {
                    'message': f'High sensory overload risk detected! ({prediction:.1%})',
                    'level': 'high',
                    'timestamp': datetime.now().isoformat()
                }
                recent_data['alerts'].append(alert_data)
                socketio.emit('alert', alert_data)
            
            time.sleep(1)  # 1Hz update rate
        except Exception as e:
            logger.error(f"Error in broadcast: {e}")
            time.sleep(1)

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
# ROUTE REGISTRATION - FIXED TO PREVENT ENDPOINT CONFLICTS
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
        return jsonify({
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "model_loaded": model is not None,
            "enhanced_model_loaded": enhanced_model.model is not None
        })
    
    @app.route('/api/activities/start', methods=['POST'])
    def start_activity_session():
        try:
            data = request.get_json()
            activity_id = data.get('activity_id')
            
            # Create a session ID (in a real app, you'd store this in a database)
            session_id = f"session_{int(datetime.now().timestamp())}"
            
            return jsonify({
                "session_id": session_id,
                "activity_id": activity_id,
                "status": "started",
                "started_at": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/activities/complete', methods=['POST'])
    def complete_activity_session():
        try:
            data = request.get_json()
            session_id = data.get('session_id')
            activity_id = data.get('activity_id')
            
            # In a real app, you'd update the session in a database
            return jsonify({
                "session_id": session_id,
                "activity_id": activity_id,
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "message": "Activity completed successfully"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/activities')
    def get_activities():
        return jsonify({
            "activities": [
                {
                    "id": 1,
                    "name": "Deep Breathing",
                    "description": "Follow the breathing circle to calm your mind",
                    "duration": 300,
                    "type": "breathing",
                    "emoji": "🌬️",
                    "color": "#4CAF50",
                    "difficulty": "beginner",
                    "age_range": "4+",
                    "benefits": ["Calming", "Focus", "Relaxation"],
                    "accessibility": ["visual", "audio"],
                    "instructions": [
                        {"text": "Get comfortable and relax your shoulders", "duration": 5, "phase": "prepare"},
                        {"text": "Breathe in slowly through your nose", "duration": 4, "phase": "inhale"},
                        {"text": "Hold your breath for a moment", "duration": 2, "phase": "hold"},
                        {"text": "Breathe out slowly through your mouth", "duration": 6, "phase": "exhale"}
                    ]
                },
                {
                    "id": 2,
                    "name": "Guided Meditation",
                    "description": "Listen to calming guidance for relaxation",
                    "duration": 600,
                    "type": "meditation",
                    "emoji": "🧘",
                    "color": "#2196F3",
                    "difficulty": "beginner",
                    "age_range": "6+",
                    "benefits": ["Relaxation", "Mindfulness", "Stress Relief"],
                    "accessibility": ["audio"],
                    "instructions": [
                        {"text": "Find a comfortable sitting position", "duration": 10, "phase": "prepare"},
                        {"text": "Close your eyes and focus on your breathing", "duration": 30, "phase": "inhale"},
                        {"text": "Notice any thoughts without judgment", "duration": 20, "phase": "hold"},
                        {"text": "Slowly return your awareness", "duration": 10, "phase": "exhale"}
                    ]
                },
                {
                    "id": 3,
                    "name": "Calming Sounds",
                    "description": "Nature sounds and white noise for relaxation",
                    "duration": 900,
                    "type": "audio",
                    "emoji": "🎵",
                    "color": "#9C27B0",
                    "difficulty": "beginner",
                    "age_range": "3+",
                    "benefits": ["Relaxation", "Sleep Aid", "Focus"],
                    "accessibility": ["audio"],
                    "instructions": [
                        {"text": "Get comfortable and close your eyes", "duration": 10, "phase": "prepare"},
                        {"text": "Focus on the calming sounds", "duration": 890, "phase": "inhale"}
                    ]
                }
            ]
        })

    @app.route('/api/activities/voice-options')
    def get_voice_options():
        return jsonify({
            "voices": [
                {
                    "id": 1,
                    "name": "Calm Female",
                    "language": "en-US",
                    "gender": "female",
                    "description": "Gentle and soothing",
                    "age_suitability": "All ages"
                },
                {
                    "id": 2,
                    "name": "Gentle Male", 
                    "language": "en-US",
                    "gender": "male",
                    "description": "Warm and reassuring",
                    "age_suitability": "All ages"
                },
                {
                    "id": 3,
                    "name": "Soothing Voice",
                    "language": "en-GB",
                    "gender": "female", 
                    "description": "Calm British accent",
                    "age_suitability": "6+"
                }
            ]
        })

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

# Start background threads
def start_background_services():
    """Start all background services"""
    # Start sensor simulation if available
    start_background_simulation()
    
    # Start real-time broadcasting
    broadcast_thread = threading.Thread(target=broadcast_sensor_data, daemon=True)
    broadcast_thread.start()
    logger.info("Real-time broadcasting started")

# Start background services when app starts
start_background_services()

# Add this at the VERY BOTTOM of your app.py file:

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("DEBUG", "False").lower() == "true"
    
    # Use 10000 as default port for Render
    socketio.run(app, 
                host="0.0.0.0", 
                port=port, 
                debug=debug, 
                allow_unsafe_werkzeug=True)