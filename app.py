#!/usr/bin/env python3
"""
Flask application factory and configuration
"""

import os
import logging
import joblib
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import sys
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, template_folder='templates')
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

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

def setup_logging():
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

def load_ml_model():
    """Load the enhanced ML model on application startup"""
    global model, threshold, model_metadata
    
    # FIRST try to load the enhanced model (NEW)
    try:
        enhanced_model_path = Path("models/enhanced_overload_model.joblib")
        if enhanced_model_path.exists():
            model = joblib.load(enhanced_model_path)
            threshold = 0.5
            
            # Create enhanced model metadata
            model_metadata = {
                "model_loaded": True,
                "model_type": "Enhanced_RandomForest",
                "training_date": "2025-09-05",
                "training_samples": 819,
                "test_accuracy": 0.933,
                "features": ["noise", "light", "motion"],
                "message": "Enhanced model trained on 819 real sensor samples"
            }
            
            logger.info("✓ Enhanced ML model loaded successfully")
            logger.info(f"✓ Model accuracy: 93.3% (trained on 819 samples)")
            return
    except Exception as e:
        logger.error(f"Error loading enhanced model: {e}")
    
    # Fall back to simple model (your existing code)
    try:
        # Try to import and use enhanced services
        try:
            # Add current directory to Python path for imports
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            from prediction_service import PredictionService
            from feature_engineer import FeatureEngineer
            
            # Create instances of the services
            prediction_service = PredictionService()
            feature_engineer = FeatureEngineer()
            
            # Load the enhanced model
            prediction_service.load_model()
            
            if prediction_service.is_loaded:
                # Use enhanced model
                model = prediction_service.model
                threshold = prediction_service.threshold
                model_metadata = prediction_service.metadata
                
                logger.info("✓ Enhanced ML model loaded successfully")
                logger.info(f"✓ Using {len(prediction_service.feature_names)} features")
                if 'f1_score' in prediction_service.metadata:
                    logger.info(f"✓ Model performance: F1={prediction_service.metadata['f1_score']:.3f}")
                
                # Store references for routes
                app.config['PREDICTION_SERVICE'] = prediction_service
                app.config['FEATURE_ENGINEER'] = feature_engineer
                return
            else:
                logger.warning("Enhanced model failed to load")
                
        except ImportError as e:
            logger.warning(f"Enhanced services not available: {e}")
        except Exception as e:
            logger.error(f"Error initializing enhanced services: {e}")
        
        # Fall back to simple model
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
        logger.info("✓ Recommendation engine loaded successfully")
        return recommendation_engine
    except ImportError as e:
        logger.error(f"Failed to load recommendation engine: {e}")
        return None
    except Exception as e:
        logger.error(f"Error initializing recommendation engine: {e}")
        return None

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

# Load models and engines on startup
load_ml_model()
recommendation_engine = load_recommendation_engine()

# Recommendation Engine Routes
@app.route('/api/recommendations', methods=['GET', 'POST'])
def get_recommendations():
    """
    Get recommendations based on sensor data or overload type
    """
    from flask import request
    
    try:
        if request.method == 'POST':
            data = request.get_json()
            overload_type = data.get('overload_type', '')
            user_id = data.get('user_id', 'default')
        else:
            # GET request - get overload type from query params
            overload_type = request.args.get('type', '')
            user_id = request.args.get('user_id', 'default')
        
        if not overload_type:
            return jsonify({"error": "No overload type provided"}), 400
        
        # Get recommendations
        if recommendation_engine:
            recommendations = recommendation_engine.get_recommendations(
                overload_type, user_id, count=3
            )
        else:
            # Fallback demo recommendations
            recommendations = [
                {
                    "id": "demo_strategy",
                    "name": "Take a calming break",
                    "description": "Find a quiet space and take deep breaths",
                    "emoji": "🧘",
                    "feedback_score": 0.75
                }
            ]
        
        return jsonify({
            "overload_type": overload_type,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def record_feedback():
    """
    Record feedback about a strategy
    """
    from flask import request
    
    try:
        data = request.get_json()
        strategy_id = data.get('strategy_id')
        was_helpful = data.get('helpful', False)
        user_id = data.get('user_id', 'default')
        
        if not strategy_id:
            return jsonify({"error": "No strategy ID provided"}), 400
        
        if recommendation_engine:
            success_rate = recommendation_engine.record_feedback(strategy_id, was_helpful, user_id)
        else:
            success_rate = 0.5  # Demo value
        
        return jsonify({
            "status": "success",
            "message": "Feedback recorded",
            "strategy_id": strategy_id,
            "success_rate": success_rate,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/profile', methods=['GET', 'POST', 'PUT'])
def manage_profile():
    """
    Create or update user profile
    """
    from flask import request
    
    try:
        if request.method == 'GET':
            user_id = request.args.get('user_id', 'default')
            if recommendation_engine:
                profile = recommendation_engine.user_profiles.get(user_id, {})
            else:
                profile = {}
            return jsonify({"profile": profile})
        
        else:  # POST or PUT
            data = request.get_json()
            user_id = data.get('user_id', 'default')
            age = data.get('age')
            preferences = data.get('preferences', {})
            
            if not age:
                return jsonify({"error": "Age is required"}), 400
            
            if recommendation_engine:
                profile = recommendation_engine.create_user_profile(user_id, age, preferences)
            else:
                profile = {"age": age, "preferences": preferences}
            
            return jsonify({
                "status": "success",
                "message": "Profile updated",
                "user_id": user_id,
                "profile": profile,
                "timestamp": datetime.now().isoformat()
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/profile')
def profile_page():
    """Serve the profile setup page"""
    from flask import render_template
    return render_template('profile.html')

# Register main routes
try:
    # Add current directory to Python path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    from routes import register_routes
    register_routes(app, model, threshold, model_metadata, THRESHOLDS)
    logger.info("Main routes registered successfully")
except ImportError as e:
    logger.error(f"Failed to register main routes: {e}")
    # Fallback basic routes
    @app.route('/')
    def home():
        return jsonify({
            "message": "Sensor Monitoring API",
            "status": "operational",
            "model_loaded": model is not None,
            "recommendation_engine_loaded": recommendation_engine is not None,
            "endpoints": {
                "health": "/health",
                "current": "/api/current",
                "predict": "/api/predict (POST)",
                "recommendations": "/api/recommendations",
                "profile": "/api/profile"
            }
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_status": "loaded" if model else "not_loaded",
            "recommendation_engine_status": "loaded" if recommendation_engine else "not_loaded"
        })
except Exception as e:
    logger.error(f"Error registering main routes: {e}")

# Start sensor simulation if available
start_background_simulation()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)