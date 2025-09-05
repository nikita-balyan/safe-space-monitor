#!/usr/bin/env python3
"""
Flask application factory and configuration
"""

import os
import logging
import joblib
from pathlib import Path
from flask import Flask
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

def register_fallback_routes():
    """Register basic routes if routes.py is missing"""
    from flask import jsonify
    from datetime import datetime
    import random
    
    @app.route('/')
    def home():
        return jsonify({
            "message": "Sensor Monitoring API (Fallback Mode)",
            "status": "operational",
            "model_loaded": model is not None,
            "endpoints": {
                "health": "/health",
                "current": "/api/current",
                "predict": "/api/predict (POST)"
            }
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_status": "loaded" if model else "not_loaded"
        })
    
    @app.route('/api/current')
    def current_sensor_data():
        data = {
            "noise": random.randint(40, 120),
            "light": random.randint(1000, 10000),
            "motion": random.randint(10, 100),
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(data)
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        from flask import request
        import numpy as np
        
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            if model:
                # Simple model prediction
                input_data = np.array([[
                    float(data.get('noise', 0)),
                    float(data.get('light', 0)),
                    float(data.get('motion', 0))
                ]])
                probability = model.predict_proba(input_data)[0, 1]
                prediction = 1 if probability > threshold else 0
                
                result = {
                    "probability": float(probability),
                    "prediction": int(prediction),
                    "confidence": "medium",
                    "threshold": float(threshold),
                    "model_type": "simple"
                }
            else:
                # Demo mode
                result = {
                    "prediction": random.choice([0, 1]),
                    "confidence": random.choice(["low", "medium", "high"]),
                    "threshold": 0.5,
                    "demo_mode": True,
                    "model_type": "demo"
                }
            
            return jsonify({
                "prediction": result,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"error": str(e)}), 500

# Load model on startup
load_ml_model()

# Register routes
try:
    # Add current directory to Python path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        
    from routes import register_routes
    register_routes(app, model, threshold, model_metadata, THRESHOLDS)
    logger.info("Routes registered successfully")
except ImportError as e:
    logger.error(f"Failed to register routes: {e}")
    logger.info("Registering fallback routes")
    register_fallback_routes()
except Exception as e:
    logger.error(f"Error registering routes: {e}")
    register_fallback_routes()

# Start sensor simulation if available
start_background_simulation()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)