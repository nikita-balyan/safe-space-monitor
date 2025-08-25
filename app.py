import os
import logging
import joblib
from pathlib import Path
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
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

def load_model():
    """Load the trained model if it exists"""
    global model, threshold, model_metadata
    
    model_path = Path("models/overload_model.joblib")
    threshold_path = Path("models/model_threshold.txt")
    
    try:
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            
            if threshold_path.exists():
                with open(threshold_path, 'r') as f:
                    threshold = float(f.read().strip())
                logger.info(f"Threshold loaded: {threshold}")
            
            model_metadata = {
                "model_type": str(type(model).__name__),
                "training_date": "2025-01-15",
                "features": ["noise", "light", "motion"]
            }
        else:
            logger.warning("No trained model found. Predictions will not be available.")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None

def start_background_simulation():
    """Start the sensor simulation in a background thread"""
    try:
        from sensor_simulator import start_sensor_simulation
        start_sensor_simulation(interval=1)  # 1Hz as requested
        logger.info("Background sensor simulation started")
    except Exception as e:
        logger.error(f"Failed to start sensor simulation: {e}")

# Load model on startup
load_model()

# Register routes
from routes import register_routes
register_routes(app, model, threshold, model_metadata, THRESHOLDS)

# Start sensor simulation
start_background_simulation()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
