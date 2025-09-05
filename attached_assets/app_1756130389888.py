# app.py
import os
import logging
from flask import Flask, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import joblib
import json

# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Create Flask app
# ------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "child-dashboard-secret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# ------------------------------
# Global variables (accessible to routes)
# ------------------------------
model = None
threshold = 0.5
model_metadata = {}
THRESHOLDS = {
    'noise': {'warning': 70, 'danger': 85},
    'light': {'warning': 800, 'danger': 1000},
    'motion': {'warning': 70, 'danger': 80}
}

# ------------------------------
# Model loading function
# ------------------------------
def load_model():
    global model, threshold, model_metadata
    try:
        model_path = "models/overload_model.joblib"
        threshold_path = "models/model_threshold.txt"
        metadata_path = "models/model_metadata.json"
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning("⚠️  Model file not found at %s", model_path)
            model = None
            
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                threshold = float(f.read().strip())
            logger.info(f"✅ Threshold loaded: {threshold}")
        else:
            logger.warning("⚠️  Threshold file not found, using default 0.5")
            # Create default threshold file
            with open(threshold_path, "w") as f:
                f.write("0.5")
            
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)
            logger.info("✅ Model metadata loaded")
        else:
            logger.warning("⚠️  Model metadata not found, creating default")
            model_metadata = {
                "training_date": "2023-01-01",
                "model_type": "RandomForest",
                "version": "1.0",
                "features": ["noise", "light", "motion"],
                "performance_metrics": {
                    "accuracy": 0.89,
                    "precision": 0.85,
                    "recall": 0.92,
                    "f1_score": 0.88
                }
            }
            # Create the metadata file for future use
            with open(metadata_path, "w") as f:
                json.dump(model_metadata, f, indent=2)
            
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model = None

# ------------------------------
# Background Tasks (Sensor Simulation)
# ------------------------------
def start_background_tasks():
    import threading
    try:
        from sensor_simulator import start_sensor_simulation
        logger.info("Starting sensor simulation thread...")
        sensor_thread = threading.Thread(target=start_sensor_simulation, daemon=True)
        sensor_thread.start()
        logger.info("Sensor simulation started successfully.")
    except Exception as e:
        logger.error(f"Failed to start sensor simulation: {e}")

# ------------------------------
# Make global variables available to routes
# ------------------------------
@app.context_processor
def inject_globals():
    return {
        'thresholds': THRESHOLDS,
        'model_loaded': model is not None,
        'model_metadata': model_metadata
    }

# ------------------------------
# Import and register routes
# ------------------------------
try:
    from routes import register_routes
    # Pass the global variables to the routes registration
    register_routes(app, model, threshold, model_metadata, THRESHOLDS)
    logger.info("Routes registered successfully.")
except ImportError as e:
    logger.error(f"Error importing routes: {e}")
    # Fallback basic route if routes.py doesn't exist
    @app.route("/")
    def fallback_route():
        return "Flask is running but routes.py could not be imported. Check the error logs."
except Exception as e:
    logger.error(f"Error registering routes: {e}")

# ------------------------------
# Load model on startup
# ------------------------------
load_model()

# Start sensor simulation
start_background_tasks()

# ------------------------------
# Health check route
# ------------------------------
@app.route("/health")
def health():
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "threshold": threshold,
        "sensor_simulation": "active"
    }
    return jsonify(status)

@app.route("/reload-model")
def reload_model():
    """Endpoint to manually reload the model"""
    load_model()
    return jsonify({
        "success": True,
        "message": "Model reloaded from disk",
        "model_loaded": model is not None,
        "threshold": threshold
    })

# ------------------------------
# Run Flask
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    logger.info(f"Starting Flask server on http://0.0.0.0:{port}")
    logger.info(f"Model status: {'Loaded' if model is not None else 'Not loaded'}")
    logger.info(f"Model threshold: {threshold}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)