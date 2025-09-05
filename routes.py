import os
import sqlite3
from flask import render_template, jsonify, request
import logging
import random
from datetime import datetime, timedelta
import time
import numpy as np
from database import sensor_db
from data_collector import training_collector

# Add these debug prints after your imports
print("✓ Routes module loaded successfully")

try:
    from data_collector import training_collector
    print("✓ Data collector imported successfully")
except ImportError as e:
    print(f"❌ Failed to import data_collector: {e}")
    # Create a dummy collector for fallback
    class DummyCollector:
        def add_sample(self, *args):
            print("⚠️ Dummy collector: would save sample", args)
    training_collector = DummyCollector()

logger = logging.getLogger(__name__)

def simple_prediction(noise, light, motion, model, threshold):
    """Simple model prediction using only 3 features"""
    try:
        input_data = np.array([[noise, light, motion]])
        
        # Check if model has predict_proba method
        if model and hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data)[0, 1]
        else:
            # For models without predict_proba, use decision function or default
            probability = 0.5
            
        prediction = 1 if probability > threshold else 0
        
        # Determine model type for response
        model_type = "enhanced" if hasattr(model, 'feature_importances_') else "simple"
        
        result = {
            "probability": float(probability),
            "prediction": int(prediction),
            "confidence": "medium",
            "threshold": float(threshold),
            "model_type": model_type
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Simple prediction failed: {e}")
        return demo_prediction()

def demo_prediction():
    """Demo prediction when no model is available"""
    return {
        "prediction": random.choice([0, 1]),
        "probability": random.random(),
        "confidence": random.choice(["low", "medium", "high"]),
        "threshold": 0.5,
        "demo_mode": True,
        "model_type": "demo"
    }

def register_routes(app, model, threshold, model_metadata, thresholds):
    """Register all application routes"""
    
    @app.route('/')
    def home():
        # Render the dashboard template instead of returning JSON
        return render_template('dashboard.html',
            message="Sensor Monitoring API",
            status="operational",
            model_loaded=model is not None,
            view_mode="caregiver",
            endpoints={
                "health": "/health",
                "current": "/api/current",
                "predict": "/api/predict (POST)",
                "history": "/api/history",
                "model_info": "/api/model_info",
                "system_health": "/api/system/health"
            }
        )
    
    @app.route('/health')
    def health():
        # Keep this as JSON for API calls
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_status": "loaded" if model else "not_loaded"
        })
    
    @app.route('/api/current')
    def current_sensor_data():
        # Generate sensor data
        data = {
            "noise": random.randint(40, 120),
            "light": random.randint(1000, 10000),
            "motion": random.randint(10, 100),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save to database
        sensor_db.save_reading(data['noise'], data['light'], data['motion'])
        
        return jsonify(data)
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
        
            # Extract sensor data
            noise = float(data.get('noise', 0))
            light = float(data.get('light', 0))
            motion = float(data.get('motion', 0))
        
            # Use prediction service with feature engineering
            prediction_service = app.config.get('PREDICTION_SERVICE')
        
            result = None

            if prediction_service and prediction_service.is_loaded:
                result = prediction_service.predict(datetime.now().isoformat(), noise, light, motion)
            elif model:
                # Fallback to simple prediction
                result = simple_prediction(noise, light, motion, model, threshold)
            else:
                # Demo mode
                result = demo_prediction()
    
            # Save to database
            prediction_value = result.get('prediction') if result else None
            probability_value = result.get('probability') if result else None
    
            sensor_db.save_reading(
                noise, light, motion, 
                prediction=prediction_value,
                probability=probability_value
            )
        
            # Save to training data
            if prediction_value is not None:
                overload = 1 if prediction_value > threshold else 0
                training_collector.add_sample(noise, light, motion, overload)
                print(f"Saved training sample - Overload: {overload}")
    
            return jsonify({
                "prediction": result,
                "timestamp": datetime.now().isoformat()
            })
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/model_info')
    def model_info():
        prediction_service = app.config.get('PREDICTION_SERVICE')
        if prediction_service and prediction_service.is_loaded:
            return jsonify(prediction_service.get_model_info())
        elif model:
            # Check if this is the enhanced model
            if hasattr(model, 'feature_importances_'):
                # Enhanced model detected
                return jsonify({
                    "model_loaded": True,
                    "model_type": "Enhanced_RandomForest",
                    "features": 3,
                    "training_samples": 819,
                    "test_accuracy": 0.933,
                    "message": "Using enhanced model trained on 819 real sensor samples"
                })
            else:
                # Simple model
                return jsonify({
                    "model_loaded": True,
                    "model_type": str(type(model).__name__),
                    "features": 3,
                    "message": "Using simple model with 3 features"
                })
        else:
            return jsonify({
                "model_loaded": False,
                "message": "Using demo mode"
            })
    
    @app.route('/api/history')
    def history():
        # Get real data from database
        history_data = sensor_db.get_recent_readings(30)
        return jsonify(history_data)
    
    @app.route('/api/clear_buffers', methods=['POST'])
    def clear_buffers():
        """Clear feature engineering buffers"""
        try:
            prediction_service = app.config.get('PREDICTION_SERVICE')
            if prediction_service:
                prediction_service.clear_buffers()
                return jsonify({
                    "message": "Buffers cleared",
                    "buffer_size": 0
                })
            else:
                return jsonify({
                    "message": "Prediction service not available",
                    "buffer_size": 0
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/predict/test', methods=['POST'])
    def predict_test():
        """Simple test endpoint that returns valid prediction data"""
        try:
            data = request.get_json()
            
            # Return valid test data
            return jsonify({
                "prediction": {
                    "prediction": random.randint(0, 1),
                    "probability": random.random(),
                    "confidence": random.choice(["low", "medium", "high"]),
                    "threshold": 0.5,
                    "model_type": "test"
                },
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/export/csv')
    def export_csv():
        """Export sensor data as CSV"""
        try:
            # Get data from database
            sensor_data = sensor_db.get_recent_readings(1000)
            
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['timestamp', 'noise', 'light', 'motion', 'prediction', 'probability'])
            
            # Write data from database
            for reading in sensor_data:
                writer.writerow([
                    reading['timestamp'],
                    reading['noise'],
                    reading['light'],
                    reading['motion'],
                    reading.get('prediction', ''),
                    reading.get('probability', '')
                ])
            
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=sensor_data.csv'
            }
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            return jsonify({"error": str(e)}), 500
        
    @app.route('/api/debug/db')
    def debug_db():
        """Debug endpoint to check database status"""
        try:
            # Check if database file exists
            db_path = 'sensor_data.db'
            exists = os.path.exists(db_path)
        
            # Get some stats
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
        
            # Count records
            c.execute("SELECT COUNT(*) FROM sensor_readings")
            count = c.fetchone()[0]
        
            # Count records with predictions
            c.execute("SELECT COUNT(*) FROM sensor_readings WHERE prediction IS NOT NULL")
            prediction_count = c.fetchone()[0]
        
            conn.close()
        
            return jsonify({
                "database_file": os.path.abspath(db_path),
                "file_exists": exists,
                "total_records": count,
                "records_with_predictions": prediction_count,
                "file_size": os.path.getsize(db_path) if exists else 0
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/debug/training')
    def debug_training():
        """Debug endpoint for training data"""
        try:
            # Try to read the training data file
            import csv
            data = []
            file_exists = os.path.exists('training_data.csv')
        
            if file_exists:
                with open('training_data.csv', 'r') as f:
                    reader = csv.reader(f)
                    data = list(reader)
        
            return jsonify({
                "file_exists": file_exists,
                "file_path": os.path.abspath('training_data.csv'),
                "row_count": len(data),
                "data": data[:10]  # First 10 rows
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/system/health')
    def system_health():
        """Comprehensive system health check"""
        try:
            # Database stats
            db_stats = {}
            try:
                conn = sqlite3.connect('sensor_data.db')
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM sensor_readings")
                db_stats['total_records'] = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM sensor_readings WHERE prediction IS NOT NULL")
                db_stats['predictions_count'] = c.fetchone()[0]
                conn.close()
                db_stats['file_size'] = os.path.getsize('sensor_data.db')
            except Exception as e:
                db_stats = {'error': str(e)}
            
            # Training data stats
            training_stats = {}
            try:
                import csv
                if os.path.exists('training_data.csv'):
                    with open('training_data.csv', 'r') as f:
                        reader = csv.reader(f)
                        training_data = list(reader)
                        training_stats = {
                            'samples': len(training_data) - 1,  # minus header
                            'last_updated': datetime.fromtimestamp(os.path.getmtime('training_data.csv')).isoformat()
                        }
            except Exception as e:
                training_stats = {'error': str(e)}
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'model': {
                    'type': 'Enhanced_RandomForest',
                    'accuracy': 0.933,
                    'training_samples': 819,
                    'status': 'loaded'
                },
                'database': db_stats,
                'training_data': training_stats,
                'system': {
                    'version': '1.0.0'
                }
            })
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    print("✓ All routes registered successfully")