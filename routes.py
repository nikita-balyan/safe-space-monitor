from flask import render_template, jsonify, request, current_app
import numpy as np
from datetime import datetime

def register_routes(app, model, threshold, model_metadata, THRESHOLDS):
    """Register all routes to the given Flask app with shared globals"""
    
    # Store references to the shared global variables
    app.config['GLOBAL_MODEL'] = model
    app.config['GLOBAL_THRESHOLD'] = threshold
    app.config['GLOBAL_MODEL_METADATA'] = model_metadata
    app.config['GLOBAL_THRESHOLDS'] = THRESHOLDS

    @app.route("/")
    def dashboard():
        view_mode = request.args.get("view", "child")
        return render_template("dashboard.html", 
                             view_mode=view_mode, 
                             thresholds=app.config['GLOBAL_THRESHOLDS'])

    @app.route("/api/current")
    def api_current_readings():
        from sensor_simulator import get_current_readings
        data = get_current_readings()
        if not data:
            return jsonify({"error": "No sensor data available"}), 404
        return jsonify(data)

    @app.route("/api/history")
    def api_historical_data():
        from sensor_simulator import get_historical_data
        data = get_historical_data()
        return jsonify(data if data else [])

    @app.route("/api/status")
    def api_status():
        from sensor_simulator import get_current_readings
        current = get_current_readings()
        return jsonify({
            "status": "active" if current else "inactive",
            "sensors_active": bool(current),
            "timestamp": current.get("timestamp") if current else None
        })

    @app.route("/api/thresholds")
    def api_thresholds():
        return jsonify(app.config['GLOBAL_THRESHOLDS'])

    @app.route("/test")
    def test_route():
        return "Flask is working! If you see this, the server is running."

    @app.route("/debug/routes")
    def debug_routes():
        """Show all registered routes"""
        routes = []
        for rule in current_app.url_map.iter_rules():
            routes.append({
                'endpoint': rule.endpoint,
                'methods': list(rule.methods),
                'path': rule.rule
            })
        return jsonify(routes)
    
    @app.route('/api/predict')
    def api_predict():
        from sensor_simulator import get_current_features, get_current_readings
        import pandas as pd
        
        if app.config['GLOBAL_MODEL'] is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        # Get current features for ML prediction
        features_data = get_current_features()
        if not features_data:
            return jsonify({"error": "Insufficient sensor data for feature extraction"}), 404
        
        # Get current readings for display
        current = get_current_readings()
        if not current:
            return jsonify({"error": "No current sensor data"}), 404
        
        try:
            # Prepare features in the same order as training
            metadata = app.config.get('GLOBAL_MODEL_METADATA', {})
            feature_names = metadata.get('feature_names', [])
            
            if feature_names:
                # Use the exact feature order from training
                feature_values = []
                for feature_name in feature_names:
                    value = features_data.get(feature_name, 0)  # Default to 0 if missing
                    feature_values.append(value)
                features_array = np.array([feature_values])
            else:
                # Fallback: try to reconstruct feature order
                feature_order = []
                
                # Add basic sensors
                feature_order.extend(['noise', 'light', 'motion'])
                
                # Add rolling features
                for sensor in ['noise', 'light', 'motion']:
                    for window in [10, 30, 60]:
                        for stat in ['mean', 'std', 'min', 'max', 'range']:
                            feature_order.append(f'{sensor}_{stat}_{window}')
                
                # Add FFT features
                for sensor in ['noise', 'light', 'motion']:
                    feature_order.append(f'{sensor}_fft_energy')
                
                feature_values = []
                for feature_name in feature_order:
                    value = features_data.get(feature_name, 0)
                    feature_values.append(value)
                features_array = np.array([feature_values])
            
            # Make prediction
            probability = app.config['GLOBAL_MODEL'].predict_proba(features_array)[0, 1]
            prediction = 1 if probability > app.config['GLOBAL_THRESHOLD'] else 0
            
            return jsonify({
                "probability": float(probability),
                "prediction": int(prediction),
                "threshold": float(app.config['GLOBAL_THRESHOLD']),
                "timestamp": current['timestamp'],
                "confidence": "high" if probability > 0.8 or probability < 0.2 else "medium",
                "features": {
                    "noise": current['noise'],
                    "light": current['light'],
                    "motion": current['motion']
                },
                "feature_count": len(feature_values) if 'feature_values' in locals() else 0
            })
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    @app.route('/api/model/info')
    def api_model_info():
        metadata = app.config.get('GLOBAL_MODEL_METADATA', {})
        return jsonify({
            "model_loaded": app.config['GLOBAL_MODEL'] is not None,
            "model_type": metadata.get('model_type', 'unknown'),
            "threshold": app.config['GLOBAL_THRESHOLD'],
            "metadata": metadata,
            "feature_count": metadata.get('num_features', 0),
            "training_date": metadata.get('training_date', 'unknown')
        })

    @app.route('/api/feedback', methods=['POST'])
    def api_feedback():
        """Accept user feedback on predictions for model improvement"""
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            # Required fields
            required_fields = ['timestamp', 'actual_overload', 'prediction_probability']
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            # Optional fields
            feedback_entry = {
                'timestamp': data['timestamp'],
                'actual_overload': int(data['actual_overload']),
                'prediction_probability': float(data['prediction_probability']),
                'predicted_overload': int(data.get('predicted_overload', 0)),
                'user_notes': data.get('user_notes', ''),
                'feedback_timestamp': datetime.now().isoformat(),
                'view_mode': data.get('view_mode', 'unknown')  # child vs caregiver
            }
            
            # Save feedback to CSV
            from sensor_simulator import FEEDBACK_FILE
            import csv
            
            # Create feedback file if it doesn't exist
            if not FEEDBACK_FILE.exists():
                with open(FEEDBACK_FILE, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'actual_overload', 'prediction_probability', 
                        'predicted_overload', 'user_notes', 'feedback_timestamp', 'view_mode'
                    ])
            
            # Append feedback
            with open(FEEDBACK_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    feedback_entry['timestamp'],
                    feedback_entry['actual_overload'],
                    feedback_entry['prediction_probability'],
                    feedback_entry['predicted_overload'],
                    feedback_entry['user_notes'],
                    feedback_entry['feedback_timestamp'],
                    feedback_entry['view_mode']
                ])
            
            return jsonify({
                "status": "success",
                "message": "Feedback recorded successfully",
                "feedback_id": feedback_entry['feedback_timestamp']
            })
            
        except Exception as e:
            return jsonify({"error": f"Failed to record feedback: {str(e)}"}), 500

    @app.route('/api/export/csv')
    def api_export_csv():
        from sensor_simulator import get_historical_data
        import csv
        import io
        from flask import Response
        
        data = get_historical_data(100)  # Get last 100 readings
        
        if not data:
            return jsonify({"error": "No data available for export"}), 404
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['timestamp', 'noise', 'light', 'motion', 'label'])
        
        # Write data
        for reading in data:
            writer.writerow([
                reading.get('timestamp', ''),
                reading.get('noise', 0),
                reading.get('light', 0),
                reading.get('motion', 0),
                reading.get('label', 0)
            ])
        
        response = Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=sensor_data.csv'}
        )
        
        return response
