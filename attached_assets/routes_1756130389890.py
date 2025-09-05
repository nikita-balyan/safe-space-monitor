# routes.py

from flask import render_template, jsonify, request, current_app
import numpy as np

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
            "timestamp": current["timestamp"] if current else None
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
        from sensor_simulator import get_current_readings
        
        if app.config['GLOBAL_MODEL'] is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        current = get_current_readings()
        if not current:
            return jsonify({"error": "No sensor data"}), 404
        
        features = np.array([[current['noise'], current['light'], current['motion']]])
        
        try:
            probability = app.config['GLOBAL_MODEL'].predict_proba(features)[0, 1]
            prediction = 1 if probability > app.config['GLOBAL_THRESHOLD'] else 0
            
            return jsonify({
                "probability": float(probability),
                "prediction": int(prediction),
                "threshold": float(app.config['GLOBAL_THRESHOLD']),
                "timestamp": current['timestamp'],
                "features": {
                    "noise": current['noise'],
                    "light": current['light'],
                    "motion": current['motion']
                }
            })
        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    @app.route('/api/model/info')
    def api_model_info():
        return jsonify({
            "model_loaded": app.config['GLOBAL_MODEL'] is not None,
            "model_type": str(type(app.config['GLOBAL_MODEL']).__name__) if app.config['GLOBAL_MODEL'] else "None",
            "threshold": app.config['GLOBAL_THRESHOLD'],
            "metadata": app.config['GLOBAL_MODEL_METADATA'],
            "features": ["noise", "light", "motion"]
        })

    @app.route('/api/model/performance')
    def api_model_performance():
        # Sample performance data
        performance_data = {
            "accuracy": 0.89,
            "precision": 0.85,
            "recall": 0.92,
            "f1_score": 0.88,
            "roc_auc": 0.93,
            "confusion_matrix": [[45, 5], [3, 47]],
            "training_history": {
                "epochs": list(range(1, 21)),
                "train_loss": [0.65, 0.52, 0.45, 0.38, 0.32, 0.28, 0.24, 0.21, 0.18, 0.16, 
                              0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06, 0.05],
                "val_loss": [0.58, 0.48, 0.42, 0.37, 0.33, 0.30, 0.27, 0.25, 0.23, 0.21,
                           0.20, 0.19, 0.18, 0.17, 0.17, 0.16, 0.16, 0.16, 0.15, 0.15],
                "train_accuracy": [0.70, 0.75, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.92,
                                 0.93, 0.94, 0.95, 0.95, 0.96, 0.96, 0.97, 0.97, 0.97, 0.98],
                "val_accuracy": [0.72, 0.76, 0.79, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88, 0.88,
                               0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89, 0.89]
            },
            "feature_importance": {
                "noise": 0.45,
                "light": 0.30,
                "motion": 0.25
            },
            "last_trained": app.config['GLOBAL_MODEL_METADATA'].get("training_date", "2023-01-01")
        }
        
        return jsonify(performance_data)

    @app.route('/api/model/confusion_matrices')
    def api_confusion_matrices():
        # Sample data showing how confusion matrix changes over epochs
        matrices = {}
        base_matrix = [[30, 20], [15, 35]]  # Initial poor performance
        
        for epoch in range(0, 20, 2):  # Every 2 epochs
            improvement = epoch * 2
            matrices[epoch] = [
                [base_matrix[0][0] + improvement, base_matrix[0][1] - improvement//2],
                [base_matrix[1][0] - improvement//2, base_matrix[1][1] + improvement]
            ]
        
        return jsonify(matrices)

    @app.route('/api/model/reload')
    def api_model_reload():
        # This would need access to the load_model function from app.py
        # For now, we'll return a message indicating this endpoint needs implementation
        return jsonify({
            "success": False,
            "message": "Model reload endpoint needs implementation in app context",
            "model_loaded": app.config['GLOBAL_MODEL'] is not None
        })