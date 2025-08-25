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
