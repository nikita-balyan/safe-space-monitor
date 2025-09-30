from flask import Flask, jsonify
import os
import random

app = Flask(__name__)

@app.route('/')
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Safe Space Monitor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 20px; background: #f0f8ff; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Safe Space Monitor</h1>
            <p>AI-Powered Sensory Regulation for Neurodiverse Children</p>
            
            <div class="status">
                <h3>✅ Application Status: Running</h3>
                <p><strong>Deployment:</strong> Successfully deployed on Render</p>
                <p><strong>Features:</strong> Real-time sensory monitoring & predictions</p>
                <p><strong>API Endpoints:</strong> 
                    <a href="/health">/health</a> | 
                    <a href="/api/current">/api/current</a> |
                    <a href="/api/recommendations">/api/recommendations</a>
                </p>
            </div>
            
            <h3>📊 Test Links:</h3>
            <ul>
                <li><a href="/dashboard">Dashboard</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/api/current">Current Sensor Data</a></li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.route('/dashboard')
def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard - Safe Space Monitor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .sensor { margin: 10px 0; padding: 10px; border-left: 4px solid #4CAF50; }
        </style>
    </head>
    <body>
        <h1>📈 Safe Space Monitor Dashboard</h1>
        <div class="sensor">
            <h3>Real-time Sensor Monitoring</h3>
            <p>Noise: <span id="noise">--</span> dB</p>
            <p>Light: <span id="light">--</span> lux</p>
            <p>Motion: <span id="motion">--</span> units</p>
            <p>Overload Risk: <span id="risk">--</span></p>
        </div>
        <p><a href="/">← Back to Home</a></p>
        
        <script>
            // Simple auto-update
            setInterval(() => {
                fetch('/api/current')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('noise').textContent = data.sensor_data.noise;
                        document.getElementById('light').textContent = data.sensor_data.light;
                        document.getElementById('motion').textContent = data.sensor_data.motion;
                        document.getElementById('risk').textContent = 
                            (data.prediction.probability * 100).toFixed(1) + '%';
                    });
            }, 2000);
        </script>
    </body>
    </html>
    """

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "Safe Space Monitor",
        "version": "1.0",
        "deployment": "Render",
        "timestamp": "2025-09-30T18:00:00Z"
    })

@app.route('/api/current')
def current_data():
    # Simple sensor simulation
    noise = random.randint(40, 120)
    light = random.randint(1000, 10000)
    motion = random.randint(10, 100)
    
    # Simple prediction logic
    risk_score = 0
    if noise > 80: risk_score += 0.4
    if light > 7000: risk_score += 0.3
    if motion > 80: risk_score += 0.3
    
    return jsonify({
        "sensor_data": {
            "noise": noise,
            "light": light,
            "motion": motion
        },
        "prediction": {
            "probability": risk_score,
            "prediction": 1 if risk_score > 0.7 else 0,
            "model_used": "threshold_based",
            "confidence": 0.85
        }
    })

@app.route('/api/recommendations')
def recommendations():
    return jsonify({
        "strategies": [
            {
                "name": "Use noise-cancelling headphones",
                "type": "auditory",
                "effectiveness": 85,
                "description": "Reduce overwhelming sounds"
            },
            {
                "name": "Move to a quieter space", 
                "type": "environmental",
                "effectiveness": 90,
                "description": "Change to calmer environment"
            },
            {
                "name": "Practice deep breathing",
                "type": "regulatory", 
                "effectiveness": 75,
                "description": "Calm nervous system"
            }
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)