from flask import Flask, jsonify, render_template
import os
import random

app = Flask(__name__)

# Simple fallback prediction without model dependencies
def simple_prediction(noise, light, motion):
    """Simple threshold-based prediction for deployment"""
    score = 0
    if noise > 70: score += 0.4
    if light > 5000: score += 0.3  
    if motion > 70: score += 0.3
    return min(score, 1.0)

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "version": "1.0"})

@app.route('/api/current')
def current_data():
    # Simulate sensor data
    noise = random.randint(40, 120)
    light = random.randint(1000, 10000)
    motion = random.randint(10, 100)
    
    probability = simple_prediction(noise, light, motion)
    
    return jsonify({
        "sensor_data": {
            "noise": noise,
            "light": light,
            "motion": motion
        },
        "prediction": {
            "probability": probability,
            "prediction": 1 if probability > 0.7 else 0,
            "model_used": "simple_threshold",
            "confidence": 0.8
        }
    })

@app.route('/api/recommendations')
def recommendations():
    return jsonify({
        "strategies": [
            {"name": "Noise-cancelling headphones", "type": "auditory", "effectiveness": 85},
            {"name": "Move to quiet space", "type": "environmental", "effectiveness": 90},
            {"name": "Deep breathing exercise", "type": "regulatory", "effectiveness": 75}
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)