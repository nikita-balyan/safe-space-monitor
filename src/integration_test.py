#!/usr/bin/env python3
"""
Integration test with sensor simulator
"""

import joblib
import numpy as np
from datetime import datetime

class OverloadPredictor:
    """Simple predictor for integration testing"""
    
    def __init__(self):
        self.model = joblib.load("../models/overload_model.joblib")
        print("Model loaded for integration testing")
    
    def predict(self, noise, light, motion):
        """Predict overload based on sensor values"""
        features = np.array([[noise, light, motion]])
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            "prediction": int(prediction),
            "probability": float(max(probability)),
            "confidence": "high" if max(probability) > 0.8 else "medium",
            "timestamp": datetime.now().isoformat()
        }

# Test the integration
if __name__ == "__main__":
    predictor = OverloadPredictor()
    
    # Simulate sensor readings
    test_readings = [
        {"noise": 50, "light": 2000, "motion": 30},
        {"noise": 95, "light": 2000, "motion": 30},
        {"noise": 50, "light": 7500, "motion": 35}
    ]
    
    print("Integration Test Results:")
    print("=" * 40)
    
    for i, reading in enumerate(test_readings, 1):
        result = predictor.predict(reading["noise"], reading["light"], reading["motion"])
        state = "OVERLOAD" if result["prediction"] == 1 else "NORMAL"
        
        print(f"Test {i}:")
        print(f"  Input: {reading}")
        print(f"  Prediction: {state} (confidence: {result['confidence']})")
        print(f"  Probability: {result['probability']:.3f}")
        print()
