#!/usr/bin/env python3
"""
Comprehensive test of the model
"""

import joblib
import pandas as pd
import json
from pathlib import Path

def test_comprehensive():
    """Test the model with various sensor values"""
    try:
        # Load model
        model = joblib.load("../models/overload_model.joblib")
        
        # Test cases
        test_cases = [
            # (noise, light, motion, expected_state)
            (45, 1500, 25, "NORMAL"),      # Normal conditions
            (95, 2000, 30, "OVERLOAD"),    # Noise overload
            (55, 7500, 35, "OVERLOAD"),    # Light overload  
            (60, 1800, 75, "OVERLOAD"),    # Motion overload
            (90, 7000, 70, "OVERLOAD"),    # Combined overload
            (35, 800, 15, "NORMAL"),       # Very calm
        ]
        
        print("Testing model predictions:")
        print("=" * 50)
        
        for i, (noise, light, motion, expected) in enumerate(test_cases, 1):
            # Make prediction
            sample = [[noise, light, motion]]
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0]
            
            actual_state = "OVERLOAD" if prediction == 1 else "NORMAL"
            confidence = max(probability)
            
            status = "?" if actual_state == expected else "?"
            
            print(f"{status} Test {i}:")
            print(f"   Sensors: {noise}dB, {light}lux, {motion}au")
            print(f"   Prediction: {actual_state} (expected: {expected})")
            print(f"   Confidence: {confidence:.3f}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_comprehensive()
