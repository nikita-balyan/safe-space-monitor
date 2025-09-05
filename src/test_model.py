#!/usr/bin/env python3
"""
Test script to verify everything works
"""

import joblib
import pandas as pd
import json
from pathlib import Path

def test_model():
    """Test the trained model"""
    try:
        # Load model
        model_path = Path("../models/overload_model.joblib")
        model = joblib.load(model_path)
        print("? Model loaded successfully")
        
        # Load metadata
        metadata_path = Path("../models/model_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print("? Metadata loaded successfully")
        
        # Load data
        data_path = Path("../data/features_data.csv")
        df = pd.read_csv(data_path)
        print("? Data loaded successfully")
        
        print(f"\nModel performance:")
        print(f"Accuracy: {metadata['accuracy']:.3f}")
        print(f"F1 Score: {metadata['f1_score']:.3f}")
        
        # Test prediction
        sample_data = [[50, 2000, 30]]  # Normal case
        prediction = model.predict(sample_data)
        probability = model.predict_proba(sample_data)
        
        print(f"\nSample prediction:")
        print(f"Input: Noise=50dB, Light=2000lux, Motion=30")
        print(f"Prediction: {'OVERLOAD' if prediction[0] == 1 else 'NORMAL'}")
        print(f"Confidence: {max(probability[0]):.3f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_model()
