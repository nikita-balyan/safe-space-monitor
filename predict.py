@'
#!/usr/bin/env python3
"""
Simple prediction test script
"""

import joblib
import numpy as np
from pathlib import Path

def test_prediction():
    """Test the trained model with sample data"""
    try:
        # Load model
        model_path = Path("../models/overload_model.joblib")
        model = joblib.load(model_path)
        
        # Sample test data
        test_data = np.array([[50, 2000, 30] + [0] * 48])  # Normal case
        
        # Make prediction
        prediction = model.predict(test_data)
        probability = model.predict_proba(test_data)
        
        print(f"Prediction: {prediction[0]}")
        print(f"Probability: {probability[0]}")
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please run model_training.py first to train the model")

if __name__ == "__main__":
    test_prediction()
'@ | Out-File -Encoding utf8 predict.py