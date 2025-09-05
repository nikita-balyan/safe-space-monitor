# test_feature_engineer.py
from feature_engineer import FeatureEngineer
from datetime import datetime, timedelta
import numpy as np

def test_feature_engineer():
    """Test the feature engineering functionality"""
    print("Testing Feature Engineer...")
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(window_sizes=[5, 10])
    
    # Add some sample data
    base_time = datetime.now()
    for i in range(15):  # Add 15 readings
        timestamp = base_time + timedelta(seconds=i*2)
        noise = 50 + np.random.normal(0, 10)  # Random noise around 50dB
        light = 3000 + np.random.normal(0, 500)  # Random light around 3000 lux
        motion = 30 + np.random.normal(0, 5)  # Random motion around 30 units
        
        feature_engineer.add_reading(timestamp, noise, light, motion)
    
    # Check if we have enough data
    print(f"Has enough data: {feature_engineer.has_enough_data()}")
    print(f"Buffer sizes: {len(feature_engineer.noise_buffer)} readings")
    
    # Extract features
    features = feature_engineer.extract_features()
    print(f"Extracted {len(features)} features:")
    
    for feature_name, value in features.items():
        print(f"  {feature_name}: {value:.3f}")
    
    return features

if __name__ == "__main__":
    test_feature_engineer()