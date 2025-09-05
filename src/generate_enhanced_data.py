#!/usr/bin/env python3
"""
Enhanced training data generator with rolling features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from collections import deque

# Configuration
data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)
output_file = f"{data_dir}/enhanced_features_data.csv"

def generate_enhanced_data(num_samples=2000):
    """Generate sensor data with rolling features"""
    
    # Base patterns (same as before)
    patterns = {
        "normal": {"noise_mean": 50, "noise_std": 15, "light_mean": 2000, "light_std": 1000, "motion_mean": 30, "motion_std": 15, "overload_prob": 0.1},
        "noise_overload": {"noise_mean": 90, "noise_std": 20, "light_mean": 2000, "light_std": 1000, "motion_mean": 30, "motion_std": 15, "overload_prob": 0.8},
        "light_overload": {"noise_mean": 50, "noise_std": 15, "light_mean": 7000, "light_std": 2000, "motion_mean": 30, "motion_std": 15, "overload_prob": 0.7},
        "motion_overload": {"noise_mean": 50, "noise_std": 15, "light_mean": 2000, "light_std": 1000, "motion_mean": 70, "motion_std": 20, "overload_prob": 0.6},
        "combined_overload": {"noise_mean": 85, "noise_std": 20, "light_mean": 6500, "light_std": 2000, "motion_mean": 65, "motion_std": 20, "overload_prob": 0.95}
    }
    
    # Buffers for rolling calculations
    noise_buffer = deque(maxlen=60)
    light_buffer = deque(maxlen=60)
    motion_buffer = deque(maxlen=60)
    
    data = []
    current_time = datetime.now()
    
    for i in range(num_samples):
        # Choose pattern and generate base values
        pattern_name = random.choice(list(patterns.keys()))
        pattern = patterns[pattern_name]
        
        # Generate with smoothing
        noise = max(30, min(120, np.random.normal(pattern["noise_mean"], pattern["noise_std"])))
        light = max(100, min(10000, np.random.normal(pattern["light_mean"], pattern["light_std"])))
        motion = max(0, min(100, np.random.normal(pattern["motion_mean"], pattern["motion_std"])))
        
        # Add to buffers
        noise_buffer.append(noise)
        light_buffer.append(light)
        motion_buffer.append(motion)
        
        # Determine label
        label = 1 if random.random() < pattern["overload_prob"] else 0
        
        # Create base record
        record = {
            'timestamp': (current_time + timedelta(seconds=i)).isoformat(),
            'noise': noise,
            'light': light,
            'motion': motion,
            'label': label
        }
        
        # Add rolling features if we have enough data
        if len(noise_buffer) >= 10:
            for sensor, buffer in [('noise', noise_buffer), ('light', light_buffer), ('motion', motion_buffer)]:
                values = list(buffer)
                
                # 10-second window features
                if len(values) >= 10:
                    recent_10 = values[-10:]
                    record[f'{sensor}_mean_10'] = np.mean(recent_10)
                    record[f'{sensor}_std_10'] = np.std(recent_10)
                    record[f'{sensor}_range_10'] = max(recent_10) - min(recent_10)
                    record[f'{sensor}_slope_10'] = (recent_10[-1] - recent_10[0]) / 10
                
                # 30-second window features
                if len(values) >= 30:
                    recent_30 = values[-30:]
                    record[f'{sensor}_mean_30'] = np.mean(recent_30)
                    record[f'{sensor}_std_30'] = np.std(recent_30)
                    record[f'{sensor}_range_30'] = max(recent_30) - min(recent_30)
                    record[f'{sensor}_slope_30'] = (recent_30[-1] - recent_30[0]) / 30
                
                # 60-second window features  
                if len(values) >= 60:
                    recent_60 = values[-60:]
                    record[f'{sensor}_mean_60'] = np.mean(recent_60)
                    record[f'{sensor}_std_60'] = np.std(recent_60)
                    record[f'{sensor}_range_60'] = max(recent_60) - min(recent_60)
                    record[f'{sensor}_slope_60'] = (recent_60[-1] - recent_60[0]) / 60
                    
                    # Simple FFT energy (approximation)
                    fft_values = np.fft.fft(recent_60)
                    record[f'{sensor}_fft_energy'] = np.sum(np.abs(fft_values[1:])**2)  # Exclude DC component
        
        data.append(record)
    
    return pd.DataFrame(data)

def main():
    """Main function"""
    print("Generating enhanced training data with rolling features...")
    
    # Generate data
    df = generate_enhanced_data(2000)
    
    # Fill NaN values with 0 for missing rolling features
    df = df.fillna(0)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} samples to {output_file}")
    
    # Show summary
    print(f"Normal samples: {len(df[df['label'] == 0])}")
    print(f"Overload samples: {len(df[df['label'] == 1])}")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {len(df.columns) - 4}")  # minus timestamp, 3 sensors, label

if __name__ == "__main__":
    main()
