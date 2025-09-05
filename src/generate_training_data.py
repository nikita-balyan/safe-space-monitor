#!/usr/bin/env python3
"""
Simple training data generator
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Create data directory if it doesn't exist
data_dir = "../data"
os.makedirs(data_dir, exist_ok=True)
output_file = f"{data_dir}/features_data.csv"

def generate_simple_data(num_samples=500):
    """Generate simple sensor data"""
    
    data = []
    current_time = datetime.now()
    
    for i in range(num_samples):
        # Randomly choose normal or overload state
        is_overload = random.random() < 0.3  # 30% chance of overload
        
        if is_overload:
            # Overload patterns
            pattern = random.choice(["noise", "light", "motion", "combined"])
            
            if pattern == "noise":
                noise = random.randint(80, 120)
                light = random.randint(1000, 3000)
                motion = random.randint(10, 40)
            elif pattern == "light":
                noise = random.randint(40, 70)
                light = random.randint(6000, 10000)
                motion = random.randint(10, 40)
            elif pattern == "motion":
                noise = random.randint(40, 70)
                light = random.randint(1000, 3000)
                motion = random.randint(60, 100)
            else:  # combined
                noise = random.randint(70, 100)
                light = random.randint(5000, 8000)
                motion = random.randint(50, 80)
        else:
            # Normal state
            noise = random.randint(40, 70)
            light = random.randint(1000, 3000)
            motion = random.randint(10, 40)
        
        # Create record
        record = {
            'timestamp': (current_time + timedelta(seconds=i)).isoformat(),
            'noise': noise,
            'light': light,
            'motion': motion,
            'label': 1 if is_overload else 0
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

def main():
    """Main function"""
    print("Generating training data...")
    
    # Generate data
    df = generate_simple_data(1000)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} samples to {output_file}")
    
    # Show summary
    print(f"Normal samples: {len(df[df['label'] == 0])}")
    print(f"Overload samples: {len(df[df['label'] == 1])}")
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    main()
