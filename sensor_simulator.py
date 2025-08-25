import os
import csv
import time
import random
import argparse
import threading
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import deque

# ==============================
# Global Configurations
# ==============================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RAW_FILE = DATA_DIR / "sensor_data.csv"
PROCESSED_FILE = DATA_DIR / "processed_data.csv"
FEATURES_FILE = DATA_DIR / "features_data.csv"
FEEDBACK_FILE = DATA_DIR / "user_feedback.csv"

# Thresholds for overload detection
THRESHOLDS = {
    "noise": {"warning": 70, "danger": 100},
    "light": {"warning": 3000, "danger": 8000},
    "motion": {"warning": 50, "danger": 80}
}

# Global data storage
latest_reading = {
    "noise": 50.0, 
    "light": 1000.0, 
    "motion": 20.0, 
    "label": 0, 
    "timestamp": datetime.now().isoformat()
}

# Enhanced data storage for feature engineering
historical_data = deque(maxlen=300)  # 5 minutes at 1Hz
feature_buffer = deque(maxlen=60)    # 60 seconds for real-time features
MAX_HISTORY = 60  # Keep last 60 seconds for API

# Threading lock for thread-safe operations
data_lock = threading.Lock()

# ==============================
# Data Simulator
# ==============================
class DataSimulator:
    """Simulates realistic sensor readings and logs them into a CSV file."""

    def __init__(self, csv_path=RAW_FILE):
        self.noise = 50.0
        self.light = 1000.0
        self.motion = 20.0
        self.csv_path = Path(csv_path)
        self._init_csv()

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "noise", "light", "motion", "label"])

    def _simulate_once(self):
        # Generate realistic sensor data with some variation
        current_hour = datetime.now().hour
        
        # Noise: quieter at night, louder during day
        base_noise = 40 if 22 <= current_hour or current_hour <= 6 else 60
        new_noise = max(30, min(120, base_noise + random.randint(-20, 40)))
        
        # Light: follows day/night cycle
        if 6 <= current_hour <= 18:
            base_light = 5000  # Daytime
        else:
            base_light = 500   # Nighttime
        new_light = max(100, min(10000, base_light + random.randint(-2000, 3000)))
        
        # Motion: random activity
        new_motion = random.randint(0, 100)

        # EMA smoothing for realistic transitions
        self.noise = 0.7 * self.noise + 0.3 * new_noise
        self.light = 0.7 * self.light + 0.3 * new_light
        self.motion = 0.7 * self.motion + 0.3 * new_motion
        
        return round(self.noise, 1), round(self.light, 1), round(self.motion, 1)

    def log_once(self):
        n, l, m = self._simulate_once()
        
        # Generate label based on overload thresholds
        # Overload = 1 if ANY sensor exceeds danger threshold
        label = 1 if (n > THRESHOLDS["noise"]["danger"] or 
                      l > THRESHOLDS["light"]["danger"] or 
                      m > THRESHOLDS["motion"]["danger"]) else 0
        
        timestamp = datetime.now().isoformat()

        # Log to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, n, l, m, label])
            
        return n, l, m, label, timestamp

# ==============================
# Feature Engineering Functions
# ==============================
def extract_rolling_features(data, window_sizes=[10, 30, 60]):
    """Extract rolling statistical features from sensor data"""
    if len(data) < min(window_sizes):
        return {}
    
    features = {}
    
    # Convert deque to DataFrame for easier processing
    df = pd.DataFrame(list(data))
    
    for window in window_sizes:
        if len(df) >= window:
            # Rolling statistics for each sensor
            for sensor in ['noise', 'light', 'motion']:
                if sensor in df.columns:
                    rolling_data = df[sensor].rolling(window=window, min_periods=1)
                    
                    features.update({
                        f'{sensor}_mean_{window}': rolling_data.mean().iloc[-1],
                        f'{sensor}_std_{window}': rolling_data.std().iloc[-1],
                        f'{sensor}_min_{window}': rolling_data.min().iloc[-1],
                        f'{sensor}_max_{window}': rolling_data.max().iloc[-1],
                        f'{sensor}_range_{window}': rolling_data.max().iloc[-1] - rolling_data.min().iloc[-1]
                    })
    
    return features

def calculate_fft_energy(data, sensor='noise'):
    """Calculate FFT energy for a sensor (simplified frequency domain feature)"""
    if len(data) < 10:
        return 0
    
    try:
        values = [reading[sensor] for reading in data if sensor in reading]
        if len(values) < 10:
            return 0
        
        # Simple FFT energy calculation
        fft = np.fft.fft(values[-30:])  # Last 30 readings
        energy = np.sum(np.abs(fft)**2)
        return float(energy)
    except:
        return 0

def get_current_features():
    """Get current sensor reading with extracted features for ML prediction"""
    with data_lock:
        if len(feature_buffer) < 10:
            return None
        
        # Get rolling features
        features = extract_rolling_features(feature_buffer)
        
        # Add FFT energy features
        for sensor in ['noise', 'light', 'motion']:
            features[f'{sensor}_fft_energy'] = calculate_fft_energy(feature_buffer, sensor)
        
        # Add current readings
        current = latest_reading.copy()
        features.update({
            'noise': current['noise'],
            'light': current['light'],
            'motion': current['motion'],
            'timestamp': current['timestamp']
        })
        
        return features

# ==============================
# Flask Compatibility Functions
# ==============================
def get_current_readings():
    """Return the latest sensor reading"""
    with data_lock:
        return latest_reading.copy()

def get_historical_data(n=60):
    """Return the last n sensor readings"""
    with data_lock:
        if not historical_data:
            return []
        # Convert deque to list and slice
        data_list = list(historical_data)
        return data_list[-n:] if len(data_list) >= n else data_list

def start_sensor_simulation(interval=1):
    """Start the sensor simulation in a background thread"""
    def run_simulation():
        global latest_reading, historical_data, feature_buffer
        sim = DataSimulator()
        
        while True:
            try:
                n, l, m, label, timestamp = sim.log_once()
                
                # Update latest reading
                with data_lock:
                    latest_reading.update({
                        "noise": n, 
                        "light": l, 
                        "motion": m, 
                        "label": label, 
                        "timestamp": timestamp
                    })
                    
                    # Add to historical data (for API)
                    historical_data.append(latest_reading.copy())
                    
                    # Add to feature buffer (for ML features)
                    feature_buffer.append(latest_reading.copy())
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in sensor simulation: {e}")
                time.sleep(interval)
    
    # Start background thread
    thread = threading.Thread(target=run_simulation, daemon=True)
    thread.start()
    print(f"Sensor simulation started with {interval}s interval")

# ==============================
# Data Collection for Training
# ==============================
def save_features_to_csv():
    """Save current features with labels to CSV for model training"""
    try:
        features = get_current_features()
        if not features:
            return
        
        # Ensure features CSV exists with headers
        if not FEATURES_FILE.exists():
            headers = ['timestamp', 'noise', 'light', 'motion', 'label'] + \
                     [f'{sensor}_{stat}_{window}' 
                      for sensor in ['noise', 'light', 'motion'] 
                      for stat in ['mean', 'std', 'min', 'max', 'range'] 
                      for window in [10, 30, 60]] + \
                     [f'{sensor}_fft_energy' for sensor in ['noise', 'light', 'motion']]
            
            with open(FEATURES_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        # Append current features
        with open(FEATURES_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Build row with all features
            row = [
                features['timestamp'],
                features['noise'],
                features['light'], 
                features['motion'],
                latest_reading['label']
            ]
            
            # Add rolling features
            for sensor in ['noise', 'light', 'motion']:
                for stat in ['mean', 'std', 'min', 'max', 'range']:
                    for window in [10, 30, 60]:
                        key = f'{sensor}_{stat}_{window}'
                        row.append(features.get(key, 0))
            
            # Add FFT features  
            for sensor in ['noise', 'light', 'motion']:
                key = f'{sensor}_fft_energy'
                row.append(features.get(key, 0))
            
            writer.writerow(row)
            
    except Exception as e:
        print(f"Error saving features: {e}")

def save_user_feedback(prediction, actual_label, timestamp):
    """Save user feedback for model improvement"""
    try:
        if not FEEDBACK_FILE.exists():
            with open(FEEDBACK_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'prediction', 'actual_label', 'correct'])
        
        with open(FEEDBACK_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            correct = 1 if prediction == actual_label else 0
            writer.writerow([timestamp, prediction, actual_label, correct])
            
    except Exception as e:
        print(f"Error saving feedback: {e}")

def collect_training_data(duration_minutes=5):
    """Collect training data with features for specified duration"""
    print(f"Collecting training data for {duration_minutes} minutes...")
    start_time = time.time()
    count = 0
    
    while time.time() - start_time < duration_minutes * 60:
        if len(feature_buffer) >= 10:  # Ensure we have enough data for features
            save_features_to_csv()
            count += 1
            
        time.sleep(1)  # Collect every second
        
    print(f"Training data collection complete. Saved {count} feature records.")

# ==============================
# CLI Entry Point
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor Simulator")
    parser.add_argument("--simulate", type=int, default=60, help="Run simulation for N seconds")
    parser.add_argument("--collect", type=int, help="Collect training data for N minutes")
    args = parser.parse_args()

    if args.collect:
        # Start simulation and collect training data
        start_sensor_simulation(interval=1)
        time.sleep(10)  # Wait for buffer to fill
        collect_training_data(args.collect)
    else:
        print(f"Running sensor simulation for {args.simulate} seconds...")
        start_sensor_simulation(interval=1)
        
        # Keep main thread alive
        time.sleep(args.simulate)
        print("Simulation complete.")
