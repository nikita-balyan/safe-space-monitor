import os
import csv
import time
import random
import argparse
import threading
import pandas as pd
from datetime import datetime
from pathlib import Path

# ==============================
# Global Configurations
# ==============================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RAW_FILE = DATA_DIR / "sensor_data.csv"
PROCESSED_FILE = DATA_DIR / "processed_data.csv"

# Thresholds
THRESHOLDS = {
    "noise": 70,
    "light": 3000,
    "motion": 50
}

# Global data storage
latest_reading = {
    "noise": 50.0, 
    "light": 1000.0, 
    "motion": 20.0, 
    "label": 0, 
    "timestamp": datetime.now().isoformat()
}

historical_data = []
MAX_HISTORY = 60  # Keep last 60 seconds

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
        
        # Generate label based on thresholds (overload detection)
        label = 1 if (n > 100 or l > 8000 or m > 80) else 0
        timestamp = datetime.now().isoformat()

        # Log to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, n, l, m, label])
            
        return n, l, m, label, timestamp

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
        return historical_data[-n:].copy() if historical_data else []

def start_sensor_simulation(interval=1):
    """Start the sensor simulation in a background thread"""
    def run_simulation():
        global latest_reading, historical_data
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
                    
                    # Add to historical data
                    historical_data.append(latest_reading.copy())
                    
                    # Keep only last MAX_HISTORY readings
                    if len(historical_data) > MAX_HISTORY:
                        historical_data.pop(0)
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Error in sensor simulation: {e}")
                time.sleep(interval)
    
    # Start background thread
    thread = threading.Thread(target=run_simulation, daemon=True)
    thread.start()
    print(f"Sensor simulation started with {interval}s interval")

# ==============================
# CLI Entry Point
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor Simulator")
    parser.add_argument("--simulate", type=int, default=60, help="Run simulation for N seconds")
    args = parser.parse_args()

    print(f"Running sensor simulation for {args.simulate} seconds...")
    start_sensor_simulation(interval=1)
    
    # Keep main thread alive
    time.sleep(args.simulate)
    print("Simulation complete.")
