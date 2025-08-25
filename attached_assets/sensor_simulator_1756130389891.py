# sensor_simulator.py
import os
import csv
import time
import random
import argparse
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# Global Configurations
# ==============================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RAW_FILE = DATA_DIR / "sensor_data.csv"
PROCESSED_FILE = DATA_DIR / "processed_data.csv"
MODEL_FILE = DATA_DIR / "model.pkl"

# Thresholds
THRESHOLDS = {
    "noise": 70,
    "light": 3000,
    "motion": 50
}

# Latest reading (for Flask integration)
latest_reading = {"noise": 0, "light": 0, "motion": 0, "label": 0, "timestamp": datetime.now().isoformat()}

# ==============================
# Data Simulator
# ==============================
class DataSimulator:
    """Simulates sensor readings and logs them into a CSV file."""

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
        new_noise = random.randint(20, 130)
        new_light = random.randint(100, 10000)
        new_motion = random.randint(0, 100)

        # EMA smoothing
        self.noise = 0.7 * self.noise + 0.3 * new_noise
        self.light = 0.7 * self.light + 0.3 * new_light
        self.motion = 0.7 * self.motion + 0.3 * new_motion
        return self.noise, self.light, self.motion

    def log_once(self):
        n, l, m = self._simulate_once()
        label = 1 if (n > 70 and m > 30) else 0
        timestamp = datetime.now().isoformat()

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), n, l, m, label])
        return n, l, m, label


# ==============================
# Data Collection
# ==============================
def collect_training_data(minutes=5):
    sim = DataSimulator()
    print(f"📡 Collecting ~{minutes} min of data into {sim.csv_path} ...")
    for sec in range(minutes * 60):
        sim.log_once()
        if sec % 60 == 0 and sec > 0:
            print(f"⏳ {sec} sec logged")
        time.sleep(1)
    print("✅ Data collection complete.")

# ==============================
# Data Processing
# ==============================
def label_data(df):
    df["noise_label"] = df["noise"].apply(lambda x: "high" if x > THRESHOLDS["noise"] else "low")
    df["light_label"] = df["light"].apply(lambda x: "bright" if x > THRESHOLDS["light"] else "dim")
    df["motion_label"] = df["motion"].apply(lambda x: "active" if x > THRESHOLDS["motion"] else "still")
    return df

def add_risk_label(df):
    df["risk_label"] = df.apply(
        lambda row: 1 if (row["noise"] > 75 or row["motion"] > 50) else 0,
        axis=1
    )
    return df

def extract_features(df):
    for col in ["noise", "light", "motion"]:
        df[f"{col}_mean"] = df[col].rolling(window=10, min_periods=1).mean()
        df[f"{col}_std"] = df[col].rolling(window=10, min_periods=1).std().fillna(0)
    return df

def process_data():
    if not RAW_FILE.exists():
        print("❌ Raw CSV not found. Run with --collect first.")
        return
    df = pd.read_csv(RAW_FILE)
    print(f"✅ Loaded {len(df)} rows from {RAW_FILE}")

    df = label_data(df)
    df = extract_features(df)
    df = add_risk_label(df)

    df.to_csv(PROCESSED_FILE, index=False)
    print(f"✅ Processed data saved to {PROCESSED_FILE}")
    return df

# ==============================
# Model Training
# ==============================
def train_model():
    if not RAW_FILE.exists():
        print("❌ No training data found. Please run with --collect first.")
        return

    df = pd.read_csv(RAW_FILE)
    X = df[["noise", "light", "motion"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model trained. Accuracy: {acc:.2f}")

    joblib.dump(model, MODEL_FILE)
    print(f"📦 Model saved at {MODEL_FILE}")

# ==============================
# Prediction
# ==============================
def predict_once():
    if not MODEL_FILE.exists():
        print("❌ No trained model found. Run with --train first.")
        return

    model = joblib.load(MODEL_FILE)
    sim = DataSimulator(csv_path=DATA_DIR / "temp.csv")
    n, l, m = sim._simulate_once()
    pred = model.predict([[n, l, m]])[0]
    print(f"📊 Noise={n:.1f}, Light={l:.1f}, Motion={m:.1f} → Prediction={pred}")

# ==============================
# Flask Compatibility
# ==============================
def get_current_readings():
    return latest_reading

def get_historical_data(n=50):
    """Return the last n sensor readings from the CSV."""
    if not RAW_FILE.exists():
        return []
    df = pd.read_csv(RAW_FILE)
    if df.empty:
        return []
            
    # Convert to list of dictionaries with proper formatting
    historical_data = df.tail(n).to_dict('records')
        
    # Ensure all records have all required fields
    for record in historical_data:
        if 'timestamp' not in record:
            record['timestamp'] = datetime.now().isoformat()
                
        return historical_data
    except Exception as e:
        print(f"Error reading historical data: {e}")
        return []

def start_sensor_simulation(interval=2):
    import threading
    def run():
        sim = DataSimulator()
        while True:
            n, l, m, label, timestamp = sim.log_once()
            latest_reading.update({"noise": n, "light": l, "motion": m, "label": label, "timestamp": datetime.now().isoformat()})
            time.sleep(interval)
    t = threading.Thread(target=run, daemon=True)
    t.start()

# ==============================
# CLI Entry Point
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor Simulator")
    parser.add_argument("--collect", type=int, help="Collect training data for N minutes")
    parser.add_argument("--process", action="store_true", help="Process raw data and extract features")
    parser.add_argument("--train", action="store_true", help="Train model on collected data")
    parser.add_argument("--predict", action="store_true", help="Simulate one reading and predict")
    args = parser.parse_args()

    if args.collect:
        collect_training_data(args.collect)
    elif args.process:
        process_data()
    elif args.train:
        train_model()
    elif args.predict:
        predict_once()
    else:
        print("⚡ Usage: python sensor_simulator.py --collect MINUTES | --process | --train | --predict")
