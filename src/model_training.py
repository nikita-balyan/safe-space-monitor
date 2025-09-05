#!/usr/bin/env python3
"""
Simple model training script
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuration
data_dir = "../data"
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

features_file = f"{data_dir}/features_data.csv"
model_file = f"{models_dir}/overload_model.joblib"

def load_data():
    """Load training data"""
    print(f"Loading data from {features_file}")
    df = pd.read_csv(features_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Normal samples: {len(df[df['label'] == 0])}")
    print(f"Overload samples: {len(df[df['label'] == 1])}")
    
    return df

def train_simple_model():
    """Train a simple model"""
    print("=== Starting Training ===")
    
    # Load data
    df = load_data()
    
    # Prepare features
    X = df[['noise', 'light', 'motion']].copy()
    y = df['label'].copy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Save model
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")
    
    # Save metadata
    metadata = {
        'model_type': 'RandomForest',
        'training_date': datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    metadata_file = f"{models_dir}/model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")
    
    return model, metadata

if __name__ == "__main__":
    model, metadata = train_simple_model()
    print(f"\nTraining completed!")
    print(f"Final F1 score: {metadata['f1_score']:.3f}")
