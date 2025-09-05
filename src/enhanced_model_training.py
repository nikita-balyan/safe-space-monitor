#!/usr/bin/env python3
"""
Enhanced model training with rolling features
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuration
data_dir = "../data"
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

features_file = f"{data_dir}/enhanced_features_data.csv"
model_file = f"{models_dir}/enhanced_overload_model.joblib"

def load_enhanced_data():
    """Load enhanced training data"""
    print(f"Loading enhanced data from {features_file}")
    df = pd.read_csv(features_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Normal samples: {len(df[df['label'] == 0])}")
    print(f"Overload samples: {len(df[df['label'] == 1])}")
    
    # Show feature columns
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'noise', 'light', 'motion', 'label']]
    print(f"Number of rolling features: {len(feature_cols)}")
    print(f"Feature examples: {feature_cols[:5]}")
    
    return df

def train_enhanced_model():
    """Train model with enhanced features"""
    print("=== Starting Enhanced Training ===")
    
    # Load data
    df = load_enhanced_data()
    
    # Prepare features (use rolling features only)
    base_cols = ['timestamp', 'noise', 'light', 'motion', 'label']
    feature_cols = [col for col in df.columns if col not in base_cols]
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train Random Forest
    print("\nTraining Enhanced Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save model
    joblib.dump(model, model_file)
    print(f"Enhanced model saved to {model_file}")
    
    # Save metadata with feature information
    metadata = {
        'model_type': 'EnhancedRandomForest',
        'training_date': datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'feature_names': feature_cols,
        'num_features': len(feature_cols)
    }
    
    metadata_file = f"{models_dir}/enhanced_model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Enhanced metadata saved to {metadata_file}")
    
    return model, metadata

if __name__ == "__main__":
    model, metadata = train_enhanced_model()
    print(f"\nEnhanced training completed!")
    print(f"Final F1 score: {metadata['f1_score']:.3f}")
    print(f"Using {metadata['num_features']} rolling features")
