#!/usr/bin/env python3
"""
ML Model Training Script for Overload Prediction
Trains Logistic Regression and Decision Tree models on sensor data with features
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

FEATURES_FILE = DATA_DIR / "features_data.csv"
MODEL_PATH = MODELS_DIR / "overload_model.joblib"
THRESHOLD_PATH = MODELS_DIR / "model_threshold.txt"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"

class OverloadPredictor:
    """ML Pipeline for overload prediction"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.threshold = 0.5
        self.metadata = {}
        
    def load_data(self):
        """Load and prepare training data"""
        if not FEATURES_FILE.exists():
            raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
        
        print(f"Loading data from {FEATURES_FILE}")
        df = pd.read_csv(FEATURES_FILE)
        
        if len(df) < 100:
            print(f"Warning: Only {len(df)} samples available. Consider collecting more data.")
        
        print(f"Dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Define feature columns (exclude timestamp and label)
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'label']]
        
        X = df[feature_cols].copy()
        y = df['label'].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Store feature names
        self.feature_names = feature_cols
        
        print(f"Features selected: {len(feature_cols)}")
        print(f"Feature names: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples") 
        print(f"Test set: {len(X_test)} samples")
        
        # Store test set for final evaluation
        self.X_test, self.y_test = X_test, y_test
        
        # Model configurations
        model_configs = {
            'logistic_regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(random_state=42, max_iter=1000))
                ]),
                'params': {
                    'lr__C': [0.1, 1.0, 10.0],
                    'lr__class_weight': ['balanced', None]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=42),
                'params': {
                    'max_depth': [3, 5, 7, 10],
                    'min_samples_split': [5, 10, 20],
                    'min_samples_leaf': [2, 5, 10],
                    'class_weight': ['balanced', None]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42, n_estimators=100),
                'params': {
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [5, 10],
                    'min_samples_leaf': [2, 5],
                    'class_weight': ['balanced', None]
                }
            }
        }
        
        best_score = 0
        results = {}
        
        # Train and evaluate each model
        for name, config in model_configs.items():
            print(f"\n--- Training {name.replace('_', ' ').title()} ---")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'],
                cv=5,
                scoring='f1',  # Focus on F1 score for imbalanced data
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate on validation set
            best_model = grid_search.best_estimator_
            val_predictions = best_model.predict(X_val)
            val_probabilities = best_model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, val_predictions),
                'precision': precision_score(y_val, val_predictions),
                'recall': recall_score(y_val, val_predictions),
                'f1': f1_score(y_val, val_predictions),
                'roc_auc': roc_auc_score(y_val, val_probabilities)
            }
            
            results[name] = {
                'model': best_model,
                'params': grid_search.best_params_,
                'metrics': metrics,
                'cv_score': grid_search.best_score_
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Validation metrics: {metrics}")
            
            # Track best model
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                self.best_model = best_model
                self.best_model_name = name
        
        self.models = results
        print(f"\nBest model: {self.best_model_name} (F1: {best_score:.3f})")
        
        return results
    
    def optimize_threshold(self, X_val, y_val):
        """Optimize prediction threshold for best F1 score"""
        if self.best_model is None:
            return 0.5
        
        probabilities = self.best_model.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            f1 = f1_score(y_val, predictions)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        
        return best_threshold
    
    def final_evaluation(self):
        """Evaluate best model on test set"""
        if self.best_model is None:
            raise ValueError("No model trained")
        
        # Predictions on test set
        test_predictions = self.best_model.predict(self.X_test)
        test_probabilities = self.best_model.predict_proba(self.X_test)[:, 1]
        
        # Apply optimized threshold
        threshold_predictions = (test_probabilities >= self.threshold).astype(int)
        
        # Calculate final metrics
        final_metrics = {
            'test_accuracy': accuracy_score(self.y_test, test_predictions),
            'test_precision': precision_score(self.y_test, test_predictions),
            'test_recall': recall_score(self.y_test, test_predictions),
            'test_f1': f1_score(self.y_test, test_predictions),
            'test_roc_auc': roc_auc_score(self.y_test, test_probabilities),
            'threshold_accuracy': accuracy_score(self.y_test, threshold_predictions),
            'threshold_precision': precision_score(self.y_test, threshold_predictions),
            'threshold_recall': recall_score(self.y_test, threshold_predictions),
            'threshold_f1': f1_score(self.y_test, threshold_predictions)
        }
        
        print("\n=== FINAL TEST RESULTS ===")
        print("Default threshold (0.5):")
        for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc']:
            print(f"  {metric}: {final_metrics[metric]:.3f}")
        
        print(f"\nOptimized threshold ({self.threshold:.3f}):")
        for metric in ['threshold_accuracy', 'threshold_precision', 'threshold_recall', 'threshold_f1']:
            print(f"  {metric}: {final_metrics[metric]:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, threshold_predictions)
        print(f"\nConfusion Matrix (threshold={self.threshold:.3f}):")
        print(cm)
        
        return final_metrics
    
    def save_model(self):
        """Save the best model and metadata"""
        if self.best_model is None:
            raise ValueError("No model to save")
        
        # Save model
        joblib.dump(self.best_model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
        
        # Save threshold
        with open(THRESHOLD_PATH, 'w') as f:
            f.write(str(self.threshold))
        print(f"Threshold saved to {THRESHOLD_PATH}")
        
        # Save metadata
        self.metadata = {
            'model_type': self.best_model_name,
            'training_date': datetime.now().isoformat(),
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'num_features': len(self.feature_names),
            'model_params': self.models[self.best_model_name]['params'],
            'metrics': self.models[self.best_model_name]['metrics']
        }
        
        with open(METADATA_PATH, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"Metadata saved to {METADATA_PATH}")
    
    def train_pipeline(self):
        """Complete training pipeline"""
        print("=== Starting ML Training Pipeline ===")
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Check class balance
        if y.sum() < 10:
            print("Warning: Very few positive samples. Consider collecting more overload data.")
        
        # Train models
        results = self.train_models(X, y)
        
        # For threshold optimization, we need validation data
        # Split again for threshold optimization (use part of training data)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X.iloc[:-len(self.X_test)], y.iloc[:-len(self.y_test)], 
            test_size=0.2, random_state=42, stratify=y.iloc[:-len(self.y_test)]
        )
        
        # Optimize threshold
        self.optimize_threshold(X_val_split, y_val_split)
        
        # Final evaluation
        final_metrics = self.final_evaluation()
        
        # Save model
        self.save_model()
        
        print("\n=== Training Complete ===")
        return final_metrics

def collect_training_data_if_needed():
    """Collect training data if features file doesn't exist or is too small"""
    if not FEATURES_FILE.exists():
        print("No features data found. Collecting training data...")
        
        # Import and run data collection
        from sensor_simulator import start_sensor_simulation, collect_training_data
        import time
        
        # Start simulation
        start_sensor_simulation(interval=1)
        print("Waiting for sensor buffer to fill...")
        time.sleep(15)
        
        # Collect 3 minutes of data
        collect_training_data(duration_minutes=3)
        print("Training data collection complete.")
    else:
        df = pd.read_csv(FEATURES_FILE)
        if len(df) < 100:
            print(f"Only {len(df)} samples found. Collecting more data...")
            
            from sensor_simulator import start_sensor_simulation, collect_training_data
            import time
            
            start_sensor_simulation(interval=1)
            time.sleep(15)
            collect_training_data(duration_minutes=2)

if __name__ == "__main__":
    # Ensure we have training data
    collect_training_data_if_needed()
    
    # Train model
    predictor = OverloadPredictor()
    metrics = predictor.train_pipeline()
    
    print(f"\nModel training completed successfully!")
    print(f"Best model: {predictor.best_model_name}")
    print(f"Final F1 score: {metrics['threshold_f1']:.3f}")