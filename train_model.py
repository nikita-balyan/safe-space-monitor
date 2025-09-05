import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

def train_enhanced_model():
    """Train an enhanced model using collected training data"""
    try:
        # Load your collected data
        df = pd.read_csv('training_data.csv')
        print(f"Loaded {len(df)} training samples")
        
        # Prepare features and labels
        X = df[['noise', 'light', 'motion']]
        y = df['overload']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        # Save model
        model_path = 'models/enhanced_overload_model.joblib'
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "training_date": datetime.now().isoformat(),
            "training_samples": len(df),
            "features": list(X.columns),
            "train_accuracy": float(train_score),
            "test_accuracy": float(test_score),
            "model_type": "RandomForest"
        }
        
        print(f"Model saved to {model_path}")
        return model, metadata
        
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

if __name__ == "__main__":
    train_enhanced_model()