# prediction_service.py
"""
Simplified Prediction Service for Real-Time Overload Detection
Fixed version with better error handling and fallbacks
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os

class PredictionService:
    """Handles model loading and predictions with robust error handling"""
    
    def __init__(self):
        self.model = None
        self.metadata = {}
        self.feature_names = ['noise', 'light', 'motion']
        self.threshold = 0.5
        self.is_loaded = False
        self.load_model()
        
    def load_model(self):
        """Load the trained model with multiple fallback options"""
        try:
            # Try to load the enhanced model first
            model_path = Path("models/enhanced_overload_model.joblib")
            metadata_path = Path("models/enhanced_model_metadata.json")
            
            if model_path.exists():
                self.model = joblib.load(model_path)
                
                # Load metadata if available
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    self.feature_names = self.metadata.get('features', ['noise', 'light', 'motion'])
                    self.threshold = self.metadata.get('threshold', 0.5)
                
                # Validate model is trained
                if hasattr(self.model, 'predict'):
                    self.is_loaded = True
                    print("✅ Enhanced model loaded successfully")
                    print(f"📊 Model info: {len(self.feature_names)} features, threshold: {self.threshold}")
                    return True
                else:
                    print("⚠️ Model file exists but model is invalid")
            
            # If no model exists, create a simple fallback
            print("🔄 No trained model found. Using fallback threshold-based prediction.")
            self.is_loaded = False
            return False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
            return False
    
    def predict(self, sensor_data):
        """Make prediction with comprehensive error handling"""
        # Use enhanced model if available and loaded
        if self.is_loaded and self.model is not None:
            try:
                # Prepare features
                features = self._prepare_features(sensor_data)
                
                # Make prediction
                prediction = self.model.predict(features)[0]
                probability = self.model.predict_proba(features)[0, 1]
                confidence = np.max(self.model.predict_proba(features)[0])
                
                return {
                    "probability": float(probability),
                    "prediction": int(prediction),
                    "confidence": float(confidence),
                    "threshold": float(self.threshold),
                    "model_used": "enhanced_random_forest",
                    "features_used": len(self.feature_names),
                    "error": None
                }
                
            except Exception as e:
                print(f"⚠️ Enhanced model prediction failed: {e}")
                # Fall through to fallback prediction
        
        # Fallback: Use simple threshold-based prediction
        return self._fallback_prediction(sensor_data)
    
    def _prepare_features(self, sensor_data):
        """Prepare features for model prediction with NaN handling"""
        try:
            # Extract and validate features
            features = []
            for feature_name in self.feature_names:
                value = sensor_data.get(feature_name, 0)
                # Handle None or invalid values
                if value is None or np.isnan(value):
                    # Use reasonable defaults based on feature type
                    if feature_name == 'noise':
                        value = 50.0
                    elif feature_name == 'light':
                        value = 500.0
                    elif feature_name == 'motion':
                        value = 0.5
                    else:
                        value = 0.0
                features.append(float(value))
            
            # Create DataFrame for proper feature naming
            feature_df = pd.DataFrame([features], columns=self.feature_names)
            
            return feature_df
            
        except Exception as e:
            print(f"❌ Feature preparation error: {e}")
            # Return default features
            default_features = [50.0, 500.0, 0.5]  # noise, light, motion
            return pd.DataFrame([default_features], columns=self.feature_names)
    
    def _fallback_prediction(self, sensor_data):
        """Fallback prediction using simple thresholds"""
        try:
            noise = sensor_data.get('noise', 50)
            light = sensor_data.get('light', 500)
            motion = sensor_data.get('motion', 0.5)
            
            # Handle NaN values in fallback
            if noise is None or np.isnan(noise):
                noise = 50
            if light is None or np.isnan(light):
                light = 500
            if motion is None or np.isnan(motion):
                motion = 0.5
            
            # Simple weighted threshold model
            risk_score = 0.0
            
            # Noise contribution (0-0.4)
            if noise > 80:
                risk_score += 0.4
            elif noise > 70:
                risk_score += 0.2
            elif noise > 60:
                risk_score += 0.1
            
            # Light contribution (0-0.4)
            if light > 800:
                risk_score += 0.4
            elif light > 600:
                risk_score += 0.2
            elif light > 400:
                risk_score += 0.1
            
            # Motion contribution (0-0.2)
            if motion > 0.8:
                risk_score += 0.2
            elif motion > 0.6:
                risk_score += 0.1
            
            probability = min(risk_score, 1.0)
            prediction = 1 if probability > self.threshold else 0
            
            # Confidence based on how far from threshold
            confidence = abs(probability - 0.5) * 2  # 0-1 scale
            
            return {
                "probability": float(probability),
                "prediction": int(prediction),
                "confidence": float(confidence),
                "threshold": float(self.threshold),
                "model_used": "fallback_threshold",
                "features_used": 3,
                "error": "Using fallback prediction"
            }
            
        except Exception as e:
            print(f"❌ Fallback prediction also failed: {e}")
            # Ultimate fallback
            return {
                "probability": 0.5,
                "prediction": 0,
                "confidence": 0.0,
                "threshold": 0.5,
                "model_used": "emergency_fallback",
                "features_used": 0,
                "error": "All prediction methods failed"
            }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {
                "model_loaded": False,
                "message": "Using fallback threshold-based prediction",
                "features": self.feature_names,
                "threshold": self.threshold
            }
        
        return {
            "model_loaded": True,
            "model_type": self.metadata.get('model_type', 'RandomForest'),
            "accuracy": self.metadata.get('test_accuracy', 0),
            "training_samples": self.metadata.get('training_samples', 0),
            "feature_count": len(self.feature_names),
            "training_date": self.metadata.get('training_date', 'unknown'),
            "threshold": self.threshold,
            "feature_importance": self.metadata.get('feature_importance', {})
        }
    
    def train_model_if_needed(self):
        """Train a new model if none exists"""
        if not self.is_loaded:
            print("🔄 No model found. Attempting to train a new one...")
            try:
                from model_training import train_enhanced_model
                model, metadata = train_enhanced_model()
                if model is not None:
                    self.model = model
                    self.metadata = metadata
                    self.is_loaded = True
                    print("✅ New model trained and loaded successfully")
                    return True
                else:
                    print("❌ Model training failed")
                    return False
            except Exception as e:
                print(f"❌ Model training attempt failed: {e}")
                return False
        return True

# Global instance for easy import
prediction_service = PredictionService()

# Initialize on import
print("🚀 Initializing Prediction Service...")
if not prediction_service.is_loaded:
    print("📝 Note: No trained model found. Using fallback prediction mode.")
    print("💡 Run 'python model_training.py' to train a model for better accuracy.")