#!/usr/bin/env python3
"""
Prediction service for real-time overload detection
"""

import joblib
import numpy as np
from pathlib import Path
import json

class PredictionService:
    """Handles model loading and predictions"""
    
    def __init__(self):
        self.model = None
        self.metadata = {}
        self.feature_names = []
        self.threshold = 0.5
        self.is_loaded = False
        
        # Create a SINGLE FeatureEngineer instance that will be reused
        from feature_engineer import FeatureEngineer
        self.feature_engineer = FeatureEngineer()
        
    def load_model(self):
        """Load the trained model and metadata"""
        try:
            # Try multiple possible model paths
            model_paths = [
                Path("models/enhanced_overload_model.joblib"),
                Path("../models/enhanced_overload_model.joblib"),
                Path("./enhanced_overload_model.joblib"),
                Path("enhanced_model.joblib"),
                Path("basic_model.joblib")
            ]
            
            model_path = None
            for path in model_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path:
                self.model = joblib.load(model_path)
                
                # Try to load metadata
                metadata_paths = [
                    Path("models/enhanced_model_metadata.json"),
                    Path("../models/enhanced_model_metadata.json"),
                    Path(f"{model_path}.metadata.json"),
                    Path("enhanced_model_metadata.json"),
                    Path("basic_model_metadata.json")
                ]
                
                metadata_loaded = False
                for meta_path in metadata_paths:
                    if meta_path.exists():
                        with open(meta_path, 'r') as f:
                            self.metadata = json.load(f)
                        metadata_loaded = True
                        break
                
                if metadata_loaded:
                    self.feature_names = self.metadata.get('feature_names', [])
                    self.threshold = self.metadata.get('threshold', 0.5)
                    self.is_loaded = True
                    
                    print("✓ Enhanced model loaded successfully")
                    print(f"✓ Using {len(self.feature_names)} features")
                    if 'f1_score' in self.metadata:
                        print(f"✓ Model performance: F1={self.metadata['f1_score']:.3f}")
                else:
                    print("⚠ Model metadata not found")
                    self.is_loaded = False
            else:
                print("⚠ No model files found")
                self.is_loaded = False
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.is_loaded = False
    
    def add_reading(self, timestamp, noise, light, motion):
        """Add a sensor reading to the feature engineer"""
        self.feature_engineer.add_reading(timestamp, noise, light, motion)
    
    def predict(self, timestamp, noise, light, motion):
        """Make a prediction for current sensor readings"""
        # First add the new reading to our feature engineer
        self.add_reading(timestamp, noise, light, motion)
        
        if not self.is_loaded or self.model is None:
            return {
                "error": "Model not loaded",
                "probability": None,
                "prediction": None,
                "confidence": "low",
                "samples_collected": len(self.feature_engineer.noise_buffer)
            }
        
        try:
            # Check if we have enough data
            if not self.feature_engineer.has_enough_data():
                return {
                    "warming_up": True,
                    "samples_collected": len(self.feature_engineer.noise_buffer),
                    "samples_needed": self.feature_engineer.max_window,
                    "probability": None,
                    "prediction": None,
                    "confidence": "low"
                }
            
            # Extract features and make prediction
            features = self.feature_engineer.extract_features()
            feature_vector = []
            
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0))
            
            probability = self.model.predict_proba([feature_vector])[0, 1]
            prediction = 1 if probability > self.threshold else 0
            
            # Determine confidence level
            if probability > 0.8 or probability < 0.2:
                confidence = "high"
            elif probability > 0.6 or probability < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            return {
                "probability": float(probability),
                "prediction": int(prediction),
                "confidence": confidence,
                "threshold": float(self.threshold),
                "timestamp": timestamp,
                "features_used": len(self.feature_names),
                "samples_available": len(self.feature_engineer.noise_buffer)
            }
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "probability": None,
                "prediction": None,
                "confidence": "low",
                "samples_collected": len(self.feature_engineer.noise_buffer)
            }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"model_loaded": False}
        
        return {
            "model_loaded": True,
            "model_type": self.metadata.get('model_type', 'unknown'),
            "accuracy": self.metadata.get('accuracy', 0),
            "recall": self.metadata.get('recall', 0),
            "f1_score": self.metadata.get('f1_score', 0),
            "feature_count": len(self.feature_names),
            "training_date": self.metadata.get('training_date', 'unknown'),
            "threshold": self.threshold
        }
    
    def clear_buffers(self):
        """Clear the feature engineer buffers"""
        self.feature_engineer.clear_buffers()
        print("✓ Feature buffers cleared")


# Global instance for easy import
prediction_service = PredictionService()