import numpy as np
import pandas as pd
from train_model import load_enhanced_model

class EnhancedSensoryModel:
    def __init__(self):
        self.model = None
        self.feature_names = ['temperature', 'humidity', 'pressure', 'light_level', 'sound_level']
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the enhanced model with error handling"""
        try:
            self.model = load_enhanced_model()
            if self.model is not None:
                print("🎯 Enhanced sensory model initialized successfully")
            else:
                print("⚠️ Enhanced model could not be initialized")
        except Exception as e:
            print(f"❌ Failed to initialize enhanced model: {str(e)}")
    
    def predict_comfort_level(self, sensor_data):
        """Predict comfort level using enhanced model with safety checks"""
        if self.model is None:
            print("⚠️ Enhanced model not available, using fallback")
            return self._fallback_prediction(sensor_data)
        
        try:
            # Extract features in correct order
            features = np.array([[
                sensor_data.get('temperature', 22.0),
                sensor_data.get('humidity', 50.0),
                sensor_data.get('pressure', 1013.0),
                sensor_data.get('light_level', 500.0),
                sensor_data.get('sound_level', 50.0)
            ]])
            
            # Create DataFrame for proper feature naming
            feature_df = pd.DataFrame(features, columns=self.feature_names)
            
            # Handle any NaN values that might still exist
            feature_df = feature_df.fillna(feature_df.mean())
            
            print(f"🔍 Prediction features: {feature_df.values[0]}")
            
            # Make prediction
            prediction = self.model.predict(feature_df)[0]
            confidence = np.max(self.model.predict_proba(feature_df)[0])
            
            print(f"🎯 Enhanced model prediction: {prediction} (confidence: {confidence:.2f})")
            
            return {
                'level': prediction,
                'confidence': confidence,
                'source': 'enhanced_model'
            }
            
        except Exception as e:
            print(f"⚠️ Enhanced model prediction failed: {str(e)}")
            return self._fallback_prediction(sensor_data)
    
    def _fallback_prediction(self, sensor_data):
        """Fallback prediction when enhanced model fails"""
        temp = sensor_data.get('temperature', 22.0)
        
        if temp < 18:
            comfort = "uncomfortable"
        elif temp > 28:
            comfort = "uncomfortable"
        else:
            comfort = "comfortable"
            
        return {
            'level': comfort,
            'confidence': 0.7,
            'source': 'fallback'
        }

# Global instance
enhanced_model = EnhancedSensoryModel()