"""
Enhanced ML Model for Sensory Overload Prediction
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedSensoryModel:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        
    def generate_training_data(self, n_samples=1000):
        """Generate realistic synthetic training data"""
        np.random.seed(42)
        
        # Base patterns for different overload types
        data = []
        
        for i in range(n_samples):
            # Normal state (60% of data)
            if i < 0.6 * n_samples:
                noise = np.random.normal(50, 15)
                light = np.random.normal(2000, 800)
                motion = np.random.normal(30, 12)
                overload = 0
                
            # Auditory overload (15% of data)
            elif i < 0.75 * n_samples:
                noise = np.random.normal(90, 20)
                light = np.random.normal(1800, 600)
                motion = np.random.normal(25, 10)
                overload = 1
                
            # Visual overload (15% of data)
            elif i < 0.9 * n_samples:
                noise = np.random.normal(55, 12)
                light = np.random.normal(6000, 1500)
                motion = np.random.normal(20, 8)
                overload = 1
                
            # Motion overload (10% of data)
            else:
                noise = np.random.normal(60, 10)
                light = np.random.normal(2200, 700)
                motion = np.random.normal(70, 15)
                overload = 1
            
            # Ensure realistic ranges
            noise = max(20, min(120, noise))
            light = max(100, min(10000, light))
            motion = max(0, min(100, motion))
            
            data.append({
                'noise': noise,
                'light': light,
                'motion': motion,
                'overload': overload,
                'timestamp': datetime.now().isoformat()
            })
        
        return pd.DataFrame(data)
    
    def extract_features(self, df):
        """Extract advanced features from sensor data"""
        features = []
        
        # Basic features
        features.extend(['noise', 'light', 'motion'])
        
        # Statistical features (simulating rolling windows)
        df['noise_rolling_mean'] = df['noise'].rolling(window=5, min_periods=1).mean()
        df['light_rolling_mean'] = df['light'].rolling(window=5, min_periods=1).mean()
        df['motion_rolling_mean'] = df['motion'].rolling(window=5, min_periods=1).mean()
        
        df['noise_rolling_std'] = df['noise'].rolling(window=5, min_periods=1).std()
        df['light_rolling_std'] = df['light'].rolling(window=5, min_periods=1).std()
        df['motion_rolling_std'] = df['motion'].rolling(window=5, min_periods=1).std()
        
        features.extend([
            'noise_rolling_mean', 'light_rolling_mean', 'motion_rolling_mean',
            'noise_rolling_std', 'light_rolling_std', 'motion_rolling_std'
        ])
        
        # Rate of change features
        df['noise_roc'] = df['noise'].diff().fillna(0)
        df['light_roc'] = df['light'].diff().fillna(0)
        df['motion_roc'] = df['motion'].diff().fillna(0)
        
        features.extend(['noise_roc', 'light_roc', 'motion_roc'])
        
        # Interaction features
        df['noise_light_interaction'] = df['noise'] * df['light'] / 1000
        df['noise_motion_interaction'] = df['noise'] * df['motion'] / 100
        df['light_motion_interaction'] = df['light'] * df['motion'] / 1000
        
        features.extend([
            'noise_light_interaction', 
            'noise_motion_interaction', 
            'light_motion_interaction'
        ])
        
        # Threshold crossing features
        df['noise_above_70'] = (df['noise'] > 70).astype(int)
        df['light_above_3000'] = (df['light'] > 3000).astype(int)
        df['motion_above_50'] = (df['motion'] > 50).astype(int)
        
        features.extend(['noise_above_70', 'light_above_3000', 'motion_above_50'])
        
        self.feature_names = features
        return df[features]
    
    def train(self, n_samples=1000):
        """Train the enhanced model"""
        print("🔄 Generating training data...")
        df = self.generate_training_data(n_samples)
        
        print("🔧 Extracting features...")
        X = self.extract_features(df)
        y = df['overload']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("🤖 Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, zero_division=0)
        self.recall = recall_score(y_test, y_pred, zero_division=0)
        
        print(f"✅ Model trained successfully!")
        print(f"📊 Accuracy: {self.accuracy:.3f}")
        print(f"🎯 Precision: {self.precision:.3f}")
        print(f"📈 Recall: {self.recall:.3f}")
        
        return self.model
    
    def predict(self, sensor_data):
        """Make prediction on new sensor data"""
        if self.model is None:
            # Fallback to simple threshold-based prediction
            return self._fallback_prediction(sensor_data)
        
        try:
            # Create feature vector
            features = self._create_feature_vector(sensor_data)
            
            # Get prediction probability
            probability = self.model.predict_proba([features])[0, 1]
            
            return {
                'probability': float(probability),
                'prediction': int(probability > 0.5),
                'confidence': float(probability if probability > 0.5 else 1 - probability),
                'model_used': 'enhanced_random_forest',
                'features_used': len(features)
            }
            
        except Exception as e:
            print(f"⚠️ Enhanced model prediction failed: {e}")
            return self._fallback_prediction(sensor_data)
    
    def _create_feature_vector(self, sensor_data):
        """Create feature vector from sensor data"""
        # This is a simplified version - in production you'd maintain state
        # for rolling windows and previous values
        
        noise = sensor_data['noise']
        light = sensor_data['light']
        motion = sensor_data['motion']
        
        # Basic features
        features = [noise, light, motion]
        
        # Statistical features (simplified)
        features.extend([noise, light, motion])  # Using current values as mean
        features.extend([5, 300, 5])  # Simplified std dev
        
        # Rate of change (simplified)
        features.extend([0, 0, 0])  # Assuming no previous data
        
        # Interaction features
        features.extend([
            noise * light / 1000,
            noise * motion / 100,
            light * motion / 1000
        ])
        
        # Threshold features
        features.extend([
            int(noise > 70),
            int(light > 3000),
            int(motion > 50)
        ])
        
        return features
    
    def _fallback_prediction(self, sensor_data):
        """Fallback prediction using simple thresholds"""
        noise = sensor_data['noise']
        light = sensor_data['light']
        motion = sensor_data['motion']
        
        # Simple weighted threshold model
        risk_score = 0.0
        
        if noise > 80:
            risk_score += 0.4
        elif noise > 70:
            risk_score += 0.2
            
        if light > 5000:
            risk_score += 0.4
        elif light > 3000:
            risk_score += 0.2
            
        if motion > 60:
            risk_score += 0.2
        elif motion > 50:
            risk_score += 0.1
            
        probability = min(risk_score, 1.0)
        
        return {
            'probability': probability,
            'prediction': int(probability > 0.5),
            'confidence': 0.7,
            'model_used': 'fallback_threshold',
            'features_used': 3
        }
    
    def save_model(self, filepath='models/enhanced_sensory_model.joblib'):
        """Save trained model to file"""
        if self.model is None:
            print("❌ No model to save. Train the model first.")
            return False
        
        import os
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
        return True
    
    def load_model(self, filepath='models/enhanced_sensory_model.joblib'):
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.accuracy = model_data['accuracy']
            self.precision = model_data['precision']
            self.recall = model_data['recall']
            
            print(f"✅ Model loaded from {filepath}")
            print(f"📊 Previous performance - Accuracy: {self.accuracy:.3f}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

# Global instance
enhanced_model = EnhancedSensoryModel()

def initialize_enhanced_model():
    """Initialize and train the enhanced model"""
    print("🚀 Initializing Enhanced Sensory Model...")
    
    # Try to load existing model first
    if enhanced_model.load_model():
        return enhanced_model
    
    # Train new model
    enhanced_model.train(n_samples=1000)
    enhanced_model.save_model()
    
    return enhanced_model

if __name__ == "__main__":
    # Train and test the model
    model = initialize_enhanced_model()
    
    # Test prediction
    test_data = {'noise': 85.5, 'light': 3200.2, 'motion': 45.1}
    prediction = model.predict(test_data)
    print(f"🧪 Test Prediction: {prediction}")