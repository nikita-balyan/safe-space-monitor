# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import os

def train_enhanced_model():
    """Train an enhanced model using collected training data with NaN handling"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Load your collected data
        df = pd.read_csv('training_data.csv')
        print(f"📊 Loaded {len(df)} training samples")
        print(f"🔍 Columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_features = ['noise', 'light', 'motion']
        required_target = 'overload'
        
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            print(f"❌ Missing required features: {missing_features}")
            return None, None
            
        if required_target not in df.columns:
            print(f"❌ Missing target column: {required_target}")
            return None, None
        
        # Prepare features and labels
        X = df[required_features]
        y = df[required_target]
        
        # Data quality check
        print(f"🔍 Data quality check:")
        print(f"   - NaN values in features: {X.isna().sum().sum()}")
        print(f"   - NaN values in target: {y.isna().sum()}")
        print(f"   - Feature shapes: {X.shape}")
        print(f"   - Target distribution:\n{y.value_counts()}")
        
        # Handle NaN values - fill with column means for features, mode for target
        X = X.fillna(X.mean())
        y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
        
        # Check if we have enough data after cleaning
        if len(X) < 10:
            print("❌ Not enough training data after cleaning")
            return None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline with imputer and classifier for robust NaN handling
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),  # Handles any NaNs during prediction
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ))
        ])
        
        # Train model
        print("🔄 Training Random Forest model with pipeline...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"✅ Training accuracy: {train_score:.3f}")
        print(f"✅ Testing accuracy: {test_score:.3f}")
        
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
            "model_type": "RandomForest_with_Imputer",
            "pipeline_steps": list(model.named_steps.keys())
        }
        
        print(f"💾 Model saved to {model_path}")
        print(f"📝 Metadata: {metadata}")
        
        return model, metadata
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        print(f"🔍 Debug info: {traceback.format_exc()}")
        return None, None

def load_enhanced_model():
    """Safely load the enhanced model with proper validation"""
    try:
        model_path = 'models/enhanced_overload_model.joblib'
        
        if not os.path.exists(model_path):
            print("⚠️ No pre-trained model found, will train new one when needed")
            return None
        
        model = joblib.load(model_path)
        
        # Validate that the model is properly trained
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            if hasattr(classifier, 'classes_'):
                print(f"✅ Loaded trained model with {len(classifier.classes_)} classes")
                return model
            else:
                print("⚠️ Model file exists but classifier is not trained")
                return None
        else:
            print("⚠️ Model file exists but pipeline structure is invalid")
            return None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_with_enhanced_model(model, features):
    """Safe prediction with the enhanced model"""
    if model is None:
        print("⚠️ Model not available for prediction")
        return None, 0.0
    
    try:
        # Convert features to DataFrame with correct column names
        feature_df = pd.DataFrame([features], columns=['noise', 'light', 'motion'])
        
        # Ensure no NaN values
        feature_df = feature_df.fillna(feature_df.mean())
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(feature_df)[0]
        confidence = np.max(probabilities)
        
        print(f"🎯 Model prediction: {prediction} (confidence: {confidence:.3f})")
        
        return prediction, confidence
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, 0.0

def validate_training_data():
    """Validate and clean training data"""
    try:
        if not os.path.exists('training_data.csv'):
            print("❌ training_data.csv not found!")
            return False
            
        df = pd.read_csv('training_data.csv')
        print("📊 Training Data Validation Report:")
        print(f"   - Total rows: {len(df)}")
        print(f"   - Columns: {df.columns.tolist()}")
        print(f"   - NaN values per column:")
        for col in df.columns:
            nan_count = df[col].isna().sum()
            print(f"     {col}: {nan_count} NaN(s)")
        
        # Check for required columns
        required_cols = ['noise', 'light', 'motion', 'overload']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Clean the data
        df_clean = df.copy()
        
        # Fill NaN values
        for col in ['noise', 'light', 'motion']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        if 'overload' in df_clean.columns:
            df_clean['overload'] = df_clean['overload'].fillna(
                df_clean['overload'].mode()[0] if len(df_clean['overload'].mode()) > 0 else 0
            )
        
        # Save cleaned data
        df_clean.to_csv('training_data.csv', index=False)
        print("✅ Data validated and cleaned successfully")
        return True
        
    except Exception as e:
        print(f"❌ Data validation error: {e}")
        return False

# Initialize model on import
enhanced_model = load_enhanced_model()

if __name__ == "__main__":
    print("🚀 Starting enhanced model training process...")
    
    # First validate the data
    if validate_training_data():
        # Then train the model
        model, metadata = train_enhanced_model()
        if model is not None:
            print("🎉 Model training completed successfully!")
            
            # Test the model with sample data
            print("🧪 Testing model with sample data...")
            test_features = {
                'noise': 65.5,
                'light': 300.0,
                'motion': 0.8
            }
            prediction, confidence = predict_with_enhanced_model(model, test_features)
            print(f"🧪 Test prediction: {prediction}, confidence: {confidence:.3f}")
        else:
            print("❌ Model training failed")
    else:
        print("❌ Data validation failed - cannot train model")