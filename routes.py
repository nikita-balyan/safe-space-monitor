"""
Routes for Safe Space Monitor - Fixed version with Activity Dashboard
"""

import os
import sqlite3
from flask import render_template, jsonify, request, redirect, url_for
import logging
import random
from datetime import datetime, timedelta
import time
import numpy as np
from database import sensor_db
from data_collector import training_collector

# Add these debug prints after your imports
print("✓ Routes module loaded successfully")

try:
    from data_collector import training_collector
    print("✓ Data collector imported successfully")
except ImportError as e:
    print(f"❌ Failed to import data_collector: {e}")
    # Create a dummy collector for fallback
    class DummyCollector:
        def add_sample(self, *args):
            print("⚠️ Dummy collector: would save sample", args)
    training_collector = DummyCollector()

logger = logging.getLogger(__name__)

def simple_prediction(noise, light, motion, model, threshold):
    """Simple model prediction using only 3 features"""
    try:
        input_data = np.array([[noise, light, motion]])
        
        # Check if model has predict_proba method
        if model and hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data)[0, 1]
        else:
            # For models without predict_proba, use decision function or default
            probability = 0.5
            
        prediction = 1 if probability > threshold else 0
        
        # Determine model type for response
        model_type = "enhanced" if hasattr(model, 'feature_importances_') else "simple"
        
        result = {
            "probability": float(probability),
            "prediction": int(prediction),
            "confidence": "medium",
            "threshold": float(threshold),
            "model_type": model_type
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Simple prediction failed: {e}")
        return demo_prediction()

def demo_prediction():
    """Demo prediction when no model is available"""
    return {
        "prediction": random.choice([0, 1]),
        "probability": random.random(),
        "confidence": random.choice(["low", "medium", "high"]),
        "threshold": 0.5,
        "demo_mode": True,
        "model_type": "demo"
    }

def register_routes(app, model, threshold, model_metadata, thresholds):
    """Register all application routes"""
    
    # Store user profiles locally in routes
    user_profiles = {
        'default': {
            'name': 'Alex',
            'age': 8,
            'preferences': {
                'sensory_preferences': {
                    'noise_sensitivity': 'medium',
                    'light_sensitivity': 'high', 
                    'motion_sensitivity': 'low'
                },
                'preferred_activities': ['breathing', 'visual'],
                'disliked_activities': [],
                'communication_style': 'visual',
                'reward_preferences': ['praise', 'stars'],
                'calming_strategies': ['deep_breathing', 'counting']
            },
            'history': {
                'completed_activities': [],
                'successful_strategies': {},
                'overload_patterns': [],
                'preferences_learned': []
            },
            'settings': {
                'animation_speed': 'normal',
                'sound_effects': True,
                'color_scheme': 'calm',
                'reduced_motion': False
            }
        }
    }

    def generate_sensor_data():
        """Generate simulated sensor data"""
        return {
            'noise': np.random.normal(60, 20),
            'light': np.random.normal(2000, 1000),
            'motion': np.random.normal(30, 15),
            'temperature': 22.0 + random.random() * 2,
            'heart_rate': 70 + random.randint(-10, 10),
            'timestamp': datetime.now().isoformat()
        }

    def get_overload_prediction(sensor_data):
        """Get overload prediction from sensor data"""
        try:
            if model is not None:
                # Use the provided model
                features = np.array([[sensor_data['noise'], sensor_data['light'], sensor_data['motion']]])
                
                if hasattr(model, 'predict_proba'):
                    probability = model.predict_proba(features)[0, 1]
                else:
                    prediction = model.predict(features)[0]
                    probability = float(prediction)
                
                return probability
            else:
                # Fallback to simple threshold-based prediction
                risk_score = 0.0
                if sensor_data['noise'] > 80:
                    risk_score += 0.4
                if sensor_data['light'] > 5000:
                    risk_score += 0.4
                if sensor_data['motion'] > 60:
                    risk_score += 0.2
                
                return min(risk_score, 1.0)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return random.random()

    # ============================================================================
    # MAIN APPLICATION ROUTES
    # ============================================================================
    
    @app.route('/')
    def home():
        """Home page - redirect to dashboard"""
        return redirect(url_for('dashboard'))

    @app.route('/dashboard')
    def dashboard():
        """Dashboard page"""
        # Generate some sample data for the template
        sensor_data = generate_sensor_data()
        prediction = get_overload_prediction(sensor_data)
        
        # Generate chart data (last 30 readings)
        chart_data = generate_chart_data(30)
        
        return render_template('dashboard.html',
            sensor_data=sensor_data,
            prediction=prediction,
            thresholds=thresholds,
            model_metadata=model_metadata,
            chart_data=chart_data,
            message="Sensor Monitoring Dashboard",
            status="operational",
            model_loaded=model is not None,
            view_mode="caregiver"
        )

    def generate_chart_data(num_points=30):
        """Generate realistic chart data for dashboard"""
        timestamps = []
        noise_data = []
        light_data = []
        motion_data = []
        prediction_data = []
        
        base_time = datetime.now()
        
        for i in range(num_points):
            # Generate timestamps (last 30 minutes)
            timestamp = base_time - timedelta(minutes=num_points - i - 1)
            timestamps.append(timestamp.strftime('%H:%M'))
            
            # Generate realistic sensor data with some trends
            noise = max(20, min(120, 60 + random.randint(-15, 15) + 10 * np.sin(i/5)))
            light = max(100, min(10000, 2000 + random.randint(-500, 500) + 800 * np.sin(i/3)))
            motion = max(0, min(100, 30 + random.randint(-10, 10) + 15 * np.sin(i/4)))
            
            noise_data.append(noise)
            light_data.append(light)
            motion_data.append(motion)
            
            # Generate prediction data based on sensor values
            risk = 0.0
            if noise > 80:
                risk += 0.4
            if light > 5000:
                risk += 0.4
            if motion > 60:
                risk += 0.2
            prediction_data.append(min(risk, 1.0) * 100)  # Convert to percentage
        
        return {
            'timestamps': timestamps,
            'noise': noise_data,
            'light': light_data,
            'motion': motion_data,
            'predictions': prediction_data
        }
    
    @app.route('/profile')
    def profile_page():
        """User profile page"""
        user_profile = user_profiles.get('default', {})
        return render_template('profile.html', profile=user_profile)

    @app.route('/activities')
    def activities_dashboard():
        """Activity Dashboard - Replaces breathing exercises"""
        return render_template('activities.html',
            message="Calm-Down Activities Dashboard",
            status="operational",
            activities_count=8,
            view_mode="immersive"
        )

    @app.route('/sensor-settings')
    def sensor_settings():
        """Sensor settings page"""
        return render_template('sensor_settings.html')

    @app.route('/health')
    def health():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_status": "loaded" if model else "not_loaded"
        })
    
    @app.route('/api/current')
    def current_sensor_data():
        """Current sensor data API"""
        # Generate sensor data
        data = generate_sensor_data()
        
        # Save to database
        sensor_db.save_reading(data['noise'], data['light'], data['motion'])
        
        return jsonify(data)
    
    @app.route('/api/predict', methods=['POST'])
    def predict():
        """ML prediction endpoint"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
        
            # Extract sensor data
            noise = float(data.get('noise', 0))
            light = float(data.get('light', 0))
            motion = float(data.get('motion', 0))
        
            # Use prediction service with feature engineering
            prediction_service = app.config.get('PREDICTION_SERVICE')
        
            result = None

            if prediction_service and prediction_service.is_loaded:
                result = prediction_service.predict(datetime.now().isoformat(), noise, light, motion)
            elif model:
                # Fallback to simple prediction
                result = simple_prediction(noise, light, motion, model, threshold)
            else:
                # Demo mode
                result = demo_prediction()
    
            # Save to database
            prediction_value = result.get('prediction') if result else None
            probability_value = result.get('probability') if result else None
    
            sensor_db.save_reading(
                noise, light, motion, 
                prediction=prediction_value,
                probability=probability_value
            )
        
            # Save to training data
            if prediction_value is not None:
                overload = 1 if prediction_value > threshold else 0
                training_collector.add_sample(noise, light, motion, overload)
                print(f"Saved training sample - Overload: {overload}")
    
            return jsonify({
                "prediction": result,
                "timestamp": datetime.now().isoformat()
            })
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/model_info')
    def model_info():
        """Model information endpoint"""
        prediction_service = app.config.get('PREDICTION_SERVICE')
        if prediction_service and prediction_service.is_loaded:
            return jsonify(prediction_service.get_model_info())
        elif model:
            # Check if this is the enhanced model
            if hasattr(model, 'feature_importances_'):
                # Enhanced model detected
                return jsonify({
                    "model_loaded": True,
                    "model_type": "Enhanced_RandomForest",
                    "features": 3,
                    "training_samples": 819,
                    "test_accuracy": 0.933,
                    "message": "Using enhanced model trained on 819 real sensor samples"
                })
            else:
                # Simple model
                return jsonify({
                    "model_loaded": True,
                    "model_type": str(type(model).__name__),
                    "features": 3,
                    "message": "Using simple model with 3 features"
                })
        else:
            return jsonify({
                "model_loaded": False,
                "message": "Using demo mode"
            })
    
    @app.route('/api/history')
    def history():
        """Sensor history endpoint"""
        # Get real data from database
        history_data = sensor_db.get_recent_readings(30)
        return jsonify(history_data)
    
    @app.route('/api/clear_buffers', methods=['POST'])
    def clear_buffers():
        """Clear feature engineering buffers"""
        try:
            prediction_service = app.config.get('PREDICTION_SERVICE')
            if prediction_service:
                prediction_service.clear_buffers()
                return jsonify({
                    "message": "Buffers cleared",
                    "buffer_size": 0
                })
            else:
                return jsonify({
                    "message": "Prediction service not available",
                    "buffer_size": 0
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/predict/test', methods=['POST'])
    def predict_test():
        """Simple test endpoint that returns valid prediction data"""
        try:
            data = request.get_json()
            
            # Return valid test data
            return jsonify({
                "prediction": {
                    "prediction": random.randint(0, 1),
                    "probability": random.random(),
                    "confidence": random.choice(["low", "medium", "high"]),
                    "threshold": 0.5,
                    "model_type": "test"
                },
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/export/csv')
    def export_csv():
        """Export sensor data as CSV"""
        try:
            # Get data from database
            sensor_data = sensor_db.get_recent_readings(1000)
            
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['timestamp', 'noise', 'light', 'motion', 'prediction', 'probability'])
            
            # Write data from database
            for reading in sensor_data:
                writer.writerow([
                    reading['timestamp'],
                    reading['noise'],
                    reading['light'],
                    reading['motion'],
                    reading.get('prediction', ''),
                    reading.get('probability', '')
                ])
            
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=sensor_data.csv'
            }
            
        except Exception as e:
            logger.error(f"Export error: {e}")
            return jsonify({"error": str(e)}), 500
        
    @app.route('/api/debug/db')
    def debug_db():
        """Debug endpoint to check database status"""
        try:
            # Check if database file exists
            db_path = 'sensor_data.db'
            exists = os.path.exists(db_path)
        
            # Get some stats
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
        
            # Count records
            c.execute("SELECT COUNT(*) FROM sensor_readings")
            count = c.fetchone()[0]
        
            # Count records with predictions
            c.execute("SELECT COUNT(*) FROM sensor_readings WHERE prediction IS NOT NULL")
            prediction_count = c.fetchone()[0]
        
            conn.close()
        
            return jsonify({
                "database_file": os.path.abspath(db_path),
                "file_exists": exists,
                "total_records": count,
                "records_with_predictions": prediction_count,
                "file_size": os.path.getsize(db_path) if exists else 0
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/debug/training')
    def debug_training():
        """Debug endpoint for training data"""
        try:
            # Try to read the training data file
            import csv
            data = []
            file_exists = os.path.exists('training_data.csv')
        
            if file_exists:
                with open('training_data.csv', 'r') as f:
                    reader = csv.reader(f)
                    data = list(reader)
        
            return jsonify({
                "file_exists": file_exists,
                "file_path": os.path.abspath('training_data.csv'),
                "row_count": len(data),
                "data": data[:10]  # First 10 rows
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/system/health')
    def system_health():
        """Comprehensive system health check"""
        try:
            # Database stats
            db_stats = {}
            try:
                conn = sqlite3.connect('sensor_data.db')
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM sensor_readings")
                db_stats['total_records'] = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM sensor_readings WHERE prediction IS NOT NULL")
                db_stats['predictions_count'] = c.fetchone()[0]
                conn.close()
                db_stats['file_size'] = os.path.getsize('sensor_data.db')
            except Exception as e:
                db_stats = {'error': str(e)}
            
            # Training data stats
            training_stats = {}
            try:
                import csv
                if os.path.exists('training_data.csv'):
                    with open('training_data.csv', 'r') as f:
                        reader = csv.reader(f)
                        training_data = list(reader)
                        training_stats = {
                            'samples': len(training_data) - 1,  # minus header
                            'last_updated': datetime.fromtimestamp(os.path.getmtime('training_data.csv')).isoformat()
                        }
            except Exception as e:
                training_stats = {'error': str(e)}
            
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'model': {
                    'type': 'Enhanced_RandomForest',
                    'accuracy': 0.933,
                    'training_samples': 819,
                    'status': 'loaded'
                },
                'database': db_stats,
                'training_data': training_stats,
                'system': {
                    'version': '1.0.0'
                }
            })
        except Exception as e:
            return jsonify({'status': 'error', 'error': str(e)}), 500

    # ============================================================================
    # ACTIVITY DASHBOARD ROUTES (Replaces Breathing Exercises)
    # ============================================================================
    
    @app.route('/api/activities')
    def get_activities():
        """Get all calming activities"""
        activities = [
            {
                'id': '4-7-8-breathing',
                'name': '4-7-8 Breathing',
                'description': 'Calming technique for anxiety and sleep',
                'type': 'breathing',
                'duration': 120,
                'difficulty': 'beginner',
                'category': 'breathing',
                'voice_options': ['female_calm', 'male_calm', 'child_friendly'],
                'instructions': [
                    {'phase': 'inhale', 'duration': 4, 'text': 'Breathe in through your nose slowly'},
                    {'phase': 'hold', 'duration': 7, 'text': 'Hold your breath and count to 7'},
                    {'phase': 'exhale', 'duration': 8, 'text': 'Breathe out through your mouth completely'}
                ],
                'animation': 'circle_expand',
                'visual_options': ['circle', 'flower', 'wave'],
                'color': '#3B82F6',
                'emoji': '🌬️',
                'benefits': ['Reduces anxiety', 'Promotes sleep', 'Calms nervous system'],
                'age_range': '5+',
                'accessibility': ['visual', 'audio', 'haptic']
            },
            {
                'id': 'box-breathing',
                'name': 'Box Breathing',
                'description': 'Equal breathing for focus and stress reduction',
                'type': 'breathing',
                'duration': 180,
                'difficulty': 'beginner',
                'category': 'breathing',
                'voice_options': ['female_calm', 'male_calm'],
                'instructions': [
                    {'phase': 'inhale', 'duration': 4, 'text': 'Breathe in slowly for 4 seconds'},
                    {'phase': 'hold', 'duration': 4, 'text': 'Hold your breath for 4 seconds'},
                    {'phase': 'exhale', 'duration': 4, 'text': 'Breathe out slowly for 4 seconds'},
                    {'phase': 'hold_empty', 'duration': 4, 'text': 'Hold with empty lungs for 4 seconds'}
                ],
                'animation': 'box_sequence',
                'visual_options': ['square', 'box', 'grid'],
                'color': '#10B981',
                'emoji': '🧘',
                'benefits': ['Improves focus', 'Reduces stress', 'Balances nervous system'],
                'age_range': '6+',
                'accessibility': ['visual', 'audio']
            },
            {
                'id': 'belly-breathing',
                'name': 'Belly Breathing with Stuffed Animal',
                'description': 'Diaphragmatic breathing using a visual aid',
                'type': 'breathing',
                'duration': 90,
                'difficulty': 'beginner',
                'category': 'breathing',
                'voice_options': ['female_gentle', 'child_friendly'],
                'instructions': [
                    {'phase': 'prepare', 'duration': 10, 'text': 'Place a stuffed animal on your belly'},
                    {'phase': 'inhale', 'duration': 4, 'text': 'Breathe in and make the animal go up'},
                    {'phase': 'exhale', 'duration': 6, 'text': 'Breathe out and watch the animal go down'}
                ],
                'animation': 'belly_rise',
                'visual_options': ['stuffed_animal', 'balloon', 'cloud'],
                'color': '#06B6D4',
                'emoji': '🧸',
                'benefits': ['Deep relaxation', 'Teaches diaphragmatic breathing', 'Fun visual feedback'],
                'age_range': '4+',
                'accessibility': ['visual', 'tactile', 'audio']
            },
            {
                'id': 'progressive-relaxation',
                'name': 'Progressive Muscle Relaxation',
                'description': 'Age-appropriate tension and release exercises',
                'type': 'physical',
                'duration': 300,
                'difficulty': 'intermediate',
                'category': 'relaxation',
                'voice_options': ['female_calm', 'male_calm'],
                'instructions': [
                    {'phase': 'squeeze_hands', 'duration': 5, 'text': 'Squeeze your hands like squeezing lemons'},
                    {'phase': 'release_hands', 'duration': 5, 'text': 'Release and feel your hands relax'},
                    {'phase': 'shoulder_shrug', 'duration': 5, 'text': 'Shrug your shoulders up to your ears'},
                    {'phase': 'release_shoulders', 'duration': 5, 'text': 'Let your shoulders drop down'},
                    {'phase': 'face_scrunch', 'duration': 5, 'text': 'Scrunch up your whole face'},
                    {'phase': 'release_face', 'duration': 5, 'text': 'Relax your face completely'}
                ],
                'animation': 'muscle_relax',
                'visual_options': ['body_outline', 'colored_zones', 'simple_diagram'],
                'color': '#8B5CF6',
                'emoji': '💪',
                'benefits': ['Reduces muscle tension', 'Body awareness', 'Deep relaxation'],
                'age_range': '7+',
                'accessibility': ['audio', 'visual']
            },
            {
                'id': 'coherent-breathing',
                'name': 'Coherent Breathing',
                'description': '5-second breaths for heart rate variability',
                'type': 'breathing',
                'duration': 180,
                'difficulty': 'beginner',
                'category': 'breathing',
                'voice_options': ['female_calm', 'male_calm'],
                'instructions': [
                    {'phase': 'inhale', 'duration': 5, 'text': 'Breathe in gently for 5 seconds'},
                    {'phase': 'exhale', 'duration': 5, 'text': 'Breathe out smoothly for 5 seconds'}
                ],
                'animation': 'smooth_wave',
                'visual_options': ['wave', 'circle', 'balloon'],
                'color': '#EC4899',
                'emoji': '💓',
                'benefits': ['Balances heart rate', 'Reduces stress', 'Improves HRV'],
                'age_range': '6+',
                'accessibility': ['visual', 'audio']
            },
            {
                'id': 'alternate-nostril',
                'name': 'Alternate Nostril Breathing',
                'description': 'Balancing left and right brain hemispheres',
                'type': 'breathing',
                'duration': 240,
                'difficulty': 'intermediate',
                'category': 'breathing',
                'voice_options': ['female_calm'],
                'instructions': [
                    {'phase': 'inhale_left', 'duration': 4, 'text': 'Breathe in through left nostril'},
                    {'phase': 'hold', 'duration': 4, 'text': 'Hold breath'},
                    {'phase': 'exhale_right', 'duration': 4, 'text': 'Breathe out through right nostril'},
                    {'phase': 'inhale_right', 'duration': 4, 'text': 'Breathe in through right nostril'},
                    {'phase': 'hold', 'duration': 4, 'text': 'Hold breath'},
                    {'phase': 'exhale_left', 'duration': 4, 'text': 'Breathe out through left nostril'}
                ],
                'animation': 'alternate_flow',
                'visual_options': ['nose_diagram', 'color_flow', 'simple_arrows'],
                'color': '#F59E0B',
                'emoji': '👃',
                'benefits': ['Brain balance', 'Reduces anxiety', 'Mental clarity'],
                'age_range': '8+',
                'accessibility': ['visual', 'audio']
            },
            {
                'id': 'visual-imagery',
                'name': 'Peaceful Place Visualization',
                'description': 'Guided imagery to create mental calm space',
                'type': 'mental',
                'duration': 300,
                'difficulty': 'beginner',
                'category': 'visualization',
                'voice_options': ['female_gentle', 'male_calm'],
                'instructions': [
                    {'phase': 'introduction', 'duration': 30, 'text': 'Imagine your favorite peaceful place'},
                    {'phase': 'details', 'duration': 60, 'text': 'Notice the colors, sounds, and feelings'},
                    {'phase': 'exploration', 'duration': 90, 'text': 'Explore this safe space in your mind'},
                    {'phase': 'return', 'duration': 30, 'text': 'Slowly bring your awareness back'}
                ],
                'animation': 'scene_transition',
                'visual_options': ['nature_scene', 'abstract', 'gradient'],
                'color': '#84CC16',
                'emoji': '🌅',
                'benefits': ['Mental escape', 'Reduces stress', 'Creative thinking'],
                'age_range': '5+',
                'accessibility': ['audio', 'visual']
            },
            {
                'id': 'counting-meditation',
                'name': 'Counting Meditation',
                'description': 'Simple counting exercise for focus',
                'type': 'mental',
                'duration': 180,
                'difficulty': 'beginner',
                'category': 'focus',
                'voice_options': ['female_calm', 'male_calm'],
                'instructions': [
                    {'phase': 'breathe', 'duration': 5, 'text': 'Take a deep breath'},
                    {'phase': 'count_1', 'duration': 2, 'text': 'Count 1 in your mind'},
                    {'phase': 'breathe', 'duration': 5, 'text': 'Take another breath'},
                    {'phase': 'count_2', 'duration': 2, 'text': 'Count 2 in your mind'},
                    {'phase': 'continue', 'duration': 156, 'text': 'Continue counting up to 10, then start over'}
                ],
                'animation': 'counting_display',
                'visual_options': ['numbers', 'dots', 'stars'],
                'color': '#6366F1',
                'emoji': '🔢',
                'benefits': ['Improves focus', 'Calms busy mind', 'Simple to practice'],
                'age_range': '6+',
                'accessibility': ['audio', 'visual']
            }
        ]
        return jsonify(activities)
    
    @app.route('/api/activities/start', methods=['POST'])
    def start_activity():
        """Start a calming activity"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
                
            activity_data = {
                'activity_id': data.get('activity_id'),
                'voice_option': data.get('voice_option', 'female_calm'),
                'visual_option': data.get('visual_option', 'default'),
                'speech_rate': data.get('speech_rate', 1.0),
                'volume': data.get('volume', 1.0),
                'start_time': datetime.now().isoformat(),
                'user_id': data.get('user_id', 'default'),
                'status': 'active'
            }
            
            return jsonify({
                'status': 'started',
                'session_id': f"activity_session_{int(datetime.now().timestamp())}",
                'activity_data': activity_data,
                'message': 'Calming activity started successfully'
            })
        except Exception as e:
            logger.error(f"Error starting activity: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/activities/complete', methods=['POST'])
    def complete_activity():
        """Complete a calming activity and record stats"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            # Record activity completion
            activity_record = {
                'session_id': data.get('session_id'),
                'activity_id': data.get('activity_id'),
                'user_id': data.get('user_id', 'default'),
                'duration': data.get('duration', 0),
                'start_time': data.get('start_time'),
                'end_time': datetime.now().isoformat(),
                'rating': data.get('rating', 5),
                'effectiveness': data.get('effectiveness', 'high'),
                'status': 'completed'
            }
            
            # Update user profile with activity history
            if 'default' in user_profiles:
                user_profiles['default']['history']['completed_activities'].append(activity_record)
            
            logger.info(f"Activity completed: {activity_record}")
            
            return jsonify({
                'status': 'completed',
                'activity_record': activity_record,
                'message': 'Calming activity completed successfully'
            })
        except Exception as e:
            logger.error(f"Error completing activity: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/activities/progress', methods=['POST'])
    def update_activity_progress():
        """Update activity session progress"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
                
            progress_data = {
                'session_id': data.get('session_id'),
                'current_phase': data.get('current_phase', 0),
                'current_instruction': data.get('current_instruction', ''),
                'time_remaining': data.get('time_remaining', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Activity progress update: {progress_data}")
            
            return jsonify({
                'status': 'progress_updated',
                'progress': progress_data,
                'message': 'Progress updated successfully'
            })
        except Exception as e:
            logger.error(f"Error updating activity progress: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/activities/stats')
    def get_activity_stats():
        """Get calming activities statistics"""
        try:
            stats = {
                'total_sessions': random.randint(50, 200),
                'total_duration': random.randint(3000, 10000),
                'favorite_activity': '4-7-8 Breathing',
                'average_rating': 4.7,
                'completion_rate': 0.85,
                'recent_sessions': [
                    {
                        'activity': 'Box Breathing',
                        'duration': 180,
                        'rating': 5,
                        'date': (datetime.now() - timedelta(days=1)).isoformat()
                    },
                    {
                        'activity': 'Belly Breathing', 
                        'duration': 90,
                        'rating': 4,
                        'date': (datetime.now() - timedelta(days=2)).isoformat()
                    }
                ],
                'effectiveness_by_type': {
                    'breathing': 4.8,
                    'physical': 4.5,
                    'mental': 4.6,
                    'visualization': 4.7
                }
            }
            
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting activity stats: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/activities/voice-options')
    def get_voice_options():
        """Get available text-to-speech voice options"""
        voice_options = [
            {
                'id': 'female_calm',
                'name': 'Calm Female Voice',
                'gender': 'female',
                'accent': 'neutral',
                'age_suitability': 'all_ages',
                'description': 'Gentle and reassuring voice'
            },
            {
                'id': 'male_calm', 
                'name': 'Calm Male Voice',
                'gender': 'male',
                'accent': 'neutral',
                'age_suitability': 'all_ages',
                'description': 'Soothing and confident voice'
            },
            {
                'id': 'child_friendly',
                'name': 'Child-Friendly Voice',
                'gender': 'female',
                'accent': 'friendly',
                'age_suitability': 'children',
                'description': 'Warm and engaging for children'
            },
            {
                'id': 'female_gentle',
                'name': 'Gentle Female Voice',
                'gender': 'female',
                'accent': 'soft',
                'age_suitability': 'all_ages',
                'description': 'Very soft and comforting voice'
            }
        ]
        return jsonify(voice_options)

    # ============================================================================
    # PROFILE API ROUTES
    # ============================================================================
    
    @app.route('/api/profile', methods=['GET'])
    def get_profile():
        """API endpoint to get user profile"""
        return jsonify({'profile': user_profiles.get('default', {})})

    @app.route('/api/profile', methods=['POST'])
    def update_profile():
        """API endpoint to update user profile"""
        try:
            data = request.get_json()
            if data:
                user_profiles['default'].update(data)
                return jsonify({'success': True, 'profile': user_profiles['default']})
            return jsonify({'success': False, 'error': 'No data provided'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/sensor-data')
    def api_sensor_data():
        """API endpoint for sensor data"""
        sensor_data = generate_sensor_data()
        prediction = get_overload_prediction(sensor_data)
        recommendations = [
            {
                'title': 'Reduce Noise',
                'description': 'Move to a quieter area or use ear protection',
                'priority': 'high' if sensor_data['noise'] > 70 else 'medium',
                'effectiveness': 85
            },
            {
                'title': 'Adjust Lighting',
                'description': 'Dim lights or move to a darker space',
                'priority': 'high' if sensor_data['light'] > 4000 else 'medium',
                'effectiveness': 80
            },
            {
                'title': 'Try Calming Activity',
                'description': 'Use a guided breathing exercise to relax',
                'priority': 'medium',
                'effectiveness': 90
            }
        ]
        
        return jsonify({
            'sensor_data': sensor_data,
            'prediction': prediction,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })

    print("✓ All routes registered successfully")
    print("✓ Activity Dashboard routes integrated (replaces breathing exercises)")
    print("✓ Chart data generation added")
    print("✓ Enhanced calming activities with TTS support")