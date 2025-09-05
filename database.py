import os
import sqlite3
import json
from datetime import datetime

class SensorDatabase:
    def __init__(self, db_path='sensor_data.db'):
        self.db_path = db_path
        print(f"Database path: {os.path.abspath(self.db_path)}")  # ADD THIS LINE
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables
        c.execute('''CREATE TABLE IF NOT EXISTS sensor_readings
                     (id INTEGER PRIMARY KEY, timestamp TEXT, 
                      noise REAL, light REAL, motion REAL,
                      prediction REAL, probability REAL)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS alerts
                     (id INTEGER PRIMARY KEY, timestamp TEXT,
                      sensor_type TEXT, value REAL, severity TEXT,
                      message TEXT)''')
        
        conn.commit()
        conn.close()
    
    def save_reading(self, noise, light, motion, prediction=None, probability=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''INSERT INTO sensor_readings 
                     (timestamp, noise, light, motion, prediction, probability)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                 (datetime.now().isoformat(), noise, light, motion, prediction, probability))
        
        conn.commit()
        conn.close()
    
    def get_recent_readings(self, limit=30):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''SELECT timestamp, noise, light, motion, prediction, probability
                     FROM sensor_readings 
                     ORDER BY timestamp DESC LIMIT ?''', (limit,))
        
        readings = []
        for row in c.fetchall():
            readings.append({
                'timestamp': row[0],
                'noise': row[1],
                'light': row[2],
                'motion': row[3],
                'prediction': row[4],
                'probability': row[5]
            })
        
        conn.close()
        return readings

# Create a global instance
sensor_db = SensorDatabase()