#!/usr/bin/env python3
"""
Real-time feature engineering for sensor data
"""

import numpy as np
from collections import deque
from datetime import datetime

class FeatureEngineer:
    """Handles real-time feature extraction"""
    
    def __init__(self, window_sizes=[10, 30, 60]):
        self.window_sizes = window_sizes
        self.max_window = max(window_sizes)
        
        # Buffers for each sensor
        self.noise_buffer = deque(maxlen=self.max_window)
        self.light_buffer = deque(maxlen=self.max_window)
        self.motion_buffer = deque(maxlen=self.max_window)
        
        # Timestamp buffer
        self.timestamp_buffer = deque(maxlen=self.max_window)
    
    def add_reading(self, timestamp, noise, light, motion):
        """Add a new sensor reading to buffers"""
        self.timestamp_buffer.append(timestamp)
        self.noise_buffer.append(float(noise))
        self.light_buffer.append(float(light))
        self.motion_buffer.append(float(motion))
    
    def has_enough_data(self):
        """Check if we have enough data for feature extraction"""
        return len(self.noise_buffer) >= min(self.window_sizes)
    
    def extract_features(self):
        """Extract rolling features from current buffers"""
        features = {}
        
        # Extract features for each sensor
        for sensor_name, buffer in [
            ('noise', self.noise_buffer),
            ('light', self.light_buffer), 
            ('motion', self.motion_buffer)
        ]:
            values = list(buffer)
            
            for window in self.window_sizes:
                if len(values) >= window:
                    window_values = values[-window:]
                    
                    # Basic statistics
                    features[f'{sensor_name}_mean_{window}'] = np.mean(window_values)
                    features[f'{sensor_name}_std_{window}'] = np.std(window_values)
                    features[f'{sensor_name}_range_{window}'] = max(window_values) - min(window_values)
                    
                    # Slope (rate of change)
                    if len(window_values) > 1:
                        features[f'{sensor_name}_slope_{window}'] = (window_values[-1] - window_values[0]) / window
                    else:
                        features[f'{sensor_name}_slope_{window}'] = 0
                    
                    # FFT energy for the largest window
                    if window == self.max_window:
                        try:
                            fft_values = np.fft.fft(window_values)
                            # Exclude DC component (first element)
                            energy = np.sum(np.abs(fft_values[1:])**2)
                            features[f'{sensor_name}_fft_energy'] = energy
                        except:
                            features[f'{sensor_name}_fft_energy'] = 0
        
        return features
    
    def clear_buffers(self):
        """Clear all buffers"""
        self.noise_buffer.clear()
        self.light_buffer.clear()
        self.motion_buffer.clear()
        self.timestamp_buffer.clear()