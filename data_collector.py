import csv
import os
from datetime import datetime

class DataCollector:
    def __init__(self, filename='training_data.csv'):
        self.filename = filename
        print(f"Data collector initialized. File: {os.path.abspath(self.filename)}")
        self.init_csv()
    
    def init_csv(self):
        # Only create file if it doesn't exist
        try:
            with open(self.filename, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'noise', 'light', 'motion', 'overload'])
                print(f"✓ Created new training data file: {self.filename}")
        except FileExistsError:
            print(f"✓ Using existing training data file: {self.filename}")
        except Exception as e:
            print(f"❌ Error creating training data file: {e}")
    
    def add_sample(self, noise, light, motion, overload):
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), noise, light, motion, overload])
            print(f"✓ Added training sample: noise={noise}, light={light}, motion={motion}, overload={overload}")
            return True
        except Exception as e:
            print(f"❌ Error saving training sample: {e}")
            return False

# Create global instance
training_collector = DataCollector()