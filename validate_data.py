import pandas as pd
import numpy as np

def validate_training_data():
    """Check and clean training data"""
    try:
        df = pd.read_csv("training_data.csv")
        print("📊 Dataset Info:")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print("\n📋 Columns:", df.columns.tolist())
        print("\n🔍 Missing Values:")
        print(df.isnull().sum())
        print("\n📈 Basic Stats:")
        print(df.describe())
        
        # Fix any missing values
        df_clean = df.fillna(df.mean())
        df_clean.to_csv("training_data.csv", index=False)
        print("✅ Data validated and cleaned")
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")

if __name__ == "__main__":
    validate_training_data()