#!/usr/bin/env python3
"""
Check data distribution and quality
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/features_data.csv")

print("Data Analysis:")
print("=" * 30)
print(f"Total samples: {len(df)}")
print(f"Normal samples: {len(df[df['label'] == 0])}")
print(f"Overload samples: {len(df[df['label'] == 1])}")
print(f"Overload ratio: {len(df[df['label'] == 1]) / len(df):.2%}")

print("\nSensor statistics:")
print("Noise - Mean:", df['noise'].mean(), "Std:", df['noise'].std())
print("Light - Mean:", df['light'].mean(), "Std:", df['light'].std())  
print("Motion - Mean:", df['motion'].mean(), "Std:", df['motion'].std())

print("\nFirst 3 samples:")
print(df.head(3))
