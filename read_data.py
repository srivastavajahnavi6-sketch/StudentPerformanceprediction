import pandas as pd

data = pd.read_csv("StudentPerformanceFactors.csv")

print("Dataset Loaded Successfully!")
print(data.head())
print("Dataset Shape:", data.shape)
print("Columns:", data.columns)