# %% 

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc 
warnings.filterwarnings('ignore')


df = pd.read_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_pivoted.csv')

# data exploration
print(f"DataFrame 'df' loaded with shape: {df.shape}")
print("-" * 30) 

# name of the columns
print(f"Columns in DataFrame: {df.columns.tolist()}")
print("-" * 30)

# NaN percentage in each column
nan_percentage = df.isna().mean() * 100
print("NaN percentage in each column:")
print(nan_percentage[nan_percentage > 0])
print("-" * 30)

# Drop columns with more than 50% NaN values
threshold = 0.5
columns_to_drop = nan_percentage[nan_percentage > threshold].index
print(f"Dropping columns with more than {threshold * 100}% NaN values: {columns_to_drop.tolist()}")
df.drop(columns=columns_to_drop, inplace=True)

# Print shape after dropping columns
print(f"Shape of DataFrame after dropping columns: {df.shape}")
print("-" * 30)

# Fill NaN values with 0
df.fillna(0, inplace=True)

# Save the cleaned DataFrame to a new CSV file
df.to_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_cleaned.csv', index=False)
print("Cleaned DataFrame saved to 'df_cleaned.csv'")
print("-" * 30)

# Explicitly call garbage collection after cleaning
gc.collect()
print("Garbage collection called after cleaning.")
print("-" * 30)