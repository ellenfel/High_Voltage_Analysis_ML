# %% 

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc 
warnings.filterwarnings('ignore')

# reading data
df = pd.read_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_pivoted.csv')

# data exploration
print(f"DataFrame 'df' loaded with shape: {df.shape}")
original_shape = df.shape
print("-" * 30)

# name of the columns
print(f"Columns in DataFrame: {df.columns.tolist()}")
print("-" * 30)

# filter out device_name 'I-Link Box 1, 2, 3,' and 'I-Link Box R&D Test'
device_names_to_exclude = ['I-Link Box 1', 'I-Link Box 2', 'I-Link Box 3', 'I-Link Box R&D Test', 'I-Link Box Tester', 'Signal Tester']
df = df[~df['device_name'].isin(device_names_to_exclude)]

# Print shape after filtering device names
device_filtered_shape = df.shape
print(f"Shape of DataFrame after filtering device names: {device_filtered_shape}")
print("-" * 30)

# Keep only the listed columns
columns_to_keep = [
    "time",
    "device_profile",
    "device_name",
    "IR_A",
    "IR_B",
    "IR_C",
    "Water_Detection_Outside",
    "current_a",
    "current_b",
    "current_c",
    "harvesting_battery_power",
    "harvesting_battery_voltage",
    "hum",
    "image_name",
    "images_urls",
    "ipec_pddata_avg_pd_a",
    "ipec_pddata_avg_pd_b",
    "ipec_pddata_avg_pd_c",
    "lid_switch",
    "mppt_batary_voltage",
    "mppt_panel_power",
    "pressure",
    "pulse_a",
    "pulse_b",
    "pulse_c",
    "temp",
    "thermocouple_a",
    "thermocouple_b",
    "thermocouple_c",
    "voltage_a",
    "voltage_b",
    "voltage_c",
    "water_detector",
    "water_image_name",
    "rssi",
    "snr",
]
df = df[columns_to_keep]
# Print shape after keeping specific columns
filtered_columns_shape = df.shape
print(f"Shape of DataFrame after keeping specific columns: {filtered_columns_shape}")
print("-" * 30)

 # List of all the columns in the DataFrame
print(f"Columns in DataFrame after filtering: {df.columns.tolist()}")
print("-" * 30)

# NaN percentage in each column
nan_percentage = df.isna().mean() * 100
print("NaN percentage in each column:")
print(nan_percentage[nan_percentage > 0])
print("-" * 30)


# Fill NaN values with 0
df.fillna(0, inplace=True)

# Save the cleaned DataFrame to a new CSV file
df.to_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_cleaned.csv', index=False)
print("Cleaned DataFrame saved to 'df_cleaned.csv'")
print("-" * 30)

# NaN percentage in each column
nan_percentage = df.isna().mean() * 100
print("NaN percentage in each column:")
print(nan_percentage[nan_percentage > 0])
print("-" * 30)

# Drop columns with more than 90% NaN values
threshold = 0.9
columns_to_drop = nan_percentage[nan_percentage > threshold].index
print(f"Dropping columns with more than {threshold * 100}% NaN values: {columns_to_drop.tolist()}")
df.drop(columns=columns_to_drop, inplace=True)

# Explicitly call garbage collection after cleaning
gc.collect()
print("Garbage collection called after cleaning.")
print("-" * 30)