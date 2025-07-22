# %% 

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc 
warnings.filterwarnings('ignore')



### SIMPLE DATA EXPLORATION ###

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
    "IR_A_value",
    "IR_B_value",
    "IR_C_value",
    "Water_Detection_Outside_value",
    "current_a_value",
    "current_b_value",
    "current_c_value",
    "harvesting_battery_power_value",
    "harvesting_battery_voltage_value",
    "hum_value",
    "image_name_value",
    "ipec_pddata_avg_pd_a_value",
    "ipec_pddata_avg_pd_b_value",
    "ipec_pddata_avg_pd_c_value",
    "lid_switch_value",
    "mppt_batary_voltage_value",
    "mppt_panel_power_value",
    "pressure_value",
    "pulse_a_value",
    "pulse_b_value",
    "pulse_c_value",
    "temp_value",
    "thermocouple_a_value",
    "thermocouple_b_value",
    "thermocouple_c_value",
    "voltage_a_value",
    "voltage_b_value",
    "voltage_c_value",
    "water_detector_value",
    "water_image_name_value",
    # "rssi", "snr" are not in the DataFrame, so they are excluded.
]

df = df[columns_to_keep]
# Print shape after keeping specific columns
filtered_columns_shape = df.shape
print(f"Shape of DataFrame after keeping specific columns: {filtered_columns_shape}")
print("-" * 30)

 # List of all the columns in the DataFrame
print(f"Columns in DataFrame after filtering: {df.columns.tolist()}")
print("-" * 30)

# Drop all the rows with NaN values
df.dropna(inplace=True)
# Print shape after dropping rows with NaN values
print(f"Shape of DataFrame after dropping rows with NaN values: {df.shape}")
print("-" * 30)

# Save the cleaned DataFrame to a new CSV file
df.to_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_cleaned.csv', index=False)
print("Cleaned DataFrame saved to 'df_cleaned.csv'")
print("-" * 30)





### DEALING WITH NaN VALUES ###

# # NaN percentage in each column
# nan_percentage = df.isna().mean() * 100
# print("NaN percentage in each column:")
# print(nan_percentage[nan_percentage > 0])
# print("-" * 30)

# # How many Rows have more than 50% NaN values
# nan_threshold = 0.9
# rows_with_nan = df[df.isna().mean(axis=1) > nan_threshold]
# print(f"rows with more than {nan_threshold * 100}% NaN values: {rows_with_nan.shape[0]}")
# print("-" * 30)

# # Total number of rows in the DataFrame
# total_rows = df.shape[0]
# print(f"number of rows in the DataFrame: {total_rows}")
# print("-" * 30)

# # Percentage of rows with more than 50% NaN values
# percentage_rows_with_nan = (rows_with_nan.shape[0] / total_rows) * 100
# print(f"Percentage of rows with more than {nan_threshold * 100}% NaN values: {percentage_rows_with_nan:.2f}%")
# print("-" * 30) 

# # 0.7 = %6.97
# # 0.6 = %38.16
# # 0.5 = %39.53
# # 0.4 = %39.54

# # How many Rows have more than a specific number of NaN values
# nan_count_threshold = 3
# nan_counts_per_row = df.isna().sum(axis=1)
# rows_with_many_nans = df[nan_counts_per_row >= nan_count_threshold]

# print(f"Number of rows with {nan_count_threshold} or more NaN values: {rows_with_many_nans.shape[0]}")
# print("-" * 30)

# # Optional: Percentage of rows with more than the specified number of NaN values
# percentage_rows_with_many_nans = (rows_with_many_nans.shape[0] / total_rows) * 100
# print(f"Percentage of rows with {nan_count_threshold} or more NaN values: {percentage_rows_with_many_nans:.2f}%")
# print("-" * 30)

# # Drop rows with more than a specific number of NaN values
# df_cleaned = df[nan_counts_per_row < nan_count_threshold].copy()
# rows_removed = total_rows - df_cleaned.shape[0]
# print(f"Rows removed : {rows_removed}")

# # Number of NaN values in each column after dropping rows
# nan_counts_after_dropping = df_cleaned.isna().sum()
# print("Number of NaN values in each column after dropping rows:")
# print(nan_counts_after_dropping[nan_counts_after_dropping > 0])
# print("-" * 30)

################################
################################
################################