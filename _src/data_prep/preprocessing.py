# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for enhanced plots
import warnings
import gc

# ML specific imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor # For LightGBM Regressor

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings('ignore')


# reading data (this part is from your existing code)
df = pd.read_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_cleaned.csv')


# One hot encoding for 'device_profile' and 'device_name'
def one_hot_encode(df, columns):
    """
    Perform one-hot encoding on specified columns of the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to encode.
    columns (list): List of column names to apply one-hot encoding.
    
    Returns:
    pd.DataFrame: DataFrame with one-hot encoded columns.
    """
    return pd.get_dummies(df, columns=columns, drop_first=True)

#columns before one-hot encoding
print("Columns before one-hot encoding:")
print(df.columns.tolist())

# Apply one-hot encoding to 'device_profile' and 'device_name'
df_encoded = one_hot_encode(df, ['device_profile', 'device_name'])

#columns after one-hot encoding
print("\nColumns after one-hot encoding:")
print(df_encoded.columns.tolist())

### Selecting target as ipec value ###

#Add target column ipec_pd  based on sum of 'ipec_pddata_avg_pd_a_value', 'ipec_pddata_avg_pd_b_value', 'ipec_pddata_avg_pd_c_value'
df_encoded['ipec_pd'] = df_encoded[['ipec_pddata_avg_pd_a_value', 'ipec_pddata_avg_pd_b_value', 'ipec_pddata_avg_pd_c_value']].sum(axis=1)

# Drop the individual ipec_pddata_avg_pd columns
columns_to_drop = ['ipec_pddata_avg_pd_a_value', 'ipec_pddata_avg_pd_b_value', 'ipec_pddata_avg_pd_c_value']
df_encoded.drop(columns=columns_to_drop, inplace=True)

# remove 'image_name_value' and 'water_detector_value' columns
columns_to_drop = ['image_name_value', 'water_detector_value']
df_encoded.drop(columns=columns_to_drop, inplace=True)

# Time-based feature engineering
def is_weekend(time_column):
    """
    Check if the time is on a weekend.

    Parameters:
    time_column (pd.Series): Series containing time in ms-epoch format.

    Returns:
    pd.Series: Boolean Series indicating if the time is on a weekend.
    """
    # Convert to datetime and check day of week (Monday=0, Sunday=6)
    return pd.to_datetime(time_column, unit='ms').dt.dayofweek >= 5

if 'time' in df_encoded.columns:
    # Convert the 'time' column to datetime objects once for efficiency.
    print("Converting 'time' to datetime objects...")
    datetime_series = pd.to_datetime(df_encoded['time'], unit='ms')

    # Add only 'hour_of_day' feature
    print("Adding 'hour_of_day' feature...")
    df_encoded['hour_of_day'] = datetime_series.dt.hour          # Hour (0-23)

    # All other granular time-based features (day_of_week, day_of_month, month, year, etc.)
    # have been removed as per request.

    print("Time-based feature 'hour_of_day' added.")

    # Drop the original 'time' column, as its information is now decomposed into other features
    print("Dropping original 'time' column...")
    df_encoded.drop(columns=['time'], inplace=True)

# Reset index after potentially dropping columns (good practice)
df_encoded.reset_index(drop=True, inplace=True)
print("DataFrame index reset.")
gc.collect() # Clean up memory after feature engineering

print("Time-based feature engineering complete.")
print(f"New df_encoded shape after time feature engineering: {df_encoded.shape}")
print("New df_encoded columns (sample):")
print(df_encoded.columns.tolist()[-5:]) # Show last 5 columns which should include new time feature


# Save the ML Ready DataFrame to a CSV file for future use
output_file = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_ml_ready.csv'
print(f"Saving ML ready DataFrame to {output_file}...")
df_encoded.to_csv(output_file, index=False)
print("Data saved successfully.")

