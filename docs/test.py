# %%

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import folium
from folium.plugins import HeatMap
import plotly.express as px

# plotting configurations
plt.style.use('fivethirtyeight')
# %matplotlib inline
pd.set_option('display.max_columns', 32)

def process_hv_data():
    import pandas as pd

    # reading data
    df = pd.read_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/hv_ts.csv')

    # renaming columns
    df.rename(columns={'ts': 'time', 'device_profile': 'device_profile', 'devname': 'device_name', 'key': 'key', 'merged_column': 'value'}, inplace=True)

    #gets all the keys other than error(no key is error in current db)
    unique_values = df['key'].unique()
    unique_values = [value for value in unique_values if 'error' not in str(value)]

    # removing rows with specific values in the 'key' column

    # Sample df so it wouldnt crash, Exculude this in colab
    df = df.sample(frac=0.1)  # Use 10% of the data for testing

    values_to_exclude = [
        'devName', 'devEUI', 'time', 'snr', 'rssi', 'protocol_version', 
        'firmware_version', 'hardware_version', 'sn', 'active', 
        'images_urls', 'water_images_urls', 'serialnumber', 'Location',
        'gpio_in_1', 'gpio_in_2', 'gpio_in_3', 'gpio_in_4', 
        'gpio_out_1', 'gpio_out_2'
    ]
    column_name_to_check = 'key'
    # Print shape before filtering
    print(f"Shape of df before filtering: {df.shape}")
    # Filter out rows where the value in the 'key' column is not in the list of values to exclude
    #df_sample = df.sample(frac=0.1)  # Use 10% of the data for testing
    df = df[~df[column_name_to_check].isin(values_to_exclude)]
    # Print shape after filtering
    print(f"Shape of df after filtering: {df.shape}")
    unique_values_after = df['key'].unique()

    return df

df = process_hv_data()



########## SECTION 1: DATA CLEANING AND PREPROCESSING ##########

# Print current count of NaN values in the original "value" column
print("Initial NaN count in 'value':", df['value'].isna().sum())

# Step 1: Make a backup copy of the original "value" column
df['clean_value'] = df['value']

# Step 2: Remove leading and trailing whitespace from the backup column
df['clean_value'] = df['clean_value'].astype(str).str.strip()

# Step 3: Convert the cleaned column to numeric. Non-convertible values become NaN.
df['clean_value'] = pd.to_numeric(df['clean_value'], errors='coerce')

# Step 4: Check what the conversion did by printing stats
print("=== Clean Value Column Stats ===")
print(f"Data type of 'clean_value': {df['clean_value'].dtype}")
print(f"Total NaN values in 'clean_value': {df['clean_value'].isna().sum()}")

# Step 5: Make a list of unique original values that were converted to NaN in 'clean_value'
nan_values = df.loc[df['clean_value'].isna(), 'value'].astype(str).str.strip()
nan_counts = nan_values.value_counts()
print("Unique original values converted to NaN and their counts:")
print(nan_counts)

# Define the conversion map (all keys in lowercase for consistency)
conversion_map = {
    "safe.png": 0,            # safe is 0
    "i-lb_closed.svg": 1,     # closed is 1
    "off": 0,                 # off assumed to be 0 (false)
    "i-lb_open.svg": 0,       # open is 0
    "i-lb closed.png": 1,     # closed is 1
    "i-lb open.png": 0,       # open is 0
    "i-lb closed1.png": 1,    # closed is 1
    "true": 1,                # true maps to 1
    "false": 0,               # false maps to 0
    "water alert.png": 1,     # water alert is 1
    "on": 1                   # on is 1
}

# Apply the conversion map to the 'value' column where 'clean_value' is NaN
# iterate through the original 'value' column where 'clean_value' became NaN
for original_str, mapped_val in conversion_map.items():
    # Find rows where clean_value is NaN AND original value matches the key (case-insensitive)
    mask = df['clean_value'].isna() & (df['value'].astype(str).str.strip().str.lower() == original_str)
    df.loc[mask, 'clean_value'] = mapped_val

df['clean_value'] = pd.to_numeric(df['clean_value'], errors='coerce')

print("\n--- After Conversion Map Application ---")
print("Data type of 'clean_value':", df['clean_value'].dtype)
print("Total NaN values in 'clean_value':", df['clean_value'].isna().sum())


### Feature Engineering: Pivoting the DataFrame ###

unique_values = df['key'].unique()

# Convert 'time' (milliseconds) to datetime objects
df['datetime'] = pd.to_datetime(df['time'], unit='ms')

# Create a binned time column by rounding to the nearest 3 seconds
df['binned_time'] = df['datetime'].dt.round('3S')

# Pivot the table using 'binned_time' as part of the index
# Use df.pivot_table because it can handle multiple values per bin using aggfunc
# We will aggregate 'clean_value' by its mean for values falling into the same 3-second window
df_pivoted = df.pivot_table(
    index=['binned_time', 'device_profile', 'device_name'],
    columns='key',
    values='clean_value',  # Use the cleaned, numeric values for pivoting
    aggfunc='mean'         # Aggregate multiple readings within a 3-second window by their mean
).reset_index()

# Remove the 'key' column name from the columns MultiIndex for cleaner access
df_pivoted.columns.name = None

# Optional: Add suffix to key column names for clarity (adjusted for pivot_table output)
new_columns = []
for col in df_pivoted.columns:
    if col in ['binned_time', 'device_profile', 'device_name']:
        new_columns.append(col)
    else:
        new_columns.append(f"{col}_value") # Add suffix to measure columns
df_pivoted.columns = new_columns

# Display the result
print("\n--- Pivoting with 3-second Time Window ---")
print("Original DataFrame shape:", df.shape)
print("Pivoted DataFrame shape:", df_pivoted.shape)
print("\nPivoted DataFrame columns:")
print(df_pivoted.columns.tolist())
print("\nFirst few rows:")
print(df_pivoted.head())

# Check for any missing values after pivot (these are now legitimate NaNs from sparse data)
print(f"\nMissing values per column:")
print(df_pivoted.isnull().sum().sort_values(ascending=False)) # Sorted to see most missing first

# Optional: Drop the intermediate 'datetime' column if no longer needed
df.drop(columns=['datetime'], inplace=True)







#Using pivot (simpler but will fail if there are duplicates)
df_pivoted = df.pivot(
    index=['time', 'device_profile', 'device_name'], 
    columns='key', 
    values='value'
).reset_index()

# Optional: Add suffix to key column names for clarity
df_pivoted.columns = [col if col in ['time', 'device_profile', 'device_name'] 
                     else f"{col}_value" for col in df_pivoted.columns]

# Display the result
print("Original DataFrame shape:", df.shape)
print("Pivoted DataFrame shape:", df_pivoted.shape)
print("\nPivoted DataFrame columns:")
print(df_pivoted.columns.tolist())
print("\nFirst few rows:")
print(df_pivoted.head())

# Optional: Check for any missing values after pivot
print(f"\nMissing values per column:")
print(df_pivoted.isnull().sum())

df_sample = df_pivoted.head(10000)



















