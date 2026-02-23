# %% 

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc 
warnings.filterwarnings('ignore')


def process_hv_data():
    import pandas as pd

    # reading data
    # Ensure this path is correct for your local setup
    df = pd.read_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/hv_ts.csv')

    # renaming columns
    df.rename(columns={'ts': 'time', 'device_profile': 'device_profile', 'devname': 'device_name', 'key': 'key', 'merged_column': 'value'}, inplace=True)

    #gets all the keys other than error (no key is error in current db)
    unique_values = df['key'].unique()
    unique_values = [value for value in unique_values if 'error' not in str(value)]

    # removing rows with specific values in the 'key' column

    #df = df.sample(frac=0.1)  # This line was for testing with a sample, removed for full data processing.

    values_to_exclude = [
        'devName', 'devEUI', 'time', 'snr', 'rssi', 'protocol_version',
        'firmware_version', 'hardware_version', 'sn', 'active',
        'images_urls', 'water_images_urls', 'serialnumber', 'Location',
        'gpio_in_1', 'gpio_in_2', 'gpio_in_3', 'gpio_in_4',
        'gpio_out_1', 'gpio_out_2', '420_ir_a_value', '420_ir_b_value',
        '420_ir_c_value', 'alarm_value'
    ]

    column_name_to_check = 'key'
    # Print shape before filtering
    print(f"Shape of df before filtering: {df.shape}")
    # Filter out rows where the value in the 'key' column is not in the list of values to exclude
    df = df[~df[column_name_to_check].isin(values_to_exclude)]
    # Print shape after filtering
    print(f"Shape of df after filtering: {df.shape}")
    unique_values_after = df['key'].unique()

    return df

# Load your DataFrame
df = process_hv_data()

# Explicitly call garbage collection after initial data loading and filtering
gc.collect()
print(f"DataFrame 'df' loaded and pre-filtered with shape: {df.shape}")
print("-" * 30)


########## SECTION 1: DATA CLEANING AND PREPROCESSING (OPTIMIZED) ##########

# Print current count of NaN values in the original "value" column
print("Initial NaN count in 'value':", df['value'].isna().sum())

# Step 1: Make a backup copy of the original "value" column
# CRITICAL CHANGE: Using .copy(deep=True) ensures 'clean_value' is a *completely independent*
# copy in memory. This guarantees that any operations on 'clean_value'
# do NOT affect the original 'df['value']' column.
df['clean_value'] = df['value'].copy(deep=True)

# Step 2: Remove leading and trailing whitespace from the backup column
# This operation is vectorized by pandas and generally memory-efficient.
df['clean_value'] = df['clean_value'].astype(str).str.strip()

# Step 3: Convert the cleaned column to numeric. Non-convertible values become NaN.
# This vectorized operation efficiently converts numeric strings and turns unconvertible ones into NaN.
df['clean_value'] = pd.to_numeric(df['clean_value'], errors='coerce')

# Step 4: Check what the conversion did by printing stats
print("=== Clean Value Column Stats ===")
print(f"Data type of 'clean_value': {df['clean_value'].dtype}")
print(f"Total NaN values in 'clean_value': {df['clean_value'].isna().sum()}")

# Step 5: Make a list of unique original values that were converted to NaN in 'clean_value'
# This diagnostic step efficiently works on the subset of data that became NaN.
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
# CRITICAL CHANGE: Replaced the inefficient Python for-loop with a single, vectorized pandas operation.
# This is the primary change to prevent memory overloads and crashes.
print("\nApplying conversion map using vectorized operations (memory-optimized)...")

# 1. Create a boolean mask identifying rows where 'clean_value' is currently NaN.
nan_mask = df['clean_value'].isna()

# 2. For the rows identified by 'nan_mask', take the *original* 'value' data.
# Convert these specific original strings to lowercase and strip whitespace *once* for mapping.
# Then, apply the 'conversion_map' using the highly optimized .map() method.
temp_mapped_values = df.loc[nan_mask, 'value'].astype(str).str.strip().str.lower().map(conversion_map)

# 3. Assign the results from 'temp_mapped_values' back to the 'clean_value' column,
# ensuring updates happen only where 'clean_value' was originally NaN.
df.loc[nan_mask, 'clean_value'] = temp_mapped_values

# Explicitly delete the temporary series to free up memory immediately, reducing peak usage.
del temp_mapped_values
gc.collect() # Trigger Python's garbage collector to free memory immediately

print("Conversion map applied.")

# Final conversion to numeric to ensure consistent data type across the entire 'clean_value' column.
df['clean_value'] = pd.to_numeric(df['clean_value'], errors='coerce')

print("\n--- After Conversion Map Application ---")
print("Data type of 'clean_value':", df['clean_value'].dtype)
print("Total NaN values in 'clean_value':", df['clean_value'].isna().sum())

# Including memory usage check at the end of this section
print("\n--- DataFrame Memory Usage (Deep) After Cleaning ---")
print(df.info(memory_usage='deep'))
gc.collect() # Final garbage collection for cleaning section


### Feature Engineering: Pivoting the DataFrame (OPTIMIZED) ###

print("\nStarting DataFrame pivot process...")

# Define the index columns for the pivot. These will become the unique rows in the pivoted DataFrame.
pivot_index_cols = ['time', 'device_profile', 'device_name']

# Perform the pivot operation.
# CRITICAL CHANGE: Using 'clean_value' for the values, as it's the cleaned numeric data.
df_pivoted = df.pivot(
    index=pivot_index_cols,
    columns='key',
    values='clean_value' # <<<--- CHANGED from 'value' to 'clean_value'
).reset_index() # .reset_index() converts the pivot index back into regular columns.

# Optional: Add suffix to key column names for clarity
df_pivoted.columns = [col if col in pivot_index_cols
                     else f"{col}_value" for col in df_pivoted.columns]

# Explicitly run garbage collection to free up memory from intermediate objects created during pivot.
gc.collect()

# Display the result
print("\n--- Pivot Operation Complete ---")
print("Original DataFrame shape (before pivot):", df.shape) # Referencing df's shape before pivot
print("Pivoted DataFrame shape:", df_pivoted.shape)
print("\nPivoted DataFrame columns (first 20):")
print(df_pivoted.columns.tolist()[:20])
if len(df_pivoted.columns) > 20:
    print(f"... and {len(df_pivoted.columns) - 20} more columns.")
print("\nFirst few rows of Pivoted DataFrame:")
print(df_pivoted.head())

# Optional: Check for any missing values after pivot
print(f"\nMissing values per column (total for each pivoted key):")
# Displaying only top 20 for brevity, the full count can be found by inspecting df_pivoted.isnull().sum()
print(df_pivoted.isnull().sum().head(20))
if len(df_pivoted.columns) > 20:
    print(f"... and missing counts for {len(df_pivoted.columns) - 20} more columns.")

# Print memory usage of the new pivoted DataFrame
print("\n--- Pivoted DataFrame Memory Usage (Deep) ---")
print(df_pivoted.info(memory_usage='deep'))
gc.collect() # Final garbage collection for pivoting section

# Keep the df_sample line as it was in your original script
df_sample = df_pivoted.head(10000)
print("\n'df_sample' created (first 10,000 rows of df_pivoted).")

#