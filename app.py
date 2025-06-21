# %%

########## SECTION 1: IMPORTING DATA AND LIBS ##########

# importing libs# importing libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import missingno as msno  ModuleNotFoundError: No module named 'numpy.rec'
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

    values_to_exclude = ['devName', 'devEUI', 'time', 'snr', 'rssi', 'protocol_version', 'firmware_version', 'hardware_version', 'sn', 'active', 'images_urls', 'water_images_urls', 'serialnumber', 'Location']
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


# Call the function
df = process_hv_data()

#for sampling
#df_sample = df.head(10000)
#unique_values = df['key'].unique()


########## SECTION 2: DATA EXPLORATION ##########

# Check for NaN values in the DataFrame
print("=== Checking for NaN values in the DataFrame ===")
print(f"Total NaN values in DataFrame: {df.isnull().sum().sum()}")


# Summarize the DataFrame
print("=== DataFrame Summary ===")
print(f"Shape: {df.shape}")
print(f"Column Names: {df.columns.tolist()}")
print("\nData Types:")
print(df.dtypes)
print("\nFirst Few Rows:")
print(df.head())
print("\nMissing Values Per Column:")
print(df.isnull().sum())
print(f"Data Type of column key: {df['key'].dtype}")

# Add unique elements for device_profile and device_name
print("\nUnique Elements:")
print(f"Unique device profiles: {df['device_profile'].nunique()} ({df['device_profile'].unique()[:5]}...)")
print(f"Unique device names: {df['device_name'].nunique()} ({df['device_name'].unique()[:5]}...)")

# Add count of different keys
print(f"\nNumber of different keys: {df['key'].nunique()}")


# Issue: NaN values in DataFrame
# Step 1: Create a quick pivot to identify NaN locations
print("\n1. Creating quick pivot to identify NaN locations...")
df_copy = df.copy()

# Quick pivot without cleaning to see where NaNs appear
df_quick_pivot = df_copy.pivot(
    index=['time', 'device_profile', 'device_name'],
    columns='key',
    values='value'
).reset_index()

# Step 2: Find NaN locations and map back to original values
print("\n2. Analyzing NaN patterns...")
nan_analysis = {}

for col in df_quick_pivot.columns:
    if col not in ['time', 'device_profile', 'device_name']:
        nan_count = df_quick_pivot[col].isna().sum()
        total_count = len(df_quick_pivot)
        if nan_count > 0:
            nan_analysis[col] = {
                'nan_count': nan_count,
                'total_count': total_count,
                'nan_percentage': (nan_count / total_count) * 100
            }

print(f"Columns with NaN values: {len(nan_analysis)}")
for key, stats in nan_analysis.items():
    print(f"  {key}: {stats['nan_count']}/{stats['total_count']} ({stats['nan_percentage']:.1f}% NaN)")

# Step 3: Sample problematic values from original data for each key with NaNs
print("\n3. Sampling original values for keys with NaN issues...")
conversion_map = {}

for key_with_nans in list(nan_analysis.keys())[:10]:  # Limit to first 10 for analysis
    print(f"\n--- Analyzing key: {key_with_nans} ---")
    
    # Get all unique values for this key from original data
    key_values = df_copy[df_copy['key'] == key_with_nans]['value'].unique()
    print(f"Unique value count: {len(key_values)}")
    
    # Sample and display unique values
    sample_size = min(20, len(key_values))
    sample_values = key_values[:sample_size]
    print(f"Sample values: {sample_values}")
    
    # Analyze potential issues
    problematic_values = []
    clean_values = []
    
    for val in sample_values:
        if pd.isna(val):
            continue
            
        val_str = str(val)
        original_val = val_str
        cleaned_val = val_str.strip()
        
        # Check for various issues
        has_spaces = original_val != cleaned_val
        is_boolean_like = cleaned_val.lower() in ['true', 'false', 't', 'f', '1', '0', 'yes', 'no', 'on', 'off']
        is_numeric_like = False
        
        try:
            float(cleaned_val)
            is_numeric_like = True
        except:
            pass
        
        if has_spaces or is_boolean_like or is_numeric_like:
            problematic_values.append({
                'original': original_val,
                'cleaned': cleaned_val,
                'has_spaces': has_spaces,
                'is_boolean_like': is_boolean_like,
                'is_numeric_like': is_numeric_like
            })
        else:
            clean_values.append(original_val)
    
    if problematic_values:
        print(f"  Problematic values found: {len(problematic_values)}")
        for pv in problematic_values[:5]:  # Show first 5
            print(f"    '{pv['original']}' -> '{pv['cleaned']}' (spaces:{pv['has_spaces']}, bool:{pv['is_boolean_like']}, num:{pv['is_numeric_like']})")
    
    conversion_map[key_with_nans] = {
        'total_unique': len(key_values),
        'problematic_count': len(problematic_values),
        'clean_count': len(clean_values),
        'sample_problematic': problematic_values[:10],  # Keep first 10 for mapping
        'sample_clean': clean_values[:5]
    }



########## SECTION 3: DATA CLEANING AND TRANSFORMATION ##########

# NEW APPROACH TO CLEAN THE 'value' COLUMN AND CHECK NAN VALUES

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
    "water alert.png": 1      # water alert is 1
}

# Create a new boolean column from the cleaned value column.
# Ensure the cleaned column is in string format and in lowercase before mapping.
df['boolean_value'] = df['clean_value'].astype(str).str.lower().map(conversion_map)

# Check the conversion results by counting each value (including any NaNs)
print("=== Boolean Value Column Stats ===")
print(df['boolean_value'].value_counts(dropna=False))

# Verify conversion for specific problematic values
# Create a temporary lowercased version of clean_value for robust matching
df['clean_value_lower'] = df['clean_value'].astype(str).str.lower()

# Define the values to check (lowercase)
values_to_check = ["i-lb_closed.svg", "safe.png"]

# Filter rows where the cleaned value matches these problematic values
filtered_test = df[df['clean_value_lower'].isin(values_to_check)][['value', 'clean_value', 'boolean_value']]

# Print the filtered rows to verify if the boolean conversion worked as expected
print("Filtered test rows for conversion:")
print(filtered_test)















# BEFORE count - original data forced to numeric
df['value_temp_numeric'] = pd.to_numeric(df['value'].astype(str).str.strip(), errors='coerce')
df_pivot_before = df.pivot_table(
   index=['time', 'device_name', 'device_profile'],
   columns='key',
   values='value_temp_numeric',
   aggfunc='mean'
)
nans_before = df_pivot_before.isnull().sum().sum()
print(f"✅ Total NaN Count BEFORE fix: {nans_before}")
df.drop(columns=['value_temp_numeric'], inplace=True)

# First, identify which keys should be numeric vs text
numeric_keys = []
text_keys = []

for key in df['key'].unique():
   key_values = df[df['key'] == key]['value'].astype(str).str.strip()
   # Try converting a sample to see if it's mostly numeric
   sample_converted = pd.to_numeric(key_values.head(100), errors='coerce')
   numeric_ratio = sample_converted.notna().sum() / len(sample_converted)
   
   if numeric_ratio > 0.8:  # If >80% of values are numeric
       numeric_keys.append(key)
   else:
       text_keys.append(key)

print(f"Numeric keys: {numeric_keys}")
print(f"Text keys: {text_keys}")

# Now create separate pivots for numeric and text data
df_numeric = df[df['key'].isin(numeric_keys)].copy()
df_text = df[df['key'].isin(text_keys)].copy()

# Clean and convert only numeric data
df_numeric['value_clean_numeric'] = pd.to_numeric(
   df_numeric['value'].astype(str).str.strip(), 
   errors='coerce'
)

# Pivot numeric data
df_numeric_pivot = df_numeric.pivot_table(
   index=['time', 'device_name', 'device_profile'],
   columns='key',
   values='value_clean_numeric',
   aggfunc='mean'
)

# Pivot text data (keep as text)
df_text_pivot = df_text.pivot_table(
   index=['time', 'device_name', 'device_profile'],
   columns='key',
   values='value',
   aggfunc='first'  # Take first value for text
)

# Combine both pivots
df_final = pd.concat([df_numeric_pivot, df_text_pivot], axis=1).reset_index()

# AFTER count
nans_after = df_final.isnull().sum().sum()
print(f"✅ Total NaN Count AFTER proper fix: {nans_after}")


print("Detailed NaN analysis:")
print(f"Numeric pivot NaNs: {df_numeric_pivot.isnull().sum().sum()}")
print(f"Text pivot NaNs: {df_text_pivot.isnull().sum().sum()}")
print(f"Combined NaNs: {df_final.isnull().sum().sum()}")

print(f"\nNumeric pivot shape: {df_numeric_pivot.shape}")
print(f"Text pivot shape: {df_text_pivot.shape}")
print(f"Combined shape: {df_final.shape}")

# Check if the numeric cleaning actually worked
print(f"\nNumeric data before cleaning - failed conversions:")
df_numeric_test = df[df['key'].isin(numeric_keys)].copy()
df_numeric_test['original_numeric'] = pd.to_numeric(df_numeric_test['value'], errors='coerce')
df_numeric_test['cleaned_numeric'] = pd.to_numeric(df_numeric_test['value'].astype(str).str.strip(), errors='coerce')

original_nans = df_numeric_test['original_numeric'].isnull().sum()
cleaned_nans = df_numeric_test['cleaned_numeric'].isnull().sum()

print(f"Original numeric NaNs: {original_nans}")
print(f"Cleaned numeric NaNs: {cleaned_nans}")
print(f"NaNs reduced by cleaning: {original_nans - cleaned_nans}")























df['value'] = df['value'].str.strip()
# Pivot the DataFrame to a wide format
df_pivot = df.pivot(
    index=['time', 'device_profile', 'device_name'], 
    columns='key', 
    values='value'
).reset_index()

# Identify all columns that came from the 'key' column
value_columns = df['key'].unique()

# Loop through these columns and convert them to numeric types
for col in value_columns:
    if col in df_pivot.columns:
        df_pivot[col] = pd.to_numeric(df_pivot[col], errors='coerce')

# Optional: You can also handle boolean-like columns if needed
# For example, for the 'alarm' column:
if 'alarm' in df_pivot.columns:
    # Convert 'true' to True, 'false' to False, and everything else to NaN
    df_pivot['alarm'] = df_pivot['alarm'].map({'true': True, 'false': False}).astype(float) # Using float to keep NaNs

print("=== Data Types After Conversion ===")
print(df_pivot.info())



# Step 1: Create conversion function based on your analysis
def clean_iot_value(value):
    """
    Clean IoT sensor values - mainly strip spaces and convert to appropriate types
    Based on analysis showing most values are floats with space padding
    """
    if pd.isna(value):
        return np.nan
    
    # Convert to string and strip all whitespace
    cleaned = str(value).strip()
    
    # Handle empty strings after stripping
    if cleaned == '' or cleaned.lower() in ['null', 'none', 'nan']:
        return np.nan
    
    # Handle boolean values (like 'alarm' field)
    if cleaned.lower() == 'true':
        return True
    elif cleaned.lower() == 'false':
        return False
    
    # Try to convert to numeric (most of your data)
    try:
        # Try integer first (for values like '0', '122', '12200')
        if '.' not in cleaned:
            return int(cleaned)
        else:
            # Float for decimal values
            return float(cleaned)
    except ValueError:
        # If conversion fails, return as cleaned string
        return cleaned

# Step 2: Apply conversion efficiently
print("Applying conversion to values...")
df_cleaned = df.copy()

# Clean the values
df_cleaned['value'] = df_cleaned['value'].apply(clean_iot_value)

print("Conversion completed!")


# Step 3: Test with a small sample first
print("\n=== TESTING ON SMALL SAMPLE ===")
sample_size = 10000
#df_test = df_cleaned.head(sample_size).copy()
df_test = df.head(sample_size).copy()  # Use original df for testing pivot

print(f"Testing pivot on {sample_size} rows...")
try:
    df_test_pivot = df_test.pivot(
        index=['time', 'device_profile', 'device_name'],
        columns='key',
        values='value'
    ).reset_index()
    
    df_test_pivot.columns.name = None
    
    # Add suffix to key columns
    df_test_pivot.columns = [col if col in ['time', 'device_profile', 'device_name'] 
                            else f"{col}_value" for col in df_test_pivot.columns]
    
    print(f"✅ SUCCESS! Test pivot completed.")
    print(f"Test pivoted shape: {df_test_pivot.shape}")
    print(f"NaN count in test pivot: {df_test_pivot.isnull().sum().sum()}")
    
    # Show sample of successful conversion
    print(f"\nSample of pivoted data:")
    print(df_test_pivot.head(3))
    
except Exception as e:
    print(f"❌ Test failed: {e}")





import pandas as pd
import numpy as np

print("=== DEBUGGING THE REAL NaN ISSUE ===")

# Let's test with the first 10,000 rows to understand the pattern
sample_size = 10000
df_debug = df.head(sample_size).copy()

print(f"Debug sample shape: {df_debug.shape}")
print(f"Unique devices in sample: {df_debug['device_name'].nunique()}")
print(f"Unique keys in sample: {df_debug['key'].nunique()}")
print(f"Sample keys: {df_debug['key'].unique()}")

# Step 1: Check the data distribution
print(f"\n=== DEVICE AND KEY DISTRIBUTION ===")
device_key_counts = df_debug.groupby(['device_name', 'key']).size().reset_index(name='count')
print(f"Device-Key combinations: {len(device_key_counts)}")
print(device_key_counts.head(10))

# Step 2: Create a pivot and see exactly what's happening
print(f"\n=== PIVOT ANALYSIS ===")
df_pivot_debug = df_debug.pivot(
    index=['time', 'device_profile', 'device_name'],
    columns='key',
    values='value'
).reset_index()

print(f"Pivot shape: {df_pivot_debug.shape}")
print(f"Total cells in pivot: {df_pivot_debug.shape[0] * df_pivot_debug.shape[1]}")
print(f"Non-index columns: {df_pivot_debug.shape[1] - 3}")
print(f"Data cells (excluding index): {df_pivot_debug.shape[0] * (df_pivot_debug.shape[1] - 3)}")

# Step 3: Analyze NaN pattern by device
print(f"\n=== NaN PATTERN BY DEVICE ===")
data_columns = [col for col in df_pivot_debug.columns if col not in ['time', 'device_profile', 'device_name']]

for device in df_pivot_debug['device_name'].unique()[:3]:  # Check first 3 devices
    device_data = df_pivot_debug[df_pivot_debug['device_name'] == device]
    print(f"\nDevice: {device}")
    print(f"  Rows: {len(device_data)}")
    
    # Check which keys this device actually has data for
    non_nan_cols = []
    for col in data_columns:
        non_nan_count = device_data[col].notna().sum()
        if non_nan_count > 0:
            non_nan_cols.append((col, non_nan_count))
    
    print(f"  Keys with data: {len(non_nan_cols)}")
    if non_nan_cols:
        print(f"  Sample keys: {[col[0] for col in non_nan_cols[:5]]}")

# Step 4: Check if different devices have different sensors
print(f"\n=== DEVICE SENSOR MAPPING ===")
device_sensors = {}
for device in df_debug['device_name'].unique():
    device_keys = df_debug[df_debug['device_name'] == device]['key'].unique()
    device_sensors[device] = set(device_keys)
    print(f"{device}: {len(device_keys)} keys - {list(device_keys)[:5]}...")

# Step 5: Calculate expected vs actual data density
print(f"\n=== DATA DENSITY ANALYSIS ===")
total_possible_combinations = len(df_pivot_debug) * len(data_columns)
actual_data_points = df_debug.shape[0]  # Original row count
nan_count = df_pivot_debug[data_columns].isna().sum().sum()
filled_count = total_possible_combinations - nan_count

print(f"Original data rows: {df_debug.shape[0]}")
print(f"Pivot rows: {len(df_pivot_debug)}")
print(f"Possible sensor columns: {len(data_columns)}")
print(f"Total possible cells: {total_possible_combinations}")
print(f"Filled cells: {filled_count}")
print(f"NaN cells: {nan_count}")
print(f"Data density: {(filled_count/total_possible_combinations)*100:.1f}%")

# Step 6: The key insight - check if this is normal IoT behavior
print(f"\n=== IOT DATA REALITY CHECK ===")
print("This high NaN count is likely NORMAL for IoT data because:")
print("1. Different device types have different sensors")
print("2. Devices send data at different intervals")
print("3. Not every device measures every parameter")

# Step 7: Verify this theory
print(f"\n=== VERIFICATION ===")
# Group by device profile to see sensor differences
profile_sensors = df_debug.groupby('device_profile')['key'].unique()
print("Sensors by device profile:")
for profile, sensors in profile_sensors.items():
    print(f"  {profile}: {len(sensors)} sensors")

# Check time intervals
print(f"\nTime range in sample:")
print(f"  Min time: {df_debug['time'].min()}")
print(f"  Max time: {df_debug['time'].max()}")
print(f"  Unique timestamps: {df_debug['time'].nunique()}")

# Final recommendation
print(f"\n=== CONCLUSION ===")
if nan_count > filled_count:
    print("✅ HIGH NaN COUNT IS EXPECTED!")
    print("   - Different devices have different sensors")
    print("   - This creates a sparse matrix when pivoted")
    print("   - Your data is probably clean already")
    print("   - Focus on analyzing devices individually or by type")
else:
    print("❌ Something else is wrong - investigate further")

print(f"\n=== RECOMMENDATION ===")
print("Try analyzing by device profile or individual devices:")
print("df_analysis = df_pivot_debug[df_pivot_debug['device_profile'] == 'specific_profile']")
print("Then check NaN counts for that subset")











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










