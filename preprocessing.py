# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for enhanced plots
import warnings
import gc
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

# Apply one-hot encoding to 'device_profile' and 'device_name'
df_encoded = one_hot_encode(df, ['device_profile', 'device_name'])

### Selecting target as ipec value ###

#Add target column ipec_pd  based on sum of 'ipec_pddata_avg_pd_a_value', 'ipec_pddata_avg_pd_b_value', 'ipec_pddata_avg_pd_c_value'
df_encoded['ipec_pd'] = df_encoded[['ipec_pddata_avg_pd_a_value', 'ipec_pddata_avg_pd_b_value', 'ipec_pddata_avg_pd_c_value']].sum(axis=1)

# Drop the individual ipec_pddata_avg_pd columns
columns_to_drop = ['ipec_pddata_avg_pd_a_value', 'ipec_pddata_avg_pd_b_value', 'ipec_pddata_avg_pd_c_value']
df_encoded.drop(columns=columns_to_drop, inplace=True)

# Is it weekend column based on 'time' ms-epoch column
def is_weekend(time_column):
    """
    Check if the time is on a weekend.
    
    Parameters:
    time_column (pd.Series): Series containing time in ms-epoch format.
    
    Returns:
    pd.Series: Boolean Series indicating if the time is on a weekend.
    """
    return pd.to_datetime(time_column, unit='ms').dt.dayofweek >= 5

# create test_df with 2 columns which are time and is_weekend column
test_df = pd.DataFrame({
    'time': df_encoded['time'],
    'is_weekend': is_weekend(df_encoded['time'])
})

# Add the is_weekend column to the main DataFrame
df_encoded['is_weekend'] = test_df['is_weekend']

# Drop the 'time' column if it exists, as it is not needed for modeling
if 'time' in df_encoded.columns:
    df_encoded.drop(columns=['time'], inplace=True)
# Reset index after dropping columns
df_encoded.reset_index(drop=True, inplace=True) 


# put the target column 'ipec_pd' at the end of the DataFrame
target_column = df_encoded.pop('ipec_pd')
df_encoded['ipec_pd'] = target_column 

gc.collect()

# Split the DataFrame into X (features) and y (target)
X = df_encoded.drop(columns=['ipec_pd'])
y = df_encoded['ipec_pd']
