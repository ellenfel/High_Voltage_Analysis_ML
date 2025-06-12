# %%

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


# reading data
df = pd.read_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/hv_ts.csv')


# renaming columns
df.rename(columns={'ts': 'time', 'device_profile': 'device_profile', 'devname': 'device_name', 'key': 'key', 'merged_column': 'value'}, inplace=True)
df_sample = df.head(100)


#gets all the keys other than error(no key is error in current db)
unique_values = df['key'].unique()
unique_values = [value for value in unique_values if 'error' not in str(value)]

# filtering out rows where 'key' contains 'error'
df = df[~df['key'].str.contains('error', case=False, na=False)]

# This works
# removing rows with specific values in the 'key' column
values_to_exclude = ['devName', 'devEUI', 'time', 'snr', 'rssi', 'protocol_version', 'firmware_version', 'hardware_version', 'sn', 'active', 'images_urls', 'water_images_urls', 'serialnumber', 'Location']
column_name_to_check = 'key'

# Filter out rows where the value in the 'key' column is not in the list of values to exclude
#df_sample = df.sample(frac=0.1)  # Use 10% of the data for testing
df = df[~df[column_name_to_check].isin(values_to_exclude)]
unique_values = df['key'].unique()




#for sampling
df_sample = df.head(10000)
unique_values = df['key'].unique()

#df = df.iloc[10:] #df is not ready for slicing, it has no header and the first 10 rows are not useful
#df['value'] = df['value'].astype('float64')



# Method 1: Using pivot_table (recommended for potential duplicate handling)
df_pivoted = df.pivot_table(
    index=['time', 'device_profile', 'device_name'], 
    columns='key', 
    values='value', 
    aggfunc='first'  # Use 'first' in case of duplicates, or 'mean' for numeric data
).reset_index()

# Flatten the column names (remove the hierarchical structure)
df_pivoted.columns.name = None

# Method 2: Using pivot (simpler but will fail if there are duplicates)
# df_pivoted = df.pivot(
#     index=['time', 'device_profile', 'device_name'], 
#     columns='key', 
#     values='value'
# ).reset_index()

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
