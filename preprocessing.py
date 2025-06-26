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


import pandas as pd
import numpy as np
import gc
import warnings
warnings.filterwarnings('ignore')

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

# --- IMPORTANT: This code assumes df_encoded, X, and y are ready from your previous cells. ---
# --- No dummy data generation is included; it works directly with your prepared data. ---

print("Starting Machine Learning Pipeline...")

# --- 1. Handle NaNs in Features (X) ---
# As discussed, for structural NaNs due to different device signal lists,
# zero-imputation is a common and often effective strategy for sensor data.
# This implies that a missing reading for a sensor means it's inactive or not present.
print(f"NaNs in X before imputation: {X.isna().sum().sum()}")
X.fillna(0, inplace=True)
print(f"NaNs in X after zero-imputation: {X.isna().sum().sum()}")
gc.collect() # Free memory after imputation


# --- 2. Data Splitting ---
# Split the data into training and testing sets.
# For time-series, a time-based split is generally preferred to prevent data leakage,
# but for simplicity and if the order isn't strictly sequential for all predictions,
# a random split is a good starting point.
print("\nSplitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
gc.collect() # Free memory after splitting


# --- 3. Feature Scaling ---
# Scaling is crucial for many ML models (like linear models, SVMs, k-NN)
# but less critical for tree-based models. However, it's good practice.
# Fit the scaler only on the training data to avoid data leakage.
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")
gc.collect() # Free memory after scaling


# --- 4. Model Selection and Training ---
# We'll use a few common regression models suitable for this type of data.
# Tree-based models (RandomForest, XGBoost, LightGBM) are generally robust
# to imputed NaNs and varying feature scales.

models = {
    'RandomForestRegressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBRegressor': XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, tree_method='hist'), # 'hist' for larger datasets
#    'LGBMRegressor': LGBBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}

print("\n--- Training and Evaluating Models ---")
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train) # Train on scaled data
    gc.collect() # Clean up memory after training each model

    print(f"Evaluating {name}...")
    y_pred = model.predict(X_test_scaled) # Predict on scaled test data

    # Evaluation Metrics for Regression
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    print(f"{name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # --- Figure Generation for Thesis ---
    print(f"Generating figures for {name}...")

    # Figure 1: Actual vs. Predicted Values Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Line for perfect prediction
    plt.title(f'{name}: Actual vs. Predicted ipec_pd Values')
    plt.xlabel('Actual ipec_pd')
    plt.ylabel('Predicted ipec_pd')
    plt.grid(True)
    plt.show()

    # Figure 2: Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', lw=2) # Line at zero for reference
    plt.title(f'{name}: Residuals Plot (Predicted vs. Error)')
    plt.xlabel('Predicted ipec_pd')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True)
    plt.show()

    # Figure 3: Distribution of Residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title(f'{name}: Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    plt.close('all') # Close all figures to free up memory
    gc.collect() # Clean up memory after plotting

print("\n--- Model Training & Evaluation Complete ---")
print("\nSummary of Model Performance:")
for name, metrics in results.items():
    print(f"Model: {name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("-" * 20)

# You now have trained models and their initial performance metrics.
# Next steps would involve hyperparameter tuning, more advanced feature engineering,
# and potentially cross-validation.

gc.collect() # Final garbage collection at the end of the cell




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# Ensure X is available from the previous steps (after NaN imputation)
# If this cell is run independently, X needs to be loaded/created here.
# Assuming X exists from the previous ML pipeline cell, after X.fillna(0, inplace=True)

print("Starting feature correlation analysis...")

# Calculate the correlation matrix
# Using X (features) after NaN imputation but before scaling to see correlations
# among the actual values.
print("Calculating correlation matrix for features (X)...")
correlation_matrix = X.corr()
print("Correlation matrix calculated.")
gc.collect() # Free memory after correlation calculation


# --- Visualization: Heatmap of the Correlation Matrix ---
print("\nGenerating correlation heatmap...")
plt.figure(figsize=(20, 18)) # Adjust figure size based on number of columns
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Features', fontsize=20)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()
plt.close('all') # Close the figure to free up memory
gc.collect()


# --- Identifying Highly Correlated Pairs ---
# Set a correlation threshold to identify highly correlated features.
# A high positive (close to 1) or high negative (close to -1) correlation indicates strong relationships.
correlation_threshold = 0.8 # You can adjust this threshold (e.g., 0.9 for very high)

print(f"\nIdentifying highly correlated feature pairs (absolute correlation > {correlation_threshold})...")

# Get unique pairs from the upper triangle of the correlation matrix (excluding self-correlation)
highly_correlated_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        col1 = correlation_matrix.columns[i]
        col2 = correlation_matrix.columns[j]
        corr_val = correlation_matrix.iloc[i, j]

        if abs(corr_val) > correlation_threshold:
            highly_correlated_pairs.append((col1, col2, corr_val))

if highly_correlated_pairs:
    print("\nHighly Correlated Feature Pairs:")
    for col1, col2, corr_val in highly_correlated_pairs:
        print(f"- '{col1}' and '{col2}': Correlation = {corr_val:.4f}")
    print("\nConsider these for potential multicollinearity or redundancy.")
else:
    print(f"\nNo feature pairs found with absolute correlation greater than {correlation_threshold}.")

gc.collect() # Final garbage collection
print("\nFeature correlation analysis complete.")
