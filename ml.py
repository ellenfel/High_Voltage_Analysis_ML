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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_ml_ready.csv')
target_column_desc = df['ipec_pd'].describe()
print(f"df.shape: {df.shape}")


# put the target column 'ipec_pd' at the end of the DataFrame
target_column = df.pop('ipec_pd')
df['ipec_pd'] = target_column 
gc.collect()

# Split the DataFrame into X (features) and y (target)
X = df.drop(columns=['ipec_pd'])
y = df['ipec_pd']

# --- 0. Setup and Configuration ---
print("--- Initializing Setup and Configuration ---")
# Directory for saving figures
FIGURE_DIR = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/docs/figures'

# Plot styling for consistency
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
FIGURE_DPI = 300 # High resolution for thesis

# --- 1. Data Splitting ---
print("\n--- 1. Splitting Data into Training and Testing Sets ---")
# A single split is still useful for a final hold-out test set.
# Cross-validation will be done on the training set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
gc.collect()

# --- 2. Feature Scaling ---
print("\n--- 2. Scaling Features ---")
# Fit on training data only to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Convert back to DataFrame to keep column names for feature importance plots
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
print("Features scaled using StandardScaler.")
gc.collect()

### 3. Model Training and Initial Evaluation
print("\n--- 4. Model Training and Initial Evaluation ---")

# Note, y_train and y_test are NOT transformed for this baseline

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains a given model, makes predictions, and evaluates its performance.
    """
    print(f"\nTraining {model_name}...")
    start_time = datetime.now()
    model.fit(X_train, y_train)
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds.")

    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)

    # Ensure predictions don't go negative if your target truly cannot be negative
    # (ipec_pd min is 0.0, so negative predictions are physically impossible)
    y_pred[y_pred < 0] = 0

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"--- {model_name} Performance ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # Plotting Actual vs. Predicted (Sample)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual ipec_pd")
    plt.ylabel("Predicted ipec_pd")
    plt.title(f'{model_name}: Actual vs. Predicted ipec_pd (Test Set)')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURE_DIR, f'{model_name.lower().replace(" ", "_")}_actual_vs_predicted.png'), dpi=FIGURE_DPI)
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title(f'{model_name}: Residuals Distribution')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{model_name.lower().replace(" ", "_")}_residuals_distribution.png'), dpi=FIGURE_DPI)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted ipec_pd")
    plt.ylabel("Residuals")
    plt.title(f'{model_name}: Residuals vs. Predicted Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{model_name.lower().replace(" ", "_")}_residuals_vs_predicted.png'), dpi=FIGURE_DPI)
    plt.show()

    gc.collect()

    return {"model": model, "rmse": rmse, "mae": mae, "r2": r2, "training_time": training_time}


# Initialize and train a RandomForestRegressor as a baseline
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 uses all available cores
rf_results = train_and_evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test, model_name="Random Forest Regressor")
