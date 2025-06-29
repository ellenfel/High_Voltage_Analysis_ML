# %%

### This is for Keras Tuner ###

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import os
from datetime import datetime

# ML specific imports
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
# Expanded metrics imports
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, \
                            mean_squared_log_error, median_absolute_error, max_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor # Keeping commented as it's not in your model list currently

# Deep Learning Imports
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # Added ReduceLROnPlateau
warnings.filterwarnings('ignore')

# reading data (this part is from your existing code)
df = pd.read_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_ml_ready.csv')

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

# Create the directory if it doesn't exist
os.makedirs(FIGURE_DIR, exist_ok=True)
print(f"Figures will be saved to: {FIGURE_DIR}")

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

all_results = {}
best_estimators = {}

# --- 3. Helper Function for Plotting ---
def generate_model_plots(model_name, y_true, y_pred, feature_importances=None, feature_names=None):
    """
    Generates and saves a set of diagnostic plots for a regression model.
    """
    print(f"Generating plots for {model_name}...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure y_true and y_pred are Series or numpy arrays for consistent plotting
    if not isinstance(y_true, (pd.Series, np.ndarray)):
        y_true = pd.Series(y_true)
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        y_pred = pd.Series(y_pred)
    
    # --- Figure 1: Actual vs. Predicted ---
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, s=50, edgecolor='k')
    # Use min/max of actuals and predictions to define the perfect line range more robustly
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'{model_name}: Actual vs. Predicted Values', fontsize=16)
    plt.xlabel('Actual ipec_pd', fontsize=12)
    plt.ylabel('Predicted ipec_pd', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_ActualVsPredicted.png'), dpi=FIGURE_DPI)
    plt.close()

    # --- Figure 4: Feature Importance (if available) ---
    if feature_importances is not None and feature_names is not None:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False).head(20) # Top 20 features

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
        plt.title(f'{model_name}: Top 20 Feature Importances', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_FeatureImportance.png'), dpi=FIGURE_DPI)
        plt.close()
        
    print(f"Plots for {model_name} saved successfully.")
    gc.collect()






### TUNER SECTION ###
# Re-run these imports to ensure keras_tuner and statsmodels are available
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt # NEW: Keras Tuner for hyperparameter optimization
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np
import pandas as pd # Needed for pd.Series in generate_model_plots
import statsmodels.api as sm # Needed for Q-Q plot

# Define FIGURE_DIR and FIGURE_DPI if they are not already in your current session
# (It's safe to run this again even if they are defined, or uncomment if they error out)
FIGURE_DIR = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/docs/figures'
FIGURE_DPI = 300
os.makedirs(FIGURE_DIR, exist_ok=True) # Ensure directory exists

# RE-DEFINE the generate_model_plots function (it has new plots)
def generate_model_plots(model_name, y_true, y_pred, feature_importances=None, feature_names=None):
    """
    Generates and saves a set of diagnostic plots for a regression model.
    Includes new plots for residuals vs. actual and Q-Q plot.
    """
    print(f"Generating plots for {model_name}...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure y_true and y_pred are Series or numpy arrays for consistent plotting
    if not isinstance(y_true, (pd.Series, np.ndarray)):
        y_true = pd.Series(y_true)
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        y_pred = pd.Series(y_pred)
    
    # --- Figure 1: Actual vs. Predicted ---
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, s=50, edgecolor='k')
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.title(f'{model_name}: Actual vs. Predicted Values', fontsize=16)
    plt.xlabel('Actual ipec_pd', fontsize=12)
    plt.ylabel('Predicted ipec_pd', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_ActualVsPredicted.png'), dpi=FIGURE_DPI)
    plt.close()


    print(f"Plots for {model_name} saved successfully.")
    # No gc.collect() here to prevent conflicts with subsequent calls, let the main script handle it.



# --- Processing Model: Deep Neural Network (DNN) with Keras Tuner ---
print("\n--- Processing Model: Deep Neural Network (DNN) with Keras Tuner ---")

# Convert scaled dataframes to numpy arrays for Keras (ensure these are available from previous cells)
# X_train_scaled, X_test_scaled, y_train, y_test must be defined in your notebook session.
X_train_scaled_np = X_train_scaled.to_numpy()
X_test_scaled_np = X_test_scaled.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Define the model-building function for Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(X_train_scaled_np.shape[1],))) # Input layer with 30 features

    # Tune the number of hidden layers (2 to 4 layers)
    for i in range(hp.Int('num_layers', 2, 4)):
        model.add(
            layers.Dense(
                # Tune number of units (neurons) in the current layer
                hp.Int(f'units_{i}', min_value=64, max_value=512, step=64),
                # Choose activation function
                activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh']),
                # Tune L2 regularization strength
                kernel_regularizer=keras.regularizers.l2(
                    hp.Float(f'l2_lambda_{i}', min_value=1e-5, max_value=1e-2, sampling='log')
                )
            )
        )
        model.add(layers.BatchNormalization()) # Batch Normalization
        # Tune dropout rate for each hidden layer
        model.add(layers.Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)))

    # Output layer
    model.add(layers.Dense(1))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]) # Broader range
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mae', 'mse'])
    return model

# Setup Keras Tuner - Hyperband is efficient for wide search spaces
# Try to increase max_epochs for trials to give the model more time to learn
# and hyperband_iterations to explore more configurations.
tuner = kt.Hyperband(
    build_model,
    objective='val_loss', # Optimize for validation loss
    max_epochs=100,       # Increased max epochs for a trial
    factor=3,            # Reduction factor for Hyperband
    hyperband_iterations=3, # Increased number of full Hyperband iterations
    directory='keras_tuner_dir', # Directory to store results
    project_name='ipec_pd_dnn_tuning_v2', # New project name to avoid conflicts
    overwrite=False # Overwrite previous results in this directory
)

# Callbacks for the Keras Tuner search process
tuner_callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1), # Re-increased patience for tuner to allow more epochs per trial
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1) # Re-increased patience for LR reduction
]

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Keras Tuner search for DNN optimal hyperparameters.")
# Run the hyperparameter search
# This process can take a significant amount of time depending on hyperband_iterations, max_epochs, and your hardware.
tuner.search(X_train_scaled_np, y_train_np,
             epochs=100, # Max epochs per trial as defined in tuner.search
             batch_size=64,
             validation_split=0.1, # Validation split for the tuner
             callbacks=tuner_callbacks,
             verbose=1)
print(f"[{datetime.now().strftime('%H:%M:%S')}] Keras Tuner search complete.")

# Get the best model found by the tuner
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_dnn_model = tuner.get_best_models(num_models=1)[0]

print(f"\nBest DNN Hyperparameters found:\n{best_hp.values}")

# Now, evaluate the best_dnn_model on the final hold-out test set
print("\nEvaluating Best DNN model on the final hold-out test set...")
y_pred_dnn = best_dnn_model.predict(X_test_scaled_np).flatten()

# Calculate ALL final metrics for DNN
mse_dnn = mean_squared_error(y_test_np, y_pred_dnn)
rmse_dnn = np.sqrt(mse_dnn)
mae_dnn = mean_absolute_error(y_test_np, y_pred_dnn)
r2_dnn = r2_score(y_test_np, y_pred_dnn)

# New metrics for DNN (ensure predictions are non-negative for MSLE/RMSLE)
msle_dnn = mean_squared_log_error(y_test_np, np.maximum(0, y_pred_dnn))
rmsle_dnn = np.sqrt(msle_dnn)
medae_dnn = median_absolute_error(y_test_np, y_pred_dnn)
max_err_dnn = max_error(y_test_np, y_pred_dnn)

# MAPE calculation for DNN (handle division by zero)
if 0 in y_test_np:
    non_zero_indices_dnn = y_test_np != 0
    if np.sum(non_zero_indices_dnn) > 0:
        mape_dnn = np.mean(np.abs((y_test_np[non_zero_indices_dnn] - y_pred_dnn[non_zero_indices_dnn]) / y_test_np[non_zero_indices_dnn])) * 100
    else:
        mape_dnn = np.nan
else:
    mape_dnn = np.mean(np.abs((y_test_np - y_pred_dnn) / y_test_np)) * 100

all_results['Simple_DNN'] = {
    'Test_MSE': mse_dnn,
    'Test_RMSE': rmse_dnn,
    'Test_MAE': mae_dnn,
    'Test_R2': r2_dnn,
    'Test_MSLE': msle_dnn,
    'Test_RMSLE': rmsle_dnn,
    'Test_MedAE': medae_dnn,
    'Test_MaxError': max_err_dnn,
    'Test_MAPE': mape_dnn
}
best_estimators['Simple_DNN'] = best_dnn_model # Store the best tuned Keras model

print(f"Simple DNN Test RMSE: {rmse_dnn:.4f}")
print(f"Simple DNN Test MAE: {mae_dnn:.4f}")
print(f"Simple DNN Test R2: {r2_dnn:.4f}")

# Generate and save plots for the DNN model
print(f"Generating and saving plots for Simple_DNN (Prediction Diagnostics)...")
generate_model_plots('Simple_DNN', y_test_np, y_pred_dnn, feature_importances=None, feature_names=None)
print(f"Plots for Simple_DNN (Prediction Diagnostics) saved successfully.")
gc.collect()

# --- Plotting Training History for the BEST DNN (after tuning) ---
# You'll need to manually re-fit the best_dnn_model to get its history if tuner.search()
# doesn't directly store the best run's history. Keras Tuner focuses on the best *model*,
# not necessarily the full history of its final training.
# A common approach is to get the best hyperparameters, build a new model with them,
# and train it once on your full training data to capture its history.

# For simplicity, if you want a history plot, we'll train the best model again
# and capture its history. This is often what you'd report in a thesis.
print("\n--- Training Best DNN Model for History Plot ---")
history_best_dnn = best_dnn_model.fit(X_train_scaled_np, y_train_np,
                                     epochs=tuner.search_space_sizes()['epochs'], # Or a set number like 200
                                     batch_size=64,
                                     validation_split=0.1,
                                     callbacks=[
                                         EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
                                         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
                                     ],
                                     verbose=1)

print("\n--- Generating BEST DNN Training History Plots ---")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plt.figure(figsize=(14, 6))

# Plot Training & Validation MAE
plt.subplot(1, 2, 1)
plt.plot(history_best_dnn.history['mae'], label='Training MAE', color='blue')
plt.plot(history_best_dnn.history['val_mae'], label='Validation MAE', color='green', linestyle='--')
plt.title('BEST DNN Training and Validation Mean Absolute Error', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Absolute Error', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

# Plot Training & Validation Loss (MSE)
plt.subplot(1, 2, 2)
plt.plot(history_best_dnn.history['loss'], label='Training Loss (MSE)', color='red')
plt.plot(history_best_dnn.history['val_loss'], label='Validation Loss (MSE)', color='purple', linestyle='--')
plt.title('BEST DNN Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plot_filepath_history = os.path.join(FIGURE_DIR, f'{timestamp}_BEST_DNN_Training_History.png')
plt.savefig(plot_filepath_history, dpi=FIGURE_DPI)
plt.close()
print(f"BEST DNN Training History Plots saved successfully to: {plot_filepath_history}")








# --- IMPORTANT: EXECUTE THIS ENTIRE CELL IMMEDIATELY AFTER Ctrl+C ON TUNER.SEARCH() ---
# DO NOT RESTART YOUR KERNEL BEFORE RUNNING THIS CELL.
# This cell will save your best model, its hyperparameters, all calculated metrics,
# and generate/save all required plots.

# --- 0. Essential Imports & Setup ---
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
import json
import gc # For garbage collection to free up memory
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, median_absolute_error, max_error
import tensorflow as tf # Required for Keras model operations and loading

# Define FIGURE_DIR and FIGURE_DPI if not already globally defined in your script
# Adjust this path if your 'docs/figures' directory is elsewhere
if 'FIGURE_DIR' not in locals():
    FIGURE_DIR = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/docs/figures'
if 'FIGURE_DPI' not in locals():
    FIGURE_DPI = 300
os.makedirs(FIGURE_DIR, exist_ok=True) # Ensure the directory for figures exists

# Define a directory for saving models and metrics if not already done
if 'MODEL_SAVE_DIR' not in locals():
    MODEL_SAVE_DIR = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/saved_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True) # Ensure the directory for models exists

# --- 1. Define the generate_model_plots function (essential for plotting) ---
# This function must be defined here to ensure it's available.
def generate_model_plots(model_name, y_true, y_pred, feature_importances=None, feature_names=None):
    """
    Generates and saves a set of diagnostic plots for a regression model,
    designed for thesis-quality visualization of skewed data.
    """
    print(f"Generating advanced prediction diagnostic plots for {model_name}...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure y_true and y_pred are numpy arrays for consistent plotting
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Calculate metrics for display on plots
    current_mse = mean_squared_error(y_true, y_pred)
    current_rmse = np.sqrt(current_mse)
    current_mae = mean_absolute_error(y_true, y_pred)
    current_r2 = r2_score(y_true, y_pred)

    # Handle cases where min/max might be problematic for empty arrays
    min_val = min(y_true.min(), y_pred.min()) if y_true.size > 0 and y_pred.size > 0 else 0
    max_val = max(y_true.max(), y_pred.max()) if y_true.size > 0 and y_pred.size > 0 else 1
    
    # --- Figure 1: Actual vs. Predicted (Overall and Zoomed) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8)) # Two subplots for overall and zoom

    # Overall plot (ax1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3, s=30, edgecolor='none', ax=ax1, color='blue')
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_title(f'{model_name}: Actual vs. Predicted Values (Overall)', fontsize=16)
    ax1.set_xlabel('Actual ipec_pd', fontsize=12)
    ax1.set_ylabel('Predicted ipec_pd', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.text(0.05, 0.95, f'RMSE: {current_rmse:.2f}\nMAE: {current_mae:.2f}\nR²: {current_r2:.2f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))


    # Zoomed-in plot (ax2) - Focusing on lower values for better visibility of PD events
    # Adjust zoom_threshold dynamically, but with a cap
    zoom_threshold = min(y_true.max() * 0.1 if y_true.size > 0 else 100, 1000) 
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, s=50, edgecolor='k', ax=ax2, color='blue')
    ax2.plot([0, zoom_threshold], [0, zoom_threshold], 'r--', lw=2, label='Perfect Prediction')
    ax2.set_xlim(-50, zoom_threshold * 1.1)
    ax2.set_ylim(-50, zoom_threshold * 1.1)
    ax2.set_title(f'{model_name}: Actual vs. Predicted Values (Zoomed)', fontsize=16)
    ax2.set_xlabel('Actual ipec_pd', fontsize=12)
    ax2.set_ylabel('Predicted ipec_pd', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_ActualVsPredicted.png'), dpi=FIGURE_DPI)
    plt.close(fig)

    # --- Figure 2: Residuals Plot vs. Predicted (Overall and Zoomed) ---
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Overall residuals (ax1)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.3, s=30, edgecolor='none', ax=ax1, color='green')
    ax1.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax1.set_title(f'{model_name}: Residuals Plot vs. Predicted (Overall)', fontsize=16)
    ax1.set_xlabel('Predicted ipec_pd', fontsize=12)
    ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Zoomed-in residuals (ax2)
    zoom_res_limit = current_rmse * 2 # Heuristic for residual spread, adjust as needed
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, s=50, edgecolor='k', ax=ax2, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax2.set_xlim(min(y_pred.min(), -50) if y_pred.size > 0 else -50, zoom_threshold * 1.1)
    ax2.set_ylim(-zoom_res_limit, zoom_res_limit)
    ax2.set_title(f'{model_name}: Residuals Plot vs. Predicted (Zoomed)', fontsize=16)
    ax2.set_xlabel('Predicted ipec_pd', fontsize=12)
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_ResidualsVsPredicted.png'), dpi=FIGURE_DPI)
    plt.close(fig)

    # --- Figure 3: Distribution of Residuals ---
    plt.figure(figsize=(10, 7))
    sns.histplot(residuals, kde=True, bins=50, color='purple')
    plt.title(f'{model_name}: Distribution of Residuals', fontsize=16)
    plt.xlabel('Residual Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_ResidualsDistribution.png'), dpi=FIGURE_DPI)
    plt.close()

    # --- Figure 4: Feature Importance (if available) ---
    if feature_importances is not None and feature_names is not None and len(feature_importances) > 0:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False).head(20) # Show top 20 features

        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
        plt.title(f'{model_name}: Top 20 Feature Importances', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_FeatureImportance.png'), dpi=FIGURE_DPI)
        plt.close()
        
    # --- Figure 5: Residuals vs. Actual Values (Overall and Zoomed) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Overall residuals vs Actual (ax1)
    sns.scatterplot(x=y_true, y=residuals, alpha=0.3, s=30, edgecolor='none', ax=ax1, color='orange')
    ax1.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax1.set_title(f'{model_name}: Residuals Plot vs. Actual (Overall)', fontsize=16)
    ax1.set_xlabel('Actual ipec_pd', fontsize=12)
    ax1.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Zoomed-in residuals vs Actual (ax2)
    sns.scatterplot(x=y_true, y=residuals, alpha=0.5, s=50, edgecolor='k', ax=ax2, color='orange')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax2.set_xlim(min(y_true.min(), -50) if y_true.size > 0 else -50, zoom_threshold * 1.1)
    ax2.set_ylim(-zoom_res_limit, zoom_res_limit)
    ax2.set_title(f'{model_name}: Residuals Plot vs. Actual (Zoomed)', fontsize=16)
    ax2.set_xlabel('Actual ipec_pd', fontsize=12)
    ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_ResidualsVsActual.png'), dpi=FIGURE_DPI)
    plt.close(fig)

    # --- Figure 6: Q-Q Plot of Residuals (for normality check) ---
    if len(residuals) > 1: # Q-Q plot needs at least 2 data points
        try:
            fig = sm.qqplot(residuals, line='45', fit=True)
            plt.title(f'{model_name}: Quantile-Quantile Plot of Residuals', fontsize=16)
            plt.xlabel('Theoretical Quantiles', fontsize=12)
            plt.ylabel('Sample Quantiles of Residuals', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_QQPlotResiduals.png'), dpi=FIGURE_DPI)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not generate Q-Q plot due to error: {e}")
    else:
        print("Warning: Not enough data points to generate Q-Q plot for residuals.")
        
    print(f"Prediction diagnostic plots for {model_name} saved successfully.")
    gc.collect()


# --- 2. Initialize dictionaries if not already (safeguard) ---
# This ensures these exist even if you restart and only run this cell.
if 'all_results' not in locals():
    all_results = {}
if 'best_estimators' not in locals():
    best_estimators = {}

# --- 3. Retrieve Best Model and Hyperparameters from Tuner ---
# This assumes 'tuner' object is available from the interrupted search
print("\n--- Retrieving Best Model and Hyperparameters from Tuner ---")
best_hp = None
best_dnn_model = None
try:
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_dnn_model = tuner.get_best_models(num_models=1)[0]
    print(f"\nBest DNN Hyperparameters Found:\n{best_hp.values}")
except Exception as e:
    print(f"ERROR: Failed to retrieve best model/hyperparameters from tuner: {e}")
    print("This often happens if you restarted the kernel or 'tuner.search()' did not run long enough to save any trials.")
    print("Cannot proceed with saving model, metrics, and plots. Please ensure 'tuner' object is active.")
    raise # Re-raise the exception to stop execution and alert the user

# --- 4. Validate Data Availability ---
print("\n--- Validating Data Availability for Evaluation & Retraining ---")
data_error = False
if 'X_test_scaled_np' not in locals() or not isinstance(X_test_scaled_np, np.ndarray) or X_test_scaled_np.size == 0:
    print("CRITICAL ERROR: X_test_scaled_np not found or is empty. Cannot perform model evaluation.")
    data_error = True
if 'y_test_np' not in locals() or not isinstance(y_test_np, np.ndarray) or y_test_np.size == 0:
    print("CRITICAL ERROR: y_test_np not found or is empty. Cannot perform model evaluation.")
    data_error = True
if 'X_train_scaled_np' not in locals() or not isinstance(X_train_scaled_np, np.ndarray) or X_train_scaled_np.size == 0:
    print("WARNING: X_train_scaled_np not found or is empty. Training history plots may be skipped.")
    # Not critical to stop, but important for history plot
if 'y_train_np' not in locals() or not isinstance(y_train_np, np.ndarray) or y_train_np.size == 0:
    print("WARNING: y_train_np not found or is empty. Training history plots may be skipped.")
    # Not critical to stop, but important for history plot

if data_error:
    raise RuntimeError("Missing critical test data (X_test_scaled_np, y_test_np). Please ensure your data loading and preprocessing cells have been run successfully.")

# --- 5. Make Predictions on Test Set ---
print("\n--- Making Predictions with Best DNN model on Test Set ---")
y_pred_dnn = best_dnn_model.predict(X_test_scaled_np).flatten()

# --- 6. Calculate ALL Evaluation Metrics for DNN ---
print("\n--- Calculating Final Evaluation Metrics ---")
y_true_eval = y_test_np.flatten() if y_test_np.ndim > 1 else y_test_np
y_pred_eval = y_pred_dnn # Already flattened

# Ensure y_true_eval and y_pred_eval are arrays of floats for metric functions
y_true_eval = y_true_eval.astype(float)
y_pred_eval = y_pred_eval.astype(float)

mse_dnn = mean_squared_error(y_true_eval, y_pred_eval)
rmse_dnn = np.sqrt(mse_dnn)
mae_dnn = mean_absolute_error(y_true_eval, y_pred_eval)
r2_dnn = r2_score(y_true_eval, y_pred_eval)

# For MSLE and RMSLE, values must be non-negative. Filter out any potential negatives.
# If ipec_pd can be 0, MSLE/RMSLE are problematic as log(0) is undefined.
# We filter for positive values for MSLE/RMSLE for robustness.
# Adding a small epsilon for predictions if they are too close to zero but should be positive
y_pred_msle = np.maximum(y_pred_eval, 1e-9) # Ensure predictions are slightly positive for log

positive_indices_for_msle = (y_true_eval > 0) & (y_pred_msle >= 0)
if np.any(positive_indices_for_msle):
    msle_dnn = mean_squared_log_error(y_true_eval[positive_indices_for_msle], y_pred_msle[positive_indices_for_msle])
    rmsle_dnn = np.sqrt(msle_dnn)
else:
    msle_dnn = np.nan
    rmsle_dnn = np.nan
    print("Warning: No positive actual values found for MSLE/RMSLE calculation after filtering.")

medae_dnn = median_absolute_error(y_true_eval, y_pred_eval)
max_err_dnn = max_error(y_true_eval, y_pred_eval)

# Calculate MAPE, handling division by zero for actual values
non_zero_actual_indices = y_true_eval != 0
if np.any(non_zero_actual_indices):
    mape_dnn = np.mean(np.abs((y_true_eval[non_zero_actual_indices] - y_pred_eval[non_zero_actual_indices]) / y_true_eval[non_zero_actual_indices])) * 100
else:
    mape_dnn = np.nan # Or some other appropriate value if all y_test_np are zero
    print("Warning: All actual values are zero; MAPE cannot be calculated meaningfully.")

print(f"Final DNN Metrics:")
print(f"  Test MSE: {mse_dnn:.4f}")
print(f"  Test RMSE: {rmse_dnn:.4f}")
print(f"  Test MAE: {mae_dnn:.4f}")
print(f"  Test R2: {r2_dnn:.4f}")
print(f"  Test MSLE: {msle_dnn:.4f}" if not np.isnan(msle_dnn) else "  Test MSLE: N/A (no positive actuals)")
print(f"  Test RMSLE: {rmsle_dnn:.4f}" if not np.isnan(rmsle_dnn) else "  Test RMSLE: N/A (no positive actuals)")
print(f"  Test MedAE: {medae_dnn:.4f}")
print(f"  Test Max Error: {max_err_dnn:.4f}")
print(f"  Test MAPE: {mape_dnn:.2f}%" if not np.isnan(mape_dnn) else "  Test MAPE: N/A (all actuals are zero)")

# --- 7. Store Results in Dictionaries and Save Metrics to JSON ---
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Use a single timestamp for all saved files from this run

all_results['Simple_DNN'] = {
    'Test_MSE': mse_dnn,
    'Test_RMSE': rmse_dnn,
    'Test_MAE': mae_dnn,
    'Test_R2': r2_dnn,
    'Test_MSLE': msle_dnn,
    'Test_RMSLE': rmsle_dnn,
    'Test_MedAE': medae_dnn,
    'Test_MaxError': max_err_dnn,
    'Test_MAPE': mape_dnn
}
best_estimators['Simple_DNN'] = best_dnn_model
print("\nFinal metrics stored in 'all_results' and best model stored in 'best_estimators'.")

# Save all_results dictionary to a JSON file
metrics_filename = f'dnn_evaluation_metrics_{current_timestamp}.json'
metrics_filepath = os.path.join(MODEL_SAVE_DIR, metrics_filename)
try:
    with open(metrics_filepath, 'w') as f:
        json.dump(all_results['Simple_DNN'], f, indent=4) # Save just the DNN metrics
    print(f"DNN evaluation metrics saved to: {metrics_filepath}")
except Exception as e:
    print(f"ERROR: Could not save evaluation metrics to JSON: {e}")


# --- 8. Generate and Save Prediction Diagnostic Plots ---
generate_model_plots('Simple_DNN', y_test_np, y_pred_dnn, feature_importances=None, feature_names=None)
print("Prediction diagnostic plots saved successfully.")
gc.collect()

# --- 9. Retrain Best DNN Model to get a full training history for plotting ---
# This creates the 'history_best_dnn' object that contains loss and MAE per epoch
print("\n--- Retraining Best DNN Model (for training history plot) ---")
history_best_dnn = None
if 'X_train_scaled_np' not in locals() or not isinstance(X_train_scaled_np, np.ndarray) or X_train_scaled_np.size == 0:
    print("WARNING: X_train_scaled_np not found or is empty. Skipping training history plot.")
elif 'y_train_np' not in locals() or not isinstance(y_train_np, np.ndarray) or y_train_np.size == 0:
    print("WARNING: y_train_np not found or is empty. Skipping training history plot.")
else:
    try:
        # Clone the model to get a fresh training history
        # Alternatively, you can use `best_dnn_model.optimizer.variables = []` if you want to reuse and clear optimizer state
        # For a clean history, re-building or cloning is often safer.
        # However, for simply getting the history from the best model, just calling .fit again works.
        # Ensure your target metric (e.g., 'val_loss') is available from the model's compile step.
        
        # Keras models can be re-fit directly; previous history is overwritten.
        # Using a small batch_size from best_hp if available, otherwise default to 64
        batch_size_val = best_hp.get('batch_size', 64) if best_hp else 64 
        history_best_dnn = best_dnn_model.fit(X_train_scaled_np, y_train_np,
                                             epochs=best_hp.get('tuner/epochs', 100) if best_hp else 100,
                                             batch_size=batch_size_val,
                                             validation_split=0.1, # Keep consistency with tuner's validation split
                                             callbacks=[
                                                 tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
                                                 tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)
                                             ],
                                             verbose=0) # Set verbose=0 to avoid lengthy output
        print("Best DNN model retrained to capture full training history.")
    except Exception as e:
        print(f"ERROR during retraining for history plot: {e}")
        print("Training history plots will not be generated.")
        history_best_dnn = None


# --- 11. Save the Best Keras Model (Architecture and Weights) ---
print("\n--- Saving the Best Keras Model ---")
model_save_path = os.path.join(MODEL_SAVE_DIR, f'best_dnn_model_{current_timestamp}.keras') # Recommended format for TF 2.x
try:
    best_dnn_model.save(model_save_path)
    print(f"Best DNN model saved to: {model_save_path}")
except Exception as e:
    print(f"ERROR: Could not save the best Keras model: {e}")

# --- 12. Save the Best Hyperparameters to a JSON file ---
if best_hp: # Only save if best_hp was successfully retrieved
    print("\n--- Saving Best Hyperparameters to File ---")
    config_filename = f'best_dnn_hyperparameters_{current_timestamp}.json'
    config_filepath = os.path.join(MODEL_SAVE_DIR, config_filename)

    try:
        with open(config_filepath, 'w') as f:
            json.dump(best_hp.values, f, indent=4)
        print(f"Best DNN Hyperparameters saved to: {config_filepath}")
    except Exception as e:
        print(f"ERROR: Could not save best hyperparameters to file: {e}")
else:
    print("Skipping saving hyperparameters as best_hp object was not retrieved.")

print("\n--- ALL REQUESTED POST-INTERRUPTION OPERATIONS COMPLETE. ---")
print("You can now safely restart your kernel if desired.")
gc.collect() # Final garbage collection





























# --- 7. Model Comparison Figures ---
def plot_metrics_comparison(results_dict, figure_dir):
    """
    Generates and saves comparison plots for all models across various metrics.
    Displays all models and highlights/orders for top performance.
    """
    print("\n--- 7. Generating Model Comparison Figures ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert results to a DataFrame for easier plotting
    results_df = pd.DataFrame(results_dict).T # Transpose to have models as rows
    results_df.index.name = 'Model'

    metrics_to_plot = {
        'Test_RMSE': {'title': 'Root Mean Squared Error (RMSE)', 'smaller_is_better': True},
        'Test_MAE': {'title': 'Mean Absolute Error (MAE)', 'smaller_is_better': True},
        'Test_R2': {'title': 'R-squared (R2 Score)', 'smaller_is_better': False},
        'Test_RMSLE': {'title': 'Root Mean Squared Log Error (RMSLE)', 'smaller_is_better': True},
        'Test_MedAE': {'title': 'Median Absolute Error (MedAE)', 'smaller_is_better': True},
        'Test_MaxError': {'title': 'Max Error', 'smaller_is_better': True},
        'Test_MAPE': {'title': 'Mean Absolute Percentage Error (MAPE)', 'smaller_is_better': True}
    }

    for metric, props in metrics_to_plot.items():
        if metric not in results_df.columns:
            print(f"Skipping plot for {metric}: not found in results.")
            continue

        plt.figure(figsize=(12, 7))
        
        # Sort based on metric
        ascending = props['smaller_is_better']
        sorted_df = results_df.sort_values(by=metric, ascending=ascending)
        
        # Create a specific color palette to highlight best models
        colors = sns.color_palette('viridis', len(sorted_df))
        if ascending: # Smaller is better, best is first (darker color usually)
             plot_colors = colors[::-1] # Reverse for lighter colors for worse models
        else: # Larger is better, best is first
            plot_colors = colors


        sns.barplot(x=sorted_df.index, y=metric, data=sorted_df, palette=plot_colors)
        
        plt.title(f'Model Comparison: {props["title"]}', fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        # Add value labels on top of bars
        for index, row in sorted_df.iterrows():
            plt.text(row.name, row[metric], f'{row[metric]:.4f}', color='black', ha="center", va='bottom', fontsize=9)

        plt.savefig(os.path.join(figure_dir, f'{timestamp}_Model_Comparison_{metric}.png'), dpi=FIGURE_DPI)
        plt.close()
        print(f"Comparison plot for {metric} saved.")
    gc.collect()

# Call the comparison plotting function after all models are processed

plot_metrics_comparison(all_results, FIGURE_DIR)