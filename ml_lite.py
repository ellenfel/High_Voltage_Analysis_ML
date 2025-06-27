# %%
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

    # --- Figure 2: Residuals Plot ---
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, s=50, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.title(f'{model_name}: Residuals Plot', fontsize=16)
    plt.xlabel('Predicted ipec_pd', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_Residuals.png'), dpi=FIGURE_DPI)
    plt.close()

    # --- Figure 3: Distribution of Residuals ---
    plt.figure(figsize=(10, 7))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title(f'{model_name}: Distribution of Residuals', fontsize=16)
    plt.xlabel('Residual Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_ResidualsDistribution.png'), dpi=FIGURE_DPI)
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


# --- 4. Model Selection, Tuning, and Cross-Validation (Original Limited Parameters) ---
print("\n--- 4. Starting Model Training, Tuning, and Evaluation (Original Limited Parameters) ---")
# Define models and their hyperparameter grids for GridSearchCV
# THESE GRIDS ARE RESTORED TO THEIR ORIGINAL LIMITED CONFIGURATION FOR FASTER RUNTIME.
models_and_params = {
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [50, 100],  
            'max_depth': [10, 20],      
            'min_samples_leaf': [1, 5]  
        }
    },
    'XGBRegressor': {
        'model': XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist', objective='reg:squarederror'),
        'params': {
            'n_estimators': [50, 100],     
            'learning_rate': [0.1],        
            'max_depth': [5],             
            'subsample': [0.8]            
        }
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [50, 100],     
            'learning_rate': [0.1],         
            'max_depth': [3]                
        }
    }
}



###FOUNDATIONAL CODE FOR K-FOLD CROSS-VALIDATION AND GRIDSEARCHCV

# Setup K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"Using K-Fold Cross-Validation with {kf.get_n_splits()} splits.")

# Store results
all_results = {}
best_estimators = {}

for name, config in models_and_params.items():
    print(f"\n--- Processing Model: {name} ---")
    
    # GridSearchCV handles the cross-validation
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=kf,
        scoring='neg_root_mean_squared_error', # Optimize for RMSE
        n_jobs=-1, # Use all available cores
        verbose=3  # Set to 2 for per-fit feedback.
    )
    
    # Fit on the training data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting GridSearchCV for {name}. You will now see progress for each fit.")
    grid_search.fit(X_train_scaled, y_train)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] GridSearchCV for {name} complete.")
    
    # Store the best estimator
    best_model = grid_search.best_estimator_
    best_estimators[name] = best_model
    
    print(f"\nBest parameters found for {name}: {grid_search.best_params_}")
    print(f"Best CV RMSE score for {name}: {-grid_search.best_score_:.4f}")

    # Evaluate the best model on the hold-out test set
    print(f"\nEvaluating best {name} model on the final hold-out test set...")
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate ALL final metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # New metrics
    # MSLE requires non-negative inputs. Clip predictions to 0 if they go negative.
    msle = mean_squared_log_error(y_test, np.maximum(0, y_pred))
    rmsle = np.sqrt(msle)
    medae = median_absolute_error(y_test, y_pred)
    max_err = max_error(y_test, y_pred)
    
    # MAPE calculation (handle division by zero if y_test contains 0)
    # A common approach for MAPE: if y_true is zero, the error is infinite or undefined.
    # We'll use a robust calculation that handles zeros gracefully.
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    # Handle cases where y_test might have zero values for MAPE
    if 0 in y_test.values: # Check if there are any zero values in y_test
        non_zero_indices = y_test != 0
        if np.sum(non_zero_indices) > 0:
            mape = np.mean(np.abs((y_test[non_zero_indices] - y_pred[non_zero_indices]) / y_test[non_zero_indices])) * 100
        else:
            mape = np.nan # Or some other indicator if all y_test are zero
    else:
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100


    all_results[name] = {
        'Test_MSE': mse,
        'Test_RMSE': rmse,
        'Test_MAE': mae,
        'Test_R2': r2,
        'Test_MSLE': msle,    # New
        'Test_RMSLE': rmsle,  # New
        'Test_MedAE': medae,  # New
        'Test_MaxError': max_err, # New
        'Test_MAPE': mape     # New
    }

    # Generate and save plots for the best model
    print(f"Generating and saving plots for {name}...")
    feature_importances = best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None
    generate_model_plots(name, y_test, y_pred, feature_importances, X_train.columns)
    print(f"Plots for {name} saved successfully.")

    gc.collect()
















# --- Added: Simple Deep Neural Network (DNN) for potentially more accuracy ---
print("\n--- Processing Model: Simple Deep Neural Network (DNN) ---")

# Convert scaled dataframes to numpy arrays for Keras
X_train_scaled_np = X_train_scaled.to_numpy()
X_test_scaled_np = X_test_scaled.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Define the DNN model for a thesis, focusing on regularization
# 'kernel_regularizer' applies L2 regularization (weight decay) to the layer's weights.
# BatchNormalization helps stabilize and accelerate training, and also acts as a mild regularizer.
# Dropout regularizes by randomly setting a fraction of input units to 0 at each update step during training.
l2_lambda = 0.001 # A common starting point for L2 regularization strength

dnn_model = keras.Sequential([
    # Input Layer explicitly defined for clarity, shape corresponds to 30 columns of X
    keras.Input(shape=(X_train_scaled_np.shape[1],)), # X has 30 columns

    # First Hidden Layer
    layers.Dense(256, kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.BatchNormalization(), # Normalize activations of the previous layer
    layers.Activation('relu'),   # Apply activation function
    layers.Dropout(0.4),         # Increased dropout rate

    # Second Hidden Layer
    layers.Dense(128, kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.4),         # Increased dropout rate

    # Third Hidden Layer
    layers.Dense(64, kernel_regularizer=keras.regularizers.l2(l2_lambda)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),         # Slightly lower dropout on smaller layer

    # Output Layer for Regression
    layers.Dense(1)
])

# Define a custom learning rate schedule or use ReduceLROnPlateau
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Explicitly set learning rate

# Compile the model
dnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

# Display model summary
dnn_model.summary()

# Add Callbacks for better training and potential speed-up
# Reduced patience for EarlyStopping to stop training sooner when overfitting starts
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1) # Reduced patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1) # Reduced patience for LR reduction

# Train the model
print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting DNN training with regularization.")
history = dnn_model.fit(X_train_scaled_np, y_train_np,
                        epochs=200, # Max epochs, EarlyStopping will manage
                        batch_size=64,
                        validation_split=0.1, # Use a portion of training data for validation
                        callbacks=[early_stopping, reduce_lr], # Apply callbacks
                        verbose=1)
print(f"[{datetime.now().strftime('%H:%M:%S')}] DNN training complete.")

# --- NEW: Plotting Training History for DNN (like your examples) ---
print("\n--- Generating DNN Training History Plots ---")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

plt.figure(figsize=(14, 6)) # Adjusted figure size for better readability

# Plot Training & Validation MAE (as your 'accuracy' proxy for regression)
plt.subplot(1, 2, 1)
plt.plot(history.history['mae'], label='Training MAE', color='blue')
plt.plot(history.history['val_mae'], label='Validation MAE', color='green', linestyle='--')
plt.title('DNN Training and Validation Mean Absolute Error', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Mean Absolute Error', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

# Plot Training & Validation Loss (MSE)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss (MSE)', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='purple', linestyle='--')
plt.title('DNN Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plot_filepath = os.path.join(FIGURE_DIR, f'{timestamp}_DNN_Training_History.png')
plt.savefig(plot_filepath, dpi=FIGURE_DPI)
plt.close()
print("DNN Training History Plots saved successfully.")
# --- END NEW ---

# Evaluate the DNN model
print("\nEvaluating Simple DNN model on the final hold-out test set...")
# Predict on the test set, then flatten the output as it comes as a 2D array (samples, 1)
y_pred_dnn = dnn_model.predict(X_test_scaled_np).flatten()

# Calculate ALL final metrics for DNN
mse_dnn = mean_squared_error(y_test_np, y_pred_dnn)
rmse_dnn = np.sqrt(mse_dnn)
mae_dnn = mean_absolute_error(y_test_np, y_pred_dnn)
r2_dnn = r2_score(y_test_np, y_pred_dnn)

# New metrics for DNN
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
best_estimators['Simple_DNN'] = dnn_model # Store the trained Keras model


print(f"Simple DNN Test RMSE: {rmse_dnn:.4f}")
print(f"Simple DNN Test MAE: {mae_dnn:.4f}")
print(f"Simple DNN Test R2: {r2_dnn:.4f}")

# Generate and save plots for the DNN model
print(f"Generating and saving plots for Simple_DNN (Prediction Diagnostics)...")
# Feature importances are not directly available for DNNs in the same way, so pass None
generate_model_plots('Simple_DNN', y_test_np, y_pred_dnn, feature_importances=None, feature_names=None)
print(f"Plots for Simple_DNN (Prediction Diagnostics) saved successfully.")
gc.collect()


print("\n\n--- 5. Final Model Performance Summary ---")
for name, metrics in all_results.items():
    print(f"\nModel: {name}")
    for metric, value in metrics.items():
        # Handle NaN values for display
        if pd.isna(value):
            print(f"  {metric}: NaN")
        else:
            print(f"  {metric}: {value:.4f}")
    print("-" * 30)









# --- Added: Simple Deep Neural Network (DNN) for potentially more accuracy ---
print("\n--- Processing Model: Simple Deep Neural Network (DNN) ---")

# Convert scaled dataframes to numpy arrays for Keras
X_train_scaled_np = X_train_scaled.to_numpy()
X_test_scaled_np = X_test_scaled.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Define the DNN model - slightly more complex for potentially better results
dnn_model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train_scaled_np.shape[1],)), # Increased neurons
    layers.Dropout(0.3), # Slightly increased dropout
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1) # Output layer for regression
])

# Define a custom learning rate schedule or use ReduceLROnPlateau
# For simplicity, let's use Adam with default learning rate or slightly adjusted
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Explicitly set learning rate

# Compile the model
dnn_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

# Add Callbacks for better training and potential speed-up
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1) # Increased patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.00001, verbose=1) # Reduce LR on plateau

# Train the model
print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting DNN training.")
history = dnn_model.fit(X_train_scaled_np, y_train_np,
                        epochs=200, # Increased max epochs, EarlyStopping will manage
                        batch_size=64,
                        validation_split=0.1, # Use a portion of training data for validation
                        callbacks=[early_stopping, reduce_lr], # Apply callbacks
                        verbose=1)
print(f"[{datetime.now().strftime('%H:%M:%S')}] DNN training complete.")

# Evaluate the DNN model
print("\nEvaluating Simple DNN model on the final hold-out test set...")
# Predict on the test set, then flatten the output as it comes as a 2D array (samples, 1)
y_pred_dnn = dnn_model.predict(X_test_scaled_np).flatten()

# Calculate ALL final metrics for DNN
mse_dnn = mean_squared_error(y_test_np, y_pred_dnn)
rmse_dnn = np.sqrt(mse_dnn)
mae_dnn = mean_absolute_error(y_test_np, y_pred_dnn)
r2_dnn = r2_score(y_test_np, y_pred_dnn)

# New metrics for DNN
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
best_estimators['Simple_DNN'] = dnn_model # Store the trained Keras model

print(f"Simple DNN Test RMSE: {rmse_dnn:.4f}")
print(f"Simple DNN Test MAE: {mae_dnn:.4f}")
print(f"Simple DNN Test R2: {r2_dnn:.4f}")

# Generate and save plots for the DNN model
print(f"Generating and saving plots for Simple_DNN...")
# Feature importances are not directly available for DNNs in the same way, so pass None
generate_model_plots('Simple_DNN', y_test_np, y_pred_dnn, feature_importances=None, feature_names=None)
print(f"Plots for Simple_DNN saved successfully.")
gc.collect()


print("\n\n--- 5. Final Model Performance Summary ---")
for name, metrics in all_results.items():
    print(f"\nModel: {name}")
    for metric, value in metrics.items():
        # Handle NaN values for display
        if pd.isna(value):
            print(f"  {metric}: NaN")
        else:
            print(f"  {metric}: {value:.4f}")
    print("-" * 30)

# Cell 2: Generate and save DNN Training History Plots
print("\n--- Generating DNN Training History Plots ---")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Check if 'history' object exists (from a previous DNN model.fit() run)
if 'history' not in locals():
    print("Error: 'history' object not found. Please ensure the DNN training cell has been run.")
else:
    plt.figure(figsize=(14, 6)) # Adjusted figure size for better readability

    # Plot Training & Validation MAE (as your 'accuracy' proxy for regression)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['mae'], label='Training MAE', color='blue')
    plt.plot(history.history['val_mae'], label='Validation MAE', color='green', linestyle='--')
    plt.title('DNN Training and Validation Mean Absolute Error', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot Training & Validation Loss (MSE)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss (MSE)', color='red')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='purple', linestyle='--')
    plt.title('DNN Training and Validation Loss', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Mean Squared Error)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plot_filepath = os.path.join(FIGURE_DIR, f'{timestamp}_DNN_Training_History.png')
    plt.savefig(plot_filepath, dpi=FIGURE_DPI)
    plt.close()
    print(f"DNN Training History Plots saved successfully to: {plot_filepath}")



















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

    # --- Figure 2: Residuals Plot ---
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, s=50, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.title(f'{model_name}: Residuals Plot', fontsize=16)
    plt.xlabel('Predicted ipec_pd', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_Residuals.png'), dpi=FIGURE_DPI)
    plt.close()

    # --- Figure 3: Distribution of Residuals ---
    plt.figure(figsize=(10, 7))
    sns.histplot(residuals, kde=True, bins=50)
    plt.title(f'{model_name}: Distribution of Residuals', fontsize=16)
    plt.xlabel('Residual Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_ResidualsDistribution.png'), dpi=FIGURE_DPI)
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

    # --- NEW Figure 5: Residuals vs. Actual Values (for homoscedasticity check) ---
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_true, y=residuals, alpha=0.5, s=50, edgecolor='k')
    plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.title(f'{model_name}: Residuals vs. Actual Values', fontsize=16)
    plt.xlabel('Actual ipec_pd', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_ResidualsVsActual.png'), dpi=FIGURE_DPI)
    plt.close()

    # --- NEW Figure 6: Q-Q Plot of Residuals (for normality check) ---
    # Need to import statsmodels.api as sm for this
    fig = sm.qqplot(residuals, line='45', fit=True)
    plt.title(f'{model_name}: Quantile-Quantile Plot of Residuals', fontsize=16)
    plt.xlabel('Theoretical Quantiles', fontsize=12)
    plt.ylabel('Sample Quantiles of Residuals', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'{timestamp}_{model_name}_QQPlotResiduals.png'), dpi=FIGURE_DPI)
    plt.close(fig) # Close the figure object returned by qqplot to prevent display in notebook
        
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

# --- 6. Correlation Analysis (from your original code) ---
print("\n--- 6. Feature Correlation Analysis ---")
print("Calculating correlation matrix...")
correlation_matrix = X.corr()

plt.figure(figsize=(22, 20))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".1f")
plt.title('Correlation Matrix of Features', fontsize=20)
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
corr_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(FIGURE_DIR, f'{corr_timestamp}_FeatureCorrelationMatrix.png'), dpi=FIGURE_DPI)
plt.close()
print("Correlation heatmap saved.")
gc.collect()

print("\n--- Pipeline Finished ---")
print("Check your figures directory for all generated plots, including comparison plots.")

# Reminder: In a Jupyter environment, all_results and best_estimators
# will be accessible in memory after execution.
# For persistence beyond the current session, you would use joblib.dump or similar.


### --- TODO for Advanced Steps (as per your list) --- ###
#
# 1. ERROR ANALYSIS:
#    - You now have residuals plots. Look for patterns:
#      - Is the error variance constant (homoscedasticity)? Or does it change with the predicted value?
#      - Are there clusters of points with high errors? You could investigate these samples in your original data.
#
# 2. ENSEMBLE MODELING (ADVANCED):
#    - You could combine your best models using a VotingRegressor.
#      from sklearn.ensemble import VotingRegressor
#      # Example:
#      # vote_reg = VotingRegressor(estimators=[('xgb', best_estimators['XGBRegressor']), ('rf', best_estimators['RandomForestRegressor'])])
#      # vote_reg.fit(X_train_scaled, y_train)
#      # This often yields a more robust, generalized model.
#
# 3. MODEL EXPLAINABILITY (XAI):
#    - For your thesis, using SHAP or LIME is highly recommended.
#      import shap
#      # Example with your best XGBoost model:
#      # explainer = shap.TreeExplainer(best_estimators['XGBRegressor'])
#      # shap_values = explainer.shap_values(X_test_scaled)
#      # shap.summary_plot(shap_values, X_test_scaled, show=False)
#      # plt.savefig(os.path.join(FIGURE_DIR, 'SHAP_Summary_Plot.png'), dpi=FIGURE_DPI)
#      # This explains not just *which* features are important, but *how* they impact predictions.

