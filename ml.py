# %%
# ==============================================================================
# 1. Imports
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import os
from datetime import datetime

# Scikit-learn and ML Frameworks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import keras
from keras import layers, regularizers

# ==============================================================================
# 2. Configuration
# ==============================================================================
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Path Constants ---
BASE_DIR = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML'
FIGURE_DIR = os.path.join(BASE_DIR, 'docs/figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'docs/results')

# Create directories if they don't exist
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Plotting Style ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
FIGURE_DPI = 300 # High resolution for thesis

# ==============================================================================
# 3. Data Loading and Preparation
# ==============================================================================
print("--- Loading and Preparing Data ---")
df = pd.read_csv(os.path.join(BASE_DIR, 'data/df_ml_ready.csv'))
print(f"Initial df.shape: {df.shape}")

# Ensure the target column 'ipec_pd' exists and move it to the end
if 'ipec_pd' in df.columns:
    target_column = df.pop('ipec_pd')
    df['ipec_pd'] = target_column
else:
    raise ValueError("Target column 'ipec_pd' not found in the DataFrame.")

# Split into features (X) and target (y)
X = df.drop(columns=['ipec_pd'])
y = df['ipec_pd']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames to retain column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
print("Features scaled using StandardScaler.")

# ==============================================================================
# 4. Universal Model Training and Evaluation Function
# ==============================================================================
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model", is_keras_model=False, keras_epochs=12):
    """
    Trains a model, evaluates its performance, and returns the results.
    """
    print(f"\n--- Training {model_name} ---")
    start_time = datetime.now()

    if is_keras_model:
        # Keras models require numpy arrays and have a different fit/predict signature
        model.fit(
            X_train.to_numpy(), y_train.to_numpy(),
            epochs=keras_epochs,
            batch_size=64,
            validation_split=0.1,
            verbose=0
        )
        y_pred = model.predict(X_test.to_numpy()).flatten()
    else:
        # Scikit-learn compatible models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds.")

    # Post-processing: ensure predictions are non-negative
    y_pred[y_pred < 0] = 0

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"--- {model_name} Performance ---")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}")

    return {"model": model, "rmse": rmse, "mae": mae, "r2": r2, "training_time": training_time}

# ==============================================================================
# 5. Train Traditional ML Models
# ==============================================================================
all_model_results = {}

# --- Define Models ---
models_to_train = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1),
    "LightGBM": LGBMRegressor(objective='regression', n_estimators=100, random_state=42, n_jobs=-1)
}

# --- Train Scikit-learn compatible models ---
for name, model in models_to_train.items():
    results = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name=name)
    all_model_results[name] = results

# ==============================================================================
# 6. Deep Neural Network Experimentation - Enhanced Architecture
# ==============================================================================
print("\n" + "="*60)
print("--- Starting Enhanced Deep Neural Network Experimentation ---")
print("="*60)

# --- Enhanced Hyperparameters ---
# Wider and deeper architecture for better feature learning
dnn_params = {
    "num_layers": 5,
    "units": [1024, 768, 512, 256, 128],
    "activations": ["relu", "relu", "relu", "relu", "relu"],
    "l2_lambdas": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
    "dropouts": [0.3, 0.25, 0.2, 0.15, 0.1],
    "learning_rate": 0.0005,  # Slightly lower for more stable training
    "epochs": 25,
    "batch_size": 512,  # Larger batch size for stable gradients
    "early_stopping_patience": 7,
    "reduce_lr_patience": 4
}

# --- Build Enhanced Model ---
dnn_model = keras.Sequential([layers.Input(shape=(X_train_scaled.shape[1],))])

# First layer - wider for initial feature extraction
dnn_model.add(layers.Dense(
    dnn_params["units"][0],
    activation=dnn_params["activations"][0],
    kernel_regularizer=regularizers.l2(dnn_params["l2_lambdas"][0]),
    kernel_initializer='he_normal'
))
dnn_model.add(layers.BatchNormalization())
dnn_model.add(layers.Dropout(dnn_params["dropouts"][0]))

# Hidden layers with progressive dimension reduction
for i in range(1, dnn_params["num_layers"]):
    dnn_model.add(layers.Dense(
        dnn_params["units"][i],
        activation=dnn_params["activations"][i],
        kernel_regularizer=regularizers.l2(dnn_params["l2_lambdas"][i]),
        kernel_initializer='he_normal'
    ))
    dnn_model.add(layers.BatchNormalization())
    dnn_model.add(layers.Dropout(dnn_params["dropouts"][i]))

# Output layer
dnn_model.add(layers.Dense(1, kernel_initializer='glorot_normal'))

# --- Compile Model with Advanced Optimizer ---
optimizer = keras.optimizers.Adam(
    learning_rate=dnn_params["learning_rate"],
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)
dnn_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
dnn_model.summary()

# --- Enhanced Training with Callbacks ---
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=dnn_params["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=dnn_params["reduce_lr_patience"],
        min_lr=1e-7,
        verbose=1
    )
]

# --- Train Model ---
print(f"Training enhanced DNN with {dnn_params['num_layers']} layers...")
print(f"Architecture: {' -> '.join(map(str, dnn_params['units']))} -> 1")

history = dnn_model.fit(
    X_train_scaled, y_train,
    batch_size=dnn_params["batch_size"],
    epochs=dnn_params["epochs"],
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# --- Evaluate Model ---
y_pred = dnn_model.predict(X_test_scaled, batch_size=dnn_params["batch_size"])
y_pred = y_pred.flatten()

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Store results
dnn_results = {
    'RMSE': rmse,
    'MAE': mae,
    'R-squared': r2,
    'predictions': y_pred,
    'history': history.history
}

print(f"\n--- Enhanced Deep Neural Network Performance ---")
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}")

all_model_results["Enhanced Deep Neural Network"] = dnn_results

# ==============================================================================
# 6.5 Enhanced Visualization of DNN Training History (Thesis Quality)
# ==============================================================================

# Create figure with constrained layout for better spacing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), 
                               constrained_layout=True)

# Custom color palette (colorblind-friendly)
train_color = '#1f77b4'   # Professional blue
val_color = '#d62728'     # Distinct red
grid_color = '#f0f0f0'    # Light gray grid
background_color = 'white'

# Apply global styling
plt.rcParams.update({
    'font.family': 'serif',          # Thesis-appropriate font
    'font.size': 12,                 # Base font size
    'axes.labelpad': 12,             # Axis label padding
    'axes.edgecolor': 'black',       # Axis edge color
    'axes.linewidth': 0.8,           # Axis line thickness
})

# ===== Loss Plot =====
# Smooth lines with increased thickness and custom styles
ax1.plot(history.history['loss'], 
         label='Training Loss', 
         color=train_color,
         linewidth=3,
         alpha=0.9,
         solid_capstyle='round')

ax1.plot(history.history['val_loss'], 
         label='Validation Loss', 
         color=val_color,
         linewidth=3,
         alpha=0.9,
         linestyle='--',  # Dashed for validation
         dash_capstyle='round')

# Formatting
ax1.set_title('Enhanced DNN Training Loss', 
             fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Epoch', fontsize=12, labelpad=10)
ax1.set_ylabel('Loss', fontsize=12, labelpad=10)
ax1.legend(frameon=True, framealpha=0.9, 
          facecolor=background_color, edgecolor='gray')
ax1.grid(True, color=grid_color, linestyle='-', linewidth=0.7)

# Set background and spines
ax1.set_facecolor(background_color)
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.8)

# ===== MAE Plot =====
ax2.plot(history.history['mae'], 
         label='Training MAE', 
         color=train_color,
         linewidth=3,
         alpha=0.9,
         solid_capstyle='round')

ax2.plot(history.history['val_mae'], 
         label='Validation MAE', 
         color=val_color,
         linewidth=3,
         alpha=0.9,
         linestyle='--',  # Dashed for validation
         dash_capstyle='round')

# Formatting
ax2.set_title('Enhanced DNN Training MAE', 
             fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Epoch', fontsize=12, labelpad=10)
ax2.set_ylabel('MAE', fontsize=12, labelpad=10)
ax2.legend(frameon=True, framealpha=0.9, 
          facecolor=background_color, edgecolor='gray')
ax2.grid(True, color=grid_color, linestyle='-', linewidth=0.7)

# Set background and spines
ax2.set_facecolor(background_color)
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.8)

# ===== Final Adjustments =====
# Set consistent y-axis scaling if needed
# min_loss = min(min(history.history['loss']), min(history.history['val_loss']))
# max_loss = max(max(history.history['loss']), max(history.history['val_loss']))
# ax1.set_ylim(min_loss * 0.95, max_loss * 1.05)

# Save in multiple formats for thesis publication
figure_base = os.path.join(FIGURE_DIR, 'enhanced_dnn_training_history')

# High-resolution PNG
plt.savefig(figure_base + '.png', dpi=600, bbox_inches='tight')

# Vector formats for publications
plt.savefig(figure_base + '.pdf', bbox_inches='tight', transparent=True)
plt.savefig(figure_base + '.svg', bbox_inches='tight', transparent=True)

print(f"Training history figures saved to: {figure_base}.[png/pdf/svg]")

plt.show()

# Garbage collection
gc.collect()

# ==============================================================================
# 7. Final Results Comparison and Export
# ==============================================================================
print("\n--- Comparing Final Model Performance ---")
results_df = pd.DataFrame({
    'Model': list(all_model_results.keys()),
    'RMSE': [res['rmse'] for res in all_model_results.values()],
    'MAE': [res['mae'] for res in all_model_results.values()],
    'R2': [res['r2'] for res in all_model_results.values()],
    'Training Time (s)': [res['training_time'] for res in all_model_results.values()]
})

# Sort by R2 in descending order for clear comparison
results_df = results_df.sort_values(by='R2', ascending=False).reset_index(drop=True)

print(results_df.to_string(index=False))

# --- Save results to CSV ---
output_path = os.path.join(RESULTS_DIR, 'model_comparison_results.csv')
results_df.to_csv(output_path, index=False)
print(f"\nResults successfully saved to: {output_path}")