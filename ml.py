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
# 6. Custom R² Metric for Keras
# ==============================================================================
import tensorflow as tf

def r2_keras(y_true, y_pred):
    """Custom R² metric for Keras training monitoring"""
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res/(SS_tot + tf.keras.backend.epsilon()))

# ==============================================================================
# 7. Deep Neural Network Experimentation
# ==============================================================================
print("\n" + "="*60)
print("--- Starting Enhanced Deep Neural Network Experimentation ---")
print("="*60)

# --- Hyperparameters ---
# Wider and deeper architecture for better feature learning
dnn_params = {
    "num_layers": 5,
    "units": [1024, 768, 512, 256, 128],
    "activations": ["relu", "relu", "relu", "relu", "relu"],
    "l2_lambdas": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
    "dropouts": [0.3, 0.25, 0.2, 0.15, 0.1],
    "learning_rate": 0.0005, # Slightly lower for more stable training
    "epochs": 25,
    "batch_size": 512, # Larger batch size for stable gradients
    "early_stopping_patience": 7,
    "reduce_lr_patience": 4
}

# --- Build the Model ---
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

# --- Compile Model with Advanced Optimizer and R² Metric ---
optimizer = keras.optimizers.Adam(
    learning_rate=dnn_params["learning_rate"],
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)
dnn_model.compile(optimizer=optimizer, loss='mse', metrics=['mae', r2_keras])
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

start_time = datetime.now()  # Add start time tracking
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

# Store results (matching format of other models)
training_time_dnn = (datetime.now() - start_time).total_seconds()
dnn_results = {
    'rmse': rmse,  # Changed key to match other models
    'mae': mae,    # Changed key to match other models
    'r2': r2,      # Changed key to match other models
    'training_time': training_time_dnn,  # Added missing training time
    'predictions': y_pred,
    'history': history.history
}

print(f"\n--- Deep Neural Network Performance ---")
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R-squared: {r2:.4f}")

all_model_results["Deep Neural Network"] = dnn_results


# ==============================================================================
# 7.5. Optimized DNN with Pre-Selected Hyperparameters
# ==============================================================================
print("\n" + "="*60)
print("--- Training Optimized DNN with Pre-Selected Hyperparameters ---")
print("="*60)

# --- Optimized Hyperparameters (from hyperparameter tuning results) ---
# NOTE: Interpreted 4 layers based on provided units_0 through units_3
optimized_dnn_params = {
    "num_layers": 4,
    "units": [512, 448, 320, 192],
    "activations": ["relu", "relu", "relu", "tanh"],
    "l2_lambdas": [5.005115883734732e-05, 0.005496054400493782, 0.0008832783459212954, 0.00013494772704967807],
    "dropouts": [0.2, 0.30000000000000004, 0.4, 0.30000000000000004],
    "learning_rate": 0.001,
    "epochs": 12, # From "tuner/epochs"
    "batch_size": 64,  # Using a common, robust batch size
    "early_stopping_patience": 5, # Reasonable default
    "reduce_lr_patience": 3 # Reasonable default
}

# --- Build the Optimized Model ---
optimized_dnn_model = keras.Sequential([layers.Input(shape=(X_train_scaled.shape[1],))])

# Add layers based on optimized parameters
for i in range(optimized_dnn_params["num_layers"]):
    optimized_dnn_model.add(layers.Dense(
        optimized_dnn_params["units"][i],
        activation=optimized_dnn_params["activations"][i],
        kernel_regularizer=regularizers.l2(optimized_dnn_params["l2_lambdas"][i]),
        kernel_initializer='he_normal'
    ))
    optimized_dnn_model.add(layers.BatchNormalization())
    optimized_dnn_model.add(layers.Dropout(optimized_dnn_params["dropouts"][i]))

# Output layer
optimized_dnn_model.add(layers.Dense(1, kernel_initializer='glorot_normal'))

# --- Compile Optimized Model ---
optimizer_optimized = keras.optimizers.Adam(
    learning_rate=optimized_dnn_params["learning_rate"],
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)
optimized_dnn_model.compile(optimizer=optimizer_optimized, loss='mse', metrics=['mae', r2_keras])

print("Optimized DNN Architecture:")
optimized_dnn_model.summary()

# --- Enhanced Training with Callbacks ---
callbacks_optimized = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=optimized_dnn_params["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=optimized_dnn_params["reduce_lr_patience"],
        min_lr=1e-7,
        verbose=1
    )
]

# --- Train Optimized Model ---
print(f"Training optimized DNN with {optimized_dnn_params['num_layers']} layers...")
print(f"Architecture: {' -> '.join(map(str, optimized_dnn_params['units']))} -> 1")

start_time_optimized = datetime.now()
history_optimized = optimized_dnn_model.fit(
    X_train_scaled, y_train,
    batch_size=optimized_dnn_params["batch_size"],
    epochs=optimized_dnn_params["epochs"],
    validation_split=0.2,
    callbacks=callbacks_optimized,
    verbose=1
)

# --- Evaluate Optimized Model ---
y_pred_optimized = optimized_dnn_model.predict(X_test_scaled, batch_size=optimized_dnn_params["batch_size"])
y_pred_optimized = y_pred_optimized.flatten()

# Ensure non-negative predictions
y_pred_optimized[y_pred_optimized < 0] = 0

# Calculate metrics
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
rmse_optimized = np.sqrt(mse_optimized)
mae_optimized = mean_absolute_error(y_test, y_pred_optimized)
r2_optimized = r2_score(y_test, y_pred_optimized)

# Store results
training_time_optimized = (datetime.now() - start_time_optimized).total_seconds()
optimized_dnn_results = {
    'rmse': rmse_optimized,
    'mae': mae_optimized,
    'r2': r2_optimized,
    'training_time': training_time_optimized,
    'predictions': y_pred_optimized,
    'history': history_optimized.history,
    'model': optimized_dnn_model # Storing model object for consistency
}

print(f"\n--- Optimized Deep Neural Network Performance ---")
print(f"RMSE: {rmse_optimized:.4f}, MAE: {mae_optimized:.4f}, R-squared: {r2_optimized:.4f}")
print(f"Training completed in {training_time_optimized:.2f} seconds.")

# Add to results dictionary
all_model_results["Optimized DNN"] = optimized_dnn_results


# ==============================================================================
# 8. Visualization of DNN Training
# ==============================================================================

# Create figure with constrained layout for better spacing
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

# Custom color palette (colorblind-friendly)
train_color = '#1f77b4' # Professional blue
val_color = '#d62728' # Distinct red
grid_color = '#f0f0f0' # Light gray grid
background_color = 'white'

# Apply global styling
plt.rcParams.update({
    'font.family': 'serif', # Thesis-appropriate font
    'font.size': 12, # Base font size
    'axes.labelpad': 12, # Axis label padding
    'axes.edgecolor': 'black', # Axis edge color
    'axes.linewidth': 0.8, # Axis line thickness
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
         linestyle='--', # Dashed for validation
         dash_capstyle='round')

# Formatting
ax1.set_title('DNN Training Loss',
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
         linestyle='--', # Dashed for validation
         dash_capstyle='round')

# Formatting
ax2.set_title('DNN Training MAE',
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

# ===== R² Plot =====
ax3.plot(history.history['r2_keras'],
         label='Training R²',
         color=train_color,
         linewidth=3,
         alpha=0.9,
         solid_capstyle='round')

ax3.plot(history.history['val_r2_keras'],
         label='Validation R²',
         color=val_color,
         linewidth=3,
         alpha=0.9,
         linestyle='--', # Dashed for validation
         dash_capstyle='round')

# Formatting
ax3.set_title('DNN Training R²',
              fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('Epoch', fontsize=12, labelpad=10)
ax3.set_ylabel('R² Score', fontsize=12, labelpad=10)
ax3.legend(frameon=True, framealpha=0.9,
           facecolor=background_color, edgecolor='gray')
ax3.grid(True, color=grid_color, linestyle='-', linewidth=0.7)

# Set background and spines
ax3.set_facecolor(background_color)
for spine in ax3.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(0.8)

# ===== Final Adjustments =====
# Save in multiple formats for thesis publication
figure_base = os.path.join(FIGURE_DIR, 'enhanced_dnn_training_history_with_r2')

# High-resolution PNG
plt.savefig(figure_base + '.png', dpi=600, bbox_inches='tight')

# Vector formats for publications
plt.savefig(figure_base + '.pdf', bbox_inches='tight', transparent=True)
plt.savefig(figure_base + '.svg', bbox_inches='tight', transparent=True)

print(f"Training history figures saved to: {figure_base}.[png/svg]")

plt.show()

# Garbage collection
gc.collect()

# ==============================================================================
# 9. Final Results Comparison and Export
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

# ==============================================================================
# 10. Model Comparison Visualizations
# ==============================================================================

# --- 10.1 Model Performance Comparison Bar Chart ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# Define professional color palette
# Increased number of colors to accommodate the new model
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4E6E58', '#5A4E6E']

# RMSE Comparison
bars1 = ax1.bar(results_df['Model'], results_df['RMSE'], color=colors, alpha=0.8)
ax1.set_title('Model Performance: RMSE', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.tick_params(axis='x', rotation=45, labelsize=9) # Adjusted label size for readability
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# MAE Comparison
bars2 = ax2.bar(results_df['Model'], results_df['MAE'], color=colors, alpha=0.8)
ax2.set_title('Model Performance: MAE', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('MAE', fontsize=12)
ax2.tick_params(axis='x', rotation=45, labelsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# R² Comparison
bars3 = ax3.bar(results_df['Model'], results_df['R2'], color=colors, alpha=0.8)
ax3.set_title('Model Performance: R² Score', fontsize=14, fontweight='bold', pad=15)
ax3.set_ylabel('R² Score', fontsize=12)
ax3.tick_params(axis='x', rotation=45, labelsize=9)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(1.0, results_df['R2'].max() * 1.1)) # Adjust y-lim dynamically

# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Save comparison chart
comparison_path = os.path.join(FIGURE_DIR, 'model_performance_comparison')
plt.savefig(comparison_path + '.png', dpi=600, bbox_inches='tight')
plt.savefig(comparison_path + '.svg', bbox_inches='tight', transparent=True)
print(f"Model comparison chart saved to: {comparison_path}.[png/svg]")
plt.show()


# --- 10.2 Prediction vs Actual Scatter Plot (Best Model) ---
best_model_name = results_df.iloc[0]['Model']
best_model_results = all_model_results[best_model_name]

fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)

# Get predictions from best model, using the stored predictions for DNNs
if 'predictions' in best_model_results:
    y_pred_best = best_model_results['predictions']
else:
    y_pred_best = best_model_results['model'].predict(X_test_scaled)
    y_pred_best[y_pred_best < 0] = 0  # Ensure non-negative

# Create scatter plot
scatter = ax.scatter(y_test, y_pred_best, alpha=0.6, c='#2E86AB', s=50, edgecolors='white', linewidth=0.5)

# Perfect prediction line (y=x)
min_val = min(min(y_test), min(y_pred_best))
max_val = max(max(y_test), max(y_pred_best))
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')

# Formatting
ax.set_title(f'Prediction vs Actual Values - {best_model_name}', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Actual Values', fontsize=14)
ax.set_ylabel('Predicted Values', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add R² annotation
r2_text = f'$R^2 = {best_model_results["r2"]:.4f}$' # Using LaTeX for R^2
ax.text(0.05, 0.95, r2_text, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save scatter plot
scatter_path = os.path.join(FIGURE_DIR, f'prediction_vs_actual_{best_model_name.lower().replace(" ", "_")}')
plt.savefig(scatter_path + '.png', dpi=600, bbox_inches='tight')
plt.savefig(scatter_path + '.svg', bbox_inches='tight', transparent=True)
print(f"Prediction vs actual plot saved to: {scatter_path}.[png/svg]")
plt.show()

# --- 10.3 Residuals Plot (Best Model) ---
residuals = y_test - y_pred_best

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

# Residuals vs Predicted
ax1.scatter(y_pred_best, residuals, alpha=0.6, c='#A23B72', s=50, edgecolors='white', linewidth=0.5)
ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
ax1.set_title(f'Residuals vs Predicted - {best_model_name}', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Predicted Values', fontsize=12)
ax1.set_ylabel('Residuals', fontsize=12)
ax1.grid(True, alpha=0.3)

# Residuals histogram
ax2.hist(residuals, bins=50, density=True, alpha=0.7, color='#F18F01', edgecolor='black', linewidth=0.5) # Use density=True for norm overlay
ax2.set_title(f'Residuals Distribution - {best_model_name}', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Residuals', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Add normal distribution overlay
from scipy import stats
mu, sigma = stats.norm.fit(residuals)
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
y_norm = stats.norm.pdf(x_norm, mu, sigma)
ax2.plot(x_norm, y_norm, 'r-', linewidth=2, alpha=0.8, label=f'Normal fit ($\\mu={mu:.3f}, \\sigma={sigma:.3f}$)') # LaTeX for mu and sigma
ax2.legend()

# Save residuals plot
residuals_path = os.path.join(FIGURE_DIR, f'residuals_analysis_{best_model_name.lower().replace(" ", "_")}')
plt.savefig(residuals_path + '.png', dpi=600, bbox_inches='tight')
plt.savefig(residuals_path + '.svg', bbox_inches='tight', transparent=True)
print(f"Residuals analysis plot saved to: {residuals_path}.[png/svg]")
plt.show()

print("\n" + "="*80)
print("All thesis-quality figures have been generated and saved!")
print("="*80)
