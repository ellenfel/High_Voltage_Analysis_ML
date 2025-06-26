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
    
    # --- Figure 1: Actual vs. Predicted ---
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, s=50, edgecolor='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
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


# --- 4. Model Selection, Tuning, and Cross-Validation ---
print("\n--- 4. Starting Model Training, Tuning, and Evaluation ---")
# Define models and their hyperparameter grids for GridSearchCV
models_and_params = {
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    'XGBRegressor': {
        'model': XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist', objective='reg:squarederror'),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    }
}


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
        verbose=3  # KEY CHANGE: Set to 2 for per-fit feedback. Use 3 for even more detail.
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
    
    # Calculate final metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    all_results[name] = {'Test_MSE': mse, 'Test_RMSE': rmse, 'Test_MAE': mae, 'Test_R2': r2}

    # Generate and save plots for the best model
    print(f"Generating and saving plots for {name}...")
    feature_importances = best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None
    generate_model_plots(name, y_test, y_pred, feature_importances, X_train.columns)
    print(f"Plots for {name} saved successfully.")

    gc.collect()

print("\n\n--- 5. Final Model Performance Summary ---")
for name, metrics in all_results.items():
    print(f"\nModel: {name}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("-" * 30)

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
print("Check your figures directory for all generated plots.")


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
