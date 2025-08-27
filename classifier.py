# %%

# ==============================================================================
# 1. Imports
# ==============================================================================
# Improvement: Consider lazy imports for unused models to reduce startup time.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import os
from datetime import datetime
from sklearn.svm import LinearSVC 

# Scikit-learn and ML Frameworks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix, roc_curve, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ==============================================================================
# 2. Configuration
# ==============================================================================
# Sets up project paths, creates necessary directories, and configures plotting styles.
# Suppresses warnings for cleaner output during model training.
# Improvement: Consider using environment variables or config files for paths.
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
# Loads data, creates binary classification target, splits into train/test sets, and scales features.
# Creates dummy data if the real dataset doesn't exist for demonstration purposes.
print("--- Loading and Preparing Data ---")
# Create a dummy CSV file if it doesn't exist for demonstration purposes
dummy_csv_path = os.path.join(BASE_DIR, 'data/df_ml_ready.csv')
if not os.path.exists(dummy_csv_path):
    print("Creating a dummy df_ml_ready.csv for demonstration...")
    os.makedirs(os.path.dirname(dummy_csv_path), exist_ok=True)
    dummy_data = {f'feature_{i+1}': np.random.randn(10000) for i in range(10)}
    dummy_data['ipec_pd'] = np.random.choice([0, 0, 0, 0, 1, 5, 10], 10000, p=[0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)


df = pd.read_csv(dummy_csv_path)
print(f"Initial df.shape: {df.shape}")

# Ensure the target column 'ipec_pd' exists
if 'ipec_pd' not in df.columns:
    raise ValueError("Target column 'ipec_pd' not found in the DataFrame.")

# Create binary classification target: 1 for PD, 0 for no PD
df['pd_binary'] = (df['ipec_pd'] != 0).astype(int)
print(f"\nClass distribution:")
print(df['pd_binary'].value_counts())
print(f"\nClass proportions:")
print(df['pd_binary'].value_counts(normalize=True))

# Drop the original multi-class PD column
df = df.drop(columns=['ipec_pd'])

# Split into features (X) and target (y)
X = df.drop(columns=['pd_binary'])
y = df['pd_binary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames to retain column names for feature importance plots
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
print("\nFeatures scaled using StandardScaler.")

# ==============================================================================
# 4. Universal Model Training and Evaluation Function
# ==============================================================================
# Defines a reusable function to train any classifier and evaluate its performance.
# Handles models with and without predict_proba method (like LinearSVC).
# Improvement: Add cross-validation and early stopping for better model selection.
def train_and_evaluate_classifier(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains a classification model, evaluates its performance, and returns the results.
    """
    print(f"\n--- Training {model_name} ---")
    start_time = datetime.now()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    # THIS IS THE KEY: Check if the model has 'predict_proba' before calling it.
    # If not, set the probabilities to None.
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = None # LinearSVC will result in None

    training_time = (datetime.now() - start_time).total_seconds()
    print(f"Training completed in {training_time:.2f} seconds.")

    # Calculate classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"--- {model_name} Performance ---")
    print(f"Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return {
        "model": model,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "training_time": training_time,
        "predictions": y_pred,
        "predictions_proba": y_pred_proba # This will be None for LinearSVC
    }

# ==============================================================================
# 5. Train and Evaluate Classification Models
# ==============================================================================
# Defines multiple ML models with appropriate hyperparameters and trains them.
# Uses a diverse set of algorithms: SVM, tree-based methods, and gradient boosting.
all_model_results = {}

# --- Define Models ---
models_to_train = {
    # Replace the slow SVC with the highly optimized LinearSVC
    "Linear SVM": LinearSVC(random_state=42, dual="auto"), 
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
    "XGBoost": XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
}

# --- Train all models ---
for name, model in models_to_train.items():
    results = train_and_evaluate_classifier(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name=name)
    all_model_results[name] = results

# ==============================================================================
# 6. Final Results Comparison and Export
# ==============================================================================
# Compiles all model results into a comparison DataFrame and exports to CSV.
# Ranks models by F1-Score to identify the best performing classifier.
# Improvement: Add statistical significance tests to compare model performance.
print("\n--- Comparing Final Classification Model Performance ---")
results_df = pd.DataFrame({
    'Model': list(all_model_results.keys()),
    'Accuracy': [res['accuracy'] for res in all_model_results.values()],
    'F1-Score': [res['f1'] for res in all_model_results.values()],
    'Precision': [res['precision'] for res in all_model_results.values()],
    'Recall': [res['recall'] for res in all_model_results.values()],
    'Training Time (s)': [res['training_time'] for res in all_model_results.values()]
})

# Sort by F1-Score in descending order for clear comparison
results_df = results_df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)

print(results_df.to_string(index=False))

# --- Save results to CSV ---
output_path = os.path.join(RESULTS_DIR, 'classification_model_comparison_results.csv')
results_df.to_csv(output_path, index=False)
print(f"\nResults successfully saved to: {output_path}")

# ==============================================================================
# 7. Classification Model Analysis and Visualization
# ==============================================================================
# Creates confusion matrix as DataFrame and generates ROC curve comparison plot.
# Focuses on essential visualizations for model performance assessment.
# Improvement: Add precision-recall curves for imbalanced datasets.

# --- 7.1 Confusion Matrix for Best Model as DataFrame ---
best_model_name = results_df.iloc[0]['Model']
best_model_results = all_model_results[best_model_name]
y_pred_best = best_model_results['predictions']

# Create confusion matrix as DataFrame and save to CSV
cm = confusion_matrix(y_test, y_pred_best)
cm_df = pd.DataFrame(cm, 
                     index=['Actual_No_PD', 'Actual_PD'], 
                     columns=['Predicted_No_PD', 'Predicted_PD'])

cm_csv_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{best_model_name.lower().replace(" ", "_")}.csv')
cm_df.to_csv(cm_csv_path)
print(f"Confusion matrix for '{best_model_name}' saved to: {cm_csv_path}")
print(f"\nConfusion Matrix - {best_model_name}:")
print(cm_df)

# --- 7.2 Classification Report ---
print(f"\n--- Detailed Classification Report for {best_model_name} ---")
report = classification_report(y_test, y_pred_best, target_names=['No PD (0)', 'PD Present (1)'])
print(report)

# --- 7.3 ROC Curve and AUC Score Comparison ---
print("\n--- Generating ROC Curve Comparison ---")
plt.figure(figsize=(10, 8))

for name, results in all_model_results.items():
    if results['predictions_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, results['predictions_proba'])
        auc_score = roc_auc_score(y_test, results['predictions_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

# Plotting the random classifier line
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)')

# Aesthetics
plt.title('ROC Curve Comparison for All Models', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True)
plt.tight_layout()

# Save ROC curve plot (PNG only)
roc_path = os.path.join(FIGURE_DIR, 'roc_curve_comparison.png')
plt.savefig(roc_path, dpi=FIGURE_DPI, bbox_inches='tight')
print(f"ROC curve comparison plot saved to: {roc_path}")
plt.show()


# ==============================================================================
# 8. Script Completion
# ==============================================================================
# Cleanup memory and signals successful completion of the analysis pipeline.
# Garbage collection helps free up memory after processing large datasets.
# Improvement: Add logging system to track execution time and resource usage.
print("\n" + "="*60)
print("--- Analysis Script Completed Successfully ---")
print("="*60)
gc.collect()