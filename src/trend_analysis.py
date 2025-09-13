# %%


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- Path Constants ---
BASE_DIR = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML'
FIGURE_DIR = os.path.join(BASE_DIR, 'docs/figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'docs/results')

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Load predictions CSV ---
csv_files = []
for root, dirs, files in os.walk(os.path.join(BASE_DIR, 'data/results')):
    for file in files:
        if file == 'predictions.csv':
            csv_files.append(os.path.join(root, file))

if not csv_files:
    raise FileNotFoundError("No predictions.csv found under data/results/")

latest_csv = sorted(csv_files)[-1]
predictions_df = pd.read_csv(latest_csv)

# --- Downsample to 500 points ---
n_points = 500
if len(predictions_df) > n_points:
    indices = np.linspace(0, len(predictions_df) - 1, n_points, dtype=int)
    predictions_df = predictions_df.iloc[indices].reset_index(drop=True)

# --- Select only actual and Optimized DNN ---
plot_df = predictions_df[['actual', 'Optimized DNN_pred']]

# --- Colors ---
colors = {
    'actual': "#B61F1F",  # Black
    'Optimized DNN_pred': '#1f77b4'  # Brown
}
labels = {
    'actual': 'Actual Values',
    'Optimized DNN_pred': 'Optimized DNN'
}

# --- Plot Trend ---
fig, ax = plt.subplots(figsize=(16, 8))

# Actual
ax.plot(plot_df.index, plot_df['actual'],
        color=colors['actual'], linewidth=3, label=labels['actual'], alpha=0.9)

# Optimized DNN
ax.plot(plot_df.index, plot_df['Optimized DNN_pred'],
        color=colors['Optimized DNN_pred'], linewidth=2, linestyle='--', alpha=0.8, label=labels['Optimized DNN_pred'])

# --- Metrics ---
r2 = r2_score(plot_df['actual'], plot_df['Optimized DNN_pred'])
rmse = np.sqrt(mean_squared_error(plot_df['actual'], plot_df['Optimized DNN_pred']))
mae = mean_absolute_error(plot_df['actual'], plot_df['Optimized DNN_pred'])

ax.set_title(f'Actual vs Optimized DNN Predictions\nR²={r2:.3f}, RMSE={rmse:.2f}, MAE={mae:.2f}',
             fontweight='bold', fontsize=16, pad=20)
ax.set_xlabel('Sample Index', fontweight='bold', fontsize=14)
ax.set_ylabel('Target Value', fontweight='bold', fontsize=14)
ax.legend(loc='upper left', frameon=True)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# --- Save High-Res Figure ---
trend_path = os.path.join(FIGURE_DIR, 'trend_actual_vs_dnn')
plt.savefig(trend_path + '.png', dpi=600, bbox_inches='tight')
plt.savefig(trend_path + '.pdf', bbox_inches='tight')
print(f"Trend analysis saved to: {trend_path}.[png/pdf]")

plt.show()
