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

