# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
BASE_DIR = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML'
FIGURE_DIR = os.path.join(BASE_DIR, 'docs/figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(BASE_DIR, 'data/df_ml_ready.csv'))

# List of signals to include - edit directly here (comment/uncomment as needed)
signals_to_include = [

    'current_a_value',
    'current_b_value',
    'current_c_value',
    'lid_switch_value',
    'mppt_batary_voltage_value',
    'pressure_value',
    'pulse_a_value',
    'pulse_b_value',
    'pulse_c_value',
    'temp_value',
    'voltage_a_value',
    'voltage_b_value',
    'voltage_c_value',
    'water_image_name_value',
    'device_profile_I-Link Box EA',
    'device_name_JB2',
    'device_name_Substation1',
    'device_name_Substation2',
    'ipec_pd',
]

print(f"Using {len(signals_to_include)} signals for correlation map")

# Filter dataframe to selected signals
df_filtered = df[signals_to_include]

# Calculate correlation matrix
corr_matrix = df_filtered.corr()

# Create correlation map
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix,
            cmap='RdBu_r',
            center=0,
            square=True,
            cbar_kws={'shrink': 0.5})

plt.title('Full Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'full_correlation_map.png'), dpi=300, bbox_inches='tight')
plt.show()
