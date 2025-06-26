# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn for enhanced plots
import warnings
import gc
warnings.filterwarnings('ignore')


# reading data (this part is from your existing code)
df = pd.read_csv('/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML/data/df_cleaned.csv')

# data exploration, create figures (this is where we add the plotting code)
print(f"DataFrame 'df' loaded with shape: {df.shape}")
# columns in DataFrame
print(f"Columns in DataFrame: {df.columns.tolist()}")
print("-" * 30)


# %% [markdown]
# ### 1. Histograms for Key Numerical Features
# These plots will show the distribution of individual important numerical features, helping to understand their ranges, central tendencies, and spread.

# %%
# Set plot style for better aesthetics
sns.set_style("whitegrid")

# Select a few representative columns for histograms
# Ensure these columns exist in your DataFrame before plotting
hist_columns = ['IR_A_value', 'current_a_value', 'temp_value', 'hum_value']
# Filter out any columns that might not exist in your actual df for robustness
existing_hist_columns = [col for col in hist_columns if col in df.columns]

if len(existing_hist_columns) > 0:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle('Histograms of Key Numerical Features', fontsize=16)
    axes = axes.flatten() # Flatten the 2x2 array of axes for easy iteration

    for i, col in enumerate(existing_hist_columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette("viridis")[i])
        axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')
        axes[i].set_xlabel(col.replace("_", " ").title())
        axes[i].set_ylabel('Frequency')

    # Hide any unused subplots if the number of columns is not a perfect square
    for j in range(len(existing_hist_columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlapping
    plt.savefig('histograms_key_features.png', dpi=300)
    plt.close()
    print("Generated 'histograms_key_features.png'")
else:
    print("None of the specified histogram columns found in DataFrame.")


# %% [markdown]
# ### 2. Pair Plot of Selected Numerical Features
# A pair plot helps visualize relationships between multiple numerical variables and their individual distributions. Given your large dataset size, a sample will be used for performance.

# %%
# Prepare data for pair plot - selecting a subset of features for readability
# Select a subset of columns for the pair plot that might show interesting correlations
selected_columns_for_pairplot = ['IR_A_value', 'current_a_value', 'voltage_a_value', 'temp_value', 'hum_value']

# Ensure the selected columns exist in the DataFrame
existing_columns_for_pairplot = [col for col in selected_columns_for_pairplot if col in df.columns]

if len(existing_columns_for_pairplot) > 1: # Need at least 2 columns for a meaningful pairplot
    # Using Seaborn's pairplot on a sample for performance with large datasets
    # Adjust the sample size (n) based on your system's memory and performance
    sample_size = min(10000, df.shape[0]) # Use max 10k rows or full df if smaller
    df_sample_for_pairplot = df[existing_columns_for_pairplot].sample(n=sample_size, random_state=42)

    print(f"Generating pair plot from a sample of {sample_size} rows...")
    pair_plot = sns.pairplot(df_sample_for_pairplot, diag_kind='kde', plot_kws={'alpha':0.6})
    pair_plot.fig.suptitle('Pair Plot of Selected Numerical Features (Sampled Data)', y=1.02, fontsize=16) # Adjust title position
    plt.savefig('pair_plot_selected_features.png', dpi=300)
    plt.close()
    print("Generated 'pair_plot_selected_features.png'")
elif len(existing_columns_for_pairplot) > 0:
    print("Not enough columns to generate a pair plot. Need at least two numerical columns.")
else:
    print("None of the specified columns for pair plot found in DataFrame.")


# Clean up memory
gc.collect()