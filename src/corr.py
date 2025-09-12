# %%

# ==============================================================================
# Correlation Analysis Script for Research Paper
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# Configuration
# ==============================================================================
warnings.filterwarnings('ignore')

# --- Path Constants ---
BASE_DIR = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML'
FIGURE_DIR = os.path.join(BASE_DIR, 'docs/figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'docs/results')

# Create directories if they don't exist
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Plotting Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_DPI = 300

# Professional color palettes for correlation analysis
CORRELATION_CMAP = 'RdBu_r'  # Red-Blue diverging colormap
CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# ==============================================================================
# Data Loading and Preparation
# ==============================================================================
print("--- Loading Data for Correlation Analysis ---")
df = pd.read_csv(os.path.join(BASE_DIR, 'data/df_ml_ready.csv'))
print(f"Dataset shape: {df.shape}")

# Ensure target column exists
if 'ipec_pd' not in df.columns:
    raise ValueError("Target column 'ipec_pd' not found in the DataFrame.")

print(f"Features: {df.shape[1]-1}")
print(f"Target variable: ipec_pd")
print(f"Sample size: {df.shape[0]:,}")

# ==============================================================================
# Correlation Matrix Calculation
# ==============================================================================
def calculate_correlations(data, method='pearson'):
    """
    Calculate correlation matrix with statistical significance
    """
    print(f"Calculating {method} correlations...")
    
    if method == 'pearson':
        corr_matrix = data.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = data.corr(method='spearman')
    else:
        raise ValueError("Method must be 'pearson' or 'spearman'")
    
    # Calculate p-values for significance testing
    n_features = len(data.columns)
    p_values = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                if method == 'pearson':
                    _, p_val = pearsonr(data.iloc[:, i], data.iloc[:, j])
                else:
                    _, p_val = spearmanr(data.iloc[:, i], data.iloc[:, j])
                p_values[i, j] = p_val
    
    p_values_df = pd.DataFrame(p_values, 
                               index=data.columns, 
                               columns=data.columns)
    
    return corr_matrix, p_values_df

# Calculate Pearson correlations
pearson_corr, pearson_p_values = calculate_correlations(df, method='pearson')

# Calculate Spearman correlations (for non-linear relationships)
spearman_corr, spearman_p_values = calculate_correlations(df, method='spearman')

print(f"Correlation matrices calculated successfully")

# ==============================================================================
# Target Variable Correlation Analysis
# ==============================================================================
def analyze_target_correlations(corr_matrix, target_col='ipec_pd', top_n=15):
    """
    Analyze correlations with target variable
    """
    print(f"Analyzing correlations with target variable: {target_col}")
    
    # Get correlations with target variable
    target_corr = corr_matrix[target_col].drop(target_col)
    
    # Sort by absolute correlation
    target_corr_abs = target_corr.abs().sort_values(ascending=False)
    
    print(f"\nTop {top_n} features correlated with {target_col}:")
    print("-" * 50)
    
    results = []
    for i, (feature, abs_corr) in enumerate(target_corr_abs.head(top_n).items()):
        actual_corr = target_corr[feature]
        results.append({
            'rank': i + 1,
            'feature': feature,
            'correlation': actual_corr,
            'abs_correlation': abs_corr
        })
        print(f"{i+1:2d}. {feature:<25} | r = {actual_corr:7.4f} (|r| = {abs_corr:.4f})")
    
    return pd.DataFrame(results)

# Analyze target correlations for both methods
pearson_target_results = analyze_target_correlations(pearson_corr, 'ipec_pd')
spearman_target_results = analyze_target_correlations(spearman_corr, 'ipec_pd')

# ==============================================================================
# Publication-Quality Correlation Heatmaps
# ==============================================================================
def create_correlation_heatmap(corr_matrix, p_values=None, title="Correlation Matrix", 
                             figsize=(16, 14), annot_size=8, significance_level=0.05,
                             save_name="correlation_heatmap"):
    """
    Create a publication-quality correlation heatmap
    """
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for non-significant correlations if p-values provided
    mask = np.zeros_like(corr_matrix, dtype=bool)
    if p_values is not None:
        mask = p_values > significance_level
        # Keep diagonal (self-correlations)
        np.fill_diagonal(mask, False)
    
    # Generate heatmap
    im = ax.imshow(corr_matrix.values, cmap=CORRELATION_CMAP, aspect='auto', 
                   vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr_matrix.index, fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=12)
    
    # Add correlation values as text annotations
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            if not (p_values is not None and mask[i, j]):
                text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.7 else 'black'
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', color=text_color, 
                       fontsize=annot_size, weight='bold')
            else:
                # Mark non-significant correlations
                ax.text(j, i, 'n.s.', ha='center', va='center', 
                       color='gray', fontsize=annot_size-2, style='italic')
    
    # Styling
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(-.5, len(corr_matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(corr_matrix.index), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    
    # Save in multiple formats
    base_path = os.path.join(FIGURE_DIR, save_name)
    plt.savefig(base_path + '.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(base_path + '.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(base_path + '.svg', bbox_inches='tight', transparent=True)
    
    print(f"Correlation heatmap saved to: {base_path}.[png/pdf/svg]")
    plt.show()
    
    return fig, ax

# Create full correlation matrices (may be large - consider subset if too many features)
n_features = df.shape[1]

if n_features <= 50:
    # Create full heatmap for datasets with reasonable number of features
    print("\n--- Creating Full Pearson Correlation Heatmap ---")
    create_correlation_heatmap(
        pearson_corr, pearson_p_values,
        title="Pearson Correlation Matrix with Statistical Significance",
        save_name="pearson_correlation_full"
    )
    
    print("\n--- Creating Full Spearman Correlation Heatmap ---")
    create_correlation_heatmap(
        spearman_corr, spearman_p_values,
        title="Spearman Correlation Matrix with Statistical Significance", 
        save_name="spearman_correlation_full"
    )
    
else:
    print(f"Large dataset with {n_features} features. Creating focused correlation maps...")
    
    # Create focused heatmap with top correlated features to target
    top_features = pearson_target_results.head(25)['feature'].tolist() + ['ipec_pd']
    
    # Remove duplicates while preserving order
    top_features_unique = []
    for feature in top_features:
        if feature not in top_features_unique:
            top_features_unique.append(feature)
    
    # Create subset correlation matrices
    df_subset = df[top_features_unique]
    pearson_subset = df_subset.corr(method='pearson')
    spearman_subset = df_subset.corr(method='spearman')
    
    print(f"\n--- Creating Focused Pearson Correlation Heatmap (Top {len(top_features_unique)-1} features) ---")
    create_correlation_heatmap(
        pearson_subset,
        title=f"Pearson Correlations: Top Features vs Target Variable",
        figsize=(14, 12),
        save_name="pearson_correlation_focused"
    )
    
    print(f"\n--- Creating Focused Spearman Correlation Heatmap (Top {len(top_features_unique)-1} features) ---")
    create_correlation_heatmap(
        spearman_subset,
        title=f"Spearman Correlations: Top Features vs Target Variable",
        figsize=(14, 12),
        save_name="spearman_correlation_focused"
    )

# ==============================================================================
# Target Variable Correlation Bar Charts
# ==============================================================================
def create_target_correlation_chart(target_results, method_name, top_n=20):
    """
    Create bar chart showing correlations with target variable
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Prepare data for top positive and negative correlations
    top_data = target_results.head(top_n)
    
    # Separate positive and negative correlations
    positive_corr = top_data[top_data['correlation'] > 0]
    negative_corr = top_data[top_data['correlation'] < 0]
    
    # Plot 1: All top correlations (sorted by absolute value)
    colors = ['red' if x < 0 else 'blue' for x in top_data['correlation']]
    bars1 = ax1.barh(range(len(top_data)), top_data['correlation'], color=colors, alpha=0.7)
    
    ax1.set_yticks(range(len(top_data)))
    ax1.set_yticklabels(top_data['feature'], fontsize=10)
    ax1.set_xlabel(f'{method_name} Correlation with ipec_pd', fontsize=12)
    ax1.set_title(f'Top {top_n} Features: {method_name} Correlations with Target', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add correlation values on bars
    for i, (bar, val) in enumerate(zip(bars1, top_data['correlation'])):
        width = bar.get_width()
        ax1.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    # Plot 2: Positive vs Negative breakdown
    if len(positive_corr) > 0 and len(negative_corr) > 0:
        # Show top positive and negative separately
        n_pos = min(10, len(positive_corr))
        n_neg = min(10, len(negative_corr))
        
        ax2.barh(range(n_pos), positive_corr.head(n_pos)['correlation'], 
                color='blue', alpha=0.7, label='Positive Correlations')
        ax2.barh(range(n_pos, n_pos + n_neg), negative_corr.head(n_neg)['correlation'], 
                color='red', alpha=0.7, label='Negative Correlations')
        
        all_features = list(positive_corr.head(n_pos)['feature']) + list(negative_corr.head(n_neg)['feature'])
        ax2.set_yticks(range(len(all_features)))
        ax2.set_yticklabels(all_features, fontsize=10)
        
    else:
        # All correlations have same sign
        bars2 = ax2.barh(range(len(top_data)), top_data['correlation'], 
                        color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_data)))
        ax2.set_yticklabels(top_data['feature'], fontsize=10)
    
    ax2.set_xlabel(f'{method_name} Correlation with ipec_pd', fontsize=12)
    ax2.set_title(f'Positive vs Negative Correlations with Target', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the chart
    chart_path = os.path.join(FIGURE_DIR, f'{method_name.lower()}_target_correlations')
    plt.savefig(chart_path + '.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(chart_path + '.pdf', bbox_inches='tight', transparent=True)
    
    print(f"{method_name} target correlation chart saved to: {chart_path}.[png/pdf]")
    plt.show()

# Create target correlation charts
create_target_correlation_chart(pearson_target_results, "Pearson", top_n=20)
create_target_correlation_chart(spearman_target_results, "Spearman", top_n=20)

# ==============================================================================
# Hierarchical Clustering of Features
# ==============================================================================
def create_correlation_clustering(corr_matrix, method_name='Pearson'):
    """
    Create hierarchical clustering of features based on correlation
    """
    print(f"\n--- Creating Correlation-Based Feature Clustering ({method_name}) ---")
    
    # Calculate distance matrix (1 - |correlation|)
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    # Create dendrogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot dendrogram
    dendro = dendrogram(linkage_matrix, labels=corr_matrix.columns, 
                       orientation='left', ax=ax1, leaf_font_size=10)
    ax1.set_title(f'Feature Clustering Dendrogram ({method_name})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Distance (1 - |correlation|)', fontsize=12)
    
    # Reorder correlation matrix based on clustering
    cluster_order = dendro['leaves']
    clustered_corr = corr_matrix.iloc[cluster_order, cluster_order]
    
    # Plot clustered correlation matrix
    im = ax2.imshow(clustered_corr.values, cmap=CORRELATION_CMAP, aspect='auto', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(clustered_corr.columns)))
    ax2.set_yticks(range(len(clustered_corr.index)))
    ax2.set_xticklabels(clustered_corr.columns, rotation=45, ha='right', fontsize=8)
    ax2.set_yticklabels(clustered_corr.index, fontsize=8)
    ax2.set_title(f'Clustered Correlation Matrix ({method_name})', 
                  fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    
    # Save clustering visualization
    cluster_path = os.path.join(FIGURE_DIR, f'{method_name.lower()}_correlation_clustering')
    plt.savefig(cluster_path + '.png', dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(cluster_path + '.pdf', bbox_inches='tight', transparent=True)
    
    print(f"Correlation clustering saved to: {cluster_path}.[png/pdf]")
    plt.show()
    
    return clustered_corr

# Create clustering visualizations (only if reasonable number of features)
if n_features <= 50:
    pearson_clustered = create_correlation_clustering(pearson_corr, "Pearson")
    spearman_clustered = create_correlation_clustering(spearman_corr, "Spearman")

# ==============================================================================
# Correlation Summary Statistics
# ==============================================================================
def generate_correlation_summary(corr_matrix, method_name):
    """
    Generate summary statistics for correlation matrix
    """
    # Get upper triangle of correlation matrix (excluding diagonal)
    upper_tri = np.triu(corr_matrix.values, k=1)
    correlations = upper_tri[upper_tri != 0]
    
    summary_stats = {
        'method': method_name,
        'n_features': len(corr_matrix.columns),
        'n_correlations': len(correlations),
        'mean_abs_correlation': np.mean(np.abs(correlations)),
        'median_abs_correlation': np.median(np.abs(correlations)),
        'max_abs_correlation': np.max(np.abs(correlations)),
        'min_abs_correlation': np.min(np.abs(correlations)),
        'std_abs_correlation': np.std(np.abs(correlations)),
        'high_corr_count': np.sum(np.abs(correlations) > 0.7),
        'moderate_corr_count': np.sum((np.abs(correlations) > 0.3) & (np.abs(correlations) <= 0.7)),
        'low_corr_count': np.sum(np.abs(correlations) <= 0.3)
    }
    
    return summary_stats

# Generate summaries
pearson_summary = generate_correlation_summary(pearson_corr, 'Pearson')
spearman_summary = generate_correlation_summary(spearman_corr, 'Spearman')

# Create summary DataFrame
summary_df = pd.DataFrame([pearson_summary, spearman_summary])

print("\n--- Correlation Analysis Summary ---")
print("="*50)
print(summary_df.to_string(index=False))

# Save all results
results_summary = {
    'pearson_target_correlations': pearson_target_results,
    'spearman_target_correlations': spearman_target_results,
    'correlation_summary_stats': summary_df
}

# Save to files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(RESULTS_DIR, f'correlation_analysis_{timestamp}')
os.makedirs(results_dir, exist_ok=True)

for name, data in results_summary.items():
    filepath = os.path.join(results_dir, f'{name}.csv')
    data.to_csv(filepath, index=False)
    print(f"Saved {name} to: {filepath}")

# Save full correlation matrices
pearson_corr.to_csv(os.path.join(results_dir, 'pearson_correlation_matrix.csv'))
spearman_corr.to_csv(os.path.join(results_dir, 'spearman_correlation_matrix.csv'))

print(f"\n{'='*60}")
print("CORRELATION ANALYSIS COMPLETE!")
print(f"{'='*60}")
print("Generated Outputs:")
print("  • Correlation heatmaps (Pearson & Spearman)")
print("  • Target variable correlation charts")
print("  • Feature clustering dendrograms")
print("  • Statistical summaries and data exports")
print(f"  • All files saved to: {results_dir}")
print(f"{'='*60}")

# ==============================================================================
# Additional Research Paper Specific Outputs
# ==============================================================================

# Create a publication-ready correlation table for the paper
def create_publication_table(target_results, method_name, top_n=10):
    """
    Create a clean table for publication
    """
    pub_table = target_results.head(top_n).copy()
    pub_table['correlation'] = pub_table['correlation'].round(4)
    pub_table['abs_correlation'] = pub_table['abs_correlation'].round(4)
    
    # Rename columns for publication
    pub_table = pub_table.rename(columns={
        'rank': 'Rank',
        'feature': 'Feature',
        'correlation': f'{method_name} r',
        'abs_correlation': 'Absolute r'
    })
    
    return pub_table

# Create publication tables
pearson_pub_table = create_publication_table(pearson_target_results, 'Pearson', 15)
spearman_pub_table = create_publication_table(spearman_target_results, 'Spearman', 15)

# Save publication tables
pearson_pub_table.to_csv(os.path.join(results_dir, 'pearson_publication_table.csv'), index=False)
spearman_pub_table.to_csv(os.path.join(results_dir, 'spearman_publication_table.csv'), index=False)

print("\nPublication-ready tables created:")
print("  • pearson_publication_table.csv")
print("  • spearman_publication_table.csv")

print("\nTop 10 Pearson correlations with target:")
print(pearson_pub_table.to_string(index=False))

print("\nScript execution completed successfully!")