# visualization.py
"""
All visualization functionality for the ML pipeline
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import gc
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Visualizer:
    """Handles all visualization operations"""
    
    def __init__(self, config):
        self.config = config
    
    def create_all_visualizations(self, all_results, y_test):
        """Create all visualizations for the analysis"""
        # DNN training history (if available)
        if "Deep Neural Network" in all_results and 'history' in all_results["Deep Neural Network"]:
            self._plot_dnn_training_history(all_results["Deep Neural Network"]['history'])
        
        # Model comparison
        self._plot_model_comparison(all_results)
        
        # Predictions vs actual
        self._plot_predictions_vs_actual(all_results, y_test)
        
        # Residuals analysis
        self._plot_residuals_analysis(all_results, y_test)
    
    def _plot_dnn_training_history(self, history):
        """Plot DNN training history"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
        
        train_color = '#1f77b4'
        val_color = '#d62728'
        
        # Loss plot
        ax1.plot(history['loss'], label='Training Loss', color=train_color, linewidth=3, alpha=0.9)
        ax1.plot(history['val_loss'], label='Validation Loss', color=val_color, 
                linewidth=3, alpha=0.9, linestyle='--')
        ax1.set_title('DNN Training Loss', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        ax2.plot(history['mae'], label='Training MAE', color=train_color, linewidth=3, alpha=0.9)
        ax2.plot(history['val_mae'], label='Validation MAE', color=val_color,
                linewidth=3, alpha=0.9, linestyle='--')
        ax2.set_title('DNN Training MAE', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # R² plot
        if 'r2_keras' in history:
            ax3.plot(history['r2_keras'], label='Training R²', color=train_color, linewidth=3, alpha=0.9)
            ax3.plot(history['val_r2_keras'], label='Validation R²', color=val_color,
                    linewidth=3, alpha=0.9, linestyle='--')
            ax3.set_title('DNN Training R²', fontsize=14, fontweight='bold', pad=15)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('R² Score', fontsize=12)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Save figure
        figure_path = os.path.join(self.config.FIGURE_DIR, 'dnn_training_history')
        plt.savefig(figure_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(figure_path + '.pdf', bbox_inches='tight', transparent=True)
        print(f"DNN training history saved to: {figure_path}.[png/pdf]")
        plt.show()
    
    def _plot_model_comparison(self, all_results):
        """Plot model performance comparison"""
        # Prepare data
        results_df = pd.DataFrame({
            'Model': list(all_results.keys()),
            'RMSE': [res['rmse'] for res in all_results.values()],
            'MAE': [res['mae'] for res in all_results.values()],
            'R2': [res['r2'] for res in all_results.values()],
            'Training Time (s)': [res['training_time'] for res in all_results.values()]
        }).sort_values(by='R2', ascending=False).reset_index(drop=True)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        colors = self.config.COLORS['primary']
        
        # RMSE comparison
        bars1 = ax1.bar(results_df['Model'], results_df['RMSE'], color=colors, alpha=0.8)
        ax1.set_title('Model Performance: RMSE', fontsize=14, fontweight='bold')
        ax1.set_ylabel('RMSE', fontsize=12)
        ax1.tick_params(axis='x', rotation=45, labelsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        self._add_value_labels(ax1, bars1)
        
        # MAE comparison
        bars2 = ax2.bar(results_df['Model'], results_df['MAE'], color=colors, alpha=0.8)
        ax2.set_title('Model Performance: MAE', fontsize=14, fontweight='bold')
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        self._add_value_labels(ax2, bars2)
        
        # R² comparison
        bars3 = ax3.bar(results_df['Model'], results_df['R2'], color=colors, alpha=0.8)
        ax3.set_title('Model Performance: R² Score', fontsize=14, fontweight='bold')
        ax3.set_ylabel('R² Score', fontsize=12)
        ax3.tick_params(axis='x', rotation=45, labelsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        self._add_value_labels(ax3, bars3)
        
        # Save
        comparison_path = os.path.join(self.config.FIGURE_DIR, 'model_performance_comparison')
        plt.savefig(comparison_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(comparison_path + '.pdf', bbox_inches='tight')
        print(f"Model comparison saved to: {comparison_path}.[png/pdf]")
        plt.show()
    
    def _plot_predictions_vs_actual(self, all_results, y_test):
        """Plot predictions vs actual for best model"""
        # Find best model
        best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['r2'])
        best_results = all_results[best_model_name]
        
        # Get predictions
        if 'predictions' in best_results:
            y_pred = best_results['predictions']
        else:
            y_pred = best_results['model'].predict(pd.DataFrame(y_test.index))
            y_pred[y_pred < 0] = 0
        
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        
        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, c='#2E86AB', s=50, edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
        
        ax.set_title(f'Prediction vs Actual Values - {best_model_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Actual Values', fontsize=14)
        ax.set_ylabel('Predicted Values', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2_text = f'$R^2 = {best_results["r2"]:.4f}$'
        ax.text(0.05, 0.95, r2_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save
        scatter_path = os.path.join(self.config.FIGURE_DIR, f'prediction_vs_actual_{best_model_name.lower().replace(" ", "_")}')
        plt.savefig(scatter_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(scatter_path + '.pdf', bbox_inches='tight')
        print(f"Prediction vs actual plot saved to: {scatter_path}.[png/pdf]")
        plt.show()
    
    def _plot_residuals_analysis(self, all_results, y_test):
        """Plot residuals analysis for best model"""
        # Find best model
        best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['r2'])
        best_results = all_results[best_model_name]
        
        # Get predictions and calculate residuals
        if 'predictions' in best_results:
            y_pred = best_results['predictions']
        else:
            y_pred = best_results['model'].predict(pd.DataFrame(y_test.index))
            y_pred[y_pred < 0] = 0
        
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, c='#A23B72', s=50, edgecolors='white', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
        ax1.set_title(f'Residuals vs Predicted - {best_model_name}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2.hist(residuals, bins=50, density=True, alpha=0.7, color='#F18F01', edgecolor='black', linewidth=0.5)
        ax2.set_title(f'Residuals Distribution - {best_model_name}', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add normal distribution overlay
        mu, sigma = stats.norm.fit(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mu, sigma)
        ax2.plot(x_norm, y_norm, 'r-', linewidth=2, alpha=0.8, 
                label=f'Normal fit ($\\mu={mu:.3f}, \\sigma={sigma:.3f}$)')
        ax2.legend()
        
        # Save
        residuals_path = os.path.join(self.config.FIGURE_DIR, f'residuals_analysis_{best_model_name.lower().replace(" ", "_")}')
        plt.savefig(residuals_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(residuals_path + '.pdf', bbox_inches='tight')
        print(f"Residuals analysis saved to: {residuals_path}.[png/pdf]")
        plt.show()
    
    def create_trend_analysis(self, predictions_df):
        """Create trend analysis visualizations"""
        pred_columns = [col for col in predictions_df.columns if col.endswith('_pred')]
        
        # Main trend comparison
        self._plot_main_trend_comparison(predictions_df, pred_columns)
        
        # Individual wide charts
        self._plot_individual_trends(predictions_df, pred_columns)
        
        # Residuals trend
        self._plot_residuals_trend(predictions_df, pred_columns)
    
    def _plot_main_trend_comparison(self, predictions_df, pred_columns):
        """Plot main trend comparison chart"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Plot actual values
        ax.plot(predictions_df.index, predictions_df['actual'], 
                color=self.config.COLORS['trend']['actual'], 
                linewidth=3, label=self.config.CLEAN_LABELS['actual'],
                alpha=0.9, zorder=10)
        
        # Plot predictions
        line_styles = ['-', '--', '-.', ':', '-', '--']
        for i, col in enumerate(pred_columns):
            ax.plot(predictions_df.index, predictions_df[col],
                    color=self.config.COLORS['trend'][col],
                    linewidth=2,
                    linestyle=line_styles[i % len(line_styles)],
                    alpha=0.8,
                    label=self.config.CLEAN_LABELS[col])
        
        ax.set_title('Model Predictions vs Actual Values - Trend Analysis', 
                     fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel('Test Sample Index', fontweight='bold', fontsize=14)
        ax.set_ylabel('Target Value (ipec_pd)', fontweight='bold', fontsize=14)
        ax.legend(loc='upper left', frameon=True, ncol=2, columnspacing=1.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        main_trend_path = os.path.join(self.config.FIGURE_DIR, 'model_predictions_trend_comparison')
        plt.savefig(main_trend_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(main_trend_path + '.pdf', bbox_inches='tight')
        print(f"Main trend comparison saved to: {main_trend_path}.[png/pdf]")
        plt.show()
    
    def _plot_individual_trends(self, predictions_df, pred_columns):
        """Plot individual wide trend charts"""
        for col in pred_columns:
            fig, ax = plt.subplots(figsize=(24, 8))
            
            # Plot actual and predicted
            ax.plot(predictions_df.index, predictions_df['actual'], 
                    color='#000000', linewidth=1.5, label='Actual Values', alpha=0.8)
            ax.plot(predictions_df.index, predictions_df[col],
                    color=self.config.COLORS['trend'][col],
                    linewidth=1.2, label=self.config.CLEAN_LABELS[col], alpha=0.9)
            
            # Calculate metrics
            r2 = r2_score(predictions_df['actual'], predictions_df[col])
            rmse = np.sqrt(mean_squared_error(predictions_df['actual'], predictions_df[col]))
            mae = mean_absolute_error(predictions_df['actual'], predictions_df[col])
            
            ax.set_title(f'{self.config.CLEAN_LABELS[col]} vs Actual Values\n'
                        f'R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}', 
                        fontweight='bold', fontsize=16, pad=20)
            ax.set_xlabel('Test Sample Index', fontweight='bold', fontsize=14)
            ax.set_ylabel('Target Value (ipec_pd)', fontweight='bold', fontsize=14)
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            model_filename = col.replace(' ', '_').replace('_pred', '').lower()
            individual_path = os.path.join(self.config.FIGURE_DIR, f'trend_{model_filename}_wide')
            plt.savefig(individual_path + '.png', dpi=300, bbox_inches='tight')
            plt.savefig(individual_path + '.pdf', bbox_inches='tight')
            print(f"Individual trend chart saved to: {individual_path}.[png/pdf]")
            plt.show()
            plt.close()
    
    def _plot_residuals_trend(self, predictions_df, pred_columns):
        """Plot residuals trend analysis"""
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot residuals for each model
        for col in pred_columns:
            residuals = predictions_df['actual'] - predictions_df[col]
            ax.plot(predictions_df.index, residuals,
                    color=self.config.COLORS['trend'][col],
                    linewidth=1.5, alpha=0.7,
                    label=f'{self.config.CLEAN_LABELS[col]} Residuals')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.8)
        ax.set_title('Model Residuals Trend Analysis (Actual - Predicted)', 
                     fontweight='bold', fontsize=16, pad=20)
        ax.set_xlabel('Test Sample Index', fontweight='bold', fontsize=14)
        ax.set_ylabel('Residuals', fontweight='bold', fontsize=14)
        ax.legend(loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        residuals_path = os.path.join(self.config.FIGURE_DIR, 'residuals_trend_analysis')
        plt.savefig(residuals_path + '.png', dpi=600, bbox_inches='tight')
        plt.savefig(residuals_path + '.pdf', bbox_inches='tight')
        print(f"Residuals trend analysis saved to: {residuals_path}.[png/pdf]")
        plt.show()
    
    def _add_value_labels(self, ax, bars):
        """Add value labels on bar charts"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)