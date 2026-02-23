# results_manager.py
"""
Results saving and analysis functionality
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from visualization import Visualizer

class ResultsManager:
    """Handles saving results and creating final analysis"""
    
    def __init__(self, config):
        self.config = config
        self.visualizer = Visualizer(config)
    
    def save_all_results(self, all_results, X_test, y_test):
        """Save all results and create final analysis"""
        # Create results comparison
        results_df = self._create_results_dataframe(all_results)
        
        # Save model comparison results
        self._save_model_comparison(results_df)
        
        # Generate and save predictions
        predictions_df = self._generate_predictions_dataframe(all_results, y_test)
        seed_dir = self._save_predictions_data(predictions_df, results_df)
        
        # Create trend analysis visualizations
        self.visualizer.create_trend_analysis(predictions_df)
        
        print(f"\n=== Results saved to: {seed_dir} ===")
        print("Generated outputs:")
        print("  • Model comparison results")
        print("  • Predictions dataset")
        print("  • All visualization charts")
        print("="*60)
        
        return results_df, predictions_df
    
    def _create_results_dataframe(self, all_results):
        """Create DataFrame with model comparison results"""
        results_df = pd.DataFrame({
            'Model': list(all_results.keys()),
            'RMSE': [res['rmse'] for res in all_results.values()],
            'MAE': [res['mae'] for res in all_results.values()],
            'R2': [res['r2'] for res in all_results.values()],
            'Training Time (s)': [res['training_time'] for res in all_results.values()]
        })
        
        # Sort by R2 in descending order
        results_df = results_df.sort_values(by='R2', ascending=False).reset_index(drop=True)
        
        print("\n--- Final Model Performance Comparison ---")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def _save_model_comparison(self, results_df):
        """Save model comparison results to CSV"""
        output_path = os.path.join(self.config.RESULTS_DIR, 'model_comparison_results.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\nModel comparison results saved to: {output_path}")
    
    def _generate_predictions_dataframe(self, all_results, y_test):
        """Generate DataFrame with all model predictions"""
        predictions_df = pd.DataFrame({'actual': y_test.values})
        
        # Add predictions from each model
        for model_name, results in all_results.items():
            if 'predictions' in results:
                predictions_df[f'{model_name}_pred'] = results['predictions']
            else:
                # Generate predictions for traditional ML models
                y_pred = results['model'].predict(y_test.index.to_frame())
                y_pred[y_pred < 0] = 0  # Ensure non-negative
                predictions_df[f'{model_name}_pred'] = y_pred
        
        return predictions_df
    
    def _save_predictions_data(self, predictions_df, results_df):
        """Save predictions and results with timestamped directory"""
        # Generate unique seed based on timestamp
        seed = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_dir = os.path.join(self.config.DATA_DIR, 'results', seed)
        os.makedirs(seed_dir, exist_ok=True)
        
        # Save predictions
        predictions_path = os.path.join(seed_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
        
        # Save model results
        results_path = os.path.join(seed_dir, 'model_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Model results saved to: {results_path}")
        
        return seed_dir