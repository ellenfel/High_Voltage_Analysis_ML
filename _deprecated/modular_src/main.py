# main.py
"""
Main script for High Voltage Analysis ML Pipeline
Orchestrates the entire machine learning workflow
"""

from config import Config
from data_handler import DataHandler
from model_trainer import ModelTrainer
from visualization import Visualizer
from results_manager import ResultsManager

def main():
    """Main execution function"""
    print("=== High Voltage Analysis ML Pipeline ===")
    
    # Initialize components
    config = Config()
    data_handler = DataHandler(config)
    model_trainer = ModelTrainer(config)
    visualizer = Visualizer(config)
    results_manager = ResultsManager(config)
    
    # Load and prepare data
    print("\n--- Loading and Preparing Data ---")
    X_train, X_test, y_train, y_test = data_handler.load_and_prepare_data()
    
    # Train all models
    print("\n--- Training Models ---")
    all_results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Generate visualizations
    print("\n--- Creating Visualizations ---")
    visualizer.create_all_visualizations(all_results, y_test)
    
    # Save results and analysis
    print("\n--- Saving Results ---")
    results_manager.save_all_results(all_results, X_test, y_test)
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()