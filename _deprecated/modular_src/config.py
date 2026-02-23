# config.py
"""
Configuration settings for the ML pipeline
"""

import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

class Config:
    """Configuration class for all pipeline settings"""
    
    def __init__(self):
        self.setup_environment()
        self.setup_paths()
        self.setup_plotting()
        self.setup_model_params()
    
    def setup_environment(self):
        """Configure environment settings"""
        warnings.filterwarnings('ignore')
    
    def setup_paths(self):
        """Setup all directory paths"""
        self.BASE_DIR = '/home/ellenfel/Desktop/repos/High_Voltage_Analysis_ML'
        self.FIGURE_DIR = os.path.join(self.BASE_DIR, 'docs/figures')
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, 'docs/results')
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        
        # Create directories if they don't exist
        for directory in [self.FIGURE_DIR, self.RESULTS_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_plotting(self):
        """Configure plotting style and parameters"""
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('viridis')
        self.FIGURE_DPI = 300
        
        # Professional color palettes
        self.COLORS = {
            'primary': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#4E6E58', '#5A4E6E'],
            'trend': {
                'actual': '#000000',
                'Random Forest_pred': '#1f77b4',
                'Gradient Boosting_pred': '#ff7f0e',
                'XGBoost_pred': '#2ca02c',
                'LightGBM_pred': '#d62728',
                'Deep Neural Network_pred': '#9467bd',
                'Optimized DNN_pred': '#8c564b'
            }
        }
        
        # Clean labels for visualization
        self.CLEAN_LABELS = {
            'actual': 'Actual Values',
            'Random Forest_pred': 'Random Forest',
            'Gradient Boosting_pred': 'Gradient Boosting',
            'XGBoost_pred': 'XGBoost',
            'LightGBM_pred': 'LightGBM',
            'Deep Neural Network_pred': 'Deep Neural Network',
            'Optimized DNN_pred': 'Optimized DNN'
        }
    
    def setup_model_params(self):
        """Define model hyperparameters"""
        # Standard DNN parameters
        self.DNN_PARAMS = {
            "num_layers": 5,
            "units": [1024, 768, 512, 256, 128],
            "activations": ["relu", "relu", "relu", "relu", "relu"],
            "l2_lambdas": [1e-4, 1e-4, 1e-4, 1e-4, 1e-4],
            "dropouts": [0.3, 0.25, 0.2, 0.15, 0.1],
            "learning_rate": 0.0005,
            "epochs": 25,
            "batch_size": 512,
            "early_stopping_patience": 7,
            "reduce_lr_patience": 4
        }
        
        # Optimized DNN parameters
        self.OPTIMIZED_DNN_PARAMS = {
            "num_layers": 4,
            "units": [512, 448, 320, 192],
            "activations": ["relu", "relu", "relu", "tanh"],
            "l2_lambdas": [5.005115883734732e-05, 0.005496054400493782, 0.0008832783459212954, 0.00013494772704967807],
            "dropouts": [0.2, 0.30000000000000004, 0.4, 0.30000000000000004],
            "learning_rate": 0.001,
            "epochs": 12,
            "batch_size": 64,
            "early_stopping_patience": 5,
            "reduce_lr_patience": 3
        }