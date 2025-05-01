"""Base class for discrete choice models"""

from abc import ABC, abstractmethod
import biogeme.biogeme_logging as blog
from biogeme.database import Database
import numpy as np
import pandas as pd

class BaseDiscreteChoiceModel(ABC):
    """Base class for all discrete choice models."""
    
    def __init__(self, data):
        """Initialize base model structure."""
        self.logger = blog.get_screen_logger(level=blog.INFO)
        
        # Debug: Check for NaN values in input data
        self._debug_check_nans(data, "Input data")
        
        # Check if data needs preprocessing (if categorical columns are not yet numeric)
        categorical_columns = ['travel_mode', 'purpose', 'fueltype', 'faretype']
        needs_preprocessing = any(col in data.columns and data[col].dtype == 'object' 
                                for col in categorical_columns)
        
        if needs_preprocessing:
            data_processed = self._preprocess_data(data)
        else:
            # Data is already preprocessed, just ensure float64 for numeric columns
            data_processed = data.copy()
            numeric_columns = [
                'dur_walking', 'dur_cycling', 'dur_driving', 
                'dur_pt_access', 'dur_pt_rail', 'dur_pt_bus', 
                'dur_pt_int_total', 'cost_driving_fuel',
                'cost_driving_con_charge', 'cost_transit',
                'driving_traffic_percent'
            ]
            for col in numeric_columns:
                if col in data_processed.columns:
                    data_processed[col] = data_processed[col].astype('float64')
        
        # Debug: Check for NaN values after preprocessing
        self._debug_check_nans(data_processed, "After preprocessing")
        
        # Convert any remaining float columns to float64 to ensure consistency
        float_cols = data_processed.select_dtypes(include=['float']).columns
        for col in float_cols:
            data_processed[col] = data_processed[col].astype('float64')
        
        # Debug: Check for NaN values after final conversion
        self._debug_check_nans(data_processed, "After final conversion")
        
        # Create database
        self.database = Database('choice_model', data_processed)
        self.test_database = Database('choice_model', data_processed)
        self.results = None
        
        # Initialize metrics attributes
        self.final_ll = None
        self.rho_squared = None
        self.rho_squared_bar = None
        self.vot_walking = None
        self.vot_cycling = None
        self.vot_pt = None
        self.vot_driving = None
        self.market_share_accuracy = None
        self.choice_accuracy = None
        self.actual_shares = None
        self.predicted_shares = None
        self.confusion_matrix = None
    
    def _debug_check_nans(self, df, stage):
        """Debug helper to check for NaN values."""
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            print(f"\nNaN check at {stage}:")
            for col in nan_cols:
                nan_count = df[col].isna().sum()
                print(f"{col}: {nan_count} NaN values")
                if nan_count > 0:
                    print(f"Sample of rows with NaN in {col}:")
                    print(df[df[col].isna()].head())
    
    def _preprocess_data(self, data):
        """
        Preprocess data to ensure compatibility with Biogeme.
        
        Parameters:
        data (pandas.DataFrame): Input data
        
        Returns:
        pandas.DataFrame: Processed data with correct dtypes
        """
        df = data.copy()
        
        # List of numeric columns that should be float64
        numeric_columns = [
            'dur_walking', 'dur_cycling', 'dur_driving', 
            'dur_pt_access', 'dur_pt_rail', 'dur_pt_bus', 
            'dur_pt_int_total', 'cost_driving_fuel',
            'cost_driving_con_charge', 'cost_transit',
            'driving_traffic_percent'
        ]
        
        # List of categorical columns that should be int64
        categorical_columns = [
            'travel_mode', 'purpose', 'fueltype', 'faretype'
        ]
        
        # Default values for categorical columns
        categorical_defaults = {
            'travel_mode': 1,  # Default to walk
            'purpose': 1,      # Default to HBW
            'fueltype': 6,     # Default to Average_Car
            'faretype': 1      # Default to full fare
        }
        
        # Debug: Check before numeric conversion
        self._debug_check_nans(df, "Before numeric conversion")
        
        # Convert numeric columns to float64
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype('float64')
        
        # Debug: Check after numeric conversion
        self._debug_check_nans(df, "After numeric conversion")
        
        # Convert categorical columns to int64 if they're not already
        for col in categorical_columns:
            if col in df.columns and df[col].dtype == 'object':
                # If column is still object type (string), we need to map it
                if col == 'travel_mode':
                    mapping = {'walk': 1, 'cycle': 2, 'pt': 3, 'drive': 4}
                elif col == 'purpose':
                    mapping = {'HBW': 1, 'HBE': 2, 'HBO': 3, 'B': 4, 'NHBO': 5}
                elif col == 'fueltype':
                    mapping = {'Petrol_Car': 1, 'Diesel_Car': 2, 'Hybrid_Car': 3,
                             'Petrol_LGV': 4, 'Diesel_LGV': 5, 'Average_Car': 6}
                elif col == 'faretype':
                    mapping = {'full': 1, '16+': 2, 'child': 3, 'dis': 4, 'free': 5}
                
                df[col] = df[col].map(mapping).fillna(categorical_defaults[col]).astype('int64')
            elif col in df.columns:
                # If column exists but is already numeric, just ensure it's int64
                df[col] = df[col].fillna(categorical_defaults[col]).astype('int64')
        
        # Debug: Check after categorical conversion
        self._debug_check_nans(df, "After categorical conversion")
        
        return df
    
    @abstractmethod
    def estimate(self):
        """Estimate model parameters. Must be implemented by subclasses."""
        pass
    
    def get_metrics(self):
        """Get standard metrics for model comparison."""
        if self.results is None:
            raise RuntimeError("Model must be estimated before getting metrics")
            
        metrics = {
            'n_parameters': len(self.results.data.betaValues),
            'n_observations': self.results.data.numberOfObservations,
        }
        
        # Add model statistics and VOT metrics
        metrics.update({
            'final_ll': self.final_ll if hasattr(self, 'final_ll') else None,
            'rho_squared': self.rho_squared if hasattr(self, 'rho_squared') else None,
            'rho_squared_bar': self.rho_squared_bar if hasattr(self, 'rho_squared_bar') else None,
            'vot_walking': self.vot_walking if hasattr(self, 'vot_walking') else None,
            'vot_cycling': self.vot_cycling if hasattr(self, 'vot_cycling') else None,
            'vot_pt': self.vot_pt if hasattr(self, 'vot_pt') else None,
            'vot_driving': self.vot_driving if hasattr(self, 'vot_driving') else None,
            'market_share_accuracy': self.market_share_accuracy if hasattr(self, 'market_share_accuracy') else None,
            'choice_accuracy': self.choice_accuracy if hasattr(self, 'choice_accuracy') else None,
            'actual_shares': self.actual_shares if hasattr(self, 'actual_shares') else None,
            'predicted_shares': self.predicted_shares if hasattr(self, 'predicted_shares') else None,
            'confusion_matrix': self.confusion_matrix.to_dict() if hasattr(self, 'confusion_matrix') else None
        })
        
        return metrics
