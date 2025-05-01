from mcbs.datasets import DatasetLoader
from mcbs.models.ltds_model import BaseLTDSModel
import pandas as pd
import numpy as np

class TestLTDSModel(BaseLTDSModel):
    """Test class that inherits from BaseLTDSModel but overrides __init__ to expose encoded data"""
    def __init__(self, data):
        # Only encode categorical variables without creating database
        self.encoded_data = self._encode_categorical_variables(data)
    
    def estimate(self):
        pass  # Required to implement abstract method

def check_nan_values():
    # Load the dataset
    loader = DatasetLoader()
    data = loader.load_dataset("ltds_dataset")
    
    print("\nChecking raw data:")
    print("-" * 60)
    check_dataframe_nans(data, "Raw Data")
    
    # Create test model instance to process the data
    print("\nChecking data after categorical encoding:")
    print("-" * 60)
    model = TestLTDSModel(data)
    encoded_data = model.encoded_data
    check_dataframe_nans(encoded_data, "Encoded Data")

def check_dataframe_nans(df, stage_name):
    total_rows = len(df)
    
    print(f"\n{stage_name} Shape: {df.shape}")
    print(f"{'Column Name':<30} {'NaN Count':<10} {'% NaN':<10}")
    print("-" * 60)
    
    # Check each column for NaN values
    has_nans = False
    for column in df.columns:
        nan_count = df[column].isna().sum()
        if nan_count > 0:
            has_nans = True
            nan_percent = (nan_count / total_rows) * 100
            print(f"{column:<30} {nan_count:<10} {nan_percent:.2f}%")
            
            # Print some additional diagnostics for columns with NaNs
            print(f"\nDiagnostics for column: {column}")
            print("Unique values (first 10):", df[column].unique()[:10])
            print("Value counts:")
            print(df[column].value_counts().head())
            
            # If categorical, check mapping
            if column in ['travel_mode', 'purpose', 'fueltype', 'faretype']:
                print(f"\nUnique values before mapping:")
                unique_vals = df[column].unique()
                print(unique_vals)
    
    if not has_nans:
        print("No NaN values found in any column!")

if __name__ == "__main__":
    check_nan_values()
