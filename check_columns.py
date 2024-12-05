from mcbs.datasets import DatasetLoader
import pandas as pd

def check_columns():
    # Load the dataset
    loader = DatasetLoader()
    data = loader.load_dataset("ltds_dataset")
    
    print("\nAvailable columns in the dataset:")
    print("-" * 60)
    for col in sorted(data.columns):
        print(col)
        
    # Check if similar column names exist
    print("\nColumns containing 'pt' or 'int':")
    pt_cols = [col for col in data.columns if 'pt' in col.lower()]
    int_cols = [col for col in data.columns if 'int' in col.lower()]
    print("\nPT-related columns:", pt_cols)
    print("Interchange-related columns:", int_cols)
    
    print("\nColumns containing 'cost' or 'charge':")
    cost_cols = [col for col in data.columns if 'cost' in col.lower()]
    charge_cols = [col for col in data.columns if 'charge' in col.lower()]
    print("\nCost-related columns:", cost_cols)
    print("Charge-related columns:", charge_cols)

if __name__ == "__main__":
    check_columns()
