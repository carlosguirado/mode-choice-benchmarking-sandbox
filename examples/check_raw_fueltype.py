from mcbs.datasets import DatasetLoader
import pandas as pd

def check_raw_fueltype():
    # Load the dataset
    loader = DatasetLoader()
    data = loader.load_dataset("ltds_dataset")
    
    print("\nFuel type analysis:")
    print("-" * 60)
    print("\nUnique values in fueltype column:")
    print(data['fueltype'].unique())
    
    print("\nValue counts for fueltype:")
    print(data['fueltype'].value_counts(dropna=False))
    
    print("\nCross-tabulation with travel_mode:")
    print(pd.crosstab(data['travel_mode'], data['fueltype'], margins=True))

if __name__ == "__main__":
    check_raw_fueltype()
