import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcbs.datasets.loader import DatasetLoader
import pandas as pd
import matplotlib.pyplot as plt

def display_dataset_info(dataset_name: str, loader: DatasetLoader):
    print(f"\nDataset: {dataset_name}")
    info = loader.get_dataset_info(dataset_name)
    for key, value in info.items():
        print(f"{key}: {value}")

def display_basic_stats(X: pd.DataFrame, y: pd.Series):
    print("\nFeature Statistics:")
    print(X.describe())
    
    print("\nTarget Variable Statistics:")
    print(y.describe())

#def plot_feature_distributions(X: pd.DataFrame):
    #fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #axes = axes.ravel()
    
    #for i, column in enumerate(X.columns[:4]):  # Plot first 4 features
        #X[column].hist(ax=axes[i])
        #axes[i].set_title(column)
    
    #plt.tight_layout()
    #plt.show()

def main():
    loader = DatasetLoader()
    print("Available datasets:", loader.list_datasets())

    for dataset_name in loader.list_datasets():
        display_dataset_info(dataset_name, loader)
        
        try:
            X, y = loader.load_dataset(dataset_name)
            display_basic_stats(X, y)
            #plot_feature_distributions(X)
        except FileNotFoundError:
            print(f"Dataset file for '{dataset_name}' not found. Skipping.")

if __name__ == "__main__":
    main()