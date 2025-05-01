# test_dataset_loader.py

from mcbs.datasets.dataset_loader import DatasetLoader
import pandas as pd
import sys

def test_dataset_loader():
    print("Starting DatasetLoader test...")
    
    try:
        print("Initializing DatasetLoader...")
        loader = DatasetLoader()
        print("DatasetLoader initialized successfully.")
        print(f"Available methods: {', '.join(method for method in dir(loader) if not method.startswith('_'))}")
    except Exception as e:
        print(f"Error initializing DatasetLoader: {e}")
        return

    try:
        print("\nTesting list_datasets method...")
        datasets = loader.list_datasets()
        print(f"Available datasets: {datasets}")
    except AttributeError as e:
        print(f"Error: The list_datasets method is not available. {e}")
        print("This might be due to an outdated DatasetLoader class. Please ensure you've updated the class definition.")
        return
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return

    for dataset in datasets:
        try:
            print(f"\nTesting get_dataset_info for {dataset}...")
            info = loader.get_dataset_info(dataset)
            print(f"Info for {dataset}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error getting info for dataset {dataset}: {e}")
            continue

        try:
            print(f"\nTesting load_dataset for {dataset}...")
            df = loader.load_dataset(dataset)
            print(f"Dataset {dataset} loaded successfully. Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            print("First few rows:")
            print(df.head())
        except Exception as e:
            print(f"Error loading dataset {dataset}: {e}")
            continue

    try:
        print("\nTesting get_all_datasets_info method...")
        all_info = loader.get_all_datasets_info()
        print("All datasets info:")
        for dataset, info in all_info.items():
            print(f"  {dataset}: {info}")
    except Exception as e:
        print(f"Error getting all datasets info: {e}")

    print("\nDatasetLoader test completed.")

if __name__ == "__main__":
    test_dataset_loader()