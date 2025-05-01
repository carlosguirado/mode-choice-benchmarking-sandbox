#!/usr/bin/env python3
"""
Example script demonstrating the dynamic dataset discovery functionality.

This script shows how MCBS can discover datasets from the remote repository in real-time,
even if they were added after the MCBS package was released.
"""

import pandas as pd
from mcbs.datasets import fetch_data, list_available_datasets, get_dataset_info

def main():
    print("Discovering available datasets from remote repository...")
    
    # Get real-time list of all available datasets
    available_datasets = list_available_datasets()
    print(f"Found {len(available_datasets)} available datasets:\n")
    
    # Print information about each dataset
    for dataset_name in available_datasets:
        try:
            info = get_dataset_info(dataset_name)
            print(f"Dataset: {dataset_name}")
            if 'description' in info:
                print(f"  Description: {info['description']}")
            if 'n_samples' in info:
                print(f"  Samples: {info['n_samples']}")
            if 'n_features' in info:
                print(f"  Features: {info['n_features']}")
            if 'target' in info:
                print(f"  Target: {info['target']}")
            print()
        except Exception as e:
            print(f"Error getting info for {dataset_name}: {str(e)}")
    
    # Prompt user to load a dataset
    print("\nWould you like to load a dataset? Enter the name or leave blank to exit:")
    dataset_input = input("> ")
    
    if dataset_input.strip():
        try:
            print(f"\nLoading dataset: {dataset_input}")
            data = fetch_data(dataset_input)
            print(f"Successfully loaded dataset with shape: {data.shape}")
            print("\nColumn names:")
            print(", ".join(data.columns))
            
            print("\nSample data (first 5 rows):")
            print(data.head())
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    main()