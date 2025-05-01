#!/usr/bin/env python3
"""
Simple script to list all available datasets in the mcbs-datasets repository.

This script demonstrates how to use the list_available_datasets() function
to discover what datasets are available in the remote repository in real-time.
"""

import sys
from mcbs.datasets import list_available_datasets, get_dataset_info

def main():
    """List all available datasets and display their information."""
    print("Discovering available datasets from the remote repository...\n")
    
    try:
        # Get the list of available datasets
        datasets = list_available_datasets()
        
        if not datasets:
            print("No datasets found in the remote repository.")
            return
        
        print(f"Found {len(datasets)} datasets:\n")
        
        # For each dataset, try to get its metadata
        for dataset_name in datasets:
            print(f"Dataset: {dataset_name}")
            try:
                # Get metadata for the dataset
                info = get_dataset_info(dataset_name)
                
                # Display metadata if available
                if info:
                    if 'description' in info:
                        print(f"  Description: {info['description']}")
                    if 'n_samples' in info:
                        print(f"  Samples: {info['n_samples']}")
                    if 'n_features' in info:
                        print(f"  Features: {info['n_features']}")
                    if 'target' in info:
                        print(f"  Target column: {info['target']}")
                    if 'filename' in info:
                        print(f"  File path: {info['filename']}")
                else:
                    print("  Metadata not available for this dataset")
            except Exception as e:
                print(f"  Error retrieving metadata: {str(e)}")
            
            print()  # Add a blank line between datasets
        
        print("To use any of these datasets, call:")
        print("from mcbs.datasets import fetch_data")
        print("data = fetch_data(\"dataset_name\")")
        
    except Exception as e:
        print(f"Error discovering datasets: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)