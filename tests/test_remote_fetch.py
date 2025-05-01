#!/usr/bin/env python3
"""
Test script to verify that remote data fetching still works after cleanup.
"""

import sys
from mcbs.datasets import list_available_datasets, fetch_data

def test_remote_fetching():
    """Test the remote data fetching functionality."""
    print("Testing remote data fetching functionality...\n")
    
    # List available datasets
    print("Listing available datasets:")
    try:
        datasets = list_available_datasets()
        print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    except Exception as e:
        print(f"Error listing available datasets: {str(e)}")
        return False
    
    # Try to fetch the swissmetro dataset
    print("\nFetching swissmetro_dataset:")
    try:
        data = fetch_data("swissmetro_dataset")
        print(f"Successfully fetched dataset with shape: {data.shape}")
    except Exception as e:
        print(f"Error fetching swissmetro_dataset: {str(e)}")
        return False
    
    print("\nRemote data fetching is working correctly!")
    return True

if __name__ == "__main__":
    success = test_remote_fetching()
    sys.exit(0 if success else 1)