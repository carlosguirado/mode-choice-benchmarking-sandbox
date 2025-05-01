#!/usr/bin/env python3
"""
Simple verification script to check that the package structure is correct.
"""

import os
import sys

def check_package_structure():
    """Check that the package structure is correct after cleanup."""
    print("Checking package structure after cleanup...\n")
    
    # Check that the datasets directory exists but not the dataset directories
    datasets_dir = os.path.join('mcbs', 'datasets')
    if not os.path.exists(datasets_dir):
        print(f"Error: {datasets_dir} directory does not exist")
        return False
    
    # Check that metadata.json exists
    metadata_path = os.path.join(datasets_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} does not exist")
        return False
    else:
        print(f"✓ {metadata_path} exists")
    
    # Check that the dataset directories don't exist
    dataset_dirs = ['ltds', 'modecanada', 'swissmetro']
    for dir_name in dataset_dirs:
        dir_path = os.path.join(datasets_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"Error: {dir_path} still exists after cleanup")
            return False
        else:
            print(f"✓ {dir_path} has been removed")
    
    # Check that the necessary Python files exist
    required_files = [
        os.path.join(datasets_dir, '__init__.py'),
        os.path.join(datasets_dir, 'dataset_loader.py')
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: {file_path} does not exist")
            return False
        else:
            print(f"✓ {file_path} exists")
    
    print("\nPackage structure is correct!")
    return True

if __name__ == "__main__":
    success = check_package_structure()
    sys.exit(0 if success else 1)