#!/usr/bin/env python3
"""
Cleanup script to remove local dataset files after packaging.

This script removes the local dataset files that were initially included in the repository
but are no longer needed since all datasets are now fetched from the remote repository.

Run this script after setting up the package to clean up the local datasets.
"""

import os
import shutil
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_datasets():
    """Remove local dataset directories."""
    # Get path to mcbs/datasets directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(current_dir, 'mcbs', 'datasets')
    
    # List of dataset directories to remove
    dataset_dirs = ['ltds', 'modecanada', 'swissmetro']
    
    for dir_name in dataset_dirs:
        dir_path = os.path.join(datasets_dir, dir_name)
        if os.path.exists(dir_path):
            logger.info(f"Removing local dataset directory: {dir_path}")
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Successfully removed {dir_path}")
            except Exception as e:
                logger.error(f"Failed to remove {dir_path}: {str(e)}")
        else:
            logger.info(f"Directory does not exist: {dir_path}")
    
    logger.info("Cleanup complete.")
    logger.info("All datasets will now be fetched from the remote repository as needed.")

if __name__ == "__main__":
    user_confirm = input("This will remove all local dataset files. The package will fetch datasets "
                         "from the remote repository when needed. Continue? (y/n): ")
    
    if user_confirm.lower() == 'y':
        cleanup_datasets()
    else:
        logger.info("Cleanup cancelled.")
        sys.exit(0)