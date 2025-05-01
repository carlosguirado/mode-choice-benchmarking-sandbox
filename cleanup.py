#!/usr/bin/env python3
"""
Master cleanup script for the MCBS package.

This script performs two major cleanup operations:
1. Removes local dataset files (from mcbs/datasets/<dataset_name>/)
2. Removes model output files (.json, .pickle, and .html)

The cleanup helps prepare the package for distribution by making it lightweight.
Essential files like metadata.json are preserved.
"""

import os
import sys
import glob
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Files to preserve (these will not be deleted even if they match the patterns)
PRESERVE_FILES = [
    'metadata.json',
    'pyproject.toml',
    'package.json'
]

def cleanup_datasets(repo_root, dry_run=True):
    """Remove local dataset directories."""
    # Get path to mcbs/datasets directory
    datasets_dir = os.path.join(repo_root, 'mcbs', 'datasets')
    
    # List of dataset directories to remove
    dataset_dirs = ['ltds', 'modecanada', 'swissmetro']
    
    dir_count = 0
    for dir_name in dataset_dirs:
        dir_path = os.path.join(datasets_dir, dir_name)
        if os.path.exists(dir_path):
            if dry_run:
                logger.info(f"Would remove local dataset directory: {dir_path}")
            else:
                logger.info(f"Removing local dataset directory: {dir_path}")
                try:
                    shutil.rmtree(dir_path)
                    dir_count += 1
                    logger.info(f"Successfully removed {dir_path}")
                except Exception as e:
                    logger.error(f"Failed to remove {dir_path}: {str(e)}")
        else:
            logger.info(f"Directory does not exist: {dir_path}")
    
    return dir_count

def cleanup_model_output_files(repo_root, dry_run=True):
    """Remove .json, .pickle, and .html files from the codebase."""
    # Find all files with the specified extensions
    patterns = ['**/*.json', '**/*.pickle', '**/*.html']
    files_to_delete = []
    
    for pattern in patterns:
        path_pattern = os.path.join(repo_root, pattern)
        files_to_delete.extend(glob.glob(path_pattern, recursive=True))
    
    # Filter out files to preserve
    files_to_delete = [f for f in files_to_delete if os.path.basename(f) not in PRESERVE_FILES]
    
    # Count files by extension
    json_count = sum(1 for f in files_to_delete if f.endswith('.json'))
    pickle_count = sum(1 for f in files_to_delete if f.endswith('.pickle'))
    html_count = sum(1 for f in files_to_delete if f.endswith('.html'))
    
    logger.info(f"Found {len(files_to_delete)} files to delete:")
    logger.info(f"  - {json_count} .json files")
    logger.info(f"  - {pickle_count} .pickle files")
    logger.info(f"  - {html_count} .html files")
    
    # Only delete files if not in dry_run mode
    if not dry_run:
        logger.info("Deleting files...")
        removed_count = 0
        
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                removed_count += 1
                # Only log every 100 files to avoid too much output
                if removed_count % 100 == 0:
                    logger.info(f"Removed {removed_count} files...")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {str(e)}")
        
        logger.info(f"Successfully removed {removed_count} files.")
    
    return len(files_to_delete)

def run_cleanup(dry_run=True):
    """Run the complete cleanup process."""
    # Get the repository root directory
    repo_root = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Repository root: {repo_root}")
    
    # Cleanup datasets
    logger.info("\n=== Cleaning up local dataset directories ===")
    num_dirs = cleanup_datasets(repo_root, dry_run)
    
    # Cleanup model output files
    logger.info("\n=== Cleaning up model output files ===")
    num_files = cleanup_model_output_files(repo_root, dry_run)
    
    if dry_run:
        logger.info("\n=== DRY RUN SUMMARY ===")
        logger.info(f"Would remove {num_dirs} dataset directories")
        logger.info(f"Would remove {num_files} model output files")
        logger.info("To actually remove these files, run with --force")
    else:
        logger.info("\n=== CLEANUP SUMMARY ===")
        logger.info(f"Removed {num_dirs} dataset directories")
        logger.info(f"Removed {num_files} model output files")
    
    logger.info("\nCleanup complete.")

if __name__ == "__main__":
    # Check if the --force flag is provided
    force = False
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        force = True

    if not force:
        print("This script will perform two cleanup operations:")
        print("1. Remove local dataset directories (mcbs/datasets/ltds, modecanada, swissmetro)")
        print("2. Remove model output files (.json, .pickle, .html)")
        print("\nEssential files like metadata.json will be preserved.")
        print("This is a dry run. No files will be deleted.")
        
        user_confirm = input("\nDo you want to continue with the dry run? (y/n): ")
        
        if user_confirm.lower() == 'y':
            # Show what would be deleted without actually deleting
            run_cleanup(dry_run=True)
        else:
            logger.info("Cleanup cancelled.")
            sys.exit(0)
    else:
        # Actually delete the files
        print("WARNING: Running in force mode. Files will be deleted.")
        user_confirm = input("Are you sure you want to continue? This cannot be undone. (y/n): ")
        
        if user_confirm.lower() == 'y':
            run_cleanup(dry_run=False)
        else:
            logger.info("Cleanup cancelled.")
            sys.exit(0)