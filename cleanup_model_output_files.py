#!/usr/bin/env python3
"""
Cleanup script to remove model output files (.json, .pickle, and .html).

This script removes the model output files that are generated during model training and
testing but are not needed for package distribution. It preserves essential files like
metadata.json.
"""

import os
import sys
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Files to preserve (these will not be deleted even if they match the patterns)
PRESERVE_FILES = [
    'metadata.json',
    'pyproject.toml'
]

def cleanup_model_output_files(dry_run=True):
    """
    Remove .json, .pickle, and .html files from the codebase.
    
    Args:
        dry_run (bool): If True, only show what would be deleted without actually deleting.
    """
    # Get the repository root directory
    repo_root = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Repository root: {repo_root}")
    
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
    if dry_run:
        logger.info("DRY RUN: No files will be deleted.")
        logger.info("To actually delete the files, run with --force")
    else:
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
    
    logger.info("Cleanup complete.")
    return len(files_to_delete)

if __name__ == "__main__":
    # Check if the --force flag is provided
    force = False
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        force = True

    if not force:
        user_confirm = input("This will remove all model output files (.json, .pickle, .html) "
                            "except for essential files like metadata.json. "
                            "Run with --force to actually delete files or continue? (y/n): ")
        
        if user_confirm.lower() == 'y':
            # Show what would be deleted without actually deleting
            num_files = cleanup_model_output_files(dry_run=True)
            print(f"\nWould delete {num_files} files.")
            print("To actually delete the files, run again with --force")
        else:
            logger.info("Cleanup cancelled.")
            sys.exit(0)
    else:
        # Actually delete the files
        cleanup_model_output_files(dry_run=False)