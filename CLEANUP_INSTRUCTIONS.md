# Cleanup Instructions for MCBS

This document explains how to complete the process of removing local dataset files and model output files from the repository to make the package lightweight.

## Background

In the updated version of MCBS, we've transitioned to a remote data fetching approach instead of bundling datasets with the package. We've also implemented a cleanup process to remove model output files (`.json`, `.pickle`, and `.html` files) that are generated during model training and testing but shouldn't be included in the package distribution.

## Cleanup Options

You have two main options for cleaning up the repository:

### Option 1: Using the Master Cleanup Script (Recommended)

The `cleanup.py` script handles both dataset removal and model output file cleanup:

```bash
# First run a dry run to see what would be deleted without actually deleting
python cleanup.py

# If you're satisfied with the changes, run with --force to actually delete files
python cleanup.py --force
```

### Option 2: Using Individual Cleanup Scripts

If you prefer to run the cleanup steps separately:

1. **Clean Up Local Dataset Files**:

```bash
# Remove local dataset directories
python cleanup_local_datasets.py
```

2. **Clean Up Model Output Files**:

```bash
# Run a dry run first
python cleanup_model_output_files.py

# Then run with --force to actually delete
python cleanup_model_output_files.py --force
```

## What Gets Cleaned Up

### Dataset Cleanup
- Removes `ltds`, `modecanada`, and `swissmetro` directories under `mcbs/datasets/`
- Preserves the `metadata.json` file which is essential for the package

### Model Output Cleanup
- Removes `.json` files (except `metadata.json` and other essential config files)
- Removes `.pickle` files (model serialization files)
- Removes `.html` files (Biogeme output reports)

## Verification Steps

After completing the cleanup, verify that:

1. Dataset directories have been removed:

```bash
ls -la mcbs/datasets/
```

2. Model output files have been removed:

```bash
find . -name "*.json" | grep -v metadata.json
find . -name "*.pickle"
find . -name "*.html"
```

3. The package size is significantly reduced:

```bash
du -h -d 1 .
```

4. Remote dataset fetching still works correctly:

```python
from mcbs.datasets import fetch_data, list_available_datasets

# Should list available datasets from remote repository
print(list_available_datasets())

# Should download a dataset from remote repository
data = fetch_data("swissmetro_dataset")
print(data.shape)
```

## Package and Distribute

Once everything is working correctly and cleaned up, you can build and distribute the package:

```bash
python -m build
python -m twine upload dist/*
```

## Note for Contributors

If you're contributing to MCBS:
- Datasets should be added to the [mcbs-datasets](https://github.com/carlosguirado/mcbs-datasets) repository, not to the main MCBS package.
- Model output files should always be added to `.gitignore` to prevent them from being committed to the repository.
- When running tests or examples locally, be aware that they might generate output files that should be cleaned up before packaging.