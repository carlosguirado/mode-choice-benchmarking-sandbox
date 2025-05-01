# Data Fetching in MCBS

## Overview

The Mode Choice Benchmarking Sandbox (MCBS) uses a remote data fetching approach to keep the package lightweight while still providing easy access to all datasets. This document explains how this works and how to use it in your code.

## How It Works

1. **No Datasets Included**: MCBS doesn't include dataset files in the package itself, making the installation much smaller and faster.

2. **Remote Repository**: All datasets are hosted in a dedicated GitHub repository: [mcbs-datasets](https://github.com/carlosguirado/mcbs-datasets)

3. **On-Demand Fetching**: When you request a dataset, MCBS:
   - First checks if it's available in your local cache
   - If not, downloads it from the remote repository
   - Stores it in your local cache for future use
   - Returns the dataset as a pandas DataFrame

4. **Local Caching**: By default, downloaded datasets are cached in `~/.mcbs/datasets/` to avoid repeated downloads.

5. **Dynamic Dataset Discovery**: MCBS can discover datasets that were added to the remote repository after your version of MCBS was released, allowing access to the latest datasets without updating the package.

## Using the Dataset API

### Basic Dataset Fetching

```python
from mcbs.datasets import fetch_data

# Fetch a dataset
swissmetro_data = fetch_data("swissmetro_dataset")

# Dataset is automatically downloaded if not in cache
ltds_data = fetch_data("ltds_dataset")

# You can also get features and target separately
X, y = fetch_data("modecanada_dataset", return_X_y=True)

# Optionally specify a custom cache location
data = fetch_data("swissmetro_dataset", local_cache_dir="/path/to/custom/cache")

# Disable NA dropping if needed
data_with_na = fetch_data("ltds_dataset", dropna=False)
```

### Discovering Available Datasets

The new dynamic dataset discovery system allows you to get a real-time list of all datasets available in the remote repository, including datasets that might have been added after your version of MCBS was released:

```python
from mcbs.datasets import list_available_datasets

# Get a real-time list of all available datasets
available_datasets = list_available_datasets()
print(f"Available datasets: {available_datasets}")
```

### Getting Dataset Metadata

You can get information about datasets:

```python
from mcbs.datasets import get_dataset_info

# Get metadata for a specific dataset
dataset_info = get_dataset_info("swissmetro_dataset")
print(f"Dataset description: {dataset_info['description']}")
print(f"Number of samples: {dataset_info['n_samples']}")
print(f"Target variable: {dataset_info['target']}")
```

## Available Datasets

The following datasets are included by default:

- `swissmetro_dataset`: Swiss inter-city travel mode choice
- `ltds_dataset`: London Travel Demand Survey with urban mode choices
- `modecanada_dataset`: Canadian inter-city travel dataset

New datasets may be added to the remote repository over time. Use `list_available_datasets()` to see all currently available datasets.

## Automatic Path Inference

If a dataset exists in the remote repository but isn't listed in the metadata, MCBS will attempt to infer its path based on naming conventions. For example, if you request "newdata_dataset", it will look for a file at "newdata/newdata.csv.gz" in the remote repository.

## Technical Details

- Datasets are downloaded using the `requests` library
- Files are cached in the same format as they are stored in the repository (typically gzipped CSV)
- The system handles various file formats (CSV, gzipped CSV, Parquet)
- Dataset discovery uses the GitHub API to list available datasets
- The `DatasetLoader` class handles all the fetching logic and can be customized if needed

## Adding New Datasets

To add a new dataset to MCBS:

1. Fork the [mcbs-datasets](https://github.com/carlosguirado/mcbs-datasets) repository
2. Follow these conventions:
   - Create a directory with the base name (e.g., "mydata")
   - Place the dataset file in that directory with the same name (e.g., "mydata/mydata.csv.gz")
   - Update the metadata.json file with information about your dataset
3. Submit a pull request

Users will be able to access your new dataset using `fetch_data("mydata_dataset")` immediately after it's merged into the repository, without needing to update their MCBS package.

## Example Script

See `dynamic_dataset_example.py` for a complete example of how to:
- Discover all available datasets in real-time
- Display metadata for each dataset
- Interactively load a dataset selected by the user