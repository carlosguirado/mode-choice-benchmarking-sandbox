# mcbs/datasets/dataset_loader.py

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import logging
import gzip
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL where datasets are hosted
GITHUB_URL = 'https://raw.githubusercontent.com/carlosguirado/mcbs-datasets/master/datasets'
GITHUB_API_URL = 'https://api.github.com/repos/carlosguirado/mcbs-datasets/contents/datasets'
METADATA_URL = 'https://raw.githubusercontent.com/carlosguirado/mcbs-datasets/master/metadata.json'
DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), '.mcbs', 'datasets')
METADATA_FILENAME = 'metadata.json'

class DatasetLoader:
    def __init__(self, use_local_cache: bool = True, local_cache_dir: Optional[str] = None):
        """Initialize the DatasetLoader.
        
        Parameters
        ----------
        use_local_cache : bool, default=True
            Whether to use local cache for datasets
        local_cache_dir : str, optional
            Directory to use for caching datasets. If None, uses ~/.mcbs/datasets
        """
        self.datasets_path = os.path.dirname(__file__)
        self.metadata_path = os.path.join(self.datasets_path, METADATA_FILENAME)
        self.use_local_cache = use_local_cache
        self.local_cache_dir = local_cache_dir if local_cache_dir else DEFAULT_CACHE_DIR
        
        # Initialize metadata from local file (will be used as fallback)
        self.datasets_metadata = self._load_local_metadata()
        
        # Create cache directory if it doesn't exist and caching is enabled
        if self.use_local_cache and not os.path.exists(self.local_cache_dir):
            os.makedirs(self.local_cache_dir, exist_ok=True)
            logger.info(f"Created local cache directory at {self.local_cache_dir}")

    def _load_local_metadata(self) -> Dict[str, Any]:
        """Load the metadata file containing dataset information from local file."""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Error decoding JSON from {self.metadata_path}. Using empty metadata.")
            return {}
        except FileNotFoundError:
            logger.warning(f"Metadata file not found at {self.metadata_path}. Using empty metadata.")
            return {}

    def fetch_remote_metadata(self) -> Dict[str, Any]:
        """Fetch metadata from the remote repository.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary containing available datasets
        """
        try:
            # Use direct URL to the metadata.json file
            metadata_url = METADATA_URL
            
            # Fetch metadata
            logger.info(f"Fetching metadata from: {metadata_url}")
            response = requests.get(metadata_url)
            response.raise_for_status()
            
            # Parse metadata
            remote_metadata = json.loads(response.text)
            
            # Cache the metadata locally
            if self.use_local_cache:
                cache_path = os.path.join(self.local_cache_dir, METADATA_FILENAME)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump(remote_metadata, f, indent=2)
                logger.info(f"Cached metadata to: {cache_path}")
                
            return remote_metadata
        
        except Exception as e:
            logger.warning(f"Error fetching remote metadata: {str(e)}. Using local metadata as fallback.")
            return self.datasets_metadata

    def discover_available_datasets(self) -> List[str]:
        """Dynamically discover available datasets in the remote repository.
        
        This method uses the GitHub API to list directories in the datasets repository,
        providing real-time information about available datasets even if they
        weren't present when the MCBS package was released.
        
        Returns
        -------
        List[str]
            List of dataset names available in the remote repository
        """
        try:
            # Fetch repository contents using GitHub API
            logger.info(f"Querying GitHub API for available datasets: {GITHUB_API_URL}")
            response = requests.get(GITHUB_API_URL)
            response.raise_for_status()
            
            # Parse response
            contents = response.json()
            
            # Filter for directories only (these should be dataset directories)
            dataset_dirs = [item['name'] for item in contents if item['type'] == 'dir']
            
            # Convert directory names to dataset names
            dataset_names = [f"{dir_name}_dataset" for dir_name in dataset_dirs]
            
            logger.info(f"Discovered {len(dataset_names)} datasets from remote repository")
            return dataset_names
            
        except Exception as e:
            logger.warning(f"Error discovering datasets from remote repository: {str(e)}. "
                           f"Using local metadata as fallback.")
            return list(self.datasets_metadata.keys())

    def get_all_available_datasets(self) -> List[str]:
        """Get a comprehensive list of all available datasets.
        
        This method combines information from:
        1. The local metadata file
        2. The remote metadata file (if accessible)
        3. Dynamic discovery of directories in the repository
        
        Returns
        -------
        List[str]
            List of all available dataset names
        """
        # Start with datasets from local metadata
        all_datasets = set(self.datasets_metadata.keys())
        
        try:
            # Add datasets from remote metadata
            remote_metadata = self.fetch_remote_metadata()
            all_datasets.update(remote_metadata.keys())
            
            # Add datasets from directory discovery
            discovered_datasets = self.discover_available_datasets()
            all_datasets.update(discovered_datasets)
            
        except Exception as e:
            logger.warning(f"Error while trying to get all available datasets: {str(e)}")
        
        return sorted(list(all_datasets))

    def fetch_data(self, dataset_name: str, return_X_y: bool = False, dropna: bool = True) -> pd.DataFrame:
        """Download a dataset from the remote repository, optionally store it locally, and return it.
        
        Parameters
        ----------
        dataset_name : str
            The name of the dataset to load.
        return_X_y : bool, default=False
            Whether to return the data split into features and target.
        dropna : bool, default=True
            Whether to drop rows with NA values.
            
        Returns
        -------
        dataset : DataFrame or tuple
            If return_X_y is False, returns the full DataFrame.
            If return_X_y is True, returns a tuple (X, y) of features and target.
        """
        # Check if dataset exists in local metadata
        if dataset_name not in self.datasets_metadata:
            # Try to fetch remote metadata
            remote_metadata = self.fetch_remote_metadata()
            
            # If dataset is in remote metadata, update local metadata
            if dataset_name in remote_metadata:
                logger.info(f"Found dataset '{dataset_name}' in remote metadata")
                self.datasets_metadata = remote_metadata
            else:
                # Try to infer the dataset path from naming convention
                # Assuming dataset_name format like "swissmetro_dataset"
                if "_dataset" in dataset_name:
                    base_name = dataset_name.replace("_dataset", "")
                    inferred_filename = f"{base_name}/{base_name}.csv.gz"
                    
                    logger.info(f"Dataset '{dataset_name}' not found in metadata. "
                                f"Attempting to infer path: {inferred_filename}")
                    
                    # Create metadata entry with inferred information
                    self.datasets_metadata[dataset_name] = {
                        "filename": inferred_filename,
                        "description": f"Inferred {base_name} dataset",
                        "n_samples": 0,
                        "n_features": 0,
                        "task": "prediction"
                    }
                else:
                    available_datasets = self.get_all_available_datasets()
                    raise ValueError(f"Dataset '{dataset_name}' not recognized. "
                                     f"Available datasets: {', '.join(available_datasets)}")
        
        dataset_info = self.datasets_metadata[dataset_name]
        
        if 'filename' not in dataset_info:
            raise ValueError(f"Dataset '{dataset_name}' is missing 'filename' in metadata.")
        
        filename = dataset_info['filename']
        
        # Check if using local cache and if file exists in cache
        if self.use_local_cache:
            cache_path = os.path.join(self.local_cache_dir, filename)
            cache_dir = os.path.dirname(cache_path)
            
            # Ensure the directory structure exists
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            
            # Use cached file if it exists
            if os.path.exists(cache_path):
                logger.info(f"Loading dataset from local cache: {cache_path}")
                df = self._load_file(cache_path)
            else:
                # Download and cache the file
                logger.info(f"Dataset not found in cache. Downloading from remote source.")
                df = self._download_and_cache_dataset(dataset_name, filename, cache_path)
        else:
            # Directly download without caching
            logger.info(f"Local cache disabled. Downloading dataset from remote source.")
            dataset_url = self._get_dataset_url(filename)
            df = self._download_dataset(dataset_url, filename)
        
        if dropna:
            df = df.dropna()
            
        logger.info(f"Successfully loaded dataset '{dataset_name}' with shape {df.shape}")
        
        if return_X_y:
            target_col = dataset_info.get('target')
            if not target_col:
                # Try to identify a target column
                potential_targets = ['choice', 'CHOICE', 'target', 'label', 'y', 'travel_mode', 'mode']
                for col in potential_targets:
                    if col in df.columns:
                        target_col = col
                        logger.info(f"Target column not specified. Using '{target_col}' as target.")
                        break
                
                if not target_col:
                    raise ValueError(f"Dataset '{dataset_name}' is missing target column information "
                                     f"and none could be inferred from common column names.")
                
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            return X, y
        else:
            return df
            
    def _get_dataset_url(self, filename: str) -> str:
        """Construct the URL for a dataset file."""
        return f"{GITHUB_URL}/{filename}"
            
    def _download_dataset(self, dataset_url: str, filename: str) -> pd.DataFrame:
        """Download a dataset from a URL."""
        logger.info(f"Downloading dataset from: {dataset_url}")
        
        try:
            response = requests.get(dataset_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            _, file_extension = os.path.splitext(filename)
            
            # Handle different file types
            if file_extension.lower() == '.gz':
                import io
                with gzip.open(io.BytesIO(response.content), 'rt') as f:
                    return pd.read_csv(f)
            elif file_extension.lower() == '.csv':
                import io
                return pd.read_csv(io.StringIO(response.text))
            elif file_extension.lower() == '.parquet':
                import io
                return pd.read_parquet(io.BytesIO(response.content))
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error downloading dataset: {str(e)}")
            
    def _download_and_cache_dataset(self, dataset_name: str, filename: str, cache_path: str) -> pd.DataFrame:
        """Download a dataset and save it to the cache directory."""
        dataset_url = self._get_dataset_url(filename)
        
        try:
            # Download the dataset
            df = self._download_dataset(dataset_url, filename)
            
            # Save to cache
            logger.info(f"Saving dataset to cache: {cache_path}")
            self._save_to_cache(df, cache_path)
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading or caching dataset '{dataset_name}': {str(e)}")
            raise
            
    def _save_to_cache(self, df: pd.DataFrame, cache_path: str) -> None:
        """Save a DataFrame to the cache directory."""
        directory = os.path.dirname(cache_path)
        os.makedirs(directory, exist_ok=True)
        
        _, file_extension = os.path.splitext(cache_path)
        
        if file_extension.lower() == '.gz':
            # Handle .csv.gz files
            if cache_path.lower().endswith('.csv.gz'):
                with gzip.open(cache_path, 'wt') as f:
                    df.to_csv(f, index=False)
            else:
                raise ValueError(f"Unsupported compressed format: {file_extension}")
        elif file_extension.lower() == '.csv':
            df.to_csv(cache_path, index=False)
        elif file_extension.lower() == '.parquet':
            df.to_parquet(cache_path, index=False)
        else:
            raise ValueError(f"Unsupported file format for caching: {file_extension}")
            
    def _load_file(self, filepath: str) -> pd.DataFrame:
        """Load a dataset file from disk."""
        _, file_extension = os.path.splitext(filepath)
        
        if file_extension.lower() == '.gz':
            # Handle .csv.gz files
            if filepath.lower().endswith('.csv.gz'):
                with gzip.open(filepath, 'rt') as f:
                    return pd.read_csv(f)
            else:
                raise ValueError(f"Unsupported compressed format: {file_extension}")
        elif file_extension.lower() == '.csv':
            return pd.read_csv(filepath)
        elif file_extension.lower() == '.parquet':
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    # Legacy method for backward compatibility
    def load_dataset(self, name: str) -> pd.DataFrame:
        """Load a dataset (legacy method, uses fetch_data internally)."""
        return self.fetch_data(name, return_X_y=False)

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific dataset."""
        # Try to ensure we have the most up-to-date metadata
        if name not in self.datasets_metadata:
            # Try to fetch from remote
            remote_metadata = self.fetch_remote_metadata()
            if name in remote_metadata:
                self.datasets_metadata = remote_metadata
            else:
                # Check if we can discover this dataset
                all_datasets = self.get_all_available_datasets()
                if name not in all_datasets:
                    raise ValueError(f"Dataset '{name}' not recognized. "
                                    f"Available datasets: {', '.join(all_datasets)}")
            
        return self.datasets_metadata.get(name, {})

    def list_datasets(self) -> List[str]:
        """List all available datasets from both local metadata and remote repository.
        
        This method provides a real-time list of datasets available in the remote repository,
        even if they weren't present when the MCBS package was released.
        
        Returns
        -------
        List[str]
            List of all available dataset names
        """
        return self.get_all_available_datasets()