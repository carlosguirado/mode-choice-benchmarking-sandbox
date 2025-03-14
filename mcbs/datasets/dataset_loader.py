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
GITHUB_URL = 'https://raw.githubusercontent.com/carlosguirado/mcbs-datasets/main/datasets'
DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), '.mcbs', 'datasets')

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
        self.metadata_path = os.path.join(self.datasets_path, 'metadata.json')
        self.datasets_metadata = self._load_metadata()
        self.use_local_cache = use_local_cache
        self.local_cache_dir = local_cache_dir if local_cache_dir else DEFAULT_CACHE_DIR

        # Create cache directory if it doesn't exist and caching is enabled
        if self.use_local_cache and not os.path.exists(self.local_cache_dir):
            os.makedirs(self.local_cache_dir, exist_ok=True)
            logger.info(f"Created local cache directory at {self.local_cache_dir}")

    def _load_metadata(self) -> Dict[str, Any]:
        """Load the metadata file containing dataset information."""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {self.metadata_path}. Please check the file format.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

    def fetch_data(self, dataset_name: str, return_X_y: bool = False, dropna: bool = True) -> pd.DataFrame:
        """Download a dataset, optionally store it locally, and return it.
        
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
        if dataset_name not in self.datasets_metadata:
            raise ValueError(f"Dataset '{dataset_name}' not recognized. Available datasets: {', '.join(self.datasets_metadata.keys())}")
        
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
                raise ValueError(f"Dataset '{dataset_name}' is missing target column information in metadata.")
                
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
            # If download fails, try to load from local package directory as fallback
            logger.warning(f"Error downloading dataset: {str(e)}. Trying local fallback...")
            local_path = os.path.join(self.datasets_path, filename)
            if os.path.exists(local_path):
                logger.info(f"Using local fallback file: {local_path}")
                return self._load_file(local_path)
            else:
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
        if name not in self.datasets_metadata:
            raise ValueError(f"Dataset '{name}' not recognized. Available datasets: {', '.join(self.datasets_metadata.keys())}")
        
        return self.datasets_metadata[name]

    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self.datasets_metadata.keys())

    def get_all_datasets_info(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all datasets."""
        return self.datasets_metadata

logger.info(f"DatasetLoader initialized. Available methods: {', '.join(method for method in dir(DatasetLoader) if not method.startswith('_'))}")