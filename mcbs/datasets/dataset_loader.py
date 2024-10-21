# mcbs/datasets/dataset_loader.py

import pandas as pd
from typing import Dict, Any, List
import os
import json
import logging
import gzip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self):
        self.datasets_path = os.path.dirname(__file__)
        self.metadata_path = os.path.join(self.datasets_path, 'metadata.json')
        self.datasets_metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {self.metadata_path}. Please check the file format.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")

    def load_dataset(self, name: str) -> pd.DataFrame:
        if name not in self.datasets_metadata:
            raise ValueError(f"Dataset '{name}' not recognized. Available datasets: {', '.join(self.datasets_metadata.keys())}")
        
        dataset_info = self.datasets_metadata[name]
        
        if 'filename' not in dataset_info:
            raise ValueError(f"Dataset '{name}' is missing 'filename' in metadata.")
        
        filepath = os.path.join(self.datasets_path, dataset_info['filename'])
        logger.info(f"Attempting to load dataset from: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        _, file_extension = os.path.splitext(filepath)
        
        if file_extension.lower() == '.gz':
            logger.info(f"Loading gzipped CSV file: {filepath}")
            with gzip.open(filepath, 'rt') as f:
                df = pd.read_csv(f)
        elif file_extension.lower() == '.csv':
            logger.info(f"Loading CSV file: {filepath}")
            df = pd.read_csv(filepath)
        elif file_extension.lower() == '.parquet':
            logger.info(f"Loading Parquet file: {filepath}")
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format for dataset '{name}': {file_extension}")
        
        logger.info(f"Successfully loaded dataset '{name}' with shape {df.shape}")
        return df

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        if name not in self.datasets_metadata:
            raise ValueError(f"Dataset '{name}' not recognized. Available datasets: {', '.join(self.datasets_metadata.keys())}")
        
        return self.datasets_metadata[name]

    def list_datasets(self) -> List[str]:
        return list(self.datasets_metadata.keys())

    def get_all_datasets_info(self) -> Dict[str, Dict[str, Any]]:
        return self.datasets_metadata

logger.info(f"DatasetLoader initialized. Available methods: {', '.join(method for method in dir(DatasetLoader) if not method.startswith('_'))}")