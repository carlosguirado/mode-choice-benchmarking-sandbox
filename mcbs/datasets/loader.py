import os
import json
import pandas as pd
import gzip
import logging
from typing import List, Dict, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetLoader:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.dataset_index: Dict[str, Dict] = {}
        try:
            self._load_dataset_index()
        except FileNotFoundError as e:
            logger.error(f"Failed to load dataset index: {e}")
            raise

    def _load_dataset_index(self):
        index_path = os.path.join(self.data_dir, "datasets.json")
        if not os.path.exists(index_path):
            logger.error(f"Dataset index file not found at {index_path}")
            raise FileNotFoundError(f"Dataset index file not found at {index_path}")
        
        try:
            with open(index_path, 'r') as f:
                self.dataset_index = json.load(f)
            logger.info("Successfully loaded dataset index")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse dataset index JSON: {e}")
            raise

    def list_datasets(self) -> List[str]:
        return list(self.dataset_index.keys())

    def get_dataset_info(self, dataset_name: str) -> Dict:
        if dataset_name not in self.dataset_index:
            logger.warning(f"Requested dataset '{dataset_name}' not found in index")
        return self.dataset_index.get(dataset_name, {})

    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        if dataset_name not in self.dataset_index:
            logger.error(f"Dataset '{dataset_name}' not found in the index")
            raise ValueError(f"Dataset '{dataset_name}' not found in the index")

        dataset_info = self.dataset_index[dataset_name]
        file_path = os.path.join(self.data_dir, dataset_info['file'])

        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found at {file_path}")
            raise FileNotFoundError(f"Dataset file not found at {file_path}")

        try:
            # Check if the file is gzipped
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt') as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(file_path)
            
            target = dataset_info['target']
            X = df.drop(columns=[target])
            y = df[target]
            
            logger.info(f"Successfully loaded dataset '{dataset_name}'")
            return X, y
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        loader = DatasetLoader()
        print("Available datasets:", loader.list_datasets())
        for dataset_name in loader.list_datasets():
            info = loader.get_dataset_info(dataset_name)
            print(f"\nDataset: {dataset_name}")
            print(f"Description: {info['description']}")
            print(f"Number of samples: {info['n_samples']}")
            print(f"Number of features: {info['n_features']}")
            print(f"Task: {info['task']}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")