import os
import json
import pandas as pd
import urllib.request
from typing import List, Dict, Tuple

class DatasetLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.dataset_index: Dict[str, Dict] = {}
        self._load_dataset_index()

    def _load_dataset_index(self):
        index_path = os.path.join(self.data_dir, "datasets.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Dataset index file not found at {index_path}")
        
        with open(index_path, 'r') as f:
            self.dataset_index = json.load(f)

    def list_datasets(self) -> List[str]:
        return list(self.dataset_index.keys())

    def get_dataset_info(self, dataset_name: str) -> Dict:
        return self.dataset_index.get(dataset_name, {})

    def load_dataset(self, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        if dataset_name not in self.dataset_index:
            raise ValueError(f"Dataset '{dataset_name}' not found in the index.")

        dataset_info = self.dataset_index[dataset_name]
        file_path = os.path.join(self.data_dir, f"{dataset_name}.csv")

        if not os.path.exists(file_path):
            self._download_dataset(dataset_name, dataset_info['url'], file_path)

        df = pd.read_csv(file_path)
        target = dataset_info['target']
        X = df.drop(columns=[target])
        y = df[target]

        return X, y

    def _download_dataset(self, dataset_name: str, url: str, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"Downloading dataset '{dataset_name}'...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Dataset '{dataset_name}' downloaded successfully.")

# Example usage
if __name__ == "__main__":
    loader = DatasetLoader()
    print("Available datasets:", loader.list_datasets())
    for dataset_name in loader.list_datasets():
        info = loader.get_dataset_info(dataset_name)
        print(f"\nDataset: {dataset_name}")
        print(f"Description: {info['description']}")
        print(f"Number of samples: {info['n_samples']}")
        print(f"Number of features: {info['n_features']}")
        print(f"Task: {info['task']}")