import unittest
import os
import json
import pandas as pd
from mcbs.datasets.loader import DatasetLoader

class TestDatasetLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary datasets.json file for testing
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        cls.test_datasets = {
            "test_dataset": {
                "file": "test_dataset.csv",
                "description": "Test dataset",
                "target": "target",
                "features": ["feature1", "feature2"],
                "n_samples": 100,
                "n_features": 2,
                "task": "classification"
            }
        }
        
        with open(os.path.join(cls.test_data_dir, 'datasets.json'), 'w') as f:
            json.dump(cls.test_datasets, f)
        
        # Create a test CSV file
        test_df = pd.DataFrame({
            "feature1": range(100),
            "feature2": range(100, 200),
            "target": [0, 1] * 50
        })
        test_df.to_csv(os.path.join(cls.test_data_dir, 'test_dataset.csv'), index=False)

    @classmethod
    def tearDownClass(cls):
        # Remove temporary test files
        os.remove(os.path.join(cls.test_data_dir, 'datasets.json'))
        os.remove(os.path.join(cls.test_data_dir, 'test_dataset.csv'))
        os.rmdir(cls.test_data_dir)

    def setUp(self):
        self.loader = DatasetLoader(data_dir=self.test_data_dir)

    def test_list_datasets(self):
        datasets = self.loader.list_datasets()
        self.assertEqual(datasets, ["test_dataset"])

    def test_get_dataset_info(self):
        info = self.loader.get_dataset_info("test_dataset")
        self.assertEqual(info, self.test_datasets["test_dataset"])

    def test_get_nonexistent_dataset_info(self):
        info = self.loader.get_dataset_info("nonexistent_dataset")
        self.assertEqual(info, {})

    def test_load_dataset(self):
        X, y = self.loader.load_dataset("test_dataset")
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(y.shape, (100,))

    def test_load_nonexistent_dataset(self):
        with self.assertRaises(ValueError):
            self.loader.load_dataset("nonexistent_dataset")

if __name__ == '__main__':
    unittest.main()