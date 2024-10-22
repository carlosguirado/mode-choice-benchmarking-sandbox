# DatasetLoader Class API

The `DatasetLoader` class handles dataset management and loading.

## Methods

### load_dataset()
```python
def load_dataset(self, name: str) -> pd.DataFrame:
    """
    Load a dataset by name.
    
    Args:
        name: Name of the dataset
    
    Returns:
        DataFrame containing the dataset
    """
```

### list_datasets()
```python
def list_datasets(self) -> List[str]:
    """
    List all available datasets.
    
    Returns:
        List of dataset names
    """
```