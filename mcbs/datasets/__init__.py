from .dataset_loader import DatasetLoader
from typing import Optional, Tuple, Union
import pandas as pd

def fetch_data(dataset_name: str, return_X_y: bool = False, 
              local_cache_dir: Optional[str] = None, dropna: bool = True) -> Union[pd.DataFrame, Tuple]:
    """Download a dataset from the MCBS repository, optionally store it locally, and return it.
    
    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load (e.g., "swissmetro_dataset", "ltds_dataset", "modecanada_dataset")
    return_X_y : bool, default=False
        Whether to return the data split into features and target.
    local_cache_dir : str, optional
        Directory to use for caching datasets. If None, uses ~/.mcbs/datasets
    dropna : bool, default=True
        Whether to drop rows with NA values.
        
    Returns
    -------
    dataset : DataFrame or tuple
        If return_X_y is False, returns the full DataFrame.
        If return_X_y is True, returns a tuple (X, y) of features and target.
    """
    loader = DatasetLoader(use_local_cache=(local_cache_dir is not None), 
                          local_cache_dir=local_cache_dir)
    return loader.fetch_data(dataset_name, return_X_y=return_X_y, dropna=dropna)

__all__ = ['DatasetLoader', 'fetch_data']