from .dataset_loader import DatasetLoader
from typing import Optional, Tuple, Union, List
import pandas as pd

def fetch_data(dataset_name: str, return_X_y: bool = False, 
              local_cache_dir: Optional[str] = None, dropna: bool = True) -> Union[pd.DataFrame, Tuple]:
    """Download a dataset from the MCBS remote repository, optionally store it locally, and return it.
    
    This function retrieves datasets from the remote GitHub repository (https://github.com/carlosguirado/mcbs-datasets)
    rather than including them in the package itself. This approach keeps the package lightweight
    while still providing easy access to all datasets.
    
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

def list_available_datasets() -> List[str]:
    """Get a real-time list of all available datasets from the remote repository.
    
    This function provides an up-to-date list of all datasets available in the remote repository,
    including datasets that might have been added after this version of MCBS was released.
    
    Returns
    -------
    List[str]
        List of available dataset names
    """
    loader = DatasetLoader()
    return loader.list_datasets()

def get_dataset_info(dataset_name: str):
    """Get metadata information about a specific dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of the dataset to get information about
        
    Returns
    -------
    dict
        Dictionary containing metadata about the dataset
    """
    loader = DatasetLoader()
    return loader.get_dataset_info(dataset_name)

__all__ = ['DatasetLoader', 'fetch_data', 'list_available_datasets', 'get_dataset_info']