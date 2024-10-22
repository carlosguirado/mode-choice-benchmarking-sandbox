# Benchmark Class API

The `Benchmark` class is the main interface for running model comparisons.

## Constructor
```python
benchmark = Benchmark(dataset_name: str)
```

## Methods

### run()
```python
def run(self, models: Dict[str, Callable]) -> pd.DataFrame:
    """
    Run the benchmark for multiple Biogeme models.
    
    Args:
        models: Dictionary of model names and their functions
    
    Returns:
        DataFrame with benchmark results
    """
```

### compare_results()
```python
def compare_results(self, results: pd.DataFrame) -> None:
    """
    Compare and display results from multiple benchmark runs.
    """
```