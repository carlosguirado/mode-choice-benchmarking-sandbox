# Installation Guide

## From PyPI
```bash
pip install mcbs
```

## Development Installation
```bash
git clone https://github.com/carlosguirado/mode-choice-benchmarking-sandbox.git
cd mode-choice-benchmarking-sandbox
pip install -e .
```

# docs/examples/basic_usage.md
# Basic Usage Example

This example demonstrates how to use MCBS to benchmark a simple multinomial logit model using the Swissmetro dataset.

```python
from mcbs.benchmarking import Benchmark
from biogeme import models
import pandas as pd

# Initialize benchmark
benchmark = Benchmark("swissmetro_dataset")

# Define a simple MNL model
def simple_mnl_model(data):
    # Model specification
    V_TRAIN = beta_cost * TRAIN_COST + beta_time * TRAIN_TIME
    V_SM = beta_cost * SM_COST + beta_time * SM_TIME
    V_CAR = ASC_CAR + beta_cost * CAR_CO + beta_time * CAR_TIME

    # Create and estimate model
    model = models.logit(V=[V_TRAIN, V_SM, V_CAR], 
                        av=[TRAIN_AV, SM_AV, CAR_AV], 
                        choices=CHOICE)
    results = model.estimate()
    return results

# Run benchmark
models = {
    "Simple MNL": simple_mnl_model,
}
results = benchmark.run(models)

# View results
benchmark.compare_results(results)
```

# docs/examples/advanced_usage.md
# Advanced Usage

Learn how to use MCBS for more complex scenarios:

## Purpose-Specific Models
```python
def purpose_specific_model(data):
    # Filter for specific trip purpose
    commuter_data = data[data['PURPOSE'] == 1]
    
    # Model specification
    ...
```

## Custom Metrics
```python
def custom_metric(results):
    return {
        'my_metric': calculate_my_metric(results)
    }

benchmark.run(models, additional_metrics=[custom_metric])
```

# docs/api/benchmark.md
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

# docs/api/dataset_loader.md
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