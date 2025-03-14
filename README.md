# Mode Choice Benchmarking Sandbox (MCBS)

A comprehensive Python package for benchmarking, analyzing, and validating discrete choice models for transportation mode choice analysis. MCBS provides a unified framework for comparing different model specifications, conducting sensitivity analysis, and visualizing results.

## Installation

You can install MCBS using pip:

```bash
pip install mcbs
```

## Key Features

### Model Estimation and Benchmarking

- **Multiple Model Types**: Support for Multinomial Logit (MNL), Nested Logit (NL), and Mixed Logit (ML) models
- **Standardized Metrics**: Compare models using log-likelihood, rho-squared, prediction accuracy, and market share
- **Cross-validation**: Evaluate model performance on training and testing datasets
- **Visualization**: Generate comparative plots showing model performance across different metrics

```python
from mcbs.benchmarking import Benchmark
from mcbs.datasets import fetch_data

# Load dataset (automatically downloads if not in local cache)
data = fetch_data("swissmetro_dataset")

# Define models and run benchmark
benchmark = Benchmark("swissmetro_dataset")
results = benchmark.run(models)
benchmark.compare_results(results)
```

### Advanced Analysis Capabilities

- **Sensitivity Analysis**: Evaluate how model predictions change with variations in key variables
  - Simulate changes in travel times, costs, and other attributes
  - Generate plots showing the evolution of market shares under different scenarios

- **Individual-Level Parameters**: For Mixed Logit models, calculate and visualize:
  - Individual-specific parameter distributions using Bayesian approaches
  - Value of Time (VOT) distributions across the population
  - Heterogeneity in preference structures

- **Model Calibration**: Automatically calibrate Alternative Specific Constants (ASCs) to match observed market shares

### Visualization and Reporting

- **Market Share Analysis**: Compare predicted vs. observed mode shares
- **Performance Plots**: Visualize how different models perform across multiple datasets
- **Parameter Distributions**: Plot distributions of random parameters and derived metrics like VOT
- **Sensitivity Curves**: Show how predicted mode shares change with variations in key variables

## Supported Datasets

- **Swissmetro** (`swissmetro_dataset`): Swiss inter-city travel mode choice
- **London Transport** (`ltds_dataset`): London Travel Demand Survey with urban mode choices
- **ModeCanada** (`modecanada_dataset`): Canadian inter-city travel dataset

Datasets are automatically downloaded from the [mcbs-datasets](https://github.com/carlosguirado/mcbs-datasets) repository on first use and cached locally:

```python
from mcbs.datasets import fetch_data

# Use default cache location (~/.mcbs/datasets)
data = fetch_data("swissmetro_dataset")

# Specify custom cache location
data = fetch_data("swissmetro_dataset", local_cache_dir="/path/to/cache")

# Get features and target separately
X, y = fetch_data("swissmetro_dataset", return_X_y=True)
```

## Example Applications

### Benchmarking Multiple Models

The package includes tools to benchmark multiple model types across different datasets:

```python
# Run benchmark_all_models.py to compare models across datasets
python benchmark_all_models.py
```

This generates comparative visualizations showing how different model types perform across datasets, plotting metrics like choice accuracy and market share accuracy against model fit.

### Sensitivity Analysis

Analyze how changes in key variables affect predicted mode shares:

```python
# Run sensitivity analysis on ModeCanada models
python sensitivity_analysis.py
```

This creates plots showing the evolution of mode shares as you modify variables like:
- Travel costs for different modes
- Travel times
- Service frequencies

### Individual Parameter Analysis

For Mixed Logit models, analyze individual-level parameters and VOT:

```python
# Generate individual parameter distributions for ModeCanada
python plot_individual_parameters_canada.py
```

This calculates individual-specific parameters using Bayesian conditioning and produces:
- Distributions of time and cost parameters
- Value of Time (VOT) distributions
- Summary statistics for preference heterogeneity

## Requirements

- Python >=3.8
- NumPy >=2.0.0
- Pandas >=2.0.0
- Biogeme >=3.2.14
- Matplotlib >=3.0.0
- Requests >=2.25.0
- SciPy (for statistical functions)
- Seaborn (for advanced visualizations)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Adding New Datasets

To add a new dataset to the MCBS ecosystem:

1. Fork the [mcbs-datasets](https://github.com/carlosguirado/mcbs-datasets) repository
2. Add your dataset following the structure guidelines in the repository's CONTRIBUTING.md file
3. Submit a pull request to the mcbs-datasets repository
4. Update the metadata.json file in the main MCBS package to include your dataset information

This design allows the package to remain lightweight while providing access to a growing collection of transportation mode choice datasets.