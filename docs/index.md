 Mode Choice Benchmarking Sandbox (MCBS)

MCBS is a specialized Python package designed to streamline the development and evaluation of transportation mode choice models. It provides researchers and practitioners with a standardized environment for implementing, testing, and comparing different discrete choice modeling approaches.

## Why MCBS?

Transportation mode choice modeling is crucial for:
- Understanding travel behavior
- Evaluating transportation policies
- Planning infrastructure investments
- Forecasting travel demand

However, comparing different modeling approaches can be challenging due to:
- Inconsistent data preprocessing
- Varying evaluation metrics
- Different implementation approaches
- Lack of standardized benchmarks

MCBS addresses these challenges by providing a unified framework for model development and evaluation.

## Key Features

- **Standardized Datasets**
  - Pre-processed, ready-to-use transportation datasets
  - Consistent formatting and variable definitions
  - Built-in data validation and quality checks

- **Comprehensive Benchmarking**
  - Standardized evaluation metrics
  - Automated model comparison tools
  - Performance visualization utilities
  - Cross-validation support

- **Biogeme Integration**
  - Seamless integration with Biogeme estimation
  - Support for common model specifications
  - Easy extension for custom models

- **Robust Evaluation Metrics**
  - Model fit statistics
  - Prediction accuracy measures
  - Market share analysis
  - Cross-validation metrics

## Quick Start

Install MCBS using pip:
```bash
pip install mcbs
```

Basic usage example:
```python
from mcbs import Benchmark

# Initialize benchmark with dataset
benchmark = Benchmark("swissmetro_dataset")

# Run your models
results = benchmark.run(models)
benchmark.compare_results(results)
```

## Available Datasets

|
 Dataset 
|
 Description 
|
 Observations 
|
 Features 
|
 Use Case 
|
|
---------
|
-------------
|
--------------
|
----------
|
-----------
|
|
 Swissmetro 
|
 Swiss high-speed rail survey 
|
 10,728 
|
 9 
|
 Inter-city mode choice 
|
|
 Mode Choice 
|
 SF Bay Area travel survey 
|
 5,029 
|
 12 
|
 Urban mode choice 
|

## Documentation

- [Installation Guide](installation): Detailed setup instructions and dependencies
- [API Reference](api): Complete documentation of all MCBS components
- [Examples](examples): Tutorials and example model implementations
- [Dataset Documentation](datasets): Detailed dataset descriptions and statistics

## Getting Help

- Pending -

## Contributing
-Pending, ideas below-

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Code style guidelines
- Development setup
- Pull request process
- Adding new datasets
- Implementing new features

## Citation

If you use MCBS in your research, please cite:
```bibtex
@software{mcbs2024,
  title = {Mode Choice Benchmarking Sandbox},
  author = {Guirado, Carlos; Wang, Shenhao; Cheng, Zhanhong; Walker, Joan},
  year = {2024},
  url = {https://github.com/carlosguirado/mode-choice-benchmarking-sandbox}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.