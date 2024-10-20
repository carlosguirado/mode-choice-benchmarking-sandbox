# Mode Choice Benchmarking Sandbox (MCBS)

MCBS is a large collection of curated benchmark datasets for evaluating and comparing econometric and machine learning algorithms in the context of transportation mode choice modeling. 

## Summary

MCBS provides:
- A collection of transportation mode choice datasets;
- Tools for loading and preprocessing datasets;
- Benchmarking utilities for comparing different machine learning models;
- Visualization tools for dataset exploration and result analysis.

## Dataset Format

All datasets in MCBS are stored in a common format:
- CSV files (gzip compressed);
- First row contains column names;
- The dependent variable/endpoint/outcome column is named 'target';
- All other columns are considered features.

## Installation

To install MCBS, run the following command:

```bash
pip install git+https://github.com/carlosguirado/mode-choice-benchmarking-sandbox.git
```

Or clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/mode-choice-benchmarking-sandbox.git
cd mode-choice-benchmarking-sandbox
pip install -e .
```

## Usage

Here's a basic example of how to use MCBS:

```python
from mcbs.datasets.loader import DatasetLoader

# Initialize the dataset loader
loader = DatasetLoader()

# List available datasets
print(loader.list_datasets())

# Load a specific dataset
X, y = loader.load_dataset("example_dataset")

# Now you can use X and y with your favorite machine learning library
# For example, with scikit-learn:
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

## Contributing

We welcome contributions to MCBS! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute.

## Citation
-Details to come-
## Support

MCBS is an open source project developed by Carlos Guirado at the University of California, Berkeley. For support, please open an issue on the GitHub repository.

## License

MCBS is released under the [MIT License](LICENSE).