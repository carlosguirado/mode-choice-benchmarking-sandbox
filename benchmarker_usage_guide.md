# Using the Mode Choice Benchmarker

This guide explains how to use the benchmarker with new models, including using the built-in Biogeme model wrapper.

## Using Existing Biogeme Models

The package provides a built-in wrapper for Biogeme models. Here's how to use it:

```python
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable
from mcbs.utils import BiogemeModelWrapper
from mcbs.benchmarker.benchmarker import ModelBenchmarker
from mcbs.datasets.dataset_loader import DatasetLoader

# Load data
loader = DatasetLoader()
data = loader.load_dataset("swissmetro_dataset")

# Create and configure your Biogeme model as usual
database = bio.Database('swissmetro', data)
CHOICE = Variable('CHOICE')
TRAIN_TT = Variable('TRAIN_TT')
# ... define other variables

# Define parameters
ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
# ... define other parameters

# Define utilities
V1 = ASC_TRAIN + B_TIME * TRAIN_TT
# ... define other utilities

# Create Biogeme model
logprob = models.logit(V, av, CHOICE)
biogeme = bio.BIOGEME(database, logprob)

# Use the built-in wrapper with benchmarker
model = BiogemeModelWrapper(data, biogeme)
benchmarker = ModelBenchmarker()
results = benchmarker.run_benchmark(
    data=data,
    models=[model],
    dataset_name="swissmetro"
)

# Print and export results
benchmarker.print_comparison()
benchmarker.export_results("biogeme_benchmark_results.csv")
```

## Key Features

The BiogemeModelWrapper automatically handles:

1. Model estimation
2. Extraction of key metrics:
   - Log likelihood
   - Rho squared values
   - Market shares
   - Choice accuracy
   - Confusion matrix

3. Formatting everything for the benchmarker

This makes it easy to benchmark any Biogeme model against existing implementations in the package without needing to rewrite your model code.

## Metrics Explained

The wrapper extracts and calculates several important metrics:

- **Log Likelihood**: Final log likelihood from model estimation
- **Rho Squared**: Goodness of fit measure
- **Market Shares**: Both actual and predicted mode shares
- **Choice Accuracy**: How well the model predicts individual choices
- **Confusion Matrix**: Detailed breakdown of prediction accuracy by mode

These metrics allow for comprehensive model comparison and evaluation.

## Custom Models

If you need to create a custom model that doesn't use Biogeme, you can create a class that inherits from BaseDiscreteChoiceModel:

```python
from mcbs.models.base import BaseDiscreteChoiceModel

class MyCustomModel(BaseDiscreteChoiceModel):
    def estimate(self):
        # Your model estimation code here
        # Must set these attributes:
        self.final_ll = ...  # Final log likelihood
        self.rho_squared = ...  # Rho squared
        self.rho_squared_bar = ...  # Adjusted rho squared
        self.market_share_accuracy = ...  # Market share prediction accuracy
        self.choice_accuracy = ...  # Choice prediction accuracy
        self.actual_shares = ...  # Dictionary of actual mode shares
        self.predicted_shares = ...  # Dictionary of predicted mode shares
        self.confusion_matrix = ...  # Pandas DataFrame of confusion matrix
        
        return self.results  # Return estimation results
```

The benchmarker will work with any model class that implements these required attributes.
