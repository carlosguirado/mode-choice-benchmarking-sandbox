# example/individual_parameters_example.py

from mcbs.datasets import DatasetLoader
from mcbs.utils.individual_parameters import SwissmetroIndividualCalculator
from mcbs.benchmarking import Benchmark
import biogeme.biogeme as bio
import biogeme.database as db

# Load data
loader = DatasetLoader()
data = loader.load_dataset("swissmetro")

# Run benchmark model first (pseudocode - need actual model setup)
benchmark = Benchmark("swissmetro")
model_results = benchmark.run_model(data)

# Calculate individual parameters
calculator = SwissmetroIndividualCalculator(
    model_results=model_results,
    data=data,
    choice_col='CHOICE',
    parameter_name='BETA_TIME'
)

# Get individual parameters
individual_betas = calculator.calculate_individual_parameters()

# Plot results
plot_individual_parameters(
    individual_betas,
    data['CHOICE'],
    'Distribution of Individual Time Parameters'
)

# Add individual parameters back to dataset
data['BETA_TIME_individual'] = individual_betas

# Print summary statistics by chosen mode
print("\nMean parameter values by chosen mode:")
print(data.groupby('CHOICE')['BETA_TIME_individual'].mean())