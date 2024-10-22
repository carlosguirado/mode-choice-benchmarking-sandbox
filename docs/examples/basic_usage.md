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
