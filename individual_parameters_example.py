# examples/individual_parameters_example.py

import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Expression, Beta, bioDraws

from mcbs.datasets import DatasetLoader
from mcbs.utils.individual_parameters import SwissmetroIndividualCalculator, plot_individual_parameters
from mcbs.benchmarking import Benchmark

def run_simple_model(data: pd.DataFrame):
    """Run a simple MNL model for demonstration"""
    # Create database
    database = db.Database("swissmetro_dataset", data)
    
    # Define variables
    CAR_TT = Expression('CAR_TT', database.data['CAR_TT'])
    TRAIN_TT = Expression('TRAIN_TT', database.data['TRAIN_TT'])
    SM_TT = Expression('SM_TT', database.data['SM_TT'])
    CAR_AV = Expression('CAR_AV_SP', database.data['CAR_AV_SP'])
    TRAIN_AV = Expression('TRAIN_AV_SP', database.data['TRAIN_AV_SP'])
    SM_AV = Expression('SM_AV', database.data['SM_AV'])
    CHOICE = Expression('CHOICE', database.data['CHOICE'])
    
    # Parameters
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    
    # Utilities
    V_CAR = ASC_CAR + B_TIME * CAR_TT
    V_TRAIN = ASC_TRAIN + B_TIME * TRAIN_TT
    V_SM = B_TIME * SM_TT
    
    # Associate utilities with alternatives
    av = {1: CAR_AV,
          2: TRAIN_AV,
          3: SM_AV}
    
    # Create logit model
    logprob = bio.bioLogLogit(V_CAR, V_TRAIN, V_SM, av, CHOICE)
    
    # Create and estimate model
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = "simple_mnl"
    
    results = biogeme.estimate()
    print("Estimation results:")
    print(results.shortSummary())
    return results

def main():
    # Load data
    loader = DatasetLoader()
    data = loader.load_dataset("swissmetro_dataset")
    
    # Run simple model
    model_results = run_simple_model(data)
    
    # Calculate individual parameters
    calculator = SwissmetroIndividualCalculator(
        model_results=model_results,
        data=data,
        choice_col='CHOICE',
        parameter_name='B_TIME'
    )
    
    # Get individual parameters
    individual_betas = calculator.calculate_individual_parameters()
    
    # Plot results 
    plot_individual_parameters(
        individual_betas,
        data['CHOICE'],
        'Distribution of Individual Time Parameters'
    )
    
    # Print summary statistics by chosen mode
    print("\nMean parameter values by chosen mode:")
    print(data.groupby('CHOICE')['B_TIME_individual'].mean())

if __name__ == '__main__':
    main()