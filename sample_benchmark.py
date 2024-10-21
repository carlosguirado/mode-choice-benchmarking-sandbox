# sample_benchmark.py

from mcbs.benchmarking.benchmark import Benchmark
from mcbs.datasets.dataset_loader import DatasetLoader

import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.models as models
from biogeme.expressions import Beta, Variable
import numpy as np

def prepare_data(data):
    """
    Prepare the dataset for estimation by removing rows where CHOICE = 0.
    """
    data_copy = data.copy()
    data_copy = data_copy[data_copy['CHOICE'] != 0]
    print(f"\nSample size after removing CHOICE = 0: {len(data_copy)}")
    print("Choice distribution:")
    print(data_copy['CHOICE'].value_counts())
    return data_copy

def calculate_probabilities(betas, data, utilities):
    """
    Calculate choice probabilities based on utilities and availability conditions.
    """
    exp_utilities = np.exp(utilities)
    availabilities = np.column_stack([
        data['TRAIN_AV'].values,
        data['SM_AV'].values,
        data['CAR_AV'].values
    ])
    
    sum_exp_utilities = np.sum(exp_utilities * availabilities, axis=1, keepdims=True)
    probabilities = exp_utilities / sum_exp_utilities
    probabilities = probabilities * availabilities
    row_sums = probabilities.sum(axis=1)
    probabilities = probabilities / row_sums[:, np.newaxis]
    
    return probabilities

def swissmetro_mnl_model(data):
    """
    Base Multinomial Logit model for the Swissmetro dataset.
    Includes only time and cost variables.
    """
    # Prepare data
    data_copy = prepare_data(data)
    database = db.Database('swissmetro', data_copy)
    
    # Define variables
    CHOICE = Variable('CHOICE')
    
    # Mode-specific variables
    TRAIN_AV = Variable('TRAIN_AV')
    TRAIN_TT = Variable('TRAIN_TT')
    TRAIN_CO = Variable('TRAIN_CO')
    
    CAR_AV = Variable('CAR_AV')
    CAR_TT = Variable('CAR_TT')
    CAR_CO = Variable('CAR_CO')
    
    SM_AV = Variable('SM_AV')
    SM_TT = Variable('SM_TT')
    SM_CO = Variable('SM_CO')

    # Utility expressions with availability conditions
    TRAIN_TT_EXPR = TRAIN_TT * (TRAIN_AV == 1)
    TRAIN_CO_EXPR = TRAIN_CO * (TRAIN_AV == 1)
    CAR_TT_EXPR = CAR_TT * (CAR_AV == 1)
    CAR_CO_EXPR = CAR_CO * (CAR_AV == 1)
    SM_TT_EXPR = SM_TT * (SM_AV == 1)
    SM_CO_EXPR = SM_CO * (SM_AV == 1)

    # Parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    B_COST = Beta('B_COST', 0, None, None, 0)

    # Utility functions
    V = {
        1: ASC_TRAIN + B_TIME * TRAIN_TT_EXPR + B_COST * TRAIN_CO_EXPR,  # Train
        2: B_TIME * SM_TT_EXPR + B_COST * SM_CO_EXPR,                    # Swissmetro (base)
        3: ASC_CAR + B_TIME * CAR_TT_EXPR + B_COST * CAR_CO_EXPR        # Car
    }

    # Availability conditions
    av = {1: TRAIN_AV, 2: SM_AV, 3: CAR_AV}

    # Calculate null log likelihood
    V_null = {1: ASC_TRAIN, 2: 0, 3: ASC_CAR}
    logprob_null = models.loglogit(V_null, av, CHOICE)
    biogeme_null = bio.BIOGEME(database, logprob_null)
    results_null = biogeme_null.estimate()
    null_log_likelihood = float(results_null.data.logLike)

    # Estimate full model
    logprob = models.loglogit(V, av, CHOICE)
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = "base_logit"
    results = biogeme.estimate()

    # Store metrics
    results.data.nullLogLike = null_log_likelihood
    results.data.rhoSquared = 1 - (results.data.logLike / null_log_likelihood)

    # Calculate utilities for prediction
    betas = results.get_beta_values()
    utilities = np.zeros((len(data_copy), 3))
    
    utilities[:, 0] = (betas['ASC_TRAIN'] + 
                      betas['B_TIME'] * data_copy['TRAIN_TT'] * data_copy['TRAIN_AV'] +
                      betas['B_COST'] * data_copy['TRAIN_CO'] * data_copy['TRAIN_AV'])
    utilities[:, 1] = (betas['B_TIME'] * data_copy['SM_TT'] * data_copy['SM_AV'] +
                      betas['B_COST'] * data_copy['SM_CO'] * data_copy['SM_AV'])
    utilities[:, 2] = (betas['ASC_CAR'] +
                      betas['B_TIME'] * data_copy['CAR_TT'] * data_copy['CAR_AV'] +
                      betas['B_COST'] * data_copy['CAR_CO'] * data_copy['CAR_AV'])

    # Calculate probabilities and prepare outputs
    y_pred = calculate_probabilities(betas, data_copy, utilities)
    y_true = data_copy['CHOICE'].values - 1

    return results, y_true, y_pred

def swissmetro_mnl_purpose_model(data):
    """
    Extended Multinomial Logit model for the Swissmetro dataset.
    Includes time, cost, and trip purpose variables.
    """
    # Prepare data
    data_copy = prepare_data(data)
    database = db.Database('swissmetro', data_copy)
    
    # Define variables
    CHOICE = Variable('CHOICE')
    PURPOSE = Variable('PURPOSE')
    
    # Mode-specific variables
    TRAIN_AV = Variable('TRAIN_AV')
    TRAIN_TT = Variable('TRAIN_TT')
    TRAIN_CO = Variable('TRAIN_CO')
    
    CAR_AV = Variable('CAR_AV')
    CAR_TT = Variable('CAR_TT')
    CAR_CO = Variable('CAR_CO')
    
    SM_AV = Variable('SM_AV')
    SM_TT = Variable('SM_TT')
    SM_CO = Variable('SM_CO')

    # Utility expressions with availability conditions
    TRAIN_TT_EXPR = TRAIN_TT * (TRAIN_AV == 1)
    TRAIN_CO_EXPR = TRAIN_CO * (TRAIN_AV == 1)
    CAR_TT_EXPR = CAR_TT * (CAR_AV == 1)
    CAR_CO_EXPR = CAR_CO * (CAR_AV == 1)
    SM_TT_EXPR = SM_TT * (SM_AV == 1)
    SM_CO_EXPR = SM_CO * (SM_AV == 1)

    # Parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
    B_TIME = Beta('B_TIME', 0, None, None, 0)
    B_COST = Beta('B_COST', 0, None, None, 0)
    
    # Purpose-specific parameters
    B_PURPOSE_CAR = Beta('B_PURPOSE_CAR', 0, None, None, 0)
    B_PURPOSE_TRAIN = Beta('B_PURPOSE_TRAIN', 0, None, None, 0)
    B_PURPOSE_SM = Beta('B_PURPOSE_SM', 0, None, None, 0)

    # Utility functions with purpose
    V = {
        1: ASC_TRAIN + B_TIME * TRAIN_TT_EXPR + B_COST * TRAIN_CO_EXPR + B_PURPOSE_TRAIN * (PURPOSE == 1),
        2: B_TIME * SM_TT_EXPR + B_COST * SM_CO_EXPR + B_PURPOSE_SM * (PURPOSE == 1),                    
        3: ASC_CAR + B_TIME * CAR_TT_EXPR + B_COST * CAR_CO_EXPR + B_PURPOSE_CAR * (PURPOSE == 1)        
    }

    # Availability conditions
    av = {1: TRAIN_AV, 2: SM_AV, 3: CAR_AV}

    # Calculate null log likelihood
    V_null = {1: ASC_TRAIN, 2: 0, 3: ASC_CAR}
    logprob_null = models.loglogit(V_null, av, CHOICE)
    biogeme_null = bio.BIOGEME(database, logprob_null)
    results_null = biogeme_null.estimate()
    null_log_likelihood = float(results_null.data.logLike)

    # Estimate full model
    logprob = models.loglogit(V, av, CHOICE)
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = "purpose_logit"
    results = biogeme.estimate()

    # Store metrics
    results.data.nullLogLike = null_log_likelihood
    results.data.rhoSquared = 1 - (results.data.logLike / null_log_likelihood)

    # Calculate utilities for prediction
    betas = results.get_beta_values()
    utilities = np.zeros((len(data_copy), 3))
    purposes = (data_copy['PURPOSE'] == 1).values
    
    utilities[:, 0] = (betas['ASC_TRAIN'] + 
                      betas['B_TIME'] * data_copy['TRAIN_TT'] * data_copy['TRAIN_AV'] +
                      betas['B_COST'] * data_copy['TRAIN_CO'] * data_copy['TRAIN_AV'] +
                      betas['B_PURPOSE_TRAIN'] * purposes)
    utilities[:, 1] = (betas['B_TIME'] * data_copy['SM_TT'] * data_copy['SM_AV'] +
                      betas['B_COST'] * data_copy['SM_CO'] * data_copy['SM_AV'] +
                      betas['B_PURPOSE_SM'] * purposes)
    utilities[:, 2] = (betas['ASC_CAR'] +
                      betas['B_TIME'] * data_copy['CAR_TT'] * data_copy['CAR_AV'] +
                      betas['B_COST'] * data_copy['CAR_CO'] * data_copy['CAR_AV'] +
                      betas['B_PURPOSE_CAR'] * purposes)

    # Calculate probabilities and prepare outputs
    y_pred = calculate_probabilities(betas, data_copy, utilities)
    y_true = data_copy['CHOICE'].values - 1

    return results, y_true, y_pred

def main():
    print("Loading benchmark...")
    benchmark = Benchmark("swissmetro_dataset")
    
    # Define models with descriptive names
    models = {
        "MNL - Base": swissmetro_mnl_model,
        "MNL - With Purpose": swissmetro_mnl_purpose_model
    }
    
    print("Running benchmark...")
    results = benchmark.run(models)
    
    print("Comparing results...")
    benchmark.compare_results(results)
    
    print("Plotting results...")
    benchmark.plot_results(results)

if __name__ == "__main__":
    main()