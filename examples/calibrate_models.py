"""
Model Calibration Script

This script demonstrates the ASC calibration process:
1. Load ModeCanada data
2. Estimate base models (MNL, NL, ML)
3. Calculate actual market shares
4. Show uncalibrated market shares
5. Calibrate ASCs (including CAR)
6. Show calibrated market shares
"""

from mcbs.datasets import DatasetLoader
from mcbs.models.modecanada_model import MultinomialLogitModel_MC, NestedLogitModel3_MC, MixedLogitModel_MC
import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, MonteCarlo, log

def calculate_actual_shares(data):
    """Calculate actual market shares from the data."""
    # Count choices for each mode
    choice_counts = data[data['choice'] == 1]['alt'].value_counts()
    total = len(data[data['choice'] == 1])
    
    # Convert to shares
    mode_mapping = {'train': 1, 'car': 2, 'bus': 3, 'air': 4}
    actual_shares = {mode_mapping[mode]: count/total for mode, count in choice_counts.items()}
    
    return actual_shares

def simulate_market_shares(model, data, betas=None):
    """Simulate market shares using provided beta values."""
    # Use provided betas or get from model
    if betas is None:
        betas = model.results.get_beta_values()
    
    # Get utility functions for simulation
    if isinstance(model, MixedLogitModel_MC):
        # For Mixed Logit, properly handle random parameters
        B_TIME = Beta('B_TIME', betas['B_TIME'], None, None, 0)
        B_TIME_S = Beta('B_TIME_S', betas['B_TIME_S'], None, None, 0)
        B_TIME_RND = B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL')
        
        V1 = (betas['ASC_TRAIN'] + 
              B_TIME_RND * model.TRAIN_TIME + 
              betas['B_COST'] * model.TRAIN_COST)
        
        V2 = (betas.get('ASC_CAR', 0) + 
              B_TIME_RND * model.CAR_TIME + 
              betas['B_COST'] * model.CAR_COST)
        
        V3 = (betas['ASC_BUS'] + 
              B_TIME_RND * model.BUS_TIME + 
              betas['B_COST'] * model.BUS_COST)
        
        V4 = (betas['ASC_AIR'] + 
              B_TIME_RND * model.AIR_TIME + 
              betas['B_COST'] * model.AIR_COST)
    else:
        # For MNL and NL, use standard utility functions
        V1 = (betas['ASC_TRAIN'] + 
              betas['B_TIME'] * model.TRAIN_TIME + 
              betas['B_COST'] * model.TRAIN_COST)
        
        V2 = (betas.get('ASC_CAR', 0) + 
              betas['B_TIME'] * model.CAR_TIME + 
              betas['B_COST'] * model.CAR_COST)
        
        V3 = (betas['ASC_BUS'] + 
              betas['B_TIME'] * model.BUS_TIME + 
              betas['B_COST'] * model.BUS_COST)
        
        V4 = (betas['ASC_AIR'] + 
              betas['B_TIME'] * model.AIR_TIME + 
              betas['B_COST'] * model.AIR_COST)
    
    # Associate utility functions with alternatives
    V = {1: V1, 2: V2, 3: V3, 4: V4}
    
    # Associate availability conditions
    av = {1: model.TRAIN_AV,
          2: model.CAR_AV,
          3: model.BUS_AV,
          4: model.AIR_AV}
    
    # Calculate choice probabilities based on model type
    if isinstance(model, NestedLogitModel3_MC):
        prob_train = models.nested(V, av, model.nests, 1)
        prob_car = models.nested(V, av, model.nests, 2)
        prob_bus = models.nested(V, av, model.nests, 3)
        prob_air = models.nested(V, av, model.nests, 4)
    else:  # MNL and Mixed Logit use logit probabilities
        prob_train = models.logit(V, av, 1)
        prob_car = models.logit(V, av, 2)
        prob_bus = models.logit(V, av, 3)
        prob_air = models.logit(V, av, 4)
    
    # Setup simulation
    if isinstance(model, MixedLogitModel_MC):
        # For Mixed Logit, integrate over random draws
        simulate = {
            'Prob. train': MonteCarlo(prob_train),
            'Prob. car': MonteCarlo(prob_car),
            'Prob. bus': MonteCarlo(prob_bus),
            'Prob. air': MonteCarlo(prob_air)
        }
    else:
        # For MNL and NL, use probabilities directly
        simulate = {
            'Prob. train': prob_train,
            'Prob. car': prob_car,
            'Prob. bus': prob_bus,
            'Prob. air': prob_air
        }
    
    # Run simulation
    biogeme = bio.BIOGEME(model.database, simulate)
    biogeme.modelName = "market_share_simulation"
    
    # Set number of draws for Mixed Logit
    if isinstance(model, MixedLogitModel_MC):
        biogeme.number_of_draws = 1000
    
    # Ensure ASC_CAR is in the betas dictionary
    sim_betas = betas.copy()
    if 'ASC_CAR' not in sim_betas:
        sim_betas['ASC_CAR'] = 0
    
    simulatedValues = biogeme.simulate(sim_betas)
    
    # Calculate market shares (mean probability for each alternative)
    market_shares = {
        1: simulatedValues['Prob. train'].mean(),
        2: simulatedValues['Prob. car'].mean(),
        3: simulatedValues['Prob. bus'].mean(),
        4: simulatedValues['Prob. air'].mean()
    }
    
    return market_shares

def calibrate_alternative_constants(model, data, actual_shares, max_iter=100, tolerance=1e-6):
    """Calibrate ASCs to match observed market shares."""
    # Get current beta values
    beta_values = model.results.get_beta_values()
    
    # Add ASC_CAR if it doesn't exist (it's not estimated, fixed at 0)
    if 'ASC_CAR' not in beta_values:
        beta_values['ASC_CAR'] = 0
    
    # Define modes to calibrate
    modes = {
        1: 'ASC_TRAIN',
        2: 'ASC_CAR',
        3: 'ASC_BUS',
        4: 'ASC_AIR'
    }
    
    # Iterative calibration
    for iteration in range(max_iter):
        # Get current predicted shares
        predicted_shares = simulate_market_shares(model, data, beta_values)
        
        # Calculate maximum absolute difference in shares
        max_diff = max(abs(actual_shares[mode] - predicted_shares[mode]) 
                      for mode in modes.keys())
        
        print(f"\nIteration {iteration + 1}")
        print("Current shares:")
        for mode in modes.keys():
            print(f"Mode {mode}: Actual = {actual_shares[mode]:.3f}, "
                  f"Predicted = {predicted_shares[mode]:.3f}")
        print(f"Max difference: {max_diff:.6f}")
        
        # Check convergence
        if max_diff < tolerance:
            print("Converged!")
            break
        
        # Adjust ASCs
        for mode_num, asc_name in modes.items():
            if mode_num in actual_shares and mode_num in predicted_shares:
                adjustment = np.log(actual_shares[mode_num] / predicted_shares[mode_num])
                beta_values[asc_name] += adjustment
    
    return beta_values

def print_shares_comparison(model_name, actual_shares, before_shares, after_shares):
    """Print a comparison of market shares before and after calibration."""
    print(f"\n{model_name} Market Shares:")
    print("Mode    Actual    Before    After")
    print("-" * 35)
    for mode in [1, 2, 3, 4]:
        print(f"{mode:4d}    {actual_shares[mode]:.3f}     {before_shares[mode]:.3f}     {after_shares[mode]:.3f}")

def main():
    """Run the calibration process."""
    # Load data
    print("Loading ModeCanada data...")
    loader = DatasetLoader()
    data = loader.load_dataset("modecanada_dataset")
    
    # Calculate actual market shares
    actual_shares = calculate_actual_shares(data)
    print("\nActual market shares:")
    for mode, share in actual_shares.items():
        print(f"Mode {mode}: {share:.3f}")
    
    # Estimate and calibrate MNL
    print("\nProcessing MNL model...")
    mnl = MultinomialLogitModel_MC(data)
    mnl.estimate()
    mnl_shares_before = simulate_market_shares(mnl, data)
    mnl.results.betas = calibrate_alternative_constants(mnl, data, actual_shares)
    mnl_shares_after = simulate_market_shares(mnl, data, mnl.results.betas)
    print_shares_comparison("MNL", actual_shares, mnl_shares_before, mnl_shares_after)
    
    # Estimate and calibrate NL
    print("\nProcessing NL model...")
    nl = NestedLogitModel3_MC(data)
    nl.estimate()
    nl_shares_before = simulate_market_shares(nl, data)
    nl.results.betas = calibrate_alternative_constants(nl, data, actual_shares)
    nl_shares_after = simulate_market_shares(nl, data, nl.results.betas)
    print_shares_comparison("NL", actual_shares, nl_shares_before, nl_shares_after)
    
    # Estimate and calibrate ML
    print("\nProcessing Mixed Logit model...")
    ml = MixedLogitModel_MC(data)
    ml.estimate()
    ml_shares_before = simulate_market_shares(ml, data)
    ml.results.betas = calibrate_alternative_constants(ml, data, actual_shares)
    ml_shares_after = simulate_market_shares(ml, data, ml.results.betas)
    print_shares_comparison("ML", actual_shares, ml_shares_before, ml_shares_after)
    
    # Print final ASC values
    print("\nFinal calibrated ASC values:")
    print("\nMNL ASCs:")
    for name, value in mnl.results.betas.items():
        if name.startswith('ASC'):
            print(f"{name}: {value:.3f}")
    
    print("\nNL ASCs:")
    for name, value in nl.results.betas.items():
        if name.startswith('ASC'):
            print(f"{name}: {value:.3f}")
    
    print("\nML ASCs:")
    for name, value in ml.results.betas.items():
        if name.startswith('ASC'):
            print(f"{name}: {value:.3f}")

if __name__ == "__main__":
    main()
