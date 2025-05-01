"""
Model Calibration Script for Swissmetro

This script demonstrates the ASC calibration process:
1. Load Swissmetro data
2. Estimate base models (MNL, NL, ML)
3. Calculate actual market shares
4. Show uncalibrated market shares
5. Calibrate ASCs
6. Show calibrated market shares
"""

from mcbs.datasets import DatasetLoader
from mcbs.models.swissmetro_model import MultinomialLogitModel_SM, NestedLogitModel_SM, MixedLogitModel_SM
import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, MonteCarlo, log

def calculate_actual_shares(data):
    """Calculate actual market shares from the data."""
    # Count choices for each mode
    choice_counts = data[data['CHOICE'] >= 1]['CHOICE'].value_counts()
    total = len(data[data['CHOICE'] >= 1])
    
    # Convert to shares
    actual_shares = {mode: count/total for mode, count in choice_counts.items()}
    
    return actual_shares

def simulate_market_shares(model, data, betas=None):
    """Simulate market shares using provided beta values."""
    # Use provided betas or get from model
    if betas is None:
        betas = model.results.get_beta_values()
    
    # Verify required parameters exist
    required_params = ['B_TIME', 'B_COST', 'ASC_TRAIN', 'ASC_SM', 'ASC_CAR']
    missing_params = [param for param in required_params if param not in betas]
    if missing_params:
        print(f"Missing required parameters: {missing_params}")
        default_values = {
            'B_TIME': 0.0,
            'B_COST': 0.0,
            'ASC_TRAIN': 0.0,
            'ASC_SM': 0.0,
            'ASC_CAR': 0.0}
        
        for param in missing_params:
            betas[param] = default_values[param]
            print(f"Initialised {missing_params} to {default_values[param]}")
            
        return betas
    
    # Get utility functions for simulation
    if isinstance(model, MixedLogitModel_SM):
        # For Mixed Logit, properly handle random parameters
        if 'B_TIME_S' not in betas:
            raise ValueError("Missing B_TIME_S parameter required for Mixed Logit model")
            
        B_TIME = Beta('B_TIME', betas['B_TIME'], None, None, 0)
        B_TIME_S = Beta('B_TIME_S', betas['B_TIME_S'], None, None, 0)
        B_TIME_RND = B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL')
        
        V1 = (betas['ASC_TRAIN'] + 
              B_TIME_RND * model.TRAIN_TT_SCALED + 
              betas['B_COST'] * model.TRAIN_COST_SCALED)
        
        V2 = (betas['ASC_SM'] + 
              B_TIME_RND * model.SM_TT_SCALED + 
              betas['B_COST'] * model.SM_COST_SCALED)
        
        V3 = (betas['ASC_CAR'] + 
              B_TIME_RND * model.CAR_TT_SCALED + 
              betas['B_COST'] * model.CAR_CO_SCALED)
    else:
        # For MNL and NL, use standard utility functions
        V1 = (betas['ASC_TRAIN'] + 
              betas['B_TIME'] * model.TRAIN_TT_SCALED + 
              betas['B_COST'] * model.TRAIN_COST_SCALED)
        
        V2 = (betas['ASC_SM'] + 
              betas['B_TIME'] * model.SM_TT_SCALED + 
              betas['B_COST'] * model.SM_COST_SCALED)
        
        V3 = (betas['ASC_CAR'] + 
              betas['B_TIME'] * model.CAR_TT_SCALED + 
              betas['B_COST'] * model.CAR_CO_SCALED)
    
    # Associate utility functions with alternatives
    V = {1: V1, 2: V2, 3: V3}
    
    # Associate availability conditions
    av = {1: model.TRAIN_AV_SP,
          2: model.SM_AV,
          3: model.CAR_AV_SP}
    
    # Calculate choice probabilities based on model type
    if isinstance(model, NestedLogitModel_SM):
        if not hasattr(model, 'nests'):
            raise ValueError("Nested Logit model missing nests structure")
        prob_train = models.nested(V, av, model.nests, 1)
        prob_sm = models.nested(V, av, model.nests, 2)
        prob_car = models.nested(V, av, model.nests, 3)
    else:  # MNL and Mixed Logit use logit probabilities
        prob_train = models.logit(V, av, 1)
        prob_sm = models.logit(V, av, 2)
        prob_car = models.logit(V, av, 3)
    
    # Setup simulation
    if isinstance(model, MixedLogitModel_SM):
        # For Mixed Logit, integrate over random draws
        simulate = {
            'Prob. train': MonteCarlo(prob_train),
            'Prob. sm': MonteCarlo(prob_sm),
            'Prob. car': MonteCarlo(prob_car)
        }
    else:
        # For MNL and NL, use probabilities directly
        simulate = {
            'Prob. train': prob_train,
            'Prob. sm': prob_sm,
            'Prob. car': prob_car
        }
    
    # Run simulation
    biogeme = bio.BIOGEME(model.database, simulate)
    biogeme.modelName = "market_share_simulation"
    
    # Set number of draws for Mixed Logit
    if isinstance(model, MixedLogitModel_SM):
        biogeme.number_of_draws = 1000
    
    try:
        simulatedValues = biogeme.simulate(betas)
    except Exception as e:
        raise RuntimeError(f"Simulation failed: {str(e)}")
    
    # Calculate market shares (mean probability for each alternative)
    market_shares = {
        1: simulatedValues['Prob. train'].mean(),
        2: simulatedValues['Prob. sm'].mean(),
        3: simulatedValues['Prob. car'].mean()
    }
    
    return market_shares

def calibrate_alternative_constants(model, data, actual_shares, max_iter=20, tolerance=1e-2):
    """Calibrate ASCs to match observed market shares."""
    # Get current beta values
    beta_values = model.results.get_beta_values()
    
    # Initialize ASCs if they don't exist
    if 'ASC_TRAIN' not in beta_values:
        beta_values['ASC_TRAIN'] = 0.0
    if 'ASC_SM' not in beta_values:
        beta_values['ASC_SM'] = 0.0  # Initialize even though it's fixed in estimation
    if 'ASC_CAR' not in beta_values:
        beta_values['ASC_CAR'] = 0.0
        
    # Verify other required parameters exist
    required_params = ['B_TIME', 'B_COST']
    missing_params = [param for param in required_params if param not in beta_values]
    if missing_params:
        raise ValueError(f"Missing required parameters for calibration: {missing_params}")
    
    # Define modes to calibrate
    modes = {
        1: 'ASC_TRAIN',
        2: 'ASC_SM',
        3: 'ASC_CAR'
    }
    
    # Iterative calibration
    for iteration in range(max_iter):
        try:
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
                    
        except Exception as e:
            raise RuntimeError(f"Calibration failed at iteration {iteration + 1}: {str(e)}")
    else:
        print(f"Warning: Maximum iterations ({max_iter}) reached without convergence")
    
    return beta_values

def print_shares_comparison(model_name, actual_shares, before_shares, after_shares):
    """Print a comparison of market shares before and after calibration."""
    print(f"\n{model_name} Market Shares:")
    print("Mode    Actual    Before    After")
    print("-" * 35)
    for mode in [1, 2, 3]:
        print(f"{mode:4d}    {actual_shares[mode]:.3f}     {before_shares[mode]:.3f}     {after_shares[mode]:.3f}")

def main():
    """Run the calibration process."""
    try:
        # Load data
        print("Loading Swissmetro data...")
        loader = DatasetLoader()
        data = loader.load_dataset("swissmetro_dataset")
        
        # Calculate actual market shares
        actual_shares = calculate_actual_shares(data)
        print("\nActual market shares:")
        for mode, share in actual_shares.items():
            print(f"Mode {mode}: {share:.3f}")
        
        # Estimate and calibrate MNL
        print("\nProcessing MNL model...")
        mnl = MultinomialLogitModel_SM(data)
        mnl.estimate()
        mnl_shares_before = simulate_market_shares(mnl, data)
        mnl.results.betas = calibrate_alternative_constants(mnl, data, actual_shares)
        mnl_shares_after = simulate_market_shares(mnl, data, mnl.results.betas)
        print_shares_comparison("MNL", actual_shares, mnl_shares_before, mnl_shares_after)
        
        # Estimate and calibrate NL
        print("\nProcessing NL model...")
        nl = NestedLogitModel_SM(data)
        nl.estimate()
        nl_shares_before = simulate_market_shares(nl, data)
        nl.results.betas = calibrate_alternative_constants(nl, data, actual_shares)
        nl_shares_after = simulate_market_shares(nl, data, nl.results.betas)
        print_shares_comparison("NL", actual_shares, nl_shares_before, nl_shares_after)
        
        # Estimate and calibrate ML
        print("\nProcessing Mixed Logit model...")
        ml = MixedLogitModel_SM(data)
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
                
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
