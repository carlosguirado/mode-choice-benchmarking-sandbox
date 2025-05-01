"""
Model Calibration Script for LTDS Models

This script provides calibration functions adapted for LTDS dataset structure:
1. Calculate actual market shares
2. Simulate market shares
3. Calibrate alternative-specific constants (ASCs)
"""

import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta

def calculate_actual_shares(data):
    """Calculate actual market shares from the LTDS data."""
    # First encode the travel modes if they haven't been encoded yet
    if data['travel_mode'].dtype == 'object':
        mode_mapping = {
            'walk': 1,
            'cycle': 2,
            'pt': 3,
            'drive': 4
        }
        data = data.copy()
        data['travel_mode'] = data['travel_mode'].map(mode_mapping)
    
    # Count choices for each mode
    choice_counts = data['travel_mode'].value_counts()
    total = len(data)
    
    # Convert to shares with integer keys
    actual_shares = {int(mode): count/total for mode, count in choice_counts.items()}
    
    # Ensure all modes are present
    for mode in [1, 2, 3, 4]:
        if mode not in actual_shares:
            actual_shares[mode] = 0.0
    
    return actual_shares

def simulate_market_shares(model, data, betas=None):
    """Simulate market shares using provided beta values."""
    # Use provided betas or get from model
    if betas is None:
        betas = model.results.get_beta_values()
    
    # Get utility functions for simulation
    V1 = model._get_utility_function(1)
    V2 = model._get_utility_function(2)
    V3 = model._get_utility_function(3)
    V4 = model._get_utility_function(4)
    
    # Associate utility functions with alternatives
    V = {1: V1, 2: V2, 3: V3, 4: V4}
    
    # Associate availability conditions (assuming all modes available)
    av = {1: 1, 2: 1, 3: 1, 4: 1}
    
    # Calculate choice probabilities based on model type
    if hasattr(model, 'nests'):  # Nested Logit
        prob_walk = models.nested(V, av, model.nests, 1)
        prob_cycle = models.nested(V, av, model.nests, 2)
        prob_pt = models.nested(V, av, model.nests, 3)
        prob_drive = models.nested(V, av, model.nests, 4)
    else:  # MNL
        prob_walk = models.logit(V, av, 1)
        prob_cycle = models.logit(V, av, 2)
        prob_pt = models.logit(V, av, 3)
        prob_drive = models.logit(V, av, 4)
    
    # Setup simulation
    simulate = {
        'Prob. walk': prob_walk,
        'Prob. cycle': prob_cycle,
        'Prob. PT': prob_pt,
        'Prob. drive': prob_drive
    }
    
    # Run simulation
    # Create database from current data if not already a database
    if isinstance(data, db.Database):
        database = data
    else:
        database = db.Database("simulation_data", data)

# Run simulation with current data's database
    biogeme = bio.BIOGEME(database, simulate)
    biogeme.modelName = "market_share_simulation"
    simulatedValues = biogeme.simulate(betas)
    
    # Calculate market shares (mean probability for each alternative)
    market_shares = {
        1: simulatedValues['Prob. walk'].mean(),
        2: simulatedValues['Prob. cycle'].mean(),
        3: simulatedValues['Prob. PT'].mean(),
        4: simulatedValues['Prob. drive'].mean()
    }
    
    return market_shares

def calibrate_alternative_constants(model, data, actual_shares, max_iter=20, tolerance=1e-2):
    """Calibrate ASCs to match observed market shares."""
    # Get current beta values
    beta_values = model.results.get_beta_values()
    
    # Define modes to calibrate (including walking)
    modes = {
        1: 'ASC_WALKING',
        2: 'ASC_CYCLING',
        3: 'ASC_PT',
        4: 'ASC_DRIVING'
    }
    
    # Initialize missing ASCs if needed
    for mode_num, asc_name in modes.items():
        if asc_name not in beta_values:
            beta_values[asc_name] = 0.0
    
    # Check if calibration is needed by simulating with current betas
    predicted_shares = simulate_market_shares(model, data, beta_values)
    max_diff = max(abs(actual_shares[mode] - predicted_shares[mode]) 
                  for mode in modes.keys())
    
    print("\nChecking if calibration is needed...")
    print("Current shares:")
    for mode in modes.keys():
        print(f"Mode {mode}: Actual = {actual_shares[mode]:.3f}, "
              f"Predicted = {predicted_shares[mode]:.3f}")
    print(f"Max difference: {max_diff:.6f}")
    
    # If shares already match within tolerance, return current betas
    if max_diff < tolerance:
        print("\nShares already match within tolerance. No calibration needed.")
        return beta_values
    
    print("\nStarting calibration...")
    
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
            print("\nConverged! ASC calibration complete.")
            return beta_values  # Exit immediately when converged
        
        # For nested logit, adjust ASCs considering the nesting structure
        if hasattr(model, 'nests'):
            # Get nesting parameter
            mu = beta_values.get('MU_PUBLIC', 1.0)
            # Adjust ASCs with nesting parameter
            for mode_num, asc_name in modes.items():
                if mode_num in actual_shares and mode_num in predicted_shares:
                    # For modes in the motorized nest (PT and car), scale by mu
                    if mode_num in [3, 4]:  # PT and car are in motorized nest
                        adjustment = mu * np.log(actual_shares[mode_num] / predicted_shares[mode_num])
                    else:  # Walk and cycle are not nested
                        adjustment = np.log(actual_shares[mode_num] / predicted_shares[mode_num])
                    beta_values[asc_name] += adjustment
        else:
            # For MNL, use standard adjustment
            for mode_num, asc_name in modes.items():
                if mode_num in actual_shares and mode_num in predicted_shares:
                    adjustment = np.log(actual_shares[mode_num] / predicted_shares[mode_num])
                    beta_values[asc_name] += adjustment
    
    print("\nWarning: Maximum iterations reached without convergence")
    return beta_values

def print_shares_comparison(model_name, actual_shares, before_shares, after_shares):
    """Print a comparison of market shares before and after calibration."""
    print(f"\n{model_name} Market Shares:")
    print("Mode    Actual    Before    After")
    print("-" * 35)
    for mode in [1, 2, 3, 4]:
        print(f"{mode:4d}    {actual_shares[mode]:.3f}     {before_shares[mode]:.3f}     {after_shares[mode]:.3f}")
