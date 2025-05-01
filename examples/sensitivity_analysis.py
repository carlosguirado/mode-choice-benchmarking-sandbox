"""
Sensitivity Analysis Script for ModeCanada Models

This script performs sensitivity analysis by:
1. Estimating models ONCE on original dataset
2. Calibrating ASCs for NL and ML models to match observed shares
3. Creating modified datasets with different cost/time scenarios
4. Simulating using calibrated models on modified datasets
5. Plotting evolution of predicted shares
"""

from mcbs.datasets import DatasetLoader
from mcbs.models.modecanada_model import MultinomialLogitModel_MC, NestedLogitModel3_MC, MixedLogitModel_MC
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, MonteCarlo, log
from calibrate_models import calculate_actual_shares, simulate_market_shares, calibrate_alternative_constants, print_shares_comparison

def modify_dataset(data, modification):
    """
    Create a modified version of the dataset based on specified changes.
    
    Args:
        data (pd.DataFrame): Original dataset
        modification (dict): Dictionary specifying the modifications
            e.g., {'mode': 'bus', 'variable': 'cost', 'change': -0.25}
    
    Returns:
        pd.DataFrame: Modified dataset
    """
    modified_data = data.copy()
    
    mode = modification['mode']
    variable = modification['variable']
    change = modification['change']
    
    # Apply modification only to specified mode
    mask = modified_data['alt'] == mode
    if variable == 'cost':
        modified_data.loc[mask, 'cost'] *= (1 + change)
    elif variable == 'time':
        # Modify both in-vehicle and out-of-vehicle time
        modified_data.loc[mask, 'ivt'] *= (1 + change)
        modified_data.loc[mask, 'ovt'] *= (1 + change)
    
    return modified_data

def estimate_base_models(data):
    """
    Estimate all model types ONCE on the original dataset and calibrate ASCs.
    First estimates uncalibrated models, then calibrates ASCs.
    
    Args:
        data (pd.DataFrame): Original dataset for estimation
    
    Returns:
        tuple: (mnl, nl, ml) Estimated and calibrated model objects
    """
    print("\nEstimating base models on original dataset...")
    
    # Calculate actual market shares
    actual_shares = calculate_actual_shares(data)
    print("\nActual market shares:")
    for mode, share in actual_shares.items():
        print(f"Mode {mode}: {share:.3f}")
    
    # First estimate uncalibrated models
    print("\nEstimating uncalibrated models...")
    
    # MNL
    print("\nEstimating MNL model")
    mnl = MultinomialLogitModel_MC(data)
    mnl.estimate()
    mnl_shares_before = simulate_market_shares(mnl, data)
    print("\nMNL predicted shares before calibration:")
    for mode, share in mnl_shares_before.items():
        print(f"Mode {mode}: {share:.3f}")
    
    # NL
    print("\nEstimating NL model")
    nl = NestedLogitModel3_MC(data)
    nl.estimate()
    nl_shares_before = simulate_market_shares(nl, data)
    print("\nNL predicted shares before calibration:")
    for mode, share in nl_shares_before.items():
        print(f"Mode {mode}: {share:.3f}")
    
    # ML
    print("\nEstimating Mixed Logit model")
    ml = MixedLogitModel_MC(data)
    ml.estimate()
    ml_shares_before = simulate_market_shares(ml, data)
    print("\nML predicted shares before calibration:")
    for mode, share in ml_shares_before.items():
        print(f"Mode {mode}: {share:.3f}")
    
    # Now calibrate ASCs
    print("\nCalibrating models...")
    
    # Calibrate MNL
    print("\nCalibrating MNL model")
    mnl.results.betas = calibrate_alternative_constants(mnl, data, actual_shares)
    mnl_shares_after = simulate_market_shares(mnl, data, mnl.results.betas)
    print_shares_comparison("MNL", actual_shares, mnl_shares_before, mnl_shares_after)
    
    # Calibrate NL
    print("\nCalibrating NL model")
    nl.results.betas = calibrate_alternative_constants(nl, data, actual_shares)
    nl_shares_after = simulate_market_shares(nl, data, nl.results.betas)
    print_shares_comparison("NL", actual_shares, nl_shares_before, nl_shares_after)
    
    # Calibrate ML
    print("\nCalibrating Mixed Logit model")
    ml.results.betas = calibrate_alternative_constants(ml, data, actual_shares)
    ml_shares_after = simulate_market_shares(ml, data, ml.results.betas)
    print_shares_comparison("ML", actual_shares, ml_shares_before, ml_shares_after)
    print(ml.results.betas)
    
    return mnl, nl, ml

def simulate_mixed_logit(model, data):
    """
    Simulate Mixed Logit model with proper random parameter integration.
    
    Args:
        model (MixedLogitModel_MC): Estimated Mixed Logit model
        data (pd.DataFrame): Dataset to simulate on
    
    Returns:
        dict: Dictionary containing predicted shares
    """
    betas = model.results.betas

    print("Betas entering MxlL simulation: ")
    print(betas)
    
    # Define random parameter for time
    B_TIME = Beta('B_TIME', betas['B_TIME'], None, None, 0)
    B_TIME_S = Beta('B_TIME_S', betas['B_TIME_S'], None, None, 0)
    B_TIME_RND = B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL')
    
    # Utility functions with random coefficient
    V1 = (betas['ASC_TRAIN'] + 
          B_TIME_RND * model.TRAIN_TIME + 
          betas['B_COST'] * model.TRAIN_COST)
    
    V2 = (betas['ASC_CAR'] + # fixed to 0
          B_TIME_RND * model.CAR_TIME + 
          betas['B_COST'] * model.CAR_COST)
    
    V3 = (betas['ASC_BUS'] + 
          B_TIME_RND * model.BUS_TIME + 
          betas['B_COST'] * model.BUS_COST)
    
    V4 = (betas['ASC_AIR'] + 
          B_TIME_RND * model.AIR_TIME + 
          betas['B_COST'] * model.AIR_COST)
    
    # Associate utility functions with alternatives
    V = {1: V1, 2: V2, 3: V3, 4: V4}
    
    # Associate availability conditions
    av = {1: model.TRAIN_AV,
          2: model.CAR_AV,
          3: model.BUS_AV,
          4: model.AIR_AV}
    
    # Calculate probabilities for each alternative
    prob_train = models.logit(V, av, 1)
    prob_car = models.logit(V, av, 2)
    prob_bus = models.logit(V, av, 3)
    prob_air = models.logit(V, av, 4)
    
    # Integrate over random parameter using Monte-Carlo
    simulate = {
        'Prob. train': MonteCarlo(prob_train),
        'Prob. car': MonteCarlo(prob_car),
        'Prob. bus': MonteCarlo(prob_bus),
        'Prob. air': MonteCarlo(prob_air)
    }
    
    # Run simulation
    biogeme = bio.BIOGEME(model.database, simulate, number_of_draws=100)
    biogeme.modelName = "mixed_logit_simulation"
    simulatedValues = biogeme.simulate(betas)
    
    # Calculate market shares
    shares = {
        1: simulatedValues['Prob. train'].mean(),
        2: simulatedValues['Prob. car'].mean(),
        3: simulatedValues['Prob. bus'].mean(),
        4: simulatedValues['Prob. air'].mean()
    }
    
    return shares

def simulate_models(data, mnl, nl, ml, scenario_name):
    """
    Simulate using all model types on a given dataset.
    
    Args:
        data (pd.DataFrame): Dataset to simulate on
        mnl (MultinomialLogitModel_MC): Estimated MNL model
        nl (NestedLogitModel3_MC): Estimated NL model
        ml (MixedLogitModel_MC): Estimated Mixed Logit model
        scenario_name (str): Name of the scenario being simulated
    
    Returns:
        dict: Dictionary containing simulation results for all models
    """
    results = {'scenario': scenario_name}
    
    # Create new model instances for simulation but use parameters from estimated models
    mnl_sim = MultinomialLogitModel_MC(data)
    mnl_sim.results = mnl.results
    mnl_shares = simulate_market_shares(mnl_sim, data)
    
    nl_sim = NestedLogitModel3_MC(data)
    nl_sim.results = nl.results
    nl_sim.nests = nl.nests  # Copy nests structure
    nl_shares = simulate_market_shares(nl_sim, data)
    
    # For Mixed Logit, use specialized simulation function
    ml_sim = MixedLogitModel_MC(data)
    ml_sim.results = ml.results
    ml_shares = simulate_mixed_logit(ml_sim, data)
    
    # Store results for each model type
    results['mnl'] = {
        'predicted_shares': mnl_shares,
    }
    
    results['nl'] = {
        'predicted_shares': nl_shares,
    }
    
    results['ml'] = {
        'predicted_shares': ml_shares,
    }
    
    return results

def plot_share_evolution(results, modification_group):
    """
    Create plots showing the evolution of predicted shares for a group of modifications.
    
    Args:
        results (list): List of results dictionaries
        modification_group (dict): Dictionary containing information about the modification group
    """
    mode = modification_group['mode']
    variable = modification_group['variable']
    scenarios = modification_group['scenarios']
    changes = modification_group['changes']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot lines for each model type
    models = ['mnl', 'nl', 'ml']
    model_names = {'mnl': 'MNL', 'nl': 'NL', 'ml': 'Mixed Logit'}
    colors = {'mnl': 'blue', 'nl': 'red', 'ml': 'green'}
    
    # Get baseline shares
    baseline_shares = {
        model: results[0][model]['predicted_shares'][modification_group['mode_num']]
        for model in models
    }
    
    # Convert changes to multipliers for x-axis if this is a cost modification
    if variable == 'cost':
        x_values = [1 + change for change in changes]  # e.g., -0.25 becomes 0.75, 0.1 becomes 1.1
        x_label = f'{mode.capitalize()} Price Multiplier'
    else:
        x_values = changes
        x_label = f'{mode.capitalize()} {variable.capitalize()} Change'
    
    # Plot evolution for each model
    for model in models:
        shares = [baseline_shares[model]]  # Start with baseline
        for scenario in scenarios[1:]:  # Skip baseline
            result = next(r for r in results if r['scenario'] == scenario)
            share = result[model]['predicted_shares'][modification_group['mode_num']]
            shares.append(share)
        
        plt.plot(x_values, shares, marker='o', label=model_names[model], color=colors[model])
    
    # Customize plot
    plt.title(f'Evolution of {mode.capitalize()} Share\nwith {variable.capitalize()} Modifications')
    plt.xlabel(x_label)
    plt.ylabel(f'Predicted {mode.capitalize()} Share')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.savefig(f'share_evolution_{mode}_{variable}.png')
    plt.close()

def run_sensitivity_analysis():
    """Run the complete sensitivity analysis."""
    # Load original dataset
    loader = DatasetLoader()
    data = loader.load_dataset("modecanada_dataset")
    
    # First estimate and calibrate models
    mnl, nl, ml = estimate_base_models(data)
    
    # Define modifications to test
    modifications = [
        # Car time modifications
        {'mode': 'car', 'variable': 'time', 'change': 0.25, 'name': 'car_time_plus25'},
        {'mode': 'car', 'variable': 'time', 'change': 0.50, 'name': 'car_time_plus50'},
        {'mode': 'car', 'variable': 'time', 'change': 1.00, 'name': 'car_time_plus100'},
        
        # Air cost modifications
        {'mode': 'air', 'variable': 'cost', 'change': 0.10, 'name': 'air_cost_plus10'},
        {'mode': 'air', 'variable': 'cost', 'change': 0.20, 'name': 'air_cost_plus20'},
        {'mode': 'air', 'variable': 'cost', 'change': 0.30, 'name': 'air_cost_plus30'},
        
        # Bus cost modifications
        {'mode': 'bus', 'variable': 'cost', 'change': -0.25, 'name': 'bus_cost_minus25'},
        {'mode': 'bus', 'variable': 'cost', 'change': -0.50, 'name': 'bus_cost_minus50'},
        {'mode': 'bus', 'variable': 'cost', 'change': -1.00, 'name': 'bus_cost_zero'},
    ]
    
    # Store all results
    all_results = []
    
    # Run baseline scenario simulation
    print("\nSimulating baseline scenario...")
    baseline_results = simulate_models(data, mnl, nl, ml, 'baseline')
    all_results.append(baseline_results)
    
    # Simulate on each modified scenario
    for mod in modifications:
        print(f"\nSimulating {mod['name']} scenario...")
        modified_data = modify_dataset(data, mod)
        results = simulate_models(modified_data, mnl, nl, ml, mod['name'])
        all_results.append(results)
    
    # Save results to JSON file
    output_file = f'sensitivity_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Define modification groups for plotting
    modification_groups = [
        {
            'mode': 'car',
            'mode_num': 2,
            'variable': 'time',
            'scenarios': ['baseline', 'car_time_plus25', 'car_time_plus50', 'car_time_plus100'],
            'changes': [0, 0.25, 0.50, 1.00]
        },
        {
            'mode': 'air',
            'mode_num': 4,
            'variable': 'cost',
            'scenarios': ['baseline', 'air_cost_plus10', 'air_cost_plus20', 'air_cost_plus30'],
            'changes': [0, 0.10, 0.20, 0.30]
        },
        {
            'mode': 'bus',
            'mode_num': 3,  # Mode number in the model (1=train, 2=car, 3=bus, 4=air)
            'variable': 'cost',
            'scenarios': ['baseline', 'bus_cost_minus25', 'bus_cost_minus50', 'bus_cost_zero'],
            'changes': [0, -0.25, -0.5, -1]  # -1.0 represents zero cost
        }
    ]
    
    # Create plots for each modification group
    for group in modification_groups:
        plot_share_evolution(all_results, group)
    
    print(f"\nSensitivity analysis complete. Results saved to {output_file}")
    print("Share evolution plots have been created for each modification group.")
    return all_results

if __name__ == "__main__":
    run_sensitivity_analysis()
