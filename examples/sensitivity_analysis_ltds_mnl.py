"""
Market Share Analysis Script for LTDS MNL Model

This script:
1. Estimates MNL model on original dataset
2. Performs sensitivity analysis with modified car cost scenarios
3. Creates plots showing evolution of predicted shares
"""

from mcbs.datasets import DatasetLoader
from mcbs.models.ltds_model import MultinomialLogitModel_L, BaseLTDSModel
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from calibrate_models_ltds import calculate_actual_shares, simulate_market_shares, calibrate_alternative_constants
import logging
import biogeme.biogeme as bio
from biogeme.database import Database
logger = bio.logger
logger.setLevel(logging.ERROR)  

def modify_dataset(data, modification):
    """
    Create a modified version of the dataset based on specified changes.
    
    Args:
        data (pd.DataFrame): Original dataset
        modification (dict): Dictionary specifying the modifications
            e.g., {'mode': 'drive', 'variable': 'cost', 'change': 0.25}
    
    Returns:
        pd.DataFrame: Modified dataset
    """
    modified_data = data.copy()
    
    mode = modification['mode']
    variable = modification['variable']
    change = modification['change']
    
    if variable == 'cost' and mode == 'drive':
        print(f"\nBefore modification:")
        print(f"Mean fuel cost: {modified_data['cost_driving_fuel'].mean():.3f}")
        print(f"Mean congestion charge: {modified_data['cost_driving_con_charge'].mean():.3f}")
        print(f"Total mean cost: {(modified_data['cost_driving_fuel'] + modified_data['cost_driving_con_charge']).mean():.3f}")
        
        # Modify both fuel cost and congestion charge
        modified_data['cost_driving_fuel'] = modified_data['cost_driving_fuel'] * (1 + change)
        modified_data['cost_driving_con_charge'] = modified_data['cost_driving_con_charge'] * (1 + change)
        
        print(f"\nAfter modification (increase of {change*100}%):")
        print(f"Mean fuel cost: {modified_data['cost_driving_fuel'].mean():.3f}")
        print(f"Mean congestion charge: {modified_data['cost_driving_con_charge'].mean():.3f}")
        print(f"Total mean cost: {(modified_data['cost_driving_fuel'] + modified_data['cost_driving_con_charge']).mean():.3f}")
    
    return modified_data

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
    
    # Get baseline shares for all modes
    baseline_shares = results[0]['mnl']['predicted_shares']
    
    # Convert changes to multipliers for x-axis
    x_values = [1 + change for change in changes]  # e.g., 0 becomes 1, 0.1 becomes 1.1
    
    # Store data for CSV
    csv_data = []
    
    # Plot evolution for each mode
    mode_names = {1: 'Walk', 2: 'Cycle', 3: 'PT', 4: 'Car'}
    colors = {1: 'green', 2: 'blue', 3: 'red', 4: 'purple'}
    
    for mode_num in [1, 2, 3, 4]:
        shares = [baseline_shares[mode_num]]  # Start with baseline
        for scenario in scenarios[1:]:  # Skip baseline
            result = next(r for r in results if r['scenario'] == scenario)
            share = result['mnl']['predicted_shares'][mode_num]
            shares.append(share)
        
        # Convert shares to percentages
        shares = [s * 100 for s in shares]
        
        # Store data for CSV
        for x, share in zip(x_values, shares):
            csv_data.append({
                'Mode': mode_names[mode_num],
                'Multiplier': x,
                'Share (%)': share
            })
        
        plt.plot(x_values, shares, marker='o', label=mode_names[mode_num], color=colors[mode_num])
    
    # Save data to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(f'share_evolution_{mode}_{variable}_mnl_data.csv', index=False)
    
    # Customize plot
    plt.title('Mode Shares vs Car Cost Multiplier')
    plt.xlabel('Car Cost Multiplier')
    plt.ylabel('Mode Share (%)')
    plt.ylim(0, 100)  # Set y-axis from 0 to 100%
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.savefig(f'share_evolution_{mode}_{variable}_mnl.png')
    plt.close()

def simulate_models(data, mnl_model, scenario_name):
    """
    Simulate using MNL model on a given dataset.
    
    Args:
        data (pd.DataFrame): Dataset to simulate on
        mnl_model (MultinomialLogitModel_L): Estimated MNL model
        scenario_name (str): Name of the scenario being simulated
    
    Returns:
        dict: Dictionary containing simulation results for MNL model
    """
    results = {'scenario': scenario_name}
    
    # Create new database with modified data
    database = Database("modified_data", data)
    
    # Create new model instance with modified data
    mnl_sim = MultinomialLogitModel_L(data)
    mnl_sim.database = database  # Use the new database
    mnl_sim.results = mnl_model.results  # Copy results including parameters
    
    # Print parameters and mean costs for verification
    betas = mnl_sim.results.get_beta_values()
    print(f"\nMNL parameters for {scenario_name}:")
    print(f"B_COST_DRIVING: {betas['B_COST_DRIVING']:.4f}")
    print(f"Mean driving cost: {(data['cost_driving_fuel'] + data['cost_driving_con_charge']).mean():.4f}")
    
    # Simulate MNL model
    mnl_shares = simulate_market_shares(mnl_sim, data)
    results['mnl'] = {
        'predicted_shares': mnl_shares
    }
    
    # Print predicted shares
    print(f"\nPredicted shares for {scenario_name}:")
    mode_names = {1: 'Walk', 2: 'Cycle', 3: 'PT', 4: 'Car'}
    for mode_num, share in mnl_shares.items():
        print(f"{mode_names[mode_num]}: {share:.3f}")
    
    return results

def run_market_share_analysis():
    """Run the market share analysis and sensitivity analysis."""
    # Load original dataset
    loader = DatasetLoader()
    data = loader.load_dataset("ltds_dataset")
    
    # Calculate actual market shares
    actual_shares = calculate_actual_shares(data)
    print("\nActual market shares:")
    for mode, share in actual_shares.items():
        print(f"Mode {mode}: {share:.3f}")
    
    # Estimate and calibrate models
    print("\nEstimating and calibrating models...")
    
    # MNL
    print("\nEstimating MNL model")
    mnl = MultinomialLogitModel_L(data)
    mnl.estimate()
    
    # Print initial MNL parameters
    print("\nInitial MNL parameters:")
    print(mnl.results.get_beta_values())
    
    mnl.results.betas = calibrate_alternative_constants(mnl, data, actual_shares)
    mnl_shares = simulate_market_shares(mnl, data)
    
    # Print calibrated MNL parameters
    print("\nCalibrated MNL parameters:")
    print(mnl.results.get_beta_values())
    
    # Create comparison table
    print("\nMarket Share Comparison:")
    comparison_data = []
    mode_names = {1: 'Walk', 2: 'Cycle', 3: 'PT', 4: 'Car'}
    
    for mode_num in [1, 2, 3, 4]:
        mode = mode_names[mode_num]
        actual = actual_shares[mode_num]
        mnl_pred = mnl_shares[mode_num]
        
        # Calculate differences
        mnl_diff = mnl_pred - actual
        
        comparison_data.append({
            'Mode': mode,
            'Actual Share': f"{actual:.3f}",
            'MNL Share': f"{mnl_pred:.3f}",
            'MNL Diff': f"{mnl_diff:+.3f}"
        })
    
    comparison_table = pd.DataFrame(comparison_data)
    print("\n" + comparison_table.to_string(index=False))
    
    # Save table to CSV
    comparison_table.to_csv('market_share_comparison_ltds_mnl.csv', index=False)
    print("\nMarket share comparison saved to market_share_comparison_ltds_mnl.csv")
    
    # Define car cost modifications to test
    modifications = [
        {'mode': 'drive', 'variable': 'cost', 'change': 0.25, 'name': 'car_cost_plus25'},
        {'mode': 'drive', 'variable': 'cost', 'change': 0.50, 'name': 'car_cost_plus50'},
        {'mode': 'drive', 'variable': 'cost', 'change': 1.00, 'name': 'car_cost_plus100'},
        {'mode': 'drive', 'variable': 'cost', 'change': 2.00, 'name': 'car_cost_plus200'},
        {'mode': 'drive', 'variable': 'cost', 'change': 5.00, 'name': 'car_cost_plus500'}
    ]
    
    # Store all results
    all_results = []
    
    # Run baseline scenario simulation
    print("\nSimulating baseline scenario...")
    baseline_results = simulate_models(data, mnl, 'baseline')
    all_results.append(baseline_results)
    
    # Simulate on each modified scenario
    for mod in modifications:
        print(f"\nSimulating {mod['name']} scenario...")
        modified_data = modify_dataset(data, mod)
        results = simulate_models(modified_data, mnl, mod['name'])
        all_results.append(results)
    
    # Save results to JSON file
    output_file = f'sensitivity_analysis_results_ltds_mnl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Define modification group for plotting
    modification_group = {
        'mode': 'drive',
        'mode_num': 4,  # Car is mode 4 in LTDS
        'variable': 'cost',
        'scenarios': ['baseline'] + [mod['name'] for mod in modifications],
        'changes': [0] + [mod['change'] for mod in modifications]
    }
    
    # Create plot for car cost modifications
    plot_share_evolution(all_results, modification_group)
    
    print(f"\nSensitivity analysis complete. Results saved to {output_file}")
    print("Share evolution plot has been created.")
    print("CSV file with plotted data has been created.")

if __name__ == "__main__":
    run_market_share_analysis()
