"""
Market Share Analysis Script for ModeCanada Models

This script:
1. Estimates models on original dataset
2. Calibrates ASCs for NL and ML models
3. Compares actual vs simulated market shares for each model
4. For Mixed Logit, performs multiple simulation runs to show variability
5. Performs sensitivity analysis with modified scenarios
6. Creates plots showing evolution of predicted shares
"""

from mcbs.datasets import DatasetLoader
from mcbs.models.modecanada_model import MultinomialLogitModel_MC, NestedLogitModel3_MC, MixedLogitModel_MC
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from calibrate_models import calculate_actual_shares, simulate_market_shares, calibrate_alternative_constants
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, bioDraws, MonteCarlo

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

def simulate_mixed_logit(model, data, num_runs=10):
    """
    Simulate Mixed Logit model multiple times to show variability due to random draws.
    
    Args:
        model (MixedLogitModel_MC): Estimated Mixed Logit model
        data (pd.DataFrame): Dataset to simulate on
        num_runs (int): Number of simulation runs
    
    Returns:
        dict: Dictionary containing mean predicted shares and their standard deviations
    """
    betas = model.results.betas
    all_shares = []
    
    for run in range(num_runs):
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
        
        # Calculate market shares for this run
        shares = {
            1: simulatedValues['Prob. train'].mean(),
            2: simulatedValues['Prob. car'].mean(),
            3: simulatedValues['Prob. bus'].mean(),
            4: simulatedValues['Prob. air'].mean()
        }
        all_shares.append(shares)
    
    # Calculate mean and standard deviation across runs
    mean_shares = {mode: np.mean([run[mode] for run in all_shares]) for mode in [1, 2, 3, 4]}
    std_shares = {mode: np.std([run[mode] for run in all_shares]) for mode in [1, 2, 3, 4]}
    
    return {'mean': mean_shares, 'std': std_shares}

def create_market_share_table(actual_shares, mnl_shares, nl_shares, ml_shares):
    """
    Create a formatted table comparing actual and simulated market shares.
    For Mixed Logit, includes mean and standard deviation across multiple runs.
    """
    mode_names = {1: 'Train', 2: 'Car', 3: 'Bus', 4: 'Air'}
    
    # Create DataFrame for comparison
    data = []
    for mode_num in [1, 2, 3, 4]:
        mode = mode_names[mode_num]
        actual = actual_shares[mode_num]
        mnl = mnl_shares[mode_num]
        nl = nl_shares[mode_num]
        ml_mean = ml_shares['mean'][mode_num]
        ml_std = ml_shares['std'][mode_num]
        
        # Calculate differences
        mnl_diff = mnl - actual
        nl_diff = nl - actual
        ml_diff = ml_mean - actual
        
        data.append({
            'Mode': mode,
            'Actual Share': f"{actual:.3f}",
            'MNL Share': f"{mnl:.3f}",
            'MNL Diff': f"{mnl_diff:+.3f}",
            'NL Share': f"{nl:.3f}",
            'NL Diff': f"{nl_diff:+.3f}",
            'ML Share': f"{ml_mean:.3f}±{ml_std:.3f}",
            'ML Diff': f"{ml_diff:+.3f}"
        })
    
    df = pd.DataFrame(data)
    return df

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
    x_values = [1 + change for change in changes]  # e.g., 0 becomes 1, 0.1 becomes 1.1
    
    # Store data for CSV
    csv_data = []
    
    # Plot evolution for each model
    for model in models:
        shares = [baseline_shares[model]]  # Start with baseline
        for scenario in scenarios[1:]:  # Skip baseline
            result = next(r for r in results if r['scenario'] == scenario)
            share = result[model]['predicted_shares'][modification_group['mode_num']]
            shares.append(share)
        
        # Convert shares to percentages
        shares = [s * 100 for s in shares]
        
        # Store data for CSV
        for x, share in zip(x_values, shares):
            csv_data.append({
                'Model': model_names[model],
                'Multiplier': x,
                'Share (%)': share
            })
        
        plt.plot(x_values, shares, marker='o', label=model_names[model], color=colors[model])
    
    # Save data to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(f'share_evolution_{mode}_{variable}_data.csv', index=False)
    
    # Customize plot
    plt.title('Scenario: Increase in Road Charges')
    plt.xlabel('Car Price Multiplier')
    plt.ylabel('Predicted Car Market Share (%)')
    plt.ylim(0, 100)  # Set y-axis from 0 to 100%
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save plot
    plt.savefig(f'share_evolution_{mode}_{variable}.png')
    plt.close()

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
        'predicted_shares': ml_shares['mean'],  # Use mean shares for plotting
    }
    
    return results

def run_market_share_analysis():
    """Run the market share analysis and sensitivity analysis."""
    # Load original dataset
    loader = DatasetLoader()
    data = loader.load_dataset("modecanada_dataset")
    
    # Calculate actual market shares
    actual_shares = calculate_actual_shares(data)
    print("\nActual market shares:")
    for mode, share in actual_shares.items():
        print(f"Mode {mode}: {share:.3f}")
    
    # Estimate and calibrate models
    print("\nEstimating and calibrating models...")
    
    # MNL
    print("\nEstimating MNL model")
    mnl = MultinomialLogitModel_MC(data)
    mnl.estimate()
    mnl.results.betas = calibrate_alternative_constants(mnl, data, actual_shares)
    mnl_shares = simulate_market_shares(mnl, data, mnl.results.betas)
    
    # NL
    print("\nEstimating NL model")
    nl = NestedLogitModel3_MC(data)
    nl.estimate()
    nl.results.betas = calibrate_alternative_constants(nl, data, actual_shares)
    print("THESE ARE THE NL BETAS:")
    print(nl.results.betas)
    nl.results.betas['MU_PUBLIC'] = 2.0
    print("THESE ARE THE UPDATED NL BETAS:")
    nl_shares = simulate_market_shares(nl, data, nl.results.betas)
    
    # ML
    print("\nEstimating Mixed Logit model")
    ml = MixedLogitModel_MC(data)
    ml.estimate()
    ml.results.betas = calibrate_alternative_constants(ml, data, actual_shares)
    ml_shares = simulate_mixed_logit(ml, data, num_runs=10)  # Run multiple simulations
    
    # Create comparison table
    print("\nMarket Share Comparison:")
    comparison_table = create_market_share_table(actual_shares, mnl_shares, nl_shares, ml_shares)
    print("\n" + comparison_table.to_string(index=False))
    
    # Save table to CSV
    comparison_table.to_csv('market_share_comparison.csv', index=False)
    print("\nMarket share comparison saved to market_share_comparison.csv")
    
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
        
        # Car cost modifications
        {'mode': 'car', 'variable': 'cost', 'change': 0.10, 'name': 'car_cost_plus10'},
        {'mode': 'car', 'variable': 'cost', 'change': 0.25, 'name': 'car_cost_plus25'},
        {'mode': 'car', 'variable': 'cost', 'change': 0.50, 'name': 'car_cost_plus50'},
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
            'mode': 'car',
            'mode_num': 2,
            'variable': 'cost',
            'scenarios': ['baseline', 'car_cost_plus10', 'car_cost_plus25', 'car_cost_plus50'],
            'changes': [0, 0.10, 0.25, 0.50]
        }
    ]
    
    # Create plots for each modification group
    for group in modification_groups:
        plot_share_evolution(all_results, group)
    
    print(f"\nSensitivity analysis complete. Results saved to {output_file}")
    print("Share evolution plots have been created for each modification group.")
    print("CSV files with plotted data have been created for each plot.")

if __name__ == "__main__":
    run_market_share_analysis()
