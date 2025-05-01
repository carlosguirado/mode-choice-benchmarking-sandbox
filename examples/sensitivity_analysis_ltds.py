"""
Market Share Analysis Script for LTDS Models

This script:
1. Estimates models on original dataset
2. Performs sensitivity analysis with modified car time and cost scenarios
3. Creates plots showing evolution of car mode share
"""

from mcbs.datasets import DatasetLoader
from mcbs.models.ltds_model import MultinomialLogitModel_L, NestedLogitModel_L, MixedLogitModel_L, BaseLTDSModel
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from calibrate_models_ltds import calculate_actual_shares, simulate_market_shares, calibrate_alternative_constants
import logging
import sys
import biogeme.biogeme as bio
from biogeme.database import Database
from biogeme import models
from biogeme.expressions import Variable

# Setup logging to write to both file and console
log_filename = f'sensitivity_analysis_ltds_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Redirect Biogeme logging
bio.logger.setLevel(logging.ERROR)

# Custom print function that writes to both console and file
def log_print(*args, **kwargs):
    message = ' '.join(map(str, args))
    logger.info(message)

def encode_categorical_variables(df):
    """
    Encode categorical variables according to LTDS dataset specifications.
    """
    df_encoded = df.copy()
    
    # Travel mode encoding
    mode_mapping = {
        'walk': 1,
        'cycle': 2,
        'pt': 3,
        'drive': 4
    }
    
    # Purpose encoding
    purpose_mapping = {
        'HBW': 1,    # home-based work
        'HBE': 2,    # home-based education
        'HBO': 3,    # home-based other
        'B': 4,      # employers' business
        'NHBO': 5    # non-home-based other
    }
    
    # Fuel type encoding
    fueltype_mapping = {
        'Petrol_Car': 1,
        'Diesel_Car': 2,
        'Hybrid_Car': 3,
        'Petrol_LGV': 4,
        'Diesel_LGV': 5,
        'Average_Car': 6
    }
    
    # Fare type encoding
    faretype_mapping = {
        'full': 1,
        '16+': 2,
        'child': 3,
        'dis': 4,    # disabled
        'free': 5
    }
    
    # Apply the mappings with default values
    if 'travel_mode' in df.columns:
        df_encoded['travel_mode'] = df['travel_mode'].map(mode_mapping).fillna(1).astype('int64')
    
    if 'purpose' in df.columns:
        df_encoded['purpose'] = df['purpose'].map(purpose_mapping).fillna(1).astype('int64')
    
    if 'fueltype' in df.columns:
        df_encoded['fueltype'] = df['fueltype'].map(fueltype_mapping).fillna(6).astype('int64')
    
    if 'faretype' in df.columns:
        df_encoded['faretype'] = df['faretype'].map(faretype_mapping).fillna(1).astype('int64')
    
    return df_encoded

def modify_dataset(data, modification):
    """
    Create a modified version of the dataset based on specified changes.
    """
    modified_data = data.copy()
    
    mode = modification['mode']
    variable = modification['variable']
    change = modification['change']
    
    if variable == 'time' and mode == 'drive':
        log_print(f"\nBefore modification:")
        log_print(f"Mean drive time: {modified_data['dur_driving'].mean():.3f}")
        
        # Modify drive time
        modified_data['dur_driving'] = modified_data['dur_driving'] * (1 + change)
        
        log_print(f"\nAfter modification (increase of {change*100}%):")
        log_print(f"Mean drive time: {modified_data['dur_driving'].mean():.3f}")
    
    elif variable == 'cost' and mode == 'drive':
        log_print(f"\nBefore modification:")
        log_print(f"Mean drive cost: {modified_data['cost_driving_fuel'].mean():.3f}")
        
        # Modify drive cost
        modified_data['cost_driving_fuel'] = modified_data['cost_driving_fuel'] * (1 + change)
        modified_data['cost_driving_total'] = modified_data['cost_driving_total'] * (1 + change)
        
        log_print(f"\nAfter modification (increase of {change*100}%):")
        log_print(f"Mean drive cost: {modified_data['cost_driving_fuel'].mean():.3f}")
    
    return modified_data

def plot_car_share_evolution(results, modification_groups):
    """
    Create plot showing the evolution of car mode share for different modifications.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot lines for each model type and modification group
    models = ['mnl', 'nl', 'mixed']
    model_names = {'mnl': 'MNL', 'nl': 'NL', 'mixed': 'Mixed Logit'}
    colors = {'time': 'blue', 'cost': 'red'}
    linestyles = {'mnl': '-', 'nl': '--', 'mixed': ':'}
    
    for group in modification_groups:
        mode = group['mode']
        variable = group['variable']
        scenarios = group['scenarios']
        changes = group['changes']
        
        # Get baseline shares
        baseline_shares = {
            model: results[0][model]['predicted_shares']
            for model in models
        }
        
        # Convert changes to multipliers for x-axis
        x_values = [1 + change for change in changes]  # e.g., 0 becomes 1, 0.1 becomes 1.1
        
        # Store data for CSV
        csv_data = []
        
        # Plot evolution for each model type
        for model in models:
            shares = [baseline_shares[model][4]]  # Start with baseline car share
            for scenario in scenarios[1:]:  # Skip baseline
                result = next(r for r in results if r['scenario'] == scenario)
                share = result[model]['predicted_shares'][4]  # Car is mode 4
                shares.append(share)
            
            # Convert shares to percentages
            shares = [s * 100 for s in shares]
            
            # Store data for CSV
            for x, share in zip(x_values, shares):
                csv_data.append({
                    'Model': model_names[model],
                    'Variable': variable,
                    'Multiplier': x,
                    'Car Share (%)': share
                })
            
            plt.plot(x_values, shares, marker='o', 
                    label=f"{model_names[model]} - {variable}", 
                    color=colors[variable],
                    linestyle=linestyles[model])
    
    # Save data to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv('car_share_evolution_data_ltds.csv', index=False)
    
    # Customize plot
    plt.title('Car Mode Share vs Time/Cost Multiplier')
    plt.xlabel('Multiplier')
    plt.ylabel('Car Mode Share (%)')
    plt.ylim(0, 100)  # Set y-axis from 0% to 100%
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot with extra width for legend
    plt.gcf().set_size_inches(12, 6)
    plt.savefig('car_share_evolution.png', bbox_inches='tight')
    plt.close()

def simulate_models(data, mnl_model, nl_model, mixed_model, scenario_name):
    """
    Simulate using all model types on a given dataset.
    """
    log_print(f"\nVerifying data modifications for {scenario_name}:")
    log_print(f"Mean drive time in input data: {data['dur_driving'].mean():.3f}")

    results = {'scenario': scenario_name}
    
    # Create new MNL model instance with modified data
    mnl_sim = MultinomialLogitModel_L(data)
    mnl_sim.results = mnl_model.results
    
    # Add second debug print here, right after creating MNL model
    log_print(f"Mean drive time in MNL model data: {mnl_sim.database.data['dur_driving'].mean():.3f}")
    
    # Calculate utilities and probabilities for MNL
    mnl_debug = calculate_utilities_and_probabilities(mnl_sim, data)
    print_debug_info("MNL", scenario_name, mnl_debug, data)
    
    # Simulate MNL model
    mnl_shares = simulate_market_shares(mnl_sim, data)
    results['mnl'] = {
        'predicted_shares': mnl_shares
    }
    
    # Create new NL model instance with modified data
    nl_sim = NestedLogitModel_L(data)
    nl_sim.results = nl_model.results
    nl_sim.nests = nl_model.nests
    
    # Add third debug print here, right after creating NL model
    log_print(f"Mean drive time in NL model data: {nl_sim.database.data['dur_driving'].mean():.3f}")
    
    # Calculate utilities and probabilities for NL
    nl_debug = calculate_utilities_and_probabilities(nl_sim, data)
    print_debug_info("NL", scenario_name, nl_debug, data)
    
    # Simulate NL model
    nl_shares = simulate_market_shares(nl_sim, data)
    results['nl'] = {
        'predicted_shares': nl_shares
    }
    
    # Create new Mixed Logit model instance with modified data
    mixed_sim = MixedLogitModel_L(data)
    mixed_sim.results = mixed_model.results
    
    # Add debug print for Mixed Logit model
    log_print(f"Mean drive time in Mixed Logit model data: {mixed_sim.database.data['dur_driving'].mean():.3f}")
    
    # Calculate utilities and probabilities for Mixed Logit
    mixed_debug = calculate_utilities_and_probabilities(mixed_sim, data)
    print_debug_info("Mixed Logit", scenario_name, mixed_debug, data)
    
    # Simulate Mixed Logit model
    mixed_shares = simulate_market_shares(mixed_sim, data)
    results['mixed'] = {
        'predicted_shares': mixed_shares
    }
    
    return results

def calculate_utilities_and_probabilities(model, data):
    """Calculate utilities and probabilities for each mode."""
    # Get parameters
    betas = model.results.get_beta_values()
    log_print("\nModel parameters:")
    for param, value in betas.items():
        log_print(f"{param}: {value:.4f}")
    
    # Initialize variables
    WALK_TIME = Variable('dur_walking')
    CYCLE_TIME = Variable('dur_cycling')
    DRIVE_TIME = Variable('dur_driving')
    PT_ACCESS = Variable('dur_pt_access')
    PT_RAIL = Variable('dur_pt_rail')
    PT_BUS = Variable('dur_pt_bus')
    PT_INT = Variable('dur_pt_int_total')
    DRIVE_COST = Variable('cost_driving_fuel')
    CCHARGE = Variable('cost_driving_con_charge')
    PT_COST = Variable('cost_transit')
    
    # Calculate utilities based on model type
    if isinstance(model, MultinomialLogitModel_L):
        # MNL utilities
        V1 = (0 + betas['B_TIME_WALKING'] * WALK_TIME)
        V2 = (betas['ASC_CYCLING'] + betas['B_TIME_CYCLING'] * CYCLE_TIME)
        V3 = (betas['ASC_PT'] + betas['B_COST'] * PT_COST + 
              betas['B_TIME_PT'] * (PT_ACCESS + PT_RAIL + PT_BUS + PT_INT))
        V4 = (betas['ASC_DRIVING'] + betas['B_TIME_DRIVING'] * DRIVE_TIME +
              betas['B_COST'] * (DRIVE_COST + CCHARGE))
    elif isinstance(model, NestedLogitModel_L):
        # NL utilities
        V1 = (0 + betas['B_TIME'] * WALK_TIME)
        V2 = (betas['ASC_CYCLING'] + betas['B_TIME'] * CYCLE_TIME)
        V3 = (betas['ASC_PT'] + betas['B_COST'] * PT_COST + 
              betas['B_TIME'] * (PT_ACCESS + PT_RAIL + PT_BUS + PT_INT))
        V4 = (betas['ASC_DRIVING'] + betas['B_TIME'] * DRIVE_TIME +
              betas['B_COST'] * (DRIVE_COST + CCHARGE))
    else:  # Mixed Logit
        # Use mean coefficients for simulation
        V1 = (0 + betas['B_TIME_WALKING'] * WALK_TIME)
        V2 = (betas['ASC_CYCLING'] + betas['B_TIME_CYCLING'] * CYCLE_TIME)
        V3 = (betas['ASC_PT'] + betas['B_COST'] * PT_COST + 
              betas['B_TIME_PT'] * (PT_ACCESS + PT_RAIL + PT_BUS + PT_INT))
        V4 = (betas['ASC_DRIVING'] + betas['B_TIME_DRIVING'] * DRIVE_TIME +
              betas['B_COST'] * (DRIVE_COST + CCHARGE))
    
    # Associate utilities with alternatives
    V = {1: V1, 2: V2, 3: V3, 4: V4}
    
    # Associate availability conditions
    av = {1: 1, 2: 1, 3: 1, 4: 1}
    
    # Calculate probabilities based on model type
    if isinstance(model, MultinomialLogitModel_L):
        prob1 = models.logit(V, av, 1)
        prob2 = models.logit(V, av, 2)
        prob3 = models.logit(V, av, 3)
        prob4 = models.logit(V, av, 4)
    elif isinstance(model, NestedLogitModel_L):
        prob1 = models.nested(V, None, model.nests, 1)
        prob2 = models.nested(V, None, model.nests, 2)
        prob3 = models.nested(V, None, model.nests, 3)
        prob4 = models.nested(V, None, model.nests, 4)
    else:  # Mixed Logit
        prob1 = models.logit(V, av, 1)
        prob2 = models.logit(V, av, 2)
        prob3 = models.logit(V, av, 3)
        prob4 = models.logit(V, av, 4)
    
    # Setup simulation
    simulate = {
        'Util. walk': V1,
        'Util. cycle': V2,
        'Util. PT': V3,
        'Util. drive': V4,
        'Prob. walk': prob1,
        'Prob. cycle': prob2,
        'Prob. PT': prob3,
        'Prob. drive': prob4
    }
    
    # Run simulation
    biogeme = bio.BIOGEME(model.database, simulate)
    results = biogeme.simulate(betas)
    
    return results

def print_debug_info(model_type, scenario, results, data):
    """Print detailed debug information for a model simulation."""
    log_print(f"\n=== {model_type} Model Debug Info for {scenario} ===")
    
    # Print mean utilities
    log_print("\nMean Utilities:")
    log_print(f"Walk:  {results['Util. walk'].mean():.4f}")
    log_print(f"Cycle: {results['Util. cycle'].mean():.4f}")
    log_print(f"PT:    {results['Util. PT'].mean():.4f}")
    log_print(f"Drive: {results['Util. drive'].mean():.4f}")
    
    # Print mean probabilities
    log_print("\nMean Choice Probabilities:")
    log_print(f"Walk:  {results['Prob. walk'].mean():.4f}")
    log_print(f"Cycle: {results['Prob. cycle'].mean():.4f}")
    log_print(f"PT:    {results['Prob. PT'].mean():.4f}")
    log_print(f"Drive: {results['Prob. drive'].mean():.4f}")
    
    # Print time information
    log_print("\nDriving Time:")
    log_print(f"Mean drive time: {data['dur_driving'].mean():.4f}")
    
    # Print utility ranges
    log_print("\nUtility Ranges:")
    log_print(f"Walk:  [{results['Util. walk'].min():.4f}, {results['Util. walk'].max():.4f}]")
    log_print(f"Cycle: [{results['Util. cycle'].min():.4f}, {results['Util. cycle'].max():.4f}]")
    log_print(f"PT:    [{results['Util. PT'].min():.4f}, {results['Util. PT'].max():.4f}]")
    log_print(f"Drive: [{results['Util. drive'].min():.4f}, {results['Util. drive'].max():.4f}]")
    
    log_print("\n" + "="*50)

def run_market_share_analysis():
    """Run the market share analysis and sensitivity analysis."""
    # Load original dataset
    loader = DatasetLoader()
    data = loader.load_dataset("ltds_dataset")
    
    # Encode categorical variables and ensure proper data types
    data = encode_categorical_variables(data)
    
    # Calculate actual market shares
    actual_shares = calculate_actual_shares(data)
    log_print("\nActual market shares:")
    for mode, share in actual_shares.items():
        log_print(f"Mode {mode}: {share:.3f}")
    
    # Estimate and calibrate models
    log_print("\nEstimating and calibrating models...")
    
    # MNL
    log_print("\nEstimating MNL model")
    mnl = MultinomialLogitModel_L(data)
    mnl.estimate()
    
    # Print initial MNL parameters
    log_print("\nInitial MNL parameters:")
    log_print(mnl.results.get_beta_values())
    
    mnl.results.betas = calibrate_alternative_constants(mnl, data, actual_shares)
    mnl_shares = simulate_market_shares(mnl, data)
    
    # Print calibrated MNL parameters
    log_print("\nCalibrated MNL parameters:")
    log_print(mnl.results.get_beta_values())
    
    # NL
    log_print("\nEstimating NL model")
    nl = NestedLogitModel_L(data)
    nl.estimate()
    
    # Print initial NL parameters
    log_print("\nInitial NL parameters:")
    log_print(nl.results.get_beta_values())
    
    nl.results.betas = calibrate_alternative_constants(nl, data, actual_shares)
    nl_shares = simulate_market_shares(nl, data)
    
    # Print calibrated NL parameters
    log_print("\nCalibrated NL parameters:")
    log_print(nl.results.get_beta_values())
    
    # Mixed Logit
    log_print("\nEstimating Mixed Logit model")
    mixed = MixedLogitModel_L(data)
    mixed.estimate()
    
    # Print initial Mixed Logit parameters
    log_print("\nInitial Mixed Logit parameters:")
    log_print(mixed.results.get_beta_values())
    
    mixed.results.betas = calibrate_alternative_constants(mixed, data, actual_shares)
    mixed_shares = simulate_market_shares(mixed, data)
    
    # Print calibrated Mixed Logit parameters
    log_print("\nCalibrated Mixed Logit parameters:")
    log_print(mixed.results.get_beta_values())
    
    # Create comparison table
    log_print("\nMarket Share Comparison:")
    comparison_data = []
    mode_names = {1: 'Walk', 2: 'Cycle', 3: 'PT', 4: 'Car'}
    
    for mode_num in [1, 2, 3, 4]:
        mode = mode_names[mode_num]
        actual = actual_shares[mode_num]
        mnl_pred = mnl_shares[mode_num]
        nl_pred = nl_shares[mode_num]
        mixed_pred = mixed_shares[mode_num]
        
        # Calculate differences
        mnl_diff = mnl_pred - actual
        nl_diff = nl_pred - actual
        mixed_diff = mixed_pred - actual
        
        comparison_data.append({
            'Mode': mode,
            'Actual Share': f"{actual:.3f}",
            'MNL Share': f"{mnl_pred:.3f}",
            'MNL Diff': f"{mnl_diff:+.3f}",
            'NL Share': f"{nl_pred:.3f}",
            'NL Diff': f"{nl_diff:+.3f}",
            'Mixed Share': f"{mixed_pred:.3f}",
            'Mixed Diff': f"{mixed_diff:+.3f}"
        })
    
    comparison_table = pd.DataFrame(comparison_data)
    log_print("\n" + comparison_table.to_string(index=False))
    
    # Save table to CSV
    comparison_table.to_csv('market_share_comparison_ltds.csv', index=False)
    log_print("\nMarket share comparison saved to market_share_comparison_ltds.csv")
    
    # Define modifications to test
    time_modifications = [
        {'mode': 'drive', 'variable': 'time', 'change': 0.10, 'name': 'car_time_plus10'},
        {'mode': 'drive', 'variable': 'time', 'change': 0.25, 'name': 'car_time_plus25'},
        {'mode': 'drive', 'variable': 'time', 'change': 0.50, 'name': 'car_time_plus50'}
    ]
    
    cost_modifications = [
        {'mode': 'drive', 'variable': 'cost', 'change': 0.10, 'name': 'car_cost_plus10'},
        {'mode': 'drive', 'variable': 'cost', 'change': 0.25, 'name': 'car_cost_plus25'},
        {'mode': 'drive', 'variable': 'cost', 'change': 0.50, 'name': 'car_cost_plus50'}
    ]
    
    # Store all results
    all_results = []
    
    # Run baseline scenario simulation
    log_print("\nSimulating baseline scenario...")
    baseline_results = simulate_models(data, mnl, nl, mixed, 'baseline')
    all_results.append(baseline_results)
    
    # Simulate time modifications
    for mod in time_modifications:
        log_print(f"\nSimulating {mod['name']} scenario...")
        modified_data = modify_dataset(data, mod)
        results = simulate_models(modified_data, mnl, nl, mixed, mod['name'])
        all_results.append(results)
    
    # Simulate cost modifications
    for mod in cost_modifications:
        log_print(f"\nSimulating {mod['name']} scenario...")
        modified_data = modify_dataset(data, mod)
        results = simulate_models(modified_data, mnl, nl, mixed, mod['name'])
        all_results.append(results)
    
    # Save results to JSON file
    output_file = f'sensitivity_analysis_results_ltds_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Define modification groups for plotting
    modification_groups = [
        {
            'mode': 'drive',
            'variable': 'time',
            'scenarios': ['baseline'] + [mod['name'] for mod in time_modifications],
            'changes': [0] + [mod['change'] for mod in time_modifications]
        },
        {
            'mode': 'drive',
            'variable': 'cost',
            'scenarios': ['baseline'] + [mod['name'] for mod in cost_modifications],
            'changes': [0] + [mod['change'] for mod in cost_modifications]
        }
    ]
    
    # Create plot for car mode share evolution
    plot_car_share_evolution(all_results, modification_groups)
    
    log_print(f"\nSensitivity analysis complete. Results saved to {output_file}")
    log_print("Car share evolution plot has been created.")
    log_print("CSV file with plotted data has been created.")

if __name__ == "__main__":
    run_market_share_analysis()
