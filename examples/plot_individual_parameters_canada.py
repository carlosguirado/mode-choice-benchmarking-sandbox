from mcbs.datasets import DatasetLoader
from mcbs.models.modecanada_model import MixedLogitModel_MC
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

def calculate_individual_parameters(model, results, n_draws=1000):
    """
    Calculate individual-specific parameters using Bayes theorem
    following Revelt and Train (2000)
    """
    # Get parameter estimates
    betas = results.get_beta_values()
    b_time = betas['B_TIME']
    b_time_s = betas['B_TIME_S']
    b_cost = betas['B_COST']
    
    # Generate population-level random draws
    np.random.seed(42)
    random_draws = np.random.normal(0, 1, n_draws)
    # Generate time parameter draws (lognormal)
    beta_draws = -np.exp(b_time + b_time_s * random_draws)
    
    # For each individual, calculate conditional distribution
    n_individuals = len(model.database.data)
    individual_betas = np.zeros(n_individuals)
    
    for i in range(n_individuals):
        # Get individual's choice and characteristics
        choice = model.database.data['CHOICE'].iloc[i]
        
        # Calculate choice probability for each draw
        probs = np.zeros(n_draws)
        for r in range(n_draws):
            beta_r = beta_draws[r]
            # Calculate utility for each alternative using beta_r
            utilities = {
                1: beta_r * model.database.data['TRAIN_TIME'].iloc[i] + b_cost * model.database.data['TRAIN_COST'].iloc[i],
                2: beta_r * model.database.data['CAR_TIME'].iloc[i] + b_cost * model.database.data['CAR_COST'].iloc[i],
                3: beta_r * model.database.data['BUS_TIME'].iloc[i] + b_cost * model.database.data['BUS_COST'].iloc[i],
                4: beta_r * model.database.data['AIR_TIME'].iloc[i] + b_cost * model.database.data['AIR_COST'].iloc[i]
            }
            
            # Calculate probability of chosen alternative
            exp_utils = np.exp([v - max(utilities.values()) for v in utilities.values()])  # Subtract max for numerical stability
            sum_exp = sum(exp_utils)
            probs[r] = exp_utils[choice-1] / sum_exp if sum_exp > 0 else 0
            
        # Calculate individual's expected beta using Bayes theorem
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            weights = probs / sum_probs  # Normalize probabilities
            individual_betas[i] = np.sum(beta_draws * weights)
        else:
            # If all probabilities are zero, use the mean
            individual_betas[i] = np.mean(beta_draws)
    
    return individual_betas

def plot_parameter_distributions(model, results):
    """Plot distribution of individual-specific parameters and VOT"""
    # Calculate individual parameters
    print("\nCalculating individual parameters...")
    individual_betas = calculate_individual_parameters(model, results)
    
    # Remove any potential infinities or NaNs
    individual_betas = individual_betas[np.isfinite(individual_betas)]
    
    # Calculate VOT ($/hour)
    # Note: time parameters are already in per-minute units, so multiply by 60 for per-hour
    # Cost parameter is in dollars, so no need to convert
    b_cost = results.get_beta_values()['B_COST']
    vot = individual_betas / b_cost  # This gives $/minute
    vot = vot * 60  # Convert to $/hour
    
    # Remove any potential infinities or NaNs
    vot = vot[np.isfinite(vot)]
    
    # Set seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Define colors
    main_color = '#3498db'  # Blue
    kde_color = '#e74c3c'   # Red
    mean_color = '#2ecc71'  # Green
    median_color = '#f1c40f' # Yellow
    
    # Plot 1: Time Parameter Distribution
    n, bins, patches = ax1.hist(individual_betas, bins=50, density=True, 
                               alpha=0.7, color=main_color, 
                               label='Individual Parameters')
    
    # Add KDE plot for time parameter
    kde = stats.gaussian_kde(individual_betas)
    x_range = np.linspace(min(individual_betas), max(individual_betas), 200)
    ax1.plot(x_range, kde(x_range), color=kde_color, 
             lw=2, label='Kernel Density Estimate')
    
    # Add vertical lines for time parameter statistics
    mean_time = np.mean(individual_betas)
    median_time = np.median(individual_betas)
    q25_time, q75_time = np.percentile(individual_betas, [25, 75])
    
    ax1.axvline(x=mean_time, color=mean_color, linestyle='--', 
                label=f'Mean: {mean_time:.3f}')
    ax1.axvline(x=median_time, color=median_color, linestyle=':', 
                label=f'Median: {median_time:.3f}')
    
    ax1.set_title('Distribution of Individual-Specific Time Parameters\n(utils per minute)', 
                  fontsize=12, pad=20)
    ax1.set_xlabel('Time Parameter Value (utils/min)', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.legend(fontsize=9)
    
    # Plot 2: Value of Time Distribution
    n, bins, patches = ax2.hist(vot, bins=50, density=True, 
                               alpha=0.7, color=main_color, 
                               label='Individual VOT')
    
    # Add KDE plot for VOT
    kde_vot = stats.gaussian_kde(vot)
    x_range_vot = np.linspace(min(vot), max(vot), 200)
    ax2.plot(x_range_vot, kde_vot(x_range_vot), color=kde_color, 
             lw=2, label='Kernel Density Estimate')
    
    # Add vertical lines for VOT statistics
    mean_vot = np.mean(vot)
    median_vot = np.median(vot)
    q25_vot, q75_vot = np.percentile(vot, [25, 75])
    
    ax2.axvline(x=mean_vot, color=mean_color, linestyle='--', 
                label=f'Mean: {mean_vot:.1f}')
    ax2.axvline(x=median_vot, color=median_color, linestyle=':', 
                label=f'Median: {median_vot:.1f}')
    
    ax2.set_title('Distribution of Individual-Specific Value of Time', 
                  fontsize=12, pad=20)
    ax2.set_xlabel('Value of Time ($/hour)', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nIndividual-Specific Time Parameter Distribution Summary:")
    print(f"Mean: {mean_time:.4f} utils/min")
    print(f"Median: {median_time:.4f} utils/min")
    print(f"Std Dev: {np.std(individual_betas):.4f}")
    print(f"25th percentile: {q25_time:.4f}")
    print(f"75th percentile: {q75_time:.4f}")
    
    print("\nIndividual-Specific Value of Time Distribution Summary ($/hour):")
    print(f"Mean: {mean_vot:.2f}")
    print(f"Median: {median_vot:.2f}")
    print(f"Std Dev: {np.std(vot):.2f}")
    print(f"25th percentile: {q25_vot:.2f}")
    print(f"75th percentile: {q75_vot:.2f}")
    
    print(f"\nFixed Cost Parameter: {b_cost:.4f} utils/$")
    
    print(f"\nParameter Interpretation:")
    print("- Individual-specific parameters are calculated using Bayes theorem")
    print("  to condition on each person's observed choices")
    print("- The time parameter follows a lognormal distribution")
    print("  transformed to negative to ensure disutility of travel time")
    print("- Time parameters are in utils per minute")
    print("- Cost parameters are in utils per dollar")
    print("- VOT is calculated as (utils/min)/(utils/$) * 60 to get $/hour")
    print("- The VOT distribution shows the monetary value individuals place")
    print("  on saving one hour of travel time")

def main():
    # Load data
    loader = DatasetLoader()
    data = loader.load_dataset("modecanada_dataset")
    
    # Take a stratified sample of 400 individuals
    sample_size = 400
    np.random.seed(42)
    
    # Get unique cases and their choices
    cases = data['case'].unique()
    case_choices = data.groupby('case')['choice'].first()
    
    # Stratified sampling of cases based on choices
    sampled_cases = []
    for choice in case_choices.unique():
        choice_cases = cases[case_choices == choice]
        n_sample = max(int(sample_size * len(choice_cases) / len(cases)), 1)
        sampled_cases.extend(np.random.choice(choice_cases, size=n_sample, replace=False))
    
    # Filter data to only include sampled cases
    sampled_data = data[data['case'].isin(sampled_cases)]
    
    # Create and estimate mixed logit model with sampled data
    print("\nEstimating Mixed Logit Model...")
    model = MixedLogitModel_MC(sampled_data)
    results = model.estimate()
    
    # Print model coefficients
    print("\nEstimated Model Coefficients:")
    print("-" * 50)
    betas = results.get_beta_values()
    for name, value in betas.items():
        print(f"{name:15s}: {value:10.4f}")
    print("-" * 50)
    
    # Plot parameter distributions
    plot_parameter_distributions(model, results)

if __name__ == '__main__':
    main()
