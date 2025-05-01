import pandas as pd
import matplotlib.pyplot as plt

def plot_share_evolution():
    """
    Create plots showing the evolution of predicted shares from CSV data.
    """
    # Set global font size
    plt.rcParams.update({'font.size': 14})
    
    # Read the CSV data
    df = pd.read_csv('car_share_evolution_data_PPT_plot.csv')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Define colors for each model
    colors = {'MNL': 'blue', 'NL': 'red'}
    
    # Plot lines for each model type
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        plt.plot(model_data['Multiplier'], 
                model_data['Car Share (%)'], 
                marker='o', 
                label=model, 
                color=colors[model])
    
    plt.xlabel('Car Cost Multiplier', fontsize=18)
    plt.ylabel('Car Market Share (%)', fontsize=18)
    plt.title('Scenario: Congestion Pricing', fontsize=20)
    plt.ylim(0, 100)  # Set y-axis from 0 to 100%
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=14)
    
    # Increase tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    plot_share_evolution()
