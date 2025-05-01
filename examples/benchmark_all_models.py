from mcbs.benchmarker.benchmarker import ModelBenchmarker
from mcbs.datasets.dataset_loader import DatasetLoader
from mcbs.models.swissmetro_model import MultinomialLogitModel_SM, NestedLogitModel_SM, MixedLogitModel_SM
from mcbs.models.ltds_model import MultinomialLogitModel_L, MultinomialLogitModelTotal_L, NestedLogitModel_L
from mcbs.models.modecanada_model import MultinomialLogitModel_MC, NestedLogitModel3_MC, MixedLogitModel_MC
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

def create_market_share_plot(combined_results):
    """Create market share accuracy vs rho bar squared plot."""
    plt.figure(figsize=(8, 6))
    datasets = combined_results['dataset'].unique()
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for dataset, color, marker in zip(datasets, colors, markers):
        dataset_data = combined_results[combined_results['dataset'] == dataset]
        plt.scatter(dataset_data['rho_squared_bar'],
                   dataset_data['market_share_accuracy'],
                   label=dataset,
                   color=color,
                   marker=marker,
                   s=100)

    plt.xlabel('Rho Bar Squared')
    plt.ylabel('Market Share Accuracy')
    plt.title('Market Share Accuracy vs Rho Bar Squared')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Set fixed axis ranges from 0 to 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('market_share_accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_choice_accuracy_plot(combined_results):
    """Create choice prediction accuracy vs rho bar squared plot."""
    plt.figure(figsize=(8, 6))
    
    # Set font sizes
    TITLE_SIZE = 20
    LABEL_SIZE = 18
    TICK_SIZE = 16
    
    datasets = combined_results['dataset'].unique()
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for dataset, color, marker in zip(datasets, colors, markers):
        dataset_data = combined_results[combined_results['dataset'] == dataset]
        plt.scatter(dataset_data['rho_squared_bar'],
                   dataset_data['choice_accuracy'],
                   label=dataset,
                   color=color,
                   marker=marker,
                   s=100)

    # Set font sizes for title and labels
    plt.xlabel('Rho Bar Squared', fontsize=LABEL_SIZE)
    plt.ylabel('Choice Prediction Accuracy', fontsize=LABEL_SIZE)
    plt.title('Choice Prediction Accuracy vs Rho Bar Squared', fontsize=TITLE_SIZE)
    
    # Set tick font sizes
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    
    # Set legend font size
    plt.legend(fontsize=TICK_SIZE)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set fixed axis ranges from 0 to 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('choice_accuracy_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Initialize dataset loader
    loader = DatasetLoader()
    
    print("Starting benchmarking process for all datasets...")
    
    # Store results for each dataset
    results_dfs = []
    
    # Swissmetro Dataset
    print("\n" + "="*50)
    print("Swissmetro Dataset Analysis")
    print("="*50)
    swissmetro_data = loader.load_dataset("swissmetro_dataset")
    print(f"Dataset shape: {swissmetro_data.shape}")
    benchmarker = ModelBenchmarker()
    models = [MultinomialLogitModel_SM, NestedLogitModel_SM, MixedLogitModel_SM]
    results_df_SM = benchmarker.run_benchmark(
        data=swissmetro_data,
        models=models,
        dataset_name="swissmetro"
    )
    if results_df_SM is not None:
        results_df_SM['dataset'] = 'Swissmetro'
        results_dfs.append(results_df_SM)
        benchmarker.print_comparison()
        benchmarker.export_results("swissmetro_benchmark_results.csv")
    
    # LTDS Dataset
    print("\n" + "="*50)
    print("London Travel Demand Survey (LTDS) Analysis")
    print("="*50)
    ltds_data = loader.load_dataset("ltds_dataset")
    print(f"Dataset shape: {ltds_data.shape}")
    benchmarker = ModelBenchmarker()
    models = [MultinomialLogitModel_L, MultinomialLogitModelTotal_L, NestedLogitModel_L]
    results_df_L = benchmarker.run_benchmark(
        data=ltds_data,
        models=models,
        dataset_name="ltds"
    )
    if results_df_L is not None:
        results_df_L['dataset'] = 'London'
        results_dfs.append(results_df_L)
        benchmarker.print_comparison()
        benchmarker.export_results("ltds_benchmark_results.csv")
    
    # ModeCanada Dataset
    print("\n" + "="*50)
    print("ModeCanada Dataset Analysis")
    print("="*50)
    modecanada_data = loader.load_dataset("modecanada_dataset")
    print(f"Dataset shape: {modecanada_data.shape}")
    benchmarker = ModelBenchmarker()
    models = [MultinomialLogitModel_MC, NestedLogitModel3_MC, MixedLogitModel_MC]
    results_df_MC = benchmarker.run_benchmark(
        data=modecanada_data,
        models=models,
        dataset_name="modecanada"
    )
    if results_df_MC is not None:
        results_df_MC['dataset'] = 'Canada'
        results_dfs.append(results_df_MC)
        benchmarker.print_comparison()
        benchmarker.export_results("modecanada_benchmark_results.csv")
    
    # Combine all results if we have any
    if results_dfs:
        combined_results = pd.concat(results_dfs, ignore_index=True)
        print("\nCombined Results:")
        print(combined_results)
        
        # Create separate plots
        create_market_share_plot(combined_results)
        create_choice_accuracy_plot(combined_results)
        
        print("\nBenchmarking complete!")
        print("Plots have been saved as 'market_share_accuracy_plot.png' and 'choice_accuracy_plot.png'")
    else:
        print("\nNo valid results to combine and plot.")

if __name__ == "__main__":
    main()
