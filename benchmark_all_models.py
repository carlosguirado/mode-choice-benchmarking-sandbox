from mcbs.benchmarker.benchmarker import ModelBenchmarker
from mcbs.datasets.dataset_loader import DatasetLoader
from mcbs.models.swissmetro_model import MultinomialLogitModel_SM, NestedLogitModel_SM, MixedLogitModel_SM
from mcbs.models.ltds_model import MultinomialLogitModel_L, MultinomialLogitModelTotal_L, NestedLogitModel_L
from mcbs.models.modecanada_model import MultinomialLogitModel_MC, NestedLogitModel3_MC, MixedLogitModel_MC
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

# def create_comparison_plots(combined_results):
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
#    datasets = combined_results['dataset'].unique()
#    colors = ['blue', 'red', 'green']
#    markers = ['o', 's', '^']
   
#    # Plot 1: Market Share Accuracy vs Rho Bar Squared
#    texts1 = []
#    for dataset, color, marker in zip(datasets, colors, markers):
#        dataset_data = combined_results[combined_results['dataset'] == dataset]
#        ax1.scatter(dataset_data['rho_squared_bar'],
#                   dataset_data['market_share_accuracy'],
#                   label=dataset,
#                   color=color,
#                   marker=marker,
#                   s=100)
       
#        for x, y, label in zip(dataset_data['rho_squared_bar'],
#                             dataset_data['market_share_accuracy'],
#                             dataset_data['model_name']):
#            texts1.append(ax1.text(x, y, label))
   
#    adjust_text(texts1, 
#               force_points=0.2,
#               force_text=0.2,
#               expand_points=(1.5, 1.5),
#               expand_text=(1.5, 1.5),
#               arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

#    ax1.set_xlabel('Rho Bar Squared')
#    ax1.set_ylabel('Market Share Accuracy')
#    ax1.set_title('Market Share Accuracy vs Rho Bar Squared')
#    ax1.grid(True, linestyle='--', alpha=0.7)
#    ax1.legend()

#    # Plot 2: Choice Prediction Accuracy vs Rho Bar Squared
#    texts2 = []
#    for dataset, color, marker in zip(datasets, colors, markers):
#        dataset_data = combined_results[combined_results['dataset'] == dataset]
#        ax2.scatter(dataset_data['rho_squared_bar'],
#                   dataset_data['choice_accuracy'],
#                   label=dataset,
#                   color=color,
#                   marker=marker,
#                   s=100)
       
#        for x, y, label in zip(dataset_data['rho_squared_bar'],
#                             dataset_data['choice_accuracy'],
#                             dataset_data['model_name']):
#            texts2.append(ax2.text(x, y, label))
   
#    adjust_text(texts2,
#               force_points=0.2, 
#               force_text=0.2,
#               expand_points=(1.5, 1.5),
#               expand_text=(1.5, 1.5),
#               arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

#    ax2.set_xlabel('Rho Bar Squared')
#    ax2.set_ylabel('Choice Prediction Accuracy')
#    ax2.set_title('Choice Prediction Accuracy vs Rho Bar Squared')
#    ax2.grid(True, linestyle='--', alpha=0.7)
#    ax2.legend()

#    plt.tight_layout()
#    plt.savefig('benchmark_comparison_plots.png', dpi=300, bbox_inches='tight')
#    plt.close()

def create_comparison_plots(combined_results):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Plot 1: Market Share Accuracy vs Rho Bar Squared
    datasets = combined_results['dataset'].unique()
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    for dataset, color, marker in zip(datasets, colors, markers):
        dataset_data = combined_results[combined_results['dataset'] == dataset]
        ax1.scatter(dataset_data['rho_squared_bar'],
        dataset_data['market_share_accuracy'],
        label=dataset,
        color=color,
        marker=marker,
        s=100)

    ax1.set_xlabel('Rho Bar Squared')
    ax1.set_ylabel('Market Share Accuracy')
    ax1.set_title('Market Share Accuracy vs Rho Bar Squared')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot 2: Choice Prediction Accuracy vs Rho Bar Squared
    for dataset, color, marker in zip(datasets, colors, markers):
        dataset_data = combined_results[combined_results['dataset'] == dataset]
        ax2.scatter(dataset_data['rho_squared_bar'],
        dataset_data['choice_accuracy'],
        label=dataset,
        color=color,
        marker=marker,
        s=100)
        
    ax2.set_xlabel('Rho Bar Squared')
    ax2.set_ylabel('Choice Prediction Accuracy')
    ax2.set_title('Choice Prediction Accuracy vs Rho Bar Squared')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    plt.tight_layout()
    plt.savefig('benchmark_comparison_plots_nolabel.png', dpi=300, bbox_inches='tight')
    plt.close()



def main():
    # Initialize dataset loader and benchmarker
    loader = DatasetLoader()
    benchmarker = ModelBenchmarker()
    
    print("Starting benchmarking process for all datasets...")
    
    # Swissmetro Dataset
    print("\n" + "="*50)
    print("Swissmetro Dataset Analysis")
    print("="*50)
    swissmetro_data = loader.load_dataset("swissmetro_dataset")
    print(f"Dataset shape: {swissmetro_data.shape}")
    models = [MultinomialLogitModel_SM, NestedLogitModel_SM, MixedLogitModel_SM]
    results_df_SM = benchmarker.run_benchmark(
        data=swissmetro_data,
        models=models,
        dataset_name="swissmetro"
    )
    benchmarker.print_comparison()
    benchmarker.export_results("swissmetro_benchmark_results.csv")
    
    # LTDS Dataset
    print("\n" + "="*50)
    print("London Travel Demand Survey (LTDS) Analysis")
    print("="*50)
    ltds_data = loader.load_dataset("ltds_dataset")
    print(f"Dataset shape: {ltds_data.shape}")
    models = [MultinomialLogitModel_L, MultinomialLogitModelTotal_L, NestedLogitModel_L]
    results_df_L = benchmarker.run_benchmark(
        data=ltds_data,
        models=models,
        dataset_name="ltds"
    )
    benchmarker.print_comparison()
    benchmarker.export_results("ltds_benchmark_results.csv")
    
    # ModeCanada Dataset
    print("\n" + "="*50)
    print("ModeCanada Dataset Analysis")
    print("="*50)
    modecanada_data = loader.load_dataset("modecanada_dataset")
    print(f"Dataset shape: {modecanada_data.shape}")
    models = [MultinomialLogitModel_MC, NestedLogitModel3_MC, MixedLogitModel_MC]
    results_df_MC = benchmarker.run_benchmark(
        data=modecanada_data,
        models=models,
        dataset_name="modecanada"
    )
    benchmarker.print_comparison()
    benchmarker.export_results("modecanada_benchmark_results.csv")
    
    # Add dataset column to each results dataframe
    results_df_SM['dataset'] = 'Swissmetro'
    results_df_L['dataset'] = 'LTDS'
    results_df_MC['dataset'] = 'ModeCanada'
    
    # Combine all results
    combined_results = pd.concat([results_df_SM, results_df_L, results_df_MC], ignore_index=True)

    print(combined_results)
    
    # Create comparison plots
    create_comparison_plots(combined_results)
    
    print("\nBenchmarking complete!")
    print("Comparison plots have been saved as 'benchmark_comparison_plots.png'")

if __name__ == "__main__":
    main()
