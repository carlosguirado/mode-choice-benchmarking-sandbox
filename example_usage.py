"""
Example usage of the benchmarker with Swissmetro, LTDS, and ModeCanada models
"""

from mcbs.datasets import DatasetLoader
from mcbs.benchmarker import ModelBenchmarker
from mcbs.models.swissmetro_model import (
    MultinomialLogitModel as SwissMetroMNL,
    NestedLogitModel as SwissMetroNL,
    MixedLogitModel as SwissMetroML
)
from mcbs.models.ltds_model import (
    MultinomialLogitModel as LTDSMNL,
    NestedLogitModel as LTDSNL1,
    NestedLogitModel2 as LTDSNL2
)
from mcbs.models.modecanada_model import (
    MultinomialLogitModel as ModeCanadaMNL,
    NestedLogitModel as ModeCanadaNL1,
    NestedLogitModel2 as ModeCanadaNL2,
    NestedLogitModel3 as ModeCanadaNL3
)

def run_swissmetro_benchmark():
    """Run benchmark for Swissmetro models"""
    print("\nRunning Swissmetro Benchmark...")
    
    # Load data
    loader = DatasetLoader()
    data = loader.load_dataset("swissmetro_dataset")
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker()
    
    # Define models to compare
    models = [SwissMetroMNL, SwissMetroNL, SwissMetroML]
    
    # Run benchmark
    results = benchmarker.run_benchmark(
        data=data,
        models=models,
        dataset_name='swissmetro'
    )
    
    # Print comparison
    print("\nSwissmetro Models Comparison:")
    print("=" * 50)
    benchmarker.print_comparison()
    
    # Save results
    benchmarker.export_results('swissmetro_benchmark_results.csv')

def run_ltds_benchmark():
    """Run benchmark for LTDS models"""
    print("\nRunning LTDS Benchmark...")
    
    # Load data
    loader = DatasetLoader()
    data = loader.load_dataset("ltds_dataset")
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker()
    
    # Define models to compare
    models = [LTDSMNL, LTDSNL1, LTDSNL2]
    
    # Run benchmark
    results = benchmarker.run_benchmark(
        data=data,
        models=models,
        dataset_name='ltds'
    )
    
    # Print comparison
    print("\nLTDS Models Comparison:")
    print("=" * 50)
    # Update model names in the comparison for clarity
    results['model_name'] = results['model_name'].replace({
        'MultinomialLogitModel': 'LTDS MNL',
        'NestedLogitModel': 'LTDS NL (Motorized)',
        'NestedLogitModel2': 'LTDS NL (Active)'
    })
    benchmarker.print_comparison()
    
    # Save results
    benchmarker.export_results('ltds_benchmark_results.csv')

def run_modecanada_benchmark():
    """Run benchmark for ModeCanada models"""
    print("\nRunning ModeCanada Benchmark...")
    
    # Load data
    loader = DatasetLoader()
    data = loader.load_dataset("modecanada_dataset")
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker()
    
    # Define models to compare
    models = [ModeCanadaMNL, ModeCanadaNL1, ModeCanadaNL2, ModeCanadaNL3]
    
    # Run benchmark
    results = benchmarker.run_benchmark(
        data=data,
        models=models,
        dataset_name='modecanada'
    )
    
    # Print comparison
    print("\nModeCanada Models Comparison:")
    print("=" * 50)
    # Update model names in the comparison for clarity
    results['model_name'] = results['model_name'].replace({
        'MultinomialLogitModel': 'ModeCanada MNL',
        'NestedLogitModel': 'ModeCanada NL (Public Transport)',
        'NestedLogitModel2': 'ModeCanada NL (Motorized)',
        'NestedLogitModel3': 'ModeCanada NL (Private vs Public)'
    })
    benchmarker.print_comparison()
    
    # Save results
    benchmarker.export_results('modecanada_benchmark_results.csv')

def main():
    # Run benchmarks for all datasets
    run_swissmetro_benchmark()
    run_ltds_benchmark()
    run_modecanada_benchmark()

if __name__ == "__main__":
    main()
