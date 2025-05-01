"""
Module for systematic comparison of discrete choice models.
Handles running multiple models and collecting their performance metrics.
"""

from typing import Dict, List, Type, Optional
import pandas as pd
from ..models.base import BaseDiscreteChoiceModel
from ..datasets.dataset_loader import DatasetLoader

class ModelBenchmarker:
    """Class to handle systematic model comparison and benchmarking."""
    
    def __init__(self):
        """Initialize the benchmarker."""
        self.results = {}
        self.metrics_df = None
        
    def run_benchmark(self, 
                     data: pd.DataFrame,
                     models: List[Type[BaseDiscreteChoiceModel]],
                     dataset_name: Optional[str] = None) -> pd.DataFrame:
        """
        Run benchmark comparison of multiple models on a dataset.
        
        Args:
            data: DataFrame containing the choice data
            models: List of model classes to benchmark
            dataset_name: Name of the dataset (for reporting)
            
        Returns:
            DataFrame containing comparison metrics
        """
        results = []
        
        for model_class in models:
            model_name = model_class.__name__
            print(f"\nEstimating {model_name}...")
            
            try:
                # Initialize and estimate model
                model = model_class(data)
                estimation_results = model.estimate()
                
                # Get metrics
                metrics = model.get_metrics()
                metrics['model_name'] = model_name
                if dataset_name:
                    metrics['dataset'] = dataset_name
                    
                # Store full results
                self.results[model_name] = {
                    'model': model,
                    'estimation_results': estimation_results,
                    'metrics': metrics
                }
                
                results.append(metrics)
                
            except Exception as e:
                print(f"Error estimating {model_name}: {str(e)}")
                continue
        
        # Create comparison DataFrame
        self.metrics_df = pd.DataFrame(results)
        if self.metrics_df is not None and not self.metrics_df.empty:
            self._sort_metrics()
        
        return self.metrics_df
    
    def _sort_metrics(self):
        """Sort metrics by rho squared bar and final log likelihood."""
        if 'rho_squared_bar' in self.metrics_df.columns:
            self.metrics_df.sort_values(['rho_squared_bar', 'final_ll'], 
                                      ascending=[False, False], 
                                      inplace=True)
        elif 'final_ll' in self.metrics_df.columns:
            self.metrics_df.sort_values('final_ll', 
                                      ascending=False, 
                                      inplace=True)
    
    def get_best_model(self, criterion: str = 'final_ll') -> str:
        """
        Get the best performing model according to specified criterion.
        
        Args:
            criterion: Metric to use for comparison
            
        Returns:
            Name of the best performing model
        """
        if self.metrics_df is None:
            raise ValueError("No benchmark results available")
            
        # If the requested criterion isn't available, fall back to final_ll
        if criterion not in self.metrics_df.columns:
            if 'final_ll' in self.metrics_df.columns:
                criterion = 'final_ll'
            else:
                raise ValueError("No valid comparison criteria available")
            
        best_idx = self.metrics_df[criterion].argmax()
        return self.metrics_df.iloc[best_idx]['model_name']
    
    def print_comparison(self):
        """Print formatted comparison of model results."""
        if self.metrics_df is None:
            print("No benchmark results available")
            return
            
        print("\nModel Comparison Results:")
        print("=" * 120)
        
        # Select and order columns for display
        display_columns = [
            'model_name',
            'final_ll',
            'rho_squared_bar',
            'market_share_accuracy',
            'choice_accuracy',
            'vot',
            'n_parameters',
            'n_observations'
        ]
        
        # Filter columns that exist in the DataFrame
        display_columns = [col for col in display_columns if col in self.metrics_df.columns]
        
        # Format metrics for display
        display_metrics = self.metrics_df[display_columns].round(4)
        
        # Print comparison
        print(display_metrics.to_string(index=False))
        
        # Print best model
        try:
            if 'rho_squared_bar' in self.metrics_df.columns:
                best_model = self.get_best_model('rho_squared_bar')
                print("\nBest performing model (by rho squared bar):", best_model)
            else:
                best_model = self.get_best_model('final_ll')
                print("\nBest performing model (by final log likelihood):", best_model)
        except ValueError as e:
            print("\nUnable to determine best model:", str(e))
    
    def export_results(self, filepath: str):
        """
        Export benchmark results to CSV.
        
        Args:
            filepath: Path to save results CSV
        """
        if self.metrics_df is not None:
            # Select and order columns for export
            export_columns = [
                'model_name',
                'final_ll',
                'rho_squared_bar',
                'market_share_accuracy',
                'choice_accuracy',
                'vot',
                'n_parameters',
                'n_observations',
                'actual_shares',
                'predicted_shares',
                'confusion_matrix'
            ]
            
            # Filter columns that exist in the DataFrame
            export_columns = [col for col in export_columns if col in self.metrics_df.columns]
            
            # Export selected columns
            self.metrics_df[export_columns].to_csv(filepath, index=False)
            print(f"\nResults exported to {filepath}")
