# mcbs/benchmarking/benchmark.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Callable, Tuple
from ..datasets import DatasetLoader

class Benchmark:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name  # Store dataset name
        self.dataset_loader = DatasetLoader()
        self.dataset = self.dataset_loader.load_dataset(dataset_name)
        self.dataset_info = self.dataset_loader.get_dataset_info(dataset_name)

    def run(self, models: Dict[str, Callable]) -> pd.DataFrame:
        """
        Run the benchmark for multiple models.
        
        Args:
            models (Dict[str, Callable]): Dictionary of model names and their corresponding functions.
                Each function should return a tuple (biogeme_results, y_true, y_pred).
        
        Returns:
            pd.DataFrame: Benchmark results for all models.
        """
        print("\nBenchmarking Information:")
        print("-------------------------")
        print(f"Dataset: {self.dataset_name}")
        print(f"Number of observations: {len(self.dataset)}")
        print("\nMetrics being benchmarked:")
        print("- Log-likelihood: Model's log-likelihood")
        print("- Null log-likelihood: Log-likelihood of the null model")
        print("- Rho-squared: McFadden's rho-squared")
        print("- Final log-likelihood: Final value of the log-likelihood")
        print("- Prediction accuracy: Proportion of correctly predicted choices")
        
        results = {}
        for model_name, model_func in models.items():
            print(f"\nRunning benchmark for {model_name}...")
            model_results, y_true, y_pred = model_func(self.dataset)
            metrics = self._extract_metrics(model_results, y_true, y_pred)
            results[model_name] = metrics
        
        return pd.DataFrame(results).T

    def _extract_metrics(self, biogeme_results: Any, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Extract metrics from model results.
        """
        metrics = {}
    
        # Final log-likelihood
        try:
            metrics['final_log_likelihood'] = float(biogeme_results.data.logLike)
        except (AttributeError, TypeError):
            metrics['final_log_likelihood'] = np.nan
    
        # Rho-squared (calculate manually if not available)
        try:
            null_ll = float(biogeme_results.data.nullLogLike)
            final_ll = float(biogeme_results.data.logLike)
            metrics['rho_squared'] = 1.0 - (final_ll / null_ll)
        except (AttributeError, TypeError, ZeroDivisionError):
            try:
                metrics['rho_squared'] = float(biogeme_results.data.rhoSquare)
            except (AttributeError, TypeError):
                metrics['rho_squared'] = np.nan
    
        # Value of Time (in CHF/hour)
        try:
            betas = biogeme_results.get_beta_values()
            time_coeff = float(betas['B_TIME'])
            cost_coeff = float(betas['B_COST'])
            # Convert to CHF/hour (time coefficient is in minutes and prices are in cents)
            vot = abs(time_coeff / cost_coeff) * 60 / 100
            metrics['value_of_time'] = float(vot)
        except (KeyError, ZeroDivisionError, TypeError) as e:
            print(f"Warning: Could not calculate Value of Time: {str(e)}")
            metrics['value_of_time'] = np.nan

    # Forecasting accuracy
        try:
            y_pred_labels = np.argmax(y_pred, axis=1)
            accuracy = np.mean(y_true == y_pred_labels)
            metrics['forecasting_accuracy'] = float(accuracy)
        except Exception as e:
            print(f"Warning: Could not calculate forecasting accuracy: {str(e)}")
            metrics['forecasting_accuracy'] = np.nan

    # Print the calculated metrics with proper formatting
        print("\nCalculated metrics:")
        for name, value in metrics.items():
            if name == 'value_of_time' and not np.isnan(value):
                print(f"{name}: {value:.2f} CHF/hour")
            elif not np.isnan(value):
                if name in ['final_log_likelihood']:
                    print(f"{name}: {value:.2f}")
                else:
                    print(f"{name}: {value:.4f}")
            else:
                print(f"{name}: Not available")

        return metrics

    def compare_results(self, results: pd.DataFrame) -> None:
        print(f"\nBenchmark Results for dataset: {self.dataset_name}")
        print("=" * 50)
        print(results)
        
        print("\nBest performing model by metric:")
        for metric in results.columns:
            if not results[metric].isnull().all():
                if metric in ['log_likelihood', 'prediction_accuracy']:
                    best_model = results[metric].idxmax()
                else:
                    best_model = results[metric].idxmin()
                print(f"{metric}: {best_model}")

    def plot_results(self, results: pd.DataFrame) -> None:
        """
        Plot benchmark results.
        """
        try:
            import matplotlib.pyplot as plt

            # Only plot metrics that have values
            metrics_to_plot = [col for col in results.columns 
                             if not results[col].isnull().all()]
            
            if not metrics_to_plot:
                print("No metrics available to plot.")
                return

            fig, axes = plt.subplots(len(metrics_to_plot), 1, 
                                   figsize=(10, 4*len(metrics_to_plot)))
            
            if len(metrics_to_plot) == 1:
                axes = [axes]

            for i, metric in enumerate(metrics_to_plot):
                results[metric].plot(kind='bar', ax=axes[i], 
                                   title=f'{metric} by Model')
                axes[i].set_ylabel(metric)
                plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib is not installed. Skipping plots.")