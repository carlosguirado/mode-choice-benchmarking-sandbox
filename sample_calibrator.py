import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Callable
from scipy.optimize import minimize

from mcbs.benchmarking.benchmark import Benchmark
from mcbs.datasets.dataset_loader import DatasetLoader


class SwissmetroCalibrator:
    def __init__(self, data: pd.DataFrame, utility_func: Callable):
        """
        Initialize calibrator for Swissmetro models.
        
        Args:
            data: Prepared Swissmetro dataset
            utility_func: Function that calculates utilities given parameters
        """
        self.data = data
        self.utility_func = utility_func
        self.mode_list = ['TRAIN', 'SM', 'CAR']
        self.observed_shares = self._calculate_observed_shares()
        
    def _calculate_observed_shares(self) -> Dict[str, float]:
        """Calculate observed mode shares from data."""
        choice_counts = (self.data['CHOICE'].value_counts() / len(self.data)).to_dict()
        return {self.mode_list[i-1]: choice_counts.get(i, 0) for i in [1, 2, 3]}
    
    def _calculate_probabilities(self, params: Dict[str, float]) -> np.ndarray:
        """Calculate choice probabilities using current parameters."""
        utilities = self.utility_func(self.data, params)
        exp_utilities = np.exp(utilities)
        
        availabilities = np.column_stack([
            self.data['TRAIN_AV'].values,
            self.data['SM_AV'].values,
            self.data['CAR_AV'].values
        ])
        
        sum_exp_utilities = np.sum(exp_utilities * availabilities, axis=1, keepdims=True)
        probabilities = exp_utilities / sum_exp_utilities
        probabilities = probabilities * availabilities
        
        row_sums = probabilities.sum(axis=1)
        probabilities = probabilities / row_sums[:, np.newaxis]
        
        return probabilities
    
    def _objective_function(self, asc_values: np.ndarray) -> float:
        """Objective function for optimization."""
        params = {
            'ASC_TRAIN': asc_values[0],
            'ASC_CAR': asc_values[1]
        }
        
        probabilities = self._calculate_probabilities(params)
        predicted_shares = probabilities.mean(axis=0)
        
        observed = np.array([self.observed_shares[mode] for mode in self.mode_list])
        return np.sum((predicted_shares - observed) ** 2)
    
    def calibrate(self, initial_ascs: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calibrate alternative specific constants.
        
        Returns:
            Dictionary of calibrated ASCs
        """
        if initial_ascs is None:
            initial_ascs = {'ASC_TRAIN': 0.0, 'ASC_CAR': 0.0}
        
        x0 = np.array([initial_ascs['ASC_TRAIN'], initial_ascs['ASC_CAR']])
        
        result = minimize(
            self._objective_function,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        return {
            'ASC_TRAIN': result.x[0],
            'ASC_CAR': result.x[1]
        }

class SwissmetroValidator:
    def __init__(self, data: pd.DataFrame, utility_func: Callable):
        """
        Initialize validator for Swissmetro models.
        
        Args:
            data: Prepared Swissmetro dataset
            utility_func: Function that calculates utilities given parameters
        """
        self.data = data
        self.utility_func = utility_func
        self.mode_list = ['TRAIN', 'SM', 'CAR']
        
    def calculate_metrics(self, params: Dict[str, float]) -> Dict[str, float]:
        """Calculate validation metrics for given parameters."""
        utilities = self.utility_func(self.data, params)
        probabilities = self._calculate_probabilities(utilities)
        
        # Convert CHOICE to 0-based index
        y_true = self.data['CHOICE'].values - 1
        y_pred = np.argmax(probabilities, axis=1)
        
        metrics = {}
        metrics['accuracy'] = (y_true == y_pred).mean()
        
        # Calculate predicted and actual shares
        actual_shares = np.zeros(3)
        predicted_shares = np.zeros(3)
        for i in range(3):
            actual_shares[i] = (y_true == i).mean()
            predicted_shares[i] = (y_pred == i).mean()
            
        metrics['share_rmse'] = np.sqrt(np.mean((actual_shares - predicted_shares) ** 2))
        
        # Log likelihood
        row_probs = probabilities[np.arange(len(y_true)), y_true]
        metrics['log_likelihood'] = np.sum(np.log(row_probs))
        
        return metrics
    
    def _calculate_probabilities(self, utilities: np.ndarray) -> np.ndarray:
        """Calculate choice probabilities from utilities."""
        exp_utilities = np.exp(utilities)
        availabilities = np.column_stack([
            self.data['TRAIN_AV'].values,
            self.data['SM_AV'].values,
            self.data['CAR_AV'].values
        ])
        
        sum_exp_utilities = np.sum(exp_utilities * availabilities, axis=1, keepdims=True)
        probabilities = exp_utilities / sum_exp_utilities
        probabilities = probabilities * availabilities
        
        row_sums = probabilities.sum(axis=1)
        return probabilities / row_sums[:, np.newaxis]
    
    def plot_shares(self, params: Dict[str, float], title: str = "Mode Shares Comparison"):
        """Plot comparison of predicted and actual shares."""
        utilities = self.utility_func(self.data, params)
        probabilities = self._calculate_probabilities(utilities)
        y_pred = np.argmax(probabilities, axis=1)
        y_true = self.data['CHOICE'].values - 1
        
        actual_shares = [np.mean(y_true == i) for i in range(3)]
        predicted_shares = [np.mean(y_pred == i) for i in range(3)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(self.mode_list))
        width = 0.35
        
        ax.bar(x - width/2, actual_shares, width, label='Actual')
        ax.bar(x + width/2, predicted_shares, width, label='Predicted')
        
        ax.set_ylabel('Share')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(self.mode_list)
        ax.legend()
        
        plt.tight_layout()
        return fig

def utility_calculator_base(data: pd.DataFrame, params: Dict[str, float]) -> np.ndarray:
    """Calculate utilities for base model."""
    utilities = np.zeros((len(data), 3))
    
    # Train utility
    utilities[:, 0] = (params.get('ASC_TRAIN', 0) +
                      params.get('B_TIME', 0) * data['TRAIN_TT'] * data['TRAIN_AV'] +
                      params.get('B_COST', 0) * data['TRAIN_CO'] * data['TRAIN_AV'])
    
    # Swissmetro utility (base alternative)
    utilities[:, 1] = (params.get('B_TIME', 0) * data['SM_TT'] * data['SM_AV'] +
                      params.get('B_COST', 0) * data['SM_CO'] * data['SM_AV'])
    
    # Car utility
    utilities[:, 2] = (params.get('ASC_CAR', 0) +
                      params.get('B_TIME', 0) * data['CAR_TT'] * data['CAR_AV'] +
                      params.get('B_COST', 0) * data['CAR_CO'] * data['CAR_AV'])
    
    return utilities

def utility_calculator_purpose(data: pd.DataFrame, params: Dict[str, float]) -> np.ndarray:
    """Calculate utilities for purpose model."""
    utilities = np.zeros((len(data), 3))
    purposes = (data['PURPOSE'] == 1).values
    
    # Train utility
    utilities[:, 0] = (params.get('ASC_TRAIN', 0) +
                      params.get('B_TIME', 0) * data['TRAIN_TT'] * data['TRAIN_AV'] +
                      params.get('B_COST', 0) * data['TRAIN_CO'] * data['TRAIN_AV'] +
                      params.get('B_PURPOSE_TRAIN', 0) * purposes)
    
    # Swissmetro utility
    utilities[:, 1] = (params.get('B_TIME', 0) * data['SM_TT'] * data['SM_AV'] +
                      params.get('B_COST', 0) * data['SM_CO'] * data['SM_AV'] +
                      params.get('B_PURPOSE_SM', 0) * purposes)
    
    # Car utility
    utilities[:, 2] = (params.get('ASC_CAR', 0) +
                      params.get('B_TIME', 0) * data['CAR_TT'] * data['CAR_AV'] +
                      params.get('B_COST', 0) * data['CAR_CO'] * data['CAR_AV'] +
                      params.get('B_PURPOSE_CAR', 0) * purposes)
    
    return utilities

def main():
    # Load and prepare data
    loader = DatasetLoader()
    data = loader.load_dataset("swissmetro_dataset")
    data = prepare_data(data)  # Using your prepare_data function
    
    # Estimate base model
    print("\nCalibrating base model...")
    base_results, _, _ = swissmetro_mnl_model(data)
    base_params = base_results.get_beta_values()
    
    # Calibrate base model
    base_calibrator = SwissmetroCalibrator(data, utility_calculator_base)
    base_calibrated_ascs = base_calibrator.calibrate()
    
    # Update base parameters with calibrated ASCs
    base_calibrated_params = base_params.copy()
    base_calibrated_params.update(base_calibrated_ascs)
    
    # Validate base model before and after calibration
    base_validator = SwissmetroValidator(data, utility_calculator_base)
    print("\nBase Model Metrics:")
    print("Before calibration:", base_validator.calculate_metrics(base_params))
    print("After calibration:", base_validator.calculate_metrics(base_calibrated_params))
    
    # Plot base model results
    base_validator.plot_shares(base_params, "Base Model - Before Calibration")
    base_validator.plot_shares(base_calibrated_params, "Base Model - After Calibration")
    
    # Estimate purpose model
    print("\nCalibrating purpose model...")
    purpose_results, _, _ = swissmetro_mnl_purpose_model(data)
    purpose_params = purpose_results.get_beta_values()
    
    # Calibrate purpose model
    purpose_calibrator = SwissmetroCalibrator(data, utility_calculator_purpose)
    purpose_calibrated_ascs = purpose_calibrator.calibrate()
    
    # Update purpose parameters with calibrated ASCs
    purpose_calibrated_params = purpose_params.copy()
    purpose_calibrated_params.update(purpose_calibrated_ascs)
    
    # Validate purpose model before and after calibration
    purpose_validator = SwissmetroValidator(data, utility_calculator_purpose)
    print("\nPurpose Model Metrics:")
    print("Before calibration:", purpose_validator.calculate_metrics(purpose_params))
    print("After calibration:", purpose_validator.calculate_metrics(purpose_calibrated_params))
    
    # Plot purpose model results
    purpose_validator.plot_shares(purpose_params, "Purpose Model - Before Calibration")
    purpose_validator.plot_shares(purpose_calibrated_params, "Purpose Model - After Calibration")
    
    plt.show()

if __name__ == "__main__":
    main()