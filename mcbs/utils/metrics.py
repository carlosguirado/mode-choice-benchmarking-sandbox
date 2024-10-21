# mcbs/utils/metrics.py

import numpy as np
from typing import Dict, Any

def calculate_metrics(biogeme_results: Any, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various metrics for mode choice models.
    
    Args:
        biogeme_results: Biogeme Results object
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
    
    Returns:
        Dict[str, float]: Dictionary of calculated metrics
    """
    metrics = {}
    
    # Metrics directly from Biogeme results
    metrics['log_likelihood'] = biogeme_results.data.logLike
    metrics['null_log_likelihood'] = biogeme_results.data.nullLogLike
    metrics['rho_squared'] = biogeme_results.data.rhoSquared
    metrics['adj_rho_squared'] = biogeme_results.data.adjRhoSquared
    
    # Additional metrics
    metrics['prediction_accuracy'] = calculate_prediction_accuracy(y_true, y_pred)
    metrics['avg_log_likelihood'] = calculate_avg_log_likelihood(y_true, y_pred)
    
    # Value of Time (assuming time and cost coefficients are available)
    time_coeff = biogeme_results.getBetaValues().get('time', None)
    cost_coeff = biogeme_results.getBetaValues().get('cost', None)
    if time_coeff is not None and cost_coeff is not None:
        metrics['value_of_time'] = calculate_value_of_time(time_coeff, cost_coeff)
    
    return metrics

def calculate_prediction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the prediction accuracy.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
    
    Returns:
        float: Prediction accuracy
    """
    return np.mean(y_true == np.argmax(y_pred, axis=1))

def calculate_avg_log_likelihood(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the average log-likelihood.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted probabilities
    
    Returns:
        float: Average log-likelihood
    """
    return np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))

def calculate_value_of_time(time_coeff: float, cost_coeff: float) -> float:
    """
    Calculate the Value of Time (VoT).
    
    Args:
        time_coeff (float): Time coefficient from the model
        cost_coeff (float): Cost coefficient from the model
    
    Returns:
        float: Value of Time
    """
    return abs(time_coeff / cost_coeff) * 60  # Assuming time is in minutes and cost is in currency units

# Additional helper functions can be added here as needed