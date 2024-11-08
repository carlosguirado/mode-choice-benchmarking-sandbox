# mcbs/utils/individual_parameters.py

import numpy as np
from typing import Dict, Any, List, Tuple
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Expression, Beta

class IndividualParameterCalculator:
    """Calculates individual-specific parameters from mixture model results"""
    
    def __init__(self, 
                 model_results: bio.ModelResults,
                 data: pd.DataFrame,
                 choice_col: str,
                 parameter_name: str):
        """
        Initialize calculator with model results and data
        
        Args:
            model_results: Estimated biogeme model results
            data: DataFrame containing choice and attribute data
            choice_col: Name of column containing choices
            parameter_name: Name of parameter to calculate individual values for
        """
        self.model_results = model_results
        self.database = db.Database("indiv_params", data)
        self.choice_col = choice_col
        self.parameter_name = parameter_name
        
    def calculate_individual_parameters(self, 
                                     n_draws: int = 1000,
                                     seed: int = 42) -> pd.Series:
        """
        Calculate individual-specific parameters using Bayes theorem
        
        Args:
            n_draws: Number of draws to use in Monte Carlo simulation
            seed: Random seed for reproducibility
            
        Returns:
            Series containing individual parameter values indexed by person ID
        """
        # Set random seed
        np.random.seed(seed)
        
        # Get parameter distribution from model results
        param_mean = self.model_results.getBetaValues()[self.parameter_name]
        param_std = self.model_results.getStdErrValues()[self.parameter_name]
        
        # Generate random draws
        random_draws = np.random.normal(param_mean, param_std, n_draws)
        
        # Calculate probabilities for each draw
        probs = []
        for beta_r in random_draws:
            # Update model formulation with this draw
            prob = self._calculate_choice_probability(beta_r)
            probs.append(prob)
            
        probs = np.array(probs)
        
        # Calculate individual parameters using Bayes formula
        beta_weights = random_draws[:, np.newaxis] * probs
        individual_betas = np.sum(beta_weights, axis=0) / np.sum(probs, axis=0)
        
        return pd.Series(individual_betas, 
                        index=self.database.data.index,
                        name=f"{self.parameter_name}_individual")
    
    def _calculate_choice_probability(self, beta_value: float) -> np.ndarray:
        """
        Calculate choice probability for a given parameter value
        
        Args:
            beta_value: Parameter value to evaluate probability for
            
        Returns:
            Array of probabilities
        """
        # This needs to be implemented specifically for each choice model
        # Here is a placeholder implementation
        raise NotImplementedError(
            "Implement specific choice probability calculation")

class SwissmetroIndividualCalculator(IndividualParameterCalculator):
    """Individual parameter calculator specifically for Swissmetro"""
    
    def _calculate_choice_probability(self, beta_value: float) -> np.ndarray:
        """
        Calculate Swissmetro choice probabilities for a parameter value
        
        Args:
            beta_value: Parameter value to evaluate
            
        Returns:
            Array of choice probabilities
        """
        # Recreate utility functions with given beta value
        # Note: This is simplified - would need actual specification
        TRAIN_ASC = Beta('TRAIN_ASC', 0, None, None, 0)
        SM_ASC = Beta('SM_ASC', 0, None, None, 0) 
        BETA_TIME = Beta('BETA_TIME', beta_value, None, None, 0)
        
        # Calculate utilities
        V_TRAIN = TRAIN_ASC + BETA_TIME * self.database.data['TRAIN_TT'] 
        V_SM = SM_ASC + BETA_TIME * self.database.data['SM_TT']
        V_CAR = BETA_TIME * self.database.data['CAR_TT']
        
        # Calculate exponentials
        e_train = np.exp(V_TRAIN)
        e_sm = np.exp(V_SM)  
        e_car = np.exp(V_CAR)
        
        # Get denominator
        denom = e_train + e_sm + e_car
        
        # Get probabilities for chosen alternative
        chosen = self.database.data[self.choice_col].values
        probs = np.where(chosen == 1, e_car/denom,
                np.where(chosen == 2, e_train/denom, e_sm/denom))
        
        return probs

def plot_individual_parameters(param_values: pd.Series,
                             chosen_alt: pd.Series,
                             title: str = None):
    """
    Plot histogram of individual parameters colored by chosen alternative
    
    Args:
        param_values: Series of individual parameter values
        chosen_alt: Series of chosen alternatives
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,6))
    
    # Plot histogram for each alternative
    for alt in sorted(chosen_alt.unique()):
        mask = chosen_alt == alt
        plt.hist(param_values[mask], 
                alpha=0.5,
                label=f'Alternative {alt}',
                bins=30)
    
    plt.xlabel('Parameter Value')
    plt.ylabel('Frequency')
    plt.title(title or 'Distribution of Individual Parameters')
    plt.legend()
    plt.grid(True)
    plt.show()