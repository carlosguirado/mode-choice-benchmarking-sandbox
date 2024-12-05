"""Wrapper utility for integrating Biogeme models with the benchmarker."""

from ..models.base import BaseDiscreteChoiceModel
import pandas as pd
import numpy as np

class BiogemeModelWrapper(BaseDiscreteChoiceModel):
    """Wrapper to use an existing Biogeme model with the benchmarker.
    
    This wrapper allows easy integration of Biogeme models with the benchmarking system
    by automatically extracting and calculating all required metrics.
    
    Example:
        >>> import biogeme.biogeme as bio
        >>> from biogeme import models
        >>> from mcbs.utils import BiogemeModelWrapper
        >>> 
        >>> # Your existing Biogeme model setup
        >>> biogeme = bio.BIOGEME(database, logprob)
        >>> 
        >>> # Wrap it for benchmarking
        >>> model = BiogemeModelWrapper(data, biogeme)
        >>> benchmarker = ModelBenchmarker()
        >>> results = benchmarker.run_benchmark(data, [model])
    """
    
    def __init__(self, data, biogeme_model, choice_column='CHOICE'):
        """Initialize wrapper with a Biogeme model.
        
        Args:
            data: DataFrame with choice data
            biogeme_model: Configured Biogeme model ready for estimation
            choice_column: Name of the choice column in data
        """
        super().__init__(data)
        self.biogeme_model = biogeme_model
        self.choice_column = choice_column
    
    def estimate(self):
        """Estimate model and extract metrics."""
        # Estimate the model
        self.results = self.biogeme_model.estimate()
        
        # Extract metrics from Biogeme results
        stats = self.results.getGeneralStatistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square for the null model'][0]
        self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
        
        # Calculate market shares and choice accuracy
        self._calculate_shares_and_accuracy()
        
        return self.results
    
    def _calculate_shares_and_accuracy(self):
        """Calculate market shares and choice accuracy from probabilities."""
        # Get simulated probabilities
        prob_vars = [v for v in self.results.data.simulatedValues.columns 
                    if v.startswith('Prob')]
        
        if not prob_vars:  # If no probability columns, try to simulate
            simulatedValues = self.biogeme_model.simulate(
                self.results.get_beta_values()
            )
        else:
            simulatedValues = self.results.data.simulatedValues
        
        # Calculate actual shares
        choices = self.database.data[self.choice_column]
        total = len(choices)
        self.actual_shares = choices.value_counts(normalize=True).to_dict()
        
        # Calculate predicted shares
        prob_means = simulatedValues.mean()
        self.predicted_shares = {
            i+1: prob_means[f'Prob. {i+1}'] 
            for i in range(len(prob_vars))
        }
        
        # Calculate market share accuracy
        total_abs_error = sum(
            abs(self.actual_shares.get(i, 0) - self.predicted_shares.get(i, 0))
            for i in set(self.actual_shares) | set(self.predicted_shares)
        )
        self.market_share_accuracy = 1 - (total_abs_error / 2)
        
        # Calculate choice accuracy
        predicted_choices = simulatedValues.idxmax(axis=1)
        predicted_choices = predicted_choices.map(
            lambda x: int(x.split()[-1])  # Extract choice number
        )
        
        # Create confusion matrix
        self.confusion_matrix = pd.crosstab(
            choices,
            predicted_choices,
            rownames=['Actual'],
            colnames=['Predicted']
        )
        
        # Calculate choice accuracy
        self.choice_accuracy = (choices == predicted_choices).mean()
