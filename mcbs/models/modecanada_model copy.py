"""
ModeCanada Mode Choice Model

This script implements mode choice models for the ModeCanada dataset, estimating choice between:
- train
- car
- bus
- air

Based on the same structure as the Swissmetro model.
"""

import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo
from biogeme.database import Database
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
from .base import BaseDiscreteChoiceModel
import pandas as pd
import numpy as np

class BaseModeCanadaModel(BaseDiscreteChoiceModel):
    """Base class for ModeCanada models with shared initialization."""
    
    def __init__(self, data):
        # Convert from long to wide format before creating database
        data = self._preprocess_data(data)
        super().__init__(data)
        self._initialize_variables()
    
    def _preprocess_data(self, data):
        """
        Preprocess the data:
        1. Convert alt column to numeric
        2. Convert from long to wide format
        """
        # Create a copy
        df = data.copy()
        
        # Define mode mapping (1-based for Biogeme)
        mode_mapping = {
            'train': 1,
            'car': 2,
            'bus': 3,
            'air': 4
        }
        
        # Convert alt column from strings to numeric
        df['alt_num'] = df['alt'].map(mode_mapping)
        
        # Create wide format dataframe
        cases = df['case'].unique()
        wide_data = []
        
        for case_id in cases:
            case_data = df[df['case'] == case_id]
            
            # Get case-specific variables
            row = {
                'case': case_id,
                'income': case_data['income'].iloc[0],
                'urban': case_data['urban'].iloc[0]
            }
            
            # Get the chosen alternative
            chosen = case_data[case_data['choice'] == 1]['alt_num'].iloc[0]
            row['CHOICE'] = chosen
            
            # Add mode-specific variables
            for mode, num in mode_mapping.items():
                mode_data = case_data[case_data['alt'] == mode]
                if len(mode_data) > 0:
                    row[f'{mode.upper()}_AV'] = 1
                    row[f'{mode.upper()}_COST'] = mode_data['cost'].iloc[0]
                    row[f'{mode.upper()}_TIME'] = mode_data['ivt'].iloc[0] + mode_data['ovt'].iloc[0]
                    if mode != 'car':  # Add frequency for transit modes
                        row[f'{mode.upper()}_FREQ'] = mode_data['freq'].iloc[0]
                else:
                    row[f'{mode.upper()}_AV'] = 0
                    row[f'{mode.upper()}_COST'] = 0
                    row[f'{mode.upper()}_TIME'] = 0
                    if mode != 'car':
                        row[f'{mode.upper()}_FREQ'] = 0
            
            wide_data.append(row)
        
        return pd.DataFrame(wide_data)
    
    def _initialize_variables(self):
        """Initialize all variables needed for ModeCanada models."""
        # Basic trip variables
        self.CHOICE = Variable('CHOICE')
        
        # Availability variables
        self.TRAIN_AV = Variable('TRAIN_AV')
        self.CAR_AV = Variable('CAR_AV')
        self.BUS_AV = Variable('BUS_AV')
        self.AIR_AV = Variable('AIR_AV')
        
        # Time and cost variables
        self.TRAIN_TIME = Variable('TRAIN_TIME')
        self.CAR_TIME = Variable('CAR_TIME')
        self.BUS_TIME = Variable('BUS_TIME')
        self.AIR_TIME = Variable('AIR_TIME')
        
        self.TRAIN_COST = Variable('TRAIN_COST')
        self.CAR_COST = Variable('CAR_COST')
        self.BUS_COST = Variable('BUS_COST')
        self.AIR_COST = Variable('AIR_COST')
        
        # Print initial statistics
        print("\nData Shape:", self.database.data.shape)
        print("\nMode choice distribution:")
        print(self.database.data['CHOICE'].value_counts().sort_index())

class MultinomialLogitModel(BaseModeCanadaModel):
    """Multinomial logit model implementation."""
    
    def estimate(self):
        """Estimate the multinomial logit model."""
        # Parameters to be estimated
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 1)  # Fixed to 0
        ASC_BUS = Beta('ASC_BUS', 0, None, None, 0)
        ASC_AIR = Beta('ASC_AIR', 0, None, None, 0)
        
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)

        # Utility functions
        V1 = (ASC_TRAIN + 
              B_TIME * self.TRAIN_TIME + 
              B_COST * self.TRAIN_COST)
        
        V2 = (ASC_CAR + 
              B_TIME * self.CAR_TIME + 
              B_COST * self.CAR_COST)
        
        V3 = (ASC_BUS + 
              B_TIME * self.BUS_TIME + 
              B_COST * self.BUS_COST)
        
        V4 = (ASC_AIR + 
              B_TIME * self.AIR_TIME + 
              B_COST * self.AIR_COST)

        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3, 4: V4}

        # Associate availability conditions
        av = {1: self.TRAIN_AV,
              2: self.CAR_AV,
              3: self.BUS_AV,
              4: self.AIR_AV}

        # Define and estimate the model
        logprob = models.loglogit(V, av, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "modecanada_mnl"
        
        # Disable HTML and Pickle generation
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square'][0]
        self.rho_squared_bar = stats['Rho-square-bar'][0]
        
        # Calculate value of time (in $/hour) for all modes
        betas = self.results.get_beta_values()
        self.vot = -60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        return self.results

    def get_metrics(self):
        """Get model metrics including VOT."""
        metrics = super().get_metrics()
        
        # Add model statistics and VOT metrics
        metrics.update({
            'final_ll': self.final_ll if hasattr(self, 'final_ll') else None,
            'rho_squared': self.rho_squared if hasattr(self, 'rho_squared') else None,
            'rho_squared_bar': self.rho_squared_bar if hasattr(self, 'rho_squared_bar') else None,
            'vot': self.vot if hasattr(self, 'vot') else None
        })
        
        return metrics

class NestedLogitModel(BaseModeCanadaModel):
    """Nested logit model implementation with public transport nest."""
    
    def estimate(self):
        """Estimate the nested logit model."""
        # Parameters to be estimated
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 1)  # Fixed to 0
        ASC_BUS = Beta('ASC_BUS', 0, None, None, 0)
        ASC_AIR = Beta('ASC_AIR', 0, None, None, 0)
        
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)
        
        # Nesting parameter
        MU_PT = Beta('MU_PT', 1, 1, 10, 0)

        # Utility functions
        V1 = (ASC_TRAIN + 
              B_TIME * self.TRAIN_TIME + 
              B_COST * self.TRAIN_COST)
        
        V2 = (ASC_CAR + 
              B_TIME * self.CAR_TIME + 
              B_COST * self.CAR_COST)
        
        V3 = (ASC_BUS + 
              B_TIME * self.BUS_TIME + 
              B_COST * self.BUS_COST)
        
        V4 = (ASC_AIR + 
              B_TIME * self.AIR_TIME + 
              B_COST * self.AIR_COST)

        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3, 4: V4}

        # Associate availability conditions
        av = {1: self.TRAIN_AV,
              2: self.CAR_AV,
              3: self.BUS_AV,
              4: self.AIR_AV}

        # Define nests: public transport modes in one nest
        pt_nest = OneNestForNestedLogit(
            nest_param=MU_PT,
            list_of_alternatives=[1, 3, 4],  # train, bus, air
            name='public_transport'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(pt_nest,))

        # Define and estimate the model
        logprob = models.lognested(V, av, nests, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "modecanada_nl"
        
        # Disable HTML and Pickle generation
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square'][0]
        self.rho_squared_bar = stats['Rho-square-bar'][0]
        
        # Calculate value of time (in $/hour) for all modes
        betas = self.results.get_beta_values()
        self.vot = -60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Store nests for later use
        self.nests = nests
        
        return self.results
    
    def get_metrics(self):
        """Get metrics including nest-specific information and VOT."""
        metrics = super().get_metrics()
        
        # Add correlation between alternatives in nests
        if hasattr(self, 'nests'):
            corr = self.nests.correlation(
                parameters=self.results.get_beta_values(),
                alternatives_names={1: 'Train', 2: 'Car', 3: 'Bus', 4: 'Air'}
            )
            metrics['nest_correlation'] = corr
        
        # Add model statistics and VOT metrics
        metrics.update({
            'final_ll': self.final_ll if hasattr(self, 'final_ll') else None,
            'rho_squared': self.rho_squared if hasattr(self, 'rho_squared') else None,
            'rho_squared_bar': self.rho_squared_bar if hasattr(self, 'rho_squared_bar') else None,
            'vot': self.vot if hasattr(self, 'vot') else None
        })
            
        return metrics

class NestedLogitModel2(BaseModeCanadaModel):
    """Nested logit model implementation with motorized modes nest."""
    
    def estimate(self):
        """Estimate the nested logit model."""
        # Parameters to be estimated
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 1)  # Fixed to 0
        ASC_BUS = Beta('ASC_BUS', 0, None, None, 0)
        ASC_AIR = Beta('ASC_AIR', 0, None, None, 0)
        
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)
        
        # Nesting parameter
        MU_MOTORIZED = Beta('MU_MOTORIZED', 1, 1, 10, 0)

        # Utility functions
        V1 = (ASC_TRAIN + 
              B_TIME * self.TRAIN_TIME + 
              B_COST * self.TRAIN_COST)
        
        V2 = (ASC_CAR + 
              B_TIME * self.CAR_TIME + 
              B_COST * self.CAR_COST)
        
        V3 = (ASC_BUS + 
              B_TIME * self.BUS_TIME + 
              B_COST * self.BUS_COST)
        
        V4 = (ASC_AIR + 
              B_TIME * self.AIR_TIME + 
              B_COST * self.AIR_COST)

        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3, 4: V4}

        # Associate availability conditions
        av = {1: self.TRAIN_AV,
              2: self.CAR_AV,
              3: self.BUS_AV,
              4: self.AIR_AV}

        # Define nests: all motorized modes in one nest
        motorized = OneNestForNestedLogit(
            nest_param=MU_MOTORIZED,
            list_of_alternatives=[1, 2, 3, 4],  # all modes are motorized
            name='motorized'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(motorized,))

        # Define and estimate the model
        logprob = models.lognested(V, av, nests, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "modecanada_nl2"
        
        # Disable HTML and Pickle generation
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square'][0]
        self.rho_squared_bar = stats['Rho-square-bar'][0]
        
        # Calculate value of time (in $/hour) for all modes
        betas = self.results.get_beta_values()
        self.vot = -60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Store nests for later use
        self.nests = nests
        
        return self.results
    
    def get_metrics(self):
        """Get metrics including nest-specific information and VOT."""
        metrics = super().get_metrics()
        
        # Add correlation between alternatives in nests
        if hasattr(self, 'nests'):
            corr = self.nests.correlation(
                parameters=self.results.get_beta_values(),
                alternatives_names={1: 'Train', 2: 'Car', 3: 'Bus', 4: 'Air'}
            )
            metrics['nest_correlation'] = corr
        
        # Add model statistics and VOT metrics
        metrics.update({
            'final_ll': self.final_ll if hasattr(self, 'final_ll') else None,
            'rho_squared': self.rho_squared if hasattr(self, 'rho_squared') else None,
            'rho_squared_bar': self.rho_squared_bar if hasattr(self, 'rho_squared_bar') else None,
            'vot': self.vot if hasattr(self, 'vot') else None
        })
            
        return metrics

class NestedLogitModel3(BaseModeCanadaModel):
    """Nested logit model implementation with private vs public transportation nest."""
    
    def estimate(self):
        """Estimate the nested logit model."""
        # Parameters to be estimated
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 1)  # Fixed to 0
        ASC_BUS = Beta('ASC_BUS', 0, None, None, 0)
        ASC_AIR = Beta('ASC_AIR', 0, None, None, 0)
        
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)
        
        # Nesting parameter
        MU_PUBLIC = Beta('MU_PUBLIC', 1, 1, 10, 0)

        # Utility functions
        V1 = (ASC_TRAIN + 
              B_TIME * self.TRAIN_TIME + 
              B_COST * self.TRAIN_COST)
        
        V2 = (ASC_CAR + 
              B_TIME * self.CAR_TIME + 
              B_COST * self.CAR_COST)
        
        V3 = (ASC_BUS + 
              B_TIME * self.BUS_TIME + 
              B_COST * self.BUS_COST)
        
        V4 = (ASC_AIR + 
              B_TIME * self.AIR_TIME + 
              B_COST * self.AIR_COST)

        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3, 4: V4}

        # Associate availability conditions
        av = {1: self.TRAIN_AV,
              2: self.CAR_AV,
              3: self.BUS_AV,
              4: self.AIR_AV}

        # Define nests: public transportation modes in one nest
        public = OneNestForNestedLogit(
            nest_param=MU_PUBLIC,
            list_of_alternatives=[1, 3, 4],  # train, bus, air
            name='public'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(public,))

        # Define and estimate the model
        logprob = models.lognested(V, av, nests, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "modecanada_nl3"
        
        # Disable HTML and Pickle generation
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square'][0]
        self.rho_squared_bar = stats['Rho-square-bar'][0]
        
        # Calculate value of time (in $/hour) for all modes
        betas = self.results.get_beta_values()
        self.vot = -60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Store nests for later use
        self.nests = nests
        
        return self.results
    
    def get_metrics(self):
        """Get metrics including nest-specific information and VOT."""
        metrics = super().get_metrics()
        
        # Add correlation between alternatives in nests
        if hasattr(self, 'nests'):
            corr = self.nests.correlation(
                parameters=self.results.get_beta_values(),
                alternatives_names={1: 'Train', 2: 'Car', 3: 'Bus', 4: 'Air'}
            )
            metrics['nest_correlation'] = corr
        
        # Add model statistics and VOT metrics
        metrics.update({
            'final_ll': self.final_ll if hasattr(self, 'final_ll') else None,
            'rho_squared': self.rho_squared if hasattr(self, 'rho_squared') else None,
            'rho_squared_bar': self.rho_squared_bar if hasattr(self, 'rho_squared_bar') else None,
            'vot': self.vot if hasattr(self, 'vot') else None
        })
            
        return metrics

def main():
    """Test the models with sample data."""
    from mcbs.datasets import DatasetLoader
    
    # Load data
    loader = DatasetLoader()
    data = loader.load_dataset("modecanada_dataset")
    
    # Print initial diagnostics
    print("\nInitial Data Shape:", data.shape)
    print("\nInitial Value counts for choice:")
    print(data['choice'].value_counts())
    print("\nInitial Value counts for alt:")
    print(data['alt'].value_counts())
    
    # Estimate MNL model
    print("\nEstimating Multinomial Logit Model...")
    mnl = MultinomialLogitModel(data)
    mnl_results = mnl.estimate()
    mnl_metrics = mnl.get_metrics()
    print("\nMNL Metrics:", mnl_metrics)
    
    # Estimate NL model with public transport nest
    print("\nEstimating Nested Logit Model (Public Transport)...")
    nl = NestedLogitModel(data)
    nl_results = nl.estimate()
    nl_metrics = nl.get_metrics()
    print("\nNL (Public Transport) Metrics:", nl_metrics)
    
    # Estimate NL model with motorized nest
    print("\nEstimating Nested Logit Model (Motorized)...")
    nl2 = NestedLogitModel2(data)
    nl2_results = nl2.estimate()
    nl2_metrics = nl2.get_metrics()
    print("\nNL (Motorized) Metrics:", nl2_metrics)
    
    # Estimate NL model with private vs public nest
    print("\nEstimating Nested Logit Model (Private vs Public)...")
    nl3 = NestedLogitModel3(data)
    nl3_results = nl3.estimate()
    nl3_metrics = nl3.get_metrics()
    print("\nNL (Private vs Public) Metrics:", nl3_metrics)

if __name__ == "__main__":
    main()
