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
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo, exp
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

    def calculate_choice_accuracy(self):
        """Calculate individual choice prediction accuracy and market shares."""
        if not hasattr(self, 'results'):
            raise RuntimeError("Model must be estimated before calculating accuracy")
            
        # Get utility functions for simulation
        V1 = self._get_utility_function(1)
        V2 = self._get_utility_function(2)
        V3 = self._get_utility_function(3)
        V4 = self._get_utility_function(4)
        
        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3, 4: V4}
        
        # Associate availability conditions
        av = {1: self.TRAIN_AV,
              2: self.CAR_AV,
              3: self.BUS_AV,
              4: self.AIR_AV}
        
        # Calculate choice probabilities
        prob_train = models.logit(V, av, 1)
        prob_car = models.logit(V, av, 2)
        prob_bus = models.logit(V, av, 3)
        prob_air = models.logit(V, av, 4)
        
        # Setup simulation
        simulate = {
            'Prob. train': prob_train,
            'Prob. car': prob_car,
            'Prob. bus': prob_bus,
            'Prob. air': prob_air
        }
        
        # Run simulation
        biogeme = bio.BIOGEME(self.database, simulate)
        biogeme.modelName = "choice_prediction"
        simulatedValues = biogeme.simulate(self.results.get_beta_values())
        
        # Calculate market shares
        actual_counts = self.database.data['CHOICE'].value_counts()
        total = len(self.database.data)
        self.actual_shares = {i: actual_counts.get(i, 0) / total for i in [1, 2, 3, 4]}
        
        # Calculate predicted shares (mean probability for each alternative)
        self.predicted_shares = {
            1: simulatedValues['Prob. train'].mean(),
            2: simulatedValues['Prob. car'].mean(),
            3: simulatedValues['Prob. bus'].mean(),
            4: simulatedValues['Prob. air'].mean()
        }
        
        # Calculate market share accuracy
        total_abs_error = sum(abs(self.actual_shares[i] - self.predicted_shares[i]) 
                            for i in [1, 2, 3, 4])
        self.market_share_accuracy = 1 - (total_abs_error / 2)
        
        print("\nMarket Shares:")
        print("Mode      Actual    Predicted")
        print("-" * 30)
        for i in [1, 2, 3, 4]:
            print(f"{i:4d}     {self.actual_shares[i]:.3f}     {self.predicted_shares[i]:.3f}")
        print(f"\nMarket Share Accuracy: {self.market_share_accuracy:.3f}")
        
        # Calculate choice accuracy
        prob_max = simulatedValues.idxmax(axis=1)
        prob_max = prob_max.replace({
            'Prob. train': 1, 
            'Prob. car': 2, 
            'Prob. bus': 3, 
            'Prob. air': 4
        })
        
        # Create confusion matrix
        data_sim = {
            'y_Actual': self.database.data['CHOICE'],
            'y_Predicted': prob_max
        }
        df = pd.DataFrame(data_sim, columns=['y_Actual', 'y_Predicted'])
        self.confusion_matrix = pd.crosstab(
            df['y_Actual'], 
            df['y_Predicted'], 
            rownames=['Actual'], 
            colnames=['Predicted']
        )

        ### THIS IS A QUICK FIX AND NEED TO DO PROPERLY
        self.confusion_matrix.insert(2, '3', 0)
        
        # Calculate accuracy
        self.choice_accuracy = (
            np.diagonal(self.confusion_matrix.to_numpy()).sum() / 
            self.confusion_matrix.to_numpy().sum()
        )
        
        print("\nConfusion Matrix:")
        print(self.confusion_matrix)
        print(f"\nChoice Prediction Accuracy: {self.choice_accuracy:.3f}")

    def calculate_choice_accuracy_nest(self):
        """Calculate individual choice prediction accuracy and market shares for nested logit models."""
        if not hasattr(self, 'results'):
            raise RuntimeError("Model must be estimated before calculating accuracy")
            
        # Get utility functions for simulation
        V1 = self._get_utility_function(1)
        V2 = self._get_utility_function(2)
        V3 = self._get_utility_function(3)
        V4 = self._get_utility_function(4)
        
        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3, 4: V4}
        
        # Associate availability conditions
        av = {1: self.TRAIN_AV,
              2: self.CAR_AV,
              3: self.BUS_AV,
              4: self.AIR_AV}
        
        # Calculate choice probabilities
        prob_train = models.nested(V, None, self.nests, 1)
        prob_car = models.nested(V, None, self.nests, 2)
        prob_bus = models.nested(V, None, self.nests, 3)
        prob_air = models.nested(V, None, self.nests, 4)
        
        # Setup simulation
        simulate = {
            'Prob. train': prob_train,
            'Prob. car': prob_car,
            'Prob. bus': prob_bus,
            'Prob. air': prob_air
        }
        
        # Run simulation
        biogeme = bio.BIOGEME(self.database, simulate)
        biogeme.modelName = "choice_prediction_nest"
        simulatedValues = biogeme.simulate(self.results.get_beta_values())
        
        # Calculate market shares
        actual_counts = self.database.data['CHOICE'].value_counts()
        total = len(self.database.data)
        self.actual_shares = {i: actual_counts.get(i, 0) / total for i in [1, 2, 3, 4]}
        
        # Calculate predicted shares (mean probability for each alternative)
        self.predicted_shares = {
            1: simulatedValues['Prob. train'].mean(),
            2: simulatedValues['Prob. car'].mean(),
            3: simulatedValues['Prob. bus'].mean(),
            4: simulatedValues['Prob. air'].mean()
        }
        
        # Calculate market share accuracy
        total_abs_error = sum(abs(self.actual_shares[i] - self.predicted_shares[i]) 
                            for i in [1, 2, 3, 4])
        self.market_share_accuracy = 1 - (total_abs_error / 2)
        
        print("\nMarket Shares:")
        print("Mode      Actual    Predicted")
        print("-" * 30)
        for i in [1, 2, 3, 4]:
            print(f"{i:4d}     {self.actual_shares[i]:.3f}     {self.predicted_shares[i]:.3f}")
        print(f"\nMarket Share Accuracy: {self.market_share_accuracy:.3f}")
        
        # Calculate choice accuracy
        prob_max = simulatedValues.idxmax(axis=1)
        prob_max = prob_max.replace({
            'Prob. train': 1, 
            'Prob. car': 2, 
            'Prob. bus': 3, 
            'Prob. air': 4
        })
        
        # Create confusion matrix
        data_sim = {
            'y_Actual': self.database.data['CHOICE'],
            'y_Predicted': prob_max
        }
        df = pd.DataFrame(data_sim, columns=['y_Actual', 'y_Predicted'])
        self.confusion_matrix = pd.crosstab(
            df['y_Actual'], 
            df['y_Predicted'], 
            rownames=['Actual'], 
            colnames=['Predicted']
        )
        
        # Calculate accuracy
        self.choice_accuracy = (
            np.diagonal(self.confusion_matrix.to_numpy()).sum() / 
            self.confusion_matrix.to_numpy().sum()
        )
        
        print("\nConfusion Matrix:")
        print(self.confusion_matrix)
        print(f"\nChoice Prediction Accuracy: {self.choice_accuracy:.3f}")

class MultinomialLogitModel_MC(BaseModeCanadaModel):
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
        
        # Enable HTML and Pickle generation
        biogeme.generateHtml = True
        biogeme.generatePickle = True

        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square for the null model'][0]
        self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
        
        # Calculate value of time (in $/hour) for all modes
        betas = self.results.get_beta_values()
        self.vot = 60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Calculate choice accuracy and market shares
        self.calculate_choice_accuracy()
        
        return self.results

    def _calculate_utilities(self, betas):
        """Calculate utilities for each alternative using MNL parameters."""
        v1 = (betas['ASC_TRAIN'] + 
              betas['B_TIME'] * self.database.data['TRAIN_TIME'] + 
              betas['B_COST'] * self.database.data['TRAIN_COST'])
        
        v2 = (betas['ASC_CAR'] + #fixed to 0
              betas['B_TIME'] * self.database.data['CAR_TIME'] + 
              betas['B_COST'] * self.database.data['CAR_COST'])
        
        v3 = (betas['ASC_BUS'] + 
              betas['B_TIME'] * self.database.data['BUS_TIME'] + 
              betas['B_COST'] * self.database.data['BUS_COST'])
        
        v4 = (betas['ASC_AIR'] + 
              betas['B_TIME'] * self.database.data['AIR_TIME'] + 
              betas['B_COST'] * self.database.data['AIR_COST'])
        
        utilities = np.column_stack([v1, v2, v3, v4])
        
        return utilities

    def _get_utility_function(self, alternative):
        """Get utility function for a specific alternative."""
        betas = self.results.get_beta_values()
        
        if alternative == 1:  # Train 
            return (betas['ASC_TRAIN'] + 
                   betas['B_TIME'] * self.TRAIN_TIME + 
                   betas['B_COST'] * self.TRAIN_COST)
        elif alternative == 2:  # Car 
            return (0 +  #betas['ASC_CAR'] + #fixed to 0
                   betas['B_TIME'] * self.CAR_TIME + 
                   betas['B_COST'] * self.CAR_COST)
        elif alternative == 3:  # Bus 
            return (betas['ASC_BUS'] + 
                   betas['B_TIME'] * self.BUS_TIME + 
                   betas['B_COST'] * self.BUS_COST)
        elif alternative == 4:  # Air
            return (betas['ASC_AIR'] + 
                   betas['B_TIME'] * self.AIR_TIME + 
                   betas['B_COST'] * self.AIR_COST)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")
    
    def get_metrics(self):
        """Get metrics including nest-specific information and VOT."""
        metrics = super().get_metrics()
        
        # Add model statistics and VOT metrics
        metrics.update({
            'final_ll': self.final_ll if hasattr(self, 'final_ll') else None,
            'rho_squared': self.rho_squared if hasattr(self, 'rho_squared') else None,
            'rho_squared_bar': self.rho_squared_bar if hasattr(self, 'rho_squared_bar') else None,
            'vot': self.vot if hasattr(self, 'vot') else None,
            'market_share_accuracy': self.market_share_accuracy if hasattr(self, 'market_share_accuracy') else None,
            'choice_accuracy': self.choice_accuracy if hasattr(self, 'choice_accuracy') else None,
            'actual_shares': self.actual_shares if hasattr(self, 'actual_shares') else None,
            'predicted_shares': self.predicted_shares if hasattr(self, 'predicted_shares') else None,
            'confusion_matrix': self.confusion_matrix.to_dict() if hasattr(self, 'confusion_matrix') else None
        })
            
        return metrics

class NestedLogitModel3_MC(BaseModeCanadaModel):
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
            list_of_alternatives=[1, 3, 4],  # train, bus
            name='public'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(public,))

        # Define and estimate the model
        logprob = models.lognested(V, av, nests, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "modecanada_nl3"
        
        # Enable HTML and Pickle generation
        biogeme.generateHtml = True
        biogeme.generatePickle = True
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square for the null model'][0]
        self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
        
        # Calculate value of time (in $/hour) for all modes
        betas = self.results.get_beta_values()
        self.vot = 60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Store nests for later use
        self.nests = nests
        
        # Calculate choice accuracy and market shares
        self.calculate_choice_accuracy_nest()
        
        return self.results

    def _calculate_utilities(self, betas):
        """Calculate utilities for each alternative using NL parameters."""
        v1 = (betas['ASC_TRAIN'] + 
              betas['B_TIME'] * self.database.data['TRAIN_TIME'] + 
              betas['B_COST'] * self.database.data['TRAIN_COST']) 
        
        v2 = (0 + betas['ASC_CAR'] + #fixed to 0
              betas['B_TIME'] * self.database.data['CAR_TIME'] + 
              betas['B_COST'] * self.database.data['CAR_COST']) 
        
        v3 = (betas['ASC_BUS'] + 
              betas['B_TIME'] * self.database.data['BUS_TIME'] + 
              betas['B_COST'] * self.database.data['BUS_COST']) 
        
        v4 = (betas['ASC_AIR'] + 
              betas['B_TIME'] * self.database.data['AIR_TIME'] + 
              betas['B_COST'] * self.database.data['AIR_COST']) 
        
        utilities = np.column_stack([v1, v2, v3, v4])
        
        # Apply nesting structure
        mu = betas['MU_PUBLIC']
        #utilities[:, [0, 2, 3]] = utilities[:, [0, 2, 3]] / mu  # Scale utilities in public transport nest
        
        return utilities

    def _get_utility_function(self, alternative):
        """Get utility function for a specific alternative."""
        betas = self.results.get_beta_values()
        mu = betas['MU_PUBLIC']
        
        if alternative == 1:  # Train (in nest)
            return (betas['ASC_TRAIN'] + 
                   betas['B_TIME'] * self.TRAIN_TIME + 
                   betas['B_COST'] * self.TRAIN_COST) 
        elif alternative == 2:  # Car (not in nest)
            return (0 + #betas['ASC_CAR'] + #fixed to 0
                   betas['B_TIME'] * self.CAR_TIME + 
                   betas['B_COST'] * self.CAR_COST)
        elif alternative == 3:  # Bus (in nest)
            return (betas['ASC_BUS'] + 
                   betas['B_TIME'] * self.BUS_TIME + 
                   betas['B_COST'] * self.BUS_COST) 
        elif alternative == 4:  # Air (in nest)
            return (betas['ASC_AIR'] + 
                   betas['B_TIME'] * self.AIR_TIME + 
                   betas['B_COST'] * self.AIR_COST) 
        else:
            raise ValueError(f"Invalid alternative: {alternative}")
    
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
            'vot': self.vot if hasattr(self, 'vot') else None,
            'market_share_accuracy': self.market_share_accuracy if hasattr(self, 'market_share_accuracy') else None,
            'choice_accuracy': self.choice_accuracy if hasattr(self, 'choice_accuracy') else None,
            'actual_shares': self.actual_shares if hasattr(self, 'actual_shares') else None,
            'predicted_shares': self.predicted_shares if hasattr(self, 'predicted_shares') else None,
            'confusion_matrix': self.confusion_matrix.to_dict() if hasattr(self, 'confusion_matrix') else None
        })
            
        return metrics

class MixedLogitModel_MC(BaseModeCanadaModel):
    """Mixed logit model implementation with random coefficients."""
    
    def estimate(self):
        """Estimate the mixed logit model."""
        # Parameters to be estimated
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 1)  # Fixed to 0
        ASC_BUS = Beta('ASC_BUS', 0, None, None, 0)
        ASC_AIR = Beta('ASC_AIR', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)

       # Define random parameter for time (lognormal to ensure negative)
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)  # Spread parameter
        B_TIME_RND = -exp(B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL'))
        #B_TIME_RND = B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL')

        # Utility functions with random coefficient
        V1 = (ASC_TRAIN + 
              B_TIME_RND * self.TRAIN_TIME + 
              B_COST * self.TRAIN_COST)
        
        V2 = (ASC_CAR + 
              B_TIME_RND * self.CAR_TIME + 
              B_COST * self.CAR_COST)
        
        V3 = (ASC_BUS + 
              B_TIME_RND * self.BUS_TIME + 
              B_COST * self.BUS_COST)
        
        V4 = (ASC_AIR + 
              B_TIME_RND * self.AIR_TIME + 
              B_COST * self.AIR_COST)

        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3, 4: V4}

        # Associate availability conditions
        av = {1: self.TRAIN_AV,
              2: self.CAR_AV,
              3: self.BUS_AV,
              4: self.AIR_AV}

        # Conditional on b_time_rnd, we have a logit model (kernel)
        prob = models.logit(V, av, self.CHOICE)

        # Integrate over b_time_rnd using Monte-Carlo
        logprob = log(MonteCarlo(prob))

        # Create and estimate the model
        biogeme = bio.BIOGEME(
            self.database, 
            logprob, 
            number_of_draws=100,  # You can adjust this
            seed=1223
        )
        biogeme.modelName = "modecanada_mixed"
        
        # Enable HTML and Pickle generation
        biogeme.generateHtml = True
        biogeme.generatePickle = True
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square for the null model'][0]
        self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
        
        # Calculate value of time (in $/hour) for all modes
        betas = self.results.get_beta_values()
        self.vot = 60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Calculate choice accuracy and market shares
        self.calculate_choice_accuracy()
        
        return self.results

    def _calculate_utilities(self, betas):
        """Calculate utilities for each alternative using Mixed Logit parameters."""
        # For mixed logit, we use the mean of the random parameter
        v1 = (betas['ASC_TRAIN'] + 
              betas['B_TIME'] * self.database.data['TRAIN_TIME'] + 
              betas['B_COST'] * self.database.data['TRAIN_COST'])
        
        v2 = (0 + #betas['ASC_CAR'] + #fixed to 0
              betas['B_TIME'] * self.database.data['CAR_TIME'] + 
              betas['B_COST'] * self.database.data['CAR_COST'])
        
        v3 = (betas['ASC_BUS'] + 
              betas['B_TIME'] * self.database.data['BUS_TIME'] + 
              betas['B_COST'] * self.database.data['BUS_COST'])
        
        v4 = (betas['ASC_AIR'] + 
              betas['B_TIME'] * self.database.data['AIR_TIME'] + 
              betas['B_COST'] * self.database.data['AIR_COST'])
        
        return np.column_stack([v1, v2, v3, v4])

    def _get_utility_function(self, alternative):
        """Get utility function for a specific alternative."""
        betas = self.results.get_beta_values()
        # For prediction, use mean of random parameter
        if alternative == 1:  # Train
            return (betas['ASC_TRAIN'] + 
                   betas['B_TIME'] * self.TRAIN_TIME + 
                   betas['B_COST'] * self.TRAIN_COST)
        elif alternative == 2:  # Car
            return (0 + #betas['ASC_CAR'] + # fixed to 0
                   betas['B_TIME'] * self.CAR_TIME + 
                   betas['B_COST'] * self.CAR_COST)
        elif alternative == 3:  # Bus
            return (betas['ASC_BUS'] + 
                   betas['B_TIME'] * self.BUS_TIME + 
                   betas['B_COST'] * self.BUS_COST)
        elif alternative == 4:  # Air
            return (betas['ASC_AIR'] + 
                   betas['B_TIME'] * self.AIR_TIME + 
                   betas['B_COST'] * self.AIR_COST)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")
    
    def get_metrics(self):
        """Get metrics including random parameter information and VOT."""
        metrics = super().get_metrics()
        
        # Add random parameter statistics
        if self.results is not None:
            beta_values = self.results.get_beta_values()
            if 'B_TIME' in beta_values and 'B_TIME_S' in beta_values:
                metrics['time_mean'] = beta_values['B_TIME']
                metrics['time_std'] = beta_values['B_TIME_S']
        
        # Add model statistics and VOT metrics
        metrics.update({
            'final_ll': self.final_ll if hasattr(self, 'final_ll') else None,
            'rho_squared': self.rho_squared if hasattr(self, 'rho_squared') else None,
            'rho_squared_bar': self.rho_squared_bar if hasattr(self, 'rho_squared_bar') else None,
            'vot': self.vot if hasattr(self, 'vot') else None,
            'market_share_accuracy': self.market_share_accuracy if hasattr(self, 'market_share_accuracy') else None,
            'choice_accuracy': self.choice_accuracy if hasattr(self, 'choice_accuracy') else None,
            'actual_shares': self.actual_shares if hasattr(self, 'actual_shares') else None,
            'predicted_shares': self.predicted_shares if hasattr(self, 'predicted_shares') else None,
            'confusion_matrix': self.confusion_matrix.to_dict() if hasattr(self, 'confusion_matrix') else None
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
    mnl = MultinomialLogitModel_MC(data)
    mnl_results = mnl.estimate()
    mnl_metrics = mnl.get_metrics()
    print("\nMNL Metrics:", mnl_metrics)
    
    # Estimate NL model with private vs public nest
    print("\nEstimating Nested Logit Model (Private vs Public)...")
    nl3 = NestedLogitModel3_MC(data)
    nl3_results = nl3.estimate()
    nl3_metrics = nl3.get_metrics()
    print("\nNL (Private vs Public) Metrics:", nl3_metrics)
    
    # Estimate Mixed Logit model
    print("\nEstimating Mixed Logit Model...")
    ml = MixedLogitModel_MC(data)
    ml_results = ml.estimate()
    ml_metrics = ml.get_metrics()
    print("\nMixL Metrics:", ml_metrics)

if __name__ == "__main__":
    main()
