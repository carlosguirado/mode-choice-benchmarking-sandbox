"""
Swissmetro Mode Choice Model

This script replicates the data preparation and model estimation from the Biogeme tutorial.
The model estimates the choice between train, car and Swissmetro for intercity trips
in Switzerland.

Based on: Michel Bierlaire's Biogeme tutorial
"""
import pandas as pd
import numpy as np
import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo
from biogeme.database import Database
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
from .base import BaseDiscreteChoiceModel
from biogeme.data.optima import read_data, normalized_weight
#from scenarios import scenario

class BaseSwissmetroModel(BaseDiscreteChoiceModel):
    """Base class for Swissmetro models with shared initialization."""
    
    def __init__(self, data):
        # Encode categorical variables before creating database
        data = self._encode_categorical_variables(data)
        super().__init__(data)
        self._initialize_variables()
        
    def _encode_categorical_variables(self, df):
        """
        Encode categorical variables according to Swissmetro dataset specifications.
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing the categorical columns
        
        Returns:
        pandas.DataFrame: DataFrame with encoded categorical variables
        """
        # Create a copy to avoid modifying the original
        df_encoded = df.copy()
        
        # Print data statistics before encoding
        print("\nBefore encoding:")
        print("Purpose values:", df_encoded['PURPOSE'].value_counts())
        print("Choice values:", df_encoded['CHOICE'].value_counts())
        
        return df_encoded
        
    def _initialize_variables(self):
        """Initialize all variables needed for Swissmetro models."""
        # Define all variables as in Biogeme tutorial
        self.GROUP = Variable('GROUP')
        self.SURVEY = Variable('SURVEY')
        self.SP = Variable('SP')
        self.ID = Variable('ID')
        self.PURPOSE = Variable('PURPOSE')
        self.FIRST = Variable('FIRST')
        self.TICKET = Variable('TICKET')
        self.WHO = Variable('WHO')
        self.LUGGAGE = Variable('LUGGAGE')
        self.AGE = Variable('AGE')
        self.MALE = Variable('MALE')
        self.INCOME = Variable('INCOME')
        self.GA = Variable('GA')
        self.ORIGIN = Variable('ORIGIN')
        self.DEST = Variable('DEST')
        self.TRAIN_AV = Variable('TRAIN_AV')
        self.CAR_AV = Variable('CAR_AV')
        self.SM_AV = Variable('SM_AV')
        self.TRAIN_TT = Variable('TRAIN_TT')
        self.TRAIN_CO = Variable('TRAIN_CO')
        self.TRAIN_HE = Variable('TRAIN_HE')
        self.SM_TT = Variable('SM_TT')
        self.SM_CO = Variable('SM_CO')
        self.SM_HE = Variable('SM_HE')
        self.SM_SEATS = Variable('SM_SEATS')
        self.CAR_TT = Variable('CAR_TT')
        self.CAR_CO = Variable('CAR_CO')
        self.CHOICE = Variable('CHOICE')

        # Print data statistics before filtering
        print("\nBefore filtering:")
        print("Purpose values:", self.database.data['PURPOSE'].value_counts())
        print("Choice values:", self.database.data['CHOICE'].value_counts())
        
        # Remove observations using Biogeme way
        exclude = ((self.PURPOSE != 1) * (self.PURPOSE != 3) + (self.CHOICE == 0)) > 0
        self.database.remove(exclude)

        # Print data statistics after filtering
        print("\nAfter filtering:")
        print("Purpose values:", self.database.data['PURPOSE'].value_counts())
        print("Choice values:", self.database.data['CHOICE'].value_counts())
        print("Train availability:", self.database.data['TRAIN_AV'].value_counts())
        print("Car availability:", self.database.data['CAR_AV'].value_counts())
        print("SM availability:", self.database.data['SM_AV'].value_counts())

        # Definition of new variables
        def define_if_not_exists(name, expression):
            if name not in self.database.data.columns:
                self.database.define_variable(name, expression)
            return Variable(name)

        # Define derived variables
        self.SM_COST = define_if_not_exists('SM_COST', 
                                           self.SM_CO * (self.GA == 0))
        self.TRAIN_COST = define_if_not_exists('TRAIN_COST', 
                                              self.TRAIN_CO * (self.GA == 0))
        self.CAR_AV_SP = define_if_not_exists('CAR_AV_SP', 
                                             self.CAR_AV * (self.SP != 0))
        self.TRAIN_AV_SP = define_if_not_exists('TRAIN_AV_SP', 
                                               self.TRAIN_AV * (self.SP != 0))
        self.TRAIN_TT_SCALED = define_if_not_exists('TRAIN_TT_SCALED', 
                                                   self.TRAIN_TT / 100)
        self.TRAIN_COST_SCALED = define_if_not_exists('TRAIN_COST_SCALED', 
                                                     self.TRAIN_COST / 100)
        self.SM_TT_SCALED = define_if_not_exists('SM_TT_SCALED', 
                                                self.SM_TT / 100)
        self.SM_COST_SCALED = define_if_not_exists('SM_COST_SCALED', 
                                                  self.SM_COST / 100)
        self.CAR_TT_SCALED = define_if_not_exists('CAR_TT_SCALED', 
                                                 self.CAR_TT / 100)
        self.CAR_CO_SCALED = define_if_not_exists('CAR_CO_SCALED', 
                                                 self.CAR_CO / 100)

    # def calculate_market_shares(self):
    #     """Calculate actual and predicted market shares."""
    #     if not hasattr(self, 'results'):
    #         raise RuntimeError("Model must be estimated before calculating shares")
            
    #     # Calculate actual shares
    #     actual_counts = self.database.data['CHOICE'].value_counts()
    #     total = len(self.database.data)
    #     self.actual_shares = {i: actual_counts.get(i, 0) / total for i in [1, 2, 3]}
        
    #     # Get model parameters
    #     betas = self.results.get_beta_values()
        
    #     # Calculate utilities using estimated parameters
    #     utilities = self._calculate_utilities(betas)
        
    #     # Calculate probabilities
    #     exp_utilities = np.exp(utilities)
    #     sum_exp_utilities = exp_utilities.sum(axis=1)
    #     probabilities = exp_utilities / sum_exp_utilities[:, np.newaxis]
        
    #     # Calculate predicted shares
    #     self.predicted_shares = {i+1: probabilities[:, i].mean() for i in range(3)}
        
    #     # Calculate market share accuracy using absolute error
    #     total_abs_error = sum(abs(self.actual_shares[i] - self.predicted_shares[i]) 
    #                         for i in [1, 2, 3])
    #     self.market_share_accuracy = 1 - (total_abs_error / 2)  # Divide by 2 since errors are double counted
        
    #     print("\nMarket Shares:")
    #     print("Mode      Actual    Predicted")
    #     print("-" * 30)
    #     for i in [1, 2, 3]:  # 1: Train, 2: SM, 3: Car
    #         print(f"{i:4d}     {self.actual_shares[i]:.3f}     {self.predicted_shares[i]:.3f}")
    #     print(f"\nMarket Share Accuracy: {self.market_share_accuracy:.3f}")

    def calculate_choice_accuracy(self):
        """Calculate individual choice prediction accuracy and market shares."""
        if not hasattr(self, 'results'):
            raise RuntimeError("Model must be estimated before calculating accuracy")
            
        # Get utility functions for simulation
        V1 = self._get_utility_function(1)
        V2 = self._get_utility_function(2)
        V3 = self._get_utility_function(3)
        
        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3}
        
        # Associate availability conditions
        av = {1: self.TRAIN_AV_SP, 2: self.SM_AV, 3: self.CAR_AV_SP}
        
        # Calculate choice probabilities
        prob_train = models.logit(V, av, 1)
        prob_SM = models.logit(V, av, 2)
        prob_car = models.logit(V, av, 3)
        
        # Setup simulation
        simulate = {
            'Prob. train': prob_train,
            'Prob. SM': prob_SM,
            'Prob. car': prob_car
        }
        
        # Run simulation
        biogeme = bio.BIOGEME(self.database, simulate)
        biogeme.modelName = "choice_prediction"
        simulatedValues = biogeme.simulate(self.results.get_beta_values())
        
        # Calculate market shares
        actual_counts = self.database.data['CHOICE'].value_counts()
        total = len(self.database.data)
        self.actual_shares = {i: actual_counts.get(i, 0) / total for i in [1, 2, 3]}
        
        # Calculate predicted shares (mean probability for each alternative)
        self.predicted_shares = {
            1: simulatedValues['Prob. train'].mean(),
            2: simulatedValues['Prob. SM'].mean(),
            3: simulatedValues['Prob. car'].mean()
        }
        
        # Calculate market share accuracy
        total_abs_error = sum(abs(self.actual_shares[i] - self.predicted_shares[i]) 
                            for i in [1, 2, 3])
        self.market_share_accuracy = 1 - (total_abs_error / 2)  # Divide by 2 since errors are double counted
        
        print("\nMarket Shares:")
        print("Mode      Actual    Predicted")
        print("-" * 30)
        for i in [1, 2, 3]:  # 1: Train, 2: SM, 3: Car
            print(f"{i:4d}     {self.actual_shares[i]:.3f}     {self.predicted_shares[i]:.3f}")
        print(f"\nMarket Share Accuracy: {self.market_share_accuracy:.3f}")
        
        # Calculate choice accuracy
        prob_max = simulatedValues.idxmax(axis=1)
        prob_max = prob_max.replace({'Prob. train': 1, 'Prob. SM': 2, 'Prob. car': 3})
        
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

    def calculate_choice_accuracy_nest(self):
        """Calculate individual choice prediction accuracy and market shares for nested logit models."""
        if not hasattr(self, 'results'):
            raise RuntimeError("Model must be estimated before calculating accuracy")
            
        # Get utility functions for simulation
        V1 = self._get_utility_function(1)
        V2 = self._get_utility_function(2)
        V3 = self._get_utility_function(3)
        
        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3}
        
        # Associate availability conditions
        av = {1: self.TRAIN_AV_SP, 2: self.SM_AV, 3: self.CAR_AV_SP}

        #V, self.nests, _, _ = scenario()
        
        # Calculate choice probabilities (CHECK THIS PART!)
        prob_train = models.nested(V, None, self.nests, 1)
        prob_SM = models.nested(V, None, self.nests, 3)
        prob_car = models.nested(V, None, self.nests, 2)
        
        # Setup simulation
        simulate = {
            #'weight': 1,
            'Prob. train': prob_train,
            'Prob. SM': prob_SM,
            'Prob. car': prob_car
        }
        
        # Run simulation
        biogeme = bio.BIOGEME(self.database, simulate)
        biogeme.modelName = "choice_prediction_nest"
        simulatedValues = biogeme.simulate(self.results.get_beta_values())
        
        # Calculate market shares
        actual_counts = self.database.data['CHOICE'].value_counts()
        total = len(self.database.data)
        self.actual_shares = {i: actual_counts.get(i, 0) / total for i in [1, 2, 3]}
        
        # Calculate predicted shares (mean probability for each alternative)
        self.predicted_shares = {
            1: simulatedValues['Prob. train'].mean(),
            2: simulatedValues['Prob. SM'].mean(),
            3: simulatedValues['Prob. car'].mean()
        }
        
        # Calculate market share accuracy
        total_abs_error = sum(abs(self.actual_shares[i] - self.predicted_shares[i]) 
                            for i in [1, 2, 3])
        self.market_share_accuracy = 1 - (total_abs_error / 2)  # Divide by 2 since errors are double counted
        
        print("\nMarket Shares:")
        print("Mode      Actual    Predicted")
        print("-" * 30)
        for i in [1, 2, 3]:  # 1: Train, 2: SM, 3: Car
            print(f"{i:4d}     {self.actual_shares[i]:.3f}     {self.predicted_shares[i]:.3f}")
        print(f"\nMarket Share Accuracy: {self.market_share_accuracy:.3f}")
        
        # Calculate choice accuracy
        prob_max = simulatedValues.idxmax(axis=1)
        prob_max = prob_max.replace({'Prob. train': 1, 'Prob. SM': 2, 'Prob. car': 3})
        
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

    def _calculate_utilities(self, betas):
            """Calculate utilities for each alternative using estimated parameters."""
            raise NotImplementedError("Subclasses must implement _calculate_utilities")

    def _get_utility_function(self, alternative):
            """Get utility function for a specific alternative."""
            raise NotImplementedError("Subclasses must implement _get_utility_function")

class MultinomialLogitModel_SM(BaseSwissmetroModel):
    """Multinomial logit model implementation."""
    
    def estimate(self):
        """Estimate the multinomial logit model."""
        # Parameters to be estimated
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_SM = Beta('ASC_SM', 0, None, None, 1)  # Fixed parameter
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)

        # Definition of the utility functions
        V1 = ASC_TRAIN + B_TIME * self.TRAIN_TT_SCALED + B_COST * self.TRAIN_COST_SCALED
        V2 = ASC_SM + B_TIME * self.SM_TT_SCALED + B_COST * self.SM_COST_SCALED
        V3 = ASC_CAR + B_TIME * self.CAR_TT_SCALED + B_COST * self.CAR_CO_SCALED

        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3}

        # Associate availability conditions
        av = {1: self.TRAIN_AV_SP, 2: self.SM_AV, 3: self.CAR_AV_SP}

        # Define and estimate the model
        logprob = models.loglogit(V, av, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "mnl_model"
        
        # Disable HTML and Pickle generation
        biogeme.generateHtml = True
        biogeme.generatePickle = True
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        
        # Store statistics using correct key names
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square for the null model'][0]
        self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
        
        # Calculate value of time (in CHF/hour)
        betas = self.results.get_beta_values()
        print("THESE ARE THE BETAS:")
        print(betas)
        self.vot = 60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Calculate market shares and accuracies
        #self.calculate_market_shares()
        self.calculate_choice_accuracy()
        
        return self.results

    def _calculate_utilities(self, betas):
        """Calculate utilities for each alternative using MNL parameters."""
        # ASC_SM is fixed to 1 for identification
        v1 = (betas['ASC_TRAIN'] + 
              betas['B_TIME'] * self.database.data['TRAIN_TT_SCALED'] + 
              betas['B_COST'] * self.database.data['TRAIN_COST_SCALED'])
        
        v2 = (#betas['ASC_SM'] +  # Fixed ASC_SM value
              betas['B_TIME'] * self.database.data['SM_TT_SCALED'] + 
              betas['B_COST'] * self.database.data['SM_COST_SCALED'])
        
        v3 = (betas['ASC_CAR'] + 
              betas['B_TIME'] * self.database.data['CAR_TT_SCALED'] + 
              betas['B_COST'] * self.database.data['CAR_CO_SCALED'])
        
        return np.column_stack([v1, v2, v3])

    def _get_utility_function(self, alternative):
        """Get utility function for a specific alternative."""
        betas = self.results.get_beta_values()
        if alternative == 1:  # Train
            return (betas['ASC_TRAIN'] + 
                   betas['B_TIME'] * self.TRAIN_TT_SCALED + 
                   betas['B_COST'] * self.TRAIN_COST_SCALED)
        elif alternative == 2:  # Swissmetro
            return (#betas['ASC_SM'] +  # Fixed ASC_SM value
                   betas['B_TIME'] * self.SM_TT_SCALED + 
                   betas['B_COST'] * self.SM_COST_SCALED)
        elif alternative == 3:  # Car
            return (betas['ASC_CAR'] + 
                   betas['B_TIME'] * self.CAR_TT_SCALED + 
                   betas['B_COST'] * self.CAR_CO_SCALED)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")

    def get_metrics(self):
        """Get model metrics including VOT."""
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

class NestedLogitModel_SM(BaseSwissmetroModel):
    """Nested logit model implementation."""
    
    def estimate(self):
        """Estimate the nested logit model."""
        # Parameters to be estimated
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_SM = Beta('ASC_SM', 0, None, None, 1)
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)
        MU = Beta('MU', 1, 1, 10, 0)  # Nesting parameter

        # Definition of the utility functions
        V1 = ASC_TRAIN + B_TIME * self.TRAIN_TT_SCALED + B_COST * self.TRAIN_COST_SCALED
        V2 = ASC_SM + B_TIME * self.SM_TT_SCALED + B_COST * self.SM_COST_SCALED
        V3 = ASC_CAR + B_TIME * self.CAR_TT_SCALED + B_COST * self.CAR_CO_SCALED

        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3}

        # Associate availability conditions
        av = {1: self.TRAIN_AV_SP, 2: self.SM_AV, 3: self.CAR_AV_SP}

        # Define nests - group existing modes (train and car)
        existing = OneNestForNestedLogit(
            nest_param=MU,
            list_of_alternatives=[1, 3],
            name='existing'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(existing,))

        # Define and estimate the model
        logprob = models.lognested(V, av, nests, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "nested_logit_model"
        
        # Disable HTML and Pickle generation
        biogeme.generateHtml = True
        biogeme.generatePickle = True
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        
        # Store statistics using correct key names
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square for the null model'][0]
        self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
        
        # Calculate value of time (in CHF/hour)
        betas = self.results.get_beta_values()
        self.vot = 60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Store nests for later use
        self.nests = nests
        
        # Calculate market shares and accuracies
        #self.calculate_market_shares()
        self.calculate_choice_accuracy_nest()
        
        return self.results

    def _calculate_utilities(self, betas):
        """Calculate utilities for each alternative using NL parameters."""
        v1 = (betas['ASC_TRAIN'] + 
              betas['B_TIME'] * self.database.data['TRAIN_TT_SCALED'] + 
              betas['B_COST'] * self.database.data['TRAIN_COST_SCALED'])
        
        v2 = (#1.0 +  # Fixed ASC_SM value
              betas['B_TIME'] * self.database.data['SM_TT_SCALED'] + 
              betas['B_COST'] * self.database.data['SM_COST_SCALED'])
        
        v3 = (betas['ASC_CAR'] + 
              betas['B_TIME'] * self.database.data['CAR_TT_SCALED'] + 
              betas['B_COST'] * self.database.data['CAR_CO_SCALED'])
        
        utilities = np.column_stack([v1, v2, v3])
        
        # Apply nesting structure
        mu = betas['MU']
        utilities[:, [0, 2]] = utilities[:, [0, 2]] / mu  # Scale utilities in the nest
        
        return utilities

    def _get_utility_function(self, alternative):
        """Get utility function for a specific alternative."""
        betas = self.results.get_beta_values()
        mu = betas['MU']
        
        if alternative == 1:  # Train (in nest)
            return (betas['ASC_TRAIN'] + 
                   betas['B_TIME'] * self.TRAIN_TT_SCALED + 
                   betas['B_COST'] * self.TRAIN_COST_SCALED) / mu
        elif alternative == 2:  # Swissmetro (not in nest)
            return (#1.0 +  # Fixed ASC_SM value
                   betas['B_TIME'] * self.SM_TT_SCALED + 
                   betas['B_COST'] * self.SM_COST_SCALED)
        elif alternative == 3:  # Car (in nest)
            return (betas['ASC_CAR'] + 
                   betas['B_TIME'] * self.CAR_TT_SCALED + 
                   betas['B_COST'] * self.CAR_CO_SCALED) / mu
        else:
            raise ValueError(f"Invalid alternative: {alternative}")
        
    def get_metrics(self):
        """Get metrics including nest-specific information and VOT."""
        metrics = super().get_metrics()
        
        # Add correlation between alternatives in nests
        if hasattr(self, 'nests'):
            corr = self.nests.correlation(
                parameters=self.results.get_beta_values(),
                alternatives_names={1: 'Train', 2: 'Swissmetro', 3: 'Car'}
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
    
class MixedLogitModel_SM(BaseSwissmetroModel):
    """Mixed logit model implementation with random coefficients."""
    
    def estimate(self):
        """Estimate the mixed logit model."""
        # Parameters to be estimated
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_SM = Beta('ASC_SM', 0, None, None, 1)
        B_COST = Beta('B_COST', 0, None, None, 0)

        # Define random parameter for time
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_TIME_S = Beta('B_TIME_S', 1, None, None, 0)  # Spread parameter
        B_TIME_RND = B_TIME + B_TIME_S * bioDraws('b_time_rnd', 'NORMAL')

        # Definition of the utility functions with random coefficient
        V1 = ASC_TRAIN + B_TIME_RND * self.TRAIN_TT_SCALED + B_COST * self.TRAIN_COST_SCALED
        V2 = ASC_SM + B_TIME_RND * self.SM_TT_SCALED + B_COST * self.SM_COST_SCALED
        V3 = ASC_CAR + B_TIME_RND * self.CAR_TT_SCALED + B_COST * self.CAR_CO_SCALED

        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3}

        # Associate availability conditions
        av = {1: self.TRAIN_AV_SP, 2: self.SM_AV, 3: self.CAR_AV_SP}

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
        biogeme.modelName = "mixed_logit_model"
        
        # Disable HTML and Pickle generation
        biogeme.generateHtml = False
        biogeme.generatePickle = False
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.getGeneralStatistics()
        
        # Store statistics using correct key names
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square for the null model'][0]
        self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
        
        # Calculate value of time (in CHF/hour)
        betas = self.results.get_beta_values()
        self.vot = 60 * betas['B_TIME'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Calculate market shares and accuracies
        #self.calculate_market_shares()
        self.calculate_choice_accuracy()
        
        return self.results

    def _calculate_utilities(self, betas):
        """Calculate utilities for each alternative using Mixed Logit parameters."""
        # For mixed logit, we use the mean of the random parameter
        v1 = (betas['ASC_TRAIN'] + 
              betas['B_TIME'] * self.database.data['TRAIN_TT_SCALED'] + 
              betas['B_COST'] * self.database.data['TRAIN_COST_SCALED'])
        
        v2 = (#1.0 +  # Fixed ASC_SM value
              betas['B_TIME'] * self.database.data['SM_TT_SCALED'] + 
              betas['B_COST'] * self.database.data['SM_COST_SCALED'])
        
        v3 = (betas['ASC_CAR'] + 
              betas['B_TIME'] * self.database.data['CAR_TT_SCALED'] + 
              betas['B_COST'] * self.database.data['CAR_CO_SCALED'])
        
        return np.column_stack([v1, v2, v3])

    def _get_utility_function(self, alternative):
        """Get utility function for a specific alternative."""
        betas = self.results.get_beta_values()
        # For prediction, use mean of random parameter
        if alternative == 1:  # Train
            return (betas['ASC_TRAIN'] + 
                   betas['B_TIME'] * self.TRAIN_TT_SCALED + 
                   betas['B_COST'] * self.TRAIN_COST_SCALED)
        elif alternative == 2:  # Swissmetro
            return (#1.0 +  # Fixed ASC_SM value
                   betas['B_TIME'] * self.SM_TT_SCALED + 
                   betas['B_COST'] * self.SM_COST_SCALED)
        elif alternative == 3:  # Car
            return (betas['ASC_CAR'] + 
                   betas['B_TIME'] * self.CAR_TT_SCALED + 
                   betas['B_COST'] * self.CAR_CO_SCALED)
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
    data = loader.load_dataset("swissmetro_dataset")
    
    # Print initial diagnostics
    print("\nInitial Data Shape:", data.shape)
    print("\nInitial Value counts for CHOICE:")
    print(data['CHOICE'].value_counts())
    
    # Estimate MNL model
    print("\nEstimating Multinomial Logit Model...")
    mnl = MultinomialLogitModel_SM(data)
    mnl_results = mnl.estimate()
    mnl_metrics = mnl.get_metrics()
    print("\nMNL Metrics:", mnl_metrics)
    
    # Estimate NL model
    print("\nEstimating Nested Logit Model...")
    nl = NestedLogitModel_SM(data)
    nl_results = nl.estimate()
    nl_metrics = nl.get_metrics()
    print("\nNL Metrics:", nl_metrics)

    # Estimate Mixed Logit model
    print("\nEstimating Mixed Logit Model...")
    ml = MixedLogitModel_SM(data)
    ml_results = ml.estimate()
    ml_metrics = ml.get_metrics()
    print("\nMixL Metrics:", ml_metrics)

if __name__ == "__main__":
    main()
