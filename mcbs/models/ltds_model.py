"""
London Transport Demand Survey Mode Choice Model

This script implements mode choice models for the LTDS dataset, estimating choice between:
- walking
- cycling
- public transport
- driving

Based on: Tim Hillel's LTDS analysis (2018)
"""

import biogeme.biogeme_logging as blog
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo, exp
from biogeme.database import Database
from biogeme.nests import OneNestForNestedLogit, NestsForNestedLogit
from .base import BaseDiscreteChoiceModel
import numpy as np
import pandas as pd

class BaseLTDSModel(BaseDiscreteChoiceModel):
    """Base class for LTDS models with shared initialization."""
    
    def __init__(self, data):
        # Encode categorical variables before creating database
        data = self._encode_categorical_variables(data)
        super().__init__(data)
        self._initialize_variables()

    def _encode_categorical_variables(self, df):
        """
        Encode categorical variables according to LTDS dataset specifications.
        
        Parameters:
        df (pandas.DataFrame): DataFrame containing the categorical columns
        
        Returns:
        pandas.DataFrame: DataFrame with encoded categorical variables
        """
        # Create a copy to avoid modifying the original
        df_encoded = df.copy()
        
        # Travel mode encoding
        mode_mapping = {
            'walk': 1,
            'cycle': 2,
            'pt': 3,
            'drive': 4
        }
        
        # Purpose encoding
        purpose_mapping = {
            'HBW': 1,    # home-based work
            'HBE': 2,    # home-based education
            'HBO': 3,    # home-based other
            'B': 4,      # employers' business
            'NHBO': 5    # non-home-based other
        }
        
        # Fuel type encoding - Fixed case to match actual data
        fueltype_mapping = {
            'Petrol_Car': 1,
            'Diesel_Car': 2,
            'Hybrid_Car': 3,
            'Petrol_LGV': 4,
            'Diesel_LGV': 5,
            'Average_Car': 6  # Fixed case from 'Average_car' to 'Average_Car'
        }
        
        # Fare type encoding
        faretype_mapping = {
            'full': 1,
            '16+': 2,
            'child': 3,
            'dis': 4,    # disabled
            'free': 5
        }
        
        # Apply the mappings
        if 'travel_mode' in df.columns:
            df_encoded['travel_mode'] = df['travel_mode'].map(mode_mapping)
        
        if 'purpose' in df.columns:
            df_encoded['purpose'] = df['purpose'].map(purpose_mapping)
        
        if 'fueltype' in df.columns:
            df_encoded['fueltype'] = df['fueltype'].map(fueltype_mapping)
        
        if 'faretype' in df.columns:
            df_encoded['faretype'] = df['faretype'].map(faretype_mapping)
        
        return df_encoded
        
    def _initialize_variables(self):
        """Initialize all variables needed for LTDS models."""
        # Basic trip variables
        self.CHOICE = Variable('travel_mode')
        
        # Time and cost variables
        self.WALK_TIME = Variable('dur_walking')
        self.CYCLE_TIME = Variable('dur_cycling')
        self.DRIVE_TIME = Variable('dur_driving')
        self.PT_ACCESS = Variable('dur_pt_access')
        self.PT_RAIL = Variable('dur_pt_rail')
        self.PT_BUS = Variable('dur_pt_bus')
        self.PT_INT = Variable('dur_pt_int_total')  # Updated from dur_pt_int
        self.DRIVE_COST = Variable('cost_driving_fuel')
        self.CCHARGE = Variable('cost_driving_con_charge')  # Updated from cost_driving_ccharge
        self.PT_COST = Variable('cost_transit')
        self.TRAFFIC = Variable('driving_traffic_percent')
        
        # Print initial statistics
        print("\nData Shape:", self.database.data.shape)
        print("\nMode choice distribution:")
        print(self.database.data['travel_mode'].value_counts())

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
        
        # Associate availability conditions (assuming all modes available)
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        
        # Calculate choice probabilities
        prob_walk = models.logit(V, av, 1)
        prob_cycle = models.logit(V, av, 2)
        prob_pt = models.logit(V, av, 3)
        prob_drive = models.logit(V, av, 4)
        
        # Setup simulation
        simulate = {
            'Prob. walk': prob_walk,
            'Prob. cycle': prob_cycle,
            'Prob. PT': prob_pt,
            'Prob. drive': prob_drive
        }
        
        # Run simulation
        biogeme = bio.BIOGEME(self.database, simulate)
        biogeme.modelName = "choice_prediction"
        simulatedValues = biogeme.simulate(self.results.get_beta_values())
        
        # Calculate market shares
        actual_counts = self.database.data['travel_mode'].value_counts()
        total = len(self.database.data)
        self.actual_shares = {i: actual_counts.get(i, 0) / total for i in [1, 2, 3, 4]}
        
        # Calculate predicted shares (mean probability for each alternative)
        self.predicted_shares = {
            1: simulatedValues['Prob. walk'].mean(),
            2: simulatedValues['Prob. cycle'].mean(),
            3: simulatedValues['Prob. PT'].mean(),
            4: simulatedValues['Prob. drive'].mean()
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
            'Prob. walk': 1, 
            'Prob. cycle': 2, 
            'Prob. PT': 3, 
            'Prob. drive': 4
        })
        
        # Create confusion matrix
        data_sim = {
            'y_Actual': self.database.data['travel_mode'],
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
        self.confusion_matrix.insert(1, '2', 0)


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
        av = {1: 1, 2: 1, 3: 1, 4: 1}
        
        # Calculate choice probabilities
        prob_walk = models.nested(V, None, self.nests, 1)
        prob_cycle = models.nested(V, None, self.nests, 2)
        prob_pt = models.nested(V, None, self.nests, 3)
        prob_drive = models.nested(V, None, self.nests, 4)
        
        # Setup simulation
        simulate = {
            'Prob. walk': prob_walk,
            'Prob. cycle': prob_cycle,
            'Prob. PT': prob_pt,
            'Prob. drive': prob_drive
        }
        
        # Run simulation
        biogeme = bio.BIOGEME(self.database, simulate)
        biogeme.modelName = "choice_prediction_nest"
        simulatedValues = biogeme.simulate(self.results.get_beta_values())
        
        # Calculate market shares
        actual_counts = self.database.data['travel_mode'].value_counts()
        total = len(self.database.data)
        self.actual_shares = {i: actual_counts.get(i, 0) / total for i in [1, 2, 3, 4]}
        
        # Calculate predicted shares (mean probability for each alternative)
        self.predicted_shares = {
            1: simulatedValues['Prob. walk'].mean(),
            2: simulatedValues['Prob. cycle'].mean(),
            3: simulatedValues['Prob. PT'].mean(),
            4: simulatedValues['Prob. drive'].mean()
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
            'Prob. walk': 1, 
            'Prob. cycle': 2, 
            'Prob. PT': 3, 
            'Prob. drive': 4
        })
        
        # Create confusion matrix
        data_sim = {
            'y_Actual': self.database.data['travel_mode'],
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
        self.confusion_matrix.insert(1, '2', 0)

        # Calculate accuracy
        self.choice_accuracy = (
            np.diagonal(self.confusion_matrix.to_numpy()).sum() / 
            self.confusion_matrix.to_numpy().sum()
        )
        
        print("\nConfusion Matrix:")
        print(self.confusion_matrix)
        print(f"\nChoice Prediction Accuracy: {self.choice_accuracy:.3f}")

class MultinomialLogitModel_L(BaseLTDSModel):
    """Multinomial logit model implementation."""
    
    def estimate(self):
        """Estimate the multinomial logit model."""
        # Parameters to be estimated
        ASC_WALKING = Beta('ASC_WALKING', 0, None, None, 1)
        ASC_CYCLING = Beta('ASC_CYCLING', 0, None, None, 0)
        ASC_PT = Beta('ASC_PT', 0, None, None, 0)
        ASC_DRIVING = Beta('ASC_DRIVING', 0, None, None, 0)
        
        B_TIME_WALKING = Beta('B_TIME_WALKING', 0, None, None, 0)
        B_TIME_CYCLING = Beta('B_TIME_CYCLING', 0, None, None, 0)
        B_TIME_DRIVING = Beta('B_TIME_DRIVING', 0, None, None, 0)
        B_COST_DRIVING = Beta('B_COST_DRIVING', 0, None, None, 0)
        B_COST_PT = Beta('B_COST_PT', 0, None, None, 0)
        B_TIME_PT_BUS = Beta('B_TIME_PT_BUS', 0, None, None, 0)
        B_TIME_PT_RAIL = Beta('B_TIME_PT_RAIL', 0, None, None, 0)
        B_TIME_PT_ACCESS = Beta('B_TIME_PT_ACCESS', 0, None, None, 0)
        B_TIME_PT_INT = Beta('B_TIME_PT_INT', 0, None, None, 0)
        B_TRAFFIC_DRIVING = Beta('B_TRAFFIC_DRIVING', 0, None, None, 0)

        # Utility functions
        V1 = ASC_WALKING + B_TIME_WALKING * self.WALK_TIME
        
        V2 = ASC_CYCLING + B_TIME_CYCLING * self.CYCLE_TIME
        
        V3 = (ASC_PT + 
              B_COST_PT * self.PT_COST + 
              B_TIME_PT_ACCESS * self.PT_ACCESS + 
              B_TIME_PT_RAIL * self.PT_RAIL + 
              B_TIME_PT_BUS * self.PT_BUS +
              B_TIME_PT_INT * self.PT_INT)
        
        V4 = (ASC_DRIVING + 
              B_TIME_DRIVING * self.DRIVE_TIME +
              B_COST_DRIVING * (self.DRIVE_COST + self.CCHARGE) +
              B_TRAFFIC_DRIVING * self.TRAFFIC)

        # Associate utility functions with alternatives (1: walk, 2: cycle, 3: PT, 4: drive)
        V = {1: V1, 2: V2, 3: V3, 4: V4}

        # Associate availability conditions (assuming all modes available)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Define and estimate the model
        logprob = models.loglogit(V, av, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "ltds_mnl"
        
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
        
        # Calculate value of time (in £/hour) for each mode
        betas = self.results.get_beta_values()
        # Walking VOT
        self.vot_walking =  betas['B_TIME_WALKING'] / betas['B_COST_PT'] if 'B_COST_PT' in betas else None
        # Cycling VOT
        self.vot_cycling =  betas['B_TIME_CYCLING'] / betas['B_COST_PT'] if 'B_COST_PT' in betas else None
        # PT VOT (using average of access, rail, bus, and interchange time)
        pt_time_coef = (betas['B_TIME_PT_ACCESS'] + betas['B_TIME_PT_RAIL'] + 
                       betas['B_TIME_PT_BUS'] + betas['B_TIME_PT_INT']) / 4
        self.vot_pt = pt_time_coef / betas['B_COST_PT'] if 'B_COST_PT' in betas else None
        # Driving VOT
        self.vot_driving = betas['B_TIME_DRIVING'] / betas['B_COST_DRIVING'] if 'B_COST_DRIVING' in betas else None
        
        # Calculate choice accuracy and market shares
        self.calculate_choice_accuracy()
        
        return self.results

    def _calculate_utilities(self, betas):
        """Calculate utilities for each alternative using MNL parameters."""
        v1 = (0 + betas['B_TIME_WALKING'] * self.database.data['dur_walking'])  # ASC_WALKING fixed to 0
        
        v2 = (betas['ASC_CYCLING'] + 
              betas['B_TIME_CYCLING'] * self.database.data['dur_cycling'])
        
        v3 = (betas['ASC_PT'] + 
              betas['B_COST_PT'] * self.database.data['cost_transit'] + 
              betas['B_TIME_PT_ACCESS'] * self.database.data['dur_pt_access'] + 
              betas['B_TIME_PT_RAIL'] * self.database.data['dur_pt_rail'] + 
              betas['B_TIME_PT_BUS'] * self.database.data['dur_pt_bus'] +
              betas['B_TIME_PT_INT'] * self.database.data['dur_pt_int_total'])
        
        v4 = (betas['ASC_DRIVING'] + 
              betas['B_TIME_DRIVING'] * self.database.data['dur_driving'] + 
              betas['B_COST_DRIVING'] * (self.database.data['cost_driving_fuel'] + 
                                       self.database.data['cost_driving_con_charge']) +
              betas['B_TRAFFIC_DRIVING'] * self.database.data['driving_traffic_percent'])
        
        return np.column_stack([v1, v2, v3, v4])

    def _get_utility_function(self, alternative):
        """Get utility function for a specific alternative."""
        betas = self.results.get_beta_values()
        if alternative == 1:  # Walking
            return (0 + betas['B_TIME_WALKING'] * self.WALK_TIME)  # ASC_WALKING fixed to 0
        elif alternative == 2:  # Cycling
            return (betas['ASC_CYCLING'] + 
                   betas['B_TIME_CYCLING'] * self.CYCLE_TIME)
        elif alternative == 3:  # PT
            return (betas['ASC_PT'] + 
                   betas['B_COST_PT'] * self.PT_COST + 
                   betas['B_TIME_PT_ACCESS'] * self.PT_ACCESS + 
                   betas['B_TIME_PT_RAIL'] * self.PT_RAIL + 
                   betas['B_TIME_PT_BUS'] * self.PT_BUS +
                   betas['B_TIME_PT_INT'] * self.PT_INT)
        elif alternative == 4:  # Driving
            return (betas['ASC_DRIVING'] + 
                   betas['B_TIME_DRIVING'] * self.DRIVE_TIME +
                   betas['B_COST_DRIVING'] * (self.DRIVE_COST + self.CCHARGE) +
                   betas['B_TRAFFIC_DRIVING'] * self.TRAFFIC)
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
            'vot_walking': self.vot_walking if hasattr(self, 'vot_walking') else None,
            'vot_cycling': self.vot_cycling if hasattr(self, 'vot_cycling') else None,
            'vot_pt': self.vot_pt if hasattr(self, 'vot_pt') else None,
            'vot_driving': self.vot_driving if hasattr(self, 'vot_driving') else None,
            'market_share_accuracy': self.market_share_accuracy if hasattr(self, 'market_share_accuracy') else None,
            'choice_accuracy': self.choice_accuracy if hasattr(self, 'choice_accuracy') else None,
            'actual_shares': self.actual_shares if hasattr(self, 'actual_shares') else None,
            'predicted_shares': self.predicted_shares if hasattr(self, 'predicted_shares') else None,
            'confusion_matrix': self.confusion_matrix.to_dict() if hasattr(self, 'confusion_matrix') else None
        })
        
        return metrics

class MultinomialLogitModelTotal_L(BaseLTDSModel):
    """Multinomial logit model implementation with single time coefficient."""
    
    def estimate(self):
        """Estimate the multinomial logit model with total time."""
        # Parameters to be estimated
        ASC_WALKING = Beta('ASC_WALKING', 0, None, None, 1)
        ASC_CYCLING = Beta('ASC_CYCLING', 0, None, None, 0)
        ASC_PT = Beta('ASC_PT', 0, None, None, 0)
        ASC_DRIVING = Beta('ASC_DRIVING', 0, None, None, 0)
        
        B_TIME_TOTAL = Beta('B_TIME_TOTAL', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)
        B_TRAFFIC_DRIVING = Beta('B_TRAFFIC_DRIVING', 0, None, None, 0)

        # Utility functions with total time
        V1 = ASC_WALKING + B_TIME_TOTAL * self.WALK_TIME
        
        V2 = ASC_CYCLING + B_TIME_TOTAL * self.CYCLE_TIME
        
        V3 = (ASC_PT + 
              B_COST * self.PT_COST + 
              B_TIME_TOTAL * (self.PT_ACCESS + self.PT_RAIL + 
                            self.PT_BUS + self.PT_INT))
        
        V4 = (ASC_DRIVING + 
              B_TIME_TOTAL * self.DRIVE_TIME +
              B_COST * (self.DRIVE_COST + self.CCHARGE) +
              B_TRAFFIC_DRIVING * self.TRAFFIC)

        # Associate utility functions with alternatives (1: walk, 2: cycle, 3: PT, 4: drive)
        V = {1: V1, 2: V2, 3: V3, 4: V4}

        # Associate availability conditions (assuming all modes available)
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Define and estimate the model
        logprob = models.loglogit(V, av, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "ltds_mnl_total"
        
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
        
        # Calculate value of time (in £/hour) using total time coefficient
        betas = self.results.get_beta_values()
        self.vot = 60 * betas['B_TIME_TOTAL'] / betas['B_COST'] if 'B_COST' in betas else None
        
        # Calculate choice accuracy and market shares
        self.calculate_choice_accuracy()
        
        return self.results

    def _calculate_utilities(self, betas):
        """Calculate utilities for each alternative using MNL parameters."""
        v1 = (0 + betas['B_TIME_TOTAL'] * self.database.data['dur_walking'])  # ASC_WALKING fixed to 0
        
        v2 = (betas['ASC_CYCLING'] + 
              betas['B_TIME_TOTAL'] * self.database.data['dur_cycling'])
        
        v3 = (betas['ASC_PT'] + 
              betas['B_COST'] * self.database.data['cost_transit'] + 
              betas['B_TIME_TOTAL'] * (self.database.data['dur_pt_access'] + 
                                     self.database.data['dur_pt_rail'] + 
                                     self.database.data['dur_pt_bus'] + 
                                     self.database.data['dur_pt_int_total']))
        
        v4 = (betas['ASC_DRIVING'] + 
              betas['B_TIME_TOTAL'] * self.database.data['dur_driving'] + 
              betas['B_COST'] * (self.database.data['cost_driving_fuel'] + 
                                self.database.data['cost_driving_con_charge']) +
              betas['B_TRAFFIC_DRIVING'] * self.database.data['driving_traffic_percent'])
        
        return np.column_stack([v1, v2, v3, v4])

    def _get_utility_function(self, alternative):
        """Get utility function for a specific alternative."""
        betas = self.results.get_beta_values()
        if alternative == 1:  # Walking
            return (0 + betas['B_TIME_TOTAL'] * self.WALK_TIME)  # ASC_WALKING fixed to 0
        elif alternative == 2:  # Cycling
            return (betas['ASC_CYCLING'] + 
                   betas['B_TIME_TOTAL'] * self.CYCLE_TIME)
        elif alternative == 3:  # PT
            return (betas['ASC_PT'] + 
                   betas['B_COST'] * self.PT_COST + 
                   betas['B_TIME_TOTAL'] * (self.PT_ACCESS + self.PT_RAIL + 
                                          self.PT_BUS + self.PT_INT))
        elif alternative == 4:  # Driving
            return (betas['ASC_DRIVING'] + 
                   betas['B_TIME_TOTAL'] * self.DRIVE_TIME +
                   betas['B_COST'] * (self.DRIVE_COST + self.CCHARGE) +
                   betas['B_TRAFFIC_DRIVING'] * self.TRAFFIC)
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

class NestedLogitModel_L(BaseLTDSModel):
    """Nested logit model implementation with motorized modes nest."""
    
    def estimate(self):
        """Estimate the nested logit model."""
        # Same parameters as MNL
        ASC_WALKING = Beta('ASC_WALKING', 0, None, None, 1)
        ASC_CYCLING = Beta('ASC_CYCLING', 0, None, None, 0)
        ASC_PT = Beta('ASC_PT', 0, None, None,0)
        ASC_DRIVING = Beta('ASC_DRIVING', 0, None, None, 0)
        
        B_TIME_WALKING = Beta('B_TIME_WALKING', 0, None, None, 0)
        B_TIME_CYCLING = Beta('B_TIME_CYCLING', 0, None, None, 0)
        B_TIME_DRIVING = Beta('B_TIME_DRIVING', 0, None, None, 0)
        B_COST_DRIVING = Beta('B_COST_DRIVING', 0, None, None, 0)
        B_COST_PT = Beta('B_COST_PT', 0, None, None, 0)
        B_TIME_PT_BUS = Beta('B_TIME_PT_BUS', 0, None, None, 0)
        B_TIME_PT_RAIL = Beta('B_TIME_PT_RAIL', 0, None, None, 0)
        B_TIME_PT_ACCESS = Beta('B_TIME_PT_ACCESS', 0, None, None, 0)
        B_TIME_PT_INT = Beta('B_TIME_PT_INT', 0, None, None, 0)
        B_TRAFFIC_DRIVING = Beta('B_TRAFFIC_DRIVING', 0, None, None, 0)
        
        # Nesting parameter
        MU_MOTORIZED = Beta('MU_MOTORIZED', 1, 1, 10, 0)

        # Utility functions (same as MNL)
        V1 = ASC_WALKING + B_TIME_WALKING * self.WALK_TIME
        
        V2 = ASC_CYCLING + B_TIME_CYCLING * self.CYCLE_TIME
        
        V3 = (ASC_PT + 
              B_COST_PT * self.PT_COST + 
              B_TIME_PT_ACCESS * self.PT_ACCESS + 
              B_TIME_PT_RAIL * self.PT_RAIL + 
              B_TIME_PT_BUS * self.PT_BUS +
              B_TIME_PT_INT * self.PT_INT)
        
        V4 = (ASC_DRIVING + 
              B_TIME_DRIVING * self.DRIVE_TIME +
              B_COST_DRIVING * (self.DRIVE_COST + self.CCHARGE) +
              B_TRAFFIC_DRIVING * self.TRAFFIC)

        # Associate utility functions with alternatives
        V = {1: V1, 2: V2, 3: V3, 4: V4}

        # Associate availability conditions
        av = {1: 1, 2: 1, 3: 1, 4: 1}

        # Define nests: motorized modes (PT and car) in one nest
        motorized = OneNestForNestedLogit(
            nest_param=MU_MOTORIZED,
            list_of_alternatives=[3, 4],
            name='motorized'
        )
        nests = NestsForNestedLogit(choice_set=list(V), tuple_of_nests=(motorized,))

        # Store nests for later use
        self.nests = nests

        # Define and estimate the model
        logprob = models.lognested(V, av, nests, self.CHOICE)
        biogeme = bio.BIOGEME(self.database, logprob)
        biogeme.modelName = "ltds_nl"
        
        # Disable HTML and Pickle generation
        biogeme.generateHtml = True
        biogeme.generatePickle = True
        
        # Calculate null log likelihood and estimate
        biogeme.calculate_null_loglikelihood(av)
        self.results = biogeme.estimate()
        
        # Get general statistics
        stats = self.results.get_general_statistics()
        self.final_ll = stats['Final log likelihood'][0]
        self.rho_squared = stats['Rho-square for the null model'][0]
        self.rho_squared_bar = stats['Rho-square-bar for the null model'][0]
        
        # Calculate value of time (in £/hour) for each mode
        betas = self.results.get_beta_values()
        # Walking VOT
        self.vot_walking = 60 * betas['B_TIME_WALKING'] / betas['B_COST_PT'] if 'B_COST_PT' in betas else None
        # Cycling VOT
        self.vot_cycling = 60 * betas['B_TIME_CYCLING'] / betas['B_COST_PT'] if 'B_COST_PT' in betas else None
        # PT VOT (using average of access, rail, bus, and interchange time)
        pt_time_coef = (betas['B_TIME_PT_ACCESS'] + betas['B_TIME_PT_RAIL'] + 
                       betas['B_TIME_PT_BUS'] + betas['B_TIME_PT_INT']) / 4
        self.vot_pt = 60 * pt_time_coef / betas['B_COST_PT'] if 'B_COST_PT' in betas else None
        # Driving VOT
        self.vot_driving = 60 * betas['B_TIME_DRIVING'] / betas['B_COST_DRIVING'] if 'B_COST_DRIVING' in betas else None
        
        # Calculate choice accuracy and market shares
        self.calculate_choice_accuracy_nest()
        
        return self.results
    
    def _get_utility_function(self, alternative):
        """Get utility function for a specific alternative."""
        betas = self.results.get_beta_values()
        if alternative == 1:  # Walking
            return (0 + betas['B_TIME_WALKING'] * self.WALK_TIME)  # ASC_WALKING fixed to 0
        elif alternative == 2:  # Cycling
            return (betas['ASC_CYCLING'] + 
                   betas['B_TIME_CYCLING'] * self.CYCLE_TIME)
        elif alternative == 3:  # PT
            return (betas['ASC_PT'] + 
                   betas['B_COST_PT'] * self.PT_COST + 
                   betas['B_TIME_PT_ACCESS'] * self.PT_ACCESS + 
                   betas['B_TIME_PT_RAIL'] * self.PT_RAIL + 
                   betas['B_TIME_PT_BUS'] * self.PT_BUS +
                   betas['B_TIME_PT_INT'] * self.PT_INT)
        elif alternative == 4:  # Driving
            return (betas['ASC_DRIVING'] + 
                   betas['B_TIME_DRIVING'] * self.DRIVE_TIME +
                   betas['B_COST_DRIVING'] * (self.DRIVE_COST + self.CCHARGE) +
                   betas['B_TRAFFIC_DRIVING'] * self.TRAFFIC)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")

    def get_metrics(self):
        """Get metrics including nest-specific information and VOT."""
        metrics = super().get_metrics()
        
        # Add correlation between alternatives in nests
        if hasattr(self, 'nests'):
            corr = self.nests.correlation(
                parameters=self.results.get_beta_values(),
                alternatives_names={1: 'Walk', 2: 'Cycle', 3: 'PT', 4: 'Drive'}
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
    

def main():
    """Test the models with sample data."""
    from mcbs.datasets import DatasetLoader
    
    # Load data
    loader = DatasetLoader()
    data = loader.load_dataset("ltds_dataset")
    
    # Print initial diagnostics
    print("\nInitial Data Shape:", data.shape)
    print("\nInitial Value counts for CHOICE:")
    print(data['travel_mode'].value_counts())
    
    # Estimate MNL model
    print("\nEstimating Multinomial Logit Model...")
    mnl = MultinomialLogitModel_L(data)
    mnl_results = mnl.estimate()
    mnl_metrics = mnl.get_metrics()
    print("\nMNL Metrics:", mnl_metrics)

    # Estimate MNL model
    print("\nEstimating Total Multinomial Logit Model...")
    mnl = MultinomialLogitModelTotal_L(data)
    mnl_results = mnl.estimate()
    mnl_metrics = mnl.get_metrics()
    print("\nMNL Metrics:", mnl_metrics)
    
    # Estimate NL model
    print("\nEstimating Nested Logit Model...")
    nl = NestedLogitModel_L(data)
    nl_results = nl.estimate()
    nl_metrics = nl.get_metrics()
    print("\nNL Metrics:", nl_metrics)


if __name__ == "__main__":
    main()
