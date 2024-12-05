# mcbs/utils/individual_parameters.py

import numpy as np
from typing import Dict, Any, List, Tuple
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme.expressions import Expression, Beta, bioDraws, exp

class IndividualParameterCalculator:
    """Calculates individual-specific parameters from mixture model results"""
    
    def __init__(self, 
                 model_results: Any,  # Changed from bio.ModelResults
                 data: pd.DataFrame,
                 choice_col: str,
                 parameter_name: str):
        """
        Initialize calculator with model results and data
        
        Args:
            model_results: Estimated biogeme model results object
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
        betas = self.model_results.getBetaValues()
        std_errs = self.model_results.getStdErrValues()
        param_mean = betas[self.parameter_name]
        param_std = std_errs[self.parameter_name]
        
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
        raise NotImplementedError(
            "Implement specific choice probability calculation")

class SwissmetroIndividualCalculator(IndividualParameterCalculator):
    """Individual parameter calculator specifically for Swissmetro"""
    
    def _calculate_choice_probability(self, beta_value: float) -> np.ndarray:
        """
        Calculate Swissmetro choice probabilities for a parameter value
        """
        # Create database expressions
        CAR_TT = Expression('CAR_TT', self.database.data['CAR_TT'])
        TRAIN_TT = Expression('TRAIN_TT', self.database.data['TRAIN_TT'])
        SM_TT = Expression('SM_TT', self.database.data['SM_TT'])
        
        # Parameters with fixed beta value
        TRAIN_ASC = Beta('TRAIN_ASC', 0, None, None, 0)
        SM_ASC = Beta('SM_ASC', 0, None, None, 0) 
        BETA_TIME = Beta('BETA_TIME', beta_value, None, None, 0)
        
        # Calculate utilities
        V_TRAIN = TRAIN_ASC + BETA_TIME * TRAIN_TT 
        V_SM = SM_ASC + BETA_TIME * SM_TT
        V_CAR = BETA_TIME * CAR_TT
        
        # Calculate exponentials of utilities
        e_train = np.exp(V_TRAIN.getValue())
        e_sm = np.exp(V_SM.getValue())  
        e_car = np.exp(V_CAR.getValue())
        
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


class RandomCoefficientCalculator:
    """Calculator for individual-specific parameters from random coefficient models"""
    
    def __init__(self, 
                 model_results: Any,
                 data: pd.DataFrame,
                 choice_col: str):
        """
        Initialize calculator
        
        Args:
            model_results: Estimated biogeme model results
            data: DataFrame containing choice and attribute data
            choice_col: Name of column containing choices
        """
        self.model_results = model_results
        self.database = db.Database("random_coef", data)
        self.choice_col = choice_col
        
    def calculate_individual_betas(self, 
                                 n_draws: int = 1000,
                                 seed: int = 42) -> pd.DataFrame:
        """
        Calculate individual-specific parameters using Bayes theorem
        
        P(β|i,x) = P(i|x,β)f(β)/P(i|x)
        
        Args:
            n_draws: Number of draws for Monte Carlo simulation
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame containing individual parameter values
        """
        np.random.seed(seed)
        
        # Get parameter estimates
        betas = self.model_results.getBetaValues()
        mu = betas['MU']  
        sigma = betas['SIGMA']
        
        # Generate random draws from lognormal distribution
        beta_draws = -np.exp(mu + sigma * np.random.normal(0, 1, (n_draws, len(self.database.data))))
        
        # Calculate choice probabilities for each draw
        all_probs = []
        for draw in range(n_draws):
            probs = self._calculate_choice_probability(beta_draws[draw])
            all_probs.append(probs)
        
        # Stack probabilities
        all_probs = np.array(all_probs)
        
        # Calculate individual parameters using Bayes formula
        # E[β|i] = ∑(β_r * P(i|β_r)) / ∑P(i|β_r)
        weights = beta_draws * all_probs
        individual_betas = np.sum(weights, axis=0) / np.sum(all_probs, axis=0)
        
        return pd.Series(individual_betas, 
                        index=self.database.data.index,
                        name='beta_time_individual')

    def _calculate_choice_probability(self, beta_time: float) -> np.ndarray:
        """
        Calculate choice probability for given time parameter
        """
        # Get other parameters from model
        betas = self.model_results.getBetaValues()
        
        # Parameters
        ASC_CAR = Beta('ASC_CAR', betas['ASC_CAR'], None, None, 0)
        ASC_TRAIN = Beta('ASC_TRAIN', betas['ASC_TRAIN'], None, None, 0)
        ASC_SM = Beta('ASC_SM', betas['ASC_SM'], None, None, 0)
        B_COST = Beta('B_COST', betas['B_COST'], None, None, 0)
        B_TIME = Beta('B_TIME', beta_time, None, None, 0)
        
        # Calculate utilities
        V_TRAIN = ASC_TRAIN + \
                  B_TIME * self.database.TRAIN_TT + \
                  B_COST * self.database.TRAIN_CO
        
        V_CAR = ASC_CAR + \
                B_TIME * self.database.CAR_TT + \
                B_COST * self.database.CAR_CO
        
        V_SM = ASC_SM + \
               B_TIME * self.database.SM_TT + \
               B_COST * self.database.SM_CO
        
        # Calculate exponentials
        e_train = np.exp(V_TRAIN.getValue())
        e_sm = np.exp(V_SM.getValue())  
        e_car = np.exp(V_CAR.getValue())
        
        # Get denominator
        denom = e_train + e_sm + e_car
        
        # Get probabilities for chosen alternative
        chosen = self.database.data[self.choice_col].values
        probs = np.where(chosen == 1, e_car/denom,
                        np.where(chosen == 2, e_train/denom, e_sm/denom))
        
        return probs

def plot_individual_betas_by_mode(betas: pd.Series,
                                chosen_mode: pd.Series,
                                title: str = "Distribution of Individual Time Parameters by Mode"):
    """
    Plot histogram of individual parameters colored by chosen mode
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10,6))
    
    modes = {1: "Car", 2: "Train", 3: "Swissmetro"}
    
    for mode in sorted(chosen_mode.unique()):
        mask = chosen_mode == mode
        plt.hist(betas[mask], 
                alpha=0.5,
                label=modes.get(mode, f"Mode {mode}"),
                bins=30)
    
    plt.xlabel('Time Parameter Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    class RandomCoefficientIndividualCalculator:
        """
        Combines random coefficient estimation with individual parameter calculation
        for the Swissmetro dataset.
        """
        def __init__(self, data: pd.DataFrame):
            """
            Initialize calculator with cleaned dataset.
        
            Args:
                data: DataFrame containing Swissmetro choice data
            """
            self.data = data
            self.setup_variables()
            self.setup_random_draws()
        
        def setup_variables(self):
            """Define model variables and parameters"""
            # Choice
            self.CHOICE = Variable('CHOICE')
        
            # Alternative specific constants
            self.ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
            self.ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
            self.ASC_SM = Beta('ASC_SM', 0, None, None, 1)
        
            # Time and cost parameters
            self.B_TIME = Beta('B_TIME', -1, None, None, 0)
            self.B_COST = Beta('B_COST', -1, None, None, 0)
        
            # Random parameter for time - lognormal distribution
            self.B_TIME_S = Beta('B_TIME_S', 0.3, None, None, 0)
            self.omega = bioDraws('omega', 'NORMAL')
            self.lambda_time = -exp(self.B_TIME + self.B_TIME_S * self.omega)
        
        def setup_random_draws(self, R: int = 1000):
            """
            Setup Monte Carlo draws for integration
        
            Args:
                R: Number of random draws
            """
            self.R = R
            self.simulation = MonteCarlo(self.calculate_choice_probability())
        
        def calculate_choice_probability(self):
            """
            Calculate choice probabilities using random coefficient specification
        
            Returns:
                Choice probability expression
            """
            # Define availability conditions
            AV_CAR = Variable('AV_CAR')
            AV_TRAIN = Variable('AV_TRAIN') 
            AV_SM = Variable('AV_SM')
        
            # Times and costs (properly scaled)
            TIME_CAR = Variable('TIME_CAR') / 60
            TIME_TRAIN = Variable('TIME_TRAIN') / 60
            TIME_SM = Variable('TIME_SM') / 60
        
            COST_CAR = Variable('COST_CAR') / 100
            COST_TRAIN = Variable('COST_TRAIN') / 100
            COST_SM = Variable('COST_SM') / 100
        
            # Utility functions
            V_CAR = (self.ASC_CAR + 
                    self.lambda_time * TIME_CAR +
                    self.B_COST * COST_CAR)
        
            V_TRAIN = (self.ASC_TRAIN + 
                    self.lambda_time * TIME_TRAIN +
                    self.B_COST * COST_TRAIN)
        
            V_SM = (self.ASC_SM +
                    self.lambda_time * TIME_SM + 
                    self.B_COST * COST_SM)
        
            # Associate utility functions with alternatives
            V = {1: V_CAR,
                2: V_TRAIN, 
                3: V_SM}
        
            # Associate availability conditions with alternatives
            av = {1: AV_CAR,
                2: AV_TRAIN,
                3: AV_SM}
        
            # Logit choice probability
            prob = bioLogit(V, av, self.CHOICE)
            return prob
    
        def calculate_individual_parameters(self):
            """
            Calculate individual-specific parameters using Bayes theorem
        
            Returns:
                DataFrame with individual parameter estimates
            """
            # Get estimated parameters
            betas = self.model.getBetaValues()
        
            # Initialize storage for individual parameters
            n_individuals = len(self.data)
            individual_params = np.zeros((n_individuals, 2))  # Time and cost parameters
        
            # Calculate posterior distribution parameters for each individual
            for i in range(n_individuals):
                # Get individual's choices
                choices = self.data.iloc[i]
            
                # Calculate likelihood of observed choices
                choice_prob = self.calculate_choice_probability_individual(choices, betas)
            
                # Calculate posterior distribution
                posterior_mean = self.calculate_posterior_mean(choice_prob, betas)
            
                # Store individual parameters
                individual_params[i] = posterior_mean
            
            return pd.DataFrame(
                individual_params,
                columns=['time_param', 'cost_param'],
                index=self.data.index
            )
    
        def calculate_choice_probability_individual(self, choices, betas):
            """
            Calculate choice probability for an individual
        
            Args:
            choices: Series with individual's choices
            betas: Dictionary of estimated parameters
            
            Returns:
            Choice probability for the individual
            """
            # Implementation of individual choice probability calculation
            # This will use the random coefficient specification
            pass
    
        def calculate_posterior_mean(self, choice_prob, betas):
            """
            Calculate posterior mean of parameters using Bayes theorem
        
            Args:
               choice_prob: Individual's choice probability
               betas: Dictionary of estimated parameters
            
            Returns:
                Posterior mean of parameters for individual
            """
            # Implementation of Bayes theorem for posterior calculation
            pass
    
        def plot_parameter_distributions(self, individual_params: pd.DataFrame):
            """
            Create visualizations of parameter distributions
        
            Args:
                individual_params: DataFrame with individual parameter estimates
            """
            import matplotlib.pyplot as plt
            import seaborn as sns
        
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
            # Time parameter distribution
            sns.histplot(data=individual_params, x='time_param', ax=ax1)
            ax1.set_title('Distribution of Individual Time Parameters')
            ax1.set_xlabel('Time Parameter')
        
            # Cost parameter distribution
            sns.histplot(data=individual_params, x='cost_param', ax=ax2)
            ax2.set_title('Distribution of Individual Cost Parameters')
            ax2.set_xlabel('Cost Parameter')
        
            plt.tight_layout()
            plt.show()

