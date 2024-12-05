# sample_benchmark.py

from mcbs.benchmarking.benchmark import Benchmark
from mcbs.datasets.dataset_loader import DatasetLoader
#from mcbs.datasets.dataset_prep import PrepareData

import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.models as models
from biogeme.expressions import Beta, Variable
from typing import Dict, Tuple, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class ModeChoiceModel:
    def __init__(self, data: pd.DataFrame):
        """Initialize the model with dataset."""
        # Store original data
        self.raw_data = data
        
        # Initialize database and results objects
        self.database = None
        self.model = None
        self.results = None
        
        # Define constants
        self.MODE_TO_NUM = {
            'train': 1,
            'car': 2,
            'bus': 3,
            'air': 4
        }
        self.NUM_TO_MODE = {v: k for k, v in self.MODE_TO_NUM.items()}
        
        # Initialize dictionaries for model components
        self.vars = {}  # Will store Biogeme variables
        self.betas = {} # Will store model parameters
        self.utilities = {} # Will store utility functions
        
        # Prepare data and define model structure
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare the data for Biogeme processing."""
        # Create a copy to avoid modifying original data
        data = self.raw_data.copy()

        # Convert alt column to numeric using the mapping
        data['alt'] = data['alt'].map(self.MODE_TO_NUM)

        # Create availability variables (following Swissmetro example format)
        data['TRAIN_AV'] = (data['alt'] == 1).astype(int)
        data['CAR_AV'] = (data['alt'] == 2).astype(int)
        data['BUS_AV'] = (data['alt'] == 3).astype(int)
        data['AIR_AV'] = (data['alt'] == 4).astype(int)

        # Convert the data to be compatible with Biogeme
        self.database = db.Database('mode_choice', data)
    
        # Define variables as in Swissmetro example
        # Mode-specific variables
        self.vars = {
            'CHOICE': Variable('choice'),
            'TRAIN_AV': Variable('TRAIN_AV'),
            'CAR_AV': Variable('CAR_AV'),
            'BUS_AV': Variable('BUS_AV'),
            'AIR_AV': Variable('AIR_AV'),
            'COST': Variable('cost'),
            'IVT': Variable('ivt'),
            'OVT': Variable('ovt'),
            'INCOME': Variable('income'),
        }

    def specify_utilities(self):
        """Specify the utility functions for each mode."""
        # Define availability-based expressions like in Swissmetro
        TRAIN_COST = self.vars['COST'] * (self.vars['TRAIN_AV'] == 1)
        CAR_COST = self.vars['COST'] * (self.vars['CAR_AV'] == 1)
        BUS_COST = self.vars['COST'] * (self.vars['BUS_AV'] == 1)
        AIR_COST = self.vars['COST'] * (self.vars['AIR_AV'] == 1)

        TRAIN_TIME = (self.vars['IVT'] + self.vars['OVT']) * (self.vars['TRAIN_AV'] == 1)
        CAR_TIME = (self.vars['IVT'] + self.vars['OVT']) * (self.vars['CAR_AV'] == 1)
        BUS_TIME = (self.vars['IVT'] + self.vars['OVT']) * (self.vars['BUS_AV'] == 1)
        AIR_TIME = (self.vars['IVT'] + self.vars['OVT']) * (self.vars['AIR_AV'] == 1)

        # Parameters to be estimated
        ASC_CAR = Beta('ASC_CAR', 0, None, None, 1)
        ASC_TRAIN = Beta('ASC_TRAIN', 0, None, None, 0)
        ASC_BUS = Beta('ASC_BUS', 0, None, None, 0)
        ASC_AIR = Beta('ASC_AIR', 0, None, None, 0)
        B_TIME = Beta('B_TIME', 0, None, None, 0)
        B_COST = Beta('B_COST', 0, None, None, 0)

        # Store betas for later use
        self.betas = {
            'ASC_TRAIN': ASC_TRAIN,
            'ASC_CAR': ASC_CAR,
            'ASC_BUS': ASC_BUS,
            'ASC_AIR': ASC_AIR,
            'B_TIME': B_TIME,
            'B_COST': B_COST
        }

        # Utility functions following Swissmetro format
        V = {
            1: ASC_TRAIN + B_TIME * TRAIN_TIME + B_COST * TRAIN_COST,  # Train
            2: ASC_CAR + B_TIME * CAR_TIME + B_COST * CAR_COST,        # Car
            3: ASC_BUS + B_TIME * BUS_TIME + B_COST * BUS_COST,        # Bus
            4: ASC_AIR + B_TIME * AIR_TIME + B_COST * AIR_COST         # Air (base)
        }

        return V

    def estimate(self):
        """Estimate the model using Biogeme."""
        # Specify the utility functions
        V = self.specify_utilities()
        
        # Availability conditions
        av = {
            1: self.vars['TRAIN_AV'],
            2: self.vars['CAR_AV'],
            3: self.vars['BUS_AV'],
            4: self.vars['AIR_AV']
        }
        
        # Define the model following Swissmetro example
        logprob = models.loglogit(V, av, self.vars['CHOICE'])
        
        # Create and estimate the model
        self.model = bio.BIOGEME(self.database, logprob)
        self.model.modelName = "mode_choice_model"
        
        # Estimate
        self.results = self.model.estimate()
        print(self.results.shortSummary())
        
        return self.results

def main():
    # Load data
    loader = DatasetLoader()
    data = loader.load_dataset("modecanada_dataset")
    
    # Print data diagnostics
    print("\nData Shape:", data.shape)
    print("\nUnique values in 'alt' column:")
    print(data['alt'].unique())
    print("\nValue counts for 'alt' column:")
    print(data['alt'].value_counts())
    
    print("\nSample of raw data:")
    print(data.head(10))
    
    print("\nData types of columns:")
    print(data.dtypes)
    
    print("\nChecking for missing values:")
    print(data.isnull().sum())
    
    # Now proceed with model estimation if data looks correct
    model = ModeChoiceModel(data)
    results = model.estimate()
    
    # Print detailed results
    print("\nDetailed Parameter Estimates:")
    print(results.getEstimatedParameters())

if __name__ == "__main__":
    main()