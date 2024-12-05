"""Base class for discrete choice models"""

from abc import ABC, abstractmethod
import biogeme.biogeme_logging as blog
from biogeme.database import Database
import numpy as np
import pandas as pd

class BaseDiscreteChoiceModel(ABC):
    """Base class for all discrete choice models."""
    
    def __init__(self, data):
        """Initialize base model structure."""
        self.logger = blog.get_screen_logger(level=blog.INFO)
        self.database = Database('choice_model', data)
        self.test_database = Database('choice_model', data)
        self.results = None
        
        # Initialize metrics attributes
        self.final_ll = None
        self.rho_squared = None
        self.rho_squared_bar = None
        self.vot_walking = None
        self.vot_cycling = None
        self.vot_pt = None
        self.vot_driving = None
        self.market_share_accuracy = None
        self.choice_accuracy = None
        self.actual_shares = None
        self.predicted_shares = None
        self.confusion_matrix = None
    
    @abstractmethod
    def estimate(self):
        """Estimate model parameters. Must be implemented by subclasses."""
        pass
    
    def get_metrics(self):
        """Get standard metrics for model comparison."""
        if self.results is None:
            raise RuntimeError("Model must be estimated before getting metrics")
            
        metrics = {
            'n_parameters': len(self.results.data.betaValues),
            'n_observations': self.results.data.numberOfObservations,
        }
        
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
