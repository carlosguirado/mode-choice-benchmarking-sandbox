"""File diagnostics.py

:author: Michel Bierlaire
:date: Wed Jun 28 16:17:43 2023

Classes for algorithm diagnostics
"""

import numpy as np
from enum import Enum, auto
from typing import NamedTuple


class OptimizationResults(NamedTuple):
    solution: np.ndarray
    messages: dict[str, str]
    convergence: bool


class Diagnostic(Enum):
    pass


class DoglegDiagnostic(Diagnostic):
    """Possible outcomes of the dogleg method"""

    NEGATIVE_CURVATURE = auto()
    CAUCHY = auto()
    PARTIAL_CAUCHY = auto()
    NEWTON = auto()
    PARTIAL_NEWTON = auto()
    DOGLEG = auto()


class ConjugateGradientDiagnostic(Diagnostic):
    """Possible outcomes of the conjugate gradient method"""

    CONVERGENCE = auto()
    OUT_OF_TRUST_REGION = auto()
    NEGATIVE_CURVATURE = auto()
    NUMERICAL_PROBLEM = auto()
