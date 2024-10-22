"""File hybrid_function.py

:author: Michel Bierlaire
:date: Sun Jul  2 15:34:16 2023

Class for the hybrid function calculation, that mixed analytical hessian and BFGS updates
"""

import logging
import numpy as np

from biogeme_optimization.bounds import Bounds
from biogeme_optimization.function import FunctionData, FunctionToMinimize
from biogeme_optimization.bfgs import bfgs

logger: logging.Logger = logging.getLogger(__name__)


class HybridFunction:
    """Object in charge of evaluating the function and its derivative,
    where the second derivative matrix can be either calculated
    analytically or using BFGS updates.

    """

    def __init__(
        self,
        the_function: FunctionToMinimize,
        proportion_analytical_hessian: float,
        bounds: Bounds,
    ) -> None:
        """Constructor

        :param the_function: object to calculate the objective
            function and its derivatives.
        :type the_function: optimization.FunctionToMinimize

        :param proportion_analytical_hessian: proportion of the iterations where
                                  the true hessian is calculated. When
                                  not, the BFGS update is used. If
                                  1.0, it is used for all
                                  iterations. If 0.0, it is not used
                                  at all.
        :type proportion_analytical_hessian: float

        :param bounds: object describing the bound constraints. Used
            to project the gradient and check optimality.
        :type bounds: Bounds

        """
        self.the_function: FunctionToMinimize = the_function
        self.proportion: float = proportion_analytical_hessian
        self.bounds: Bounds = bounds
        self.number_of_analytical_hessians: int = 0
        self.number_of_matrices: int = 0
        self.number_of_numerical_issues: int = 0
        self.previous_x: np.ndarray | None = None
        self.previous_gradient: np.ndarray | None = None
        self.previous_hessian: np.ndarray | None = None

    def can_calculate_analytical(self) -> bool:
        """Determines if the analytical hessian can be calculated or not"""

        if self.proportion == 0:
            return False
        if self.proportion == 1:
            return True
        if self.number_of_matrices == 0:
            return True
        return (
            float(self.number_of_analytical_hessians) / float(self.number_of_matrices)
            <= self.proportion
        )

    def calculate_function(self, iterate: np.ndarray) -> float:
        """Calculates the canonical_value of the function

        :param iterate: values
        :type iterate: numpy.array
        """
        self.the_function.set_variables(iterate)
        return self.the_function.f()

    def calculate_function_and_derivatives(
        self, iterate: np.ndarray
    ) -> FunctionData | None:
        """Calculates the function, its gradient, and the hessian, or its approximation

        :param iterate: values
        :type iterate: numpy.array
        """
        if self.can_calculate_analytical():
            function_data = self._calculate_function_and_derivatives_analytical(iterate)
        else:
            function_data = self._calculate_function_and_derivatives_bfgs(iterate)
        if function_data is None:
            return None
        self.previous_x = iterate
        self.previous_gradient = function_data.gradient
        self.previous_hessian = function_data.hessian
        return function_data

    def _calculate_function_and_derivatives_bfgs(
        self, iterate: np.ndarray
    ) -> FunctionData | None:
        """Calculates the function, its gradient, and the BFGS
            approximation of the hessian

        :param iterate: values
        :type iterate: numpy.array

        """
        self.the_function.set_variables(iterate)
        if self.the_function.check_optimality(bounds=self.bounds):
            return None
        if self.previous_x is not None:
            if self.the_function.check_insufficient_progress(self.previous_x):
                return None
        function_data = self.the_function.f_g()
        if self.previous_x is None:
            hessian = np.eye(self.the_function.dimension())
        else:
            delta_x = iterate - self.previous_x
            delta_gradient = function_data.gradient - self.previous_gradient
            hessian = bfgs(self.previous_hessian, delta_x, delta_gradient)

        self.number_of_matrices += 1
        final_function_data = FunctionData(
            function=function_data.function,
            gradient=function_data.gradient,
            hessian=hessian,
        )
        return final_function_data

    def _calculate_function_and_derivatives_analytical(
        self, iterate: np.ndarray
    ) -> FunctionData | None:
        """Calculates the function, its gradient, and the hessian

        :param iterate: values
        :type iterate: numpy.array
        """
        self.the_function.set_variables(iterate)
        if self.the_function.check_optimality(bounds=self.bounds):
            return None
        if self.previous_x is not None:
            if self.the_function.check_insufficient_progress(self.previous_x):
                return None
        function_data = self.the_function.f_g_h()
        self.number_of_analytical_hessians += 1
        self.number_of_matrices += 1

        # If there is a numerical problem with the
        # Hessian, we apply BFGS

        if np.linalg.norm(function_data.hessian) > 1.0e100:
            self.number_of_numerical_issues += 1
            logger.warning(
                f'Numerical problem with the second '
                f'derivative matrix. '
                f'Norm = {np.linalg.norm(function_data.hessian)}. '
                f'Replaced by the BFGS approximation.'
            )
            return self._calculate_function_and_derivatives_bfgs(iterate)

        return function_data

    def message(self) -> str:
        """Calculates and reports the proportion of calculation of the analytical
            hessian among all matrices involved

        :return: message reporting the proportion.
        :rtype: str
        """
        if self.number_of_matrices == 0:
            return ''

        actual_prop = (
            100
            * float(self.number_of_analytical_hessians)
            / float(self.number_of_matrices)
            if self.number_of_matrices != 0
            else 0
        )

        numerical = (
            f' [{self.number_of_numerical_issues} numerical issues]'
            if self.number_of_numerical_issues != 0
            else ''
        )

        the_message = (
            f'{self.number_of_analytical_hessians}/{self.number_of_matrices} '
            f'= {actual_prop:.1f}%{numerical}'
        )
        return the_message
