"""File linesearch.py

:author: Michel Bierlaire
:date: Thu Jun 22 15:12:21 2023

Functions for line search algorithms
"""

from abc import ABC, abstractmethod
import logging
import numpy as np
from biogeme_optimization.diagnostics import OptimizationResults
from biogeme_optimization.exceptions import OptimizationError
from biogeme_optimization.bfgs import inverse_bfgs
from biogeme_optimization.algebra import schnabel_eskow_direction
from biogeme_optimization.function import FunctionToMinimize

logger: logging.Logger = logging.getLogger(__name__)


class DirectionManagement(ABC):
    """Abstract class for the generation of a descent direction."""

    def __init__(self, function: FunctionToMinimize):
        """Constructor

        :param function: the function to minimize

        """
        self.the_function = function

    @abstractmethod
    def get_direction(self, iterate: np.ndarray) -> np.ndarray | None:
        """Obtain a descent direction

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        """


class NewtonDirection(DirectionManagement):
    """Class for the generation of the Newton direction."""

    def get_direction(self, iterate: np.ndarray) -> np.ndarray | None:
        """Update the data necessary to calculate the direction.

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        :return: descent direction, or None if the current iterate is optimal
        :rtype: numpy.array (n x 1)
        """
        self.the_function.set_variables(iterate)
        evaluation = self.the_function.f_g_h()
        optimal = self.the_function.check_optimality()
        if optimal:
            return None
        direction = schnabel_eskow_direction(evaluation.gradient, evaluation.hessian)
        return direction


class InverseBfgsDirection(DirectionManagement):
    """Class for the generation of the inverse BFGS direction."""

    def __init__(
        self,
        function: FunctionToMinimize,
        first_inverse_approximation: np.ndarray | None = None,
    ) -> None:
        """Constructor

        :param first_inverse_approximation: first inverse approximation of the Hessian
        :type first_inverse_approximation: numpy.array (n x n)
        """
        super().__init__(function)

        if first_inverse_approximation is None:
            self.inverse_hessian_approx = np.identity(self.the_function.dimension())
        else:
            if first_inverse_approximation.ndim != self.the_function.dimension():
                error_msg = (
                    f'Incompatible dimensions: {self.the_function.dimension()} '
                    f'and {first_inverse_approximation.ndim}'
                )
                raise OptimizationError(error_msg)
            self.inverse_hessian_approx = first_inverse_approximation
        self.last_iterate = None
        self.last_gradient = None

    def get_direction(self, iterate: np.ndarray) -> np.ndarray | None:
        """Obtain a descent direction

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        :return: descent direction, or None if the current iterate is optimal.
        """
        self.the_function.set_variables(iterate)
        evaluation = self.the_function.f_g()
        optimal = self.the_function.check_optimality()
        if optimal:
            return None
        # The first time, there is nothing to update
        if self.last_iterate is not None:
            delta_x = iterate - self.last_iterate
            delta_g = evaluation.gradient - self.last_gradient
            # Update the approximation, if possible
            try:
                self.inverse_hessian_approx = inverse_bfgs(
                    self.inverse_hessian_approx, delta_x, delta_g
                )
            except OptimizationError as e:
                logger.warning(e)
        self.last_iterate = iterate
        self.last_gradient = evaluation.gradient

        # Calculate and return the direction
        direction = -self.inverse_hessian_approx @ evaluation.gradient
        return direction


def line_search(
    fct: FunctionToMinimize,
    iterate: np.ndarray,
    descent_direction: np.ndarray,
    alpha0: float = 1.0,
    beta1: float = 1.0e-4,
    beta2: float = 0.99,
    lbd: float = 2.0,
    maxiter: int = 1000,
) -> float:
    """
    Calculate a step along a direction that satisfies both Wolfe conditions

    :param fct: object to calculate the objective function and its derivatives.
    :type fct: function_to_minimize

    :param iterate: current iterate.
    :type iterate: numpy.array

    :param descent_direction: descent direction.
    :type descent_direction: numpy.array

    :param alpha0: first step to test.
    :type alpha0: float

    :param beta1: parameter of the first Wolfe condition.
    :type beta1: float

    :param beta2: parameter of the second Wolfe condition.
    :type beta2: float

    :param lbd: expansion factor for a short step.
    :type lbd: float

    :param maxiter: the algorithm stops if this number of iterations
        is reached. Default: 1000

    :return: a step verifying both Wolfe conditions
    :rtype: float

    :raises OptimizationError: if ``lbd`` :math:`\\leq` 1
    :raises OptimizationError: if ``alpha0`` :math:`\\leq` 0
    :raises OptimizationError: if ``beta1`` :math:`\\geq` beta2
    :raises OptimizationError: if ``descent_direction`` is not a descent
                                             direction

    """
    if lbd <= 1:
        raise OptimizationError(f'lambda is {lbd} and must be > 1')
    if alpha0 <= 0:
        raise OptimizationError(f'alpha0 is {alpha0} and must be > 0')
    if beta1 >= beta2:
        error_msg = (
            f'Incompatible Wolfe cond. parameters: '
            f'beta1= {beta1} is greater than '
            f'beta2={beta2}'
        )
        raise OptimizationError(error_msg)

    fct.set_variables(iterate)
    evaluation = fct.f_g()
    deriv = np.inner(evaluation.gradient, descent_direction)

    if deriv >= 0:
        raise OptimizationError(
            f'descent_direction is not a descent direction: {deriv} >= 0'
        )

    alpha = alpha0
    alpha_left = 0
    alpha_right = np.inf
    for _ in range(maxiter):
        candidate = iterate + alpha * descent_direction
        fct.set_variables(candidate)
        candidate_evaluation = fct.f_g()
        # First Wolfe condition violated?
        if candidate_evaluation.function > evaluation.function + alpha * beta1 * deriv:
            alpha_right = alpha
            alpha = (alpha_left + alpha_right) / 2.0
        # Second Wolfe condition violated?
        elif np.inner(candidate_evaluation.gradient, descent_direction) < beta2 * deriv:
            alpha_left = alpha
            if alpha_right == np.inf:
                alpha *= lbd
            else:
                alpha = (alpha_left + alpha_right) / 2.0
        else:
            return alpha
    else:
        raise OptimizationError(
            f'Line search algorithm could not find a step verifying both Wolfe '
            f'conditions after {maxiter} iterations.'
        )


def minimization_with_line_search(
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    direction_management: DirectionManagement,
    maxiter: int = 1000,
) -> OptimizationResults:
    """Minimization method with inexact line search (Wolfe conditions)

    :param the_function: function to minimize
    :type the_function: FunctionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param direction_management: object in charge of providing the
        descent directions and to check optimality
    :type direction_management: DirectionManagement

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000
    :type maxiter: int

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    :raises OptimizationError:

    """
    xk = starting_point

    for k in range(maxiter):
        direction = direction_management.get_direction(xk)
        if direction is None:
            # The current iterate is optimal
            messages = the_function.messages
            messages['Number of iterations'] = k
            optimization_results = OptimizationResults(
                solution=xk,
                messages=messages,
                convergence=True,
            )
            break
        alpha = line_search(the_function, xk, direction)
        delta = alpha * direction
        xk = xk + delta
    else:
        messages = the_function.messages
        messages['Cause of termination'] = (
            f'Maximum number of iterations reached: {maxiter}'
        )

        optimization_results = OptimizationResults(
            solution=xk,
            messages=messages,
            convergence=False,
        )
    return optimization_results


def newton_line_search(
    *,
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    maxiter: int = 1000,
) -> OptimizationResults:
    """
    Newton method with inexact line search (Wolfe conditions)

    :param the_function: object to calculate the objective function and its derivatives.
    :type the_function: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000
    :type maxiter: int

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    """
    messages = {'Algorithm': 'Unconstrained Newton with line search'}
    newton_direction = NewtonDirection(the_function)
    optimization_results = minimization_with_line_search(
        the_function, starting_point, newton_direction, maxiter
    )
    messages.update(optimization_results.messages)
    return OptimizationResults(
        solution=optimization_results.solution,
        messages=messages,
        convergence=optimization_results.convergence,
    )


def bfgs_line_search(
    *,
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    init_bfgs: np.ndarray | None = None,
    maxiter: int = 1000,
) -> OptimizationResults:
    """BFGS method with inexact line search (Wolfe conditions)

    :param the_function: object to calculate the objective function and its derivatives.
    :type the_function: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param init_bfgs: matrix used to initialize BFGS. If None, the
                     identity matrix is used. Default: None.
    :type init_bfgs: numpy.array

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000
    :type maxiter: int

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResult

    """
    messages = {'Algorithm': 'Inverse BFGS with line search'}
    bfgs_direction = InverseBfgsDirection(
        function=the_function, first_inverse_approximation=init_bfgs
    )
    optimization_results = minimization_with_line_search(
        the_function, starting_point, bfgs_direction, maxiter
    )
    messages.update(optimization_results.messages)
    return OptimizationResults(
        solution=optimization_results.solution,
        messages=messages,
        convergence=optimization_results.convergence,
    )
