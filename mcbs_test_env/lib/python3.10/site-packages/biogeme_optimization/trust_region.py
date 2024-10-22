"""File trust_region.py

:author: Michel Bierlaire
:date: Fri Jun 23 09:27:17 2023

Functions for trust region algorithms
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from biogeme_optimization.diagnostics import (
    OptimizationResults,
    DoglegDiagnostic,
    ConjugateGradientDiagnostic,
    Diagnostic,
)
from biogeme_optimization.exceptions import OptimizationError
from biogeme_optimization.algebra import schnabel_eskow_direction
from biogeme_optimization.bfgs import bfgs
from biogeme_optimization.function import FunctionToMinimize, FunctionData

logger = logging.getLogger(__name__)


class QuadraticModel(ABC):
    """Abstract class for the generation of the quadratic model."""

    def __init__(self, function: FunctionToMinimize):
        """Constructor

        :param function: the function to minimize
        :type function: FunctionToMinimize

        """
        self.the_function: FunctionToMinimize = function

    @abstractmethod
    def get_f_g_h(self, iterate: np.ndarray) -> FunctionData:
        """Obtain the vector g and the matrix h characterizing the
            model, as well as the canonical_value of the function

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        :return: function, gradient and hessian
        :rtype: tuple(float, numpy.array (nx1), numpy.array (nxn)
        """


class NewtonModel(QuadraticModel):
    """Call implementing the quadratic model based on the analytical hessian"""

    def get_f_g_h(self, iterate: np.ndarray) -> FunctionData:
        """Obtain the vector g and the matrix h characterizing the model

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        :return: function, gradient and hessian
        :rtype: tuple(float, numpy.array (nx1), numpy.array (nxn)
        """
        self.the_function.set_variables(iterate)
        evaluation = self.the_function.f_g_h()
        optimal = self.the_function.check_optimality()
        if optimal:
            return FunctionData(function=None, gradient=None, hessian=None)
        return evaluation


class BfgsModel(QuadraticModel):
    """Call implementing the quadratic model based on the BFGS approximation"""

    def __init__(
        self,
        function: FunctionToMinimize,
        first_approximation: np.ndarray | None = None,
    ):
        """Constructor

        :param first_approximation: first approximation of the Hessian
        :type first_approximation: numpy.array (n x n)

        :return: function, gradient and hessian
        :rtype: tuple(float, numpy.array (nx1), numpy.array (nxn)
        """
        super().__init__(function)

        if first_approximation is None:
            self.hessian_approx: np.ndarray = np.identity(self.the_function.dimension())
        else:
            if first_approximation.ndim != self.the_function.dimension():
                error_msg = (
                    f'Incompatible dimensions: {self.the_function.dimension()} '
                    f'and {first_approximation.ndim}'
                )
                raise OptimizationError(error_msg)
            self.hessian_approx = first_approximation
        self.last_iterate: np.ndarray | None = None
        self.last_gradient: np.ndarray | None = None

    def get_f_g_h(self, iterate: np.ndarray) -> FunctionData:
        """Obtain the vector g and the matrix h characterizing the model

        :param iterate: current iterate
        :type iterate: numpy.array (n x 1)

        """
        self.the_function.set_variables(iterate)
        evaluation = self.the_function.f_g()
        optimal = self.the_function.check_optimality()
        if optimal:
            return FunctionData(function=None, gradient=None, hessian=None)
        # The first time, there is nothing to update
        if self.last_iterate is not None:
            delta_x = iterate - self.last_iterate
            delta_g = evaluation.gradient - self.last_gradient
            # Update the approximation, if possible
            try:
                self.hessian_approx = bfgs(self.hessian_approx, delta_x, delta_g)
            except OptimizationError as err:
                logger.warning(err)
        self.last_iterate = iterate
        self.last_gradient = evaluation.gradient
        return FunctionData(
            function=evaluation.function,
            gradient=evaluation.gradient,
            hessian=self.hessian_approx,
        )


def trust_region_intersection(
    inside_direction: np.ndarray,
    crossing_direction: np.ndarray,
    radius: float,
    check_step: bool = True,
) -> float:
    """Calculates the intersection with the boundary of the trust region.

    Consider a trust region of radius :math:`\\delta`, centered at
    :math:`\\hat{x}`. Let :math:`x_c` be in the trust region, and
    :math:`d_c = x_c - \\hat{x}`, so that :math:`\\|d_c\\| \\leq
    \\delta`. Let :math:`x_d` be out of the trust region, and
    :math:`d_d = x_d - \\hat{x}`, so that :math:`\\|d_d\\| \\geq
    \\delta`.  We calculate :math:`\\lambda` such that

    .. math:: \\| d_c + \\lambda (d_d - d_c)\\| = \\delta

    :param inside_direction: xc - xhat.
    :type inside_direction: numpy.array

    :param crossing_direction: dd - dc
    :type crossing_direction: numpy.array

    :param radius: radius of the trust region.
    :type radius: float

    :param check_step: if True, an exception is raised if the canonical_value of
        the step is out of the interval [0,1]
    :type check_step: bool

    :return: :math:`\\lambda` such that :math:`\\| d_c +
              \\lambda (d_d - d_c)\\| = \\delta`

    :rtype: float

    """
    a = np.inner(crossing_direction, crossing_direction)
    b = 2 * np.inner(inside_direction, crossing_direction)
    c = np.inner(inside_direction, inside_direction) - radius**2
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0:
        raise OptimizationError('No intersection')
    step = (-b + np.sqrt(discriminant)) / (2 * a)
    if check_step:
        if step < 0:
            raise OptimizationError('Starting point outside trust region')
        if step > 1:
            raise OptimizationError('Target point inside trust region')
    return step


def cauchy_newton_dogleg(
    gradient: np.ndarray, hessian: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Calculate the Cauchy, the Newton and the dogleg points.

    The Cauchy point is defined as

    .. math:: d_c = - \\frac{\\nabla f(x)^T \\nabla f(x)}
                    {\\nabla f(x)^T \\nabla^2 f(x)
                    \\nabla f(x)} \\nabla f(x)

    The Newton point :math:`d_n` verifies Newton equation:

    .. math:: H_s d_n = - \\nabla f(x)

    where :math:`H_s` is a positive definite matrix generated with the
    method by `Schnabel and Eskow (1999)`_.

    The Dogleg point is

    .. math:: d_d = \\eta d_n

    where

    .. math:: \\eta = 0.2 + 0.8 \\frac{\\alpha^2}{\\beta |\\nabla f(x)^T d_n|}

    and :math:`\\alpha= \\nabla f(x)^T \\nabla f(x)`,
    :math:`\\beta=\\nabla f(x)^T \\nabla^2 f(x)\\nabla f(x).`

    :param gradient: gradient :math:`\\nabla f(x)`

    :type gradient: numpy.array

    :param hessian: hessian :math:`\\nabla^2 f(x)`

    :type hessian: numpy.array

    :return: tuple with Cauchy point, Newton point, Dogleg point. The
        Cauchy point in None if the curvature is negative along the
        gradient direction. The Newton and the Dogleg directions are
        None if the curvature is negative along the Newton direction.
    :rtype: numpy.array, numpy.array, numpy.array

    """
    alpha = np.inner(gradient, gradient)
    beta = np.inner(gradient, hessian @ gradient)
    if beta > 0:
        d_cauchy = -(alpha / beta) * gradient
    else:
        d_cauchy = None
    try:
        d_newton = schnabel_eskow_direction(
            gradient=gradient, hessian=hessian, check_convexity=True
        )
    except OptimizationError:
        return d_cauchy, None, None
    eta = 0.2 + (0.8 * alpha * alpha / (beta * abs(np.inner(gradient, d_newton))))
    return d_cauchy, d_newton, eta * d_newton


def dogleg(
    gradient: np.ndarray, hessian: np.ndarray, radius: float
) -> tuple[np.ndarray, DoglegDiagnostic]:
    """
    Find an approximation of the trust region sub-problem using
    the dogleg method

    :param gradient: gradient of the quadratic model.
    :type gradient: numpy.array
    :param hessian: hessian of the quadratic model.
    :type hessian: numpy.array
    :param radius: radius of the trust region.
    :type radius: float

    :return: d, diagnostic where

          - d is an approximate solution of the trust region subproblem
          - diagnostic is the nature of the solution:

             * NEGATIVE_CURVATURE if negative curvature along Cauchy direction
               (i.e. along the gradient)
             * CAUCHY if Cauchy step
             * PARTIAL_CAUCHY partial Cauchy step
             * NEWTON if Newton step
             * PARTIAL_NEWTON if partial Newton step
             * DOGLEG if Dogleg

    :rtype: numpy.array, int
    """

    # Check if the inputs are valid NumPy arrays
    if not isinstance(gradient, np.ndarray) or not isinstance(hessian, np.ndarray):
        raise OptimizationError('gradient and hessian must be NumPy arrays')
    if gradient.ndim != 1 or hessian.ndim != 2:
        raise OptimizationError(
            'gradient must be a 1-dimensional array and '
            'hessian must be a 2-dimensional array'
        )
    if gradient.shape[0] != hessian.shape[0] or hessian.shape[0] != hessian.shape[1]:
        raise OptimizationError("gradient and hessian must have compatible dimensions")

    cauchy_dir, newton_dir, dogleg_dir = cauchy_newton_dogleg(gradient, hessian)

    # Check if the model is convex along the gradient direction
    if cauchy_dir is None:
        d_star = -radius * gradient / np.linalg.norm(gradient)
        return d_star, DoglegDiagnostic.NEGATIVE_CURVATURE

    alpha = np.inner(gradient, gradient)
    beta = np.inner(gradient, hessian @ gradient)
    norm_dc = alpha * np.sqrt(alpha) / beta
    if norm_dc >= radius:
        # The Cauchy point is outside the trust
        # region. We move along the Cauchy
        # direction until the border of the trust
        # region.
        d_star = (radius / norm_dc) * cauchy_dir
        return d_star, DoglegDiagnostic.PARTIAL_CAUCHY

    # Check the convexity of the model along Newton direction
    if newton_dir is None:
        # Return the Cauchy point
        return cauchy_dir, DoglegDiagnostic.CAUCHY

    # Compute Newton point
    norm_dn = np.linalg.norm(newton_dir)

    if norm_dn <= radius:
        # Newton point is inside the trust region
        return newton_dir, DoglegDiagnostic.NEWTON

    # Compute the dogleg point

    eta = 0.2 + (0.8 * alpha * alpha / (beta * abs(np.inner(gradient, newton_dir))))

    partial_newton = eta * norm_dn

    if partial_newton <= radius:
        # Dogleg point is inside the trust region
        d_star = (radius / norm_dn) * newton_dir
        return d_star, DoglegDiagnostic.PARTIAL_NEWTON

    # Between Cauchy and dogleg
    crossing_direction = dogleg_dir - cauchy_dir
    lbd = trust_region_intersection(cauchy_dir, crossing_direction, radius)
    d_star = cauchy_dir + lbd * crossing_direction
    return d_star, DoglegDiagnostic.DOGLEG


def truncated_conjugate_gradient(
    gradient: np.ndarray, hessian: np.ndarray, radius: float, tol: float = 1.0e-6
) -> tuple[np.ndarray, ConjugateGradientDiagnostic]:
    """Find an approximation of the trust region sub-problem using the
    truncated conjugate gradient method

    :param gradient: gradient of the quadratic model.
    :type gradient: numpy.array

    :param hessian: hessian of the quadratic model.
    :type hessian: numpy.array

    :param radius: radius of the trust region.
    :type radius: float

    :param tol: tolerance on the norm of the gradient, used as a stopping criterion
    :type tol: float

    :return: d, diagnostic, where

          - d is the approximate solution of the trust region sub-problem,
          - diagnostic is the nature of the solution:

            * CONVERGENCE for convergence,
            * OUT_OF_TRUST_REGION if out of the trust region,
            * NEGATIVE_CURVATURE if negative curvature detected.
            * NUMERICAL_PROBLEM if a numerical problem has been encountered

    :rtype: numpy.array, ConjugateGradientDiagnostic

    """
    dimension = len(gradient)
    xk = np.zeros(dimension)
    gk = gradient
    dk = -gk
    for _ in range(dimension):
        try:
            curvature = np.inner(dk, hessian @ dk)
            if curvature <= 0:
                # Negative curvature has been detected
                diagnostic = ConjugateGradientDiagnostic.NEGATIVE_CURVATURE
                step = trust_region_intersection(xk, dk, radius, check_step=False)
                solution = xk + step * dk
                return solution, diagnostic
            alphak = -np.inner(dk, gk) / curvature
            xkp1 = xk + alphak * dk
            if np.isnan(xkp1).any() or np.linalg.norm(xkp1) > radius:
                # Out of the trust region
                diagnostic = ConjugateGradientDiagnostic.OUT_OF_TRUST_REGION
                step = trust_region_intersection(xk, dk, radius, check_step=False)
                solution = xk + step * dk
                return solution, diagnostic
            xk = xkp1
            gkp1 = hessian @ xk + gradient
            betak = np.inner(gkp1, gkp1) / np.inner(gk, gk)
            dk = -gkp1 + betak * dk
            gk = gkp1
            if np.linalg.norm(gkp1) <= tol:
                diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
                step = xk
                return step, diagnostic
        except ValueError:
            # Numerical problem. We follow the last direction until
            # the border of the trust region
            diagnostic = ConjugateGradientDiagnostic.NUMERICAL_PROBLEM
            step = trust_region_intersection(xk, dk, radius, check_step=False)
            solution = xk + step * dk
            return solution, diagnostic
    diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
    step = xk
    return step, diagnostic


def minimization_with_trust_region(
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    quadratic_model: QuadraticModel,
    solving_trust_region_subproblem: Callable[
        [np.ndarray, np.ndarray, float], tuple[np.ndarray, Diagnostic]
    ],
    maxiter: int = 1000,
    initial_radius: float = 1.0,
    eta1: float = 0.01,
    eta2: float = 0.9,
):
    """Minimization method with trust region

    :param the_function: object to calculate the objective function and its derivatives.
    :type the_function: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param quadratic_model: object in charge of generating the quadratic model
    :type quadratic_model: QuadraticModel

    :param solving_trust_region_subproblem: function in charge of
        solving the trust region sub-problem


    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.
    :type maxiter: int

    :param initial_radius: initial radius of the trust region.
    :type initial_radius: float

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    """
    xk = starting_point
    the_output: FunctionData = quadratic_model.get_f_g_h(xk)
    value_iterate = the_output.function
    g = the_output.gradient
    h = the_output.hessian
    radius = initial_radius
    max_delta = np.finfo(float).max
    min_delta = np.finfo(float).eps
    for k in range(maxiter):
        # If the iterate is optimal, the returned canonical_value is None
        if value_iterate is None:
            messages = the_function.messages
            messages['Number of iterations'] = k
            optimization_results = OptimizationResults(
                solution=xk, messages=messages, convergence=True
            )
            break
        step, _ = solving_trust_region_subproblem(g, h, radius)
        candidate = xk + step
        the_function.set_variables(candidate)
        # Calculate the canonical_value of the function
        value_candidate = the_function.f()
        if value_candidate >= value_iterate:
            radius = np.linalg.norm(step) / 2.0
            status = '-'
            continue

        num = value_iterate - value_candidate
        denominator = -np.inner(step, g) - 0.5 * np.inner(step, h @ step)
        rho = num / denominator
        if rho < eta1:
            # Failure: reduce the trust region
            radius = np.linalg.norm(step) / 2.0
            status = '-'
            continue

        # Candidate accepted
        xk = candidate
        the_output: FunctionData = quadratic_model.get_f_g_h(xk)
        value_iterate = the_output.function
        g = the_output.gradient
        h = the_output.hessian
        if rho >= eta2:
            # Enlarge the trust region
            radius = min(2 * radius, max_delta)
            status = '++'
        else:
            status = '+'
        if radius <= min_delta:
            messages = the_function.messages
            messages['Cause of termination'] = f'Trust region is too small: {radius}'
            optimization_results = OptimizationResults(
                solution=xk, messages=messages, convergence=False
            )
            break
    else:
        messages = the_function.messages
        messages['Cause of termination'] = (
            f'Maximum number of iterations reached: {maxiter}'
        )
        optimization_results = OptimizationResults(
            solution=xk, messages=messages, convergence=False
        )

    return optimization_results


def newton_trust_region(
    *,
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    use_dogleg: bool = False,
    maxiter: int = 1000,
    initial_radius: float = 1.0,
    eta1: float = 0.01,
    eta2: float = 0.9,
):
    """Newton method with trust region

    :param the_function: object to calculate the objective function and its derivatives.
    :type the_function: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param use_dogleg: True if the trust region sub-problem is solved using
        the dogleg method. False is it is solved with the truncated
        conjugate gradient algorithm.
    :type use_dogleg: bool

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.
    :type maxiter: int

    :param initial_radius: initial radius of the trust region. Default: 100.
    :type initial_radius: float

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    """
    the_model = NewtonModel(the_function)

    if use_dogleg:
        return minimization_with_trust_region(
            the_function,
            starting_point,
            the_model,
            dogleg,
            maxiter,
            initial_radius,
            eta1,
            eta2,
        )
    return minimization_with_trust_region(
        the_function,
        starting_point,
        the_model,
        truncated_conjugate_gradient,
        maxiter,
        initial_radius,
        eta1,
        eta2,
    )


def bfgs_trust_region(
    *,
    the_function: FunctionToMinimize,
    starting_point: np.ndarray,
    init_bfgs: np.ndarray | None = None,
    use_dogleg: bool = False,
    maxiter: int = 1000,
    initial_radius: float = 1.0,
    eta1: float = 0.01,
    eta2: float = 0.9,
):
    """BFGS method with trust region

    :param the_function: object to calculate the objective function and its derivatives.
    :type the_function: optimization.functionToMinimize

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param init_bfgs: matrix used to initialize BFGS. If None, the
                     identity matrix is used. Default: None.
    :type init_bfgs: numpy.array

    :param use_dogleg: True if the trust region subproblem is solved using
        the dogleg method. False is it is solved with the truncated
        conjugate gradient algorithm.
    :type use_dogleg: bool

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.
    :type maxiter: int

    :param initial_radius: initial radius of the trust region. Default: 100.
    :type initial_radius: float

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    """
    the_model = BfgsModel(the_function, init_bfgs)

    if use_dogleg:
        return minimization_with_trust_region(
            the_function,
            starting_point,
            the_model,
            dogleg,
            maxiter,
            initial_radius,
            eta1,
            eta2,
        )
    return minimization_with_trust_region(
        the_function,
        starting_point,
        the_model,
        truncated_conjugate_gradient,
        maxiter,
        initial_radius,
        eta1,
        eta2,
    )
