"""File simple_bounds.py

:author: Michel Bierlaire
:date: Wed Jun 28 11:40:55 2023

Functions for the trust region algorithm with simple bounds
"""

import logging
import numpy as np

from biogeme_optimization.bounds import Bounds
from biogeme_optimization.exceptions import OptimizationError
from biogeme_optimization.function import FunctionToMinimize
from biogeme_optimization.hybrid_function import HybridFunction
from biogeme_optimization.format import Column, FormattedColumns
from biogeme_optimization.diagnostics import OptimizationResults

logger = logging.getLogger(__name__)


def simple_bounds_newton_algorithm(
    *,
    the_function: FunctionToMinimize,
    bounds: Bounds,
    starting_point: np.ndarray,
    variable_names: list[str] | None = None,
    proportion_analytical_hessian: float = 1.0,
    first_radius: float = 1.0,
    conjugate_gradient_tol: float = np.finfo(np.float64).eps ** 0.3333,
    maxiter: int = 1000,
    eta1: float = 0.01,
    eta2: float = 0.9,
    enlarging_factor: float = 10.0,
) -> OptimizationResults:
    """Trust region algorithm for problems with simple bounds

    :param the_function: object to calculate the objective function and its derivatives.
    :type the_function: optimization.functionToMinimize

    :param bounds: bounds on the unsorted_set_of_variables
    :type bounds: class Bounds

    :param starting_point: starting point
    :type starting_point: numpy.array

    :param variable_names: names of the unsorted_set_of_variables, for reporting purposes
    :type variable_names: list(str)

    :param proportion_analytical_hessian: proportion of the iterations where
                                  the true hessian is calculated. When
                                  not, the BFGS update is used. If
                                  1.0, it is used for all
                                  iterations. If 0.0, it is not used
                                  at all.
    :type proportion_analytical_hessian: float

    :param first_radius: initial radius of the trust region. Default: 100.
    :type first_radius: float

    :param conjugate_gradient_tol: the conjugate gradient algorithm stops when this
                  precision is reached.  Default:
                  :math:`\\varepsilon^{\\frac{1}{3}}`

    :type conjugate_gradient_tol: float

    :param maxiter: the algorithm stops if this number of iterations
                    is reached. Default: 1000.

    :type maxiter: int

    :param eta1: threshold for failed iterations. Default: 0.01.
    :type eta1: float

    :param eta2: threshold for very successful iterations. Default 0.9.
    :type eta2: float

    :param enlarging_factor: if an iteration is very successful, the
                            radius of the trust region is multiplied
                            by this factor. Default 10.
    :type enlarging_factor: float

    :return: named tuple:

            - solution is the solution found,
            - message is a dictionary reporting various aspects
              related to the run of the algorithm.
            - convergence is a bool which is True if the algorithm has converged

    :rtype: OptimizationResults

    :raises biogeme.exceptions.BiogemeError: if the dimensions of the
             matrix initBfgs do not match the length of starting_point.

    """
    dimension = bounds.dimension
    if len(starting_point) != dimension:
        raise OptimizationError(
            f'Incompatible size:' f' {len(starting_point)} and {dimension}'
        )

    if not bounds.feasible(starting_point):
        logger.warning(
            'Initial point not feasible. '
            'It will be projected onto the feasible domain.'
        )

    if the_function.needs_reset():
        warning_msg = (
            'It seems that the function has been previously used in another '
            'context, as previously calculated values are stored. It will not '
            'have any consequence on the algorithm. It will just modify the '
            'statistics about function evaluation, as these previous calculations '
            'will be accounted for. It is was not intended, call the method '
            'reset() of the function before using it.'
        )
        logger.warning(warning_msg)

    # Establish the name of the algorithm, for reporting

    if proportion_analytical_hessian < 0 or proportion_analytical_hessian > 1:
        error_msg = (
            'Proportion must be between 0 and 1, not {proportion_analytical_hessian}'
        )
        raise OptimizationError(error_msg)

    if proportion_analytical_hessian == 1.0:
        algo = 'Newton with trust region for simple bound constraints'
    elif proportion_analytical_hessian == 0.0:
        algo = 'BFGS with trust region for simple bound constraints'
    else:
        algo = (
            f'Hybrid Newton [{100*proportion_analytical_hessian}%] with trust '
            f'region for simple bound constraints'
        )

    columns = [Column(title='Iter.', width=5)]

    if variable_names is not None:
        columns += [Column(title=name[:15], width=15) for name in variable_names]
    columns += [
        Column(title='Function', width=12),
        Column(title='Relgrad', width=10),
        Column(title='Radius', width=8),
        Column(title='Rho', width=8),
        Column(title=' ', width=4),
    ]

    the_formatter = FormattedColumns(columns)

    the_hybrid_function = HybridFunction(
        the_function, proportion_analytical_hessian, bounds=bounds
    )

    iterate = bounds.project(starting_point)
    current_function = the_hybrid_function.calculate_function_and_derivatives(iterate)
    if current_function is None:
        # The starting point is optimal
        messages = the_function.messages
        messages['Algorithm'] = algo
        messages['Number of iterations'] = '0'
        return OptimizationResults(
            solution=iterate, messages=messages, convergence=True
        )

    radius = first_radius
    max_delta = 1.0e10
    min_delta = np.sqrt(np.finfo(float).eps)
    rho = 0.0
    status = ''

    logger.info(the_formatter.formatted_title())

    for k in range(maxiter):

        def logmessage() -> None:
            """Send messages to the logger"""
            values_to_report = [k]
            if variable_names is not None:
                values_to_report += list(iterate)
            values_to_report += [
                current_function.function,
                the_function.relative_gradient_norm,
                float(radius),
                rho,
                status,
            ]
            logger.info(the_formatter.formatted_row(values_to_report))

        if (
            np.isnan(current_function.function).any()
            or np.isnan(current_function.gradient).any()
            or np.isnan(current_function.hessian).any()
        ):
            radius /= 2.0
            if radius <= min_delta:
                messages = the_function.messages
                messages['Algorithm'] = algo
                message = f'Trust region is too small: {radius}'
                messages['Cause of termination'] = message
                messages['Number of iterations'] = f'{k + 1}'
                messages['Proportion of Hessian calculation'] = (
                    the_hybrid_function.message()
                )
                logmessage()
                return OptimizationResults(
                    solution=iterate, messages=messages, convergence=False
                )

            status = '-'
            logmessage()
            continue
        try:
            # Solve the quadratic problem in the subspace defined by the GCP
            candidate, _ = bounds.truncated_conjugate_gradient_subspace(
                iterate=iterate,
                gradient=current_function.gradient,
                hessian=current_function.hessian,
                radius=radius,
                tol=conjugate_gradient_tol,
            )
            if np.isnan(candidate).any():
                radius /= 2.0
                if radius <= min_delta:
                    messages = the_function.messages
                    messages['Algorithm'] = algo
                    message = f'Trust region is too small: {radius}'
                    messages['Cause of termination'] = message
                    messages['Number of iterations'] = f'{k+1}'
                    messages['Proportion of Hessian calculation'] = (
                        the_hybrid_function.message()
                    )
                    logmessage()
                    return OptimizationResults(
                        solution=iterate, messages=messages, convergence=False
                    )

                status = '-'
                logmessage()
                continue

            try:
                function_value_candidate = the_hybrid_function.calculate_function(
                    candidate
                )
                failed = not np.isfinite(function_value_candidate)
                step = None
                if not failed:
                    num = current_function.function - function_value_candidate
                    step = candidate - iterate
                    denom = -np.inner(step, current_function.gradient) - 0.5 * np.inner(
                        step, current_function.hessian @ step
                    )
                    rho = num / denom
                    failed = rho < eta1
                if failed:
                    if step is None:
                        radius /= 2.0
                    else:
                        radius = min(radius / 2.0, np.linalg.norm(step, np.inf) / 2.0)
                    if radius <= min_delta:
                        messages = the_function.messages
                        messages['Algorithm'] = algo
                        message = f'Trust region is too small: {radius}'
                        messages['Cause of termination'] = message
                        messages['Number of iterations'] = f'{k+1}'
                        messages['Proportion of Hessian calculation'] = (
                            the_hybrid_function.message()
                        )
                        logmessage()
                        return OptimizationResults(
                            solution=iterate, messages=messages, convergence=False
                        )

                    status = '-'
                    logmessage()
                    continue
            except RuntimeError as e:
                logger.warning(e)
                radius /= 2.0
                if radius <= min_delta:
                    messages = the_function.messages
                    messages['Algorithm'] = algo
                    message = f'Trust region is too small: {radius}'
                    messages['Cause of termination'] = message
                    messages['Number of iterations'] = f'{k+1}'
                    messages['Proportion of Hessian calculation'] = (
                        the_hybrid_function.message()
                    )
                    logmessage()
                    return OptimizationResults(
                        solution=iterate, messages=messages, convergence=False
                    )
                failed = True
                status = '-'
                logmessage()
                continue

            # Candidate accepted
            try:
                candidate_function = (
                    the_hybrid_function.calculate_function_and_derivatives(candidate)
                )
            except OptimizationError:
                # Failure: reduce the trust region
                radius = min(radius / 2.0, np.linalg.norm(step, np.inf) / 2.0)
                if radius <= min_delta:
                    messages = the_function.messages
                    messages['Algorithm'] = algo
                    message = f'Trust region is too small: {radius}'
                    messages['Cause of termination'] = message
                    messages['Number of iterations'] = f'{k+1}'
                    messages['Proportion of Hessian calculation'] = (
                        the_hybrid_function.message()
                    )
                    logmessage()
                    return OptimizationResults(
                        solution=iterate, messages=messages, convergence=False
                    )
                status = '-'
                logmessage()

                continue

            if candidate_function is None:
                # the_matrix stopping criterion has been detected
                messages = the_function.messages
                messages['Algorithm'] = algo
                messages['Number of iterations'] = f'{k+1}'
                messages['Proportion of Hessian calculation'] = (
                    the_hybrid_function.message()
                )
                logmessage()
                return OptimizationResults(
                    solution=candidate, messages=messages, convergence=True
                )

            iterate = candidate
            current_function = candidate_function

            if rho >= eta2:
                # Enlarge the trust region
                radius = min(enlarging_factor * radius, max_delta)
                status = '++'
            else:
                status = '+'

            logmessage()
        except KeyboardInterrupt:
            warning_msg = (
                'You have pressed Ctrl-C. The iterations of the optimization algorithms '
                'are interrupted, but the script will continue to run. If you want to interrupt the script, '
                'press Ctrl-C again'
            )
            logger.warning(warning_msg)
            messages = the_function.messages
            messages['Algorithm'] = algo
            messages['Number of iterations'] = f'{k + 1}'
            message = 'Iterations interrupted by the user'
            messages['Cause of termination'] = message
            messages['Proportion of Hessian calculation'] = (
                the_hybrid_function.message()
            )
            logmessage()
            return OptimizationResults(
                solution=iterate, messages=messages, convergence=False
            )

    the_function.calculate_relative_projected_gradient(bounds)
    messages = the_function.messages
    messages['Algorithm'] = algo
    message = f'Maximum number of iterations reached: {maxiter}'
    messages['Cause of termination'] = message
    messages['Number of iterations'] = f'{maxiter}'
    messages['Proportion of Hessian calculation'] = the_hybrid_function.message()
    return OptimizationResults(solution=iterate, messages=messages, convergence=False)
