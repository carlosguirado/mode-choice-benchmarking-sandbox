"""File bounds.py.

:author: Michel Bierlaire
:date: Wed Jun 21 09:56:40 2023

Class in charge of the management of bound constraints
"""

from __future__ import annotations
import logging
import numpy as np

from biogeme_optimization.exceptions import OptimizationError
from biogeme_optimization.diagnostics import ConjugateGradientDiagnostic

logger: logging.Logger = logging.getLogger(__name__)


def inner_product_subspace(
    vector_a: np.ndarray, vector_b: np.ndarray, subspace_indices
) -> float:
    """Given two input vectors `vector_a` and `vector_b`, this
    function calculates the inner product of the entries in `vector_a`
    and `vector_b` that correspond to the provided `subspace_indices`.

    :param vector_a: The first input vector.
    :type vector_a: numpy.ndarray

    :param vector_b: The second input vector.
    :type vector_b: numpy.ndarray

    :param subspace_indices: The indices of the selected entries.
    :type subspace_indices: iterable

    :return: The inner product of the selected entries.
    :rtype: float

    :raises IndexError: If the `subspace_indices` are out of bounds
        for `vector_a` or `vector_b`.

    """

    # Extract the selected entries from the input vectors
    selected_a = vector_a[list(subspace_indices)]
    selected_b = vector_b[list(subspace_indices)]

    # Calculate the inner product of the selected entries
    inner_product = np.dot(selected_a, selected_b)

    return inner_product


def matrix_vector_mult_subspace(
    matrix: np.ndarray, vector: np.ndarray, subspace_indices
) -> np.ndarray:
    """Performs matrix-vector multiplication involving selected entries of the vector.

    Given a square matrix and a vector, this function calculates the multiplication
    between the selected columns of the matrix and the selected entries of the vector
    defined by a set of indices.

    :param matrix: Input square matrix of shape (n, n).
    :type matrix: numpy.ndarray

    :param vector: Input vector of length n.
    :type vector: numpy.ndarray

    :param subspace_indices: Set of indices specifying the selected
        entries of the vector and columns of the matrix.

    :type subspace_indices: set or numpy.ndarray

    :return: Resulting vector of length n.
    :rtype: numpy.ndarray

    :raises ValueError: If the dimensions of the matrix and vector are incompatible.
    :raises IndexError: If any index in subspace_indices is out of range.

    """
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError(
            f'Incompatible dimensions between matrix [{matrix.shape[1]}] '
            f'and vector [{vector.shape[0]}]'
        )

    if any(index >= vector.shape[0] for index in subspace_indices):
        raise IndexError(
            'Subspace index out of range for the vector of size {vector.shape[0]}'
        )

    # Extract the selected entries from the input vector
    selected_vector = vector[list(subspace_indices)] if subspace_indices else vector

    # Select the columns of the matrix based on the subspace indices
    selected_matrix = matrix[:, list(subspace_indices)] if subspace_indices else matrix

    # Perform matrix-vector multiplication using the selected entries and columns
    result = np.dot(selected_matrix, selected_vector)

    return result


class Bounds:
    """This class is designed for the management of simple bound constraints."""

    INFINITY = np.sqrt(np.finfo(float).max)

    def __init__(self, bounds: list[tuple[float | None, float | None]]):
        """Constructor.

        :param bounds: list of tuples (ell,u) containing the lower and
                  upper bounds for each free parameter.

        :type bounds: list(tuple)

        :raises biogeme.exceptions.OptimizationError: if the bounds are incompatible

        """

        def none_to_minus_infinity(value: float) -> float:
            """Returns -infinity if the input is None, otherwise returns the input canonical_value

            :param value: The input canonical_value.
            :type value: float

            :return: The modified canonical_value. If x is None, returns
                -infinity; otherwise, returns x.
            :rtype: float

            """
            if value is None:
                return -self.INFINITY

            return value

        def none_to_plus_infinity(value: float) -> float:
            """Returns +infinity if the input is None, otherwise
                returns the input canonical_value.

            :param value: The input canonical_value.
            :type value: float

            :return: The modified canonical_value. If x is None, returns
                +infinity; otherwise, returns x.
            :rtype: float

            """
            if value is None:
                return self.INFINITY

            return value

        self.bounds = bounds
        """list of tuples (ell, u) containing the lower and upper bounds for
            each free parameter.
        """

        self.dimension = len(
            bounds
        )  #: number of optimization unsorted_set_of_variables

        self.lower_bounds = np.array(
            [none_to_minus_infinity(bound[0]) for bound in bounds]
        )
        """ List of lower bounds """

        self.upper_bounds = np.array(
            [none_to_plus_infinity(bound[1]) for bound in bounds]
        )
        """ List of upper bounds """

        wrong_bounds = np.array(
            [
                bound[0] > bound[1]
                for bound in bounds
                if bound[0] is not None and bound[1] is not None
            ]
        )
        if wrong_bounds.any():
            error_msg = (
                f'Incompatible bounds for indice(s) '
                f'{np.nonzero(wrong_bounds)[0]}: '
                f'{[bounds[i] for i in np.nonzero(wrong_bounds)[0]]}'
            )
            raise OptimizationError(error_msg)

    @classmethod
    def from_bounds(cls, lower: np.ndarray, upper: np.ndarray) -> Bounds:
        """Ctor using two vectors of bounds

        :param lower: vector of lower bounds
        :type lower: numpy.array

        :param upper: vector of upper bounds
        :type upper: numpy.array

        """
        if len(lower) != len(upper):
            error_msg = f'Inconsistent length of arrays: {len(lower)} and {len(upper)}'
            raise OptimizationError(error_msg)

        list_of_tuples = list(zip(lower, upper))
        return cls(list_of_tuples)

    def __str__(self) -> str:
        """Magic string."""
        return self.bounds.__str__()

    def __repr__(self) -> str:
        return self.bounds.__repr__()

    def project(self, point: np.ndarray) -> np.ndarray:
        """Project a point onto the feasible domain defined by the bounds.

        :param point: point to project
        :type point: numpy.array

        :return: projected point
        :rtype: numpy.array

        :raises biogeme.exceptions.OptimizationError: if the dimensions
                are inconsistent
        """

        if len(point) != self.dimension:
            raise OptimizationError(
                f'Incompatible size: {len(point)}' f' and {self.dimension}'
            )

        new_point = np.clip(point, self.lower_bounds, self.upper_bounds)
        return new_point

    def intersect(self, other_bounds: Bounds) -> Bounds:
        """
        Create a bounds object representing the intersection of two regions.

        :param other_bounds: other bound object that must be intersected.
        :type other_bounds: class Bounds

        :return:  bound object, intersection of the two.
        :rtype: class Bounds

        :raises biogeme.exceptions.OptimizationError: if the dimensions
              are inconsistent
        """

        if other_bounds.dimension != self.dimension:
            raise OptimizationError(
                f'Incompatible size: {other_bounds.dimension} ' f'and {self.dimension}'
            )

        new_bounds = [
            (
                np.maximum(self.lower_bounds[i], other_bounds.lower_bounds[i]),
                np.minimum(self.upper_bounds[i], other_bounds.upper_bounds[i]),
            )
            for i in range(self.dimension)
        ]

        # Note that an exception with be raised at the creation of the
        # new 'Bounds' object is the bounds are incompatible.

        result = Bounds(new_bounds)
        return result

    def intersection_with_trust_region(
        self, point: np.ndarray, radius: float
    ) -> Bounds:
        """Create a Bounds object representing the intersection
            between the bounds and the trust region:
             point-radius <= x <= point+radius
            and
             ell <= x <= u

        :param point: center of the trust region
        :type point: numpy.array

        :param radius: radius of the trust region (infinity norm)
        :type radius: float

        :return: intersection between the feasible region and the trust region
        :rtype: class Bounds

        :raises OptimizationError: if the point is not feasible

        """
        if not self.feasible(point):
            error_msg = f'Center of the trust region of radius {radius} is not feasible'
            raise OptimizationError(error_msg)
        trust_region = Bounds.from_bounds(point - radius, point + radius)
        return self.intersect(trust_region)

    def get_bounds_for_trust_region_subproblem(
        self, point: np.ndarray, radius: float
    ) -> Bounds:
        """Create a Bounds object representing the bounds on the step
            for the trust region sub-problem. It is the intersection between:
              -radius <= d <= radius
            and
              ell-point <= d <= u-point

        :param point: center of the trust region
        :type point: numpy.array

        :param radius: radius of the trust region (infinity norm)
        :type radius: float

        :return: intersection between the feasible region and the trust region
        :rtype: class Bounds

        :raises OptimizationError: if the dimensions are inconsistent

        """
        if len(point) != self.dimension:
            raise OptimizationError(
                f'Incompatible size: {len(point)}' f' and {self.dimension}'
            )
        trust_region = Bounds.from_bounds(
            np.full(self.dimension, -radius),
            np.full(self.dimension, radius),
        )
        shifted_bounds = Bounds.from_bounds(
            self.lower_bounds - point, self.upper_bounds - point
        )
        return shifted_bounds.intersect(trust_region)

    def subspace(self, selected_variables: np.ndarray) -> Bounds:
        """Generate a Bounds object for selected unsorted_set_of_variables

        :param selected_variables: boolean vector. If an entry is True,
                   the corresponding unsorted_set_of_variables is considered.
        :type selected_variables: numpy.array(bool)

        :return: bound object
        :rtype: class Bounds

        :raises biogeme.exceptions.OptimizationError: if the dimensions
                are inconsistent
        """
        if len(selected_variables) != self.dimension:
            raise OptimizationError(
                f'Incompatible size: ' f'{len(selected_variables)} and {self.dimension}'
            )

        selected_bounds = [
            bound
            for bound, selected in zip(self.bounds, selected_variables)
            if selected
        ]
        return Bounds(selected_bounds)

    def feasible(self, point: np.ndarray) -> bool:
        """Check if point verifies the bound constraints

        :param point: point to project
        :type point: numpy.array

        :return: True if point is feasible, False otherwise.
        :rtype: bool

        :raises OptimizationError: if the dimensions are inconsistent
        """
        if point.size != self.dimension:
            raise OptimizationError(
                f'Incompatible size: ' f'point:{point.size} and {self.dimension}'
            )

        if np.any(self.lower_bounds - point > np.finfo(float).eps):
            return False
        if np.any(point - self.upper_bounds > np.finfo(float).eps):
            return False
        return True

    def maximum_step(
        self, point: np.ndarray, direction: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Calculates the maximum step that can be performed
        along a direction while staying feasible.

        :param point: reference point
        :type point: numpy.array

        :param direction: direction
        :type direction: numpy.array

        :return: the largest alpha such that point + alpha * direction is feasible
                 and the list of indices achieving this canonical_value.
        :rtype: float, int

        :raises biogeme.exceptions.OptimizationError: if the point is infeasible

        """

        if not self.feasible(point):
            raise OptimizationError(f'Infeasible point: {point}')

        def calculate_alpha(
            point_element: np.ndarray,
            lower: np.ndarray,
            upper: np.ndarray,
            direction_element: np.ndarray,
        ) -> np.ndarray:
            """Calculate the canonical_value of alpha on numpy arrays"""
            result = np.empty_like(point_element, dtype=float)

            upper_bound_exists = upper != np.inf
            lower_bound_exists = lower != -np.inf
            strictly_positive_entries = direction_element > np.finfo(float).eps
            strictly_negative_entries = direction_element < -np.finfo(float).eps

            # If the upper bound exist and the direction is positive,
            # we calculate the distance to the bound
            mask1 = upper_bound_exists & strictly_positive_entries
            result[mask1] = (upper[mask1] - point_element[mask1]) / direction_element[
                mask1
            ]

            # If there is no upper bound, and the direction is
            # positive, the step is infinite.
            mask2 = ~upper_bound_exists & strictly_positive_entries
            result[mask2] = np.inf

            # If the lower bound exist, and the direction is negative,
            # we calculate the distance to the bound.
            mask3 = lower_bound_exists & strictly_negative_entries
            result[mask3] = (lower[mask3] - point_element[mask3]) / direction_element[
                mask3
            ]

            # If the lower bound does not exist, and the direction is
            # negative, the step is infinite.
            mask4 = ~lower_bound_exists & strictly_negative_entries
            result[mask4] = np.inf

            # If the direction is zero, or very close to zero, the
            # step is infinite
            mask5 = ~strictly_positive_entries & ~strictly_negative_entries
            result[mask5] = np.inf

            return result

        alpha = calculate_alpha(point, self.lower_bounds, self.upper_bounds, direction)
        the_minimum = np.amin(alpha)
        return the_minimum, np.where(alpha == the_minimum)[0]

    def activity(
        self, point: np.ndarray, epsilon: float = np.finfo(float).eps
    ) -> np.ndarray:
        """Determines the activity status of each variable.

        :param point: point for which the activity must be determined.
        :type point: numpy.array

        :param epsilon: a bound is considered active if the distance
                        to it is less than epsilon.
        :type epsilon: float

        :return: a vector, same length as point, where each entry reports
            the activity of the corresponding variable:

            - 0 if no bound is active
            - -1 if the lower bound is active
            - 1 if the upper bound is active

        :rtype: numpy.array

        :raises biogeme.exceptions.OptimizationError: if the vector point is
                    not feasible

        :raises biogeme.exceptions.OptimizationError: if the dimensions
                    of x and bounds do not match.

        """
        if len(point) != self.dimension:
            raise OptimizationError(
                f'Incompatible size: {len(point)}' f' and {self.dimension}'
            )
        if not self.feasible(point):
            raise OptimizationError(
                f'{point} is not feasible for the ' f'bounds {self.bounds}'
            )

        activity_status = np.zeros(self.dimension, dtype=int)
        lb_mask = np.logical_and(
            self.lower_bounds != -np.inf, point - self.lower_bounds <= epsilon
        )
        ub_mask = np.logical_and(
            self.upper_bounds != np.inf, self.upper_bounds - point <= epsilon
        )

        activity_status[lb_mask] = -1
        activity_status[ub_mask] = 1

        return activity_status

    def active_constraints(
        self, point: np.ndarray, epsilon: float = np.finfo(float).eps
    ) -> set[int]:
        """Determines the indices of active constraints for a given point.

        :param point: The point for which to determine the active constraints.
        :type point: numpy.array

        :param epsilon: The tolerance used to determine the activity
            status of constraints.  the_matrix constraint is considered active
            if the distance to it is less than epsilon.
        :type epsilon: float, optional

        :return: the_matrix set containing the indices of active constraints.
        :rtype: set

        :raises biogeme.exceptions.OptimizationError: If the
            dimensions are inconsistent.
        :raises biogeme.exceptions.OptimizationError: If the point is
            infeasible.
        """

        activity_status = self.activity(point, epsilon)
        return {i for i, status in enumerate(activity_status) if status != 0}

    def inactive_constraints(
        self, point: np.ndarray, epsilon: float = np.finfo(float).eps
    ) -> set[int]:
        """Determines the indices of inactive constraints for a given point.

        :param point: The point for which to determine the active constraints.
        :type point: numpy.array

        :param epsilon: The tolerance used to determine the activity
            status of constraints.  the_matrix constraint is considered active
            if the distance to it is less than epsilon.
        :type epsilon: float, optional

        :return: the_matrix set containing the indices of inactive constraints.
        :rtype: set

        :raises biogeme.exceptions.OptimizationError: If the
            dimensions are inconsistent.
        :raises biogeme.exceptions.OptimizationError: If the point is
            infeasible.

        """

        activity_status = self.activity(point, epsilon)
        return {i for i, status in enumerate(activity_status) if status == 0}

    def breakpoints(
        self, point: np.ndarray, direction: np.ndarray
    ) -> list[tuple[int, float]]:
        """Projects the direction direction, starting from point,
        on the intersection of the bound constraints

        :param point: current point
        :type point: numpy.array

        :param direction: search direction
        :type direction: numpy.array

        :return: list of tuple (index, canonical_value), where index is the index
                 of the variable, and canonical_value the canonical_value of the corresponding
                 breakpoint.
        :rtype: list(tuple(int,float))

        :raises biogeme.exceptions.OptimizationError: if the dimensions
                are inconsistent

        :raises biogeme.exceptions.OptimizationError: if point is infeasible

        """
        if len(direction) != self.dimension:
            raise OptimizationError(
                f'Incompatible size: {self.dimension}' f' and {len(direction)}'
            )

        if not self.feasible(point):
            raise OptimizationError('Infeasible point')

        positive_mask = direction > np.finfo(float).eps
        negative_mask = direction < -np.finfo(float).eps

        upper_valid = self.upper_bounds != np.inf
        lower_valid = self.lower_bounds != -np.inf

        upper_breaks = np.where(
            positive_mask & upper_valid, (self.upper_bounds - point) / direction, np.inf
        )
        lower_breaks = np.where(
            negative_mask & lower_valid, (self.lower_bounds - point) / direction, np.inf
        )

        break_points = [(i, val) for i, val in enumerate(upper_breaks) if val < np.inf]
        break_points.extend(
            [(i, val) for i, val in enumerate(lower_breaks) if val < np.inf]
        )
        if not break_points:
            raise OptimizationError("Infeasible point")

        return sorted(break_points, key=lambda the_point: the_point[1])

    def projected_direction(
        self, x_current: np.ndarray, direction: np.ndarray
    ) -> np.ndarray:
        """Calculates the projected direction.

        Given the current point x_current and the direction,
        the function determines the projected direction:

        .. math:: \\lim_{\\alpha \to 0^+} P[x + \\alpha d] - x

        :param x_current: current point
        :type x_current: numpy.array

        :param direction: direction to follow
        :type direction: numpy.array

        :return: the projected direction
        :rtype: numpy.array

        :raises OptimizationError: if the dimensions are inconsistent
        :raises OptimizationError: if the current point is infeasible

        """
        activity_status = self.activity(x_current)
        beyond_upper = np.logical_and(activity_status == 1, direction > 0)
        beyond_lower = np.logical_and(activity_status == -1, direction < 0)
        projected_direction = direction
        projected_direction[beyond_upper] = 0
        projected_direction[beyond_lower] = 0

        return projected_direction

    def generalized_cauchy_point(
        self, x_current: np.ndarray, g_current: np.ndarray, h_current: np.ndarray
    ) -> np.ndarray:
        """Implementation of Step 2 of the Specific Algorithm by
        `Conn et al. (1988)`_.

        .. _`Conn et al. (1988)`: https://www.ams.org/journals/mcom/1988-50-182/S0025-5718-1988-0929544-3/S0025-5718-1988-0929544-3.pdf

        The quadratic model is defined as

        .. math:: m(x) = f(x_k) + (x - x_k)^T g_k +
                  \\frac{1}{2} (x-x_k)^T H (x-x_k).

        :param x_current: current point
        :type x_current: numpy.array. Dimension n.

        :param g_current: vector g involved in the quadratic model definition.
        :type g_current: numpy.array. Dimension n.

        :param h_current: matrix h_current involved in the quadratic model definition.
        :type h_current: numpy.array. Dimension n x n.

        :return: generalized Cauchy point based on inexact line search.
        :rtype: numpy.array. Dimension n.

        :raises biogeme.exceptions.OptimizationError: if the dimensions are
               inconsistent
        :raises biogeme.exceptions.OptimizationError: if xk is infeasible
        """
        if not isinstance(x_current, np.ndarray):
            raise OptimizationError(f'Not an array {x_current}: {type(x_current)}')

        if len(x_current) != self.dimension:
            raise OptimizationError(
                f'Incompatible size: {len(x_current)}' f' and {self.dimension}'
            )
        if len(g_current) != self.dimension:
            raise OptimizationError(
                f'Incompatible size: {len(g_current)}' f' and {self.dimension}'
            )
        if np.isnan(g_current).any():
            raise OptimizationError(f'Invalid gradient: {g_current}')

        if h_current.shape[0] != self.dimension or h_current.shape[1] != self.dimension:
            raise OptimizationError(
                f'Incompatible size: {h_current.shape}' f' and {self.dimension}'
            )

        if np.isnan(h_current).any():
            raise OptimizationError(f'Invalid hessian: {h_current}')

        if not self.feasible(x_current):
            raise OptimizationError(
                f'Infeasible iterate {x_current} '
                '[LB: {self.lower_bounds} UB: {self.upper_bounds}]'
            )

        x = x_current
        g = g_current - h_current @ x_current
        direction = self.projected_direction(x_current, -g_current)

        fprime = np.inner(g_current, direction)
        if fprime >= 0:
            if self.dimension <= 10:
                logger.warning(
                    f'GCP: direction {direction} is not a descent direction at {x}.'
                )
            else:
                logger.warning('GCP: not a descent direction.')

        f_second = np.inner(direction, h_current @ direction)

        set_active_constraints = set()

        while len(set_active_constraints) < self.dimension:
            # Calculate the maximum step that can be done in the current direction
            delta_t, _ = self.maximum_step(x, direction)
            # Test whether the GCP has been found
            if f_second > 0:
                ratio = -fprime / f_second
                if 0 < ratio < delta_t:
                    x = x + ratio * direction
                    return x

            # In theory, x + delta_t * direction must be feasible. However,
            # there may be some numerical problem. Therefore, we
            # project it on the feasible domain to make sure to obtain
            # a feasible point.
            x_plus = self.project(x + delta_t * direction)

            # Note that there may be more than one variable activated
            # at x_plus. Therefore, we recalculate the activity
            # status.
            set_active_constraints_plus = self.active_constraints(x_plus)
            activated = set_active_constraints_plus - set_active_constraints

            x = x_plus
            set_active_constraints = set_active_constraints_plus

            # Update line derivatives
            bd = np.zeros_like(direction)
            bd[list(activated)] = direction[list(activated)]
            b = h_current @ bd

            d_dot_g = np.sum([direction[i] * g[i] for i in activated])
            fprime += delta_t * f_second - np.inner(b, x) - d_dot_g
            f_second += np.inner(b, bd - 2 * direction)
            direction[list(activated)] = 0.0
            if np.all(direction == 0) or fprime >= 0:
                return x

        return x

    def truncated_conjugate_gradient(
        self, gradient: np.ndarray, hessian: np.ndarray, tol: float = 1.0e-6
    ) -> tuple[np.ndarray, ConjugateGradientDiagnostic]:
        """Find an approximation of the trust region sub-problem using
            the truncated conjugate gradient method, where the step d
            mus verify the bound constraints.

        :param gradient: gradient of the quadratic model.
        :type gradient: numpy.array

        :param hessian: hessian of the quadratic model.
        :type hessian: numpy.array

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
        dimension = gradient.size
        if hessian.shape[0] != dimension:
            error_msg = f'Incompatible sizes: hessian:{hessian.shape[0]} and gradient:{dimension}'
            raise OptimizationError(error_msg)

        if hessian.shape[0] != hessian.shape[1]:
            error_msg = (
                f'Matrix must be square and not {hessian.shape[0]} x {hessian.shape[1]}'
            )
            raise OptimizationError(error_msg)

        xk = np.zeros(dimension)
        hx_plus_g = gradient
        dk = -gradient
        for _ in range(dimension):
            try:
                if np.linalg.norm(hx_plus_g) <= tol:
                    diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
                    step = xk
                    return step, diagnostic

                curvature = np.inner(dk, hessian @ dk)
                if curvature <= 0:
                    # Negative curvature has been detected
                    diagnostic = ConjugateGradientDiagnostic.NEGATIVE_CURVATURE
                    step, _ = self.maximum_step(xk, dk)
                    solution = xk + step * dk
                    return solution, diagnostic
                alpha_k = -np.inner(dk, hx_plus_g) / curvature
                xkp1 = xk + alpha_k * dk
                if np.isnan(xkp1).any() or not self.feasible(xkp1):
                    # Out of the trust region
                    diagnostic = ConjugateGradientDiagnostic.OUT_OF_TRUST_REGION
                    step, _ = self.maximum_step(xk, dk)
                    solution = xk + step * dk
                    return solution, diagnostic
                xk = xkp1
                next_hx_plus_g = hessian @ xk + gradient
                beta_k = np.inner(next_hx_plus_g, next_hx_plus_g) / np.inner(
                    hx_plus_g, hx_plus_g
                )
                dk = -next_hx_plus_g + beta_k * dk
                hx_plus_g = next_hx_plus_g
            except ValueError:
                # Numerical problem. We follow the last direction
                # until the border of the trust region
                diagnostic = ConjugateGradientDiagnostic.NUMERICAL_PROBLEM
                step, _ = self.maximum_step(xk, dk)
                solution = xk + step * dk
                return solution, diagnostic
        # In theory, after n iterations, the algorithm has converged
        diagnostic = ConjugateGradientDiagnostic.CONVERGENCE
        step = xk
        return step, diagnostic

    def truncated_conjugate_gradient_subspace(
        self,
        iterate: np.ndarray,
        gradient: np.ndarray,
        hessian: np.ndarray,
        radius: float,
        tol: float = np.finfo(np.float64).eps ** 0.3333,
    ) -> tuple[np.ndarray, ConjugateGradientDiagnostic]:
        """Find an approximation of the solution of the trust region
        subproblem using the truncated conjugate gradient method within
        the subspace of free unsorted_set_of_variables. Free unsorted_set_of_variables are those
        corresponding to inactive constraints at the generalized Cauchy
        point.

        :param iterate: current iterate.
        :type iterate: numpy.array

        :param gradient: gradient of the quadratic model.
        :type gradient: numpy.array

        :param hessian: hessian of the quadrartic model.
        :type hessian: numpy.array

        :param radius: radius of the trust region.
        :type radius: float

        :param tol: tolerance used for stopping criterion.
        :type tol: float

        :return: d, diagnostic, where

              - d is the approximate solution of the trust region subproblem,
              - diagnostic is the nature of the solution:

                * CONVERGENCE for convergence,
                * OUT_OF_TRUST_REGION if out of the trust region,
                * NEGATIVE_CURVATURE if negative curvature detected.
                * NUMERICAL_PROBLEM if a numerical problem has been encountered

        :rtype: numpy.array, ConjugateGradientDiagnostic

        :raises OptimizationError: if the dimensions are inconsistent

        """
        if (
            np.isnan(iterate).any()
            or np.isnan(gradient).any()
            or np.isnan(hessian).any()
        ):
            raise OptimizationError(
                'Invalid input: iterate, gradient, or hessian contains NaN values.'
            )

        # For numerical reasons, it is better to project the iterate
        # on the bounds, although it should be feasible.
        iterate = self.project(iterate)

        # First, we calculate the intersection between the trust region on
        # the bounds. The trust region is also a bound constraint (based
        # on infinity norm) of radius 'radius', centered at iterate.
        intersection = self.intersection_with_trust_region(iterate, radius)

        # Then, we calculate the generalized Cauchy point
        gcp = intersection.generalized_cauchy_point(iterate, gradient, hessian)

        bounds_trust_region_subproblem = self.get_bounds_for_trust_region_subproblem(
            iterate, radius
        )

        # We fix the unsorted_set_of_variables active at the GCP
        activity_status: np.ndarray = intersection.activity(gcp)
        free_variables: np.ndarray = activity_status == 0
        fixed_variables: np.ndarray = activity_status != 0

        if not free_variables.any():
            return gcp, ConjugateGradientDiagnostic.CONVERGENCE

        bounds_subspace = bounds_trust_region_subproblem.subspace(free_variables)
        gradient_subspace = gradient[free_variables]
        hessian_subspace = hessian[free_variables][:, free_variables]

        candidate, diagnostic = bounds_subspace.truncated_conjugate_gradient(
            gradient_subspace, hessian_subspace, tol
        )

        solution = gcp
        solution[free_variables] = iterate[free_variables] + candidate
        return solution, diagnostic
