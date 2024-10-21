"""File algebra.py

:author: Michel Bierlaire
:date: Thu Jun 22 15:09:06 2023 

Function related to linear algebra
"""

import logging
import numpy as np
import scipy.linalg as la
from biogeme_optimization.exceptions import OptimizationError


logger = logging.getLogger(__name__)


def schnabel_eskow(
    the_matrix: np.ndarray,
    tau: float = np.finfo(np.float64).eps ** 0.3333,
    taubar: float = np.finfo(np.float64).eps ** 0.6666,
    mu: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modified Cholesky factorization by `Schnabel and Eskow (1999)`_.

    .. _`Schnabel and Eskow (1999)`: https://doi.org/10.1137/s105262349833266x

    If the matrix is 'safely' positive definite, the output is the
    classical Cholesky factor. If not, the diagonal elements are
    inflated in order to make it positive definite. The factor :math:`L`
    is such that :math:`the_matrix + E = PLL^TP^T`, where :math:`E` is a diagonal
    matrix contaninig the terms added to the diagonal, :math:`P` is a
    permutation matrix, and :math:`L` is w lower triangular matrix.

    :param the_matrix: matrix to factorize. Must be square and symmetric.
    :type the_matrix: numpy.array
    :param tau: tolerance factor.
                Default: :math:`\\varepsilon^{\\frac{1}{3}}`.
                See `Schnabel and Eskow (1999)`_
    :type tau: float
    :param taubar: tolerance factor.
                   Default: :math:`\\varepsilon^{\\frac{2}{3}}`.
                   See `Schnabel and Eskow (1999)`_
    :type taubar: float
    :param mu: tolerance factor.
               Default: 0.1.  See `Schnabel and Eskow (1999)`_
    :type mu: float

    :return: tuple :math:`L`, :math:`E`, :math:`P`,
                    where :math:`the_matrix + E = PLL^TP^T`.
    :rtype: numpy.array, numpy.array, numpy.array

    :raises biogeme.exceptions.OptimizationError: if the matrix the_matrix is not square.
    :raises biogeme.exceptions.OptimizationError: if the matrix the_matrix is not symmetric.
    """

    def pivot(j: int) -> None:
        the_matrix[j, j] = np.sqrt(the_matrix[j, j])
        for i in range(j + 1, dim):
            the_matrix[j, i] = the_matrix[i, j] = the_matrix[i, j] / the_matrix[j, j]
            the_matrix[i, j + 1 : i + 1] -= (
                the_matrix[i, j] * the_matrix[j + 1 : i + 1, j]
            )
            the_matrix[j + 1 : i + 1, i] = the_matrix[i, j + 1 : i + 1]

    def permute(i: int, j: int):
        the_matrix[[i, j]] = the_matrix[[j, i]]
        diagonal_matrix[[i, j]] = diagonal_matrix[[j, i]]
        the_matrix[:, [i, j]] = the_matrix[:, [j, i]]
        permutation_matrix[:, [i, j]] = permutation_matrix[:, [j, i]]

    the_matrix = the_matrix.astype(np.float64)
    dim = the_matrix.shape[0]
    if the_matrix.shape[1] != dim:
        raise OptimizationError('The matrix must be square')

    if not np.array_equal(the_matrix, the_matrix.T):
        raise OptimizationError('The matrix must be square and symmetric')

    diagonal_matrix = np.zeros(dim, dtype=np.float64)
    permutation_matrix = np.identity(dim)
    phase_one = True
    gamma = abs(the_matrix.diagonal()).max()
    j = 0
    while j < dim and phase_one is True:
        a_max = np.max(the_matrix.diagonal()[j:])
        a_min = np.min(the_matrix.diagonal()[j:])
        if a_max < taubar * gamma or a_min < -mu * a_max:
            phase_one = False
            break

        # Pivot on maximum diagonal of remaining submatrix
        i = j + np.argmax(the_matrix.diagonal()[j:])
        if i != j:
            # Switch rows and columns of i and j of the_matrix
            permute(i, j)
        if (
            j < dim - 1
            and np.min(
                the_matrix.diagonal()[j + 1 :]
                - the_matrix[j + 1 :, j] ** 2 / the_matrix.diagonal()[j]
            )
            < -mu * gamma
        ):
            phase_one = False  # go to phase two
        else:
            # perform jth iteration of factorization
            pivot(j)
            j += 1

    # Phase two, the_matrix not positive-definite
    if not phase_one:
        if j == dim - 1:
            diagonal_matrix[-1] = delta = -the_matrix[-1, -1] + max(
                tau * (-the_matrix[-1, -1]) / (1 - tau), taubar * gamma
            )
            the_matrix[-1, -1] += delta
            the_matrix[-1, -1] = np.sqrt(the_matrix[-1, -1])
        else:
            delta_prev = 0.0
            gerschgorin = np.zeros(dim)
            k = j - 1  # k = number of iterations performed in phase one
            # Calculate lower Gerschgorin bounds of the_matrix[k+1]
            for i in range(k + 1, dim):
                gerschgorin[i] = (
                    the_matrix[i, i]
                    - np.sum(np.abs(the_matrix[i, k + 1 : i]))
                    - np.sum(np.abs(the_matrix[i + 1 : dim, i]))
                )
            # Modified Cholesky Decomposition
            for j in range(k + 1, dim - 2):
                # Pivot on maximum lower Gerschgorin bound estimate
                i = j + np.argmax(gerschgorin[j:])
                if i != j:
                    # Switch rows and columns of i and j of the_matrix
                    permute(i, j)
                # Calculate E[j, j] and add to diagonal
                norm_j = np.sum(np.abs(the_matrix[j + 1 : dim, j]))
                diagonal_matrix[j] = delta = max(
                    0, -the_matrix[j, j] + max(norm_j, taubar * gamma), delta_prev
                )
                if delta > 0:
                    the_matrix[j, j] += delta
                    delta_prev = delta  # delta_prev will contain E_inf
                # Update Gerschgorin bound estimates
                if the_matrix[j, j] != norm_j:
                    temp = 1.0 - norm_j / the_matrix[j, j]
                    gerschgorin[j + 1 :] += np.abs(the_matrix[j + 1 :, j]) * temp
                # perform jth iteration of factorization
                pivot(j)

            # Final 2 by 2 submatrix
            eigenvalues = np.linalg.eigvalsh(the_matrix[-2:, -2:])
            eigenvalues.sort()
            diagonal_matrix[-2] = diagonal_matrix[-1] = delta = max(
                0,
                -eigenvalues[0]
                + np.maximum(
                    tau * (eigenvalues[1] - eigenvalues[0]) / (1 - tau), taubar * gamma
                ),
                delta_prev,
            )
            if delta > 0:
                the_matrix[-2, -2] += delta
                the_matrix[-1, -1] += delta
                delta_prev = delta
            the_matrix[-2, -2] = np.sqrt(
                the_matrix[-2, -2]
            )  # overwrites the_matrix[-2, -2]
            the_matrix[-1, -2] = (
                the_matrix[-1, -2] / the_matrix[-2, -2]
            )  # overwrites the_matrix[-1, -2]
            the_matrix[-2, -1] = the_matrix[-1, -2]
            # overwrites the_matrix[-1, -1]
            the_matrix[-1, -1] = np.sqrt(
                the_matrix[-1, -1] - the_matrix[-1, -2] * the_matrix[-1, -2]
            )

    lower_triangular_matrix = np.tril(the_matrix)
    return (
        lower_triangular_matrix,
        np.diag(permutation_matrix @ diagonal_matrix),
        permutation_matrix,
    )


def schnabel_eskow_direction(
    gradient: np.ndarray, hessian: np.ndarray, check_convexity: bool = False
) -> np.ndarray:
    """Calculate a descent direction using the Schnabel-Eskow factorization"""
    if check_convexity:
        lower, diagonal, permutation = schnabel_eskow(hessian)
        corrections = diagonal > 0
        if np.any(corrections):
            raise OptimizationError('The quadratic model is not convex.')
    else:
        lower, _, permutation = schnabel_eskow(hessian)
    y3 = -permutation.T @ gradient
    y2 = la.solve_triangular(lower, y3, lower=True, overwrite_b=True)
    y1 = la.solve_triangular(lower.T, y2, lower=False, overwrite_b=True)
    d = permutation @ y1
    return d
