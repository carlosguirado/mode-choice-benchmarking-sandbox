"""File bfgs.py

:author: Michel Bierlaire
:date: Sat Jun 24 17:34:20 2023

Functions for the BFGS algorithm
"""

import logging

import numpy as np

from biogeme_optimization.exceptions import OptimizationError

logger = logging.getLogger(__name__)


def bfgs(hessian_approx: np.ndarray, delta_x: np.ndarray, delta_g: np.ndarray):
    """Update the BFGS matrix. Formula (13.12) of `Bierlaire (2015)`_
    where the method proposed by `Powell (1977)`_ is applied

    .. _`Bierlaire (2015)`: http://optimizationprinciplesalgorithms.com/
    .. _`Powell (1977)`: https://link.springer.com/content/pdf/10.1007/BFb0067703.pdf

    :param hessian_approx: current approximation of the Hessian
    :type hessian_approx: numpy.array (2D)

    :param delta_x: difference between two consecutive iterates.
    :type delta_x: numpy.array (1D)

    :param delta_g: difference between two consecutive gradients.
    :type delta_g: numpy.array (1D)

    :return: updated approximation of the inverse of the Hessian.
    :rtype: numpy.array (2D)

    """
    if np.all(delta_x == 0):
        raise OptimizationError('In BFGS, the step cannot be zero')

    if np.all(delta_g == 0):
        raise OptimizationError('In BFGS, the gradient difference cannot be zero')
    h_d = hessian_approx @ delta_x
    d_h_d = np.inner(delta_x, h_d)
    denominator = np.inner(delta_x, delta_g)
    if denominator >= 0.2 * d_h_d:
        eta = delta_g
    else:
        theta = 0.8 * d_h_d / (d_h_d - denominator)
        eta = theta * delta_g + (1 - theta) * h_d

    delta_x_eta = np.inner(delta_x, eta)
    hessian_update = np.outer(eta, eta) / delta_x_eta - np.outer(h_d, h_d) / d_h_d
    return hessian_approx + hessian_update


def inverse_bfgs(
    inverse_hessian_approx: np.ndarray, delta_x: np.ndarray, delta_g: np.ndarray
):
    """Update the inverse BFGS matrix. Formula (13.13) of `Bierlaire (2015)`_

    .. _`Bierlaire (2015)`: http://optimizationprinciplesalgorithms.com/

    :param inverse_hessian_approx: current approximation of the inverse of the Hessian
    :type inverse_hessian_approx: numpy.array (2D)

    :param delta_x: difference between two consecutive iterates.
    :type delta_x: numpy.array (1D)

    :param delta_g: difference between two consecutive gradients.
    :type delta_g: numpy.array (1D)

    :return: updated approximation of the inverse of the Hessian.
    :rtype: numpy.array (2D)
    """
    dimension = len(delta_x)

    denominator = np.inner(delta_x, delta_g)
    if denominator <= 0.0:
        raise OptimizationError(
            f"Unable to perform inverse BFGS update as d'y = {denominator} <= 0"
        )
    d_y = np.outer(delta_x, delta_g)
    y_d = np.outer(delta_g, delta_x)
    d_d = np.outer(delta_x, delta_x)
    eye = np.identity(dimension)
    return (
        (eye - (d_y / denominator))
        @ inverse_hessian_approx
        @ (eye - (y_d / denominator))
    ) + d_d / denominator
