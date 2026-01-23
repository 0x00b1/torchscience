"""Least squares fit of Euler polynomial series."""

import torch
from torch import Tensor

from ._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
)
from ._euler_polynomial_e_vandermonde import euler_polynomial_e_vandermonde


def euler_polynomial_e_fit(
    x: Tensor,
    y: Tensor,
    degree: int,
) -> EulerPolynomialE:
    """Least squares fit of Euler polynomial series to data.

    Parameters
    ----------
    x : Tensor
        Sample points.
    y : Tensor
        Sample values at x.
    degree : int
        Degree of fitting polynomial.

    Returns
    -------
    EulerPolynomialE
        Fitted Euler polynomial series.
    """
    # Build Vandermonde matrix
    V = euler_polynomial_e_vandermonde(x, degree)

    # Solve least squares: V @ c = y
    # Using pseudo-inverse for numerical stability
    coeffs, _, _, _ = torch.linalg.lstsq(V, y.unsqueeze(-1))

    return euler_polynomial_e(coeffs.squeeze(-1))
