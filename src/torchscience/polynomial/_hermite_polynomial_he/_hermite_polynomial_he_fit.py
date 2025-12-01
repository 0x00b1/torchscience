import torch
from torch import Tensor

from ._hermite_polynomial_he import HermitePolynomialHe
from ._hermite_polynomial_he_vandermonde import (
    hermite_polynomial_he_vandermonde,
)


def hermite_polynomial_he_fit(
    x: Tensor,
    y: Tensor,
    degree: int,
) -> HermitePolynomialHe:
    """Fit Probabilists' Hermite series to data using least squares.

    Parameters
    ----------
    x : Tensor
        Sample points, shape (n,). Can be any real values (unbounded domain).
    y : Tensor
        Sample values, shape (n,).
    degree : int
        Degree of fitting polynomial.

    Returns
    -------
    HermitePolynomialHe
        Fitted Hermite series.

    Notes
    -----
    Uses the normal equations via torch.linalg.lstsq for numerical stability.

    Unlike bounded domain polynomials, Hermite polynomials have unbounded
    domain (-inf, inf), so no domain check is performed.

    Examples
    --------
    >>> x = torch.linspace(-2, 2, 10)
    >>> y = x**2
    >>> c = hermite_polynomial_he_fit(x, y, degree=2)
    """
    # No domain check for Hermite polynomials since domain is (-inf, inf)

    # Build Vandermonde matrix
    V = hermite_polynomial_he_vandermonde(x, degree)

    # Solve least squares: V @ coeffs = y
    # Use lstsq for numerical stability
    result = torch.linalg.lstsq(V, y.unsqueeze(-1))
    coeffs = result.solution.squeeze(-1)

    return HermitePolynomialHe(coeffs=coeffs)
