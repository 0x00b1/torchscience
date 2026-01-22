import torch
from torch import Tensor

from torchscience.polynomial._exceptions import DomainError

from ._laguerre_polynomial_l import (
    LaguerrePolynomialL,
    laguerre_polynomial_l,
)
from ._laguerre_polynomial_l_vandermonde import (
    laguerre_polynomial_l_vandermonde,
)


def laguerre_polynomial_l_fit(
    x: Tensor,
    y: Tensor,
    degree: int,
) -> LaguerrePolynomialL:
    """Fit Laguerre series to data using least squares.

    Parameters
    ----------
    x : Tensor
        Sample points, shape (n,). Must be in [0, inf).
    y : Tensor
        Sample values, shape (n,).
    degree : int
        Degree of fitting polynomial.

    Returns
    -------
    LaguerrePolynomialL
        Fitted Laguerre series.

    Raises
    ------
    DomainError
        If any sample points are outside [0, inf).

    Notes
    -----
    Uses the normal equations via torch.linalg.lstsq for numerical stability.

    Examples
    --------
    >>> x = torch.linspace(0, 5, 10)
    >>> y = torch.exp(-x)
    >>> c = laguerre_polynomial_l_fit(x, y, degree=3)
    """
    domain = LaguerrePolynomialL.DOMAIN

    if (x < domain[0]).any():
        raise DomainError(
            f"Fitting points must be in [{domain[0]}, {domain[1]}) "
            f"for LaguerrePolynomialL"
        )

    # Build Vandermonde matrix
    V = laguerre_polynomial_l_vandermonde(x, degree)

    # Solve least squares: V @ coeffs = y
    # Use lstsq for numerical stability
    result = torch.linalg.lstsq(V, y.unsqueeze(-1))
    coeffs = result.solution.squeeze(-1)

    return laguerre_polynomial_l(coeffs)
