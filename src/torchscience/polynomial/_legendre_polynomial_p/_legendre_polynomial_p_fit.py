import torch
from torch import Tensor

from torchscience.polynomial._exceptions import DomainError

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)
from ._legendre_polynomial_p_vandermonde import (
    legendre_polynomial_p_vandermonde,
)


def legendre_polynomial_p_fit(
    x: Tensor,
    y: Tensor,
    degree: int,
) -> LegendrePolynomialP:
    """Fit Legendre series to data using least squares.

    Parameters
    ----------
    x : Tensor
        Sample points, shape (n,). Must be in [-1, 1].
    y : Tensor
        Sample values, shape (n,).
    degree : int
        Degree of fitting polynomial.

    Returns
    -------
    LegendrePolynomialP
        Fitted Legendre series.

    Raises
    ------
    DomainError
        If any sample points are outside [-1, 1].

    Notes
    -----
    Uses the normal equations via torch.linalg.lstsq for numerical stability.

    Examples
    --------
    >>> x = torch.linspace(-1, 1, 10)
    >>> y = x**2
    >>> c = legendre_polynomial_p_fit(x, y, degree=2)
    """
    domain = LegendrePolynomialP.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        raise DomainError(
            f"Fitting points must be in [{domain[0]}, {domain[1]}] "
            f"for LegendrePolynomialP"
        )

    # Build Vandermonde matrix
    V = legendre_polynomial_p_vandermonde(x, degree)

    # Solve least squares: V @ coeffs = y
    # Use lstsq for numerical stability
    result = torch.linalg.lstsq(V, y.unsqueeze(-1))
    coeffs = result.solution.squeeze(-1)

    return legendre_polynomial_p(coeffs)
