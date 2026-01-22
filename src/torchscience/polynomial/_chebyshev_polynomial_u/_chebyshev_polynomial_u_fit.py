import torch
from torch import Tensor

from torchscience.polynomial._exceptions import DomainError

from ._chebyshev_polynomial_u import (
    ChebyshevPolynomialU,
    chebyshev_polynomial_u,
)
from ._chebyshev_polynomial_u_vandermonde import (
    chebyshev_polynomial_u_vandermonde,
)


def chebyshev_polynomial_u_fit(
    x: Tensor,
    y: Tensor,
    degree: int,
) -> ChebyshevPolynomialU:
    """Fit Chebyshev U series to data using least squares.

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
    ChebyshevPolynomialU
        Fitted Chebyshev U series.

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
    >>> c = chebyshev_polynomial_u_fit(x, y, degree=2)
    """
    domain = ChebyshevPolynomialU.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        raise DomainError(
            f"Fitting points must be in [{domain[0]}, {domain[1]}] "
            f"for ChebyshevPolynomialU"
        )

    # Build Vandermonde matrix
    V = chebyshev_polynomial_u_vandermonde(x, degree)

    # Solve least squares: V @ coeffs = y
    # Use lstsq for numerical stability
    result = torch.linalg.lstsq(V, y.unsqueeze(-1))
    coeffs = result.solution.squeeze(-1)

    return chebyshev_polynomial_u(coeffs)
