import torch
from torch import Tensor

from torchscience.polynomial._exceptions import DomainError

from ._gegenbauer_polynomial_c import GegenbauerPolynomialC
from ._gegenbauer_polynomial_c_vandermonde import (
    gegenbauer_polynomial_c_vandermonde,
)


def gegenbauer_polynomial_c_fit(
    x: Tensor,
    y: Tensor,
    degree: int,
    lambda_: Tensor,
) -> GegenbauerPolynomialC:
    """Fit Gegenbauer series to data using least squares.

    Parameters
    ----------
    x : Tensor
        Sample points, shape (n,). Must be in [-1, 1].
    y : Tensor
        Sample values, shape (n,).
    degree : int
        Degree of fitting polynomial.
    lambda_ : Tensor
        Parameter lambda > -1/2.

    Returns
    -------
    GegenbauerPolynomialC
        Fitted Gegenbauer series.

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
    >>> c = gegenbauer_polynomial_c_fit(x, y, degree=2, lambda_=torch.tensor(1.0))
    """
    domain = GegenbauerPolynomialC.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        raise DomainError(
            f"Fitting points must be in [{domain[0]}, {domain[1]}] "
            f"for GegenbauerPolynomialC"
        )

    # Ensure lambda_ is a tensor
    if not isinstance(lambda_, Tensor):
        lambda_ = torch.tensor(lambda_, dtype=x.dtype, device=x.device)

    # Build Vandermonde matrix
    V = gegenbauer_polynomial_c_vandermonde(x, degree, lambda_)

    # Solve least squares: V @ coeffs = y
    # Use lstsq for numerical stability
    result = torch.linalg.lstsq(V, y.unsqueeze(-1))
    coeffs = result.solution.squeeze(-1)

    return GegenbauerPolynomialC(coeffs=coeffs, lambda_=lambda_)
