import torch
from torch import Tensor

from .._domain_error import DomainError
from ._jacobi_polynomial_p import JacobiPolynomialP, jacobi_polynomial_p
from ._jacobi_polynomial_p_vandermonde import jacobi_polynomial_p_vandermonde


def jacobi_polynomial_p_fit(
    x: Tensor,
    y: Tensor,
    degree: int,
    alpha: Tensor | float,
    beta: Tensor | float,
) -> JacobiPolynomialP:
    """Fit Jacobi series to data using least squares.

    Parameters
    ----------
    x : Tensor
        Sample points, shape (n,). Must be in [-1, 1].
    y : Tensor
        Sample values, shape (n,).
    degree : int
        Degree of fitting polynomial.
    alpha : Tensor or float
        Jacobi parameter alpha, must be > -1.
    beta : Tensor or float
        Jacobi parameter beta, must be > -1.

    Returns
    -------
    JacobiPolynomialP
        Fitted Jacobi series.

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
    >>> c = jacobi_polynomial_p_fit(x, y, degree=2, alpha=0.5, beta=0.5)
    """
    domain = JacobiPolynomialP.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        raise DomainError(
            f"Fitting points must be in [{domain[0]}, {domain[1]}] "
            f"for JacobiPolynomialP"
        )

    # Convert alpha and beta to tensors if needed
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=x.dtype, device=x.device)

    # Build Vandermonde matrix
    V = jacobi_polynomial_p_vandermonde(x, degree, alpha, beta)

    # Solve least squares: V @ coeffs = y
    # Use lstsq for numerical stability
    result = torch.linalg.lstsq(V, y.unsqueeze(-1))
    coeffs = result.solution.squeeze(-1)

    return jacobi_polynomial_p(coeffs, alpha, beta)
