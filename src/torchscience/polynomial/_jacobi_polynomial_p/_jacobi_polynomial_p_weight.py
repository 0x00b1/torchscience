import warnings

import torch
from torch import Tensor

from ._jacobi_polynomial_p import JacobiPolynomialP


def jacobi_polynomial_p_weight(
    x: Tensor,
    alpha: Tensor | float,
    beta: Tensor | float,
) -> Tensor:
    """Compute Jacobi weight function.

    The weight function is w(x) = (1-x)^α * (1+x)^β, which appears in the
    orthogonality relation for Jacobi polynomials:

        integral_{-1}^{1} P_m^{(α,β)}(x) P_n^{(α,β)}(x) w(x) dx = 0  for m != n

    Parameters
    ----------
    x : Tensor
        Points at which to evaluate weight.
    alpha : Tensor or float
        Jacobi parameter α, must be > -1.
    beta : Tensor or float
        Jacobi parameter β, must be > -1.

    Returns
    -------
    Tensor
        Weight values w(x) = (1-x)^α * (1+x)^β with same shape as x.

    Warnings
    --------
    UserWarning
        If any evaluation points are outside the natural domain [-1, 1].

    Notes
    -----
    The weight function w(x) = (1-x)^α * (1+x)^β is used for:
    - Computing orthogonality integrals
    - Gauss-Jacobi quadrature
    - Weighted least squares fitting

    For α=β=0 (Legendre), w(x) = 1.
    For α=β=-1/2 (Chebyshev T), w(x) = 1/sqrt(1-x²).
    For α=β=1/2 (Chebyshev U), w(x) = sqrt(1-x²).

    Examples
    --------
    >>> jacobi_polynomial_p_weight(torch.tensor([0.0, 0.5]), alpha=0.0, beta=0.0)
    tensor([1., 1.])  # Legendre weight is 1

    >>> jacobi_polynomial_p_weight(torch.tensor([0.0, 0.5]), alpha=1.0, beta=1.0)
    tensor([1.0000, 0.5625])  # (1-0)*(1+0)=1, (1-0.5)*(1+0.5)=0.75, ^1 each
    """
    domain = JacobiPolynomialP.DOMAIN

    if ((x < domain[0]) | (x > domain[1])).any():
        warnings.warn(
            f"Evaluating JacobiPolynomialP weight function outside natural domain "
            f"[{domain[0]}, {domain[1]}].",
            stacklevel=2,
        )

    # Convert alpha and beta to tensors if needed
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=x.dtype, device=x.device)

    # w(x) = (1-x)^α * (1+x)^β
    return torch.pow(1 - x, alpha) * torch.pow(1 + x, beta)
