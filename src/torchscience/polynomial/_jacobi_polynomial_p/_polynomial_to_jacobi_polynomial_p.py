import torch
from torch import Tensor

from .._polynomial import Polynomial
from ._jacobi_polynomial_p import JacobiPolynomialP
from ._jacobi_polynomial_p_add import jacobi_polynomial_p_add
from ._jacobi_polynomial_p_mulx import jacobi_polynomial_p_mulx


def polynomial_to_jacobi_polynomial_p(
    p: Polynomial,
    alpha: Tensor | float,
    beta: Tensor | float,
) -> JacobiPolynomialP:
    """Convert power polynomial to Jacobi series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.
    alpha : Tensor or float
        Jacobi parameter α, must be > -1.
    beta : Tensor or float
        Jacobi parameter β, must be > -1.

    Returns
    -------
    JacobiPolynomialP
        Equivalent Jacobi series.

    Notes
    -----
    Uses Horner's method in Jacobi basis:
        p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n
             = c_0 + x*(c_1 + x*(c_2 + ... + x*c_n))

    Starting from c_n, we repeatedly multiply by x (in Jacobi basis)
    and add the next coefficient.

    Examples
    --------
    >>> p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
    >>> c = polynomial_to_jacobi_polynomial_p(p, alpha=0.0, beta=0.0)
    >>> c.coeffs  # x^2 in Legendre basis
    tensor([0.3333, 0.0000, 0.6667])
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    # Convert alpha and beta to tensors if needed
    if not isinstance(alpha, Tensor):
        alpha = torch.tensor(alpha, dtype=coeffs.dtype, device=coeffs.device)
    if not isinstance(beta, Tensor):
        beta = torch.tensor(beta, dtype=coeffs.dtype, device=coeffs.device)

    if n == 0:
        return JacobiPolynomialP(
            coeffs=torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device),
            alpha=alpha,
            beta=beta,
        )

    # Start with highest degree coefficient
    result = JacobiPolynomialP(
        coeffs=coeffs[..., -1:].clone(), alpha=alpha.clone(), beta=beta.clone()
    )

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = jacobi_polynomial_p_mulx(result)
        result = jacobi_polynomial_p_add(
            result,
            JacobiPolynomialP(
                coeffs=coeffs[..., i : i + 1].clone(),
                alpha=alpha.clone(),
                beta=beta.clone(),
            ),
        )

    return result
