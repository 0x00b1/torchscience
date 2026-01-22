import torch

from torchscience.polynomial._polynomial import Polynomial

from ._chebyshev_polynomial_u import (
    ChebyshevPolynomialU,
    chebyshev_polynomial_u,
)
from ._chebyshev_polynomial_u_add import chebyshev_polynomial_u_add
from ._chebyshev_polynomial_u_mulx import chebyshev_polynomial_u_mulx


def polynomial_to_chebyshev_polynomial_u(
    p: Polynomial,
) -> ChebyshevPolynomialU:
    """Convert power polynomial to Chebyshev U series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.

    Returns
    -------
    ChebyshevPolynomialU
        Equivalent Chebyshev U series.

    Notes
    -----
    Uses Horner's method in Chebyshev U basis:
        p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n
             = c_0 + x*(c_1 + x*(c_2 + ... + x*c_n))

    Starting from c_n, we repeatedly multiply by x (in Chebyshev U basis)
    and add the next coefficient.

    Examples
    --------
    >>> p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
    >>> c = polynomial_to_chebyshev_polynomial_u(p)
    >>> c  # x^2 = (U_0 + U_2)/4 since U_2 = 4x^2 - 1, so x^2 = (U_2 + 1)/4
    """
    coeffs = p.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    if n == 0:
        return chebyshev_polynomial_u(
            torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # Start with highest degree coefficient
    result = chebyshev_polynomial_u(coeffs[..., -1:].clone())

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = chebyshev_polynomial_u_mulx(result)
        result = chebyshev_polynomial_u_add(
            result, chebyshev_polynomial_u(coeffs[..., i : i + 1].clone())
        )

    return result
