import torch

from torchscience.polynomial._polynomial import Polynomial

from ._chebyshev_polynomial_t import (
    ChebyshevPolynomialT,
    chebyshev_polynomial_t,
)
from ._chebyshev_polynomial_t_add import chebyshev_polynomial_t_add
from ._chebyshev_polynomial_t_mulx import chebyshev_polynomial_t_mulx


def polynomial_to_chebyshev_polynomial_t(
    p: Polynomial,
) -> ChebyshevPolynomialT:
    """Convert power polynomial to Chebyshev series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.

    Returns
    -------
    ChebyshevPolynomialT
        Equivalent Chebyshev series.

    Notes
    -----
    Uses Horner's method in Chebyshev basis:
        p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n
             = c_0 + x*(c_1 + x*(c_2 + ... + x*c_n))

    Starting from c_n, we repeatedly multiply by x (in Chebyshev basis)
    and add the next coefficient.

    Examples
    --------
    >>> p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
    >>> c = polynomial_to_chebyshev_polynomial_t(p)
    >>> c  # x^2 = (T_0 + T_2)/2
    ChebyshevPolynomialT(tensor([0.5, 0.0, 0.5]))
    """
    # The polynomial IS the coefficients tensor
    coeffs = p.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    if n == 0:
        return chebyshev_polynomial_t(
            torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # Start with highest degree coefficient
    result = chebyshev_polynomial_t(coeffs[..., -1:].clone())

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = chebyshev_polynomial_t_mulx(result)
        result = chebyshev_polynomial_t_add(
            result, chebyshev_polynomial_t(coeffs[..., i : i + 1].clone())
        )

    return result
