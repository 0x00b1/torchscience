import torch

from torchscience.polynomial._polynomial import Polynomial

from ._chebyshev_polynomial_v import ChebyshevPolynomialV
from ._chebyshev_polynomial_v_add import chebyshev_polynomial_v_add
from ._chebyshev_polynomial_v_mulx import chebyshev_polynomial_v_mulx


def polynomial_to_chebyshev_polynomial_v(
    p: Polynomial,
) -> ChebyshevPolynomialV:
    """Convert power polynomial to Chebyshev V series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.

    Returns
    -------
    ChebyshevPolynomialV
        Equivalent Chebyshev V series.

    Notes
    -----
    Uses Horner's method in Chebyshev V basis:
        p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n
             = c_0 + x*(c_1 + x*(c_2 + ... + x*c_n))

    Starting from c_n, we repeatedly multiply by x (in Chebyshev V basis)
    and add the next coefficient.

    Examples
    --------
    >>> p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
    >>> c = polynomial_to_chebyshev_polynomial_v(p)
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    if n == 0:
        return ChebyshevPolynomialV(
            coeffs=torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # Start with highest degree coefficient
    result = ChebyshevPolynomialV(coeffs=coeffs[..., -1:].clone())

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = chebyshev_polynomial_v_mulx(result)
        result = chebyshev_polynomial_v_add(
            result, ChebyshevPolynomialV(coeffs=coeffs[..., i : i + 1].clone())
        )

    return result
