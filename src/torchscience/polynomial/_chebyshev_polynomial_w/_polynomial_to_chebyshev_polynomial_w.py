import torch

from torchscience.polynomial._polynomial import Polynomial

from ._chebyshev_polynomial_w import ChebyshevPolynomialW
from ._chebyshev_polynomial_w_add import chebyshev_polynomial_w_add
from ._chebyshev_polynomial_w_mulx import chebyshev_polynomial_w_mulx


def polynomial_to_chebyshev_polynomial_w(
    p: Polynomial,
) -> ChebyshevPolynomialW:
    """Convert power polynomial to Chebyshev W series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.

    Returns
    -------
    ChebyshevPolynomialW
        Equivalent Chebyshev W series.

    Notes
    -----
    Uses Horner's method in Chebyshev W basis:
        p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n
             = c_0 + x*(c_1 + x*(c_2 + ... + x*c_n))

    Starting from c_n, we repeatedly multiply by x (in Chebyshev W basis)
    and add the next coefficient.

    Examples
    --------
    >>> p = Polynomial(coeffs=torch.tensor([0.0, 0.0, 1.0]))  # x^2
    >>> c = polynomial_to_chebyshev_polynomial_w(p)
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    if n == 0:
        return ChebyshevPolynomialW(
            coeffs=torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # Start with highest degree coefficient
    result = ChebyshevPolynomialW(coeffs=coeffs[..., -1:].clone())

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = chebyshev_polynomial_w_mulx(result)
        result = chebyshev_polynomial_w_add(
            result, ChebyshevPolynomialW(coeffs=coeffs[..., i : i + 1].clone())
        )

    return result
