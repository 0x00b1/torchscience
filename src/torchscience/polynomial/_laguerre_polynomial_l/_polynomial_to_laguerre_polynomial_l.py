import torch

from torchscience.polynomial._polynomial import Polynomial

from ._laguerre_polynomial_l import LaguerrePolynomialL
from ._laguerre_polynomial_l_add import laguerre_polynomial_l_add
from ._laguerre_polynomial_l_mulx import laguerre_polynomial_l_mulx


def polynomial_to_laguerre_polynomial_l(
    p: Polynomial,
) -> LaguerrePolynomialL:
    """Convert power polynomial to Laguerre series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.

    Returns
    -------
    LaguerrePolynomialL
        Equivalent Laguerre series.

    Notes
    -----
    Uses Horner's method in Laguerre basis:
        p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n
             = c_0 + x*(c_1 + x*(c_2 + ... + x*c_n))

    Starting from c_n, we repeatedly multiply by x (in Laguerre basis)
    and add the next coefficient.

    Examples
    --------
    >>> p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
    >>> c = polynomial_to_laguerre_polynomial_l(p)
    >>> c.coeffs  # x^2 = 2 - 4*L_1 + 2*L_2 = 2*L_0 - 4*L_1 + 2*L_2
    tensor([ 2., -4.,  2.])
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    if n == 0:
        return LaguerrePolynomialL(
            coeffs=torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # Start with highest degree coefficient
    result = LaguerrePolynomialL(coeffs=coeffs[..., -1:].clone())

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = laguerre_polynomial_l_mulx(result)
        result = laguerre_polynomial_l_add(
            result, LaguerrePolynomialL(coeffs=coeffs[..., i : i + 1].clone())
        )

    return result
