import torch

from torchscience.polynomial._polynomial import Polynomial

from ._legendre_polynomial_p import LegendrePolynomialP
from ._legendre_polynomial_p_add import legendre_polynomial_p_add
from ._legendre_polynomial_p_mulx import legendre_polynomial_p_mulx


def polynomial_to_legendre_polynomial_p(
    p: Polynomial,
) -> LegendrePolynomialP:
    """Convert power polynomial to Legendre series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.

    Returns
    -------
    LegendrePolynomialP
        Equivalent Legendre series.

    Notes
    -----
    Uses Horner's method in Legendre basis:
        p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n
             = c_0 + x*(c_1 + x*(c_2 + ... + x*c_n))

    Starting from c_n, we repeatedly multiply by x (in Legendre basis)
    and add the next coefficient.

    Examples
    --------
    >>> p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
    >>> c = polynomial_to_legendre_polynomial_p(p)
    >>> c.coeffs  # x^2 = (2*P_2 + P_0)/3
    tensor([0.3333, 0.0000, 0.6667])
    """
    coeffs = p.coeffs
    n = coeffs.shape[-1]

    if n == 0:
        return LegendrePolynomialP(
            coeffs=torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # Start with highest degree coefficient
    result = LegendrePolynomialP(coeffs=coeffs[..., -1:].clone())

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = legendre_polynomial_p_mulx(result)
        result = legendre_polynomial_p_add(
            result, LegendrePolynomialP(coeffs=coeffs[..., i : i + 1].clone())
        )

    return result
