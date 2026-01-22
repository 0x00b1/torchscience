import torch

from torchscience.polynomial._polynomial import Polynomial

from ._hermite_polynomial_h import (
    HermitePolynomialH,
    hermite_polynomial_h,
)
from ._hermite_polynomial_h_add import hermite_polynomial_h_add
from ._hermite_polynomial_h_mulx import hermite_polynomial_h_mulx


def polynomial_to_hermite_polynomial_h(
    p: Polynomial,
) -> HermitePolynomialH:
    """Convert power polynomial to Physicists' Hermite series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.

    Returns
    -------
    HermitePolynomialH
        Equivalent Hermite series.

    Notes
    -----
    Uses Horner's method in Hermite basis:
        p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n
             = c_0 + x*(c_1 + x*(c_2 + ... + x*c_n))

    Starting from c_n, we repeatedly multiply by x (in Hermite basis)
    and add the next coefficient.

    Examples
    --------
    >>> p = polynomial(torch.tensor([-2.0, 0.0, 4.0]))  # -2 + 4x^2
    >>> c = polynomial_to_hermite_polynomial_h(p)
    >>> c  # -2 + 4x^2 = H_2 (since H_2 = 4x^2 - 2)
    HermitePolynomialH(tensor([0., 0., 1.]))
    """
    coeffs = p.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    if n == 0:
        return hermite_polynomial_h(
            torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # Start with highest degree coefficient
    result = hermite_polynomial_h(coeffs[..., -1:].clone())

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = hermite_polynomial_h_mulx(result)
        result = hermite_polynomial_h_add(
            result, hermite_polynomial_h(coeffs[..., i : i + 1].clone())
        )

    return result
