import torch

from torchscience.polynomial._polynomial._polynomial import Polynomial

from ._hermite_polynomial_he import (
    HermitePolynomialHe,
    hermite_polynomial_he,
)
from ._hermite_polynomial_he_add import hermite_polynomial_he_add
from ._hermite_polynomial_he_mulx import hermite_polynomial_he_mulx


def polynomial_to_hermite_polynomial_he(
    p: Polynomial,
) -> HermitePolynomialHe:
    """Convert power polynomial to Probabilists' Hermite series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.

    Returns
    -------
    HermitePolynomialHe
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
    >>> p = polynomial(torch.tensor([-1.0, 0.0, 1.0]))  # -1 + x^2
    >>> c = polynomial_to_hermite_polynomial_he(p)
    >>> c  # -1 + x^2 = He_2 (since He_2 = x^2 - 1)
    HermitePolynomialHe(tensor([0., 0., 1.]))
    """
    # Convert to plain tensor for operations
    coeffs = p.as_subclass(torch.Tensor)
    n = coeffs.shape[-1]

    if n == 0:
        return hermite_polynomial_he(
            torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device)
        )

    # Start with highest degree coefficient
    result = hermite_polynomial_he(coeffs[..., -1:].clone())

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = hermite_polynomial_he_mulx(result)
        result = hermite_polynomial_he_add(
            result, hermite_polynomial_he(coeffs[..., i : i + 1].clone())
        )

    return result
