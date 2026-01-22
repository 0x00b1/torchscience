import torch
from torch import Tensor

from torchscience.polynomial._polynomial import Polynomial

from ._gegenbauer_polynomial_c import (
    GegenbauerPolynomialC,
    gegenbauer_polynomial_c,
)
from ._gegenbauer_polynomial_c_add import gegenbauer_polynomial_c_add
from ._gegenbauer_polynomial_c_mulx import gegenbauer_polynomial_c_mulx


def polynomial_to_gegenbauer_polynomial_c(
    p: Polynomial,
    lambda_: Tensor,
) -> GegenbauerPolynomialC:
    """Convert power polynomial to Gegenbauer series.

    Parameters
    ----------
    p : Polynomial
        Power polynomial.
    lambda_ : Tensor
        Parameter lambda > -1/2.

    Returns
    -------
    GegenbauerPolynomialC
        Equivalent Gegenbauer series.

    Notes
    -----
    Uses Horner's method in Gegenbauer basis:
        p(x) = c_0 + c_1*x + c_2*x^2 + ... + c_n*x^n
             = c_0 + x*(c_1 + x*(c_2 + ... + x*c_n))

    Starting from c_n, we repeatedly multiply by x (in Gegenbauer basis)
    and add the next coefficient.

    Examples
    --------
    >>> from torchscience.polynomial import polynomial
    >>> p = polynomial(torch.tensor([0.0, 0.0, 1.0]))  # x^2
    >>> c = polynomial_to_gegenbauer_polynomial_c(p, torch.tensor(1.0))
    >>> # x^2 in Gegenbauer basis with lambda=1
    """
    # Get coefficients as plain tensor
    coeffs = p.as_subclass(Tensor)
    n = coeffs.shape[-1]

    # Ensure lambda_ is a tensor
    if not isinstance(lambda_, Tensor):
        lambda_ = torch.tensor(
            lambda_, dtype=coeffs.dtype, device=coeffs.device
        )

    if n == 0:
        return gegenbauer_polynomial_c(
            torch.zeros(1, dtype=coeffs.dtype, device=coeffs.device),
            lambda_,
        )

    # Start with highest degree coefficient
    result = gegenbauer_polynomial_c(coeffs[..., -1:].clone(), lambda_)

    # Horner's method: multiply by x, add next coefficient
    for i in range(n - 2, -1, -1):
        result = gegenbauer_polynomial_c_mulx(result)
        result = gegenbauer_polynomial_c_add(
            result,
            gegenbauer_polynomial_c(coeffs[..., i : i + 1].clone(), lambda_),
        )

    return result
