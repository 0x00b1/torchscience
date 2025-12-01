import torch

from ._laguerre_polynomial_l import LaguerrePolynomialL
from ._laguerre_polynomial_l_multiply import laguerre_polynomial_l_multiply


def laguerre_polynomial_l_pow(
    a: LaguerrePolynomialL,
    n: int,
) -> LaguerrePolynomialL:
    """Raise Laguerre series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : LaguerrePolynomialL
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    LaguerrePolynomialL
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = laguerre_polynomial_l(torch.tensor([1.0, 1.0]))  # L_0 + L_1
    >>> b = laguerre_polynomial_l_pow(a, 2)
    >>> # (L_0 + L_1)^2 using Laguerre linearization
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # a^0 = L_0 = 1
        ones_shape = list(a.coeffs.shape)
        ones_shape[-1] = 1
        return LaguerrePolynomialL(
            coeffs=torch.ones(
                ones_shape, dtype=a.coeffs.dtype, device=a.coeffs.device
            )
        )

    if n == 1:
        return LaguerrePolynomialL(coeffs=a.coeffs.clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = LaguerrePolynomialL(coeffs=base.coeffs.clone())
            else:
                result = laguerre_polynomial_l_multiply(result, base)
        base = laguerre_polynomial_l_multiply(base, base)
        n //= 2

    return result
