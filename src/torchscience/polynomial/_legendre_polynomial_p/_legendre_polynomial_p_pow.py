import torch

from ._legendre_polynomial_p import (
    LegendrePolynomialP,
    legendre_polynomial_p,
)
from ._legendre_polynomial_p_multiply import legendre_polynomial_p_multiply


def legendre_polynomial_p_pow(
    a: LegendrePolynomialP,
    n: int,
) -> LegendrePolynomialP:
    """Raise Legendre series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : LegendrePolynomialP
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    LegendrePolynomialP
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = legendre_polynomial_p(torch.tensor([1.0, 1.0]))  # P_0 + P_1
    >>> b = legendre_polynomial_p_pow(a, 2)
    >>> # (P_0 + P_1)^2 using Legendre linearization
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    coeffs = a.as_subclass(torch.Tensor)

    if n == 0:
        # a^0 = P_0 = 1
        ones_shape = list(coeffs.shape)
        ones_shape[-1] = 1
        return legendre_polynomial_p(
            torch.ones(ones_shape, dtype=coeffs.dtype, device=coeffs.device)
        )

    if n == 1:
        return legendre_polynomial_p(coeffs.clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = legendre_polynomial_p(
                    base.as_subclass(torch.Tensor).clone()
                )
            else:
                result = legendre_polynomial_p_multiply(result, base)
        base = legendre_polynomial_p_multiply(base, base)
        n //= 2

    return result
