import torch

from ._hermite_polynomial_h import (
    HermitePolynomialH,
    hermite_polynomial_h,
)
from ._hermite_polynomial_h_multiply import hermite_polynomial_h_multiply


def hermite_polynomial_h_pow(
    a: HermitePolynomialH,
    n: int,
) -> HermitePolynomialH:
    """Raise Physicists' Hermite series to a non-negative integer power.

    Uses binary exponentiation for efficiency.

    Parameters
    ----------
    a : HermitePolynomialH
        Base series.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    HermitePolynomialH
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> a = hermite_polynomial_h(torch.tensor([1.0, 1.0]))  # H_0 + H_1
    >>> b = hermite_polynomial_h_pow(a, 2)
    >>> # (H_0 + H_1)^2 using Hermite linearization
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # a^0 = H_0 = 1
        ones_shape = list(a.shape)
        ones_shape[-1] = 1
        return hermite_polynomial_h(
            torch.ones(ones_shape, dtype=a.dtype, device=a.device)
        )

    if n == 1:
        return hermite_polynomial_h(a.clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = hermite_polynomial_h(base.clone())
            else:
                result = hermite_polynomial_h_multiply(result, base)
        base = hermite_polynomial_h_multiply(base, base)
        n //= 2

    return result
