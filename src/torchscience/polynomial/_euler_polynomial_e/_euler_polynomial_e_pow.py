"""Power of Euler polynomial series."""

import torch

from ._euler_polynomial_e import (
    EulerPolynomialE,
    euler_polynomial_e,
)
from ._euler_polynomial_e_multiply import euler_polynomial_e_multiply


def euler_polynomial_e_pow(
    a: EulerPolynomialE,
    n: int,
) -> EulerPolynomialE:
    """Raise Euler polynomial series to a power.

    Parameters
    ----------
    a : EulerPolynomialE
        Series to raise to power.
    n : int
        Non-negative integer power.

    Returns
    -------
    EulerPolynomialE
        Series a^n.

    Raises
    ------
    ValueError
        If n is negative.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"Power must be a non-negative integer, got {n}")

    if n == 0:
        # Return 1 (which is E_0)
        result_shape = list(a.shape)
        result_shape[-1] = 1
        coeffs = torch.ones(result_shape, dtype=a.dtype, device=a.device)
        return euler_polynomial_e(coeffs)

    if n == 1:
        return euler_polynomial_e(a.as_subclass(torch.Tensor).clone())

    # Binary exponentiation
    result = None
    base = a

    while n > 0:
        if n % 2 == 1:
            if result is None:
                result = euler_polynomial_e(
                    base.as_subclass(torch.Tensor).clone()
                )
            else:
                result = euler_polynomial_e_multiply(result, base)
        base = euler_polynomial_e_multiply(base, base)
        n //= 2

    return result
