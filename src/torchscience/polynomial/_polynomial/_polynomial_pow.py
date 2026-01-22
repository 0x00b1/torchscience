import torch

from ._polynomial import Polynomial, polynomial
from ._polynomial_multiply import polynomial_multiply


def polynomial_pow(p: Polynomial, n: int) -> Polynomial:
    """Raise polynomial to non-negative integer power.

    Uses binary exponentiation (repeated squaring) for efficiency.

    Parameters
    ----------
    p : Polynomial
        Base polynomial.
    n : int
        Non-negative integer exponent.

    Returns
    -------
    Polynomial
        p raised to power n.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> p = polynomial(torch.tensor([1.0, 1.0]))  # 1 + x
    >>> polynomial_pow(p, 3)  # 1 + 3x + 3x^2 + x^3
    Polynomial(tensor([1., 3., 3., 1.]))
    """
    if n < 0:
        raise ValueError(f"Exponent must be non-negative, got {n}")

    if n == 0:
        # Return constant 1 with same dtype/device
        # p IS the coefficient tensor now
        one = torch.ones(1, dtype=p.dtype, device=p.device)
        # Handle batch dimensions
        if p.dim() > 1:
            batch_shape = p.shape[:-1]
            one = one.expand(*batch_shape, 1)
        return polynomial(one)

    if n == 1:
        return p

    # Binary exponentiation
    result = None
    base = p

    while n > 0:
        if n & 1:  # If current bit is 1
            if result is None:
                result = base
            else:
                result = polynomial_multiply(result, base)
        base = polynomial_multiply(base, base)
        n >>= 1

    return result
